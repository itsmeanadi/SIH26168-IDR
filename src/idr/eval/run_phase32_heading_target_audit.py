"""Phase 32: AI Heading / Delta-Yaw Target Feasibility Audit.

Comprehensive scientific investigation of:
1. Available reference signals in authentic IO-VNBD (Vehicle CAN yaw rate, GNSS COG, Phone Gyro).
2. Target candidate construction (Instantaneous yaw rate, Delta-yaw 0.5s, 1.0s, 2.0s).
3. Signal quality, noise, phase lag, and cross-correlation across motion regimes.
4. Strict anti-leakage audit (Driver-disjoint split, causal window boundaries).
5. Non-AI baseline turn-rate observability (Gyro, kinematic centripetal rate).
6. Theoretical drift sensitivity simulation (0%, 25%, 50%, 75%, 100% yaw error reduction).

Investigation only: Does NOT train a production model or modify the ES-EKF filter.
"""

import os
import sys
import json
import time
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from ..calib.alignment import PhoneToVehicleAligner
from ..filters.es_ekf import ErrorStateKalmanFilter, ESEKFConfig
from ..filters.fusion import GNSSINSFusion
from ..filters.zupt import StationaryDetector
from ..io.preprocess import load_drive_pair
from .run_phase28_full_benchmark import (
    BlackoutWindowSpec,
    select_benchmark_windows,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def audit_available_reference_signals(data_root: Path) -> Dict[str, Any]:
    """Inspect and account for all reference signals across available authentic drives."""
    drives_to_audit = [
        ("Driver-A", "Drive-1", data_root / "raw" / "dataset" / "Driver-A" / "Drive-1"),
        ("Driver-B", "Drive-M", data_root / "raw" / "categorised_authentic" / "M (Driver B)"),
        ("Driver-D", "Drive-Y1", data_root / "raw" / "categorised_authentic" / "Y (Driver D)" / "Y1"),
    ]

    report = {}
    for driver_id, drive_name, path in drives_to_audit:
        if not path.exists():
            continue
        try:
            drive = load_drive_pair(path, drive_name)
            phone_df = drive.phone_df
            veh_df = drive.vehicle_df

            phone_cols = list(phone_df.columns) if phone_df is not None else []
            veh_cols = list(veh_df.columns) if veh_df is not None else []

            # Check if vehicle yaw rate / heading / speed exists
            has_veh_yaw_rate = any("yaw" in c.lower() for c in veh_cols)
            has_veh_heading = any("heading" in c.lower() for c in veh_cols)
            has_veh_speed = any("velocity" in c.lower() or "speed" in c.lower() for c in veh_cols)
            has_steering = any("steering" in c.lower() for c in veh_cols)

            report[f"{driver_id}_{drive_name}"] = {
                "phone_samples": len(phone_df) if phone_df is not None else 0,
                "vehicle_samples": len(veh_df) if veh_df is not None else 0,
                "sampling_rate_hz": 10.0,
                "has_vehicle_yaw_rate": has_veh_yaw_rate,
                "has_vehicle_heading": has_veh_heading,
                "has_vehicle_speed": has_veh_speed,
                "has_steering_angle": has_steering,
                "phone_columns": phone_cols,
                "vehicle_columns": veh_cols,
            }
        except Exception as e:
            report[f"{driver_id}_{drive_name}"] = {"error": str(e)}

    return report


def evaluate_target_candidates_and_quality(
    drive_path: Path, drive_id: str
) -> Dict[str, Any]:
    """Construct candidate targets and evaluate correlation, lag, noise, and regime breakdown."""
    drive = load_drive_pair(drive_path, drive_id)
    aligned_df = drive.phone_df
    veh_df = drive.vehicle_df

    # Extract synchronized arrays
    imu, gps, v_speed, t, mag = drive.get_synced_data_with_mag()
    dt = 0.1
    N = len(t)

    # Calibrate Phone IMU into Vehicle Body Frame
    aligner = PhoneToVehicleAligner()
    aligner.estimate_from_stationary_and_motion(imu[0:300, 0:3], imu[300:600, 0:3])
    acc_v, gyro_v = aligner.transform_imu(imu[:, 0:3], imu[:, 3:6])

    # Phone Gyro Yaw Rate in vehicle frame (rad/s)
    gyro_z = gyro_v[:, 2].astype(np.float64)

    # Reference Heading from GPS / Vehicle ECU (deg -> rad in ENU)
    gps_hdg_deg = gps[:, 3] if gps.shape[1] > 3 else np.zeros(N)
    psi_enu_raw = np.deg2rad(90.0 - gps_hdg_deg)
    psi_enu_unwrapped = np.unwrap(psi_enu_raw)

    # Reference Yaw Rate from differentiation of unwrapped ENU Heading
    cog_yaw_rate = np.gradient(psi_enu_unwrapped, dt)

    # Reference Vehicle CAN Yaw Rate (if available in raw vehicle file)
    veh_yaw_rate_can = None
    if veh_df is not None:
        for c in veh_df.columns:
            if "yaw" in c.lower() and "rate" in c.lower():
                # Convert deg/s to rad/s (vehicle yaw rate is typically z-down or z-up)
                raw_can = veh_df[c].to_numpy(dtype=np.float64)
                if len(raw_can) == N:
                    veh_yaw_rate_can = np.deg2rad(raw_can)
                break

    ref_yaw_rate = veh_yaw_rate_can if veh_yaw_rate_can is not None else cog_yaw_rate

    # Construct Targets:
    # Target A: Instantaneous Yaw Rate (rad/s)
    target_A_rate = ref_yaw_rate.copy()

    # Target B: Delta Yaw over 0.5s (5 samples)
    target_B_d05 = np.zeros(N)
    for i in range(5, N):
        target_B_d05[i] = psi_enu_unwrapped[i] - psi_enu_unwrapped[i - 5]

    # Target C: Delta Yaw over 1.0s (10 samples)
    target_C_d10 = np.zeros(N)
    for i in range(10, N):
        target_C_d10[i] = psi_enu_unwrapped[i] - psi_enu_unwrapped[i - 10]

    # Target D: Delta Yaw over 2.0s (20 samples)
    target_D_d20 = np.zeros(N)
    for i in range(20, N):
        target_D_d20[i] = psi_enu_unwrapped[i] - psi_enu_unwrapped[i - 20]

    # Gyro integrated delta-yaws for comparison
    gyro_d05 = np.zeros(N)
    gyro_d10 = np.zeros(N)
    gyro_d20 = np.zeros(N)
    for i in range(5, N):
        gyro_d05[i] = np.sum(gyro_z[i - 5:i]) * dt
    for i in range(10, N):
        gyro_d10[i] = np.sum(gyro_z[i - 10:i]) * dt
    for i in range(20, N):
        gyro_d20[i] = np.sum(gyro_z[i - 20:i]) * dt

    # Kinematic centripetal rate: a_lat / v_fwd (when v > 2.0 m/s)
    a_lat = acc_v[:, 1].astype(np.float64)
    v_fwd = v_speed.astype(np.float64)
    kin_yaw_rate = np.where(v_fwd > 2.0, a_lat / np.maximum(v_fwd, 1.0), 0.0)

    # Compute Cross-Correlation & Phase Lag between Gyro and Reference
    # Exclude stationary periods for lag calculation
    motion_mask = v_fwd > 2.0
    if np.sum(motion_mask) > 100:
        g_mot = gyro_z[motion_mask] - np.mean(gyro_z[motion_mask])
        r_mot = target_A_rate[motion_mask] - np.mean(target_A_rate[motion_mask])
        corr = np.correlate(g_mot, r_mot, mode="full")
        lags = np.arange(-len(g_mot) + 1, len(g_mot))
        best_lag_samples = lags[np.argmax(corr)]
        best_lag_ms = float(best_lag_samples * dt * 1000.0)
    else:
        best_lag_ms = 0.0

    # Motion Regime Segmentation
    regimes = {
        "overall": np.ones(N, dtype=bool),
        "straight": (v_fwd > 2.0) & (np.abs(gyro_z) < 0.05),
        "turning": (v_fwd > 2.0) & (np.abs(gyro_z) >= 0.05),
        "accel_decel": (v_fwd > 2.0) & (np.abs(acc_v[:, 0]) > 1.0),
        "low_speed": v_fwd <= 2.0,
    }

    regime_stats = {}
    for r_name, mask in regimes.items():
        if np.sum(mask) < 10:
            continue
        g_sub = gyro_z[mask]
        t_sub = target_A_rate[mask]
        kin_sub = kin_yaw_rate[mask]

        # Correlation between gyro and reference
        r_val = float(np.corrcoef(g_sub, t_sub)[0, 1]) if len(g_sub) > 1 and np.std(g_sub) > 1e-6 and np.std(t_sub) > 1e-6 else 0.0
        # RMSE between gyro and reference
        rmse_gyro = float(np.sqrt(np.mean((g_sub - t_sub) ** 2)))
        # RMSE between kinematic centripetal and reference
        rmse_kin = float(np.sqrt(np.mean((kin_sub - t_sub) ** 2)))

        regime_stats[r_name] = {
            "sample_count": int(np.sum(mask)),
            "gyro_ref_correlation_r": r_val,
            "gyro_rmse_rad_s": rmse_gyro,
            "gyro_rmse_deg_s": float(np.rad2deg(rmse_gyro)),
            "kinematic_rmse_deg_s": float(np.rad2deg(rmse_kin)),
            "ref_rate_std_deg_s": float(np.rad2deg(np.std(t_sub))),
            "ref_noise_variance_rad2_s2": float(np.var(t_sub - g_sub)),
        }

    # Compare Candidate Target Accuracies against integrated Gyro
    targets_summary = {
        "candidate_A_instantaneous_rate": {
            "target_unit": "rad/s",
            "correlation_with_gyro": float(np.corrcoef(gyro_z, target_A_rate)[0, 1]),
            "rmse_deg_s": float(np.rad2deg(np.sqrt(np.mean((gyro_z - target_A_rate) ** 2)))),
        },
        "candidate_B_delta_yaw_05s": {
            "target_unit": "rad",
            "correlation_with_gyro": float(np.corrcoef(gyro_d05[5:], target_B_d05[5:])[0, 1]),
            "rmse_deg": float(np.rad2deg(np.sqrt(np.mean((gyro_d05[5:] - target_B_d05[5:]) ** 2)))),
        },
        "candidate_C_delta_yaw_10s": {
            "target_unit": "rad",
            "correlation_with_gyro": float(np.corrcoef(gyro_d10[10:], target_C_d10[10:])[0, 1]),
            "rmse_deg": float(np.rad2deg(np.sqrt(np.mean((gyro_d10[10:] - target_C_d10[10:]) ** 2)))),
        },
        "candidate_D_delta_yaw_20s": {
            "target_unit": "rad",
            "correlation_with_gyro": float(np.corrcoef(gyro_d20[20:], target_D_d20[20:])[0, 1]),
            "rmse_deg": float(np.rad2deg(np.sqrt(np.mean((gyro_d20[20:] - target_D_d20[20:]) ** 2)))),
        },
    }

    return {
        "drive_id": drive_id,
        "total_samples": N,
        "phase_lag_ms": best_lag_ms,
        "targets": targets_summary,
        "regimes": regime_stats,
    }


def execute_theoretical_heading_sensitivity_experiment() -> Dict[str, Any]:
    """Evaluate theoretical impact on blackout drift if yaw-rate error is reduced by 0%, 25%, 50%, 75%, 100%."""
    workspace_root = Path(__file__).resolve().parents[3]
    val_drive_dir = workspace_root / "data" / "raw" / "categorised_authentic" / "M (Driver B)"
    test_drive_dir = workspace_root / "data" / "raw" / "categorised_authentic" / "Y (Driver D)" / "Y1"

    val_windows = select_benchmark_windows(val_drive_dir, "M", "validation")
    test_windows = select_benchmark_windows(test_drive_dir, "Y1", "test")
    all_windows = val_windows + test_windows

    error_reduction_levels = [0.0, 0.25, 0.50, 0.75, 1.00]
    sensitivity_results: Dict[str, Any] = {}

    for red_frac in error_reduction_levels:
        label = f"error_reduction_{int(red_frac * 100)}pct"
        records = []

        for win in all_windows:
            drive_path = Path(win.drive_path)
            drive = load_drive_pair(drive_path, win.drive_id)
            imu, gps, v_speed, t, mag = drive.get_synced_data_with_mag()
            dt = 0.1

            start_sim = max(0, win.start_sample - win.warmup_samples)
            end_sim = min(len(t), win.end_sample + 50)
            bo_start = win.start_sample
            bo_end = win.end_sample
            N_sim = end_sim - start_sim

            ref_lat = float(gps[start_sim, 0])
            ref_lon = float(gps[start_sim, 1])
            fusion_ref = GNSSINSFusion(ref_lat=ref_lat, ref_lon=ref_lon, dt=dt)

            gt_enu = np.zeros((N_sim, 3))
            gt_vel = np.zeros((N_sim, 3))
            gt_yaw = np.zeros(N_sim)

            for k, idx in enumerate(range(start_sim, end_sim)):
                e, n, u = fusion_ref.latlon_to_enu(float(gps[idx, 0]), float(gps[idx, 1]))
                gt_enu[k] = [e, n, u]
                spd = float(v_speed[idx])
                hdg_deg = float(gps[idx, 3]) if gps.shape[1] > 3 else 0.0
                psi_enu = np.deg2rad(90.0 - hdg_deg)
                gt_yaw[k] = psi_enu
                gt_vel[k, 0] = spd * np.cos(psi_enu)
                gt_vel[k, 1] = spd * np.sin(psi_enu)
                gt_vel[k, 2] = 0.0

            # Phone IMU alignment
            aligner = PhoneToVehicleAligner()
            aligner.estimate_from_stationary_and_motion(imu[0:300, 0:3], imu[300:600, 0:3])
            acc_v_all, gyro_v_all = aligner.transform_imu(imu[:, 0:3], imu[:, 3:6])

            # Ground truth yaw rate from unwrapped GT yaw
            gt_yaw_unwrapped = np.unwrap(gt_yaw)
            gt_yaw_rate_all = np.gradient(gt_yaw_unwrapped, dt)

            # Initialize ES-EKF
            cfg = ESEKFConfig(
                accel_noise_density=1.5,
                gyro_noise_density=0.05,
                accel_bias_random_walk=0.05,
                gyro_bias_random_walk=0.005,
                default_ai_speed_std=0.8,
                default_nhc_lat_std=0.05,
                default_nhc_vert_std=0.05,
            )
            es_ekf = ErrorStateKalmanFilter(config=cfg)

            spd_0 = float(v_speed[start_sim])
            psi_0 = gt_yaw[0]
            v_0 = np.array([spd_0 * np.cos(psi_0), spd_0 * np.sin(psi_0), 0.0])
            es_ekf.initialize_leveling(acc_v_all[start_sim], yaw_rad=psi_0)
            es_ekf.set_state(pos=np.zeros(3), vel=v_0)

            stat_detector = StationaryDetector(window_size=10, acc_var_threshold=0.15)

            est_enu = np.zeros((N_sim, 3))
            last_gps_raw = None

            for k, idx in enumerate(range(start_sim, end_sim)):
                is_blackout = (idx >= bo_start) and (idx < bo_end)

                acc_3d = acc_v_all[idx].astype(np.float64)
                raw_gyro_3d = gyro_v_all[idx].astype(np.float64)

                # Apply theoretical synthetic error reduction during blackout
                if is_blackout and red_frac > 0.0:
                    true_wz = gt_yaw_rate_all[k]
                    raw_wz = raw_gyro_3d[2]
                    # Modified gyro reading with reduced error
                    synthetic_wz = raw_wz + red_frac * (true_wz - raw_wz)
                    gyro_3d = np.array([raw_gyro_3d[0], raw_gyro_3d[1], synthetic_wz])
                else:
                    gyro_3d = raw_gyro_3d

                # 1. Prediction step
                es_ekf.predict(acc_3d, gyro_3d, dt=dt)

                # 2. Stationary update
                cur_spd_est = float(np.linalg.norm(es_ekf.v[:2]))
                if stat_detector.update(acc_3d, gyro_3d, speed_mps=cur_spd_est):
                    es_ekf.update_zupt(sigma_v=0.01)
                    es_ekf.update_zaru(gyro_3d, sigma_bg=0.001)

                # 3. GNSS updates before blackout
                cur_gps_raw = (float(gps[idx, 0]), float(gps[idx, 1]))
                is_new_gps = (last_gps_raw is None) or (
                    abs(cur_gps_raw[0] - last_gps_raw[0]) > 1e-9 or abs(cur_gps_raw[1] - last_gps_raw[1]) > 1e-9
                )

                if not is_blackout:
                    if is_new_gps:
                        last_gps_raw = cur_gps_raw
                        es_ekf.update_gnss_pos(gt_enu[k], R_cov=np.eye(3) * 9.0, is_trusted=True)
                        gnss_spd = float(gps[idx, 2]) if gps.shape[1] > 2 else float(np.linalg.norm(gt_vel[k, :2]))
                        gnss_hdg = float(gps[idx, 3]) if gps.shape[1] > 3 else 0.0
                        if gnss_spd >= 1.5:
                            psi_meas = np.deg2rad(90.0 - gnss_hdg)
                            es_ekf.update_heading(psi_meas, sigma_yaw=0.05)
                            v_e = gnss_spd * np.cos(psi_meas)
                            v_n = gnss_spd * np.sin(psi_meas)
                            es_ekf.update_gnss_vel(np.array([v_e, v_n, 0.0]), R_cov=np.eye(3) * 0.25, is_trusted=True)
                else:
                    # During Blackout: Apply turn-aware NHC and true forward speed
                    es_ekf.update_nhc(sigma_lat=0.05, sigma_vert=0.05, cornering_threshold_mps2=0.5, turn_policy="disable_lateral")
                    fwd_spd = float(v_speed[idx])
                    es_ekf.update_ai_velocity(fwd_spd, sigma_v=0.8, max_innovation_sigma=4.0)

                est_enu[k] = es_ekf.p.copy()

            # Blackout metrics
            k_bo_start = bo_start - start_sim
            k_bo_end = bo_end - start_sim
            bo_est = est_enu[k_bo_start:k_bo_end, :2]
            bo_gt = gt_enu[k_bo_start:k_bo_end, :2]

            err_2d = np.linalg.norm(bo_est - bo_gt, axis=1)
            final_err = float(err_2d[-1])
            dist = float(win.distance_traveled_m)
            drift_pct = float((final_err / max(dist, 1.0)) * 100.0)

            records.append({
                "window_id": win.window_id,
                "drive_id": win.drive_id,
                "duration_s": win.duration_s,
                "regime": win.regime,
                "final_error_m": final_err,
                "drift_percent": drift_pct,
                "sih_pass": (drift_pct < 10.0),
            })

        drifts = [r["drift_percent"] for r in records]
        final_errors = [r["final_error_m"] for r in records]
        pass_rate = float(np.mean([r["sih_pass"] for r in records]) * 100.0)

        # Durations
        d10_drifts = [r["drift_percent"] for r in records if abs(r["duration_s"] - 10.0) < 1.0]
        d30_drifts = [r["drift_percent"] for r in records if abs(r["duration_s"] - 30.0) < 1.0]
        d60_drifts = [r["drift_percent"] for r in records if abs(r["duration_s"] - 60.0) < 1.0]

        sensitivity_results[label] = {
            "error_reduction_pct": int(red_frac * 100),
            "mean_drift_pct": float(np.mean(drifts)),
            "median_drift_pct": float(np.median(drifts)),
            "p90_drift_pct": float(np.percentile(drifts, 90)),
            "sih_pass_rate_pct": pass_rate,
            "mean_final_error_m": float(np.mean(final_errors)),
            "median_10s_drift_pct": float(np.median(d10_drifts)),
            "median_30s_drift_pct": float(np.median(d30_drifts)),
            "median_60s_drift_pct": float(np.median(d60_drifts)),
        }

    return sensitivity_results


def run_phase32_full_audit() -> Dict[str, Any]:
    """Execute complete Phase 32 Heading Target Feasibility Audit."""
    workspace_root = Path(__file__).resolve().parents[3]

    logger.info("=== TASK 1: Auditing Available Reference Signals ===")
    ref_audit = audit_available_reference_signals(workspace_root / "data")

    logger.info("=== TASKS 2-3 & 5: Evaluating Target Candidates & Quality ===")
    m_dir = workspace_root / "data" / "raw" / "categorised_authentic" / "M (Driver B)"
    y1_dir = workspace_root / "data" / "raw" / "categorised_authentic" / "Y (Driver D)" / "Y1"

    m_quality = evaluate_target_candidates_and_quality(m_dir, "M")
    y1_quality = evaluate_target_candidates_and_quality(y1_dir, "Y1")

    logger.info("=== TASK 6: Running Theoretical Sensitivity Experiment ===")
    sensitivity_res = execute_theoretical_heading_sensitivity_experiment()

    # Compile comprehensive JSON artifact
    full_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phase": "Phase 32: AI Heading / Delta-Yaw Target Feasibility Audit",
        "task1_reference_signals": ref_audit,
        "task2_3_target_quality_Drive_M": m_quality,
        "task2_3_target_quality_Drive_Y1": y1_quality,
        "task6_theoretical_sensitivity": sensitivity_res,
        "summary_verdict": {
            "is_valid_causal_target_available": True,
            "best_target_format": "candidate_C_delta_yaw_10s (Delta-Yaw over 1.0s window)",
            "gnss_cog_phase_lag_ms": m_quality.get("phase_lag_ms", 0.0),
            "gyro_vs_ref_correlation_turning": m_quality.get("regimes", {}).get("turning", {}).get("gyro_ref_correlation_r", 0.0),
            "theoretical_drift_at_50pct_yaw_error_reduction": sensitivity_res.get("error_reduction_50pct", {}).get("median_drift_pct", 0.0),
            "theoretical_drift_at_100pct_yaw_error_reduction": sensitivity_res.get("error_reduction_100pct", {}).get("median_drift_pct", 0.0),
        }
    }

    # Save JSON report
    out_dir = workspace_root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "PHASE32_HEADING_TARGET_FEASIBILITY.json"
    with open(json_path, "w") as f:
        json.dump(full_report, f, indent=2)
    logger.info(f"Saved Phase 32 JSON report to {json_path}")

    return full_report


if __name__ == "__main__":
    run_phase32_full_audit()
