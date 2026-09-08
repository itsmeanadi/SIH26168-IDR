"""Phase 34: Dynamic Pitch/Roll Gravity Alignment Evaluation & Validation.

Evaluates adaptive dynamic gravity-alignment constraints for the 15-state ES-EKF
across all 30 authentic IO-VNBD blackout windows (15 Drive M, 15 Drive Y1).

Formulations tested:
1. baseline_phase33: Phase 33 baseline (unlocked GNSS, decoupled NHC, fixed leveling sigma 0.3)
2. adaptive_kinematic_sigma: Dynamic sigma scaling with centripetal (|w_z * v|) and longitudinal (|a_x|) acceleration
3. quasi_steady_norm_gate: Specific force magnitude gating (| ||a|| - g0 | < 1.5 m/s^2)
4. dynamic_gravity_alignment_full: Adaptive sigma + norm gating + decoupled horizontal axes
5. no_gravity_leveling: Leveling completely disabled (ablation reference)
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..config import set_seed
from ..filters.es_ekf import (
    ErrorStateKalmanFilter,
    ESEKFConfig,
    quat_to_rot,
    quat_multiply,
    exp_quaternion,
    rot_to_euler_rpy,
    skew_symmetric,
)
from ..filters.fusion import GNSSINSFusion
from ..filters.zupt import StationaryDetector
from ..calib.alignment import PhoneToVehicleAligner
from ..models.velocity_net import VelocityEstimatorNet
from .run_phase28_full_benchmark import (
    BlackoutWindowSpec,
    get_cached_drive_data,
    select_benchmark_windows,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase34GravityAlignment")


def run_adaptive_gravity_leveling_step(
    es_ekf: ErrorStateKalmanFilter,
    acc_meas: np.ndarray,
    forward_speed: float = 0.0,
    yaw_rate: float = 0.0,
    fwd_acc: float = 0.0,
    base_sigma: float = 0.3,
    centripetal_noise_coeff: float = 0.5,
    longitudinal_noise_coeff: float = 0.5,
    max_sigma: float = 5.0,
    norm_gate_tolerance: float = 2.5,
) -> Tuple[bool, Dict[str, Any]]:
    """Execute dynamic pitch/roll gravity alignment with adaptive covariance."""
    a_m = np.array(acc_meas, dtype=np.float64).reshape(3)
    norm_a = float(np.linalg.norm(a_m))
    g0 = 9.80665

    # Check 1: Reject severe shocks / bumps when specific force norm deviates excessively from gravity
    if abs(norm_a - g0) > norm_gate_tolerance:
        return False, {"accepted": False, "reason": f"Specific force norm {norm_a:.2f} deviated from g0 by > {norm_gate_tolerance} m/s^2"}

    # Kinematic accelerations in vehicle body frame
    centripetal = float(yaw_rate * forward_speed)
    a_kin = np.array([fwd_acc, centripetal, 0.0], dtype=np.float64)

    z_g = a_m - a_kin - es_ekf.ba
    h_g = -es_ekf.rotation_matrix.T @ es_ekf.g_nav

    z = z_g[0:2]
    h_x = h_g[0:2]

    H = np.zeros((2, 15), dtype=np.float64)
    # Attitude error Jacobian: [h_g]_x (acts on roll and pitch, zero yaw coupling)
    H[0:2, 6:9] = skew_symmetric(h_g)[0:2, :]
    # Accelerometer bias Jacobian
    H[0:2, 9:12] = np.eye(3)[0:2, :]

    # Adaptive standard deviations scaling with kinematic dynamic intensity
    sigma_pitch = float(np.clip(np.sqrt(base_sigma**2 + (longitudinal_noise_coeff * abs(fwd_acc))**2), base_sigma, max_sigma))
    sigma_roll = float(np.clip(np.sqrt(base_sigma**2 + (centripetal_noise_coeff * abs(centripetal))**2), base_sigma, max_sigma))

    R_cov = np.diag([sigma_pitch**2, sigma_roll**2]).astype(np.float64)

    return es_ekf.update_measurement(z, h_x, H, R_cov, "dynamic_gravity_alignment", gate_threshold=es_ekf.config.chi2_gate_2d)


@dataclass
class Phase34WindowResult:
    window_id: str
    drive_id: str
    duration_s: float
    regime: str
    config_name: str
    distance_m: float
    final_pos_error_m: float
    drift_percent: float
    pitch_error_deg: float
    roll_error_deg: float
    yaw_error_deg: float
    speed_rmse_mps: float
    ai_accepted: int
    ai_rejected: int
    leveling_accepted: int
    leveling_rejected: int


def run_phase34_simulation(
    window: BlackoutWindowSpec,
    config_name: str,
    vel_model: Optional[VelocityEstimatorNet],
) -> Phase34WindowResult:
    """Simulate a single blackout window under the specified gravity alignment configuration."""
    imu, gps, v_speed, t = get_cached_drive_data(window.drive_path, window.drive_id)
    dt = 0.1
    start_sim = max(0, window.start_sample - window.warmup_samples)
    end_sim = min(len(t), window.end_sample + 50)
    bo_start = window.start_sample
    bo_end = window.end_sample
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

    aligner = PhoneToVehicleAligner()
    aligner.estimate_from_stationary_and_motion(imu[0:300, 0:3], imu[300:600, 0:3])
    acc_v_all, gyro_v_all = aligner.transform_imu(imu[:, 0:3], imu[:, 3:6])

    # Precompute AI model inferences
    ai_speeds = np.zeros(N_sim)
    if vel_model is not None:
        vel_model.eval()
        wins = []
        for k, idx in enumerate(range(start_sim, end_sim)):
            w_start = max(0, idx - 49)
            win = np.hstack([acc_v_all[w_start:idx + 1], gyro_v_all[w_start:idx + 1]]).T
            if win.shape[1] < 50:
                win = np.pad(win, ((0, 0), (50 - win.shape[1], 0)), mode="edge")
            wins.append(win)
        wins_arr = np.stack(wins, axis=0)
        with torch.no_grad():
            wins_t = torch.tensor(wins_arr, dtype=torch.float32)
            preds = vel_model(wins_t).squeeze(-1)
            ai_speeds = np.clip(preds.cpu().numpy(), 0.0, None)

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
    last_gps_raw = None
    prev_spd = spd_0

    k_bo_start = bo_start - start_sim
    k_bo_end = bo_end - start_sim

    est_p = np.zeros((N_sim, 3))
    est_v = np.zeros((N_sim, 3))
    est_rpy = np.zeros((N_sim, 3))

    ai_acc_cnt = 0
    ai_rej_cnt = 0
    lev_acc_cnt = 0
    lev_rej_cnt = 0

    for k, idx in enumerate(range(start_sim, end_sim)):
        is_blackout = (idx >= bo_start) and (idx < bo_end)
        acc_3d = acc_v_all[idx].astype(np.float64)
        gyro_3d = gyro_v_all[idx].astype(np.float64)

        # 1. Prediction step
        es_ekf.predict(acc_3d, gyro_3d, dt=dt)
        cur_v_b = es_ekf.rotation_matrix.T @ es_ekf.v
        fwd_spd = float(cur_v_b[0])
        acc_x_kin = (fwd_spd - prev_spd) / dt
        prev_spd = fwd_spd
        eff_yr = float(gyro_3d[2] - es_ekf.bg[2])

        # 2. Dynamic Gravity Alignment execution according to config
        if config_name == "baseline_phase33":
            if abs(acc_x_kin) < 2.0 and abs(eff_yr) < 0.2:
                acc_l, _ = es_ekf.update_gravity_leveling(acc_3d, forward_speed=fwd_spd, yaw_rate=eff_yr, fwd_acc=acc_x_kin, sigma_level=0.3)
                if acc_l: lev_acc_cnt += 1
                else: lev_rej_cnt += 1
        elif config_name == "adaptive_kinematic_sigma":
            acc_l, _ = run_adaptive_gravity_leveling_step(
                es_ekf, acc_3d, forward_speed=fwd_spd, yaw_rate=eff_yr, fwd_acc=acc_x_kin,
                base_sigma=0.3, centripetal_noise_coeff=0.8, longitudinal_noise_coeff=0.6,
                norm_gate_tolerance=10.0
            )
            if acc_l: lev_acc_cnt += 1
            else: lev_rej_cnt += 1
        elif config_name == "quasi_steady_norm_gate":
            if abs(np.linalg.norm(acc_3d) - 9.80665) < 1.2:
                acc_l, _ = es_ekf.update_gravity_leveling(acc_3d, forward_speed=fwd_spd, yaw_rate=eff_yr, fwd_acc=acc_x_kin, sigma_level=0.3)
                if acc_l: lev_acc_cnt += 1
                else: lev_rej_cnt += 1
        elif config_name == "dynamic_gravity_alignment_full":
            acc_l, _ = run_adaptive_gravity_leveling_step(
                es_ekf, acc_3d, forward_speed=fwd_spd, yaw_rate=eff_yr, fwd_acc=acc_x_kin,
                base_sigma=0.25, centripetal_noise_coeff=0.6, longitudinal_noise_coeff=0.5,
                norm_gate_tolerance=1.8
            )
            if acc_l: lev_acc_cnt += 1
            else: lev_rej_cnt += 1
        elif config_name == "no_gravity_leveling":
            pass  # Completely disabled

        # 3. Stationary / ZUPT
        cur_spd_est = float(np.linalg.norm(es_ekf.v[:2]))
        if ai_speeds[k] > 0.1:
            cur_spd_est = max(cur_spd_est, ai_speeds[k])
        is_stat = stat_detector.update(acc_3d, gyro_3d, speed_mps=cur_spd_est)
        if is_stat:
            es_ekf.update_zupt(sigma_v=0.01)
            es_ekf.update_zaru(gyro_3d, sigma_bg=0.001)

        # 4. GNSS or Blackout updates
        cur_gps_raw = (float(gps[idx, 0]), float(gps[idx, 1]))
        is_new_gps = (last_gps_raw is None) or (abs(cur_gps_raw[0] - last_gps_raw[0]) > 1e-9 or abs(cur_gps_raw[1] - last_gps_raw[1]) > 1e-9)

        if not is_blackout:
            if is_new_gps:
                last_gps_raw = cur_gps_raw
                gnss_pos = gt_enu[k]
                es_ekf.update_gnss_pos(gnss_pos, R_cov=np.eye(3) * 9.0, is_trusted=True)
                gnss_spd = float(gps[idx, 2]) if gps.shape[1] > 2 else float(np.linalg.norm(gt_vel[k, :2]))
                gnss_hdg = float(gps[idx, 3]) if gps.shape[1] > 3 else 0.0
                if gnss_spd >= 1.5:
                    psi_meas = np.deg2rad(90.0 - gnss_hdg)
                    es_ekf.update_heading(psi_meas, sigma_yaw=0.05)
                    v_e = gnss_spd * np.cos(psi_meas)
                    v_n = gnss_spd * np.sin(psi_meas)
                    es_ekf.update_gnss_vel(np.array([v_e, v_n, 0.0]), R_cov=np.eye(3) * 0.25, is_trusted=True)
        else:
            # During Blackout
            # NHC
            es_ekf.update_nhc(sigma_lat=0.05, sigma_vert=0.05, cornering_threshold_mps2=0.5)
            # AI Speed Update
            acc_ai, _ = es_ekf.update_ai_velocity(float(ai_speeds[k]), sigma_v=0.8, max_innovation_sigma=4.0)
            if acc_ai: ai_acc_cnt += 1
            else: ai_rej_cnt += 1

        est_p[k] = es_ekf.p.copy()
        est_v[k] = es_ekf.v.copy()
        est_rpy[k] = es_ekf.euler_angles

    bo_est_p = est_p[k_bo_start:k_bo_end]
    bo_gt_p = gt_enu[k_bo_start:k_bo_end]
    bo_est_v = est_v[k_bo_start:k_bo_end]
    bo_gt_v = gt_vel[k_bo_start:k_bo_end]

    final_pos_err = float(np.linalg.norm(bo_est_p[-1, :2] - bo_gt_p[-1, :2]))
    dist_m = max(1.0, float(np.sum(np.linalg.norm(np.diff(bo_gt_p[:, :2], axis=0), axis=1))))
    drift_pct = (final_pos_err / dist_m) * 100.0

    r_final, p_final, y_final = est_rpy[k_bo_end - 1]
    yaw_err_deg = float(np.rad2deg(min(abs(y_final - gt_yaw[k_bo_end - 1]), 2 * np.pi - abs(y_final - gt_yaw[k_bo_end - 1]))))
    pitch_err_deg = float(np.rad2deg(abs(p_final)))
    roll_err_deg = float(np.rad2deg(abs(r_final)))

    spd_est = np.linalg.norm(bo_est_v[:, :2], axis=1)
    spd_gt = np.linalg.norm(bo_gt_v[:, :2], axis=1)
    spd_rmse = float(np.sqrt(np.mean((spd_est - spd_gt)**2)))

    return Phase34WindowResult(
        window_id=window.window_id,
        drive_id=window.drive_id,
        duration_s=window.duration_s,
        regime=window.regime,
        config_name=config_name,
        distance_m=dist_m,
        final_pos_error_m=final_pos_err,
        drift_percent=drift_pct,
        pitch_error_deg=pitch_err_deg,
        roll_error_deg=roll_err_deg,
        yaw_error_deg=yaw_err_deg,
        speed_rmse_mps=spd_rmse,
        ai_accepted=ai_acc_cnt,
        ai_rejected=ai_rej_cnt,
        leveling_accepted=lev_acc_cnt,
        leveling_rejected=lev_rej_cnt,
    )


def run_phase34_benchmark():
    set_seed(42)
    logger.info("=== Starting Phase 34 Dynamic Gravity Alignment Benchmark ===")

    # 1. Load Model
    model_path = Path("models/authentic/velocity_net.pt")
    vel_model = None
    if model_path.exists():
        vel_model = VelocityEstimatorNet(in_channels=6, hidden_dim=64, num_layers=2)
        vel_model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        vel_model.eval()

    # 2. Load Benchmark Windows
    data_dir = Path("data/raw/categorised_authentic")
    m_dir = data_dir / "M (Driver B)"
    y1_dir = data_dir / "Y (Driver D)" / "Y1"

    windows_m = select_benchmark_windows(m_dir, "M", "validation")
    windows_y1 = select_benchmark_windows(y1_dir, "Y1", "held_out_test")
    all_windows = windows_m + windows_y1

    configs = [
        "baseline_phase33",
        "adaptive_kinematic_sigma",
        "quasi_steady_norm_gate",
        "dynamic_gravity_alignment_full",
        "no_gravity_leveling",
    ]

    all_results: Dict[str, List[Phase34WindowResult]] = {c: [] for c in configs}

    for cfg in configs:
        logger.info("Evaluating configuration: %s", cfg)
        for w in all_windows:
            res = run_phase34_simulation(w, cfg, vel_model)
            all_results[cfg].append(res)

    # Process and summarize results
    summary: Dict[str, Any] = {}
    for cfg in configs:
        recs = all_results[cfg]
        drifts = [r.drift_percent for r in recs]
        errs = [r.final_pos_error_m for r in recs]
        p_errs = [r.pitch_error_deg for r in recs]
        r_errs = [r.roll_error_deg for r in recs]
        y_errs = [r.yaw_error_deg for r in recs]
        spd_rmses = [r.speed_rmse_mps for r in recs]

        # Breakdowns by duration
        d10 = [r.drift_percent for r in recs if r.duration_s == 10.0]
        d30 = [r.drift_percent for r in recs if r.duration_s == 30.0]
        d60 = [r.drift_percent for r in recs if r.duration_s == 60.0]

        summary[cfg] = {
            "mean_drift_pct": float(np.mean(drifts)),
            "median_drift_pct": float(np.median(drifts)),
            "p90_drift_pct": float(np.percentile(drifts, 90)),
            "median_10s_drift_pct": float(np.median(d10)),
            "median_30s_drift_pct": float(np.median(d30)),
            "median_60s_drift_pct": float(np.median(d60)),
            "mean_final_error_m": float(np.mean(errs)),
            "median_final_error_m": float(np.median(errs)),
            "mean_pitch_error_deg": float(np.mean(p_errs)),
            "mean_roll_error_deg": float(np.mean(r_errs)),
            "mean_yaw_error_deg": float(np.mean(y_errs)),
            "mean_speed_rmse_mps": float(np.mean(spd_rmses)),
            "total_ai_accepted": sum(r.ai_accepted for r in recs),
            "total_ai_rejected": sum(r.ai_rejected for r in recs),
            "total_leveling_accepted": sum(r.leveling_accepted for r in recs),
            "total_leveling_rejected": sum(r.leveling_rejected for r in recs),
        }

    # Save JSON Report
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "PHASE34_DYNAMIC_GRAVITY_ALIGNMENT.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved JSON report to %s", json_path)

    print("\n" + "=" * 90)
    print("PHASE 34 DYNAMIC GRAVITY ALIGNMENT BENCHMARK SUMMARY")
    print("=" * 90)
    print(f"{'Configuration':<32} | {'Median Drift':<12} | {'P90 Drift':<12} | {'10s Drift':<10} | {'30s Drift':<10} | {'60s Drift':<10} | {'Mean Pitch':<10}")
    print("-" * 90)
    for cfg in configs:
        s = summary[cfg]
        print(f"{cfg:<32} | {s['median_drift_pct']:>10.1f}% | {s['p90_drift_pct']:>10.1f}% | {s['median_10s_drift_pct']:>8.1f}% | {s['median_30s_drift_pct']:>8.1f}% | {s['median_60s_drift_pct']:>8.1f}% | {s['mean_pitch_error_deg']:>8.2f}°")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    run_phase34_benchmark()
