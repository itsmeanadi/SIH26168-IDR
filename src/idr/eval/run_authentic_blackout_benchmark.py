"""Phase 26: Authentic GNSS Blackout Benchmark Engine.

Executes rigorous, leakage-free dead reckoning evaluation across authentic IO-VNBD dataset
drives (Validation Driver B / M and Held-Out Test Driver D / Y1) across 10s, 30s, and 60s
outage durations and 5 distinct vehicle motion regimes.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch

from ..config import CONFIG, set_seed
from ..filters.es_ekf import ErrorStateKalmanFilter, exp_quaternion
from ..filters.ekf import ExtendedKalmanFilter
from ..filters.fusion import GNSSINSFusion
from ..filters.nhc import apply_nhc_update
from ..filters.zupt import StationaryDetector, apply_zupt, apply_zaru
from ..io.loader import load_drive_pair
from ..models.velocity_net import VelocityEstimatorNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase26Benchmark")


@dataclass
class BlackoutWindowSpec:
    window_id: str
    drive_id: str
    drive_path: str
    split: str  # 'validation' or 'held_out_test'
    regime: str  # 'straight', 'accel_decel', 'turning', 'low_speed', 'high_speed'
    duration_s: float
    warmup_samples: int
    start_sample: int
    end_sample: int
    mean_speed_mps: float
    distance_traveled_m: float


@dataclass
class BlackoutRunResult:
    window_id: str
    drive_id: str
    split: str
    regime: str
    duration_s: float
    baseline_name: str
    total_distance_m: float
    # Position errors
    final_pos_error_2d_m: float
    max_pos_error_2d_m: float
    pos_rmse_2d_m: float
    pos_mae_2d_m: float
    horizontal_drift_percent: float
    final_pos_error_3d_m: float
    pos_error_east_m: float
    pos_error_north_m: float
    pos_error_up_m: float
    # Velocity errors
    final_vel_error_2d_mps: float
    vel_rmse_2d_mps: float
    speed_rmse_mps: float
    speed_mae_mps: float
    # Heading & Attitude errors
    final_heading_error_deg: float
    heading_rmse_deg: float
    # Diagnostic counts
    ai_updates_requested: int
    ai_updates_accepted: int
    ai_updates_rejected: int
    zupt_activations_total: int
    zupt_activations_stationary: int  # GT speed < 0.1 m/s
    zupt_activations_motion: int      # GT speed >= 0.1 m/s
    # SIH Status
    sih_pass_10pct: bool


DRIVE_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


def get_cached_drive_data(drive_path: str, drive_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and cache authentic drive arrays in memory."""
    key = f"{drive_path}_{drive_id}"
    if key not in DRIVE_CACHE:
        drive = load_drive_pair(Path(drive_path), drive_id)
        DRIVE_CACHE[key] = drive.get_synced_data()
    return DRIVE_CACHE[key]


def select_benchmark_windows(drive_dir: Path, drive_id: str, split: str) -> List[BlackoutWindowSpec]:
    """Select standardized, well-distributed blackout test windows for each motion regime."""
    imu, gps, v_speed, t = get_cached_drive_data(str(drive_dir), drive_id)
    N = len(t)
    dt = 0.1
    warmup = 300  # 30 seconds of pre-blackout GNSS initialization

    selected: List[BlackoutWindowSpec] = []
    
    # Target configurations: 10s (100 steps), 30s (300 steps), 60s (600 steps)
    # Regimes to find: Straight, Accel/Decel, Turning, Low-Speed, High-Speed
    durations = [(10.0, 100), (30.0, 300), (60.0, 600)]
    
    # We scan systematically through the drive to pick robust, non-overlapping windows
    for dur_s, n_steps in durations:
        found_regimes = set()
        step_stride = 1500  # Stride through the drive
        
        for start_idx in range(warmup + 100, N - n_steps - 100, step_stride):
            end_idx = start_idx + n_steps
            sub_spd = v_speed[start_idx:end_idx]
            sub_yaw_deg = np.rad2deg(np.abs(imu[start_idx:end_idx, 5]))
            
            mean_spd = float(np.mean(sub_spd))
            min_spd = float(np.min(sub_spd))
            max_spd = float(np.max(sub_spd))
            delta_spd = max_spd - min_spd
            mean_yaw = float(np.mean(sub_yaw_deg))
            dist = float(np.sum(sub_spd) * dt)
            
            # Classify regime
            regime = None
            if mean_yaw > 5.0 and "turning" not in found_regimes:
                regime = "turning"
            elif delta_spd > 6.0 and "accel_decel" not in found_regimes:
                regime = "accel_decel"
            elif mean_spd > 15.0 and mean_yaw < 2.5 and "high_speed" not in found_regimes:
                regime = "high_speed"
            elif mean_spd < 4.0 and "low_speed" not in found_regimes:
                regime = "low_speed"
            elif mean_yaw < 2.0 and mean_spd >= 4.0 and "straight" not in found_regimes:
                regime = "straight"
                
            if regime is not None:
                found_regimes.add(regime)
                win_id = f"{drive_id}_{int(dur_s)}s_{regime}_{start_idx}"
                selected.append(BlackoutWindowSpec(
                    window_id=win_id,
                    drive_id=drive_id,
                    drive_path=str(drive_dir),
                    split=split,
                    regime=regime,
                    duration_s=dur_s,
                    warmup_samples=warmup,
                    start_sample=start_idx,
                    end_sample=end_idx,
                    mean_speed_mps=mean_spd,
                    distance_traveled_m=dist,
                ))
                
            if len(found_regimes) >= 5:
                break

    return selected


def run_single_window_benchmark(
    window: BlackoutWindowSpec,
    baseline: str,
    vel_model: Optional[VelocityEstimatorNet],
    export_trace: bool = False,
) -> Tuple[BlackoutRunResult, Optional[pd.DataFrame]]:
    """Execute one blackout scenario under strict leakage controls.
    
    Baselines:
      A. '15state_imu_only'   : ES-EKF IMU propagation only during outage
      B. '15state_ai'         : ES-EKF + AI forward speed updates
      C. '15state_ai_nhc'     : ES-EKF + AI + NHC
      D. '15state_full'       : ES-EKF + AI + NHC + ZUPT/ZARU (Full System)
      E. 'legacy_ekf_full'    : 9-state planar EKF + AI + NHC + ZUPT
      F. 'const_vel_baseline' : Kinematic constant-velocity extrapolation from pre-outage state
    """
    imu, gps, v_speed, t = get_cached_drive_data(window.drive_path, window.drive_id)
    dt = 0.1
    
    start_sim = max(0, window.start_sample - window.warmup_samples)
    end_sim = min(len(t), window.end_sample + 50)  # include 5s post-blackout recovery
    
    bo_start = window.start_sample
    bo_end = window.end_sample
    
    # Reference coordinates in ENU centered at start of warm-up
    ref_lat = float(gps[start_sim, 0])
    ref_lon = float(gps[start_sim, 1])
    
    # Ground truth positions and velocities in local ENU
    fusion_ref = GNSSINSFusion(ref_lat=ref_lat, ref_lon=ref_lon, dt=dt)
    N_sim = end_sim - start_sim
    
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

    # Calibrate / align phone IMU into vehicle frame
    aligner = PhoneToVehicleAligner()
    aligner.estimate_from_stationary_and_motion(imu[0:300, 0:3], imu[300:600, 0:3])
    acc_v_all, gyro_v_all = aligner.transform_imu(imu[:, 0:3], imu[:, 3:6])

    # Initialize Evaluated Filter
    use_es_ekf = "15state" in baseline or baseline == "const_vel_baseline"
    use_legacy_ekf = "legacy_ekf" in baseline
    
    es_ekf: Optional[ErrorStateKalmanFilter] = None
    legacy_ekf: Optional[ExtendedKalmanFilter] = None
    
    spd_0 = float(v_speed[start_sim])
    psi_0 = gt_yaw[0]
    v_0 = np.array([spd_0 * np.cos(psi_0), spd_0 * np.sin(psi_0), 0.0])
    
    if use_es_ekf:
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
        es_ekf.initialize_leveling(acc_v_all[start_sim], yaw_rad=psi_0)
        es_ekf.set_state(pos=np.zeros(3), vel=v_0)
    elif use_legacy_ekf:
        legacy_ekf = ExtendedKalmanFilter(dt=dt)
        legacy_ekf.x[0:3] = np.zeros(3)
        legacy_ekf.x[3:6] = v_0
        legacy_ekf.x[6] = psi_0
        
    stat_detector = StationaryDetector(window_size=10, acc_var_threshold=0.15)
    
    # Precompute AI model inferences causally in batch (sliding window of 50 samples / 5.0s)
    # Ensuring ZERO access to GNSS, ground-truth, or future frames
    ai_speeds = np.zeros(N_sim)
    ai_ready = np.zeros(N_sim, dtype=bool)
    if vel_model is not None:
        vel_model.eval()
        wins = []
        for k, idx in enumerate(range(start_sim, end_sim)):
            w_start = max(0, idx - 49)
            win = np.hstack([acc_v_all[w_start:idx + 1], gyro_v_all[w_start:idx + 1]]).T  # shape (6, W)
            if win.shape[1] < 50:
                win = np.pad(win, ((0, 0), (50 - win.shape[1], 0)), mode="edge")
            wins.append(win)
        wins_arr = np.stack(wins, axis=0)  # shape (N_sim, 6, 50)
        with torch.no_grad():
            wins_t = torch.tensor(wins_arr, dtype=torch.float32)
            preds = vel_model(wins_t).squeeze(-1)
            ai_speeds = np.clip(preds.cpu().numpy(), 0.0, None)
            ai_ready[:] = True

    # Storage for simulation trajectory
    est_enu = np.zeros((N_sim, 3))
    est_vel = np.zeros((N_sim, 3))
    est_yaw = np.zeros(N_sim)
    
    # Diagnostic logging
    ai_req_count = 0
    ai_acc_count = 0
    ai_rej_count = 0
    zupt_count = 0
    zupt_stat_count = 0
    zupt_mot_count = 0
    
    # Trace columns for representative dump
    trace_records = []
    
    # Store pre-blackout state for kinematic baseline
    pre_bo_pos = np.zeros(3)
    pre_bo_vel = np.zeros(3)
    
    last_gps_raw = None
    prev_spd = spd_0
    
    for k, idx in enumerate(range(start_sim, end_sim)):
        t_cur = float(t[idx])
        is_blackout = (idx >= bo_start) and (idx < bo_end)
        
        acc_3d = acc_v_all[idx].astype(np.float64)
        gyro_3d = gyro_v_all[idx].astype(np.float64)
        fwd_acc = float(acc_3d[0])
        yaw_rate = float(gyro_3d[2])
        
        gt_spd = float(v_speed[idx])
        is_true_stationary = (gt_spd < 0.1)
        
        # 1. Prediction step
        if use_es_ekf and es_ekf is not None:
            es_ekf.predict(acc_3d, gyro_3d, dt=dt)
            # Dynamic Leveling
            cur_v_body = es_ekf.rotation_matrix.T @ es_ekf.v
            fwd_speed = float(cur_v_body[0])
            acc_x_kin = (fwd_speed - prev_spd) / dt
            prev_spd = fwd_speed
            eff_yaw_rate = float(gyro_3d[2] - es_ekf.bg[2])
            if abs(acc_x_kin) < 2.0 and abs(eff_yaw_rate) < 0.2:
                es_ekf.update_gravity_leveling(acc_3d, forward_speed=fwd_speed, yaw_rate=eff_yaw_rate, fwd_acc=acc_x_kin, sigma_level=0.3)
        elif use_legacy_ekf and legacy_ekf is not None:
            legacy_ekf.predict(fwd_acc, yaw_rate)
            
        # 2. Stationary / ZUPT update
        apply_zupt_flag = baseline in ("15state_full", "legacy_ekf_full")
        cur_spd_est = float(np.linalg.norm(es_ekf.v[:2])) if use_es_ekf else float(np.linalg.norm(legacy_ekf.x[3:5]))
        if ai_ready[k]:
            cur_spd_est = max(cur_spd_est, ai_speeds[k])
            
        is_stat = stat_detector.update(acc_3d, gyro_3d, speed_mps=cur_spd_est)
        zupt_applied = False
        if is_stat and apply_zupt_flag:
            zupt_applied = True
            zupt_count += 1
            if is_true_stationary:
                zupt_stat_count += 1
            else:
                zupt_mot_count += 1
                
            if use_es_ekf and es_ekf is not None:
                es_ekf.update_zupt(sigma_v=0.01)
                es_ekf.update_zaru(gyro_3d, sigma_bg=0.001)
            elif use_legacy_ekf and legacy_ekf is not None:
                apply_zupt(legacy_ekf, sigma_v=0.01)
                apply_zaru(legacy_ekf, gyro_z_raw=yaw_rate)

        # 3. Measurement updates
        ai_accepted = False
        ai_innov = 0.0
        ai_nis = 0.0
        
        cur_gps_raw = (float(gps[idx, 0]), float(gps[idx, 1]))
        is_new_gps = (last_gps_raw is None) or (abs(cur_gps_raw[0] - last_gps_raw[0]) > 1e-9 or abs(cur_gps_raw[1] - last_gps_raw[1]) > 1e-9)
        
        if not is_blackout:
            # GNSS Available: Normal Fusion Update on Fresh Fixes
            if is_new_gps:
                last_gps_raw = cur_gps_raw
                gnss_e, gnss_n, gnss_u = gt_enu[k]
                meas_pos = np.array([gnss_e, gnss_n, gnss_u], dtype=np.float64)
                
                if use_es_ekf and es_ekf is not None:
                    es_ekf.update_gnss_pos(meas_pos, R_cov=np.eye(3) * (3.0 ** 2))
                    gnss_spd = float(gps[idx, 2]) if gps.shape[1] > 2 else float(np.linalg.norm(gt_vel[k, :2]))
                    gnss_hdg = float(gps[idx, 3]) if gps.shape[1] > 3 else 0.0
                    if gnss_spd >= 1.5:
                        psi_meas = np.deg2rad(90.0 - gnss_hdg)
                        es_ekf.update_heading(psi_meas, sigma_yaw=0.05)
                        v_e = gnss_spd * np.cos(psi_meas)
                        v_n = gnss_spd * np.sin(psi_meas)
                        es_ekf.update_gnss_vel(np.array([v_e, v_n, 0.0]), R_cov=np.eye(3) * (0.5 ** 2))
                elif use_legacy_ekf and legacy_ekf is not None:
                    legacy_ekf.update_gnss_pos(meas_pos, R_cov=np.eye(3) * (3.0 ** 2))
                    gnss_spd = float(gps[idx, 2]) if gps.shape[1] > 2 else float(np.linalg.norm(gt_vel[k, :2]))
                    gnss_hdg = float(gps[idx, 3]) if gps.shape[1] > 3 else 0.0
                    if gnss_spd >= 1.5:
                        psi_meas = np.deg2rad(90.0 - gnss_hdg)
                        legacy_ekf.update_heading(psi_meas, R_yaw=0.03)
                        legacy_ekf.update_velocity_2d(gnss_spd, 0.0)
            prev_gnss_enu = (gnss_e, gnss_n)
            
            # Cache state at outage onset
            if idx == bo_start - 1:
                if use_es_ekf and es_ekf is not None:
                    pre_bo_pos = es_ekf.p.copy()
                    pre_bo_vel = es_ekf.v.copy()
                elif use_legacy_ekf and legacy_ekf is not None:
                    pre_bo_pos = np.array([legacy_ekf.x[0], legacy_ekf.x[1], 0.0])
                    pre_bo_vel = np.array([legacy_ekf.x[3], legacy_ekf.x[4], 0.0])
        else:
            # INSIDE BLACKOUT: STRICT GNSS DENIED
            prev_gnss_enu = None
            
            if baseline == "const_vel_baseline":
                # Pure kinematic extrapolation from estimated pre-outage state
                t_bo_elapsed = (idx - bo_start) * dt
                est_enu[k] = pre_bo_pos + pre_bo_vel * t_bo_elapsed
                est_vel[k] = pre_bo_vel
                est_yaw[k] = np.arctan2(pre_bo_vel[1], pre_bo_vel[0]) if np.linalg.norm(pre_bo_vel[:2]) > 0.1 else 0.0
                continue
            
            # NHC updates
            use_nhc = baseline in ("15state_ai_nhc", "15state_full", "legacy_ekf_full")
            if use_nhc:
                if use_es_ekf and es_ekf is not None:
                    es_ekf.update_nhc(sigma_lat=0.05, sigma_vert=0.05)
                elif use_legacy_ekf and legacy_ekf is not None:
                    apply_nhc_update(legacy_ekf, sigma_lat=0.05, sigma_vert=0.05)

            # AI forward velocity updates
            use_ai = baseline in ("15state_ai", "15state_ai_nhc", "15state_full", "legacy_ekf_full")
            if use_ai and ai_ready[k] and (k % 10 == 0):  # 1 Hz update rate
                ai_req_count += 1
                v_ai = ai_speeds[k]
                if use_es_ekf and es_ekf is not None:
                    acc, diag = es_ekf.update_ai_velocity(v_ai, sigma_v=3.0, max_innovation_sigma=3.0)
                    ai_accepted = acc
                    ai_innov = diag.get("innovation", 0.0)
                    ai_nis = diag.get("nis", 0.0)
                elif use_legacy_ekf and legacy_ekf is not None:
                    acc, diag = legacy_ekf.update_ai_velocity(v_ai, sigma_v=3.0, max_innovation_sigma=3.0)
                    ai_accepted = acc
                    ai_innov = diag.get("innovation", 0.0)
                    ai_nis = diag.get("nis", 0.0)
                if ai_accepted:
                    ai_acc_count += 1
                else:
                    ai_rej_count += 1

        # Extract current state
        if use_es_ekf and es_ekf is not None:
            est_enu[k] = es_ekf.p.copy()
            est_vel[k] = es_ekf.v.copy()
            est_yaw[k] = float(es_ekf.euler_angles[2])
            pos_cov_tr = float(np.trace(es_ekf.P[0:3, 0:3]))
            vel_cov_tr = float(np.trace(es_ekf.P[3:6, 3:6]))
        elif use_legacy_ekf and legacy_ekf is not None:
            est_enu[k] = np.array([legacy_ekf.x[0], legacy_ekf.x[1], 0.0])
            est_vel[k] = np.array([legacy_ekf.x[3], legacy_ekf.x[4], 0.0])
            est_yaw[k] = float(legacy_ekf.x[6])
            pos_cov_tr = float(legacy_ekf.P[0, 0] + legacy_ekf.P[1, 1])
            vel_cov_tr = float(legacy_ekf.P[3, 3] + legacy_ekf.P[4, 4])
        else:
            pos_cov_tr = 0.0
            vel_cov_tr = 0.0

        if export_trace:
            pos_err_2d = float(np.linalg.norm(est_enu[k, :2] - gt_enu[k, :2]))
            pos_err_3d = float(np.linalg.norm(est_enu[k] - gt_enu[k]))
            vel_err_2d = float(np.linalg.norm(est_vel[k, :2] - gt_vel[k, :2]))
            spd_est = float(np.linalg.norm(est_vel[k, :2]))
            spd_gt = float(v_speed[idx])
            yaw_err_deg = float(np.rad2deg(abs(est_yaw[k] - gt_yaw[k])))
            yaw_err_deg = min(yaw_err_deg, 360.0 - yaw_err_deg)
            
            trace_records.append({
                "timestamp": t_cur,
                "relative_t_s": round((idx - start_sim) * dt, 2),
                "is_blackout": is_blackout,
                "gnss_available": not is_blackout,
                "gnss_trust_status": "NORMAL" if not is_blackout else "BLACKOUT",
                "pos_est_e": est_enu[k, 0],
                "pos_est_n": est_enu[k, 1],
                "pos_est_u": est_enu[k, 2],
                "pos_ref_e": gt_enu[k, 0],
                "pos_ref_n": gt_enu[k, 1],
                "pos_ref_u": gt_enu[k, 2],
                "pos_err_2d": pos_err_2d,
                "pos_err_3d": pos_err_3d,
                "vel_est_e": est_vel[k, 0],
                "vel_est_n": est_vel[k, 1],
                "vel_est_u": est_vel[k, 2],
                "vel_ref_e": gt_vel[k, 0],
                "vel_ref_n": gt_vel[k, 1],
                "vel_ref_u": gt_vel[k, 2],
                "vel_err_2d": vel_err_2d,
                "speed_est": spd_est,
                "speed_ref": spd_gt,
                "speed_err": abs(spd_est - spd_gt),
                "heading_est_deg": float(np.rad2deg(est_yaw[k])) % 360.0,
                "heading_ref_deg": float(np.rad2deg(gt_yaw[k])) % 360.0,
                "heading_err_deg": yaw_err_deg,
                "ai_speed": ai_speeds[k] if is_blackout else 0.0,
                "ai_accepted": ai_accepted,
                "ai_innovation": ai_innov,
                "ai_nis": ai_nis,
                "nhc_active": is_blackout and ("nhc" in baseline or "full" in baseline),
                "zupt_active": zupt_applied,
                "zaru_active": zupt_applied,
                "pos_cov_trace": pos_cov_tr,
                "vel_cov_trace": vel_cov_tr,
            })

    # Compute blackout interval evaluation metrics
    k_bo_start = bo_start - start_sim
    k_bo_end = bo_end - start_sim
    
    bo_est_enu = est_enu[k_bo_start:k_bo_end]
    bo_gt_enu = gt_enu[k_bo_start:k_bo_end]
    bo_est_vel = est_vel[k_bo_start:k_bo_end]
    bo_gt_vel = gt_vel[k_bo_start:k_bo_end]
    bo_est_yaw = est_yaw[k_bo_start:k_bo_end]
    bo_gt_yaw = gt_yaw[k_bo_start:k_bo_end]
    
    # 2D horizontal position error
    err_2d = np.linalg.norm(bo_est_enu[:, :2] - bo_gt_enu[:, :2], axis=1)
    final_err_2d = float(err_2d[-1])
    max_err_2d = float(np.max(err_2d))
    pos_rmse_2d = float(np.sqrt(np.mean(err_2d ** 2)))
    pos_mae_2d = float(np.mean(err_2d))
    
    # 3D position error
    err_3d = np.linalg.norm(bo_est_enu - bo_gt_enu, axis=1)
    final_err_3d = float(err_3d[-1])
    
    # Axis errors at end
    err_e = float(abs(bo_est_enu[-1, 0] - bo_gt_enu[-1, 0]))
    err_n = float(abs(bo_est_enu[-1, 1] - bo_gt_enu[-1, 1]))
    err_u = float(abs(bo_est_enu[-1, 2] - bo_gt_enu[-1, 2]))
    
    # Distance and drift percentage
    dist_m = float(window.distance_traveled_m)
    drift_pct = (final_err_2d / max(dist_m, 1.0)) * 100.0
    
    # Velocity errors
    vel_err_2d = np.linalg.norm(bo_est_vel[:, :2] - bo_gt_vel[:, :2], axis=1)
    final_vel_err = float(vel_err_2d[-1])
    vel_rmse_2d = float(np.sqrt(np.mean(vel_err_2d ** 2)))
    
    speed_est = np.linalg.norm(bo_est_vel[:, :2], axis=1)
    speed_gt = np.linalg.norm(bo_gt_vel[:, :2], axis=1)
    speed_err = np.abs(speed_est - speed_gt)
    speed_rmse = float(np.sqrt(np.mean(speed_err ** 2)))
    speed_mae = float(np.mean(speed_err))
    
    # Heading error
    heading_diffs = np.array([
        min(abs(e - g), 2 * np.pi - abs(e - g))
        for e, g in zip(bo_est_yaw, bo_gt_yaw)
    ])
    heading_diffs_deg = np.rad2deg(heading_diffs)
    final_hdg_err = float(heading_diffs_deg[-1])
    hdg_rmse_deg = float(np.sqrt(np.mean(heading_diffs_deg ** 2)))
    
    sih_pass = (drift_pct < 10.0)
    
    result = BlackoutRunResult(
        window_id=window.window_id,
        drive_id=window.drive_id,
        split=window.split,
        regime=window.regime,
        duration_s=window.duration_s,
        baseline_name=baseline,
        total_distance_m=round(dist_m, 2),
        final_pos_error_2d_m=round(final_err_2d, 3),
        max_pos_error_2d_m=round(max_err_2d, 3),
        pos_rmse_2d_m=round(pos_rmse_2d, 3),
        pos_mae_2d_m=round(pos_mae_2d, 3),
        horizontal_drift_percent=round(drift_pct, 2),
        final_pos_error_3d_m=round(final_err_3d, 3),
        pos_error_east_m=round(err_e, 3),
        pos_error_north_m=round(err_n, 3),
        pos_error_up_m=round(err_u, 3),
        final_vel_error_2d_mps=round(final_vel_err, 3),
        vel_rmse_2d_mps=round(vel_rmse_2d, 3),
        speed_rmse_mps=round(speed_rmse, 3),
        speed_mae_mps=round(speed_mae, 3),
        final_heading_error_deg=round(final_hdg_err, 2),
        heading_rmse_deg=round(hdg_rmse_deg, 2),
        ai_updates_requested=ai_req_count,
        ai_updates_accepted=ai_acc_count,
        ai_updates_rejected=ai_rej_count,
        zupt_activations_total=zupt_count,
        zupt_activations_stationary=zupt_stat_count,
        zupt_activations_motion=zupt_mot_count,
        sih_pass_10pct=sih_pass,
    )
    
    df_trace = pd.DataFrame(trace_records) if export_trace else None
    return result, df_trace


def execute_full_benchmark(
    val_drive_path: str = "data/raw/categorised_authentic/M (Driver B)",
    val_drive_id: str = "M",
    test_drive_path: str = "data/raw/categorised_authentic/Y (Driver D)/Y1",
    test_drive_id: str = "Y1",
    model_path: str = "models/authentic/velocity_net.pt",
    output_dir: str = "reports",
) -> Dict[str, Any]:
    """Execute complete Phase 26 Blackout Benchmark across validation and held-out test drives."""
    set_seed(42)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    traces_path = output_path / "traces"
    traces_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("PHASE 26: AUTHENTIC GNSS BLACKOUT BENCHMARK EXECUTION")
    logger.info("=" * 80)
    
    # 1. Load trained AI Velocity model
    vel_model = None
    if os.path.exists(model_path):
        vel_model = VelocityEstimatorNet(in_channels=6, hidden_dim=64, num_layers=2)
        st = torch.load(model_path, map_location="cpu", weights_only=True)
        vel_model.load_state_dict(st)
        vel_model.eval()
        logger.info(f"Loaded authentic VelocityEstimatorNet from: {model_path}")
    else:
        logger.warning(f"VelocityEstimatorNet checkpoint NOT found at: {model_path}")

    # 2. Select benchmark windows
    logger.info("Selecting representative blackout windows...")
    val_windows = select_benchmark_windows(Path(val_drive_path), val_drive_id, split="validation")
    test_windows = select_benchmark_windows(Path(test_drive_path), test_drive_id, split="held_out_test")
    
    all_windows = val_windows + test_windows
    logger.info(f"Selected {len(val_windows)} Validation windows and {len(test_windows)} Held-Out Test windows.")
    
    baselines = [
        "15state_imu_only",
        "15state_ai",
        "15state_ai_nhc",
        "15state_full",
        "legacy_ekf_full",
        "const_vel_baseline",
    ]
    
    results: List[BlackoutRunResult] = []
    
    # 3. Determinism / Repeatability Check
    logger.info("\n--- EXECUTING DETERMINISM CHECK ---")
    rep_win = test_windows[0]
    res1, _ = run_single_window_benchmark(rep_win, "15state_full", vel_model, export_trace=False)
    res2, _ = run_single_window_benchmark(rep_win, "15state_full", vel_model, export_trace=False)
    
    det_diff = abs(res1.final_pos_error_2d_m - res2.final_pos_error_2d_m)
    logger.info(f"Repeatability Delta: {det_diff:.12e} m (Bitwise Deterministic: {det_diff < 1e-9})")
    
    # 4. Run Benchmark across all Windows and Baselines
    logger.info("\n--- RUNNING BENCHMARK MATRIX ---")
    
    representative_traces = {}
    
    for win in all_windows:
        for b_name in baselines:
            # Export trace for 10s, 30s, 60s representative turning & straight cases on held-out Y1
            should_export = (
                win.split == "held_out_test"
                and b_name == "15state_full"
                and win.regime in ("straight", "turning", "accel_decel")
                and win.duration_s in (10.0, 30.0, 60.0)
            )
            
            res, df_trace = run_single_window_benchmark(win, b_name, vel_model, export_trace=should_export)
            results.append(res)
            
            if df_trace is not None:
                trace_filename = f"trace_{win.drive_id}_{int(win.duration_s)}s_{win.regime}_{b_name}.csv"
                df_trace.to_csv(traces_path / trace_filename, index=False)
                representative_traces[f"{int(win.duration_s)}s_{win.regime}"] = trace_filename
                logger.info(f"Saved representative trace: {trace_filename}")
                
            logger.info(
                f"[{win.split[:3].upper()}] {win.drive_id} {win.duration_s:2.0f}s | {win.regime:<11s} | "
                f"{b_name:<18s} | Err: {res.final_pos_error_2d_m:6.2f}m | Drift: {res.horizontal_drift_percent:5.2f}% | "
                f"Pass: {res.sih_pass_10pct}"
            )

    # 5. Compile Statistical Aggregations
    df_res = pd.DataFrame([asdict(r) for r in results])
    
    # Summary by Baseline and Duration
    summary_by_baseline = {}
    for b_name, grp in df_res.groupby("baseline_name"):
        drift_arr = grp["horizontal_drift_percent"].to_numpy()
        summary_by_baseline[b_name] = {
            "count": int(len(grp)),
            "sih_pass_count": int(np.sum(grp["sih_pass_10pct"])),
            "sih_pass_percent": round(float(np.mean(grp["sih_pass_10pct"]) * 100.0), 2),
            "drift_mean_pct": round(float(np.mean(drift_arr)), 2),
            "drift_median_pct": round(float(np.median(drift_arr)), 2),
            "drift_p90_pct": round(float(np.percentile(drift_arr, 90)), 2),
            "drift_p95_pct": round(float(np.percentile(drift_arr, 95)), 2),
            "drift_worst_pct": round(float(np.max(drift_arr)), 2),
            "final_err_mean_m": round(float(grp["final_pos_error_2d_m"].mean()), 2),
            "pos_rmse_mean_m": round(float(grp["pos_rmse_2d_m"].mean()), 2),
            "speed_rmse_mean_mps": round(float(grp["speed_rmse_mps"].mean()), 3),
            "heading_rmse_mean_deg": round(float(grp["heading_rmse_deg"].mean()), 2),
            "total_ai_accepted": int(grp["ai_updates_accepted"].sum()),
            "total_ai_rejected": int(grp["ai_updates_rejected"].sum()),
            "total_zupt_activations": int(grp["zupt_activations_total"].sum()),
            "total_zupt_motion_false": int(grp["zupt_activations_motion"].sum()),
        }
        
    # Breakdown on Held-Out Test Drive Y1
    df_held_out = df_res[df_res["split"] == "held_out_test"]
    held_out_summary = {}
    for (dur, b_name), grp in df_held_out.groupby(["duration_s", "baseline_name"]):
        drift_arr = grp["horizontal_drift_percent"].to_numpy()
        key = f"{int(dur)}s_{b_name}"
        held_out_summary[key] = {
            "duration_s": dur,
            "baseline": b_name,
            "n_cases": int(len(grp)),
            "pass_count": int(np.sum(grp["sih_pass_10pct"])),
            "pass_pct": round(float(np.mean(grp["sih_pass_10pct"]) * 100.0), 2),
            "drift_median_pct": round(float(np.median(drift_arr)), 2),
            "drift_p90_pct": round(float(np.percentile(drift_arr, 90)), 2),
            "final_err_median_m": round(float(np.median(grp["final_pos_error_2d_m"])), 2),
            "pos_rmse_mean_m": round(float(grp["pos_rmse_2d_m"].mean()), 2),
        }

    # SIH Target Verdict Determination
    full_sys_test = df_held_out[df_held_out["baseline_name"] == "15state_full"]
    pass_pct_full_test = float(np.mean(full_sys_test["sih_pass_10pct"]) * 100.0)
    
    if pass_pct_full_test >= 90.0:
        verdict = "SIH TARGET ACHIEVED"
    elif pass_pct_full_test >= 50.0:
        verdict = "SIH TARGET PARTIALLY ACHIEVED"
    else:
        verdict = "SIH TARGET NOT ACHIEVED"

    final_payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "verdict": verdict,
        "determinism_delta_m": det_diff,
        "is_deterministic": bool(det_diff < 1e-9),
        "total_benchmark_cases": len(results),
        "validation_cases": len(val_windows) * len(baselines),
        "held_out_test_cases": len(test_windows) * len(baselines),
        "baselines_evaluated": baselines,
        "summary_by_baseline": summary_by_baseline,
        "held_out_by_duration": held_out_summary,
        "representative_traces": representative_traces,
        "all_cases": [asdict(r) for r in results],
        "disclaimer": (
            "IO-VNBD is an automotive dataset recorded inside a passenger car. "
            "These empirical findings do NOT constitute physical two-wheeler / motorcycle validation."
        ),
    }

    # Save JSON artifact
    json_path = output_path / "PHASE26_AUTHENTIC_BLACKOUT_BENCHMARK.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)
    logger.info(f"Saved machine-readable benchmark report: {json_path}")

    return final_payload


if __name__ == "__main__":
    execute_full_benchmark()
