"""Phase 31A: Magnetometer & Heading Stabilization Ablation Benchmark Runner.

Evaluates 6 standardized configurations across all 30 authentic blackout windows (15 Drive M, 15 Drive Y1):
A. phase30a_baseline                : Phase-30A ES-EKF (unlocked GNSS gate, decoupled NHC, no mag)
B. mag_only                         : Phase-30A + tilt-compensated magnetometer heading update with causal gating
C. heading_stabilization_only       : Phase-30A + pre-blackout heading stabilization (standstill ZARU/ZUPT, dynamic COG)
D. mag_plus_heading_stabilization   : Mag heading + Pre-blackout heading stabilization
E. turn_aware_nhc_only              : Phase-30A + turn-aware lateral NHC disable during cornering
F. phase31a_all_mechanisms          : All Phase-31A improvements combined (Mag + Pre-BO Heading Stab + Turn-aware NHC)

Strict Scientific Controls:
- Frozen VelocityEstimatorNet weights (no retraining)
- Frozen Driver D / Drive Y1 test set (no tuning)
- Exact same authentic 30 blackout windows
- Zero future data, zero GT injection
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
import torch

from ..calib.alignment import PhoneToVehicleAligner
from ..calib.mag_calib import MagnetometerCalibrator
from ..sensors.mag_heading import compute_tilt_compensated_heading
from ..sensors.mag_disturbance import MagneticDisturbanceDetector
from ..filters.es_ekf import ErrorStateKalmanFilter, ESEKFConfig
from ..filters.ekf import ExtendedKalmanFilter
from ..filters.fusion import GNSSINSFusion
from ..filters.zupt import StationaryDetector
from ..models.train_all import VelocityEstimatorNet
from ..io.preprocess import load_drive_pair
from .run_phase28_full_benchmark import (
    BlackoutWindowSpec,
    select_benchmark_windows,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DRIVE_WITH_MAG_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


def get_cached_drive_data_with_mag(
    drive_path: str, drive_id: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and cache authentic drive arrays with 3D magnetometer in memory."""
    key = f"{drive_path}_{drive_id}"
    if key not in DRIVE_WITH_MAG_CACHE:
        drive = load_drive_pair(Path(drive_path), drive_id)
        DRIVE_WITH_MAG_CACHE[key] = drive.get_synced_data_with_mag()
    return DRIVE_WITH_MAG_CACHE[key]


@dataclass
class Phase31AResultRecord:
    window_id: str
    drive_id: str
    split: str
    regime: str
    duration_s: float
    config_name: str
    distance_m: float
    # Pre-blackout entry errors
    pre_bo_pos_error_m: float
    pre_bo_vel_error_mps: float
    pre_bo_yaw_error_deg: float
    # Outage error metrics
    final_pos_error_2d_m: float
    max_pos_error_2d_m: float
    pos_rmse_2d_m: float
    drift_percent: float
    speed_rmse_mps: float
    # Heading metrics at checkpoints
    heading_error_10s_deg: float
    heading_error_30s_deg: float
    heading_error_60s_deg: float
    final_heading_error_deg: float
    heading_rmse_deg: float
    final_yaw_std_deg: float
    # Acceptance & Diagnostic metrics
    ai_accepted: int
    ai_rejected: int
    mag_accepted: int
    mag_rejected: int
    mag_disturbance_count: int
    nhc_active_count: int
    nhc_turn_disabled_count: int
    sih_pass_10pct: bool


def run_single_phase31a_simulation(
    window: BlackoutWindowSpec,
    config_name: str,
    vel_model: Optional[VelocityEstimatorNet],
) -> Phase31AResultRecord:
    """Execute a single window simulation under the designated Phase 31A configuration."""
    imu, gps, v_speed, t, mag = get_cached_drive_data_with_mag(window.drive_path, window.drive_id)
    dt = 0.1

    start_sim = max(0, window.start_sample - window.warmup_samples)
    end_sim = min(len(t), window.end_sample + 50)

    bo_start = window.start_sample
    bo_end = window.end_sample

    ref_lat = float(gps[start_sim, 0])
    ref_lon = float(gps[start_sim, 1])
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

    # Phone-to-vehicle IMU alignment
    aligner = PhoneToVehicleAligner()
    aligner.estimate_from_stationary_and_motion(imu[0:300, 0:3], imu[300:600, 0:3])
    acc_v_all, gyro_v_all = aligner.transform_imu(imu[:, 0:3], imu[:, 3:6])

    # Magnetometer calibrator & disturbance detector
    mag_calib = MagnetometerCalibrator(R_p2v=aligner.R_phone_to_vehicle)
    # Fit hard-iron offset on pre-blackout warmup window
    mag_calib.fit_min_max_sphere(mag[start_sim:bo_start])
    mag_v_all = mag_calib.transform_and_calibrate(mag)

    mag_detector = MagneticDisturbanceDetector(
        ref_field_uT=45.0,
        norm_tolerance_uT=12.0,
        max_variance_uT2=25.0,
        max_rate_discrepancy_deg_s=45.0,
    )

    # Configuration feature toggles:
    # A. 'phase30a_baseline'              : use_mag=False, pre_bo_stab=False, turn_nhc='decouple'
    # B. 'mag_only'                       : use_mag=True,  pre_bo_stab=False, turn_nhc='decouple'
    # C. 'heading_stabilization_only'     : use_mag=False, pre_bo_stab=True,  turn_nhc='decouple'
    # D. 'mag_plus_heading_stabilization' : use_mag=True,  pre_bo_stab=True,  turn_nhc='decouple'
    # E. 'turn_aware_nhc_only'            : use_mag=False, pre_bo_stab=False, turn_nhc='disable_lateral'
    # F. 'phase31a_all_mechanisms'        : use_mag=True,  pre_bo_stab=True,  turn_nhc='disable_lateral'

    use_mag = config_name in ("mag_only", "mag_plus_heading_stabilization", "phase31a_all_mechanisms")
    pre_bo_stab = config_name in ("heading_stabilization_only", "mag_plus_heading_stabilization", "phase31a_all_mechanisms")
    turn_nhc_policy = "disable_lateral" if config_name in ("turn_aware_nhc_only", "phase31a_all_mechanisms") else "decouple"

    # Precompute causal AI predictions (50-sample sliding window)
    ai_speeds = np.zeros(N_sim)
    ai_ready = np.zeros(N_sim, dtype=bool)
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
            ai_ready[:] = True

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
    est_vel = np.zeros((N_sim, 3))
    est_yaw = np.zeros(N_sim)
    est_yaw_std = np.zeros(N_sim)

    ai_acc_count = 0
    ai_rej_count = 0
    mag_acc_count = 0
    mag_rej_count = 0
    mag_dist_count = 0
    nhc_active_count = 0
    nhc_turn_disabled_count = 0

    last_gps_raw = None
    prev_spd = spd_0

    pre_bo_k = bo_start - start_sim - 1

    for k, idx in enumerate(range(start_sim, end_sim)):
        t_cur = float(t[idx])
        is_blackout = (idx >= bo_start) and (idx < bo_end)

        acc_3d = acc_v_all[idx].astype(np.float64)
        gyro_3d = gyro_v_all[idx].astype(np.float64)
        mag_3d = mag_v_all[idx].astype(np.float64)
        fwd_acc = float(acc_3d[0])
        yaw_rate = float(gyro_3d[2])

        # 1. Prediction step
        es_ekf.predict(acc_3d, gyro_3d, dt=dt)
        cur_v_body = es_ekf.rotation_matrix.T @ es_ekf.v
        fwd_speed = float(cur_v_body[0])
        acc_x_kin = (fwd_speed - prev_spd) / dt
        prev_spd = fwd_speed
        eff_yaw_rate = float(gyro_3d[2] - es_ekf.bg[2])
        
        if abs(acc_x_kin) < 2.0 and abs(eff_yaw_rate) < 0.2:
            es_ekf.update_gravity_leveling(
                acc_3d,
                forward_speed=fwd_speed,
                yaw_rate=eff_yaw_rate,
                fwd_acc=acc_x_kin,
                sigma_level=0.3,
            )

        # 2. Stationary / ZUPT update
        cur_spd_est = float(np.linalg.norm(es_ekf.v[:2]))
        if ai_ready[k]:
            cur_spd_est = max(cur_spd_est, ai_speeds[k])

        is_stat = stat_detector.update(acc_3d, gyro_3d, speed_mps=cur_spd_est)
        if is_stat:
            es_ekf.update_zupt(sigma_v=0.01)
            # Standstill heading stabilization: tightly observe gyro bias
            sigma_zaru = 0.0005 if pre_bo_stab else 0.001
            es_ekf.update_zaru(gyro_3d, sigma_bg=sigma_zaru)

        # 3. Magnetometer Processing & Gating
        roll, pitch, _ = es_ekf.euler_angles
        psi_mag = compute_tilt_compensated_heading(mag_3d, roll_rad=roll, pitch_rad=pitch)
        is_clean_mag, diag_dist = mag_detector.update(mag_3d, psi_mag, eff_yaw_rate, t_cur)
        if not is_clean_mag:
            mag_dist_count += 1

        if use_mag:
            # During motion, update magnetic heading if clean
            if is_clean_mag:
                acc_mag, _ = es_ekf.update_magnetic_heading(
                    psi_mag, sigma_yaw=0.15, is_clean=True, gate_threshold=es_ekf.config.chi2_gate_1d
                )
                if acc_mag:
                    mag_acc_count += 1
                else:
                    mag_rej_count += 1
            else:
                mag_rej_count += 1

        # 4. GNSS or Blackout updates
        cur_gps_raw = (float(gps[idx, 0]), float(gps[idx, 1]))
        is_new_gps = (last_gps_raw is None) or (
            abs(cur_gps_raw[0] - last_gps_raw[0]) > 1e-9 or abs(cur_gps_raw[1] - last_gps_raw[1]) > 1e-9
        )

        if not is_blackout:
            if is_new_gps:
                last_gps_raw = cur_gps_raw
                gnss_pos = gt_enu[k]

                # Warmup GNSS updates (unlocked trusted gate)
                es_ekf.update_gnss_pos(gnss_pos, R_cov=np.eye(3) * 9.0, gate_threshold=None, is_trusted=True)

                gnss_spd = float(gps[idx, 2]) if gps.shape[1] > 2 else float(np.linalg.norm(gt_vel[k, :2]))
                gnss_hdg = float(gps[idx, 3]) if gps.shape[1] > 3 else 0.0

                # Pre-blackout heading stabilization logic:
                # Only use GNSS COG as heading measurement when speed >= 1.5 m/s (or 2.0 m/s with pre_bo_stab)
                min_cog_speed = 2.0 if pre_bo_stab else 1.5
                if gnss_spd >= min_cog_speed:
                    psi_meas = np.deg2rad(90.0 - gnss_hdg)
                    # Adaptive yaw measurement variance based on speed
                    sigma_cog = max(0.03, 0.15 / max(gnss_spd, 1.0)) if pre_bo_stab else 0.05
                    es_ekf.update_heading(psi_meas, sigma_yaw=sigma_cog, gate_threshold=None)
                    v_e = gnss_spd * np.cos(psi_meas)
                    v_n = gnss_spd * np.sin(psi_meas)
                    es_ekf.update_gnss_vel(np.array([v_e, v_n, 0.0]), R_cov=np.eye(3) * 0.25, gate_threshold=None, is_trusted=True)
        else:
            # During Blackout
            # NHC update with turn policy
            acc_nhc, diag_nhc = es_ekf.update_nhc(
                sigma_lat=0.05,
                sigma_vert=0.05,
                cornering_threshold_mps2=0.5,
                turn_policy=turn_nhc_policy,
            )
            if acc_nhc:
                nhc_active_count += 1
            if diag_nhc.get("is_dynamic_cornering", False) and turn_nhc_policy == "disable_lateral":
                nhc_turn_disabled_count += 1

            # AI speed update
            if ai_ready[k]:
                val_speed = float(ai_speeds[k])
                acc_ai, _ = es_ekf.update_ai_velocity(val_speed, sigma_v=0.8, max_innovation_sigma=4.0)
                if acc_ai:
                    ai_acc_count += 1
                else:
                    ai_rej_count += 1

        # Record trajectory & covariance
        est_enu[k] = es_ekf.p.copy()
        est_vel[k] = es_ekf.v.copy()
        _, _, cur_psi = es_ekf.euler_angles
        est_yaw[k] = cur_psi
        est_yaw_std[k] = np.rad2deg(np.sqrt(es_ekf.P[8, 8]))

    # Pre-blackout entry error at k_bo_start - 1
    pre_bo_pos_err = float(np.linalg.norm(est_enu[pre_bo_k, :2] - gt_enu[pre_bo_k, :2]))
    pre_bo_vel_err = float(np.linalg.norm(est_vel[pre_bo_k, :2] - gt_vel[pre_bo_k, :2]))
    pre_bo_yaw_err = float(
        np.rad2deg(
            min(
                abs(est_yaw[pre_bo_k] - gt_yaw[pre_bo_k]),
                2 * np.pi - abs(est_yaw[pre_bo_k] - gt_yaw[pre_bo_k]),
            )
        )
    )

    # Blackout metrics
    k_bo_start = bo_start - start_sim
    k_bo_end = bo_end - start_sim

    bo_est_enu = est_enu[k_bo_start:k_bo_end]
    bo_gt_enu = gt_enu[k_bo_start:k_bo_end]
    bo_est_vel = est_vel[k_bo_start:k_bo_end]
    bo_gt_vel = gt_vel[k_bo_start:k_bo_end]
    bo_est_yaw = est_yaw[k_bo_start:k_bo_end]
    bo_gt_yaw = gt_yaw[k_bo_start:k_bo_end]

    err_2d = np.linalg.norm(bo_est_enu[:, :2] - bo_gt_enu[:, :2], axis=1)
    final_err_2d = float(err_2d[-1])
    max_err_2d = float(np.max(err_2d))
    pos_rmse_2d = float(np.sqrt(np.mean(err_2d ** 2)))

    spd_est = np.linalg.norm(bo_est_vel[:, :2], axis=1)
    spd_gt = np.linalg.norm(bo_gt_vel[:, :2], axis=1)
    speed_rmse = float(np.sqrt(np.mean((spd_est - spd_gt) ** 2)))

    yaw_diffs = np.array(
        [
            min(abs(bo_est_yaw[i] - bo_gt_yaw[i]), 2 * np.pi - abs(bo_est_yaw[i] - bo_gt_yaw[i]))
            for i in range(len(bo_est_yaw))
        ]
    )
    yaw_err_deg = np.rad2deg(yaw_diffs)
    final_yaw_err = float(yaw_err_deg[-1])
    yaw_rmse = float(np.sqrt(np.mean(yaw_err_deg ** 2)))

    # Heading error at 10s, 30s, 60s checkpoints
    idx_10s = min(len(yaw_err_deg) - 1, 100)
    idx_30s = min(len(yaw_err_deg) - 1, 300)
    idx_60s = min(len(yaw_err_deg) - 1, 600)

    hdg_err_10s = float(yaw_err_deg[idx_10s])
    hdg_err_30s = float(yaw_err_deg[idx_30s])
    hdg_err_60s = float(yaw_err_deg[idx_60s])

    dist = float(window.distance_traveled_m)
    drift_pct = float((final_err_2d / max(dist, 1.0)) * 100.0)
    sih_pass = (drift_pct < 10.0)

    return Phase31AResultRecord(
        window_id=window.window_id,
        drive_id=window.drive_id,
        split=window.split,
        regime=window.regime,
        duration_s=window.duration_s,
        config_name=config_name,
        distance_m=dist,
        pre_bo_pos_error_m=pre_bo_pos_err,
        pre_bo_vel_error_mps=pre_bo_vel_err,
        pre_bo_yaw_error_deg=pre_bo_yaw_err,
        final_pos_error_2d_m=final_err_2d,
        max_pos_error_2d_m=max_err_2d,
        pos_rmse_2d_m=pos_rmse_2d,
        drift_percent=drift_pct,
        speed_rmse_mps=speed_rmse,
        heading_error_10s_deg=hdg_err_10s,
        heading_error_30s_deg=hdg_err_30s,
        heading_error_60s_deg=hdg_err_60s,
        final_heading_error_deg=final_yaw_err,
        heading_rmse_deg=yaw_rmse,
        final_yaw_std_deg=float(est_yaw_std[-1]),
        ai_accepted=ai_acc_count,
        ai_rejected=ai_rej_count,
        mag_accepted=mag_acc_count,
        mag_rejected=mag_rej_count,
        mag_disturbance_count=mag_dist_count,
        nhc_active_count=nhc_active_count,
        nhc_turn_disabled_count=nhc_turn_disabled_count,
        sih_pass_10pct=sih_pass,
    )


def execute_phase31a_ablation() -> Dict[str, Any]:
    """Run the complete Phase 31A 6-configuration ablation across all 30 authentic blackout windows."""
    workspace_root = Path(__file__).resolve().parents[3]
    dataset_dir = workspace_root / "data" / "authentic" / "dataset"
    model_path = workspace_root / "models" / "authentic" / "velocity_net.pt"

    # Load frozen AI model
    vel_model = VelocityEstimatorNet(in_channels=6, hidden_dim=64, num_layers=2)
    if model_path.exists():
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        vel_model.load_state_dict(state_dict)
        logger.info(f"Loaded frozen VelocityEstimatorNet from {model_path}")
    else:
        logger.warning(f"VelocityEstimatorNet not found at {model_path}, proceeding model-free")
        vel_model = None

    # Load 30 authentic blackout windows
    val_drive_dir = workspace_root / "data" / "raw" / "categorised_authentic" / "M (Driver B)"
    test_drive_dir = workspace_root / "data" / "raw" / "categorised_authentic" / "Y (Driver D)" / "Y1"

    val_windows = select_benchmark_windows(val_drive_dir, "M", "validation")
    test_windows = select_benchmark_windows(test_drive_dir, "Y1", "test")
    all_windows = val_windows + test_windows
    logger.info(f"Loaded {len(all_windows)} standardized benchmark windows (15 Val M, 15 Test Y1)")

    configs = [
        "phase30a_baseline",
        "mag_only",
        "heading_stabilization_only",
        "mag_plus_heading_stabilization",
        "turn_aware_nhc_only",
        "phase31a_all_mechanisms",
    ]

    all_records: List[Phase31AResultRecord] = []
    t0 = time.time()

    for cfg in configs:
        logger.info(f"--- Running Ablation Configuration: {cfg} ---")
        for win in all_windows:
            rec = run_single_phase31a_simulation(win, cfg, vel_model)
            all_records.append(rec)

    elapsed = time.time() - t0
    logger.info(f"Completed {len(all_records)} simulations in {elapsed:.2f}s")

    # Compute comprehensive summary statistics
    summary = generate_ablation_summary(all_records, configs)

    # Save JSON report
    out_dir = workspace_root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "PHASE31A_HEADING_STABILIZATION.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved Phase 31A JSON report to {json_path}")

    return summary


def generate_ablation_summary(records: List[Phase31AResultRecord], configs: List[str]) -> Dict[str, Any]:
    """Aggregate ablation metrics by configuration, split, regime, and duration."""
    summary: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_simulations": len(records),
        "configurations": {},
    }

    for cfg in configs:
        cfg_recs = [r for r in records if r.config_name == cfg]
        if not cfg_recs:
            continue

        drifts = [r.drift_percent for r in cfg_recs]
        pos_errors = [r.final_pos_error_2d_m for r in cfg_recs]
        pre_bo_yaw_errs = [r.pre_bo_yaw_error_deg for r in cfg_recs]
        hdg_10s = [r.heading_error_10s_deg for r in cfg_recs]
        hdg_30s = [r.heading_error_30s_deg for r in cfg_recs]
        hdg_60s = [r.heading_error_60s_deg for r in cfg_recs]
        final_hdg = [r.final_heading_error_deg for r in cfg_recs]
        yaw_stds = [r.final_yaw_std_deg for r in cfg_recs]

        # Splits
        m_recs = [r for r in cfg_recs if r.drive_id == "M"]
        y1_recs = [r for r in cfg_recs if r.drive_id == "Y1"]

        # Regimes
        straight_recs = [r for r in cfg_recs if r.regime == "straight"]
        accel_recs = [r for r in cfg_recs if r.regime == "accel_decel"]
        low_speed_recs = [r for r in cfg_recs if r.regime == "low_speed"]
        turning_recs = [r for r in cfg_recs if r.regime == "turning"]

        # Durations
        d10_recs = [r for r in cfg_recs if abs(r.duration_s - 10.0) < 1.0]
        d30_recs = [r for r in cfg_recs if abs(r.duration_s - 30.0) < 1.0]
        d60_recs = [r for r in cfg_recs if abs(r.duration_s - 60.0) < 1.0]

        def get_sub_metrics(sub_list: List[Phase31AResultRecord]) -> Dict[str, float]:
            if not sub_list:
                return {}
            sub_drifts = [r.drift_percent for r in sub_list]
            sub_hdg = [r.final_heading_error_deg for r in sub_list]
            sub_pos = [r.final_pos_error_2d_m for r in sub_list]
            return {
                "count": len(sub_list),
                "mean_drift_pct": float(np.mean(sub_drifts)),
                "median_drift_pct": float(np.median(sub_drifts)),
                "p90_drift_pct": float(np.percentile(sub_drifts, 90)),
                "worst_drift_pct": float(np.max(sub_drifts)),
                "sih_pass_rate_pct": float(np.mean([r.sih_pass_10pct for r in sub_list]) * 100.0),
                "mean_final_pos_error_m": float(np.mean(sub_pos)),
                "mean_final_heading_error_deg": float(np.mean(sub_hdg)),
            }

        total_ai_acc = sum(r.ai_accepted for r in cfg_recs)
        total_ai_rej = sum(r.ai_rejected for r in cfg_recs)
        ai_acc_rate = float(total_ai_acc / max(1, total_ai_acc + total_ai_rej) * 100.0)

        total_mag_acc = sum(r.mag_accepted for r in cfg_recs)
        total_mag_rej = sum(r.mag_rejected for r in cfg_recs)
        mag_acc_rate = float(total_mag_acc / max(1, total_mag_acc + total_mag_rej) * 100.0)

        summary["configurations"][cfg] = {
            "overall": {
                "mean_drift_pct": float(np.mean(drifts)),
                "median_drift_pct": float(np.median(drifts)),
                "p90_drift_pct": float(np.percentile(drifts, 90)),
                "p95_drift_pct": float(np.percentile(drifts, 95)),
                "worst_drift_pct": float(np.max(drifts)),
                "sih_pass_rate_pct": float(np.mean([r.sih_pass_10pct for r in cfg_recs]) * 100.0),
                "mean_pre_bo_pos_error_m": float(np.mean([r.pre_bo_pos_error_m for r in cfg_recs])),
                "mean_pre_bo_yaw_error_deg": float(np.mean(pre_bo_yaw_errs)),
                "mean_heading_error_10s_deg": float(np.mean(hdg_10s)),
                "mean_heading_error_30s_deg": float(np.mean(hdg_30s)),
                "mean_heading_error_60s_deg": float(np.mean(hdg_60s)),
                "mean_final_heading_error_deg": float(np.mean(final_hdg)),
                "mean_final_pos_error_m": float(np.mean(pos_errors)),
                "mean_final_yaw_std_deg": float(np.mean(yaw_stds)),
                "ai_acceptance_rate_pct": ai_acc_rate,
                "mag_acceptance_rate_pct": mag_acc_rate,
                "mag_accepted_total": total_mag_acc,
                "mag_rejected_total": total_mag_rej,
                "mag_disturbance_count_total": sum(r.mag_disturbance_count for r in cfg_recs),
                "nhc_active_total": sum(r.nhc_active_count for r in cfg_recs),
                "nhc_turn_disabled_total": sum(r.nhc_turn_disabled_count for r in cfg_recs),
            },
            "by_split": {
                "Drive_M_validation": get_sub_metrics(m_recs),
                "Drive_Y1_test": get_sub_metrics(y1_recs),
            },
            "by_regime": {
                "straight": get_sub_metrics(straight_recs),
                "accel_decel": get_sub_metrics(accel_recs),
                "low_speed": get_sub_metrics(low_speed_recs),
                "turning": get_sub_metrics(turning_recs),
            },
            "by_duration": {
                "10s": get_sub_metrics(d10_recs),
                "30s": get_sub_metrics(d30_recs),
                "60s": get_sub_metrics(d60_recs),
            },
        }

    return summary


if __name__ == "__main__":
    execute_phase31a_ablation()
