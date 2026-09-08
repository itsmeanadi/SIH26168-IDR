"""Phase 30A: Causal Ablation Benchmark Runner.

Evaluates the exact causal impact of:
1. Phase 28 baseline (uncorrected ES-EKF with strict warmup GNSS gate and coupled NHC)
2. 15state + GNSS warmup gating correction only
3. 15state + NHC cornering decoupling only
4. 15state + BOTH corrections (Phase 30A candidate)
5. Legacy 9-state planar EKF (cross-architecture reference)

Evaluates on all 30 standardized authentic blackout windows (15 Drive M, 15 Drive Y1).
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
from ..filters.es_ekf import ErrorStateKalmanFilter, ESEKFConfig
from ..filters.ekf import ExtendedKalmanFilter
from ..filters.fusion import GNSSINSFusion
from ..filters.zupt import StationaryDetector
from ..models.train_all import VelocityEstimatorNet
from ..io.preprocess import load_drive_pair
from .run_phase28_full_benchmark import (
    BlackoutWindowSpec,
    get_cached_drive_data,
    select_benchmark_windows,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class AblationResultRecord:
    window_id: str
    drive_id: str
    split: str
    regime: str
    duration_s: float
    baseline: str
    distance_m: float
    # Pre-blackout entry error
    pre_bo_pos_error_m: float
    pre_bo_vel_error_mps: float
    pre_bo_yaw_error_deg: float
    # Outage error metrics
    final_pos_error_2d_m: float
    max_pos_error_2d_m: float
    pos_rmse_2d_m: float
    drift_percent: float
    speed_rmse_mps: float
    final_heading_error_deg: float
    heading_rmse_deg: float
    # Diagnostic counts
    ai_accepted: int
    ai_rejected: int
    nhc_cornering_decoupled_count: int
    sih_pass_10pct: bool


def run_single_ablation(
    window: BlackoutWindowSpec,
    ablation_name: str,
    vel_model: Optional[VelocityEstimatorNet],
    export_trace: bool = False,
) -> Tuple[AblationResultRecord, Optional[List[Dict[str, Any]]]]:
    """Execute one ablation run under strict causal provenance."""
    imu, gps, v_speed, t = get_cached_drive_data(window.drive_path, window.drive_id)
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

    # Calibrate phone IMU into vehicle frame
    aligner = PhoneToVehicleAligner()
    aligner.estimate_from_stationary_and_motion(imu[0:300, 0:3], imu[300:600, 0:3])
    acc_v_all, gyro_v_all = aligner.transform_imu(imu[:, 0:3], imu[:, 3:6])

    # Determine Ablation Configuration Flags
    # Baselines:
    # 1. 'phase28_baseline'      : Strict GNSS gate (11.345), Standard NHC (coupled)
    # 2. 'gnss_warmup_fix_only'  : Unlocked GNSS gate, Standard NHC (coupled)
    # 3. 'nhc_decouple_only'     : Strict GNSS gate, Decoupled NHC on cornering
    # 4. 'phase30a_both_fixes'   : Unlocked GNSS gate, Decoupled NHC on cornering
    # 5. 'legacy_ekf_full'       : 9-state planar EKF reference
    
    use_legacy = (ablation_name == "legacy_ekf_full")
    unlock_gnss = ablation_name in ("gnss_warmup_fix_only", "phase30a_both_fixes")
    decouple_nhc = ablation_name in ("nhc_decouple_only", "phase30a_both_fixes")

    es_ekf: Optional[ErrorStateKalmanFilter] = None
    legacy_ekf: Optional[ExtendedKalmanFilter] = None

    spd_0 = float(v_speed[start_sim])
    psi_0 = gt_yaw[0]
    v_0 = np.array([spd_0 * np.cos(psi_0), spd_0 * np.sin(psi_0), 0.0])

    if not use_legacy:
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
    else:
        legacy_ekf = ExtendedKalmanFilter(dt=dt)
        legacy_ekf.x[0:3] = np.zeros(3)
        legacy_ekf.x[3:6] = v_0
        legacy_ekf.x[6] = psi_0

    stat_detector = StationaryDetector(window_size=10, acc_var_threshold=0.15)

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

    est_enu = np.zeros((N_sim, 3))
    est_vel = np.zeros((N_sim, 3))
    est_yaw = np.zeros(N_sim)

    ai_acc_count = 0
    ai_rej_count = 0
    nhc_decoupled_count = 0

    trace_records = []
    last_gps_raw = None
    prev_spd = spd_0

    pre_bo_k = bo_start - start_sim - 1

    for k, idx in enumerate(range(start_sim, end_sim)):
        t_cur = float(t[idx])
        is_blackout = (idx >= bo_start) and (idx < bo_end)

        acc_3d = acc_v_all[idx].astype(np.float64)
        gyro_3d = gyro_v_all[idx].astype(np.float64)
        fwd_acc = float(acc_3d[0])
        yaw_rate = float(gyro_3d[2])

        # 1. Prediction step
        if es_ekf is not None:
            es_ekf.predict(acc_3d, gyro_3d, dt=dt)
            cur_v_body = es_ekf.rotation_matrix.T @ es_ekf.v
            fwd_speed = float(cur_v_body[0])
            acc_x_kin = (fwd_speed - prev_spd) / dt
            prev_spd = fwd_speed
            eff_yaw_rate = float(gyro_3d[2] - es_ekf.bg[2])
            if abs(acc_x_kin) < 2.0 and abs(eff_yaw_rate) < 0.2:
                es_ekf.update_gravity_leveling(acc_3d, forward_speed=fwd_speed, yaw_rate=eff_yaw_rate, fwd_acc=acc_x_kin, sigma_level=0.3)
        elif legacy_ekf is not None:
            legacy_ekf.predict(fwd_acc, yaw_rate)

        # 2. Stationary / ZUPT update
        cur_spd_est = float(np.linalg.norm(es_ekf.v[:2])) if es_ekf is not None else float(np.linalg.norm(legacy_ekf.x[3:5]))
        if ai_ready[k]:
            cur_spd_est = max(cur_spd_est, ai_speeds[k])

        is_stat = stat_detector.update(acc_3d, gyro_3d, speed_mps=cur_spd_est)
        if is_stat:
            if es_ekf is not None:
                es_ekf.update_zupt(sigma_v=0.01)
                es_ekf.update_zaru(gyro_3d, sigma_bg=0.001)
            elif legacy_ekf is not None:
                legacy_ekf.update_ai_velocity(0.0, sigma_v=0.01)

        # 3. GNSS or Blackout updates
        cur_gps_raw = (float(gps[idx, 0]), float(gps[idx, 1]))
        is_new_gps = (last_gps_raw is None) or (abs(cur_gps_raw[0] - last_gps_raw[0]) > 1e-9 or abs(cur_gps_raw[1] - last_gps_raw[1]) > 1e-9)

        if not is_blackout:
            if is_new_gps:
                last_gps_raw = cur_gps_raw
                gnss_pos = gt_enu[k]

                if es_ekf is not None:
                    # Gating policy
                    gate_p = None if unlock_gnss else es_ekf.config.chi2_gate_3d
                    es_ekf.update_gnss_pos(gnss_pos, R_cov=np.eye(3) * 9.0, gate_threshold=gate_p, is_trusted=unlock_gnss)

                    gnss_spd = float(gps[idx, 2]) if gps.shape[1] > 2 else float(np.linalg.norm(gt_vel[k, :2]))
                    gnss_hdg = float(gps[idx, 3]) if gps.shape[1] > 3 else 0.0
                    if gnss_spd >= 1.5:
                        psi_meas = np.deg2rad(90.0 - gnss_hdg)
                        gate_h = None if unlock_gnss else es_ekf.config.chi2_gate_1d
                        es_ekf.update_heading(psi_meas, sigma_yaw=0.05, gate_threshold=gate_h)
                        v_e = gnss_spd * np.cos(psi_meas)
                        v_n = gnss_spd * np.sin(psi_meas)
                        gate_v = None if unlock_gnss else es_ekf.config.chi2_gate_3d
                        es_ekf.update_gnss_vel(np.array([v_e, v_n, 0.0]), R_cov=np.eye(3) * 0.25, gate_threshold=gate_v, is_trusted=unlock_gnss)
                elif legacy_ekf is not None:
                    legacy_ekf.update_gnss_pos(gnss_pos, R_cov=np.eye(3) * 9.0)
                    gnss_spd = float(gps[idx, 2]) if gps.shape[1] > 2 else float(np.linalg.norm(gt_vel[k, :2]))
                    gnss_hdg = float(gps[idx, 3]) if gps.shape[1] > 3 else 0.0
                    if gnss_spd >= 1.5:
                        psi_meas = np.deg2rad(90.0 - gnss_hdg)
                        legacy_ekf.update_heading(psi_meas, R_yaw=0.03)
                        legacy_ekf.update_velocity_2d(gnss_spd, 0.0)
        else:
            # During Blackout
            if es_ekf is not None:
                # NHC
                c_thresh = 0.5 if decouple_nhc else 1e9
                _, diag_nhc = es_ekf.update_nhc(sigma_lat=0.05, sigma_vert=0.05, cornering_threshold_mps2=c_thresh)
                if diag_nhc.get("is_dynamic_cornering", False):
                    nhc_decoupled_count += 1

                # AI speed update
                if ai_ready[k]:
                    val_speed = float(ai_speeds[k])
                    acc_ai, _ = es_ekf.update_ai_velocity(val_speed, sigma_v=0.8, max_innovation_sigma=4.0)
                    if acc_ai:
                        ai_acc_count += 1
                    else:
                        ai_rej_count += 1
            elif legacy_ekf is not None:
                if ai_ready[k]:
                    val_speed = float(ai_speeds[k])
                    acc_ai, _ = legacy_ekf.update_ai_velocity(val_speed, sigma_v=0.8, max_innovation_sigma=4.0)
                    if acc_ai:
                        ai_acc_count += 1
                    else:
                        ai_rej_count += 1
                legacy_ekf.update_velocity_2d(cur_spd_est, 0.0, R_cov=np.diag([0.8**2, 0.05**2]))

        # Record trajectory
        if es_ekf is not None:
            est_enu[k] = es_ekf.p.copy()
            est_vel[k] = es_ekf.v.copy()
            _, _, cur_psi = es_ekf.euler_angles
            est_yaw[k] = cur_psi
        elif legacy_ekf is not None:
            est_enu[k] = legacy_ekf.x[0:3].copy()
            est_vel[k] = legacy_ekf.x[3:6].copy()
            est_yaw[k] = float(legacy_ekf.x[6])

    # Pre-blackout entry error at k_bo_start - 1
    pre_bo_pos_err = float(np.linalg.norm(est_enu[pre_bo_k, :2] - gt_enu[pre_bo_k, :2]))
    pre_bo_vel_err = float(np.linalg.norm(est_vel[pre_bo_k, :2] - gt_vel[pre_bo_k, :2]))
    pre_bo_yaw_err = float(np.rad2deg(min(abs(est_yaw[pre_bo_k] - gt_yaw[pre_bo_k]), 2 * np.pi - abs(est_yaw[pre_bo_k] - gt_yaw[pre_bo_k]))))

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

    yaw_diffs = np.array([min(abs(bo_est_yaw[i] - bo_gt_yaw[i]), 2 * np.pi - abs(bo_est_yaw[i] - bo_gt_yaw[i])) for i in range(len(bo_est_yaw))])
    yaw_err_deg = np.rad2deg(yaw_diffs)
    final_yaw_err = float(yaw_err_deg[-1])
    yaw_rmse = float(np.sqrt(np.mean(yaw_err_deg ** 2)))

    dist = float(window.distance_traveled_m)
    drift_pct = float((final_err_2d / max(dist, 1.0)) * 100.0)
    sih_pass = (drift_pct < 10.0)

    record = AblationResultRecord(
        window_id=window.window_id,
        drive_id=window.drive_id,
        split=window.split,
        regime=window.regime,
        duration_s=window.duration_s,
        baseline=ablation_name,
        distance_m=dist,
        pre_bo_pos_error_m=pre_bo_pos_err,
        pre_bo_vel_error_mps=pre_bo_vel_err,
        pre_bo_yaw_error_deg=pre_bo_yaw_err,
        final_pos_error_2d_m=final_err_2d,
        max_pos_error_2d_m=max_err_2d,
        pos_rmse_2d_m=pos_rmse_2d,
        drift_percent=drift_pct,
        speed_rmse_mps=speed_rmse,
        final_heading_error_deg=final_yaw_err,
        heading_rmse_deg=yaw_rmse,
        ai_accepted=ai_acc_count,
        ai_rejected=ai_rej_count,
        nhc_cornering_decoupled_count=nhc_decoupled_count,
        sih_pass_10pct=sih_pass,
    )

    return record, None


def execute_phase30a_ablation():
    """Run complete 30-window ablation across all 5 configurations."""
    print("=" * 80)
    print("PHASE 30A: CAUSAL ABLATION BENCHMARK RUNNER")
    print("=" * 80)

    # 1. Load authentic model
    model_path = Path("models/authentic/velocity_net.pt")
    vel_model = None
    if model_path.exists():
        vel_model = VelocityEstimatorNet()
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        vel_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        vel_model.eval()
        logger.info(f"Loaded authentic VelocityEstimatorNet from: {model_path}")

    # 2. Select windows (same 15 on M, 15 on Y1)
    val_dir = Path("data/raw/categorised_authentic/M (Driver B)")
    test_dir = Path("data/raw/categorised_authentic/Y (Driver D)/Y1")

    val_windows = select_benchmark_windows(val_dir, "M", "validation")
    test_windows = select_benchmark_windows(test_dir, "Y1", "held_out_test")
    all_windows = val_windows + test_windows

    ablations = [
        "phase28_baseline",
        "gnss_warmup_fix_only",
        "nhc_decouple_only",
        "phase30a_both_fixes",
        "legacy_ekf_full",
    ]

    results: List[AblationResultRecord] = []
    
    print(f"Executing {len(all_windows)} windows x {len(ablations)} configurations = {len(all_windows) * len(ablations)} total runs...")

    for win in all_windows:
        for abl in ablations:
            rec, _ = run_single_ablation(win, abl, vel_model)
            results.append(rec)
            status_str = "PASS" if rec.sih_pass_10pct else "FAIL"
            logger.info(f"[{rec.split[:3].upper()}] {rec.drive_id} {int(rec.duration_s):2d}s | {rec.regime:<11} | {rec.baseline:<22} | EntryErr: {rec.pre_bo_pos_error_m:6.2f}m | FinalErr: {rec.final_pos_error_2d_m:6.2f}m | Drift: {rec.drift_percent:7.2f}% | {status_str}")

    # Aggregate summaries by baseline
    summary_by_baseline = {}
    for abl in ablations:
        b_recs = [r for r in results if r.baseline == abl]
        drifts = [r.drift_percent for r in b_recs]
        final_errs = [r.final_pos_error_2d_m for r in b_recs]
        entry_errs = [r.pre_bo_pos_error_m for r in b_recs]
        pass_cnt = sum(1 for r in b_recs if r.sih_pass_10pct)
        
        summary_by_baseline[abl] = {
            "count": len(b_recs),
            "sih_pass_count": pass_cnt,
            "sih_pass_percent": round(pass_cnt / len(b_recs) * 100.0, 2),
            "entry_err_mean_m": round(float(np.mean(entry_errs)), 2),
            "entry_err_median_m": round(float(np.median(entry_errs)), 2),
            "drift_mean_pct": round(float(np.mean(drifts)), 2),
            "drift_median_pct": round(float(np.median(drifts)), 2),
            "drift_p90_pct": round(float(np.percentile(drifts, 90)), 2),
            "drift_p95_pct": round(float(np.percentile(drifts, 95)), 2),
            "drift_worst_pct": round(float(np.max(drifts)), 2),
            "final_err_mean_m": round(float(np.mean(final_errs)), 2),
            "final_err_median_m": round(float(np.median(final_errs)), 2),
            "speed_rmse_mean_mps": round(float(np.mean([r.speed_rmse_mps for r in b_recs])), 3),
            "heading_rmse_mean_deg": round(float(np.mean([r.heading_rmse_deg for r in b_recs])), 2),
            "total_ai_accepted": sum(r.ai_accepted for r in b_recs),
            "total_ai_rejected": sum(r.ai_rejected for r in b_recs),
            "total_nhc_decoupled": sum(r.nhc_cornering_decoupled_count for r in b_recs),
        }

    # Summary by duration for Phase 30A both fixes
    summary_by_duration = {}
    for dur in [10.0, 30.0, 60.0]:
        for abl in ["phase28_baseline", "phase30a_both_fixes", "legacy_ekf_full"]:
            sub = [r for r in results if r.baseline == abl and r.duration_s == dur]
            drifts = [r.drift_percent for r in sub]
            final_errs = [r.final_pos_error_2d_m for r in sub]
            entry_errs = [r.pre_bo_pos_error_m for r in sub]
            pass_cnt = sum(1 for r in sub if r.sih_pass_10pct)
            summary_by_duration[f"{int(dur)}s_{abl}"] = {
                "duration_s": dur,
                "baseline": abl,
                "n_cases": len(sub),
                "pass_count": pass_cnt,
                "pass_pct": round(pass_cnt / len(sub) * 100.0, 2),
                "entry_err_median_m": round(float(np.median(entry_errs)), 2),
                "drift_median_pct": round(float(np.median(drifts)), 2),
                "drift_p90_pct": round(float(np.percentile(drifts, 90)), 2),
                "final_err_median_m": round(float(np.median(final_errs)), 2),
            }

    # Save JSON report
    report_dict = {
        "benchmark_phase": "30A",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_runs": len(results),
        "configurations_evaluated": ablations,
        "summary_by_baseline": summary_by_baseline,
        "summary_by_duration": summary_by_duration,
        "detailed_results": [asdict(r) for r in results],
    }

    out_json = Path("reports/PHASE30A_CAUSAL_ABLATION.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    logger.info(f"Saved machine-readable ablation report: {out_json}")

    print("\n" + "=" * 80)
    print("PHASE 30A ABLATION SUMMARY")
    print("=" * 80)
    print(f"{'Baseline':<24} | {'EntryErr (m)':<12} | {'Drift Med %':<12} | {'Drift P90 %':<12} | {'Pass Rate':<10}")
    print("-" * 80)
    for abl in ablations:
        s = summary_by_baseline[abl]
        print(f"{abl:<24} | {s['entry_err_median_m']:12.2f} | {s['drift_median_pct']:12.2f} | {s['drift_p90_pct']:12.2f} | {s['sih_pass_percent']:9.1f}%")


if __name__ == "__main__":
    execute_phase30a_ablation()
