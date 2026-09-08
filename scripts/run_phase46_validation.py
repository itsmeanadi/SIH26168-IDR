"""Phase 46 Validation Script: Authentic Model -> End-to-End Navigation Impact Validation.

Evaluates:
1. Direct AI Metrics Comparison on Untouched Test Drive Y1 (Driver D)
   - Speed MAE, RMSE, Mean Bias, Stationary Error, Regime Errors (Low/Med/High).
2. Controlled Authentic Blackout Benchmark (10s, 30s, 60s) on both Drive M (Val) and Drive Y1 (Test)
   across 4 configurations:
   - Config 1: Pure IMU + existing constraints (ES-EKF + ZUPT/ZARU + NHC + Leveling, No AI)
   - Config 2: Historical Synthetic AI (Checkpoint A)
   - Config 3: Authentic Huber AI (Checkpoint B)
   - Config 4: Authentic Huber + Stationary-Gate AI (Checkpoint C)
3. Offline Qualitative Evaluation on Real Physical Walking Session (exp_20260908_172204_two_wheeler)
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from idr.eval.run_authentic_blackout_benchmark import (
    select_benchmark_windows,
    run_single_window_benchmark,
    get_cached_drive_data,
    BlackoutWindowSpec,
    BlackoutRunResult,
)
from idr.io.loader import load_drive_pair
from idr.models.velocity_net import VelocityEstimatorNet
from idr.eval.experiment_runner import ExperimentHarness

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase46Validation")

CHECKPOINTS = {
    "historical_synthetic": {
        "name": "Historical Synthetic (Mock)",
        "path": Path("models/checkpoints/historical_synthetic_velocity_net.pt"),
        "role": "historical_synthetic",
    },
    "authentic_huber": {
        "name": "Authentic Huber (No Gate Loss)",
        "path": Path("models/checkpoints/authentic_huber_velocity_net.pt"),
        "role": "authentic_huber",
    },
    "authentic_gate": {
        "name": "Authentic Huber + Stationary Gate",
        "path": Path("models/checkpoints/authentic_gate_velocity_net.pt"),
        "role": "authentic_gate",
    },
}


def compute_sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def run_direct_ai_comparison(
    drive_folder: Path, drive_id: str, split_name: str
) -> Dict[str, Any]:
    """Evaluate speed estimation metrics for all models on a synchronized drive."""
    drive = load_drive_pair(drive_folder, drive_id)
    phone_imu, phone_gps, v_speed, t = drive.get_synced_data()

    window_size = 50
    stride = 10
    min_len = len(phone_imu)

    windows_list = []
    target_list = []
    for i in range(0, min_len - window_size + 1, stride):
        windows_list.append(phone_imu[i : i + window_size].T)
        target_list.append(v_speed[i + window_size - 1])

    windows = np.ascontiguousarray(np.stack(windows_list), dtype=np.float32)
    targets = np.array(target_list, dtype=np.float32)

    results = {}
    for key, info in CHECKPOINTS.items():
        ckpt_path = info["path"]
        model = VelocityEstimatorNet()
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
        model.eval()

        preds_list = []
        batch_size = 256
        with torch.no_grad():
            for b in range(0, len(windows), batch_size):
                b_win = torch.from_numpy(windows[b : b + batch_size])
                out = model(b_win)
                preds_list.append(out.squeeze(-1).numpy())
        preds = np.concatenate(preds_list)

        mae = float(np.mean(np.abs(preds - targets)))
        rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
        bias = float(np.mean(preds - targets))

        mask_stat = targets < 0.1
        mask_low = targets < 1.0
        mask_med = (targets >= 1.0) & (targets <= 10.0)
        mask_high = targets > 10.0

        stat_mae = float(np.mean(np.abs(preds[mask_stat] - targets[mask_stat]))) if np.any(mask_stat) else 0.0
        stat_mean_pred = float(np.mean(preds[mask_stat])) if np.any(mask_stat) else 0.0
        low_mae = float(np.mean(np.abs(preds[mask_low] - targets[mask_low]))) if np.any(mask_low) else 0.0
        med_mae = float(np.mean(np.abs(preds[mask_med] - targets[mask_med]))) if np.any(mask_med) else 0.0
        high_mae = float(np.mean(np.abs(preds[mask_high] - targets[mask_high]))) if np.any(mask_high) else 0.0

        results[key] = {
            "checkpoint_name": info["name"],
            "checkpoint_sha256": compute_sha256(ckpt_path),
            "split": split_name,
            "drive_id": drive_id,
            "sample_count": len(targets),
            "mae_mps": round(mae, 4),
            "mae_kmh": round(mae * 3.6, 2),
            "rmse_mps": round(rmse, 4),
            "rmse_kmh": round(rmse * 3.6, 2),
            "mean_bias_mps": round(bias, 4),
            "stationary_mae_mps": round(stat_mae, 4),
            "stationary_mean_pred_mps": round(stat_mean_pred, 4),
            "low_speed_mae_mps": round(low_mae, 4),
            "med_speed_mae_mps": round(med_mae, 4),
            "high_speed_mae_mps": round(high_mae, 4),
            "pred_min_mps": round(float(np.min(preds)), 4),
            "pred_max_mps": round(float(np.max(preds)), 4),
            "pred_mean_mps": round(float(np.mean(preds)), 4),
        }

    return results


def run_controlled_blackout_suite(
    drive_dir: Path, drive_id: str, split_name: str
) -> List[Dict[str, Any]]:
    """Run standardized blackout suite across all 4 configurations on a drive."""
    windows = select_benchmark_windows(drive_dir, drive_id, split_name)
    logger.info(f"Selected {len(windows)} benchmark windows for {drive_id} ({split_name})")

    suite_results = []

    # Configs:
    # 1. Pure IMU + constraints (no AI) -> baseline='15state_imu_only', vel_model=None
    # 2. Historical synthetic AI -> baseline='15state_full', vel_model=Checkpoint A
    # 3. Authentic Huber AI -> baseline='15state_full', vel_model=Checkpoint B
    # 4. Authentic stationary-gate AI -> baseline='15state_full', vel_model=Checkpoint C
    configs = [
        ("pure_imu_no_ai", "Pure IMU + ZUPT/ZARU + NHC + Leveling", None, "15state_imu_only"),
        ("historical_synthetic_ai", "Historical Synthetic AI + EKF", CHECKPOINTS["historical_synthetic"]["path"], "15state_full"),
        ("authentic_huber_ai", "Authentic Huber AI + EKF", CHECKPOINTS["authentic_huber"]["path"], "15state_full"),
        ("authentic_gate_ai", "Authentic Huber + Gate AI + EKF", CHECKPOINTS["authentic_gate"]["path"], "15state_full"),
    ]

    for cfg_key, cfg_label, ckpt_path, baseline_type in configs:
        ai_model = None
        if ckpt_path is not None and ckpt_path.exists():
            ai_model = VelocityEstimatorNet()
            ai_model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
            ai_model.eval()

        for win in windows:
            res, _ = run_single_window_benchmark(
                window=win,
                baseline=baseline_type,
                vel_model=ai_model,
                export_trace=False,
            )
            res_dict = asdict(res)
            res_dict["config_key"] = cfg_key
            res_dict["config_label"] = cfg_label
            res_dict["baseline_name"] = cfg_label
            res_dict["checkpoint_sha256"] = compute_sha256(ckpt_path) if ckpt_path else None
            suite_results.append(res_dict)

    return suite_results


def run_physical_walking_comparison() -> Dict[str, Any]:
    """Evaluate the 3 checkpoints on the real recorded GPS-denied physical walking session."""
    session_csv = Path("data/sessions/exp_20260908_172204_two_wheeler/telemetry.csv")
    if not session_csv.exists():
        return {}

    df = pd.read_csv(session_csv)
    # Extract vehicle-frame IMU
    ax = df["acc_veh_x"].to_numpy(dtype=np.float32)
    ay = df["acc_veh_y"].to_numpy(dtype=np.float32)
    az = df["acc_veh_z"].to_numpy(dtype=np.float32)
    gx = df["gyro_veh_x"].to_numpy(dtype=np.float32)
    gy = df["gyro_veh_y"].to_numpy(dtype=np.float32)
    gz = df["gyro_veh_z"].to_numpy(dtype=np.float32)
    is_stat = df["is_stationary"].to_numpy(dtype=np.int32)

    features = np.column_stack([ax, ay, az, gx, gy, gz])
    N = len(features)
    window_size = 50
    stride = 10

    # Build windows
    windows_list = []
    win_indices = []
    for i in range(0, N - window_size + 1, stride):
        windows_list.append(features[i : i + window_size].T)
        win_indices.append(i + window_size - 1)

    windows = np.ascontiguousarray(np.stack(windows_list), dtype=np.float32)

    # Motion intervals in exp_20260908_172204:
    # Stationary start: samples 0 to 440 (~0-8s)
    # Walking: samples 440 to 1100 (~8-20s)
    # Stationary stop: samples 1100 to end
    stat_start_idx = [k for k, idx in enumerate(win_indices) if idx < 440]
    walk_idx = [k for k, idx in enumerate(win_indices) if 440 <= idx < 1100]
    stop_idx = [k for k, idx in enumerate(win_indices) if idx >= 1100]

    phys_results = {}
    for key, info in CHECKPOINTS.items():
        ckpt_path = info["path"]
        model = VelocityEstimatorNet()
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
        model.eval()

        with torch.no_grad():
            out = model(torch.from_numpy(windows))
            preds = out.squeeze(-1).numpy()

        p_stat_start = preds[stat_start_idx] if stat_start_idx else np.array([0.0])
        p_walk = preds[walk_idx] if walk_idx else np.array([0.0])
        p_stop = preds[stop_idx] if stop_idx else np.array([0.0])

        phys_results[key] = {
            "checkpoint_name": info["name"],
            "checkpoint_sha256": compute_sha256(ckpt_path),
            "stationary_initial_mean_mps": round(float(np.mean(p_stat_start)), 3),
            "stationary_initial_peak_mps": round(float(np.max(p_stat_start)), 3),
            "walking_mean_mps": round(float(np.mean(p_walk)), 3),
            "walking_peak_mps": round(float(np.max(p_walk)), 3),
            "stopping_mean_mps": round(float(np.mean(p_stop)), 3),
            "stopping_final_mps": round(float(p_stop[-1]) if len(p_stop) > 0 else 0.0, 3),
        }

    return phys_results


def aggregate_blackout_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate drift, position, and velocity metrics by config and duration."""
    df = pd.DataFrame(results)
    summary = {}

    for cfg_key in df["config_key"].unique():
        sub_cfg = df[df["config_key"] == cfg_key]
        cfg_label = sub_cfg["config_label"].iloc[0]

        summary[cfg_key] = {
            "config_label": cfg_label,
            "by_duration": {},
            "overall": {
                "median_drift_pct": round(float(sub_cfg["horizontal_drift_percent"].median()), 2),
                "p90_drift_pct": round(float(sub_cfg["horizontal_drift_percent"].quantile(0.90)), 2),
                "p95_drift_pct": round(float(sub_cfg["horizontal_drift_percent"].quantile(0.95)), 2),
                "worst_drift_pct": round(float(sub_cfg["horizontal_drift_percent"].max()), 2),
                "median_final_pos_err_m": round(float(sub_cfg["final_pos_error_2d_m"].median()), 2),
                "p90_final_pos_err_m": round(float(sub_cfg["final_pos_error_2d_m"].quantile(0.90)), 2),
                "speed_rmse_mps": round(float(sub_cfg["speed_rmse_mps"].mean()), 3),
                "heading_rmse_deg": round(float(sub_cfg["heading_rmse_deg"].mean()), 2),
                "sih_pass_rate_pct": round(float(sub_cfg["sih_pass_10pct"].mean() * 100.0), 1),
                "total_windows_tested": len(sub_cfg),
            },
        }

        for dur in [10.0, 30.0, 60.0]:
            sub_dur = sub_cfg[sub_cfg["duration_s"] == dur]
            if len(sub_dur) > 0:
                summary[cfg_key]["by_duration"][f"{int(dur)}s"] = {
                    "median_drift_pct": round(float(sub_dur["horizontal_drift_percent"].median()), 2),
                    "p90_drift_pct": round(float(sub_dur["horizontal_drift_percent"].quantile(0.90)), 2),
                    "p95_drift_pct": round(float(sub_dur["horizontal_drift_percent"].quantile(0.95)), 2),
                    "worst_drift_pct": round(float(sub_dur["horizontal_drift_percent"].max()), 2),
                    "median_final_pos_err_m": round(float(sub_dur["final_pos_error_2d_m"].median()), 2),
                    "p90_final_pos_err_m": round(float(sub_dur["final_pos_error_2d_m"].quantile(0.90)), 2),
                    "speed_rmse_mps": round(float(sub_dur["speed_rmse_mps"].mean()), 3),
                    "heading_rmse_deg": round(float(sub_dur["heading_rmse_deg"].mean()), 2),
                    "sih_pass_rate_pct": round(float(sub_dur["sih_pass_10pct"].mean() * 100.0), 1),
                    "window_count": len(sub_dur),
                }

    return summary


def main():
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("\n=======================================================")
    print("  PHASE 46: AUTHENTIC MODEL -> NAVIGATION IMPACT VALIDATION")
    print("=======================================================")

    # 1. Direct AI Comparison on Drive Y1 (Driver D - Untouched Held-Out Test)
    drive_y1_dir = Path("data/raw/categorised_authentic/Y (Driver D)/Y1")
    logger.info("Running Direct AI Metrics Comparison on Untouched Test Set (Drive Y1 / Driver D)...")
    direct_y1 = run_direct_ai_comparison(drive_y1_dir, "Y1", "HELD_OUT_TEST")

    # Direct AI Comparison on Drive M (Driver B - Validation)
    drive_m_dir = Path("data/raw/categorised_authentic/M (Driver B)")
    logger.info("Running Direct AI Metrics Comparison on Validation Set (Drive M / Driver B)...")
    direct_m = run_direct_ai_comparison(drive_m_dir, "M", "VALIDATION")

    direct_report = {
        "held_out_test_y1": direct_y1,
        "validation_m": direct_m,
    }
    with open(reports_dir / "PHASE46_AI_DIRECT_COMPARISON.json", "w", encoding="utf-8") as f:
        json.dump(direct_report, f, indent=2)

    # 2. Blackout Suite on Drive M (Validation) and Drive Y1 (Test)
    logger.info("Executing Controlled Blackout Suite on Validation Drive M...")
    bo_results_m = run_controlled_blackout_suite(drive_m_dir, "M", "validation")

    logger.info("Executing Controlled Blackout Suite on Untouched Test Drive Y1...")
    bo_results_y1 = run_controlled_blackout_suite(drive_y1_dir, "Y1", "held_out_test")

    all_bo_results = bo_results_m + bo_results_y1
    agg_summary_m = aggregate_blackout_metrics(bo_results_m)
    agg_summary_y1 = aggregate_blackout_metrics(bo_results_y1)
    agg_summary_all = aggregate_blackout_metrics(all_bo_results)

    bo_report = {
        "metadata": {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_blackout_runs": len(all_bo_results),
            "durations_s": [10.0, 30.0, 60.0],
            "configs_tested": ["Pure IMU", "Historical Synthetic", "Authentic Huber", "Authentic Gate"],
        },
        "validation_drive_m": agg_summary_m,
        "held_out_test_drive_y1": agg_summary_y1,
        "combined_benchmark": agg_summary_all,
        "individual_runs": all_bo_results,
    }
    with open(reports_dir / "PHASE46_BLACKOUT_BENCHMARK_RESULTS.json", "w", encoding="utf-8") as f:
        json.dump(bo_report, f, indent=2)

    # 3. Physical Walking Session Offline Evaluation
    logger.info("Evaluating Physical Walking Session (exp_20260908_172204_two_wheeler)...")
    phys_report = run_physical_walking_comparison()
    with open(reports_dir / "PHASE46_PHYSICAL_WALK_COMPARISON.json", "w", encoding="utf-8") as f:
        json.dump(phys_report, f, indent=2)

    print("\n--- PHASE 46 DIRECT AI COMPARISON (Drive Y1 Test) ---")
    for k, v in direct_y1.items():
        print(f"[{v['checkpoint_name']}] MAE: {v['mae_mps']:.3f} m/s ({v['mae_kmh']:.2f} km/h), RMSE: {v['rmse_mps']:.3f} m/s, Stat Err: {v['stationary_mae_mps']:.3f} m/s, Bias: {v['mean_bias_mps']:.3f} m/s")

    print("\n--- PHASE 46 BLACKOUT BENCHMARK (Combined Val + Test) ---")
    for k, v in agg_summary_all.items():
        print(f"[{v['config_label']}] Median Drift: {v['overall']['median_drift_pct']}%, P90 Drift: {v['overall']['p90_drift_pct']}%, Worst Drift: {v['overall']['worst_drift_pct']}%, P90 Pos Err: {v['overall']['p90_final_pos_err_m']}m, SIH Pass Rate: {v['overall']['sih_pass_rate_pct']}%")

    print("\n--- PHASE 46 PHYSICAL WALKING TELEMETRY CHECK ---")
    for k, v in phys_report.items():
        print(f"[{v['checkpoint_name']}] Stat Initial: {v['stationary_initial_mean_mps']:.2f} m/s, Walk Peak: {v['walking_peak_mps']:.2f} m/s, Walk Mean: {v['walking_mean_mps']:.2f} m/s, Stop Final: {v['stopping_final_mps']:.2f} m/s")


if __name__ == "__main__":
    main()
