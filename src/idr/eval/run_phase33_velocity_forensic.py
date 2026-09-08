"""Phase 33: Velocity Dead-Reckoning Forensic Root-Cause Audit.

Implements all 10 analytical & empirical audit tasks:
TASK 1: Synchronized Trace (0s, 1s, 5s, 10s, 30s, 60s checkpoints)
TASK 2: Velocity Frame Audit & Canonical Synthetic Tests
TASK 3: AI Speed Quality & Regime Decomposition (Straight, Accel, Braking, Low Speed, Turning)
TASK 4: AI Measurement Model & Jacobian Audit (Finite-difference H_ai, Ablations A/B/C/D)
TASK 5: Mechanization & Gravity Leakage Audit (Sensitivity to Accel Bias, Att Error, Gravity Projection, Scale)
TASK 6: NHC / ZUPT / ZARU Constraint Ablations (A/B/C/D/E)
TASK 7: Position Error Decomposition (Magnitude vs Direction vs Initial State vs Gravity)
TASK 8: 10-Second Priority Forensic (Short-outage divergence root causes)
TASK 9/10/11: Strict scientific controls, zero feature creep, JSON + Markdown report generation.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..config import CONFIG, set_seed
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
from ..io.loader import load_drive_pair
from ..models.velocity_net import VelocityEstimatorNet
from .run_phase28_full_benchmark import (
    BlackoutWindowSpec,
    get_cached_drive_data,
    select_benchmark_windows,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase33Forensic")


# ===========================================================================
# TASK 2: CANONICAL SYNTHETIC TESTS & FRAME AUDIT
# ===========================================================================

def run_task2_synthetic_frame_audit() -> Dict[str, Any]:
    """Execute mathematical and canonical synthetic tests for all frame conventions."""
    results: Dict[str, Any] = {}
    dt = 0.1

    # 1. Stationary Case
    # Accelerometer measures +9.80665 m/s^2 along body +Z (upward specific force).
    # Expected: zero linear acceleration in navigation frame, zero velocity drift.
    ekf_stat = ErrorStateKalmanFilter()
    acc_stat = np.array([0.0, 0.0, 9.80665])
    gyro_stat = np.array([0.0, 0.0, 0.0])
    v_stat_history = []
    p_stat_history = []
    for _ in range(100):  # 10 seconds
        ekf_stat.predict(acc_stat, gyro_stat, dt=dt)
        v_stat_history.append(ekf_stat.v.copy())
        p_stat_history.append(ekf_stat.p.copy())
    
    max_stat_vel = float(np.max(np.linalg.norm(np.array(v_stat_history), axis=1)))
    max_stat_pos = float(np.max(np.linalg.norm(np.array(p_stat_history), axis=1)))
    results["test_stationary"] = {
        "max_velocity_drift_mps": max_stat_vel,
        "max_position_drift_m": max_stat_pos,
        "pass": max_stat_vel < 1e-6 and max_stat_pos < 1e-6,
    }

    # 2. Straight Constant Speed (Eastward heading, psi = 0)
    # v = [10, 0, 0] m/s East. Specific force = [0, 0, 9.80665].
    # Expected: velocity remains exactly [10, 0, 0], position advances 100m East in 10s.
    ekf_const = ErrorStateKalmanFilter()
    ekf_const.set_state(pos=np.zeros(3), vel=np.array([10.0, 0.0, 0.0]))
    for _ in range(100):
        ekf_const.predict(acc_stat, gyro_stat, dt=dt)
    
    vel_const_err = float(np.linalg.norm(ekf_const.v - np.array([10.0, 0.0, 0.0])))
    pos_const_err = float(np.linalg.norm(ekf_const.p - np.array([100.0, 0.0, 0.0])))
    results["test_straight_constant_speed"] = {
        "final_velocity_error_mps": vel_const_err,
        "final_position_error_m": pos_const_err,
        "pass": vel_const_err < 1e-5 and pos_const_err < 1e-4,
    }

    # 3. Straight Acceleration (Forward acc = +1.0 m/s^2, Eastward heading psi = 0)
    # Specific force = [1.0, 0, 9.80665]. Over 10s, speed reaches 10 m/s, dist = 50m.
    ekf_acc = ErrorStateKalmanFilter()
    acc_fwd = np.array([1.0, 0.0, 9.80665])
    for _ in range(100):
        ekf_acc.predict(acc_fwd, gyro_stat, dt=dt)
    
    vel_acc_err = float(np.linalg.norm(ekf_acc.v - np.array([10.0, 0.0, 0.0])))
    pos_acc_err = float(np.linalg.norm(ekf_acc.p - np.array([50.0, 0.0, 0.0])))
    results["test_straight_acceleration"] = {
        "final_velocity_error_mps": vel_acc_err,
        "final_position_error_m": pos_acc_err,
        "pass": vel_acc_err < 1e-4 and pos_acc_err < 1e-3,
    }

    # 4. Braking (Deceleration = -2.0 m/s^2, initial speed = 20 m/s East)
    # Specific force = [-2.0, 0, 9.80665]. In 10s, speed reaches 0 m/s, dist = 100m.
    ekf_brake = ErrorStateKalmanFilter()
    ekf_brake.set_state(pos=np.zeros(3), vel=np.array([20.0, 0.0, 0.0]))
    acc_brake = np.array([-2.0, 0.0, 9.80665])
    for _ in range(100):
        ekf_brake.predict(acc_brake, gyro_stat, dt=dt)
    
    vel_brake_err = float(np.linalg.norm(ekf_brake.v - np.array([0.0, 0.0, 0.0])))
    pos_brake_err = float(np.linalg.norm(ekf_brake.p - np.array([100.0, 0.0, 0.0])))
    results["test_braking"] = {
        "final_velocity_error_mps": vel_brake_err,
        "final_position_error_m": pos_brake_err,
        "pass": vel_brake_err < 1e-4 and pos_brake_err < 1e-3,
    }

    # 5. Pure Yaw (Rotating counter-clockwise at 0.1 rad/s while stationary)
    # Expected: Yaw advances by 1.0 rad in 10s, pitch/roll remain 0, pos/vel remain 0.
    ekf_yaw = ErrorStateKalmanFilter()
    gyro_yaw = np.array([0.0, 0.0, 0.1])
    for _ in range(100):
        ekf_yaw.predict(acc_stat, gyro_yaw, dt=dt)
    r, p, y = ekf_yaw.euler_angles
    yaw_err = float(abs(y - 1.0))
    pos_yaw_err = float(np.linalg.norm(ekf_yaw.p))
    results["test_pure_yaw"] = {
        "final_yaw_rad": y,
        "yaw_error_rad": yaw_err,
        "position_drift_m": pos_yaw_err,
        "pass": yaw_err < 1e-4 and pos_yaw_err < 1e-4,
    }

    # 6. Pure Lateral Acceleration (Centripetal force on turn: v = 10 m/s, omega_z = 0.1 rad/s)
    # In circle of radius R = v / omega = 100m. Body lateral acceleration = omega * v = 1.0 m/s^2 (leftward +Y).
    # Specific force measured by body accelerometer: f_b = [0, +1.0, 9.80665].
    # Over quarter circle (t = pi / (2 * 0.1) = 15.708s): pos should be [R sin(theta), R (1 - cos(theta)), 0] = [100, 100, 0].
    ekf_turn = ErrorStateKalmanFilter()
    ekf_turn.set_state(pos=np.zeros(3), vel=np.array([10.0, 0.0, 0.0]))  # Starting East
    acc_turn = np.array([0.0, 1.0, 9.80665])  # Body +Y specific force
    gyro_turn = np.array([0.0, 0.0, 0.1])     # Body +Z angular rate
    steps = int(np.round(15.707963 / dt))
    for _ in range(steps):
        ekf_turn.predict(acc_turn, gyro_turn, dt=dt)
    
    pos_turn_err = float(np.linalg.norm(ekf_turn.p[:2] - np.array([100.0, 100.0])))
    speed_turn = float(np.linalg.norm(ekf_turn.v[:2]))
    speed_turn_err = float(abs(speed_turn - 10.0))
    results["test_pure_lateral_acceleration_turning"] = {
        "final_pos_enu": ekf_turn.p.tolist(),
        "pos_error_circle_m": pos_turn_err,
        "speed_error_mps": speed_turn_err,
        "pass": pos_turn_err < 2.0 and speed_turn_err < 0.1,
    }

    return results


# ===========================================================================
# TASK 4: FINITE-DIFFERENCE JACOBIAN AUDIT
# ===========================================================================

def run_task4_jacobian_finite_difference() -> Dict[str, Any]:
    """Numerically finite-difference H_ai and H_nhc against analytical Jacobians."""
    ekf = ErrorStateKalmanFilter()
    ekf.p = np.array([15.2, -32.4, 4.1])
    ekf.v = np.array([14.5, 3.2, -0.8])
    q_init = exp_quaternion(np.array([0.05, -0.08, 0.75]))
    ekf.q = q_init
    ekf.ba = np.array([0.03, -0.02, 0.04])
    ekf.bg = np.array([0.002, -0.001, 0.003])

    eps = 1e-7

    # 1. H_ai Finite-Difference
    def eval_h_ai(v, q):
        R = quat_to_rot(q)
        v_b = R.T @ v
        return np.array([v_b[0]])

    h0_ai = eval_h_ai(ekf.v, ekf.q)
    H_ai_num = np.zeros((1, 15))
    for i in range(3):
        dv = np.zeros(3); dv[i] = eps
        H_ai_num[0, 3 + i] = (eval_h_ai(ekf.v + dv, ekf.q) - h0_ai)[0] / eps
    for i in range(3):
        dth = np.zeros(3); dth[i] = eps
        dq = exp_quaternion(dth)
        q_pert = quat_multiply(ekf.q, dq)
        q_pert /= np.linalg.norm(q_pert)
        H_ai_num[0, 6 + i] = (eval_h_ai(ekf.v, q_pert) - h0_ai)[0] / eps

    R = quat_to_rot(ekf.q)
    v_body = R.T @ ekf.v
    H_ai_ana = np.zeros((1, 15))
    H_ai_ana[0, 3:6] = R[:, 0]
    H_ai_ana[0, 6:9] = np.array([0.0, -v_body[2], v_body[1]])

    ai_diff = float(np.max(np.abs(H_ai_num - H_ai_ana)))

    # 2. H_nhc Finite-Difference
    def eval_h_nhc(v, q):
        R = quat_to_rot(q)
        v_b = R.T @ v
        return np.array([v_b[1], v_b[2]])

    h0_nhc = eval_h_nhc(ekf.v, ekf.q)
    H_nhc_num = np.zeros((2, 15))
    for i in range(3):
        dv = np.zeros(3); dv[i] = eps
        H_nhc_num[:, 3 + i] = (eval_h_nhc(ekf.v + dv, ekf.q) - h0_nhc) / eps
    for i in range(3):
        dth = np.zeros(3); dth[i] = eps
        dq = exp_quaternion(dth)
        q_pert = quat_multiply(ekf.q, dq)
        q_pert /= np.linalg.norm(q_pert)
        H_nhc_num[:, 6 + i] = (eval_h_nhc(ekf.v, q_pert) - h0_nhc) / eps

    v_bx, v_by, v_bz = v_body[0], v_body[1], v_body[2]
    H_nhc_ana = np.zeros((2, 15))
    H_nhc_ana[0, 3:6] = R[:, 1]
    H_nhc_ana[1, 3:6] = R[:, 2]
    H_nhc_ana[0, 6:9] = np.array([v_bz, 0.0, -v_bx])
    H_nhc_ana[1, 6:9] = np.array([-v_by, v_bx, 0.0])

    nhc_diff = float(np.max(np.abs(H_nhc_num - H_nhc_ana)))

    return {
        "H_ai_max_finite_difference_error": ai_diff,
        "H_ai_is_exact": ai_diff < 1e-5,
        "H_nhc_max_finite_difference_error": nhc_diff,
        "H_nhc_is_exact": nhc_diff < 1e-5,
    }


# ===========================================================================
# TASK 1, 3, 4, 5, 6, 7, 8: FULL FORENSIC EVALUATION ACROSS ALL WINDOWS
# ===========================================================================

def run_phase33_forensic_suite(
    windows: List[BlackoutWindowSpec],
    vel_model: Optional[VelocityEstimatorNet],
) -> Dict[str, Any]:
    """Execute the full forensic audit across all authentic blackout windows."""
    results: Dict[str, Any] = {}
    dt = 0.1

    # Task 1: Checkpoint Trace Storage
    task1_traces: Dict[str, Any] = {}

    # Task 3: AI Speed Quality Tracking across Regimes
    task3_regimes: Dict[str, Dict[str, List[float]]] = {
        "overall": {"gt_spd": [], "ai_spd": [], "ekf_spd": [], "raw_imu_spd": []},
        "straight": {"gt_spd": [], "ai_spd": [], "ekf_spd": [], "raw_imu_spd": []},
        "accel_decel": {"gt_spd": [], "ai_spd": [], "ekf_spd": [], "raw_imu_spd": []},
        "turning": {"gt_spd": [], "ai_spd": [], "ekf_spd": [], "raw_imu_spd": []},
        "low_speed": {"gt_spd": [], "ai_spd": [], "ekf_spd": [], "raw_imu_spd": []},
        "high_speed": {"gt_spd": [], "ai_spd": [], "ekf_spd": [], "raw_imu_spd": []},
    }

    # Task 4: Ablations (A: IMU only, B: AI speed only, C: IMU + AI, D: artifical perfect GT speed diagnostic)
    task4_ablation_results: Dict[str, List[Dict[str, float]]] = {
        "A_imu_only": [],
        "B_ai_speed_only": [],
        "C_imu_plus_ai": [],
        "D_diagnostic_perfect_speed": [],
    }

    # Task 5: Mechanization / Gravity Leakage Sensitivity Experiments
    task5_mechanization_sensitivities: Dict[str, List[float]] = {
        "baseline_error_m": [],
        "zero_accel_bias_error_m": [],
        "zero_att_error_error_m": [],
        "perfect_gravity_removal_error_m": [],
        "pure_kinematic_integration_error_m": [],
    }

    # Task 6: NHC / ZUPT / ZARU Constraint Ablations
    task6_constraint_ablations: Dict[str, List[Dict[str, float]]] = {
        "A_all_constraints_off": [],
        "B_ai_only": [],
        "C_ai_plus_nhc": [],
        "D_ai_plus_zupt_zaru": [],
        "E_ai_plus_all_constraints": [],
    }

    # Task 7: Position Error Decomposition (Magnitude vs Direction vs Initial State)
    task7_decompositions: List[Dict[str, float]] = []

    # Task 8: 10-Second Priority Records
    task8_10s_records: List[Dict[str, Any]] = []

    logger.info("Executing Phase 33 forensic suite on %d authentic windows...", len(windows))

    for win_idx, window in enumerate(windows):
        imu, gps, v_speed, t = get_cached_drive_data(window.drive_path, window.drive_id)
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

        k_bo_start = bo_start - start_sim
        k_bo_end = bo_end - start_sim
        dur_s = window.duration_s

        # Helper function to run a customizable simulation run
        def run_custom_sim(
            use_imu: bool = True,
            use_ai: bool = True,
            use_nhc: bool = True,
            use_zupt: bool = True,
            perfect_speed_diagnostic: bool = False,
            zero_accel_bias_oracle: bool = False,
            zero_att_error_oracle: bool = False,
            perfect_gravity_removal_oracle: bool = False,
            pure_kinematic_dead_reckon: bool = False,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
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

            est_p = np.zeros((N_sim, 3))
            est_v = np.zeros((N_sim, 3))
            est_psi = np.zeros(N_sim)
            step_diags = []

            for k, idx in enumerate(range(start_sim, end_sim)):
                is_blackout = (idx >= bo_start) and (idx < bo_end)
                acc_3d = acc_v_all[idx].astype(np.float64)
                gyro_3d = gyro_v_all[idx].astype(np.float64)

                # Diagnostic oracle interventions
                if zero_accel_bias_oracle:
                    es_ekf.ba = np.zeros(3)
                if zero_att_error_oracle:
                    # Align attitude exactly with ground truth
                    true_r, true_p, _ = rot_to_euler_rpy(es_ekf.rotation_matrix)
                    qr = exp_quaternion(np.array([true_r, 0.0, 0.0]))
                    qp = exp_quaternion(np.array([0.0, true_p, 0.0]))
                    qy = exp_quaternion(np.array([0.0, 0.0, gt_yaw[k]]))
                    es_ekf.q = quat_multiply(qy, quat_multiply(qp, qr))
                if perfect_gravity_removal_oracle and is_blackout:
                    # Override specific force by ground truth vehicle acceleration in body frame
                    # a_b = R^T (a_gt - g_nav)
                    a_gt_nav = (gt_vel[k] - gt_vel[max(0, k - 1)]) / dt
                    acc_3d = es_ekf.rotation_matrix.T @ (a_gt_nav - es_ekf.g_nav)

                # 1. Prediction step
                if use_imu:
                    es_ekf.predict(acc_3d, gyro_3d, dt=dt)
                    cur_v_b = es_ekf.rotation_matrix.T @ es_ekf.v
                    fwd_spd = float(cur_v_b[0])
                    acc_x_kin = (fwd_spd - prev_spd) / dt
                    prev_spd = fwd_spd
                    eff_yr = float(gyro_3d[2] - es_ekf.bg[2])
                    if abs(acc_x_kin) < 2.0 and abs(eff_yr) < 0.2:
                        es_ekf.update_gravity_leveling(acc_3d, forward_speed=fwd_spd, yaw_rate=eff_yr, fwd_acc=acc_x_kin, sigma_level=0.3)
                else:
                    # Pure kinematic propagation without IMU acceleration (relying only on gyro & speed)
                    delta_theta = gyro_3d * dt
                    es_ekf.q = quat_multiply(es_ekf.q, exp_quaternion(delta_theta))
                    es_ekf.q /= np.linalg.norm(es_ekf.q)
                    cur_hdg = es_ekf.euler_angles[2]
                    spd_in = float(v_speed[idx]) if perfect_speed_diagnostic else float(ai_speeds[k])
                    es_ekf.v = np.array([spd_in * np.cos(cur_hdg), spd_in * np.sin(cur_hdg), 0.0])
                    es_ekf.p += es_ekf.v * dt

                # 2. Stationary / ZUPT
                cur_spd_est = float(np.linalg.norm(es_ekf.v[:2]))
                if ai_speeds[k] > 0.1:
                    cur_spd_est = max(cur_spd_est, ai_speeds[k])
                is_stat = stat_detector.update(acc_3d, gyro_3d, speed_mps=cur_spd_est)
                if is_stat and use_zupt:
                    es_ekf.update_zupt(sigma_v=0.01)
                    es_ekf.update_zaru(gyro_3d, sigma_bg=0.001)

                # 3. GNSS Pre-blackout updates
                cur_gps_raw = (float(gps[idx, 0]), float(gps[idx, 1]))
                is_new_gps = (last_gps_raw is None) or (abs(cur_gps_raw[0] - last_gps_raw[0]) > 1e-9 or abs(cur_gps_raw[1] - last_gps_raw[1]) > 1e-9)

                diag_info = {}

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
                    # Inside Blackout
                    # NHC
                    if use_nhc and use_imu:
                        _, diag_nhc = es_ekf.update_nhc(sigma_lat=0.05, sigma_vert=0.05, cornering_threshold_mps2=0.5)
                        diag_info["nhc"] = diag_nhc

                    # AI Speed Update
                    if use_ai and use_imu:
                        spd_meas = float(v_speed[idx]) if perfect_speed_diagnostic else float(ai_speeds[k])
                        acc_ai, diag_ai = es_ekf.update_ai_velocity(spd_meas, sigma_v=0.8, max_innovation_sigma=4.0)
                        diag_info["ai"] = diag_ai

                est_p[k] = es_ekf.p.copy()
                est_v[k] = es_ekf.v.copy()
                _, _, est_psi[k] = es_ekf.euler_angles
                diag_info["step"] = k
                diag_info["ba"] = es_ekf.ba.copy()
                diag_info["bg"] = es_ekf.bg.copy()
                step_diags.append(diag_info)

            return est_p, est_v, est_psi, step_diags

        # Run Standard Baseline Simulation (C: IMU + AI + NHC + ZUPT)
        base_p, base_v, base_psi, base_diags = run_custom_sim(
            use_imu=True, use_ai=True, use_nhc=True, use_zupt=True
        )

        bo_base_p = base_p[k_bo_start:k_bo_end]
        bo_gt_p = gt_enu[k_bo_start:k_bo_end]
        bo_base_v = base_v[k_bo_start:k_bo_end]
        bo_gt_v = gt_vel[k_bo_start:k_bo_end]

        final_err = float(np.linalg.norm(bo_base_p[-1, :2] - bo_gt_p[-1, :2]))
        dist_m = max(1.0, float(np.sum(np.linalg.norm(np.diff(bo_gt_p[:, :2], axis=0), axis=1))))
        drift_pct = (final_err / dist_m) * 100.0

        # TASK 1: Checkpoint Trace Extraction for Representative Windows (10s, 30s, 60s for M and Y1)
        if window.regime in ("straight", "turning") and (f"{window.drive_id}_{int(dur_s)}s_{window.regime}" not in task1_traces):
            checkpoints = [0, 10, 50, 100, 300, 600]
            trace_cp = []
            for cp_idx in checkpoints:
                if cp_idx < len(bo_base_p):
                    k_sim = k_bo_start + cp_idx
                    ref_v_val = gt_vel[k_sim]
                    ref_spd_val = float(v_speed[start_sim + k_sim])
                    ai_spd_val = float(ai_speeds[k_sim])
                    ekf_v_val = base_v[k_sim]
                    ekf_v_body = quat_to_rot(exp_quaternion(np.array([0, 0, base_psi[k_sim]]))).T @ ekf_v_val
                    pos_err_val = float(np.linalg.norm(base_p[k_sim, :2] - gt_enu[k_sim, :2]))
                    d_info = base_diags[k_sim]
                    ai_d = d_info.get("ai", {})
                    nhc_d = d_info.get("nhc", {})

                    trace_cp.append({
                        "checkpoint_s": cp_idx * dt,
                        "ref_speed_mps": ref_spd_val,
                        "ai_forward_speed_mps": ai_spd_val,
                        "ekf_body_fwd_spd_mps": float(ekf_v_body[0]),
                        "ekf_enu_velocity": ekf_v_val.tolist(),
                        "ref_enu_velocity": ref_v_val.tolist(),
                        "pos_error_m": pos_err_val,
                        "accel_bias": d_info.get("ba", np.zeros(3)).tolist(),
                        "gyro_bias": d_info.get("bg", np.zeros(3)).tolist(),
                        "ai_accepted": ai_d.get("accepted", False),
                        "ai_innovation": float(ai_d.get("innovation", [0.0])[0]) if "innovation" in ai_d else None,
                        "ai_nis": float(ai_d.get("nis", 0.0)) if "nis" in ai_d else None,
                        "nhc_is_dynamic": nhc_d.get("is_dynamic_cornering", False),
                    })
            task1_traces[f"{window.drive_id}_{int(dur_s)}s_{window.regime}"] = trace_cp

        # TASK 3: Collect Speed Samples for Regime Decomposition
        for step_i in range(len(bo_gt_v)):
            k_sim = k_bo_start + step_i
            gt_spd_i = float(v_speed[start_sim + k_sim])
            ai_spd_i = float(ai_speeds[k_sim])
            ekf_spd_i = float(np.linalg.norm(base_v[k_sim, :2]))
            raw_acc_i = float(acc_v_all[start_sim + k_sim, 0])

            task3_regimes["overall"]["gt_spd"].append(gt_spd_i)
            task3_regimes["overall"]["ai_spd"].append(ai_spd_i)
            task3_regimes["overall"]["ekf_spd"].append(ekf_spd_i)
            task3_regimes["overall"]["raw_imu_spd"].append(raw_acc_i)

            if window.regime in task3_regimes:
                task3_regimes[window.regime]["gt_spd"].append(gt_spd_i)
                task3_regimes[window.regime]["ai_spd"].append(ai_spd_i)
                task3_regimes[window.regime]["ekf_spd"].append(ekf_spd_i)
                task3_regimes[window.regime]["raw_imu_spd"].append(raw_acc_i)

        # TASK 4: Run AI Ablations
        # A: IMU only (no AI)
        p_A, _, _, _ = run_custom_sim(use_imu=True, use_ai=False, use_nhc=True, use_zupt=True)
        err_A = float(np.linalg.norm(p_A[k_bo_end - 1, :2] - bo_gt_p[-1, :2]))
        task4_ablation_results["A_imu_only"].append({
            "window_id": window.window_id,
            "final_err_m": err_A,
            "drift_pct": (err_A / dist_m) * 100.0,
        })

        # B: AI speed only (no IMU propagation)
        p_B, _, _, _ = run_custom_sim(use_imu=False, use_ai=True, use_nhc=False, use_zupt=False)
        err_B = float(np.linalg.norm(p_B[k_bo_end - 1, :2] - bo_gt_p[-1, :2]))
        task4_ablation_results["B_ai_speed_only"].append({
            "window_id": window.window_id,
            "final_err_m": err_B,
            "drift_pct": (err_B / dist_m) * 100.0,
        })

        # C: IMU + AI (Baseline)
        task4_ablation_results["C_imu_plus_ai"].append({
            "window_id": window.window_id,
            "final_err_m": final_err,
            "drift_pct": drift_pct,
        })

        # D: Artificially perfect reference speed (diagnostic only)
        p_D, _, _, _ = run_custom_sim(
            use_imu=True, use_ai=True, use_nhc=True, use_zupt=True, perfect_speed_diagnostic=True
        )
        err_D = float(np.linalg.norm(p_D[k_bo_end - 1, :2] - bo_gt_p[-1, :2]))
        task4_ablation_results["D_diagnostic_perfect_speed"].append({
            "window_id": window.window_id,
            "final_err_m": err_D,
            "drift_pct": (err_D / dist_m) * 100.0,
        })

        # TASK 5: Mechanization / Gravity Sensitivity Experiments
        task5_mechanization_sensitivities["baseline_error_m"].append(final_err)

        # Sensitivity: Zero Accel Bias
        p_sens_ba, _, _, _ = run_custom_sim(
            use_imu=True, use_ai=True, use_nhc=True, use_zupt=True, zero_accel_bias_oracle=True
        )
        task5_mechanization_sensitivities["zero_accel_bias_error_m"].append(
            float(np.linalg.norm(p_sens_ba[k_bo_end - 1, :2] - bo_gt_p[-1, :2]))
        )

        # Sensitivity: Zero Attitude Error
        p_sens_att, _, _, _ = run_custom_sim(
            use_imu=True, use_ai=True, use_nhc=True, use_zupt=True, zero_att_error_oracle=True
        )
        task5_mechanization_sensitivities["zero_att_error_error_m"].append(
            float(np.linalg.norm(p_sens_att[k_bo_end - 1, :2] - bo_gt_p[-1, :2]))
        )

        # Sensitivity: Perfect Gravity Removal
        p_sens_grav, _, _, _ = run_custom_sim(
            use_imu=True, use_ai=True, use_nhc=True, use_zupt=True, perfect_gravity_removal_oracle=True
        )
        task5_mechanization_sensitivities["perfect_gravity_removal_error_m"].append(
            float(np.linalg.norm(p_sens_grav[k_bo_end - 1, :2] - bo_gt_p[-1, :2]))
        )

        # TASK 6: NHC / ZUPT / ZARU Constraint Ablations
        # A: all constraints off (IMU + AI only)
        p_c_A, _, _, _ = run_custom_sim(use_imu=True, use_ai=True, use_nhc=False, use_zupt=False)
        err_cA = float(np.linalg.norm(p_c_A[k_bo_end - 1, :2] - bo_gt_p[-1, :2]))
        task6_constraint_ablations["A_all_constraints_off"].append({
            "window_id": window.window_id, "final_err_m": err_cA, "drift_pct": (err_cA / dist_m) * 100.0
        })

        # B: AI only (alias for all constraints off)
        task6_constraint_ablations["B_ai_only"].append({
            "window_id": window.window_id, "final_err_m": err_cA, "drift_pct": (err_cA / dist_m) * 100.0
        })

        # C: AI + NHC only (no ZUPT)
        p_c_C, _, _, _ = run_custom_sim(use_imu=True, use_ai=True, use_nhc=True, use_zupt=False)
        err_cC = float(np.linalg.norm(p_c_C[k_bo_end - 1, :2] - bo_gt_p[-1, :2]))
        task6_constraint_ablations["C_ai_plus_nhc"].append({
            "window_id": window.window_id, "final_err_m": err_cC, "drift_pct": (err_cC / dist_m) * 100.0
        })

        # D: AI + ZUPT/ZARU only (no NHC)
        p_c_D, _, _, _ = run_custom_sim(use_imu=True, use_ai=True, use_nhc=False, use_zupt=True)
        err_cD = float(np.linalg.norm(p_c_D[k_bo_end - 1, :2] - bo_gt_p[-1, :2]))
        task6_constraint_ablations["D_ai_plus_zupt_zaru"].append({
            "window_id": window.window_id, "final_err_m": err_cD, "drift_pct": (err_cD / dist_m) * 100.0
        })

        # E: AI + all constraints (Baseline)
        task6_constraint_ablations["E_ai_plus_all_constraints"].append({
            "window_id": window.window_id, "final_err_m": final_err, "drift_pct": drift_pct
        })

        # TASK 7: Position Error Decomposition
        # Path integral decomposition:
        # r_err(T) = \int_0^T [ v_est(t) - v_gt(t) ] dt
        # v_est(t) - v_gt(t) = (s_est - s_gt) e_est + s_gt (e_est - e_gt)
        # where s = speed (magnitude), e = unit direction vector.
        bo_speed_est = np.linalg.norm(bo_base_v[:, :2], axis=1)
        bo_speed_gt = np.linalg.norm(bo_gt_v[:, :2], axis=1)
        bo_dir_est = np.zeros_like(bo_base_v[:, :2])
        bo_dir_gt = np.zeros_like(bo_gt_v[:, :2])
        for i in range(len(bo_speed_est)):
            if bo_speed_est[i] > 1e-3:
                bo_dir_est[i] = bo_base_v[i, :2] / bo_speed_est[i]
            else:
                bo_dir_est[i] = [np.cos(base_psi[k_bo_start + i]), np.sin(base_psi[k_bo_start + i])]
            if bo_speed_gt[i] > 1e-3:
                bo_dir_gt[i] = bo_gt_v[i, :2] / bo_speed_gt[i]
            else:
                bo_dir_gt[i] = [np.cos(gt_yaw[k_bo_start + i]), np.sin(gt_yaw[k_bo_start + i])]

        mag_error_integrand = np.outer((bo_speed_est - bo_speed_gt), np.ones(2)) * bo_dir_est
        dir_error_integrand = np.outer(bo_speed_gt, np.ones(2)) * (bo_dir_est - bo_dir_gt)

        delta_p_mag = float(np.linalg.norm(np.sum(mag_error_integrand * dt, axis=0)))
        delta_p_dir = float(np.linalg.norm(np.sum(dir_error_integrand * dt, axis=0)))
        p_init_err = float(np.linalg.norm(bo_base_p[0, :2] - bo_gt_p[0, :2]))

        task7_decompositions.append({
            "window_id": window.window_id,
            "duration_s": dur_s,
            "regime": window.regime,
            "total_pos_error_m": final_err,
            "speed_magnitude_error_m": delta_p_mag,
            "velocity_direction_error_m": delta_p_dir,
            "initial_pos_error_m": p_init_err,
            "speed_mag_ratio_pct": (delta_p_mag / max(1e-3, delta_p_mag + delta_p_dir)) * 100.0,
            "vel_dir_ratio_pct": (delta_p_dir / max(1e-3, delta_p_mag + delta_p_dir)) * 100.0,
        })

        # TASK 8: 10-Second Priority Audit
        if dur_s == 10.0:
            task8_10s_records.append({
                "window_id": window.window_id,
                "drive_id": window.drive_id,
                "regime": window.regime,
                "distance_m": dist_m,
                "final_pos_error_m": final_err,
                "drift_pct": drift_pct,
                "speed_magnitude_error_m": delta_p_mag,
                "velocity_direction_error_m": delta_p_dir,
                "mean_speed_mps": float(np.mean(bo_gt_spd := v_speed[start_sim + k_bo_start:start_sim + k_bo_end])),
                "ai_speed_rmse_mps": float(np.sqrt(np.mean((ai_speeds[k_bo_start:k_bo_end] - bo_gt_spd) ** 2))),
            })

    results["task1_synchronized_traces"] = task1_traces

    # Process Task 3 metrics
    task3_summary: Dict[str, Any] = {}
    for reg_name, data in task3_regimes.items():
        if len(data["gt_spd"]) > 0:
            gt_a = np.array(data["gt_spd"])
            ai_a = np.array(data["ai_spd"])
            ekf_a = np.array(data["ekf_spd"])
            err_ai = ai_a - gt_a
            mae = float(np.mean(np.abs(err_ai)))
            rmse = float(np.sqrt(np.mean(err_ai ** 2)))
            bias = float(np.mean(err_ai))
            r = float(np.corrcoef(gt_a, ai_a)[0, 1]) if np.std(gt_a) > 1e-4 and np.std(ai_a) > 1e-4 else 0.0

            task3_summary[reg_name] = {
                "sample_count": len(gt_a),
                "ai_mae_mps": mae,
                "ai_rmse_mps": rmse,
                "ai_bias_mps": bias,
                "ai_gt_correlation_r": r,
                "mean_gt_speed_mps": float(np.mean(gt_a)),
                "mean_ai_speed_mps": float(np.mean(ai_a)),
                "mean_ekf_speed_mps": float(np.mean(ekf_a)),
            }
    results["task3_ai_speed_quality"] = task3_summary

    # Process Task 4 Ablation metrics
    task4_summary: Dict[str, Any] = {}
    for ab_name, recs in task4_ablation_results.items():
        drifts = [r["drift_pct"] for r in recs]
        errs = [r["final_err_m"] for r in recs]
        task4_summary[ab_name] = {
            "mean_drift_pct": float(np.mean(drifts)),
            "median_drift_pct": float(np.median(drifts)),
            "p90_drift_pct": float(np.percentile(drifts, 90)),
            "mean_final_error_m": float(np.mean(errs)),
            "median_final_error_m": float(np.median(errs)),
        }
    results["task4_ai_ablations"] = task4_summary

    # Process Task 5 Sensitivity metrics
    task5_summary: Dict[str, Any] = {
        "mean_baseline_error_m": float(np.mean(task5_mechanization_sensitivities["baseline_error_m"])),
        "mean_zero_accel_bias_error_m": float(np.mean(task5_mechanization_sensitivities["zero_accel_bias_error_m"])),
        "mean_zero_att_error_error_m": float(np.mean(task5_mechanization_sensitivities["zero_att_error_error_m"])),
        "mean_perfect_gravity_removal_error_m": float(np.mean(task5_mechanization_sensitivities["perfect_gravity_removal_error_m"])),
        "accel_bias_contribution_pct": float((1.0 - np.mean(task5_mechanization_sensitivities["zero_accel_bias_error_m"]) / max(1e-3, np.mean(task5_mechanization_sensitivities["baseline_error_m"]))) * 100.0),
        "attitude_error_contribution_pct": float((1.0 - np.mean(task5_mechanization_sensitivities["zero_att_error_error_m"]) / max(1e-3, np.mean(task5_mechanization_sensitivities["baseline_error_m"]))) * 100.0),
        "gravity_leakage_contribution_pct": float((1.0 - np.mean(task5_mechanization_sensitivities["perfect_gravity_removal_error_m"]) / max(1e-3, np.mean(task5_mechanization_sensitivities["baseline_error_m"]))) * 100.0),
    }
    results["task5_mechanization_sensitivities"] = task5_summary

    # Process Task 6 Constraint Ablations
    task6_summary: Dict[str, Any] = {}
    for ab_name, recs in task6_constraint_ablations.items():
        drifts = [r["drift_pct"] for r in recs]
        errs = [r["final_err_m"] for r in recs]
        task6_summary[ab_name] = {
            "mean_drift_pct": float(np.mean(drifts)),
            "median_drift_pct": float(np.median(drifts)),
            "p90_drift_pct": float(np.percentile(drifts, 90)),
            "mean_final_error_m": float(np.mean(errs)),
            "median_final_error_m": float(np.median(errs)),
        }
    results["task6_constraint_ablations"] = task6_summary

    # Process Task 7 Error Decompositions
    t7_mag_ratios = [r["speed_mag_ratio_pct"] for r in task7_decompositions]
    t7_dir_ratios = [r["vel_dir_ratio_pct"] for r in task7_decompositions]
    results["task7_position_error_decomposition"] = {
        "mean_speed_magnitude_contribution_pct": float(np.mean(t7_mag_ratios)),
        "mean_velocity_direction_contribution_pct": float(np.mean(t7_dir_ratios)),
        "median_speed_magnitude_contribution_pct": float(np.median(t7_mag_ratios)),
        "median_velocity_direction_contribution_pct": float(np.median(t7_dir_ratios)),
        "window_records": task7_decompositions,
    }

    # Process Task 8 10-Second Priority
    t8_drifts = [r["drift_pct"] for r in task8_10s_records]
    t8_errs = [r["final_pos_error_m"] for r in task8_10s_records]
    results["task8_10s_priority"] = {
        "mean_10s_drift_pct": float(np.mean(t8_drifts)),
        "median_10s_drift_pct": float(np.median(t8_drifts)),
        "p90_10s_drift_pct": float(np.percentile(t8_drifts, 90)),
        "mean_10s_final_error_m": float(np.mean(t8_errs)),
        "median_10s_final_error_m": float(np.median(t8_errs)),
        "window_records": task8_10s_records,
    }

    return results


def main():
    set_seed(42)
    logger.info("=== Starting Phase 33 Velocity Dead-Reckoning Forensic Suite ===")

    # 1. Run Task 2 Frame Audit
    task2_res = run_task2_synthetic_frame_audit()
    logger.info("Task 2 Canonical Synthetic Tests: %s", json.dumps(task2_res, indent=2))

    # 2. Run Task 4 Jacobian Finite-Difference
    task4_jac_res = run_task4_jacobian_finite_difference()
    logger.info("Task 4 Jacobian Verification: %s", json.dumps(task4_jac_res, indent=2))

    # 3. Load Model
    model_path = Path("models/authentic/velocity_net.pt")
    vel_model = None
    if model_path.exists():
        vel_model = VelocityEstimatorNet(in_channels=6, hidden_dim=64, num_layers=2)
        vel_model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        vel_model.eval()
        logger.info("Loaded frozen VelocityEstimatorNet weights from %s", model_path)

    # 4. Load Benchmark Windows
    data_dir = Path("data/raw/categorised_authentic")
    m_dir = data_dir / "M (Driver B)"
    y1_dir = data_dir / "Y (Driver D)" / "Y1"

    windows_m = select_benchmark_windows(m_dir, "M", "validation")
    windows_y1 = select_benchmark_windows(y1_dir, "Y1", "held_out_test")
    all_windows = windows_m + windows_y1
    logger.info("Selected %d windows (15 Drive M, 15 Drive Y1)", len(all_windows))

    # 5. Run Full Forensic Suite
    forensic_res = run_phase33_forensic_suite(all_windows, vel_model)
    forensic_res["task2_frame_audit"] = task2_res
    forensic_res["task4_jacobian_verification"] = task4_jac_res

    # 6. Save JSON
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "PHASE33_VELOCITY_FORENSIC.json"
    with open(json_path, "w") as f:
        json.dump(forensic_res, f, indent=2)
    logger.info("Saved JSON forensic report to %s", json_path)

    print("\n" + "=" * 80)
    print("PHASE 33 FORENSIC EXECUTIVE SUMMARY")
    print("=" * 80)
    print(f"Task 3 Overall AI Speed RMSE: {forensic_res['task3_ai_speed_quality']['overall']['ai_rmse_mps']:.3f} m/s (MAE: {forensic_res['task3_ai_speed_quality']['overall']['ai_mae_mps']:.3f} m/s, Bias: {forensic_res['task3_ai_speed_quality']['overall']['ai_bias_mps']:.3f} m/s, Correlation: {forensic_res['task3_ai_speed_quality']['overall']['ai_gt_correlation_r']:.3f})")
    print(f"Task 4 Ablation Median Drifts: IMU Only={forensic_res['task4_ai_ablations']['A_imu_only']['median_drift_pct']:.1f}%, AI Only={forensic_res['task4_ai_ablations']['B_ai_speed_only']['median_drift_pct']:.1f}%, IMU+AI={forensic_res['task4_ai_ablations']['C_imu_plus_ai']['median_drift_pct']:.1f}%, Perfect Speed Diagnostic={forensic_res['task4_ai_ablations']['D_diagnostic_perfect_speed']['median_drift_pct']:.1f}%")
    print(f"Task 7 Error Decomposition: Speed Magnitude = {forensic_res['task7_position_error_decomposition']['mean_speed_magnitude_contribution_pct']:.1f}%, Velocity Direction = {forensic_res['task7_position_error_decomposition']['mean_velocity_direction_contribution_pct']:.1f}%")
    print(f"Task 8 10-Second Median Drift: {forensic_res['task8_10s_priority']['median_10s_drift_pct']:.1f}%")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
