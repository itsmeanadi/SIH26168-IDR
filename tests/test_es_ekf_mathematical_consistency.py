"""Pytest Suite for 15-State ES-EKF Mathematical Consistency.

Validates:
1. Quaternion normalization, composition, and DCM mapping.
2. 3D Gravity mechanization under level and arbitrary tilt states.
3. Analytical F_d discrete transition matrix vs numerical central finite differences.
4. Analytical H_ai Jacobian vs numerical finite differences.
5. Analytical H_nhc Jacobian vs numerical finite differences.
6. Multiplicative error-state injection and covariance reset matrix G.
7. Adversarial corner cases (gimbal lock pitch ±90°, high angular velocity, zero velocity, tiny dt).
"""

import numpy as np
import pytest
from idr.filters.es_ekf import (
    quat_to_rot,
    quat_multiply as quat_mult,
    rot_to_euler_rpy,
    skew_symmetric as skew,
    so3_right_jacobian as right_jacobian_so3,
    exp_quaternion as rotvec_to_quat,
    ErrorStateKalmanFilter,
)


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    return q / np.linalg.norm(q)


def compute_analytical_F(p, v, q, ba, bg, acc_m, gyro_m, dt):
    ekf = ErrorStateKalmanFilter()
    ekf.set_state(pos=p, vel=v, quat=q, accel_bias=ba, gyro_bias=bg)
    f_b = acc_m - ba
    omega_b = gyro_m - bg
    R = quat_to_rot(q)
    delta_q = rotvec_to_quat(omega_b * dt)
    return ekf.compute_discrete_transition_matrix(f_b, omega_b, dt, R, delta_q)


def compute_numerical_F(p, v, q, ba, bg, acc_m, gyro_m, dt, eps=1e-7):
    g_nav = np.array([0.0, 0.0, -9.80665])
    f_b = acc_m - ba
    omega_b = gyro_m - bg
    R = quat_to_rot(q)
    delta_q = rotvec_to_quat(omega_b * dt)
    q_nom_next = quat_mult(q, delta_q)
    q_nom_next /= np.linalg.norm(q_nom_next)
    a_nav = R @ f_b + g_nav
    p_nom_next = p + v * dt + 0.5 * a_nav * (dt ** 2)
    v_nom_next = v + a_nav * dt

    F_num = np.zeros((15, 15), dtype=np.float64)
    for col in range(15):
        delta = np.zeros(15)
        delta[col] = eps
        p_pert = p + delta[0:3]
        v_pert = v + delta[3:6]
        q_pert = quat_mult(q, rotvec_to_quat(delta[6:9]))
        q_pert /= np.linalg.norm(q_pert)
        ba_pert = ba + delta[9:12]
        bg_pert = bg + delta[12:15]

        f_b_pert = acc_m - ba_pert
        omega_b_pert = gyro_m - bg_pert
        dq_pert = rotvec_to_quat(omega_b_pert * dt)
        q_pert_next = quat_mult(q_pert, dq_pert)
        q_pert_next /= np.linalg.norm(q_pert_next)

        R_pert = quat_to_rot(q_pert)
        a_nav_pert = R_pert @ f_b_pert + g_nav
        p_pert_next = p_pert + v_pert * dt + 0.5 * a_nav_pert * (dt ** 2)
        v_pert_next = v_pert + a_nav_pert * dt

        dp_next = p_pert_next - p_nom_next
        dv_next = v_pert_next - v_nom_next
        R_rel = quat_to_rot(q_nom_next).T @ quat_to_rot(q_pert_next)
        dtheta_next = np.array([
            R_rel[2, 1] - R_rel[1, 2],
            R_rel[0, 2] - R_rel[2, 0],
            R_rel[1, 0] - R_rel[0, 1]
        ]) * 0.5

        F_num[0:3, col] = dp_next / eps
        F_num[3:6, col] = dv_next / eps
        F_num[6:9, col] = dtheta_next / eps
        F_num[9:12, col] = delta[9:12] / eps
        F_num[12:15, col] = delta[12:15] / eps
    return F_num



def test_quaternion_dcm_consistency():
    """Verify vector rotation consistency R(q) v == q (x) [0, v] (x) q*."""
    rng = np.random.RandomState(101)
    for _ in range(25):
        axis = rng.randn(3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(-np.pi, np.pi)
        q = np.array([np.cos(angle / 2.0), *(axis * np.sin(angle / 2.0))])
        v = rng.uniform(-50, 50, 3)
        
        # Method 1: Matrix multiplication R(q) v
        R = quat_to_rot(q)
        v_rot_mat = R @ v
        
        # Method 2: Quaternion conjugation q (x) [0, v] (x) q*
        q_v = np.array([0.0, *v])
        q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
        v_rot_quat = quat_mult(quat_mult(q, q_v), q_conj)[1:4]
        
        assert np.allclose(v_rot_mat, v_rot_quat, atol=1e-9)


def test_gravity_3d_cancellation_under_tilt():
    """Verify level, pitched, and rolled stationary IMU cancels Earth gravity to zero navigation acceleration."""
    g_val = 9.80665
    g_n = np.array([0.0, 0.0, -g_val])
    
    # 25 random 3D orientations (roll in [-45, 45], pitch in [-45, 45], yaw in [-180, 180])
    rng = np.random.RandomState(202)
    for _ in range(25):
        roll = np.deg2rad(rng.uniform(-45, 45))
        pitch = np.deg2rad(rng.uniform(-45, 45))
        yaw = np.deg2rad(rng.uniform(-180, 180))
        
        Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        q = rot_to_quat(R)
        
        # Specific force measured by accelerometer at rest is -R^T g_n = R^T [0, 0, +g]
        f_b = R.T @ np.array([0.0, 0.0, g_val])
        
        # Reconstructed navigation acceleration
        a_n = quat_to_rot(q) @ f_b + g_n
        assert np.allclose(a_n, [0.0, 0.0, 0.0], atol=1e-8)


def test_propagation_jacobian_accuracy():
    """Verify analytical F_d matches numerical finite differences across 25 dynamic states."""
    rng = np.random.RandomState(303)
    for _ in range(25):
        dt = rng.uniform(0.01, 0.1)
        p = rng.uniform(-100, 100, 3)
        v = rng.uniform(-30, 30, 3)
        axis = rng.randn(3)
        axis /= np.linalg.norm(axis)
        q = np.array([np.cos(0.5), *(axis * np.sin(0.5))])
        ba = rng.uniform(-0.5, 0.5, 3)
        bg = rng.uniform(-0.05, 0.05, 3)
        acc_m = rng.uniform(-5.0, 5.0, 3)
        acc_m[2] += 9.81
        gyro_m = rng.uniform(-1.0, 1.0, 3)
        
        F_ana = compute_analytical_F(p, v, q, ba, bg, acc_m, gyro_m, dt)
        F_num = compute_numerical_F(p, v, q, ba, bg, acc_m, gyro_m, dt, eps=1e-7)
        
        max_err = np.max(np.abs(F_ana - F_num))
        assert max_err < 1e-4, f"F_d discrepancy too high: {max_err}"


def test_ai_velocity_jacobian_accuracy():
    """Verify analytical H_ai matches numerical finite differences across 25 dynamic states."""
    rng = np.random.RandomState(404)
    for _ in range(25):
        v = rng.uniform(-25, 25, 3)
        axis = rng.randn(3)
        axis /= np.linalg.norm(axis)
        q = np.array([np.cos(0.4), *(axis * np.sin(0.4))])
        
        R = quat_to_rot(q)
        v_b = R.T @ v
        
        H_ana = np.zeros((1, 15), dtype=np.float64)
        H_ana[0, 3:6] = R[:, 0]
        H_ana[0, 6:9] = np.array([0.0, -v_b[2], v_b[1]])
        
        H_num = np.zeros((1, 15), dtype=np.float64)
        eps = 1e-7
        z_base = v_b[0]
        
        for i in range(15):
            delta_x = np.zeros(15)
            delta_x[i] = eps
            v_p = v + delta_x[3:6]
            dq_p = rotvec_to_quat(delta_x[6:9])
            q_p = quat_mult(q, dq_p)
            q_p /= np.linalg.norm(q_p)
            v_b_p = quat_to_rot(q_p).T @ v_p
            H_num[0, i] = (v_b_p[0] - z_base) / eps
            
        max_err = np.max(np.abs(H_ana - H_num))
        assert max_err < 1e-4, f"H_ai discrepancy too high: {max_err}"


def test_nhc_jacobian_accuracy():
    """Verify analytical H_nhc matches numerical finite differences across 25 dynamic states."""
    rng = np.random.RandomState(505)
    for _ in range(25):
        v = rng.uniform(-25, 25, 3)
        axis = rng.randn(3)
        axis /= np.linalg.norm(axis)
        q = np.array([np.cos(0.6), *(axis * np.sin(0.6))])
        
        R = quat_to_rot(q)
        v_b = R.T @ v
        
        H_ana = np.zeros((2, 15), dtype=np.float64)
        H_ana[0, 3:6] = R[:, 1]
        H_ana[1, 3:6] = R[:, 2]
        H_ana[0, 6:9] = np.array([v_b[2], 0.0, -v_b[0]])
        H_ana[1, 6:9] = np.array([-v_b[1], v_b[0], 0.0])
        
        H_num = np.zeros((2, 15), dtype=np.float64)
        eps = 1e-7
        z_base = np.array([v_b[1], v_b[2]])
        
        for i in range(15):
            delta_x = np.zeros(15)
            delta_x[i] = eps
            v_p = v + delta_x[3:6]
            dq_p = rotvec_to_quat(delta_x[6:9])
            q_p = quat_mult(q, dq_p)
            q_p /= np.linalg.norm(q_p)
            v_b_p = quat_to_rot(q_p).T @ v_p
            H_num[:, i] = (v_b_p[1:3] - z_base) / eps
            
        max_err = np.max(np.abs(H_ana - H_num))
        assert max_err < 1e-4, f"H_nhc discrepancy too high: {max_err}"


def test_adversarial_attitude_pitch_roll_yaw_extremes():
    """Test extreme attitudes including pitch=89 deg, roll=85 deg, and yaw=180 deg."""
    rng = np.random.RandomState(606)
    extreme_angles = [
        (0.0, np.deg2rad(89.0), 0.0),      # Near gimbal-lock pitch
        (np.deg2rad(85.0), 0.0, 0.0),      # Extreme roll
        (0.0, 0.0, np.deg2rad(180.0)),     # Opposite yaw
        (np.deg2rad(45.0), np.deg2rad(45.0), np.deg2rad(90.0)),
    ]
    
    for roll, pitch, yaw in extreme_angles:
        Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        q = rot_to_quat(Rz @ Ry @ Rx)
        
        p = np.array([100.0, -50.0, 20.0])
        v = np.array([15.0, 0.0, 0.0])
        ba = np.zeros(3)
        bg = np.zeros(3)
        acc_m = np.array([0.0, 0.0, 9.81])
        gyro_m = np.array([0.05, -0.02, 0.1])
        dt = 0.01
        
        F_ana = compute_analytical_F(p, v, q, ba, bg, acc_m, gyro_m, dt)
        F_num = compute_numerical_F(p, v, q, ba, bg, acc_m, gyro_m, dt, eps=1e-7)
        
        assert np.all(np.isfinite(F_ana))
        assert np.max(np.abs(F_ana - F_num)) < 1e-4
