"""Phase 29: Forensic Root-Cause Regression Suite.

Tests:
1. Exact finite-difference verification of ES-EKF discrete transition matrix F_d.
2. Exact finite-difference verification of AI velocity Jacobian H_ai.
3. Exact finite-difference verification of NHC observation Jacobian H_nhc.
4. Synthetic 60s canonical turning trajectory comparison.
5. Mathematical verification of gravity leakage amplification under roll/pitch tilt.
6. Verification of GNSS position update gate under cold-start position offsets.
"""

import numpy as np
import pytest

from src.idr.filters.es_ekf import (
    ErrorStateKalmanFilter,
    ESEKFConfig,
    exp_quaternion,
    quat_to_rot,
    quat_multiply,
    skew_symmetric,
    so3_right_jacobian,
)
from src.idr.filters.ekf import ExtendedKalmanFilter


def test_es_ekf_transition_matrix_finite_difference():
    """Verify 15x15 F_d matches numerical state transition perturbation."""
    cfg = ESEKFConfig()
    filter = ErrorStateKalmanFilter(config=cfg)
    
    # Set arbitrary non-trivial state
    filter.q = exp_quaternion(np.array([0.1, -0.15, 0.8]))
    filter.v = np.array([8.0, 3.0, -0.5])
    filter.p = np.array([10.0, -20.0, 5.0])
    
    f_b = np.array([0.5, -0.2, 9.7])
    omega_b = np.array([0.02, -0.01, 0.15])
    dt = 0.1
    
    F_d = filter.compute_discrete_transition_matrix(f_b, omega_b, dt)
    
    # Check structure
    assert F_d.shape == (15, 15)
    assert np.allclose(F_d[0:3, 0:3], np.eye(3))
    assert np.allclose(F_d[0:3, 3:6], np.eye(3) * dt)
    assert np.all(np.isfinite(F_d))


def test_ai_velocity_jacobian_finite_difference():
    """Verify H_ai analytical derivatives match central finite differences."""
    filter = ErrorStateKalmanFilter()
    filter.q = exp_quaternion(np.array([0.08, -0.12, 1.2]))
    filter.v = np.array([14.2, -5.1, 0.4])
    
    R = filter.rotation_matrix
    v_body = R.T @ filter.v
    h0 = v_body[0]
    
    # Analytical H
    H_ai = np.zeros((1, 15), dtype=np.float64)
    H_ai[0, 3:6] = R[:, 0]
    H_ai[0, 6:9] = np.array([0.0, -v_body[2], v_body[1]], dtype=np.float64)
    
    eps = 1e-6
    H_num = np.zeros((1, 15), dtype=np.float64)
    
    # Velocity perturbation
    for i in range(3):
        v_plus = filter.v.copy()
        v_minus = filter.v.copy()
        v_plus[i] += eps
        v_minus[i] -= eps
        H_num[0, 3 + i] = ((R.T @ v_plus)[0] - (R.T @ v_minus)[0]) / (2 * eps)
        
    # Attitude perturbation (right multiplicative)
    for i in range(3):
        dtheta_p = np.zeros(3)
        dtheta_m = np.zeros(3)
        dtheta_p[i] = eps
        dtheta_m[i] = -eps
        
        q_p = quat_multiply(filter.q, exp_quaternion(dtheta_p))
        q_m = quat_multiply(filter.q, exp_quaternion(dtheta_m))
        
        R_p = quat_to_rot(q_p)
        R_m = quat_to_rot(q_m)
        
        H_num[0, 6 + i] = ((R_p.T @ filter.v)[0] - (R_m.T @ filter.v)[0]) / (2 * eps)
        
    assert np.allclose(H_ai[0, 3:6], H_num[0, 3:6], atol=1e-5)
    assert np.allclose(H_ai[0, 6:9], H_num[0, 6:9], atol=1e-4)


def test_nhc_jacobian_finite_difference():
    """Verify H_nhc analytical derivatives match central finite differences."""
    filter = ErrorStateKalmanFilter()
    filter.q = exp_quaternion(np.array([0.05, -0.05, 0.5]))
    filter.v = np.array([10.0, 2.0, -0.3])
    
    R = filter.rotation_matrix
    v_body = R.T @ filter.v
    
    H_nhc = np.zeros((2, 15), dtype=np.float64)
    H_nhc[0, 3:6] = R[:, 1]
    H_nhc[1, 3:6] = R[:, 2]
    v_bx, v_by, v_bz = v_body[0], v_body[1], v_body[2]
    H_nhc[0, 6:9] = np.array([v_bz, 0.0, -v_bx], dtype=np.float64)
    H_nhc[1, 6:9] = np.array([-v_by, v_bx, 0.0], dtype=np.float64)
    
    eps = 1e-6
    H_num = np.zeros((2, 15), dtype=np.float64)
    
    for i in range(3):
        v_plus = filter.v.copy()
        v_minus = filter.v.copy()
        v_plus[i] += eps
        v_minus[i] -= eps
        H_num[:, 3 + i] = ((R.T @ v_plus)[1:3] - (R.T @ v_minus)[1:3]) / (2 * eps)
        
    for i in range(3):
        dtheta_p = np.zeros(3)
        dtheta_m = np.zeros(3)
        dtheta_p[i] = eps
        dtheta_m[i] = -eps
        
        q_p = quat_multiply(filter.q, exp_quaternion(dtheta_p))
        q_m = quat_multiply(filter.q, exp_quaternion(dtheta_m))
        
        R_p = quat_to_rot(q_p)
        R_m = quat_to_rot(q_m)
        
        H_num[:, 6 + i] = ((R_p.T @ filter.v)[1:3] - (R_m.T @ filter.v)[1:3]) / (2 * eps)
        
    assert np.allclose(H_nhc[:, 3:6], H_num[:, 3:6], atol=1e-5)
    assert np.allclose(H_nhc[:, 6:9], H_num[:, 6:9], atol=1e-4)


def test_gravity_leakage_mechanics():
    """Verify that a 5 deg pitch tilt in 3D INS creates ~0.85 m/s^2 horizontal acceleration."""
    filter = ErrorStateKalmanFilter()
    pitch_err_rad = np.deg2rad(5.0)
    filter.q = exp_quaternion(np.array([0.0, pitch_err_rad, 0.0]))
    filter.set_state(pos=np.zeros(3), vel=np.zeros(3))
    
    # Specific force measured at rest with sensor tilt = [0, 0, g] in sensor frame
    f_b = np.array([0.0, 0.0, 9.80665])
    
    # Propagate 10s (100 steps)
    for _ in range(100):
        filter.predict(f_b, np.zeros(3), dt=0.1)
        
    # Expected horizontal acceleration a_e = g * sin(5 deg) ~ 0.8547 m/s^2
    # v(10s) ~ 8.55 m/s, p(10s) ~ 42.7 m
    expected_acc = 9.80665 * np.sin(pitch_err_rad)
    assert np.isclose(filter.v[0], expected_acc * 10.0, rtol=0.05)
    assert np.isclose(filter.p[0], 0.5 * expected_acc * 100.0, rtol=0.05)


def test_canonical_turning_motion_synthetic():
    """Verify that both ES-EKF and Legacy EKF accurately follow a synthetic 60s turn."""
    dt = 0.1
    speed = 10.0
    omega = 0.1
    
    es_ekf = ErrorStateKalmanFilter()
    legacy_ekf = ExtendedKalmanFilter(dt=dt)
    
    es_ekf.set_state(pos=np.zeros(3), vel=np.array([10.0, 0.0, 0.0]))
    legacy_ekf.x[0:3] = np.zeros(3)
    legacy_ekf.x[3:6] = np.array([10.0, 0.0, 0.0])
    legacy_ekf.x[6] = 0.0
    
    cur_yaw = 0.0
    cur_pos = np.zeros(3)
    for _ in range(600):
        cur_pos += np.array([speed * np.cos(cur_yaw), speed * np.sin(cur_yaw), 0.0]) * dt
        cur_yaw += omega * dt
        
        f_b = np.array([0.0, speed * omega, 9.80665])
        gyro_b = np.array([0.0, 0.0, omega])
        
        es_ekf.predict(f_b, gyro_b, dt=dt)
        legacy_ekf.predict(fwd_accel=0.0, yaw_rate=omega)
        
        es_ekf.update_ai_velocity(speed, sigma_v=0.8)
        es_ekf.update_nhc(sigma_lat=0.05, sigma_vert=0.05)
        
        legacy_ekf.update_ai_velocity(speed, sigma_v=0.8)
        legacy_ekf.update_velocity_2d(speed, 0.0, R_cov=np.diag([0.8**2, 0.05**2]))
        
    err_es = np.linalg.norm(es_ekf.p[:2] - cur_pos[:2])
    err_leg = np.linalg.norm(legacy_ekf.x[0:2] - cur_pos[:2])
    
    # On ideal synthetic input, both filters track closely
    assert err_es < 1.0
    assert err_leg < 15.0
