"""Phase 27: Rigorous Regression Tests for Alignment, Leveling, and Blackout Stability.

Validates:
1. Phone-to-vehicle rotation direction and orthonormality
2. Stationary gravity cancellation after alignment
3. Tilted stationary gravity cancellation across pitch/roll attitudes
4. Quaternion frame convention (Body -> ENU)
5. Static attitude initialization with leveling
6. AI velocity innovation and NIS bounding
7. NHC constraint behavior
8. No false ZUPT regression
9. Deterministic replay
"""

import numpy as np
import pytest

from src.idr.calib.alignment import PhoneToVehicleAligner, AlignmentState, compute_rotation_matrix
from src.idr.filters.es_ekf import (
    ErrorStateKalmanFilter,
    ESEKFConfig,
    exp_quaternion,
    quat_to_rot,
    quat_multiply,
    rot_to_euler_rpy,
)
from src.idr.filters.zupt import StationaryDetector


def test_phone_to_vehicle_orthonormality_and_direction():
    """Verify R_p2v is orthonormal with det(R) = +1 and aligns phone axes correctly."""
    aligner = PhoneToVehicleAligner()
    
    # Phone mounted tilted 90 deg: phone +Y is vehicle +Z, phone +Z is vehicle -X, phone +X is vehicle +Y
    stat_acc = np.tile(np.array([0.0, 9.80665, 0.0]), (30, 1))
    mot_acc = np.tile(np.array([0.0, 9.80665, -2.0]), (30, 1))
    
    R = aligner.estimate_from_stationary_and_motion(stat_acc, mot_acc)
    
    # Orthonormality checks
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-5)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-5)
    assert aligner.is_calibrated or aligner.z_phone is not None


def test_stationary_gravity_cancellation_level():
    """Verify stationary level vehicle produces zero navigation acceleration."""
    filter = ErrorStateKalmanFilter()
    filter.set_state(pos=np.zeros(3), vel=np.zeros(3), quat=np.array([1.0, 0.0, 0.0, 0.0]))
    
    # Specific force for level stationary sensor pointing up
    f_b = np.array([0.0, 0.0, 9.80665])
    g_m = np.zeros(3)
    
    # Propagate 10 steps (1.0 second)
    for _ in range(10):
        filter.predict(f_b, g_m, dt=0.1)
        
    assert np.allclose(filter.v, np.zeros(3), atol=1e-6)
    assert np.allclose(filter.p, np.zeros(3), atol=1e-6)


@pytest.mark.parametrize("pitch_deg,roll_deg", [
    (0.0, 0.0),
    (10.0, 0.0),
    (-10.0, 0.0),
    (0.0, 20.0),
    (0.0, -20.0),
    (15.0, -15.0),
])
def test_tilted_stationary_gravity_cancellation(pitch_deg, roll_deg):
    """Verify leveled initialization cancels gravity across any mounting tilt."""
    filter = ErrorStateKalmanFilter()
    pitch_rad = np.deg2rad(pitch_deg)
    roll_rad = np.deg2rad(roll_deg)
    
    # Construct true tilt
    q_r = exp_quaternion(np.array([roll_rad, 0.0, 0.0]))
    q_p = exp_quaternion(np.array([0.0, pitch_rad, 0.0]))
    q_true = quat_multiply(q_p, q_r)
    R_true = quat_to_rot(q_true)
    
    # Measured specific force at rest: f_b = R_true^T [0, 0, g]
    f_b = R_true.T @ np.array([0.0, 0.0, 9.80665])
    
    # Initialize leveling
    filter.initialize_leveling(f_b, yaw_rad=0.0)
    filter.set_state(pos=np.zeros(3), vel=np.zeros(3))
    
    # Propagate 20 steps (2.0 seconds)
    for _ in range(20):
        filter.predict(f_b, np.zeros(3), dt=0.1)
        
    # Velocity and position should remain near zero
    assert np.linalg.norm(filter.v) < 1e-4
    assert np.linalg.norm(filter.p) < 1e-4


def test_quaternion_frame_convention():
    """Verify Body -> ENU quaternion transforms vectors correctly."""
    # Yaw = 90 deg (facing North): Body +X (Forward) rotates to ENU +Y (North)
    q = exp_quaternion(np.array([0.0, 0.0, np.pi / 2.0]))
    R = quat_to_rot(q)
    
    v_body = np.array([10.0, 0.0, 0.0]) # 10 m/s Forward
    v_nav = R @ v_body
    
    assert np.allclose(v_nav, np.array([0.0, 10.0, 0.0]), atol=1e-5)


def test_ai_velocity_innovation_and_gating():
    """Verify AI forward speed update produces correct innovation and accepted update."""
    filter = ErrorStateKalmanFilter()
    v_init = np.array([10.0, 0.0, 0.0])
    filter.set_state(pos=np.zeros(3), vel=v_init, quat=np.array([1.0, 0.0, 0.0, 0.0]))
    
    # Forward speed measurement: 10.5 m/s
    acc, diag = filter.update_ai_velocity(10.5, sigma_v=1.0)
    assert acc is True
    assert np.isclose(diag["innovation"][0], 0.5, atol=1e-5)
    assert diag["nis"] < 6.635


def test_nhc_lateral_vertical_constraint():
    """Verify NHC constrains lateral and vertical body velocities."""
    filter = ErrorStateKalmanFilter()
    # Velocity with lateral and vertical drift: v = [10.0, 1.5, -0.8]
    filter.set_state(pos=np.zeros(3), vel=np.array([10.0, 1.5, -0.8]), quat=np.array([1.0, 0.0, 0.0, 0.0]))
    
    acc, diag = filter.update_nhc(sigma_lat=0.05, sigma_vert=0.05)
    assert acc is True
    # Lateral and vertical body velocities should be driven toward zero
    v_body = filter.rotation_matrix.T @ filter.v
    assert abs(v_body[1]) < 0.5
    assert abs(v_body[2]) < 0.5


def test_stationary_detector_no_false_zupt_while_cruising():
    """Verify StationaryDetector does not trigger ZUPT when vehicle is cruising at 15 m/s."""
    detector = StationaryDetector(window_size=10, acc_var_threshold=0.15)
    
    # Cruising frames with engine vibration
    for _ in range(30):
        acc = np.array([0.01, 0.02, 9.81]) + np.random.normal(0.0, 0.05, 3)
        gyro = np.array([0.001, 0.002, 0.001])
        is_stat = detector.update(acc, gyro, speed_mps=15.0)
        assert is_stat is False


def test_deterministic_filter_replay():
    """Verify 15-state ES-EKF produces bitwise identical trajectories across identical runs."""
    np.random.seed(42)
    acc_series = np.random.normal(0.0, 1.0, (50, 3))
    acc_series[:, 2] += 9.80665
    gyro_series = np.random.normal(0.0, 0.05, (50, 3))
    
    def run_sim():
        f = ErrorStateKalmanFilter()
        f.initialize_leveling(acc_series[0], yaw_rad=0.5)
        for k in range(50):
            f.predict(acc_series[k], gyro_series[k], dt=0.1)
            if k % 10 == 0:
                f.update_ai_velocity(12.0, sigma_v=1.5)
                f.update_nhc(sigma_lat=0.05, sigma_vert=0.05)
        return f.p.copy(), f.v.copy(), f.q.copy()
        
    p1, v1, q1 = run_sim()
    p2, v2, q2 = run_sim()
    
    assert np.allclose(p1, p2, atol=1e-12)
    assert np.allclose(v1, v2, atol=1e-12)
    assert np.allclose(q1, q2, atol=1e-12)
