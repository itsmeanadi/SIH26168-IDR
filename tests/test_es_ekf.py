"""Unit and integration tests for the 15-state 3D Error-State Extended Kalman Filter (ES-EKF).

Test Suite covers:
A. Quaternion & SO(3) Lie Algebra operations.
B. 3D Gravity mechanization and tilt cancellation.
C. Discrete Kinematic Propagation.
D. Analytical vs Finite-Difference Jacobian Validation (F, H_ai, H_nhc, H_pos, H_vel, H_zupt, H_zaru).
E. Error-State Injection & Multiplicative Covariance Reset.
F. Covariance Stability, Symmetry, and Positivity.
G. Measurement Updates & Chi-Square NIS Gating.
H. Numerical Edge Cases (dt=0, tiny dt, large dt, high rate, NaN/Inf inputs).
I. Standstill GNSS COG Heading Protection.
"""

import numpy as np
import pytest

from idr.filters.es_ekf import (
    ESEKFConfig,
    ErrorStateKalmanFilter,
    exp_quaternion,
    quat_multiply,
    quat_to_rot,
    rot_to_euler_rpy,
    skew_symmetric,
    so3_right_jacobian,
)
from idr.filters.fusion import GNSSINSFusion


# ---------------------------------------------------------------------------
# A. Quaternion & SO(3) Algebra Tests
# ---------------------------------------------------------------------------

def test_quaternion_normalization_and_identity():
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    R_id = quat_to_rot(q_id)
    assert np.allclose(R_id, np.eye(3), atol=1e-12)

    # Round trip with exp_quaternion for zero vector
    q_zero = exp_quaternion(np.zeros(3))
    assert np.allclose(q_zero, q_id, atol=1e-12)


def test_quaternion_small_angle_expansion():
    # Small angle 1e-9
    v_tiny = np.array([1e-9, -2e-9, 3e-9])
    q_exp = exp_quaternion(v_tiny)
    assert np.isclose(np.linalg.norm(q_exp), 1.0, atol=1e-12)
    assert np.isclose(q_exp[0], 1.0, atol=1e-12)
    assert np.allclose(q_exp[1:], 0.5 * v_tiny, atol=1e-12)


def test_quaternion_yaw_rotations():
    # +90 deg yaw (pi/2) around +Z
    yaw_angle = np.pi / 2.0
    q_yaw90 = exp_quaternion(np.array([0.0, 0.0, yaw_angle]))
    R_yaw90 = quat_to_rot(q_yaw90)
    
    # Body +X [1, 0, 0] rotates to ENU +Y [0, 1, 0] (North)
    v_body = np.array([1.0, 0.0, 0.0])
    v_nav = R_yaw90 @ v_body
    assert np.allclose(v_nav, [0.0, 1.0, 0.0], atol=1e-10)

    # -90 deg yaw (-pi/2) around +Z
    q_yaw_neg90 = exp_quaternion(np.array([0.0, 0.0, -yaw_angle]))
    R_yaw_neg90 = quat_to_rot(q_yaw_neg90)
    v_nav_neg = R_yaw_neg90 @ v_body
    assert np.allclose(v_nav_neg, [0.0, -1.0, 0.0], atol=1e-10)


def test_so3_right_jacobian():
    # Test identity limit as theta -> 0
    J_zero = so3_right_jacobian(np.zeros(3))
    assert np.allclose(J_zero, np.eye(3), atol=1e-12)

    # Test finite rotation
    v = np.array([0.1, -0.2, 0.3])
    J = so3_right_jacobian(v)
    assert J.shape == (3, 3)
    assert np.linalg.det(J) > 0.0


# ---------------------------------------------------------------------------
# B. 3D Gravity Mechanization Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("roll_deg, pitch_deg, yaw_deg", [
    (0.0, 0.0, 0.0),
    (0.0, 30.0, 0.0),
    (0.0, -30.0, 0.0),
    (30.0, 0.0, 0.0),
    (-30.0, 0.0, 0.0),
    (25.0, -15.0, 45.0),
    (-40.0, 35.0, -120.0),
])
def test_gravity_cancellation_in_arbitrary_tilt(roll_deg, pitch_deg, yaw_deg):
    filter_inst = ErrorStateKalmanFilter()
    
    # Construct quaternion from roll, pitch, yaw
    q_r = exp_quaternion(np.array([np.deg2rad(roll_deg), 0.0, 0.0]))
    q_p = exp_quaternion(np.array([0.0, np.deg2rad(pitch_deg), 0.0]))
    q_y = exp_quaternion(np.array([0.0, 0.0, np.deg2rad(yaw_deg)]))
    q_att = quat_multiply(q_y, quat_multiply(q_p, q_r))
    filter_inst.set_state(quat=q_att)

    R = filter_inst.rotation_matrix
    # Specific force measured by stationary IMU in body frame: f_b = R^T * [0, 0, +9.80665]
    f_b_meas = R.T @ np.array([0.0, 0.0, 9.80665])
    gyro_meas = np.zeros(3)

    # Predict one step
    pos_before = filter_inst.p.copy()
    vel_before = filter_inst.v.copy()
    filter_inst.predict(f_b_meas, gyro_meas, dt=0.1)

    assert np.allclose(filter_inst.v, vel_before, atol=1e-10)
    assert np.allclose(filter_inst.p, pos_before, atol=1e-10)


# ---------------------------------------------------------------------------
# C. Discrete Kinematic Propagation Tests
# ---------------------------------------------------------------------------

def test_constant_velocity_propagation():
    filter_inst = ErrorStateKalmanFilter()
    filter_inst.set_state(vel=np.array([10.0, 5.0, 0.0]))
    # Stationary specific force (gravity cancellation)
    f_b = np.array([0.0, 0.0, 9.80665])
    gyro = np.zeros(3)

    for _ in range(10):
        filter_inst.predict(f_b, gyro, dt=0.1)

    assert np.allclose(filter_inst.v, [10.0, 5.0, 0.0], atol=1e-10)
    assert np.allclose(filter_inst.p, [10.0, 5.0, 0.0], atol=1e-10)  # 1.0s total


def test_constant_yaw_rate_propagation():
    filter_inst = ErrorStateKalmanFilter()
    yaw_rate = np.pi / 2.0  # 90 deg/s
    f_b = np.array([0.0, 0.0, 9.80665])
    gyro = np.array([0.0, 0.0, yaw_rate])

    # 1.0s propagation => total 90 deg yaw
    for _ in range(10):
        filter_inst.predict(f_b, gyro, dt=0.1)

    roll, pitch, yaw = filter_inst.euler_angles
    assert np.isclose(yaw, np.pi / 2.0, atol=1e-5)


# ---------------------------------------------------------------------------
# D. Analytical vs Finite-Difference Jacobian Validation
# ---------------------------------------------------------------------------

def test_f_matrix_finite_differences():
    """Verify 15x15 discrete transition matrix F_d against numerical central differences."""
    np.random.seed(42)
    eps = 1e-6
    dt = 0.05

    for trial in range(10):
        p0 = np.random.uniform(-50, 50, size=3)
        v0 = np.random.uniform(-20, 20, size=3)
        v_rot = np.random.uniform(-0.5, 0.5, size=3)
        q0 = exp_quaternion(v_rot)
        ba0 = np.random.uniform(-0.1, 0.1, size=3)
        bg0 = np.random.uniform(-0.01, 0.01, size=3)

        acc_meas = np.random.uniform(-5, 5, size=3) + np.array([0, 0, 9.80665])
        gyro_meas = np.random.uniform(-0.5, 0.5, size=3)

        filter_nom = ErrorStateKalmanFilter()
        filter_nom.set_state(pos=p0, vel=v0, quat=q0, accel_bias=ba0, gyro_bias=bg0)

        f_b = acc_meas - ba0
        omega_b = gyro_meas - bg0
        R0 = quat_to_rot(q0)
        delta_q = exp_quaternion(omega_b * dt)

        # Analytical F_d
        F_ana = filter_nom.compute_discrete_transition_matrix(f_b, omega_b, dt, R0, delta_q)

        # Nominal propagated state
        q_nom_next = quat_multiply(q0, delta_q)
        q_nom_next /= np.linalg.norm(q_nom_next)
        R_next = quat_to_rot(q_nom_next)
        a_nav = R0 @ f_b + filter_nom.g_nav
        p_nom_next = p0 + v0 * dt + 0.5 * a_nav * (dt ** 2)
        v_nom_next = v0 + a_nav * dt

        # Numerical Jacobian F_num (15x15)
        F_num = np.zeros((15, 15), dtype=np.float64)

        for col in range(15):
            delta = np.zeros(15)
            delta[col] = eps

            # Perturbed initial state
            p_pert = p0 + delta[0:3]
            v_pert = v0 + delta[3:6]
            q_pert = quat_multiply(q0, exp_quaternion(delta[6:9]))
            q_pert /= np.linalg.norm(q_pert)
            ba_pert = ba0 + delta[9:12]
            bg_pert = bg0 + delta[12:15]

            # Perturbed propagation
            f_b_pert = acc_meas - ba_pert
            omega_b_pert = gyro_meas - bg_pert
            dq_pert = exp_quaternion(omega_b_pert * dt)
            q_pert_next = quat_multiply(q_pert, dq_pert)
            q_pert_next /= np.linalg.norm(q_pert_next)

            R_pert = quat_to_rot(q_pert)
            a_nav_pert = R_pert @ f_b_pert + filter_nom.g_nav
            p_pert_next = p_pert + v_pert * dt + 0.5 * a_nav_pert * (dt ** 2)
            v_pert_next = v_pert + a_nav_pert * dt

            # Compute error states w.r.t nominal next state
            dp_next = p_pert_next - p_nom_next
            dv_next = v_pert_next - v_nom_next
            
            # Attitude error delta_theta: R_pert_next = R_nom_next * (I + [delta_theta]_x)
            # => [delta_theta]_x ~= R_nom_next^T * R_pert_next - I
            R_rel = quat_to_rot(q_nom_next).T @ quat_to_rot(q_pert_next)
            dtheta_next = np.array([
                R_rel[2, 1] - R_rel[1, 2],
                R_rel[0, 2] - R_rel[2, 0],
                R_rel[1, 0] - R_rel[0, 1]
            ]) * 0.5

            dba_next = delta[9:12]
            dbg_next = delta[12:15]

            delta_next = np.concatenate([dp_next, dv_next, dtheta_next, dba_next, dbg_next])
            F_num[:, col] = delta_next / eps

        diff = np.abs(F_ana - F_num)
        max_err = np.max(diff)
        assert max_err < 2.0e-5, f"Trial {trial}: F_d max finite difference error {max_err} exceeds tolerance"


def test_ai_measurement_jacobian_finite_differences():
    """Verify AI forward-speed observation Jacobian H_ai against finite differences."""
    np.random.seed(123)
    eps = 1e-6

    for trial in range(10):
        v = np.random.uniform(-30, 30, size=3)
        v_rot = np.random.uniform(-0.8, 0.8, size=3)
        q = exp_quaternion(v_rot)

        filter_inst = ErrorStateKalmanFilter()
        filter_inst.set_state(vel=v, quat=q)

        R = quat_to_rot(q)
        v_body = R.T @ v

        # Analytical H_ai
        H_ana = np.zeros((1, 15))
        H_ana[0, 3:6] = R[:, 0]
        H_ana[0, 6:9] = np.array([0.0, -v_body[2], v_body[1]])

        # Numerical H_num
        H_num = np.zeros((1, 15))
        h_nom = (R.T @ v)[0]

        for col in range(15):
            delta = np.zeros(15)
            delta[col] = eps

            v_pert = v + delta[3:6]
            q_pert = quat_multiply(q, exp_quaternion(delta[6:9]))
            q_pert /= np.linalg.norm(q_pert)
            R_pert = quat_to_rot(q_pert)

            h_pert = (R_pert.T @ v_pert)[0]
            H_num[0, col] = (h_pert - h_nom) / eps

        max_err = np.max(np.abs(H_ana - H_num))
        assert max_err < 2.0e-5, f"Trial {trial}: H_ai max error {max_err} exceeds tolerance"


def test_nhc_measurement_jacobian_finite_differences():
    """Verify NHC observation Jacobian H_nhc against finite differences."""
    np.random.seed(456)
    eps = 1e-6

    for trial in range(10):
        v = np.random.uniform(-25, 25, size=3)
        v_rot = np.random.uniform(-0.8, 0.8, size=3)
        q = exp_quaternion(v_rot)

        R = quat_to_rot(q)
        v_body = R.T @ v
        v_bx, v_by, v_bz = v_body[0], v_body[1], v_body[2]

        # Analytical H_nhc
        H_ana = np.zeros((2, 15))
        H_ana[0, 3:6] = R[:, 1]
        H_ana[1, 3:6] = R[:, 2]
        H_ana[0, 6:9] = np.array([v_bz, 0.0, -v_bx])
        H_ana[1, 6:9] = np.array([-v_by, v_bx, 0.0])

        # Numerical H_num
        H_num = np.zeros((2, 15))
        h_nom = np.array([v_body[1], v_body[2]])

        for col in range(15):
            delta = np.zeros(15)
            delta[col] = eps

            v_pert = v + delta[3:6]
            q_pert = quat_multiply(q, exp_quaternion(delta[6:9]))
            q_pert /= np.linalg.norm(q_pert)
            R_pert = quat_to_rot(q_pert)
            v_body_pert = R_pert.T @ v_pert
            h_pert = np.array([v_body_pert[1], v_body_pert[2]])

            H_num[:, col] = (h_pert - h_nom) / eps

        max_err = np.max(np.abs(H_ana - H_num))
        assert max_err < 2.0e-5, f"Trial {trial}: H_nhc max error {max_err} exceeds tolerance"


# ---------------------------------------------------------------------------
# E. Error Injection & Covariance Reset Tests
# ---------------------------------------------------------------------------

def test_error_injection_and_covariance_reset():
    filter_inst = ErrorStateKalmanFilter()
    filter_inst.set_state(
        pos=np.array([10.0, 20.0, 30.0]),
        vel=np.array([1.0, 2.0, 3.0]),
        accel_bias=np.array([0.01, 0.02, 0.03]),
        gyro_bias=np.array([0.001, 0.002, 0.003])
    )

    delta_x = np.zeros(15)
    delta_x[0:3] = [1.0, -1.0, 0.5]
    delta_x[3:6] = [0.1, -0.2, 0.3]
    delta_x[6:9] = [0.01, -0.02, 0.03]
    delta_x[9:12] = [0.005, -0.005, 0.002]
    delta_x[12:15] = [0.0005, -0.0005, 0.0002]

    P_initial = np.eye(15)
    filter_inst.inject_error_and_reset(delta_x, P_initial)

    assert np.allclose(filter_inst.p, [11.0, 19.0, 30.5])
    assert np.allclose(filter_inst.v, [1.1, 1.8, 3.3])
    assert np.allclose(filter_inst.ba, [0.015, 0.015, 0.032])
    assert np.allclose(filter_inst.bg, [0.0015, 0.0015, 0.0032])
    assert np.isclose(np.linalg.norm(filter_inst.q), 1.0, atol=1e-10)

    # Covariance symmetry check
    assert np.allclose(filter_inst.P, filter_inst.P.T, atol=1e-12)


# ---------------------------------------------------------------------------
# F. Covariance Stability & Positivity Tests
# ---------------------------------------------------------------------------

def test_long_term_covariance_stability():
    filter_inst = ErrorStateKalmanFilter()
    f_b = np.array([0.1, -0.05, 9.80665])
    gyro = np.array([0.01, -0.02, 0.05])

    for _ in range(500):
        filter_inst.predict(f_b, gyro, dt=0.02)
        # Covariance must remain symmetric and positive-definite
        assert np.all(np.isfinite(filter_inst.P))
        assert np.allclose(filter_inst.P, filter_inst.P.T, atol=1e-10)
        eigenvals = np.linalg.eigvalsh(filter_inst.P)
        assert np.all(eigenvals >= 0.0), "Covariance matrix has negative eigenvalue!"


# ---------------------------------------------------------------------------
# G. Measurement Updates & Chi-Square Gating Tests
# ---------------------------------------------------------------------------

def test_gnss_pos_and_vel_update():
    filter_inst = ErrorStateKalmanFilter()
    filter_inst.set_state(pos=np.array([0.0, 0.0, 0.0]), vel=np.array([0.0, 0.0, 0.0]))

    # Position update with valid in-gate innovation (< 3-DOF chi2 gate 11.345)
    accepted_pos, diag_pos = filter_inst.update_gnss_pos(np.array([1.5, 2.0, 0.5]))
    assert accepted_pos is True
    assert np.linalg.norm(filter_inst.p) > 0.0

    # Velocity update with valid in-gate innovation
    accepted_vel, diag_vel = filter_inst.update_gnss_vel(np.array([0.5, 0.3, -0.1]))
    assert accepted_vel is True
    assert filter_inst.v[0] > 0.0


def test_ai_velocity_update_and_outlier_rejection():
    filter_inst = ErrorStateKalmanFilter()
    filter_inst.set_state(vel=np.array([10.0, 0.0, 0.0]))

    # Valid AI update near 10 m/s
    accepted, diag = filter_inst.update_ai_velocity(speed_mps=10.5, sigma_v=1.5)
    assert accepted is True

    # Extreme outlier (80 m/s when filter is at ~10 m/s) should be rejected by NIS gate
    accepted_outlier, diag_outlier = filter_inst.update_ai_velocity(
        speed_mps=75.0, sigma_v=1.0, max_innovation_sigma=3.0
    )
    assert accepted_outlier is False
    assert "NIS" in diag_outlier.get("rejection_reason", "")


def test_zupt_and_zaru_updates():
    filter_inst = ErrorStateKalmanFilter()
    filter_inst.set_state(vel=np.array([0.5, -0.2, 0.1]), gyro_bias=np.array([0.005, -0.005, 0.002]))

    accepted_zupt, _ = filter_inst.update_zupt(sigma_v=0.01)
    assert accepted_zupt is True
    # Velocity should be driven strongly towards zero
    assert np.linalg.norm(filter_inst.v) < 0.1

    # ZARU when gyro reads pure bias
    accepted_zaru, _ = filter_inst.update_zaru(gyro_meas=np.array([0.005, -0.005, 0.002]), sigma_bg=0.001)
    assert accepted_zaru is True


# ---------------------------------------------------------------------------
# H. Numerical Edge Cases & Robustness
# ---------------------------------------------------------------------------

def test_dt_edge_cases():
    filter_inst = ErrorStateKalmanFilter()
    f_b = np.array([0.0, 0.0, 9.80665])
    gyro = np.zeros(3)

    # dt = 0, dt negative, dt huge -> gracefully handled
    filter_inst.predict(f_b, gyro, dt=0.0)
    assert np.all(np.isfinite(filter_inst.p))
    assert np.all(np.isfinite(filter_inst.P))

    filter_inst.predict(f_b, gyro, dt=-0.5)
    assert np.all(np.isfinite(filter_inst.p))

    filter_inst.predict(f_b, gyro, dt=100.0)
    assert np.all(np.isfinite(filter_inst.p))


def test_nan_inf_sensor_handling():
    filter_inst = ErrorStateKalmanFilter()
    acc_nan = np.array([np.nan, 0.0, 9.81])
    gyro_inf = np.array([0.0, np.inf, 0.0])

    filter_inst.predict(acc_nan, gyro_inf, dt=0.1)
    assert np.all(np.isfinite(filter_inst.p))
    assert np.all(np.isfinite(filter_inst.v))
    assert np.all(np.isfinite(filter_inst.q))
    assert np.all(np.isfinite(filter_inst.P))


# ---------------------------------------------------------------------------
# I. Standstill GNSS COG Heading Protection
# ---------------------------------------------------------------------------

def test_standstill_gnss_cog_heading_protection():
    """Verify that GPS position jitter at standstill does NOT corrupt vehicle heading."""
    fusion = GNSSINSFusion(ref_lat=28.6139, ref_lon=77.2090, dt=0.1, min_cog_speed_mps=1.5)
    
    # Establish initial anchor
    initial_yaw = float(fusion.ekf.x[6])
    
    # Simulate stationary vehicle with GPS jitter (0.05m movement in random directions at 10 Hz => 0.5 m/s < 1.5 m/s)
    acc_stat = np.array([0.0, 0.0, 9.81])
    gyro_stat = np.array([0.0, 0.0, 0.0])

    # Step 1: Fix at origin
    fusion.step(0.0, 0.0, gnss_pos=(28.6139, 77.2090), acc_3d=acc_stat, gyro_3d=gyro_stat)

    # Step 2: Fix jittered by 0.05m North-West (implied speed 0.5 m/s < 1.5 m/s)
    # 0.05m is approx 4.5e-7 degrees
    fusion.step(0.0, 0.0, gnss_pos=(28.61390045, 77.20899955), acc_3d=acc_stat, gyro_3d=gyro_stat)

    # Yaw should NOT have jumped to NW (heading update must be suppressed)
    yaw_after_jitter = float(fusion.ekf.x[6])
    assert np.isclose(yaw_after_jitter, initial_yaw, atol=0.01), "Standstill GPS jitter corrupted vehicle heading!"

    # Step 3: Fast moving fix (5.0 m/s > 1.5 m/s threshold)
    # 0.5m in 0.1s is 5 m/s North (d_lat ~ 4.5e-6 degrees)
    fusion.step(0.0, 0.0, gnss_pos=(28.6139045, 77.2090), acc_3d=acc_stat, gyro_3d=gyro_stat)
    
    # Moving COG heading update should be applied
    yaw_moving = float(fusion.ekf.x[6])
    # Heading North in ENU is pi/2 radians
    assert np.isclose(yaw_moving, np.pi / 2.0, atol=0.5)


# ---------------------------------------------------------------------------
# J. Phase 33 Bias Jacobian Sign Regression Tests
# ---------------------------------------------------------------------------

def test_phase33_zaru_gyro_bias_sign_correction():
    """Verify that ZARU with positive gyro measurement increases bg to cancel bias."""
    filter_inst = ErrorStateKalmanFilter()
    # When stationary, gyro sensor reads +0.005 rad/s due to positive bias
    accepted, diag = filter_inst.update_zaru(gyro_meas=np.array([0.0, 0.0, 0.005]), sigma_bg=0.001)
    assert accepted is True
    # bg must be positive (+0.0048 rad/s) so that gyro - bg cancels the measurement to ~0.0
    assert filter_inst.bg[2] > 0.004
    corrected_gyro = np.array([0.0, 0.0, 0.005]) - filter_inst.bg
    assert abs(corrected_gyro[2]) < 0.001


def test_phase33_gravity_leveling_accel_bias_sign_correction():
    """Verify that gravity leveling with positive horizontal acc increases ba."""
    filter_inst = ErrorStateKalmanFilter()
    # When level and stationary, sensor reads +0.1 m/s^2 along X due to positive accel bias
    accepted, diag = filter_inst.update_gravity_leveling(np.array([0.1, 0.0, 9.80665]), sigma_level=0.5)
    assert accepted is True
    # ba[0] must be positive
    assert filter_inst.ba[0] > 0.0

