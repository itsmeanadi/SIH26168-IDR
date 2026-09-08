"""Unit and safety tests for Phase 31A Magnetometer Heading & Disturbance Rejection."""

import pytest
import numpy as np
from pathlib import Path

from src.idr.calib.mag_calib import MagnetometerCalibrator
from src.idr.sensors.mag_heading import compute_tilt_compensated_heading
from src.idr.sensors.mag_disturbance import MagneticDisturbanceDetector
from src.idr.filters.es_ekf import ErrorStateKalmanFilter, ESEKFConfig, exp_quaternion, quat_multiply


# ===========================================================================
# 1. Canonical Tilt-Compensated Heading Tests
# ===========================================================================

def test_canonical_level_orientations():
    """Verify tilt-compensated heading matches exact ENU conventions on level ground."""
    # North: Body +X points North -> B_meas in body = [0, 0, -45] or horizontal North component along +X
    # When Vehicle points North: magnetic North is forward along +X body -> mx = 45, my = 0
    psi_north = compute_tilt_compensated_heading(np.array([45.0, 0.0, 0.0]), roll_rad=0.0, pitch_rad=0.0)
    assert np.isclose(psi_north, np.pi / 2, atol=1e-5), f"Expected North (pi/2), got {psi_north}"

    # East: Body +X points East -> magnetic North is to the left (+Y body) -> mx = 0, my = -45 (North is at -Y relative to body if East is +X)
    # When Vehicle points East: North is at +90 deg relative to heading, so in body frame: mx = 0, my = -45 (North is -Y)
    psi_east = compute_tilt_compensated_heading(np.array([0.0, -45.0, 0.0]), roll_rad=0.0, pitch_rad=0.0)
    assert np.isclose(psi_east, 0.0, atol=1e-5), f"Expected East (0.0), got {psi_east}"

    # South: Body +X points South -> magnetic North is backward (-X body) -> mx = -45, my = 0
    psi_south = compute_tilt_compensated_heading(np.array([-45.0, 0.0, 0.0]), roll_rad=0.0, pitch_rad=0.0)
    assert np.isclose(psi_south, -np.pi / 2, atol=1e-5), f"Expected South (-pi/2), got {psi_south}"

    # West: Body +X points West -> magnetic North is to the right (-Y body) -> mx = 0, my = +45
    psi_west = compute_tilt_compensated_heading(np.array([0.0, 45.0, 0.0]), roll_rad=0.0, pitch_rad=0.0)
    assert np.isclose(abs(psi_west), np.pi, atol=1e-5), f"Expected West (+/-pi), got {psi_west}"


def test_tilt_compensation_invariance():
    """Verify heading remains invariant under pitch and roll rotations of Earth magnetic field."""
    # Earth field vector in ENU frame: [0, 40, -20] uT (pointing North and Down)
    B_enu = np.array([0.0, 40.0, -20.0])

    # Vehicle is pointing North (yaw = pi/2), but pitched up by 15 deg and rolled right by 10 deg
    roll = np.deg2rad(10.0)
    pitch = np.deg2rad(15.0)

    # Rotation matrix Body -> ENU for (roll, pitch, yaw=pi/2)
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # yaw = pi/2
    R_b2enu = Rz @ Ry @ Rx

    # Magnetic vector measured in Body frame: B_body = R_b2enu.T @ B_enu
    B_body = R_b2enu.T @ B_enu

    # Compute tilt-compensated heading
    psi_est = compute_tilt_compensated_heading(B_body, roll_rad=roll, pitch_rad=pitch)
    assert np.isclose(psi_est, np.pi / 2, atol=1e-3), f"Tilt compensation failed: got {np.rad2deg(psi_est)} deg, expected 90 deg"


# ===========================================================================
# 2. Magnetometer Calibration Tests
# ===========================================================================

def test_hard_and_soft_iron_calibration():
    """Verify min-max calibration accurately centers and scales synthetic ellipsoidal data."""
    np.random.seed(42)
    # Generate points on a sphere
    phi = np.random.uniform(0, 2*np.pi, 200)
    costheta = np.random.uniform(-1, 1, 200)
    theta = np.arccos(costheta)
    r = 45.0

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    # Apply synthetic hard-iron offset [10, -20, 5] and soft-iron scale [0.8, 1.2, 1.0]
    true_offset = np.array([10.0, -20.0, 5.0])
    true_scale = np.array([0.8, 1.2, 1.0])
    mag_distorted = (np.stack([x, y, z], axis=1) / true_scale) + true_offset

    calib = MagnetometerCalibrator()
    calib.fit_min_max_sphere(mag_distorted)

    # Check recovered bias is close to true offset
    assert np.allclose(calib.hard_iron_bias, true_offset, atol=3.0), f"Hard iron bias error: {calib.hard_iron_bias}"

    # Check calibrated data norm is near 45 uT
    mag_clean = calib.transform_and_calibrate(mag_distorted)
    norms = np.linalg.norm(mag_clean, axis=1)
    assert abs(np.mean(norms) - 45.0) < 4.0


# ===========================================================================
# 3. Magnetic Disturbance Detector Tests
# ===========================================================================

def test_disturbance_detector_clean_vs_severe():
    """Verify disturbance detector accepts clean fields and rejects disturbances."""
    detector = MagneticDisturbanceDetector(ref_field_uT=45.0, norm_tolerance_uT=10.0, max_variance_uT2=25.0)

    # 1. Clean steady measurements
    for i in range(10):
        mag = np.array([40.0 + np.random.normal(0, 0.5), 0.0, -20.0])
        is_clean, diag = detector.update(mag, psi_mag=np.pi/2, gyro_z_rad_s=0.0, t_sec=i * 0.1)
    
    assert is_clean is True
    assert diag["level"] == "clean"
    assert diag["quality_score"] == 1.0

    # 2. Severe norm anomaly (e.g. 90 uT near vehicle motor)
    mag_disturbed = np.array([80.0, 30.0, -40.0])  # norm ~ 94 uT
    is_clean_dist, diag_dist = detector.update(mag_disturbed, psi_mag=np.pi/2, gyro_z_rad_s=0.0, t_sec=1.1)
    assert is_clean_dist is False
    assert diag_dist["level"] in ["suspicious", "severe"]

    # 3. Sudden heading jump without gyro rate support
    is_clean_jump, diag_jump = detector.update(np.array([40.0, 0.0, -20.0]), psi_mag=0.0, gyro_z_rad_s=0.0, t_sec=1.2)
    assert is_clean_jump is False
    assert abs(diag_jump["mag_rate_deg_s"]) > 100.0


# ===========================================================================
# 4. ES-EKF Magnetic Heading Gating & Safety Tests
# ===========================================================================

def test_es_ekf_magnetic_heading_update():
    """Verify ES-EKF accepts clean magnetic heading and rejects disturbed updates."""
    ekf = ErrorStateKalmanFilter()
    initial_yaw_std = np.sqrt(ekf.P[8, 8])

    # Clean update with consistent heading
    accepted, diag = ekf.update_magnetic_heading(heading_rad=0.0, sigma_yaw=0.1, is_clean=True)
    assert accepted is True
    post_yaw_std = np.sqrt(ekf.P[8, 8])
    assert post_yaw_std < initial_yaw_std, "Accepted magnetic update must reduce yaw uncertainty"

    # Disturbed update must be rejected
    rejected, diag_rej = ekf.update_magnetic_heading(heading_rad=1.5, sigma_yaw=0.1, is_clean=False)
    assert rejected is False
    assert "Magnetic disturbance" in diag_rej["rejection_reason"]


def test_turn_aware_nhc_policy():
    """Verify turn-aware NHC policy switches to 1-DOF (vertical only) during sharp dynamic cornering."""
    ekf = ErrorStateKalmanFilter()
    ekf.v = np.array([12.0, 0.0, 0.0])  # 12 m/s forward
    ekf.last_gyro = np.array([0.0, 0.0, 0.3])  # 0.3 rad/s turn -> a_lat = 3.6 m/s^2

    # Under 'disable_lateral' policy, lateral NHC is disabled, only vertical is updated
    accepted, diag = ekf.update_nhc(turn_policy="disable_lateral", cornering_threshold_mps2=0.5)
    assert accepted is True
    assert diag["is_dynamic_cornering"] is True
    assert diag["turn_policy"] == "disable_lateral"
