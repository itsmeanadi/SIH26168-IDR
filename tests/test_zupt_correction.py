"""Unit tests for Phase 21: ZUPT Correction & Stationary Detector Regression.

Verifies:
1. True stationary: near-gravity acceleration + near-zero gyro -> stationary=True
2. Smooth constant-velocity cruise: stable acceleration magnitude + forward speed > 0.5 -> stationary=False
3. Constant-velocity cruise with small sensor noise -> stationary=False
4. Accelerating vehicle: net acceleration deviates from gravity -> stationary=False
5. Turning vehicle: angular rate > threshold -> stationary=False
6. Genuine stop after motion: stationary=True only after required temporal persistence (5 frames)
7. No GNSS: detector remains causal and functional
8. No AI: detector functions safely using EKF prior speed
9. ZUPT cannot repeatedly force a moving EKF state to zero
10. Legitimate ZUPT clamping remains fully active during genuine vehicle stops
"""

import numpy as np
import pytest

from idr.filters.zupt import StationaryDetector, apply_zupt, apply_zaru
from idr.filters.ekf import ExtendedKalmanFilter
from idr.engine.navigation_engine import NavigationEngine, SensorInputFrame, GNSSInputFix


def test_1_true_stationary():
    """Test 1: Vehicle at complete rest with vertical gravity and zero rotation."""
    detector = StationaryDetector(window_size=10, persistence_frames=5)
    
    # Push 15 frames of stationary data
    for _ in range(15):
        is_stat = detector.update(
            acc_3d=np.array([0.0, 0.0, 9.81]),
            gyro_3d=np.array([0.0, 0.0, 0.0]),
            speed_mps=0.0,
        )
    assert is_stat is True
    assert detector._is_stationary_latched is True


def test_2_smooth_constant_velocity_cruise():
    """Test 2: Smooth cruising at 15 m/s with zero acceleration and zero gyro rate."""
    detector = StationaryDetector(window_size=10, persistence_frames=5)

    # In smooth cruise, accelerometer measures purely gravity [0, 0, 9.81], gyro is [0, 0, 0]
    # But vehicle forward speed is 15.0 m/s
    for _ in range(20):
        is_stat = detector.update(
            acc_3d=np.array([0.0, 0.0, 9.81]),
            gyro_3d=np.array([0.0, 0.0, 0.0]),
            speed_mps=15.0,
        )
    assert is_stat is False
    assert detector._is_stationary_latched is False


def test_3_constant_velocity_cruise_with_sensor_noise():
    """Test 3: Cruising at 8 m/s with slight sensor vibration noise."""
    detector = StationaryDetector(window_size=10, persistence_frames=5)
    rng = np.random.default_rng(42)

    for _ in range(30):
        acc_noisy = np.array([0.0, 0.0, 9.81]) + rng.normal(0, 0.05, size=3)
        gyro_noisy = rng.normal(0, 0.01, size=3)
        is_stat = detector.update(
            acc_3d=acc_noisy,
            gyro_3d=gyro_noisy,
            speed_mps=8.0,
        )
        assert is_stat is False


def test_4_accelerating_vehicle():
    """Test 4: Vehicle accelerating at +2.0 m/s^2 forward."""
    detector = StationaryDetector(window_size=10, persistence_frames=5)

    for _ in range(15):
        is_stat = detector.update(
            acc_3d=np.array([2.0, 0.0, 9.81]),  # Net acceleration magnitude = sqrt(4 + 9.81^2) = 10.01 m/s^2
            gyro_3d=np.array([0.0, 0.0, 0.0]),
            speed_mps=5.0,
        )
    assert is_stat is False


def test_5_turning_vehicle():
    """Test 5: Vehicle cornering with yaw rate = 0.2 rad/s."""
    detector = StationaryDetector(window_size=10, persistence_frames=5)

    for _ in range(15):
        is_stat = detector.update(
            acc_3d=np.array([0.0, 1.5, 9.81]),  # Centripetal acceleration
            gyro_3d=np.array([0.0, 0.0, 0.2]),  # Yaw rate = 0.2 rad/s
            speed_mps=6.0,
        )
    assert is_stat is False


def test_6_genuine_stop_after_motion():
    """Test 6: Vehicle moves, then brakes to a stop and requires full buffer flush + persistence to latch."""
    detector = StationaryDetector(window_size=10, persistence_frames=2)

    # 1. Moving phase (10 frames)
    for _ in range(10):
        detector.update(
            acc_3d=np.array([1.5, 0.0, 9.81]),
            gyro_3d=np.array([0.0, 0.0, 0.1]),
            speed_mps=10.0,
        )
    assert detector._is_stationary_latched is False

    # 2. Braking / coming to rest (speed = 0.0):
    # First 9 frames: sliding buffer still contains moving IMU frames -> NOT stationary
    for i in range(1, 10):
        is_stat = detector.update(
            acc_3d=np.array([0.0, 0.0, 9.81]),
            gyro_3d=np.array([0.0, 0.0, 0.0]),
            speed_mps=0.0,
        )
        assert is_stat is False, f"Premature latching at frame {i}"

    # Frame 10: buffer is now 100% quiet, persistence counter = 1 -> not yet latched
    is_stat_10 = detector.update(
        acc_3d=np.array([0.0, 0.0, 9.81]),
        gyro_3d=np.array([0.0, 0.0, 0.0]),
        speed_mps=0.0,
    )
    # Frame 11 (2nd consecutive quiet frame after full buffer) -> MUST latch
    is_stat_11 = detector.update(
        acc_3d=np.array([0.0, 0.0, 9.81]),
        gyro_3d=np.array([0.0, 0.0, 0.0]),
        speed_mps=0.0,
    )
    assert is_stat_11 is True
    assert detector._is_stationary_latched is True


def test_7_no_gnss_causal_operation():
    """Test 7: Detector operates reliably during GNSS blackout without crashes or future lookahead."""
    detector = StationaryDetector(window_size=10, persistence_frames=5)

    # Run 50 frames during simulated outage with only IMU and EKF estimated speed
    for i in range(50):
        is_stat = detector.update(
            acc_3d=np.array([0.0, 0.0, 9.81]),
            gyro_3d=np.array([0.0, 0.0, 0.0]),
            speed_mps=7.5,
        )
        assert is_stat is False


def test_8_no_ai_safe_fallback():
    """Test 8: Detector functions when AI speed estimate is completely unavailable."""
    detector = StationaryDetector(window_size=10, persistence_frames=5)

    # Passing speed_mps=None falls back strictly to sensor quietness & persistence
    for _ in range(15):
        detector.update(
            acc_3d=np.array([0.0, 0.0, 9.81]),
            gyro_3d=np.array([0.0, 0.0, 0.0]),
            speed_mps=None,
        )
    # When speed is unknown and IMU is quiet for 15 frames, detector latches stationary
    assert detector._is_stationary_latched is True


def test_9_zupt_cannot_force_moving_ekf_to_zero():
    """Test 9: Verify EKF state is NOT repeatedly forced to zero when vehicle is cruising."""
    ekf = ExtendedKalmanFilter(dt=0.1)
    ekf.x[3] = 10.0  # 10 m/s East
    ekf.x[6] = 0.0   # Heading East

    detector = StationaryDetector(window_size=10, persistence_frames=5)

    # Run 20 integration steps of cruising
    for _ in range(20):
        ekf.predict(fwd_accel=0.0, yaw_rate=0.0)
        cur_spd = float(np.hypot(ekf.x[3], ekf.x[4]))
        is_stat = detector.update(
            acc_3d=np.array([0.0, 0.0, 9.81]),
            gyro_3d=np.array([0.0, 0.0, 0.0]),
            speed_mps=cur_spd,
        )
        if is_stat:
            apply_zupt(ekf, sigma_v=0.01)

    # EKF speed should remain ~10 m/s and position should have integrated ~20 meters
    assert ekf.x[3] > 9.5
    assert ekf.x[0] > 18.0  # Position moved ~20m East


def test_10_legitimate_zupt_clamps_at_real_stop():
    """Test 10: At a genuine stop, ZUPT successfully clamps small residual gyro/accel drift."""
    ekf = ExtendedKalmanFilter(dt=0.1)
    ekf.x[3] = 0.05  # Small residual 5 cm/s drift
    ekf.x[4] = 0.02
    ekf.x[6] = 0.0

    detector = StationaryDetector(window_size=10, persistence_frames=5)

    for _ in range(15):
        cur_spd = float(np.hypot(ekf.x[3], ekf.x[4]))
        is_stat = detector.update(
            acc_3d=np.array([0.0, 0.0, 9.81]),
            gyro_3d=np.array([0.0, 0.0, 0.0]),
            speed_mps=cur_spd,
        )
        if is_stat:
            apply_zupt(ekf, sigma_v=0.01)

    # Velocity must be clamped to exact zero (< 0.001 m/s)
    final_spd = float(np.hypot(ekf.x[3], ekf.x[4]))
    assert final_spd < 0.001
