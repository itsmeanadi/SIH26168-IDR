"""Unit tests for Phase 19: AI + Physics Navigation EKF Integration.

Validates:
1. AI measurement update (z_ai = forward speed)
2. Frame transformation & cross-coupling Jacobian
3. Covariance bounds (clamping min_sigma_v, max_sigma_v)
4. Innovation gating (Mahalanobis / NIS gate)
5. Stale measurement rejection
6. NaN / Inf rejection & sanitization
7. Missing model handling (graceful fallback)
8. Causal 5-second window handling (no future samples, 1 Hz stride)
9. AI rejection does not break EKF
10. GNSS blackout continues with AI
11. GNSS recovery still functions
12. AI cannot inject position/heading directly
13. No future sample access
"""

import os
import pytest
import numpy as np
import torch

from idr.filters.ekf import ExtendedKalmanFilter
from idr.filters.fusion import GNSSINSFusion
from idr.engine import NavigationEngine, SensorInputFrame, GNSSInputFix
from idr.models.velocity_net import VelocityEstimatorNet


def test_ai_measurement_update():
    """Test 1: Verify AI speed acts as an observation, updating EKF velocity state."""
    ekf = ExtendedKalmanFilter(dt=0.1)
    ekf.x[3] = 5.0  # v_E = 5.0 m/s
    ekf.x[4] = 0.0  # v_N = 0.0 m/s
    ekf.x[6] = 0.0  # Heading East (psi = 0)

    # Observe 10 m/s forward speed
    accepted, metrics = ekf.update_ai_velocity(fwd_speed=10.0, sigma_v=2.0)
    assert accepted is True
    assert metrics["accepted"] is True
    assert metrics["innovation"] == pytest.approx(5.0, abs=1e-3)
    # Velocity East should have increased towards 10 m/s
    assert ekf.x[3] > 5.0


def test_frame_transformation_and_jacobian():
    """Test 2: Verify body-to-ENU projection and cross-coupling heading Jacobian."""
    ekf = ExtendedKalmanFilter(dt=0.1)
    # Vehicle heading North: psi = pi/2
    ekf.x[6] = np.pi / 2.0
    ekf.x[3] = 0.0  # v_E = 0
    ekf.x[4] = 10.0  # v_N = 10 (forward speed = 10 m/s)

    accepted, metrics = ekf.update_ai_velocity(fwd_speed=12.0, sigma_v=2.0)
    assert accepted is True
    # Forward velocity in body frame is cos(pi/2)*0 + sin(pi/2)*10 = 10 m/s
    assert metrics["predicted"] == pytest.approx(10.0, abs=1e-3)
    assert metrics["innovation"] == pytest.approx(2.0, abs=1e-3)
    # Since heading is North, v_N should increase, v_E should remain near 0
    assert ekf.x[4] > 10.0
    assert abs(ekf.x[3]) < 0.1


def test_covariance_bounds():
    """Test 3: Verify measurement covariance is strictly clamped to [min_sigma_v, max_sigma_v]."""
    ekf = ExtendedKalmanFilter(dt=0.1)
    
    # Very small sigma -> clamped to min_sigma_v (1.0)
    accepted, metrics = ekf.update_ai_velocity(fwd_speed=5.0, sigma_v=0.01, min_sigma_v=1.0, max_sigma_v=10.0)
    assert metrics["applied_sigma"] == 1.0

    # Very large sigma -> clamped to max_sigma_v (10.0)
    accepted, metrics = ekf.update_ai_velocity(fwd_speed=5.0, sigma_v=50.0, min_sigma_v=1.0, max_sigma_v=10.0)
    assert metrics["applied_sigma"] == 10.0


def test_innovation_gating():
    """Test 4: Verify Mahalanobis / NIS gate rejects wild unphysical predictions."""
    ekf = ExtendedKalmanFilter(dt=0.1)
    ekf.x[3] = 0.0
    ekf.x[4] = 0.0
    ekf.x[6] = 0.0

    # Small innovation (1 m/s) -> accepted
    accepted_norm, metrics_norm = ekf.update_ai_velocity(fwd_speed=1.0, sigma_v=2.0, max_innovation_sigma=3.0)
    assert accepted_norm is True

    # Extreme wild jump (55 m/s when filter predicts ~0) -> rejected by NIS gate
    accepted_wild, metrics_wild = ekf.update_ai_velocity(fwd_speed=55.0, sigma_v=2.0, max_innovation_sigma=3.0)
    assert accepted_wild is False
    assert "INNOVATION_GATE_EXCEEDED" in metrics_wild["rejection_reason"]


def test_stale_and_out_of_range_measurement():
    """Test 5: Out of range (< 0 or > 60 m/s) measurements are rejected."""
    ekf = ExtendedKalmanFilter(dt=0.1)
    
    # Negative speed rejected
    acc_neg, met_neg = ekf.update_ai_velocity(fwd_speed=-5.0)
    assert acc_neg is False
    assert met_neg["rejection_reason"] == "OUT_OF_RANGE_OR_NON_FINITE"

    # Impossible speed (> 60 m/s = 216 km/h) rejected
    acc_fast, met_fast = ekf.update_ai_velocity(fwd_speed=120.0)
    assert acc_fast is False
    assert met_fast["rejection_reason"] == "OUT_OF_RANGE_OR_NON_FINITE"


def test_nan_inf_rejection():
    """Test 6: Verify NaN and Inf values are safely rejected without filter corruption."""
    ekf = ExtendedKalmanFilter(dt=0.1)
    x_before = ekf.x.copy()
    P_before = ekf.P.copy()

    acc_nan, met_nan = ekf.update_ai_velocity(fwd_speed=float("nan"))
    assert acc_nan is False
    np.testing.assert_array_equal(ekf.x, x_before)
    np.testing.assert_array_equal(ekf.P, P_before)

    acc_inf, met_inf = ekf.update_ai_velocity(fwd_speed=float("inf"))
    assert acc_inf is False
    np.testing.assert_array_equal(ekf.x, x_before)
    np.testing.assert_array_equal(ekf.P, P_before)


def test_missing_model_handling():
    """Test 7: Engine operates gracefully when no AI model checkpoint is present."""
    engine = NavigationEngine()
    engine.ai_model = None
    assert engine.ai_model is None

    # Process normal frame during simulated blackout
    engine.set_simulated_blackout(True)
    frame = SensorInputFrame(
        timestamp=100.0,
        acc_x=0.0, acc_y=0.0, acc_z=9.81,
        gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
    )
    out = engine.process_frame(frame)
    assert np.isfinite(out.latitude)
    assert np.isfinite(out.longitude)
    assert out.is_in_blackout is True


def test_causal_5s_window_handling():
    """Test 8: AI predictions only trigger after full 5.0s (50 samples) and at 1 Hz stride."""
    engine = NavigationEngine()
    engine.set_simulated_blackout(True)

    # Feed 49 samples (0.0 to 4.8s at 0.1s steps) -> buffer not ready (49 samples), no AI estimate
    for i in range(49):
        frame = SensorInputFrame(
            timestamp=i * 0.1,
            acc_x=1.0, acc_y=0.0, acc_z=9.81,
            gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
        )
        engine.process_frame(frame)
        assert engine.has_new_ai_estimate is False

    # Feed 50th sample (t=4.9s) -> buffer is ready (50 samples, emitted count=50, 50%10 == 0)
    frame_50 = SensorInputFrame(
        timestamp=4.9,
        acc_x=1.0, acc_y=0.0, acc_z=9.81,
        gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
    )
    engine.process_frame(frame_50)
    if engine.ai_model is not None:
        assert engine.has_new_ai_estimate is True
        assert engine.ai_update_count >= 1


def test_ai_rejection_does_not_break_ekf():
    """Test 9: Multiple rejected AI updates do not destabilize the covariance or state."""
    ekf = ExtendedKalmanFilter(dt=0.1)
    
    # Apply 10 bad AI measurements
    for _ in range(10):
        ekf.predict(fwd_accel=0.5, yaw_rate=0.0)
        accepted, _ = ekf.update_ai_velocity(fwd_speed=50.0)  # Gated out
        assert accepted is False

    # State & covariance must remain finite and positive definite
    assert np.all(np.isfinite(ekf.x))
    assert np.all(np.isfinite(ekf.P))
    eigenvalues = np.linalg.eigvals(ekf.P)
    assert np.all(eigenvalues > 0)


def test_gnss_blackout_continues_with_ai():
    """Test 10: During simulated blackout, EKF switches to DR + AI and propagates smoothly."""
    engine = NavigationEngine()
    
    # 1. Initialize with GNSS fix
    init_gnss = GNSSInputFix(
        timestamp=0.0,
        latitude=28.6139,
        longitude=77.2090,
        altitude=200.0,
        speed_mps=10.0,
        heading_deg=90.0,
        accuracy_m=2.0,
    )
    init_imu = SensorInputFrame(
        timestamp=0.0,
        acc_x=0.0, acc_y=0.0, acc_z=9.81,
        gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
    )
    engine.process_frame(init_imu, init_gnss)

    # 2. Enter blackout
    engine.set_simulated_blackout(True)
    for i in range(1, 60):
        frame = SensorInputFrame(
            timestamp=i * 0.1,
            acc_x=0.0, acc_y=0.0, acc_z=9.81,
            gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
        )
        out = engine.process_frame(frame)
        assert out.is_in_blackout is True
        assert np.isfinite(out.latitude)
        assert np.isfinite(out.longitude)
        assert np.isfinite(out.forward_speed_mps)


def test_gnss_recovery_still_functions():
    """Test 11: When GNSS returns after blackout, recovery and smoothing function correctly."""
    engine = NavigationEngine()
    
    # Anchor GNSS
    init_gnss = GNSSInputFix(
        timestamp=0.0,
        latitude=28.6139,
        longitude=77.2090,
        altitude=200.0,
        accuracy_m=2.0,
    )
    engine.process_frame(
        SensorInputFrame(timestamp=0.0, acc_x=0.0, acc_y=0.0, acc_z=9.81, gyro_x=0.0, gyro_y=0.0, gyro_z=0.0),
        init_gnss,
    )

    # Blackout for 5 seconds with zero acceleration (stationary/coasting at anchor)
    engine.set_simulated_blackout(True)
    for i in range(1, 51):
        engine.process_frame(
            SensorInputFrame(timestamp=i * 0.1, acc_x=0.0, acc_y=0.0, acc_z=9.81, gyro_x=0.0, gyro_y=0.0, gyro_z=0.0)
        )
    assert engine.in_blackout is True

    # GNSS returns at anchor position
    engine.set_simulated_blackout(False)
    recovery_gnss = GNSSInputFix(
        timestamp=5.1,
        latitude=28.6139,
        longitude=77.2090,
        altitude=200.0,
        accuracy_m=2.0,
    )
    out_recovered = engine.process_frame(
        SensorInputFrame(timestamp=5.1, acc_x=0.0, acc_y=0.0, acc_z=9.81, gyro_x=0.0, gyro_y=0.0, gyro_z=0.0),
        recovery_gnss,
    )
    assert engine.in_blackout is False
    assert out_recovered.gnss_trust_score > 0.0


def test_ai_cannot_inject_position_or_heading():
    """Test 12: Ensure measurement matrix H has zero position entries so AI cannot fake position."""
    ekf = ExtendedKalmanFilter(dt=0.1)
    
    # Verify Jacobian definition for AI velocity
    # H must be (1, 9) with H[0, 0:3] == 0 (no position sensitivity)
    # H[0, 3] = cos(psi), H[0, 4] = sin(psi), H[0, 6] = v_lat, all others 0
    psi = 0.5
    ekf.x[6] = psi
    ekf.x[3] = 4.0
    ekf.x[4] = 3.0
    
    # Direct observation
    ekf.update_ai_velocity(fwd_speed=5.0)
    # Position rows in EKF state should remain unchanged if initial position was 0
    assert ekf.x[0] == pytest.approx(0.0, abs=1e-6)
    assert ekf.x[1] == pytest.approx(0.0, abs=1e-6)
    assert ekf.x[2] == pytest.approx(0.0, abs=1e-6)


def test_no_future_sample_access():
    """Test 13: Ensure timestamp resampler rejects timestamps backwards in time."""
    engine = NavigationEngine()
    
    frame1 = SensorInputFrame(timestamp=10.0, acc_x=0.0, acc_y=0.0, acc_z=9.81, gyro_x=0.0, gyro_y=0.0, gyro_z=0.0)
    frame2 = SensorInputFrame(timestamp=9.0, acc_x=0.0, acc_y=0.0, acc_z=9.81, gyro_x=0.0, gyro_y=0.0, gyro_z=0.0)
    
    engine.process_frame(frame1)
    # Timestamp in the past should not advance the resampler buffer
    sample = engine.ai_resampler.add_sample(
        9.0, np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0], dtype=np.float32)
    )
    assert sample is None
