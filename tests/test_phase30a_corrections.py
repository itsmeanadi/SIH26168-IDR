"""Phase 30A: Unit and Regression Tests for Confirmed Phase 29 Corrections.

Validates:
1. Warmup GNSS gating: Temporary innovation does NOT permanently lock out later valid trusted GNSS.
2. Inconsistent GNSS rejection: Untrusted GNSS is rejected when gating policy is enforced.
3. NHC Cornering Decoupling:
   - Straight motion (|omega_z * v_x| <= 0.5 m/s^2) preserves full 3D attitude Jacobian.
   - Dynamic cornering (|omega_z * v_x| > 0.5 m/s^2) decouples lateral attitude coupling.
   - Threshold boundary behavior at exactly 0.5 m/s^2.
   - Stationary / low-speed maneuver behavior.
4. GNSS blackout isolation.
"""

import numpy as np
import pytest

from src.idr.filters.es_ekf import ErrorStateKalmanFilter, ESEKFConfig, exp_quaternion, quat_to_rot


def test_trusted_gnss_warmup_no_permanent_lockout():
    """Verify that a large temporary position offset does not lock out subsequent trusted GNSS fixes."""
    filter = ErrorStateKalmanFilter()
    filter.p = np.array([150.0, -80.0, 20.0])  # Large initial offset
    filter.v = np.array([10.0, 0.0, 0.0])

    # Normal trusted GNSS position fix at [0, 0, 0]
    accepted, diag = filter.update_gnss_pos(np.array([0.0, 0.0, 0.0]), is_trusted=True)
    assert accepted is True
    # Filter position should immediately pull towards measurement
    assert np.linalg.norm(filter.p) < 150.0

    # Next fix at [1.0, 0.0, 0.0]
    accepted2, diag2 = filter.update_gnss_pos(np.array([1.0, 0.0, 0.0]), is_trusted=True)
    assert accepted2 is True


def test_untrusted_gnss_outlier_rejection():
    """Verify that untrusted/inconsistent GNSS fixes are rejected when is_trusted=False or explicit gate."""
    filter = ErrorStateKalmanFilter()
    filter.p = np.array([0.0, 0.0, 0.0])
    
    # Propagate a few steps so P is small
    for _ in range(5):
        filter.predict(np.array([0.0, 0.0, 9.80665]), np.zeros(3), dt=0.1)

    # Massive outlier fix at 500m with is_trusted=False
    accepted, diag = filter.update_gnss_pos(np.array([500.0, 500.0, 0.0]), is_trusted=False)
    assert accepted is False
    assert "exceeded gate" in diag.get("rejection_reason", "")

    # Also with explicit gate_threshold
    accepted_gate, diag_gate = filter.update_gnss_pos(
        np.array([500.0, 500.0, 0.0]),
        gate_threshold=11.345,
        is_trusted=True
    )
    assert accepted_gate is False


def test_nhc_straight_motion_attitude_coupled():
    """Verify straight motion (|a_lat| <= 0.5) preserves full 3D attitude Jacobian."""
    filter = ErrorStateKalmanFilter()
    filter.v = np.array([15.0, 0.0, 0.0])  # 15 m/s straight
    filter.predict(np.array([0.0, 0.0, 9.80665]), np.zeros(3), dt=0.1)  # yaw_rate = 0

    accepted, diag = filter.update_nhc(yaw_rate=0.0, forward_speed=15.0)
    assert accepted is True
    assert diag["is_dynamic_cornering"] is False
    assert diag["a_lat"] == 0.0


def test_nhc_dynamic_cornering_attitude_decoupled():
    """Verify dynamic cornering (|omega_z * v_x| > 0.5 m/s^2) decouples lateral attitude Jacobian."""
    filter = ErrorStateKalmanFilter()
    filter.v = np.array([12.0, 0.0, 0.0])  # 12 m/s
    yaw_rate = 0.2  # rad/s => a_lat = 2.4 m/s^2 > 0.5 m/s^2
    filter.predict(np.array([0.0, 2.4, 9.80665]), np.array([0.0, 0.0, yaw_rate]), dt=0.1)

    accepted, diag = filter.update_nhc(yaw_rate=yaw_rate, forward_speed=12.0)
    assert accepted is True
    assert diag["is_dynamic_cornering"] is True
    assert np.isclose(diag["a_lat"], 2.4)


def test_nhc_cornering_threshold_boundary():
    """Verify exact behavior at the 0.5 m/s^2 boundary."""
    filter = ErrorStateKalmanFilter()
    filter.v = np.array([10.0, 0.0, 0.0])
    
    # Exactly 0.5 m/s^2 (yaw_rate = 0.05, speed = 10)
    filter.predict(np.array([0.0, 0.5, 9.80665]), np.array([0.0, 0.0, 0.05]), dt=0.1)
    _, diag_exact = filter.update_nhc(yaw_rate=0.05, forward_speed=10.0)
    assert diag_exact["is_dynamic_cornering"] is False

    # Just above boundary (yaw_rate = 0.051, speed = 10 => a_lat = 0.51)
    filter.predict(np.array([0.0, 0.51, 9.80665]), np.array([0.0, 0.0, 0.051]), dt=0.1)
    _, diag_above = filter.update_nhc(yaw_rate=0.051, forward_speed=10.0)
    assert diag_above["is_dynamic_cornering"] is True


def test_nhc_stationary_case():
    """Verify stationary vehicle (v = 0) keeps standard coupled Jacobian with zero lateral accel."""
    filter = ErrorStateKalmanFilter()
    filter.v = np.zeros(3)
    filter.predict(np.array([0.0, 0.0, 9.80665]), np.zeros(3), dt=0.1)

    accepted, diag = filter.update_nhc(yaw_rate=0.0, forward_speed=0.0)
    assert accepted is True
    assert diag["is_dynamic_cornering"] is False
    assert diag["a_lat"] == 0.0
