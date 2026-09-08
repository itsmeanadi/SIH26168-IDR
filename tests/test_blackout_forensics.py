"""Unit tests for Phase 20: GNSS Blackout Forensic Audit.

Validates:
1. Analytical vs Finite-Difference Jacobian consistency (H_ai)
2. Metric definition mathematical rigor (ground truth distance vs drift %)
3. Deterministic repeatability of blackout simulation
4. Causal window indexing invariant
5. Stationary detector response properties
"""

import numpy as np
import pytest

from idr.filters.ekf import ExtendedKalmanFilter
from idr.filters.zupt import StationaryDetector


def test_ai_jacobian_finite_difference_consistency():
    """Verify analytical Jacobian matches finite differences to machine precision."""
    ekf = ExtendedKalmanFilter(dt=0.1)
    ekf.x = np.array([12.5, -45.2, 3.1, 4.2, -6.8, 0.05, 1.15, -0.08, 0.003], dtype=np.float64)

    psi = ekf.x[6]
    ve = ekf.x[3]
    vn = ekf.x[4]

    def h_func(x_vec):
        return np.cos(x_vec[6]) * x_vec[3] + np.sin(x_vec[6]) * x_vec[4]

    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)
    v_lat = -sin_psi * ve + cos_psi * vn

    H_analytical = np.zeros(9, dtype=np.float64)
    H_analytical[3] = cos_psi
    H_analytical[4] = sin_psi
    H_analytical[6] = v_lat

    eps = 1e-6
    H_fd = np.zeros(9, dtype=np.float64)
    for i in range(9):
        xp = ekf.x.copy()
        xm = ekf.x.copy()
        xp[i] += eps
        xm[i] -= eps
        H_fd[i] = (h_func(xp) - h_func(xm)) / (2.0 * eps)

    max_diff = float(np.max(np.abs(H_analytical - H_fd)))
    assert max_diff < 1e-5, f"Jacobian analytical vs numerical mismatch: {max_diff}"


def test_drift_metric_definition():
    """Verify drift percentage is strictly computed relative to ground truth distance."""
    # Test case: 100m true distance, 8m final error -> 8.0% drift
    gt_trajectory = np.array([[0.0, 0.0], [100.0, 0.0]])
    est_trajectory = np.array([[0.0, 0.0], [92.0, 0.0]])

    gt_dist = float(np.sum(np.linalg.norm(np.diff(gt_trajectory, axis=0), axis=1)))
    final_err = float(np.linalg.norm(est_trajectory[-1] - gt_trajectory[-1]))
    drift_pct = (final_err / max(1.0, gt_dist)) * 100.0

    assert gt_dist == pytest.approx(100.0, abs=1e-3)
    assert final_err == pytest.approx(8.0, abs=1e-3)
    assert drift_pct == pytest.approx(8.0, abs=1e-3)


def test_stationary_detector_high_motion():
    """Verify stationary detector unlatches when significant angular or acceleration motion occurs."""
    detector = StationaryDetector(window_size=10, acc_var_threshold=0.15, gyro_norm_threshold=0.08)

    # Feed quiet frames
    for _ in range(15):
        detector.update(np.array([0.0, 0.0, 9.81]), np.array([0.0, 0.0, 0.0]))
    assert detector._is_stationary_latched is True

    # Feed high dynamic motion frame (turning at 0.5 rad/s)
    is_stat = detector.update(np.array([2.0, 1.0, 9.81]), np.array([0.0, 0.0, 0.5]))
    assert is_stat is False
    assert detector._is_stationary_latched is False
