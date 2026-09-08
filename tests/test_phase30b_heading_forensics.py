"""Unit and regression tests for Phase 30B Heading Observability Forensics."""

import pytest
import numpy as np
from pathlib import Path

from src.idr.filters.es_ekf import ErrorStateKalmanFilter, ESEKFConfig, exp_quaternion, quat_multiply
from src.idr.eval.run_phase30b_heading_forensic import (
    audit_gyro_dataset_statistics,
    audit_magnetometer_data,
    audit_ai_heading_feasibility,
)


def test_stationary_gyro_statistics_bounds():
    """Verify that empirical stationary gyro bias from authentic dataset is strictly bounded."""
    stats = audit_gyro_dataset_statistics()
    if stats.get("status") == "error":
        pytest.skip("Raw zip not available in local test env")

    # Stationary gyro z-bias must be within physical MEMS consumer bounds (|b| < 0.5 deg/s)
    mean_bias = abs(stats["mean_stationary_bias_deg_s"])
    assert mean_bias < 0.5, f"Stationary gyro bias {mean_bias} deg/s exceeds realistic consumer MEMS specs"
    
    # Typical 30s open-loop drift from stationary bias alone must be < 2.0 deg
    drift_30s = stats["typical_drift_rate_30s_deg"]
    assert abs(drift_30s) < 2.0, f"Expected 30s stationary drift {drift_30s} deg is unexpectedly large"


def test_controlled_yaw_perturbation_rotates_heading():
    """Verify that quaternion injection correctly applies controlled yaw offset."""
    es_ekf = ErrorStateKalmanFilter()
    p0 = np.array([0.0, 0.0, 0.0])
    v0 = np.array([10.0, 0.0, 0.0])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    es_ekf.set_state(pos=p0, vel=v0)
    
    # Inject 15 deg yaw offset
    pert_deg = 15.0
    dq = exp_quaternion(np.array([0.0, 0.0, np.deg2rad(pert_deg)]))
    es_ekf.q = quat_multiply(es_ekf.q, dq)
    es_ekf.q = es_ekf.q / np.linalg.norm(es_ekf.q)
    
    yaw_deg = np.rad2deg(es_ekf.euler_angles[2])
    assert abs(yaw_deg - pert_deg) < 1e-4, f"Yaw injection failed: got {yaw_deg} deg, expected {pert_deg} deg"


def test_nhc_turning_dynamics_decoupling():
    """Verify that during sharp dynamic turning, lateral NHC decouples or can be gated."""
    es_ekf = ErrorStateKalmanFilter()
    es_ekf.v = np.array([15.0, 0.0, 0.0])  # 15 m/s forward
    es_ekf.last_gyro = np.array([0.0, 0.0, 0.2])  # 0.2 rad/s turn -> a_lat = 3.0 m/s^2
    
    # Update NHC with threshold 0.5 m/s^2
    accepted, diag = es_ekf.update_nhc(sigma_lat=0.05, sigma_vert=0.05, cornering_threshold_mps2=0.5)
    assert diag["is_dynamic_cornering"] is True, "Dynamic cornering was not detected at a_lat = 3.0 m/s^2"
    assert diag["a_lat"] >= 0.5


def test_magnetometer_audit_detection():
    """Verify magnetometer audit accurately inspects 3-axis fields and flags distortions."""
    mag_audit = audit_magnetometer_data()
    if not mag_audit.get("magnetometer_available_in_raw"):
        pytest.skip("Raw dataset not present for magnetometer check")
    
    assert mag_audit["units"] == "microTesla (uT)"
    assert mag_audit["sampling_rate_hz"] == 10.0
    for drive_id, res in mag_audit["drives_audited"].items():
        assert res["sample_count"] > 1000
        assert res["nan_count"] == 0
        # Earth field is typically 25-65 uT
        assert 20.0 <= res["mean_norm_uT"] <= 70.0


def test_covariance_observability_ill_conditioning():
    """Verify condition number and eigenvalues tracking on ES-EKF covariance."""
    es_ekf = ErrorStateKalmanFilter()
    P = es_ekf.P
    eigvals = np.linalg.eigvalsh(P)
    cond = float(np.max(eigvals) / max(1e-12, np.min(eigvals)))
    assert np.all(eigvals > 0), "Covariance matrix must be strictly positive definite"
    assert cond > 1.0, "Condition number must be >= 1.0"
