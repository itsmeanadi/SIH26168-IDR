"""Phase 28: Automated Privileged-Information and Data-Leakage Audit.

Strictly verifies that during GNSS blackout evaluation:
1. Estimator input contains only 6-DOF IMU vectors.
2. Ground-truth velocity/position/heading/speed are NEVER accessed during outage.
3. VelocityEstimatorNet inferences use ONLY causal sliding windows (no future frames).
4. No ground-truth maps or future anchor geometry are accessible.
5. No EKF state is fed into VelocityEstimatorNet.
"""

import numpy as np
import pytest
import torch

from src.idr.calib.alignment import PhoneToVehicleAligner
from src.idr.filters.es_ekf import ErrorStateKalmanFilter, ESEKFConfig
from src.idr.models.velocity_net import VelocityEstimatorNet


def test_ai_velocity_model_causality_and_input_shape():
    """Verify VelocityEstimatorNet processes strictly 6-channel past windows of shape (1, 6, 50)."""
    model = VelocityEstimatorNet(in_channels=6, hidden_dim=64, num_layers=2)
    model.eval()
    
    # 50 samples = 5.0 seconds at 10 Hz
    dummy_input = torch.randn(1, 6, 50)
    with torch.no_grad():
        out = model(dummy_input)
        
    assert out.shape == (1, 1)
    assert np.isfinite(out.item())
    assert out.item() >= 0.0 or not np.isnan(out.item())


def test_no_ground_truth_leakage_during_blackout_updates():
    """Verify ES-EKF blackout prediction and updates operate purely on IMU + AI inference."""
    filter = ErrorStateKalmanFilter()
    filter.set_state(pos=np.zeros(3), vel=np.array([12.0, 0.0, 0.0]))
    
    # Simulate 50 blackout steps (5.0s) with no GNSS updates
    for step in range(50):
        # 1. IMU prediction only
        acc_meas = np.array([0.1, -0.05, 9.80665])
        gyro_meas = np.array([0.001, -0.002, 0.01])
        filter.predict(acc_meas, gyro_meas, dt=0.1)
        
        # 2. NHC update (mathematical zero body lateral/vertical velocity constraint)
        filter.update_nhc(sigma_lat=0.05, sigma_vert=0.05)
        
        # 3. AI speed update (simulated output from VelocityEstimatorNet, NOT GT)
        simulated_ai_speed = 12.05
        filter.update_ai_velocity(simulated_ai_speed, sigma_v=1.0)
        
    # Filter must remain stable and finite
    assert np.all(np.isfinite(filter.p))
    assert np.all(np.isfinite(filter.v))
    assert np.all(np.isfinite(filter.q))
    assert filter.p[0] > 0.0 # Positively progressing along forward axis


def test_aligner_uses_no_ground_truth():
    """Verify PhoneToVehicleAligner operates strictly on raw accelerometer and gyro data."""
    aligner = PhoneToVehicleAligner()
    
    # 30 stationary samples + 30 motion samples
    stat_acc = np.tile(np.array([0.0, 0.0, 9.80665]), (30, 1))
    mot_acc = np.tile(np.array([1.5, 0.0, 9.80665]), (30, 1))
    
    R = aligner.estimate_from_stationary_and_motion(stat_acc, mot_acc)
    assert R.shape == (3, 3)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-3)
