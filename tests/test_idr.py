"""Unit tests for IDR components."""

import numpy as np
import pytest
import torch

from idr.models.imu_denoise import IMUDenoiseNet
from idr.models.velocity_net import VelocityEstimatorNet
from idr.filters.ekf import ExtendedKalmanFilter
from idr.filters.nhc import apply_nhc_update
from idr.eval.metrics import compute_navigation_metrics

def test_imu_denoise_net_forward():
    model = IMUDenoiseNet(in_channels=6, hidden_dim=32, out_channels=6)
    x = torch.randn(4, 6, 50)
    out = model(x)
    assert out.shape == (4, 6)

def test_velocity_net_forward():
    model = VelocityEstimatorNet(in_channels=6, hidden_dim=32, num_layers=1)
    x = torch.randn(2, 6, 50)
    out = model(x)
    assert out.shape == (2, 1)
    assert torch.all(out >= 0.0)  # Speed must be non-negative

def test_ekf_prediction_and_nhc():
    ekf = ExtendedKalmanFilter(dt=0.1)
    # Predict forward motion
    ekf.predict(fwd_accel=1.0, yaw_rate=0.0)
    assert ekf.x[3] > 0.0 or ekf.x[4] > 0.0 or ekf.x[0] > 0.0

    # Apply NHC constraint
    apply_nhc_update(ekf, sigma_lat=0.05, sigma_vert=0.05)
    assert np.isfinite(ekf.x).all()

def test_navigation_metrics():
    # 100m straight line
    gt = np.column_stack([np.linspace(0, 100, 100), np.zeros(100)])
    pred = np.column_stack([np.linspace(0, 100, 100), np.linspace(0, 5, 100)])  # 5m final lateral drift
    metrics = compute_navigation_metrics(pred, gt)
    
    assert abs(metrics.total_distance_m - 100.0) < 1.0
    assert abs(metrics.final_drift_m - 5.0) < 0.1
    assert abs(metrics.drift_percent - 5.0) < 0.2
    assert metrics.drift_percent < 10.0  # Under acceptance criterion
