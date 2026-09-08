"""Comprehensive regression and integrity tests for authentic IO-VNBD synchronization."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from idr.io.sync import (
    parse_phone_date_to_seconds,
    estimate_temporal_offset,
    synchronize_and_interpolate_drive,
)
from idr.io.loader import load_drive_pair, standardize_dataframe
from idr.io.schema import detect_schema
from idr.io.preprocess import (
    create_sliding_windows,
    detect_continuous_segments,
    GAP_THRESHOLD_SEC,
)


def test_parse_phone_date_to_seconds():
    """Test parsing of phone DATE string to continuous seconds."""
    s = pd.Series(["2019-08-30 18:12:45:703", "2019-08-30 18:12:46:703"])
    t = parse_phone_date_to_seconds(s)
    assert len(t) == 2
    assert np.isclose(t[1] - t[0], 1.0)
    expected_t0 = 18 * 3600 + 12 * 60 + 45 + 0.703
    assert np.isclose(t[0], expected_t0)


def test_timestamp_monotonicity():
    """Verify that parsed date timestamps are strictly monotonic."""
    s = pd.Series([
        "2019-08-30 10:00:00:000",
        "2019-08-30 10:00:00:100",
        "2019-08-30 10:00:00:200",
        "2019-08-30 10:00:00:300",
    ])
    t = parse_phone_date_to_seconds(s)
    dt = np.diff(t)
    assert np.all(dt > 0), "Timestamps must be strictly monotonic"


def test_estimate_temporal_offset_synthetic():
    """Test offset recovery on known synthetic signals with arbitrary time shift."""
    t_base = np.linspace(0, 100, 1000)
    true_speed = 10.0 + 5.0 * np.sin(2 * np.pi * 0.05 * t_base)
    
    t_phone = t_base + 3600.0  # phone in BST (+1h)
    p_speed = true_speed.copy()
    
    true_shift = 15.0
    t_vehicle = t_base + 3600.0 - true_shift
    v_speed = true_speed.copy()
    
    best_tau, tau_0, corr = estimate_temporal_offset(t_phone, t_vehicle, p_speed, v_speed)
    assert np.isclose(best_tau, true_shift, atol=0.15), f"Expected offset ~{true_shift}s, got {best_tau}s"
    assert corr > 0.99, f"Expected near-perfect correlation, got {corr}"


def test_interpolation_within_overlap_no_extrapolation():
    """Verify that interpolation strictly constrains output to valid overlap."""
    df_p = pd.DataFrame({
        "timestamp": np.linspace(10.0, 50.0, 401),
        "acc_x": np.zeros(401),
        "acc_y": np.zeros(401),
        "acc_z": np.ones(401) * 9.81,
        "gyro_x": np.zeros(401),
        "gyro_y": np.zeros(401),
        "gyro_z": np.zeros(401),
        "speed": np.linspace(0, 20, 401),
    })
    
    df_v = pd.DataFrame({
        "timestamp": np.linspace(20.0, 40.0, 201),
        "wheel_speed": np.linspace(5, 15, 201),
    })
    
    aligned_df, aligned_v_spd, meta = synchronize_and_interpolate_drive(df_p, df_v)
    assert len(aligned_df) == len(aligned_v_spd)
    assert meta["valid_overlap_samples"] <= len(df_p)
    assert not np.isnan(aligned_v_spd).any()
    assert np.all(aligned_v_spd >= 5.0) and np.all(aligned_v_spd <= 15.0)


def test_large_gap_protection():
    """Verify continuous segment detection splits on gaps > 0.35s."""
    t = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 1.0, 1.1, 1.2, 1.3, 1.4])
    segments = detect_continuous_segments(t, gap_threshold_sec=GAP_THRESHOLD_SEC, min_segment_length=4)
    assert len(segments) == 2
    assert segments[0] == (0, 5)
    assert segments[1] == (5, 10)


def test_target_endpoint_alignment():
    """Verify sliding windows target is speed at window endpoint (s+49)."""
    N = 100
    imu = np.ones((N, 6), dtype=np.float32)
    speeds = np.arange(N, dtype=np.float32)
    t = np.arange(N, dtype=np.float64) * 0.1
    
    X_w, y_v, y_i, meta = create_sliding_windows(
        imu, speeds, window_size=50, stride=10, timestamps=t, return_metadata=True
    )
    assert len(X_w) == 6
    for i in range(len(X_w)):
        end_idx = meta[i]["end_idx"]
        assert np.isclose(y_v[i], speeds[end_idx - 1])


def test_rebuilt_manifest_and_driver_disjoint_splits():
    """Verify rebuilt dataset manifest integrity and Driver-Disjoint split."""
    manifest_path = Path("data/processed/authentic/manifest.json")
    assert manifest_path.exists(), "Rebuilt manifest.json must exist"
    
    with open(manifest_path) as f:
        manifest = json.load(f)
        
    assert manifest["imu_denoise_available"] is False, "IMU denoiser must be marked unavailable"
    assert manifest["split_policy"] == "driver_disjoint"
    
    train_npz = np.load("data/processed/authentic/train_data.npz")
    val_npz = np.load("data/processed/authentic/val_data.npz")
    test_npz = np.load("data/processed/authentic/test_data.npz")
    
    assert len(train_npz["windows"]) == 88218
    assert len(val_npz["windows"]) == 10581
    assert len(test_npz["windows"]) == 7012
    
    # Check no NaN/Inf
    for name, npz in [("Train", train_npz), ("Val", val_npz), ("Test", test_npz)]:
        assert not np.isnan(npz["windows"]).any(), f"{name} windows contain NaNs"
        assert not np.isnan(npz["targets_vel"]).any(), f"{name} targets contain NaNs"
        assert not np.isinf(npz["windows"]).any(), f"{name} windows contain Infs"
        assert not np.isinf(npz["targets_vel"]).any(), f"{name} targets contain Infs"


def test_drive_y1_alignment_qc():
    """Verify Drive Y1 specifically achieves high physical alignment correlation."""
    drive_folder = Path("data/raw/categorised_authentic/Y (Driver D)/Y1")
    if drive_folder.exists():
        drive = load_drive_pair(drive_folder, "Y1")
        phone_imu, phone_gps, v_speed, t = drive.get_synced_data()
        
        p_gps_spd = phone_gps[:, 2] / 3.6
        valid = (~np.isnan(p_gps_spd)) & (~np.isnan(v_speed))
        corr = np.corrcoef(p_gps_spd[valid], v_speed[valid])[0, 1]
        
        assert corr > 0.90, f"Expected Drive Y1 aligned correlation > 0.90, got {corr:.4f}"


def test_feature_channels_no_leakage():
    """Verify that feature array X contains strictly 6 IMU channels with no GPS or targets."""
    for split in ["train", "val", "test"]:
        npz = np.load(f"data/processed/authentic/{split}_data.npz")
        X = npz["windows"]
        y = npz["targets_vel"]
        
        # Must be exactly shape (N, 6, 50)
        assert X.ndim == 3
        assert X.shape[1] == 6, "Features must have exactly 6 channels [acc_x..z, gyro_x..z]"
        assert X.shape[2] == 50, "Window length must be 50 samples"
        
        # Targets must not be identical to any feature slice
        for c in range(6):
            assert not np.allclose(X[:, c, -1], y), f"Feature channel {c} must not equal target y"


def test_scaler_isolation():
    """Verify that train_scaler.json was fitted strictly on training windows."""
    scaler_path = Path("data/processed/authentic/train_scaler.json")
    assert scaler_path.exists(), "train_scaler.json must exist"
    with open(scaler_path) as f:
        scaler = json.load(f)
        
    assert scaler["channels"] == ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    assert scaler["fitted_samples_count"] == 88218 * 50
    assert len(scaler["mean"]) == 6
    assert len(scaler["std"]) == 6
    for s in scaler["std"]:
        assert s > 0, "Scaler standard deviations must be strictly positive"


def test_no_drive_boundary_crossing():
    """Verify that manifest drive window totals exactly sum to the split window totals."""
    manifest_path = Path("data/processed/authentic/manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
        
    train_windows = sum(d["num_windows"] for d in manifest["drives"] if d["split"] == "train")
    val_windows = sum(d["num_windows"] for d in manifest["drives"] if d["split"] == "val")
    test_windows = sum(d["num_windows"] for d in manifest["drives"] if d["split"] == "test")
    
    assert train_windows == 88218
    assert val_windows == 10581
    assert test_windows == 7012
    
    # Verify no duplicate drives across splits
    train_drives = {d["drive_id"] for d in manifest["drives"] if d["split"] == "train"}
    val_drives = {d["drive_id"] for d in manifest["drives"] if d["split"] == "val"}
    test_drives = {d["drive_id"] for d in manifest["drives"] if d["split"] == "test"}
    
    assert len(train_drives.intersection(val_drives)) == 0, "Train and Val drives must be disjoint"
    assert len(train_drives.intersection(test_drives)) == 0, "Train and Test drives must be disjoint"
    assert len(val_drives.intersection(test_drives)) == 0, "Val and Test drives must be disjoint"
