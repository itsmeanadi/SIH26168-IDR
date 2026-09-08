"""Comprehensive tests for authentic IO-VNBD extraction, continuous segmentation, and preprocessing."""

import json
from pathlib import Path
import numpy as np
import pytest
import torch
from idr.config import BASE_DIR, CONFIG, DRIVER_GROUP_MAP
from idr.io.preprocess import (
    GAP_THRESHOLD_SEC,
    apply_scaler,
    create_sliding_windows,
    detect_continuous_segments,
    fit_training_scaler,
    IDRWindowDataset,
)

RAW_AUTH_DIR = BASE_DIR / "data" / "raw" / "categorised_authentic"
PROCESSED_AUTH_DIR = BASE_DIR / "data" / "processed" / "authentic"
QC_REPORT_JSON = BASE_DIR / "reports" / "authentic_dataset_qc.json"


def test_authentic_extraction_and_drive_pairing():
    """Verify authentic extraction contains exactly 72 1:1 paired drives (144 CSVs)."""
    assert RAW_AUTH_DIR.exists(), f"Directory missing: {RAW_AUTH_DIR}"
    p_csvs = [f for f in RAW_AUTH_DIR.glob("**/*.csv") if f.name.startswith("S-") or f.name.startswith("S_")]
    v_csvs = [f for f in RAW_AUTH_DIR.glob("**/*.csv") if f.name.lower().startswith("v-") or f.name.lower().startswith("v_")]

    assert len(p_csvs) == 72, f"Expected 72 phone CSVs, found {len(p_csvs)}"
    assert len(v_csvs) == 72, f"Expected 72 vehicle CSVs, found {len(v_csvs)}"


def test_driver_disjoint_three_way_independence():
    """Verify ALL THREE pairwise intersections are empty: Train∩Val=∅, Train∩Test=∅, Val∩Test=∅."""
    cfg = CONFIG["dataset"]

    def resolve_drivers(prefixes):
        drivers = set()
        for p in prefixes:
            for d_name, d_groups in DRIVER_GROUP_MAP.items():
                if any(p.startswith(g) or g.startswith(p) for g in d_groups):
                    drivers.add(d_name)
        return drivers

    train_drivers = resolve_drivers(cfg.train_drives)
    val_drivers = resolve_drivers(cfg.val_drives)
    test_drivers = resolve_drivers(cfg.test_drives)

    assert len(train_drivers & val_drivers) == 0, f"Train and Val share: {train_drivers & val_drivers}"
    assert len(train_drivers & test_drivers) == 0, f"Train and Test share: {train_drivers & test_drivers}"
    assert len(val_drivers & test_drivers) == 0, f"Val and Test share: {val_drivers & test_drivers}"


def test_continuous_segment_gap_detector():
    """Verify that timestamp gaps > 0.35s are detected and split into separate continuous segments."""
    # Construct synthetic timestamps: 100 samples continuous (dt=0.1), gap of 5.0s, 100 samples continuous
    t1 = np.arange(100) * 0.1
    t2 = t1[-1] + 5.0 + np.arange(1, 101) * 0.1
    timestamps = np.concatenate([t1, t2])

    segments = detect_continuous_segments(timestamps, gap_threshold_sec=0.35, min_segment_length=50)
    assert len(segments) == 2
    assert segments[0] == (0, 100)
    assert segments[1] == (100, 200)


def test_no_window_crosses_a_gap():
    """Verify that sliding windows NEVER cross a timestamp gap."""
    t1 = np.arange(100) * 0.1
    t2 = t1[-1] + 10.0 + np.arange(1, 101) * 0.1
    timestamps = np.concatenate([t1, t2])

    imu_dummy = np.random.randn(200, 6).astype(np.float32)
    speed_dummy = np.ones(200, dtype=np.float32) * 10.0

    X_w, y_v, y_i, meta = create_sliding_windows(
        imu_dummy,
        speed_dummy,
        window_size=50,
        stride=10,
        timestamps=timestamps,
        gap_threshold_sec=0.35,
        return_metadata=True,
    )

    # In 100 samples with window=50, stride=10 -> (100-50)//10 + 1 = 6 windows per segment = 12 total
    assert len(X_w) == 12
    assert len(meta) == 12

    # Check that no window spans index 99 to 100
    for m in meta:
        assert not (m["start_idx"] < 100 and m["end_idx"] > 100), f"Window crossed gap: {m}"
        # Assert dt across window is continuous (< 5.5s for 50 samples)
        assert (m["t_end"] - m["t_start"]) < 5.5


def test_training_only_scaler_fitting_and_application():
    """Verify that scaler parameters are fitted strictly on training data and correctly applied."""
    X_train = np.ones((100, 6, 50), dtype=np.float32) * 5.0
    X_test = np.ones((50, 6, 50), dtype=np.float32) * 10.0

    scaler = fit_training_scaler(X_train)
    assert np.allclose(scaler["mean"], [5.0] * 6)
    assert scaler["fitted_samples_count"] == 100 * 50

    # Apply to train and test
    norm_train = apply_scaler(X_train, scaler)
    norm_test = apply_scaler(X_test, scaler)

    # For constant array with std epsilon fallback, (5-5)/1 = 0
    assert np.allclose(norm_train, 0.0)
    # Test should be (10-5)/1 = 5.0
    assert np.allclose(norm_test, 5.0)


def test_authentic_processed_dataset_integrity():
    """Verify generated NPZ files and manifest.json in data/processed/authentic/."""
    assert PROCESSED_AUTH_DIR.exists()
    train_npz = PROCESSED_AUTH_DIR / "train_data.npz"
    val_npz = PROCESSED_AUTH_DIR / "val_data.npz"
    test_npz = PROCESSED_AUTH_DIR / "test_data.npz"
    manifest_file = PROCESSED_AUTH_DIR / "manifest.json"
    scaler_file = PROCESSED_AUTH_DIR / "train_scaler.json"

    assert train_npz.exists()
    assert val_npz.exists()
    assert test_npz.exists()
    assert manifest_file.exists()
    assert scaler_file.exists()

    with np.load(train_npz) as d:
        w_train = d["windows"]
        y_train = d["targets_vel"]
        assert w_train.shape[1:] == (6, 50)
        assert len(w_train) == len(y_train)
        assert len(w_train) > 80000

    with np.load(val_npz) as d:
        w_val = d["windows"]
        assert w_val.shape[1:] == (6, 50)
        assert len(w_val) > 9000

    with np.load(test_npz) as d:
        w_test = d["windows"]
        assert w_test.shape[1:] == (6, 50)
        assert len(w_test) > 6000

    with open(manifest_file) as f:
        m = json.load(f)
        assert m["split_policy"] == "driver_disjoint"
        assert m["target_vel"] == "reference_vehicle_forward_speed_mps"


def test_idr_window_dataset_torch_compatibility():
    """Verify IDRWindowDataset loads preprocessed numpy arrays seamlessly as PyTorch tensors."""
    train_npz = PROCESSED_AUTH_DIR / "train_data.npz"
    with np.load(train_npz) as d:
        dataset = IDRWindowDataset(d["windows"][:100], d["targets_vel"][:100], d["targets_imu"][:100])

    assert len(dataset) == 100
    win, target_v, target_i = dataset[0]
    assert isinstance(win, torch.Tensor)
    assert win.shape == (6, 50)
    assert isinstance(target_v, torch.Tensor)
    assert target_v.ndim == 0 or target_v.shape == ()
    assert isinstance(target_i, torch.Tensor)
    assert target_i.shape == (6,)
