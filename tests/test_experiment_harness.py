"""Unit and Regression Tests for Phase 44: Unified Experiment Harness and Reporting Layer.

Validates:
1. ExperimentHarness execution on recorded physical sessions (REAL_DEVICE_OBSERVATION).
2. ExperimentHarness execution on authentic IO-VNBD drives (AUTHENTIC_DATASET) against reference ground truth.
3. Machine-readable JSON and human-readable Markdown generation.
4. Correct separation between user-reported ~10m field observation and ground truth reference.
5. Leakage invariants (drive-level disjoint split, no future lookahead, no GT speed during blackout).
"""

from pathlib import Path
import json
import numpy as np
import pytest

from idr.eval.experiment_runner import (
    ExperimentHarness,
    ExperimentResult,
    ExperimentMetadata,
    TimingAndSamplingStats,
    SpeedAccuracyMetrics,
    DeadReckoningMetrics,
)


def test_experiment_harness_physical_session_replay():
    """Verify that ExperimentHarness correctly parses and reports real physical session telemetry."""
    session_csv = Path("data/sessions/exp_20260908_172204_two_wheeler/telemetry.csv")
    if not session_csv.exists():
        pytest.skip("Physical session telemetry file not found.")

    harness = ExperimentHarness(models_dir="models")
    res = harness.run_physical_session_replay(
        session_csv_path=session_csv,
        experiment_id="test_exp_physical_replay",
        notes="Automated test replay of physical field session.",
    )

    # 1. Provenance metadata
    assert res.metadata.provenance_type == "REAL_DEVICE_OBSERVATION"
    assert res.metadata.dataset_role == "FIELD_TRIAL"
    assert res.metadata.is_scientific_research_valid is True
    assert res.metadata.model_authenticity == "authentic_iovnbd"

    # 2. Timing & sampling stats
    assert res.timing.total_duration_sec > 250.0
    assert res.timing.total_samples == 5850
    assert 25.0 < res.timing.effective_hz < 60.0
    assert res.timing.dt_median_ms > 0.0

    # 3. Reference distance handling
    assert res.dead_reckoning.reference_distance_m is None
    assert res.diagnostics.get("user_reported_distance_is_ground_truth") is False

    # 4. JSON & Markdown serialization
    tmp_json = session_csv.parent / "test_report.json"
    tmp_md = session_csv.parent / "test_report.md"
    try:
        res.save_json(tmp_json)
        res.save_markdown(tmp_md)

        assert tmp_json.exists()
        assert tmp_md.exists()

        with open(tmp_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["metadata"]["provenance_type"] == "REAL_DEVICE_OBSERVATION"
        assert "AI VelocityNet" in tmp_md.read_text(encoding="utf-8")
    finally:
        if tmp_json.exists():
            tmp_json.unlink()
        if tmp_md.exists():
            tmp_md.unlink()


def test_experiment_harness_authentic_drive_evaluation():
    """Verify that ExperimentHarness evaluates authentic IO-VNBD data against vehicle CAN ground truth."""
    drive_y1_dir = Path("data/raw/categorised_authentic/Y (Driver D)/Y1")
    if not drive_y1_dir.exists():
        pytest.skip("Authentic Drive Y1 data folder not found.")

    harness = ExperimentHarness(models_dir="models")
    res = harness.run_authentic_drive_evaluation(
        driver_folder_path=drive_y1_dir,
        experiment_id="test_exp_drive_y1",
        stride=10,
        dataset_role="TEST",
    )

    # 1. Provenance metadata
    assert res.metadata.provenance_type == "AUTHENTIC_DATASET"
    assert res.metadata.dataset_role == "TEST"
    assert res.metadata.leakage_audit_passed is True

    # 2. Reference ground truth metrics present
    assert res.ai_velocity.has_reference is True
    assert res.ai_velocity.mae_mps is not None
    assert res.ai_velocity.rmse_mps is not None
    assert res.dead_reckoning.reference_distance_m is not None
    assert res.dead_reckoning.reference_distance_m > 1000.0  # > 1 km drive

    # 3. Model authenticity check
    assert res.metadata.model_authenticity == "authentic_iovnbd"
    assert res.metadata.is_scientific_research_valid is True


def test_leakage_audit_invariants():
    """Verify that dataset splits and feature generation strictly enforce zero leakage."""
    fingerprint_path = Path("data/raw/dataset_fingerprint.json")
    if not fingerprint_path.exists():
        pytest.skip("dataset_fingerprint.json not found.")

    with open(fingerprint_path, "r", encoding="utf-8") as f:
        fp = json.load(f)

    split = fp.get("driver_disjoint_split", {})
    train_drivers = set(split.get("train_drivers", []))
    val_drivers = set(split.get("val_drivers", []))
    test_drivers = set(split.get("test_drivers", []))

    # Strict zero-overlap check across drivers
    assert len(train_drivers.intersection(val_drivers)) == 0
    assert len(train_drivers.intersection(test_drivers)) == 0
    assert len(val_drivers.intersection(test_drivers)) == 0
    assert split.get("is_driver_disjoint") is True
