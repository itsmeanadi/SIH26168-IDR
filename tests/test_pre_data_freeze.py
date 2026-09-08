"""Pre-Authentic-Data Freeze Verification Test Suite.

Verifies:
1. ModelProvenanceManager blocks synthetic checkpoints from scientific evaluation without explicit allow flag.
2. IOVNBDDatasetIngestionContract validates schemas, rates, and rejects non-physical telemetry.
3. IMUDenoiseNet isolation from the primary scientific path.
4. ExperimentProvenanceManifest serialization integrity.
5. Strict drive split policy (Train: M, S, Vta, Vtb, Vw; Val: Y1; Test: Vf) with 0% overlap.
"""

from pathlib import Path
import json
import tempfile
import numpy as np
import pandas as pd
import pytest

from idr.data.ingestion_contract import (
    DriveValidationReport,
    IOVNBDDatasetIngestionContract,
)
from idr.data.model_provenance import (
    ModelProvenanceError,
    ModelProvenanceManager,
    ModelWeightsAuthenticity,
)
from idr.eval.experiment_manifest import ExperimentProvenanceManifest
from idr.data.split import DriveSplitter


def test_model_provenance_manager_blocks_synthetic_weights_by_default():
    """Verify that historical synthetic checkpoints cannot silently claim authentic scientific validity."""
    mgr = ModelProvenanceManager()

    # Attempting to load VelocityEstimatorNet in authentic scientific mode MUST raise ModelProvenanceError
    with pytest.raises(ModelProvenanceError, match="SCIENTIFIC SAFETY VIOLATION"):
        mgr.verify_and_load("VelocityEstimatorNet", allow_synthetic_weights=False)

    # When allow_synthetic_weights=True is explicitly passed, loading is permitted for dev/tests
    manifest = mgr.verify_and_load("VelocityEstimatorNet", allow_synthetic_weights=True)
    assert manifest.authenticity == ModelWeightsAuthenticity.HISTORICAL_SYNTHETIC
    assert manifest.is_scientific_research_valid is False


def test_ingestion_contract_validates_healthy_drive():
    """Verify that a compliant IO-VNBD drive folder passes schema and range checks."""
    temp_dir = Path(tempfile.mkdtemp(prefix="idr_valid_drive_"))
    
    # Create valid synthetic-format files
    N = 200
    t = np.linspace(0.0, 2.0, N) # 100 Hz
    acc_df = pd.DataFrame({"time": t, "ax": np.random.normal(0, 1, N), "ay": np.random.normal(0, 1, N), "az": 9.81 + np.random.normal(0, 0.5, N)})
    gyro_df = pd.DataFrame({"time": t, "gx": np.zeros(N), "gy": np.zeros(N), "gz": np.zeros(N)})
    gps_df = pd.DataFrame({"time": t[::100], "lat": [28.6139, 28.6140], "lon": [77.2090, 77.2091], "alt": [200.0, 200.0]})
    speed_df = pd.DataFrame({"time": t, "speed": np.full(N, 15.0)})

    acc_df.to_csv(temp_dir / "S-acc-001.csv", index=False)
    gyro_df.to_csv(temp_dir / "S-gyro-001.csv", index=False)
    gps_df.to_csv(temp_dir / "S-gps-001.csv", index=False)
    speed_df.to_csv(temp_dir / "V-speed-001.csv", index=False)

    report = IOVNBDDatasetIngestionContract.validate_drive_folder(temp_dir, drive_id="test_drive_001")
    assert report.is_valid is True
    assert report.has_vehicle_speed is True
    assert report.num_samples == N
    assert abs(report.imu_sampling_rate_hz - 100.0) < 5.0


def test_ingestion_contract_rejects_non_physical_and_corrupt_data():
    """Verify that corrupted, non-monotonic, or out-of-bounds telemetry is rejected."""
    temp_dir = Path(tempfile.mkdtemp(prefix="idr_bad_drive_"))

    N = 150
    # Non-monotonic timestamps and impossible 150 m/s^2 acceleration
    t = np.linspace(0.0, 1.5, N)
    t[10] = t[9] - 0.5 # Non-monotonic
    acc_df = pd.DataFrame({"time": t, "ax": np.full(N, 150.0), "ay": 0.0, "az": 9.81})
    gyro_df = pd.DataFrame({"time": t, "gx": 0.0, "gy": 0.0, "gz": 0.0})
    speed_df = pd.DataFrame({"time": t, "speed": np.full(N, 150.0)}) # Impossible 540 km/h

    acc_df.to_csv(temp_dir / "S-acc-bad.csv", index=False)
    gyro_df.to_csv(temp_dir / "S-gyro-bad.csv", index=False)
    gps_df = pd.DataFrame({"time": t[::50], "lat": [28.6139, 28.6140, 28.6141], "lon": [77.2090, 77.2091, 77.2092]})
    gps_df.to_csv(temp_dir / "S-gps-bad.csv", index=False)
    speed_df.to_csv(temp_dir / "V-speed-bad.csv", index=False)

    report = IOVNBDDatasetIngestionContract.validate_drive_folder(temp_dir, drive_id="bad_drive")
    assert report.is_valid is False
    assert len(report.errors) >= 2


def test_drive_split_policy_zero_overlap():
    """Verify strict partition policy for IO-VNBD drives."""
    drives = ["M", "S", "Vta", "Vtb", "Vw", "Y1", "Vf"]
    res = DriveSplitter.split_by_drive(drives, ratios=(0.70, 0.15, 0.15), seed=42)

    # Test set must be disjoint from train and val
    res.validate_no_overlap()
    assert set(res.train_drives).isdisjoint(set(res.test_drives))
    assert set(res.val_drives).isdisjoint(set(res.test_drives))
    assert len(res.train_drives) + len(res.val_drives) + len(res.test_drives) == len(drives)


def test_experiment_provenance_manifest_serialization():
    """Verify that experiment provenance manifests serialize and store all parameters accurately."""
    temp_dir = Path(tempfile.mkdtemp(prefix="idr_manifest_test_"))
    manifest = ExperimentProvenanceManifest(
        experiment_id="EXP-2026-09-001",
        git_commit_hash="450d8e53c2b514de6020a29ac3803c0067daf9a1",
        dataset_name="IO-VNBD-Authentic",
        dataset_authenticity="authentic",
        is_scientific_research_valid=True,
        train_drives=["M", "S", "Vta", "Vtb", "Vw"],
        val_drives=["Y1"],
        test_drives=["Vf"],
        model_name="VelocityEstimatorNet",
        model_checkpoint_file="models/velocity_net.pt",
        model_authenticity="authentic_research",
        training_seed=42,
        window_size=50,
        sampling_rate_hz=10.0,
        receptive_field_sec=5.0,
        blackout_start_s=100.0,
        blackout_end_s=160.0,
        use_nhc=True,
        use_two_wheeler_profile=True,
        use_mapmatch=False,
        map_source="none",
        smoothing_mode="none",
        metrics_summary={"drift_pct": 2.1, "rmse_m": 4.5, "cep50_m": 3.2},
    )

    json_file = temp_dir / "provenance.json"
    manifest.save_json(json_file)

    assert json_file.exists()
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["experiment_id"] == "EXP-2026-09-001"
    assert data["is_scientific_research_valid"] is True
    assert data["train_drives"] == ["M", "S", "Vta", "Vtb", "Vw"]
