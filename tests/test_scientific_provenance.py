"""Comprehensive test suite for Scientific Data & Evaluation Infrastructure.

Covers:
1. Dataset Provenance and Registry
2. Synthetic vs Authentic Isolation & Safety Blocking
3. Drive-Level Group-Aware Splitting (Deterministic, Zero Overlap, Leakage Detection)
4. Training Safety Gate & Audit Record Generation
5. GNSS Blackout Gate Dual-Stream Isolation (Tests A through E)
6. Map-Matching Ground-Truth Leakage Safeguards
7. AI Invocation & Evaluation Provenance Auditing
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch

from idr.data.provenance import (
    DatasetAuthenticity,
    DatasetProvenance,
    DatasetRegistry,
    SyntheticDataBlockedError,
    DatasetNotFoundError,
    DataLeakageError,
    UnsafeEvaluationError,
    get_dataset_registry,
)
from idr.data.split import DriveSplitter, SplitResult
from idr.data.safety import TrainingSafetyGate, TrainingProvenanceRecord
from idr.eval.blackout_gate import GNSSBlackoutGate
from idr.eval.evaluation_harness import (
    ScientificEvaluator,
    EvaluationSpec,
    EvaluationRunResult,
)
from idr.mapmatch.osm_graph import OSMGraphLoader
from idr.models.velocity_net import VelocityEstimatorNet


# =====================================================================
# 1. DATASET PROVENANCE & REGISTRY TESTS
# =====================================================================

def test_dataset_provenance_creation_and_serialization():
    prov = DatasetProvenance(
        dataset_name="test-authentic-dataset",
        source="University Lab RTK Dataset",
        authenticity=DatasetAuthenticity.AUTHENTIC,
        vehicle_type="motorcycle",
        sensor_type="bosch_bmi088",
        sampling_rate_hz=100.0,
        ground_truth_type="novatel_span_rtk",
        license_status="cc_by_4.0",
        drive_ids=["drive_01", "drive_02", "drive_03"],
        notes="Authentic experimental dataset.",
    )

    assert prov.is_authentic() is True
    assert prov.is_synthetic() is False
    assert prov.is_field() is False

    # Dict roundtrip
    p_dict = prov.to_dict()
    assert p_dict["authenticity"] == "authentic"
    reconstructed = DatasetProvenance.from_dict(p_dict)
    assert reconstructed.dataset_name == prov.dataset_name
    assert reconstructed.authenticity == DatasetAuthenticity.AUTHENTIC

    # File / YAML roundtrip
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = Path(tmpdir) / "test_manifest.yaml"
        prov.save_manifest(yaml_path)
        loaded = DatasetProvenance.from_file(yaml_path)
        assert loaded.dataset_name == "test-authentic-dataset"
        assert loaded.sampling_rate_hz == 100.0
        assert loaded.drive_ids == ["drive_01", "drive_02", "drive_03"]


def test_dataset_registry_resolution_and_isolation():
    registry = DatasetRegistry()
    
    synth_prov = DatasetProvenance(
        dataset_name="mock-synth",
        source="simulated",
        authenticity=DatasetAuthenticity.SYNTHETIC,
    )
    auth_prov = DatasetProvenance(
        dataset_name="real-benchmark",
        source="published",
        authenticity=DatasetAuthenticity.AUTHENTIC,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        synth_dir = Path(tmpdir) / "synth_data"
        synth_dir.mkdir()
        auth_dir = Path(tmpdir) / "auth_data"
        auth_dir.mkdir()

        registry.register(synth_prov, data_path=synth_dir)
        registry.register(auth_prov, data_path=auth_dir)

        # Authentic dataset resolves cleanly
        res_path, res_prov = registry.resolve("real-benchmark", allow_synthetic=False)
        assert res_path == auth_dir
        assert res_prov.is_authentic() is True

        # Synthetic dataset BLOCKS unless allow_synthetic=True
        with pytest.raises(SyntheticDataBlockedError):
            registry.resolve("mock-synth", allow_synthetic=False)

        # Explicit allow_synthetic succeeds
        res_path, res_prov = registry.resolve("mock-synth", allow_synthetic=True)
        assert res_path == synth_dir
        assert res_prov.is_synthetic() is True


# =====================================================================
# 2. DRIVE-LEVEL SPLIT TESTS (ZERO LEAKAGE)
# =====================================================================

def test_drive_splitter_determinism_and_ratios():
    drives = [f"drive_{i:02d}" for i in range(20)]
    
    split1 = DriveSplitter.split_by_drive(drives, ratios=(0.70, 0.15, 0.15), seed=42)
    split2 = DriveSplitter.split_by_drive(drives, ratios=(0.70, 0.15, 0.15), seed=42)

    assert split1.train_drives == split2.train_drives
    assert split1.val_drives == split2.val_drives
    assert split1.test_drives == split2.test_drives

    assert len(split1.train_drives) == 14
    assert len(split1.val_drives) == 3
    assert len(split1.test_drives) == 3
    assert split1.total_drives == 20


def test_drive_splitter_strict_zero_overlap():
    drives = [f"seq_{i}" for i in range(50)]
    split = DriveSplitter.split_by_drive(drives, ratios=(0.60, 0.20, 0.20), seed=123)

    train_set = set(split.train_drives)
    val_set = set(split.val_drives)
    test_set = set(split.test_drives)

    # Strictly disjoint sets
    assert len(train_set & val_set) == 0, "Leakage between train and val!"
    assert len(train_set & test_set) == 0, "Leakage between train and test!"
    assert len(val_set & test_set) == 0, "Leakage between val and test!"

    # All items accounted for exactly once
    assert train_set | val_set | test_set == set(drives)


def test_split_result_detects_manual_data_leakage():
    # Intentionally craft a contaminated split with shared sequence
    contaminated = SplitResult(
        train_drives=["drive_A", "drive_B", "drive_C"],
        val_drives=["drive_C", "drive_D"],  # drive_C leaked!
        test_drives=["drive_E"],
        ratios=(0.6, 0.2, 0.2),
        seed=42,
        total_drives=5,
    )
    with pytest.raises(DataLeakageError) as exc_info:
        contaminated.validate_no_overlap()
    assert "drive_C" in str(exc_info.value)


# =====================================================================
# 3. TRAINING SAFETY GATE TESTS
# =====================================================================

def test_training_safety_gate_blocks_synthetic_by_default():
    registry = DatasetRegistry()
    synth_prov = DatasetProvenance(
        dataset_name="synthetic-iovnbd-mock",
        source="download_data.py",
        authenticity=DatasetAuthenticity.SYNTHETIC,
    )
    registry.register(synth_prov)

    # Must raise SyntheticDataBlockedError when allow_synthetic=False
    with pytest.raises(SyntheticDataBlockedError):
        TrainingSafetyGate.enforce(
            dataset_name="synthetic-iovnbd-mock",
            allow_synthetic=False,
            registry=registry,
        )

    # Must pass when allow_synthetic=True
    prov = TrainingSafetyGate.enforce(
        dataset_name="synthetic-iovnbd-mock",
        allow_synthetic=True,
        registry=registry,
    )
    assert prov.dataset_name == "synthetic-iovnbd-mock"


def test_training_provenance_record_saving():
    record = TrainingProvenanceRecord(
        model_name="VelocityEstimatorNet",
        dataset_name="authentic-iovnbd",
        authenticity="authentic",
        is_scientific_research_valid=True,
        allow_synthetic_flag=False,
        train_drives=["drive_1", "drive_2"],
        val_drives=["drive_3"],
        num_train_samples=1000,
        num_val_samples=200,
        epochs_trained=30,
        notes="Authentic training run",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        json_file = Path(tmpdir) / "training_provenance.json"
        record.save(json_file)
        assert json_file.exists()
        import json
        with open(json_file) as f:
            data = json.load(f)
            assert data["is_scientific_research_valid"] is True
            assert data["authenticity"] == "authentic"
            assert data["epochs_trained"] == 30


# =====================================================================
# 4. GNSS BLACKOUT GATE DUAL-STREAM TESTS (Tests A through E)
# =====================================================================

def test_gnss_blackout_gate_five_invariants():
    """Verify the 5 critical blackout invariants:
    A. GNSS is present in the reference stream.
    B. GNSS is absent from the navigation input during blackout.
    C. GNSS resumes afterward.
    D. Reference GNSS remains available for evaluation.
    E. No hidden GNSS-derived state is injected during blackout.
    """
    N = 100
    dt = 0.1
    # Blackout between index 30 and 70 (t = 3.0s to 7.0s)
    gate = GNSSBlackoutGate(
        blackout_start_s=3.0,
        blackout_end_s=7.0,
        blackout_start_idx=30,
        blackout_end_idx=70,
    )

    for i in range(N):
        t_s = i * dt
        imu = np.ones(6, dtype=np.float32)
        raw_gnss = (52.4000 + i * 0.0001, -1.5000 + i * 0.0001, 15.0, 0.0)

        nav_gnss, is_bo = gate.process_step(
            index=i,
            timestamp_s=t_s,
            imu_data=imu,
            raw_gnss=raw_gnss,
        )

        if 30 <= i < 70:
            # INVARIANT B: GNSS must be absent from navigation during blackout
            assert is_bo is True
            assert nav_gnss is None, f"Leakage at step {i}: nav_gnss was {nav_gnss}"
        else:
            # INVARIANT C: GNSS is active outside blackout
            assert is_bo is False
            assert nav_gnss is not None, f"GNSS did not resume at step {i}"

    # INVARIANT A: Reference stream has GNSS throughout blackout
    for s in gate.reference_stream:
        assert s.reference_gnss is not None, "Reference GNSS was lost!"

    # INVARIANT D: Reference positions remain available for evaluation
    ref_coords = gate.get_reference_positions()
    assert ref_coords.shape == (N, 2)
    assert not np.isnan(ref_coords).any()

    # INVARIANT E: Structural verification
    checks = gate.verify_no_state_leakage()
    assert checks["reference_retained_during_blackout"] is True
    assert checks["navigation_gnss_strictly_blocked"] is True
    assert checks["navigation_resumed_after_blackout"] is True


# =====================================================================
# 5. MAP-MATCHING GROUND-TRUTH SAFETY TESTS
# =====================================================================

def test_osm_graph_loader_blocks_ground_truth_waypoints_by_default():
    loader = OSMGraphLoader()
    dummy_waypoints = np.random.randn(20, 2)

    # Must raise UnsafeEvaluationError without explicit unsafe flag
    with pytest.raises(UnsafeEvaluationError) as exc_info:
        loader.build_from_waypoints(dummy_waypoints, unsafe_allow_ground_truth_graph=False)
    assert "MAP-MATCHING SAFETY VIOLATION" in str(exc_info.value)

    # Safe independent synthetic grid generator works without ground truth
    grid_g = loader.build_synthetic_grid(center_xy=(0.0, 0.0), extent_m=400.0, grid_step_m=100.0)
    assert len(grid_g.nodes) > 0
    assert len(grid_g.edges) > 0


# =====================================================================
# 6. EVALUATION HARNESS & AI PROVENANCE TESTS
# =====================================================================

def test_scientific_evaluator_ai_provenance_tracking():
    registry = DatasetRegistry()
    synth_prov = DatasetProvenance(
        dataset_name="synthetic-iovnbd-mock",
        source="mock_gen",
        authenticity=DatasetAuthenticity.SYNTHETIC,
    )
    registry.register(synth_prov)

    evaluator = ScientificEvaluator(registry=registry)

    N = 100
    imu_data = np.zeros((N, 6), dtype=np.float32)
    imu_data[:, 0] = 0.5  # 0.5 m/s^2 accel
    gps_latlon = np.zeros((N, 2), dtype=np.float64)
    gps_latlon[:, 0] = 52.4068 + np.arange(N) * 0.00001
    gps_latlon[:, 1] = -1.5197 + np.arange(N) * 0.00001

    spec = EvaluationSpec(
        dataset_name="synthetic-iovnbd-mock",
        sequence_ids=["seq_1"],
        blackout_start_s=3.0,
        blackout_end_s=7.0,
        allow_synthetic=True,
        use_ai_velocity=True,
        use_nhc=True,
        use_kalmannet=False,
    )

    vel_model = VelocityEstimatorNet()
    result = evaluator.evaluate_scenario(
        imu_data=imu_data,
        gps_latlon=gps_latlon,
        spec=spec,
        vel_model=vel_model,
    )

    # Provenance assertions
    audit = result.provenance_audit
    assert audit.ai_model_invoked is True
    assert audit.ai_model_name == "VelocityEstimatorNet"
    assert audit.ai_observation_source == "neural_inference"
    assert audit.kalmannet_invoked is False
    assert audit.navigation_gnss_leakage_safe is True
    assert audit.is_scientific_research_valid is False  # Because dataset is synthetic

    # Metrics computed
    assert "final_drift_m" in result.metrics
    assert "drift_percent" in result.metrics
    assert result.blackout_points == 40  # steps 30 to 70


def test_scientific_evaluator_blocks_synthetic_evaluation_without_flag():
    registry = DatasetRegistry()
    synth_prov = DatasetProvenance(
        dataset_name="synthetic-iovnbd-mock",
        source="mock_gen",
        authenticity=DatasetAuthenticity.SYNTHETIC,
    )
    registry.register(synth_prov)

    evaluator = ScientificEvaluator(registry=registry)
    spec = EvaluationSpec(
        dataset_name="synthetic-iovnbd-mock",
        sequence_ids=["seq_1"],
        blackout_start_s=2.0,
        blackout_end_s=5.0,
        allow_synthetic=False,  # BLOCKED!
    )

    with pytest.raises(SyntheticDataBlockedError):
        evaluator.evaluate_scenario(
            imu_data=np.zeros((50, 6)),
            gps_latlon=np.zeros((50, 2)),
            spec=spec,
        )
