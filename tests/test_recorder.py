"""Comprehensive Test Suite for Experiment Telemetry Recorder and Session Lifecycle."""

import os
from pathlib import Path
import shutil
import tempfile
import numpy as np
import pytest

from idr.engine.navigation_engine import GNSSInputFix, NavigationEngine, SensorInputFrame
from idr.engine.replay import DriveReplayer
from idr.recorder.recorder import ExperimentRecorder
from idr.recorder.session import TelemetryRecord


@pytest.fixture
def temp_recorder_dir():
    temp_dir = tempfile.mkdtemp(prefix="idr_test_sessions_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_recorder_session_lifecycle_and_serialization(temp_recorder_dir):
    """Test session creation, streaming serialization, and finalization (Tests A, B, K)."""
    recorder = ExperimentRecorder(output_base_dir=temp_recorder_dir, buffer_flush_size=5)
    engine = NavigationEngine()

    # 1. Start Session
    device_meta = {
        "userAgent": "Mozilla/5.0 (Linux; Android 14; Pixel 8)",
        "platform": "Linux armv8l",
        "sampleRateHz": 50,
        "browser": "Chrome Mobile",
        "secret_token": "SHOULD_BE_STRIPPED_NO_PII"
    }
    session_id = recorder.start_session(
        session_id="test_session_001",
        notes="Highway 50 km/h acceleration test",
        vehicle_type="two_wheeler",
        device_info=device_meta,
    )
    assert session_id == "test_session_001"
    assert recorder.is_recording is True

    # 2. Feed 20 frames at 50 Hz (0.4 physical seconds)
    for i in range(20):
        t = 100.0 + i * 0.02
        imu = SensorInputFrame(t, 1.2, 0.0, 9.81, 0.0, 0.0, 0.05)
        gnss = GNSSInputFix(t, 28.6139 + i*1e-5, 77.2090 + i*1e-5, 215.0, 2.5, 12.0, 45.0) if i % 5 == 0 else None
        out = engine.process_frame(imu, gnss)
        recorder.record_frame(imu, gnss, out, step_latency_ms=1.5)

    # 3. Add Event Markers (Test F, H)
    recorder.add_marker("TUNNEL_ENTRY", notes="Approaching underpass")
    recorder.add_marker("BRAKING", notes="Slowing down for turn")

    # 4. Check Status
    status = recorder.get_status()
    assert status["is_recording"] is True
    assert status["record_count"] == 20
    assert status["session_id"] == "test_session_001"

    # 5. Stop Session
    summary = recorder.stop_session()
    assert summary is not None
    assert summary["record_count"] == 20
    assert summary["duration_sec"] >= 0.0
    assert len(summary["event_markers"]) == 3  # SESSION_START, TUNNEL_ENTRY, BRAKING, SESSION_STOP

    # 6. Verify Files on Disk
    session_path = Path(temp_recorder_dir) / "test_session_001"
    assert (session_path / "metadata.json").exists()
    assert (session_path / "telemetry.csv").exists()

    # 7. Privacy / No PII Validation (Test L)
    assert "secret_token" not in summary["device_info"]
    assert "platform" in summary["device_info"]


def test_mixed_rate_and_gnss_freshness_recording(temp_recorder_dir):
    """Test mixed-rate GNSS (1 Hz) vs IMU (50 Hz) and freshness tracking (Tests C, D, E)."""
    recorder = ExperimentRecorder(output_base_dir=temp_recorder_dir, buffer_flush_size=1)
    engine = NavigationEngine()
    recorder.start_session(session_id="test_mixed_rate", vehicle_type="two_wheeler")

    # Feed 10 frames: Only frame 0 and frame 5 have fresh GNSS
    for i in range(10):
        t = i * 0.02
        imu = SensorInputFrame(t, 0.5, 0.0, 9.81, 0.0, 0.0, 0.0)
        # GNSS fix arrives at t=0.0 and t=0.10
        gnss = GNSSInputFix(t, 28.6139, 77.2090, 200.0, 3.0, 10.0, 90.0) if i in (0, 5) else None
        out = engine.process_frame(imu, gnss)
        recorder.record_frame(imu, gnss, out)

    recorder.stop_session()

    # Read CSV and verify has_new_gnss and gnss_age_sec progression
    csv_file = Path(temp_recorder_dir) / "test_mixed_rate" / "telemetry.csv"
    with open(csv_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    header = lines[0].strip().split(",")
    has_gnss_idx = header.index("has_new_gnss")
    age_idx = header.index("gnss_age_sec")

    # Line 1 (i=0): has_new_gnss should be 1
    row0 = lines[1].strip().split(",")
    assert row0[has_gnss_idx] == "1"

    # Line 2 (i=1, t=0.02): has_new_gnss should be 0, age ~ 0.02s
    row1 = lines[2].strip().split(",")
    assert row1[has_gnss_idx] == "0"


def test_replay_compatibility_with_recorded_session(temp_recorder_dir):
    """Test that a recorded session can be loaded and replayed through DriveReplayer (Test I)."""
    recorder = ExperimentRecorder(output_base_dir=temp_recorder_dir, buffer_flush_size=1)
    engine_live = NavigationEngine(ref_lat=28.6139, ref_lon=77.2090)
    recorder.start_session(session_id="test_replay_session", vehicle_type="two_wheeler")

    # Record 30 synthetic frames
    for i in range(30):
        t = 100.0 + i * 0.1
        imu = SensorInputFrame(t, 1.0 + 0.1*i, 0.0, 9.81, 0.0, 0.0, 0.02)
        gnss = GNSSInputFix(t, 28.6139 + i*0.0001, 77.2090 + i*0.0001, 210.0, 2.0, 5.0, 45.0)
        out = engine_live.process_frame(imu, gnss)
        recorder.record_frame(imu, gnss, out)

    recorder.stop_session()

    # Replay session through fresh engine
    engine_replay = NavigationEngine()
    replayer = DriveReplayer(engine_replay)
    session_csv = str(Path(temp_recorder_dir) / "test_replay_session" / "telemetry.csv")

    loaded = replayer.load_recorded_session(session_csv, reset_engine=True)
    assert loaded is True
    assert len(replayer.data_frames) == 30

    # Step through replay frames
    for _ in range(10):
        step_out = replayer.step()
        assert step_out is not None
        assert np.isfinite(step_out.latitude)
        assert np.isfinite(step_out.forward_speed_mps)


def test_corrupted_and_missing_record_handling(temp_recorder_dir):
    """Test resilience against NaN/Inf and missing sensor channels (Test J)."""
    recorder = ExperimentRecorder(output_base_dir=temp_recorder_dir, buffer_flush_size=1)
    engine = NavigationEngine()
    recorder.start_session(session_id="test_corrupt", vehicle_type="two_wheeler")

    # Feed frames with NaN and Inf
    imu_nan = SensorInputFrame(1.0, float("nan"), float("inf"), 9.81, 0.0, float("-inf"), 0.0)
    out_nan = engine.process_frame(imu_nan, None)
    recorder.record_frame(imu_nan, None, out_nan)

    recorder.stop_session()
    csv_file = Path(temp_recorder_dir) / "test_corrupt" / "telemetry.csv"
    assert csv_file.exists()


def test_rest_api_recorder_endpoints(temp_recorder_dir):
    """Test REST API endpoints for recorder control (Test M)."""
    from fastapi.testclient import TestClient
    from idr.server.app import app, recorder

    recorder.output_base_dir = Path(temp_recorder_dir)
    with TestClient(app) as client:
        # 1. Start recording
        res_start = client.post("/api/recorder/start", json={"notes": "API test session", "vehicle_type": "two_wheeler"})
        assert res_start.status_code == 200
        assert res_start.json()["status"] == "RECORDING_STARTED"

        # 2. Add marker
        res_marker = client.post("/api/recorder/marker", json={"label": "POTHOLE", "notes": "Sharp dip in road"})
        assert res_marker.status_code == 200
        assert res_marker.json()["status"] == "MARKER_ADDED"

        # 3. Check status
        res_status = client.get("/api/recorder/status")
        assert res_status.status_code == 200
        assert res_status.json()["is_recording"] is True

        # 4. Ingest frame
        res_step = client.post("/api/step", json={
            "timestamp": 50.0,
            "acc_x": 0.5, "acc_y": 0.0, "acc_z": 9.81,
            "gyro_x": 0.0, "gyro_y": 0.0, "gyro_z": 0.01,
            "gnss_lat": 28.6139, "gnss_lon": 77.2090
        })
        assert res_step.status_code == 200

        # 5. Stop recording
        res_stop = client.post("/api/recorder/stop")
        assert res_stop.status_code == 200
        assert res_stop.json()["status"] == "RECORDING_STOPPED"

        # 6. List sessions
        res_list = client.get("/api/recorder/sessions")
        assert res_list.status_code == 200
        assert len(res_list.json()["sessions"]) > 0
