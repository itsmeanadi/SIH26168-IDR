"""Unit tests for IDR Real-Time Engine, USPs, and Replay."""

import numpy as np
import pytest

from idr.engine.blackspot import BlackspotTracker
from idr.engine.crash_detector import CrashDetector, CrashState
from idr.engine.gnss_trust import GNSSTrustEngine, GNSSTrustStatus
from idr.engine.health import HealthDiagnosticEngine, NavigationMode
from idr.engine.navigation_engine import (
    GNSSInputFix,
    NavigationEngine,
    SensorInputFrame,
)
from idr.engine.replay import DriveReplayer


def test_gnss_trust_engine_consistent_and_anomaly():
    trust_engine = GNSSTrustEngine(max_jump_dist_m=20.0, max_innovation_sigma=3.0)
    pred_dr = np.array([100.0, 100.0, 0.0])
    cov = np.eye(3) * 4.0

    # 1. Consistent GNSS Fix
    gnss_good = np.array([101.0, 99.5, 0.0])
    res_good = trust_engine.evaluate_fix(gnss_good, pred_dr, cov, timestamp=1.0)
    assert res_good.is_trusted is True
    assert res_good.status == GNSSTrustStatus.TRUSTED

    # 2. Suspicious Jump / Multipath Teleportation (50m jump in 0.1s)
    gnss_jump = np.array([150.0, 150.0, 0.0])
    res_jump = trust_engine.evaluate_fix(gnss_jump, pred_dr, cov, timestamp=1.1, inertial_speed_mps=5.0)
    assert res_jump.is_trusted is False
    assert res_jump.status in (GNSSTrustStatus.SUSPICIOUS_JUMP, GNSSTrustStatus.REJECTED_SPOOFED)


def test_blackspot_tracker_lifecycle():
    tracker = BlackspotTracker(min_duration_sec=1.0)
    
    # Start outage
    tracker.on_outage_start(28.6139, 77.2090, timestamp=10.0)
    assert tracker.active_outage is not None

    # Step inside blackout
    tracker.on_dr_update(28.6145, 77.2095, delta_dist_m=25.0, pos_uncertainty_m=2.5, timestamp=15.0)
    assert tracker.active_outage.dr_distance_m == 25.0
    assert tracker.active_outage.duration_sec == 5.0

    # End blackout
    record = tracker.on_outage_end(28.6150, 77.2100, reacquisition_jump_m=3.2, timestamp=16.0)
    assert record is not None
    assert record.final_reacquisition_error_m == 3.2
    assert tracker.active_outage is None

    geojson = tracker.to_geojson()
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 1


def test_crash_detector_physics_heuristic():
    detector = CrashDetector(impact_g_threshold=3.0, stillness_duration_sec=1.0)

    # 1. Normal driving
    alert = detector.update(
        acc_3d=np.array([1.0, 0.0, 9.81]),
        gyro_3d=np.array([0.0, 0.0, 0.0]),
        current_speed_mps=15.0,
        current_lat=28.6139,
        current_lon=77.2090,
        timestamp=0.0,
    )
    assert alert is None
    assert detector.state == CrashState.NORMAL

    # 2. High G Impact (5g shock = ~49 m/s^2)
    alert = detector.update(
        acc_3d=np.array([45.0, 20.0, 9.81]),
        gyro_3d=np.array([1.0, 5.0, 2.0]),  # rotational surge
        current_speed_mps=15.0,
        current_lat=28.6139,
        current_lon=77.2090,
        timestamp=1.0,
    )
    assert detector.state in (CrashState.IMPACT_DETECTED, CrashState.CONFIRMING_STILLNESS)

    # 3. Post-impact Stillness (v=0 for 1.2s)
    alert_confirmed = detector.update(
        acc_3d=np.array([0.0, 0.0, 9.81]),
        gyro_3d=np.array([0.0, 0.0, 0.0]),
        current_speed_mps=0.0,
        current_lat=28.6139,
        current_lon=77.2090,
        timestamp=2.5,
    )
    assert alert_confirmed is not None
    assert alert_confirmed.is_confirmed is True
    assert "EMERGENCY" in alert_confirmed.emergency_message


def test_navigation_engine_end_to_end_frame_processing():
    engine = NavigationEngine(ref_lat=28.6139, ref_lon=77.2090, vehicle_type="two_wheeler")

    # Step 1: Normal GNSS + IMU
    imu1 = SensorInputFrame(timestamp=0.0, acc_x=1.0, acc_y=0.0, acc_z=9.81, gyro_x=0.0, gyro_y=0.0, gyro_z=0.0)
    gnss1 = GNSSInputFix(timestamp=0.0, latitude=28.6139, longitude=77.2090, accuracy_m=2.5)
    out1 = engine.process_frame(imu1, gnss1)

    assert out1.nav_mode in (NavigationMode.GNSS_INS_FULL, NavigationMode.GNSS_DEGRADED)
    assert out1.is_in_blackout is False
    assert out1.gnss_trust_score >= 0.8

    # Step 2: Trigger Simulated Outage (Tunnel)
    engine.set_simulated_blackout(True)
    imu2 = SensorInputFrame(timestamp=0.1, acc_x=1.0, acc_y=0.0, acc_z=9.81, gyro_x=0.0, gyro_y=0.0, gyro_z=0.05)
    out2 = engine.process_frame(imu2, None)

    assert out2.is_in_blackout is True
    assert out2.nav_mode in (NavigationMode.DEAD_RECKONING_NHC_AI, NavigationMode.DEAD_RECKONING_PURE)

    # Step 3: Reacquisition
    engine.set_simulated_blackout(False)
    imu3 = SensorInputFrame(timestamp=0.2, acc_x=1.0, acc_y=0.0, acc_z=9.81, gyro_x=0.0, gyro_y=0.0, gyro_z=0.0)
    gnss3 = GNSSInputFix(timestamp=0.2, latitude=28.613901, longitude=77.209001, accuracy_m=2.5)
    out3 = engine.process_frame(imu3, gnss3)

    assert out3.nav_mode in (NavigationMode.REACQUISITION_SMOOTHING, NavigationMode.GNSS_INS_FULL)


def test_drive_replayer_synthetic():
    engine = NavigationEngine()
    replayer = DriveReplayer(engine)
    ok = replayer.load_iovnbd_drive("test_drive")
    assert ok is True
    assert len(replayer.data_frames) > 100

    out = replayer.step()
    assert out is not None
    assert isinstance(out.latitude, float)


def test_phone_to_vehicle_aligner_robustness():
    """Regression tests for PhoneToVehicleAligner against LinAlgError, NaN, Inf, and degenerate covariances."""
    from idr.calib.alignment import PhoneToVehicleAligner

    aligner = PhoneToVehicleAligner()

    # 1. Single-sample input (N=1, exact scenario that previously caused LinAlgError)
    R1 = aligner.estimate_from_stationary_and_motion(
        stationary_acc=np.array([[0.0, 0.0, 9.81]], dtype=np.float32),
        motion_acc=np.array([[0.0, 0.0, 9.81]], dtype=np.float32),
    )
    assert np.all(np.isfinite(R1))
    assert R1.shape == (3, 3)
    assert abs(np.linalg.det(R1) - 1.0) < 1e-2

    # 2. NaN inputs
    R_nan = aligner.estimate_from_stationary_and_motion(
        stationary_acc=np.array([[np.nan, np.nan, 9.81]], dtype=np.float32),
        motion_acc=np.array([[np.nan, 0.0, np.nan]], dtype=np.float32),
    )
    assert np.all(np.isfinite(R_nan))
    assert abs(np.linalg.det(R_nan) - 1.0) < 1e-2

    # 3. Inf inputs
    R_inf = aligner.estimate_from_stationary_and_motion(
        stationary_acc=np.array([[np.inf, 0.0, 9.81]], dtype=np.float32),
        motion_acc=np.array([[0.0, np.inf, 0.0]], dtype=np.float32),
    )
    assert np.all(np.isfinite(R_inf))
    assert abs(np.linalg.det(R_inf) - 1.0) < 1e-2

    # 4. Constant / Zero-variance degenerate samples
    R_const = aligner.estimate_from_stationary_and_motion(
        stationary_acc=np.ones((10, 3), dtype=np.float32) * 9.81,
        motion_acc=np.ones((10, 3), dtype=np.float32) * 9.81,
    )
    assert np.all(np.isfinite(R_const))
    assert abs(np.linalg.det(R_const) - 1.0) < 1e-2

    # 5. Genuine dynamic forward motion samples
    stat = np.array([[0.0, 0.0, 9.81], [0.01, -0.01, 9.80]], dtype=np.float32)
    mot = np.array([
        [1.0, 0.0, 9.81],
        [2.0, 0.0, 9.81],
        [3.0, 0.0, 9.81],
        [2.5, 0.0, 9.81],
    ], dtype=np.float32)
    R_dyn = aligner.estimate_from_stationary_and_motion(stationary_acc=stat, motion_acc=mot)
    assert np.all(np.isfinite(R_dyn))
    assert abs(np.linalg.det(R_dyn) - 1.0) < 1e-2


def test_navigation_engine_non_finite_and_degenerate_inputs():
    """Verify NavigationEngine does not crash or produce HTTP 500 when fed non-finite or degenerate frames."""
    engine = NavigationEngine(ref_lat=28.6139, ref_lon=77.2090)

    # 1. Single frame with stationary gravity
    imu_single = SensorInputFrame(timestamp=0.0, acc_x=0.0, acc_y=0.0, acc_z=9.81, gyro_x=0.0, gyro_y=0.0, gyro_z=0.0)
    out1 = engine.process_frame(imu_single, None)
    assert np.isfinite(out1.latitude)
    assert np.isfinite(out1.forward_speed_mps)

    # 2. NaN accelerometer input
    imu_nan = SensorInputFrame(timestamp=0.1, acc_x=float("nan"), acc_y=0.0, acc_z=9.81, gyro_x=0.0, gyro_y=0.0, gyro_z=0.0)
    out2 = engine.process_frame(imu_nan, None)
    assert np.isfinite(out2.latitude)

    # 3. Inf accelerometer input
    imu_inf = SensorInputFrame(timestamp=0.2, acc_x=float("inf"), acc_y=0.0, acc_z=9.81, gyro_x=0.0, gyro_y=0.0, gyro_z=0.0)
    out3 = engine.process_frame(imu_inf, None)
    assert np.isfinite(out3.latitude)


def test_navigation_engine_gnss_anchor_and_replay_isolation():
    """Regression test: Ensure DriveReplayer preloading does not corrupt engine reference,
    and NavigationEngine properly anchors to arbitrary geographic locations without thousands of km jumps.
    """
    # 1. Initialize engine
    engine = NavigationEngine()
    replayer = DriveReplayer(engine)

    # 2. Preload a dataset with reset_engine=False (as done at server startup)
    replayer.load_iovnbd_drive("Vf", reset_engine=False)
    assert engine.has_gps_anchor is False

    # 3. Supply a real-world GNSS fix from Delhi, India (~28.6139, 77.2090)
    imu = SensorInputFrame(timestamp=1000.0, acc_x=0.0, acc_y=0.0, acc_z=9.81, gyro_x=0.0, gyro_y=0.0, gyro_z=0.0)
    gnss_delhi = GNSSInputFix(timestamp=1000.0, latitude=28.6139, longitude=77.2090, altitude=216.0, accuracy_m=5.0)
    out = engine.process_frame(imu, gnss_delhi)

    # Engine must have anchored to Delhi, NOT UK (lat ~52.4)
    assert engine.has_gps_anchor is True
    assert abs(engine.ref_lat - 28.6139) < 1e-4
    assert abs(engine.ref_lon - 77.2090) < 1e-4
    assert abs(out.latitude - 28.6139) < 1e-4
    assert abs(out.longitude - 77.2090) < 1e-4
    assert out.gnss_trust_score >= 0.8
    assert out.gnss_status == "TRUSTED"

    # 4. Supply a series of frames in Bangalore, India after engine reset
    engine.reset()
    assert engine.has_gps_anchor is False

    gnss_blr = GNSSInputFix(timestamp=2000.0, latitude=12.9716, longitude=77.5946, altitude=920.0, accuracy_m=4.0)
    out_blr = engine.process_frame(imu, gnss_blr)
    assert engine.has_gps_anchor is True
    assert abs(engine.ref_lat - 12.9716) < 1e-4
    assert abs(engine.ref_lon - 77.5946) < 1e-4
    assert abs(out_blr.latitude - 12.9716) < 1e-4
    assert abs(out_blr.longitude - 77.5946) < 1e-4
    assert out_blr.gnss_trust_score >= 0.8
    assert out_blr.gnss_status == "TRUSTED"

