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


def test_aligner_multi_stage_stateful_calibration():
    """Comprehensive test suite for the 3-stage PhoneToVehicleAligner state machine:
    1. Stationary startup gravity estimation without false completion
    2. Arbitrary phone yaw mounting recovery
    3. PCA sign ambiguity resolution
    4. Degenerate motion rejection
    5. Vibration rejection
    6. Braking resilience
    7. Lean/cornering resilience
    8. NaN/Inf rejection
    """
    from idr.calib.alignment import AlignmentState, PhoneToVehicleAligner, compute_rotation_matrix

    # ── TEST 1: Stationary startup (Gravity only, must NOT falsely finish) ──
    aligner = PhoneToVehicleAligner(min_stationary_samples=20, min_motion_samples=25)
    # Tilted mounting: pitch=25 deg, roll=15 deg, yaw=30 deg
    R_true = compute_rotation_matrix(pitch=np.deg2rad(25.0), roll=np.deg2rad(15.0), yaw=np.deg2rad(30.0))
    # In phone frame, gravity reaction is R_true.T @ [0, 0, 9.81]
    g_phone = R_true.T @ np.array([0.0, 0.0, 9.81])

    for _ in range(100):
        # Add small stationary noise
        noise = np.random.normal(0, 0.01, 3)
        state = aligner.update(acc_raw=g_phone + noise, is_stationary=True)

    assert aligner.state == AlignmentState.CALIBRATING_FORWARD
    assert aligner.is_calibrated is False
    assert aligner.z_phone is not None
    # Estimated vertical axis must align with true upward vehicle vertical axis
    true_z_phone = g_phone / np.linalg.norm(g_phone)
    assert np.dot(aligner.z_phone, true_z_phone) > 0.99

    # ── TEST 2: Arbitrary yaw mounting (Forward axis estimation from motion) ──
    # Vehicle accelerates forward from 0 to 10 m/s: a_vehicle = [1.5, 0, 9.81]
    for i in range(30):
        acc_v = np.array([1.5 + np.random.normal(0, 0.05), 0.0, 9.81])
        acc_p = R_true.T @ acc_v
        speed = i * (10.0 / 30.0)
        state = aligner.update(acc_raw=acc_p, is_stationary=False, speed_mps=speed)

    assert aligner.state == AlignmentState.CALIBRATED
    assert aligner.is_calibrated is True
    # Test IMU transformation: forward acceleration in phone frame must rotate back to +X in vehicle frame
    test_fwd_phone = R_true.T @ np.array([2.0, 0.0, 9.81])
    acc_v_est, _ = aligner.transform_imu(test_fwd_phone.reshape(1, 3), np.zeros((1, 3)))
    assert acc_v_est[0, 0] > 1.8  # Positive forward acceleration recovered
    assert abs(acc_v_est[0, 1]) < 0.2  # Lateral acceleration near zero

    # ── TEST 3: PCA Sign Ambiguity (Forward vs Reverse direction resolution) ──
    aligner_sign = PhoneToVehicleAligner(min_stationary_samples=15, min_motion_samples=20)
    for _ in range(20):
        aligner_sign.update(acc_raw=np.array([0.0, 0.0, 9.81]), is_stationary=True)
    # Accelerate forward in phone frame along +Y_phone (phone mounted rotated 90 deg)
    for i in range(25):
        acc_p = np.array([0.0, 2.0 + np.random.normal(0, 0.02), 9.81])
        aligner_sign.update(acc_raw=acc_p, is_stationary=False, speed_mps=i * 0.5)
    assert aligner_sign.is_calibrated is True
    # In phone frame, [0, 1, 0] is forward -> transformed X_v must be positive
    v_out, _ = aligner_sign.transform_imu(np.array([[0.0, 1.0, 9.81]]), np.zeros((1, 3)))
    assert v_out[0, 0] > 0.9

    # ── TEST 4: Degenerate motion (Constant/near-zero acceleration rejected) ──
    aligner_degen = PhoneToVehicleAligner(min_stationary_samples=15, min_motion_samples=20)
    for _ in range(20):
        aligner_degen.update(acc_raw=np.array([0.0, 0.0, 9.81]), is_stationary=True)
    # Feed near-zero energy motion samples
    for _ in range(50):
        aligner_degen.update(acc_raw=np.array([0.001, 0.001, 9.81]), is_stationary=False)
    assert aligner_degen.state == AlignmentState.CALIBRATING_FORWARD
    assert aligner_degen.is_calibrated is False

    # ── TEST 5: Vibration without longitudinal motion rejected ──
    aligner_vibe = PhoneToVehicleAligner(min_stationary_samples=15, min_motion_samples=20)
    for _ in range(20):
        aligner_vibe.update(acc_raw=np.array([0.0, 0.0, 9.81]), is_stationary=True)
    # Circular / isotropic high-frequency vibration
    for i in range(50):
        vx = 0.3 * np.sin(i * 1.5)
        vy = 0.3 * np.cos(i * 1.5)
        aligner_vibe.update(acc_raw=np.array([vx, vy, 9.81]), is_stationary=False)
    # Isotropic motion has eigenvalue ratio ~ 1.0 -> must be rejected
    assert aligner_vibe.state == AlignmentState.CALIBRATING_FORWARD
    assert aligner_vibe.is_calibrated is False

    # ── TEST 6: Braking resilience (Subsequent braking does not corrupt locked forward axis) ──
    # aligner is already calibrated from TEST 2
    R_locked = aligner.R_phone_to_vehicle.copy()
    # Feed heavy braking (-3.0 m/s^2 forward)
    for _ in range(30):
        acc_brake_p = R_true.T @ np.array([-3.0, 0.0, 9.81])
        aligner.update(acc_raw=acc_brake_p, is_stationary=False, speed_mps=2.0)
    # Rotation matrix must remain unchanged and locked
    assert np.allclose(aligner.R_phone_to_vehicle, R_locked)
    assert aligner.is_calibrated is True

    # ── TEST 7: Lean and cornering acceleration separation ──
    aligner_lean = PhoneToVehicleAligner(min_stationary_samples=15, min_motion_samples=25)
    for _ in range(20):
        aligner_lean.update(acc_raw=np.array([0.0, 0.0, 9.81]), is_stationary=True)
    # Driving with strong forward acceleration (2.0 m/s^2) and mild lateral lean (0.5 m/s^2)
    for i in range(30):
        acc_lean_p = np.array([2.0 + np.random.normal(0, 0.05), 0.5 * np.sin(i * 0.2), 9.81])
        aligner_lean.update(acc_raw=acc_lean_p, is_stationary=False, speed_mps=i * 0.4)
    assert aligner_lean.is_calibrated is True
    # Forward axis must align with X (not Y)
    assert aligner_lean.x_phone[0] > 0.9

    # ── TEST 8: NaN / Inf input rejection ──
    aligner_nan = PhoneToVehicleAligner()
    aligner_nan.update(acc_raw=np.array([np.nan, 0.0, 9.81]), is_stationary=True)
    aligner_nan.update(acc_raw=np.array([np.inf, 0.0, 9.81]), is_stationary=False)
    assert aligner_nan.is_calibrated is False
    assert np.all(np.isfinite(aligner_nan.R_phone_to_vehicle))


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


def test_navigation_engine_gnss_freshness_caching():
    """Verify that cached GNSS fixes attached to high-frequency IMU frames do not repeatedly
    trigger EKF measurement updates or accumulate false rejections.
    """
    engine = NavigationEngine()
    base_lat, base_lon = 28.6139, 77.2090
    cached_fix = GNSSInputFix(timestamp=1000.0, latitude=base_lat, longitude=base_lon, accuracy_m=5.0)

    # Frame 0: First fix arrival -> must trigger new GNSS update
    imu0 = SensorInputFrame(timestamp=1000.0, acc_x=0.0, acc_y=0.0, acc_z=9.81, gyro_x=0.0, gyro_y=0.0, gyro_z=0.0)
    out0 = engine.process_frame(imu0, cached_fix)
    assert out0.gnss_trust_score >= 0.95
    p_pos_initial = engine.fusion.ekf.P[0, 0]

    # Frames 1..59 (59 Hz rate, same cached fix)
    for i in range(1, 60):
        t = 1000.0 + i * (1.0 / 59.0)
        imu = SensorInputFrame(timestamp=t, acc_x=0.0, acc_y=0.0, acc_z=9.81, gyro_x=0.0, gyro_y=0.0, gyro_z=0.0)
        out = engine.process_frame(imu, cached_fix)
        assert out.gnss_trust_score >= 0.95
        assert out.gnss_status == "TRUSTED"
        assert out.nav_mode in (NavigationMode.STATIONARY_ZUPT, NavigationMode.GNSS_INS_FULL)

    # EKF covariance must NOT have collapsed 60x; it should stay close to initial P (bounded by process noise)
    assert engine.fusion.ekf.P[0, 0] > 1.0

    # Frame 60: New 1 Hz GNSS fix arrives with updated timestamp
    new_fix = GNSSInputFix(timestamp=1001.0, latitude=base_lat + 1e-5, longitude=base_lon + 1e-5, accuracy_m=5.0)
    imu60 = SensorInputFrame(timestamp=1001.0, acc_x=0.0, acc_y=0.0, acc_z=9.81, gyro_x=0.0, gyro_y=0.0, gyro_z=0.0)
    out60 = engine.process_frame(imu60, new_fix)
    assert out60.gnss_trust_score >= 0.95
    assert out60.gnss_status == "TRUSTED"


def test_navigation_engine_stationary_android_scenario():
    """Simulate 5 seconds of realistic stationary Android phone sensor stream:
    - ~59 Hz IMU
    - ~1 Hz GNSS with ~1m natural stationary GPS wander
    - Verify GNSS Trust remains TRUSTED (>90%) throughout.
    """
    engine = NavigationEngine()
    base_lat, base_lon = 28.6139, 77.2090
    current_gps = GNSSInputFix(timestamp=1000.0, latitude=base_lat, longitude=base_lon, accuracy_m=5.0)

    for i in range(295):  # 5 seconds at 59 Hz
        t = 1000.0 + i * (1.0 / 59.0)

        # Every 59 frames (~1 sec), GPS receiver emits a new fix with ~1m wander
        if i % 59 == 0 and i > 0:
            current_gps = GNSSInputFix(
                timestamp=t,
                latitude=base_lat + (0.000008 if (i // 59) % 2 == 0 else -0.000008),
                longitude=base_lon + (0.000008 if (i // 59) % 2 == 1 else -0.000008),
                accuracy_m=5.0,
            )

        imu = SensorInputFrame(timestamp=t, acc_x=0.01, acc_y=0.01, acc_z=9.81, gyro_x=0.001, gyro_y=0.001, gyro_z=0.001)
        out = engine.process_frame(imu, current_gps)

        # Trust score must NEVER drop into DEGRADED (0.50) while stationary with good GPS
        assert out.gnss_trust_score >= 0.90
        assert out.gnss_status == "TRUSTED"
        if i >= 10:
            assert out.nav_mode == NavigationMode.STATIONARY_ZUPT

    # Inject genuine spoofing jump (150m) at t = 1005.0
    spoofed_fix = GNSSInputFix(timestamp=1005.0, latitude=base_lat + 0.0015, longitude=base_lon + 0.0015, accuracy_m=5.0)
    imu_jump = SensorInputFrame(timestamp=1005.0, acc_x=0.01, acc_y=0.01, acc_z=9.81, gyro_x=0.001, gyro_y=0.001, gyro_z=0.001)
    out_jump = engine.process_frame(imu_jump, spoofed_fix)

    # Anomaly detector must catch the 150m jump immediately
    assert out_jump.gnss_status in ("SUSPICIOUS_JUMP", "REJECTED_SPOOFED")
    assert out_jump.gnss_trust_score < 0.90


def test_navigation_engine_stale_gnss_outage_and_recovery():
    """Verify complete 3-phase lifecycle:
    1. Healthy 59 Hz IMU + 1 Hz GNSS
    2. GNSS stops updating (stale cached fix attached) -> transitions to DEAD_RECKONING & Blackspot tracking after timeout
    3. New GNSS fix arrives -> reacquisition smoother triggers, blackout closes, and returns to GNSS navigation.
    """
    engine = NavigationEngine(gnss_stale_timeout_sec=2.0)
    base_lat, base_lon = 28.6139, 77.2090
    dt_imu = 1.0 / 59.0

    # Phase 1: 3 seconds of healthy stationary / slow drift 59 Hz IMU + 1 Hz GNSS
    current_gps = GNSSInputFix(timestamp=1000.0, latitude=base_lat, longitude=base_lon, accuracy_m=3.0)
    for i in range(177):  # 3 seconds
        t = 1000.0 + i * dt_imu
        if i % 59 == 0 and i > 0:
            # 1 Hz fresh GNSS fix with ~0.5m wander
            current_gps = GNSSInputFix(
                timestamp=t,
                latitude=base_lat + (0.000005 if (i // 59) % 2 == 0 else -0.000005),
                longitude=base_lon,
                accuracy_m=3.0,
            )
        imu = SensorInputFrame(
            timestamp=t,
            acc_x=0.01,
            acc_y=0.01,
            acc_z=9.81,
            gyro_x=0.001,
            gyro_y=0.001,
            gyro_z=0.001,
        )
        out = engine.process_frame(imu, current_gps)
        assert out.gnss_status == "TRUSTED"
        assert out.is_in_blackout is False

    last_healthy_gps = current_gps

    # Phase 2: GNSS stops updating completely! Same cached fix is attached for 4 seconds (t=1003 to 1007)
    entered_dr = False
    for i in range(177, 177 + 236):  # 4 seconds
        t = 1000.0 + i * dt_imu
        # Simulate active forward motion during outage with road vibration so stationary detector is False
        vibe = 0.5 * np.sin(i * 0.5)
        imu = SensorInputFrame(
            timestamp=t,
            acc_x=0.5 + vibe,
            acc_y=vibe,
            acc_z=9.81,
            gyro_x=0.01,
            gyro_y=0.01,
            gyro_z=0.01,
        )
        out = engine.process_frame(imu, last_healthy_gps)  # Same cached GPS fix!

        gnss_age = t - engine.last_gnss_arrival_time
        if gnss_age <= 2.0:
            # Within stale window: still considered valid without duplicate EKF update
            assert out.gnss_status == "TRUSTED"
            assert out.is_in_blackout is False
        else:
            # Beyond 2.0s stale timeout: MUST transition to DEAD_RECKONING & Blackspot outage
            entered_dr = True
            assert out.is_in_blackout is True
            assert out.gnss_status == "BLACKOUT"
            assert out.nav_mode in (
                NavigationMode.DEAD_RECKONING_PURE,
                NavigationMode.DEAD_RECKONING_NHC_AI,
                NavigationMode.STATIONARY_ZUPT,
            )
            assert out.active_blackspot_id is not None
            assert out.pos_uncertainty_m > 0.0

    assert entered_dr is True

    # Phase 3: Fresh GNSS fix arrives after tunnel/outage exit at t = 1007.5
    t_reacq = 1007.5
    new_gps = GNSSInputFix(
        timestamp=t_reacq,
        latitude=base_lat + 0.00002,
        longitude=base_lon + 0.00002,
        accuracy_m=3.0,
    )
    imu_reacq = SensorInputFrame(
        timestamp=t_reacq,
        acc_x=0.0,
        acc_y=0.0,
        acc_z=9.81,
        gyro_x=0.0,
        gyro_y=0.0,
        gyro_z=0.0,
    )
    out_reacq = engine.process_frame(imu_reacq, new_gps)

    # Must immediately exit blackout and trigger reacquisition smoothing
    assert out_reacq.is_in_blackout is False
    assert out_reacq.nav_mode == NavigationMode.REACQUISITION_SMOOTHING
    assert out_reacq.gnss_status == "TRUSTED"
    assert out_reacq.gnss_trust_score >= 0.8


def test_ai_input_coordinate_frame_consistency_and_inference():
    """Comprehensive test suite verifying that all 6 features supplied to VelocityEstimatorNet
    are strictly expressed in the vehicle coordinate frame:
    Contract: [a_v_x, a_v_y, a_v_z, gyro_v_x, gyro_v_y, gyro_v_z]

    Tests:
    A. Identity orientation passthrough
    B. 90-degree yaw mounting rotation recovery
    C. Arbitrary 3D mounting (yaw + pitch + roll) 6-axis reconstruction
    D. Feature ordering and channel semantics
    E. Double-rotation prevention
    F. Calibration-incomplete handling
    G. NaN/Inf input robustness
    H. Live VelocityEstimatorNet neural network inference
    """
    from idr.calib.alignment import AlignmentState, PhoneToVehicleAligner, compute_rotation_matrix

    # ── TEST A: Identity Orientation ──
    engine_id = NavigationEngine()
    engine_id.aligner.R_phone_to_vehicle = np.eye(3, dtype=np.float32)
    engine_id.aligner.state = AlignmentState.CALIBRATED
    engine_id.aligner.is_calibrated = True

    imu_id = SensorInputFrame(
        timestamp=100.0,
        acc_x=1.2,
        acc_y=-0.3,
        acc_z=9.81,
        gyro_x=0.01,
        gyro_y=-0.02,
        gyro_z=0.05,
    )
    engine_id.process_frame(imu_id, None)
    buf_id = engine_id.imu_window_buffer[-1]
    expected_id = np.array([1.2, -0.3, 9.81, 0.01, -0.02, 0.05], dtype=np.float32)
    assert np.allclose(buf_id, expected_id, atol=1e-4)

    # ── TEST B: 90-Degree Yaw Mount (Landscape) ──
    # R_90 rotates +X_vehicle into +Y_phone (phone rotated 90 deg clockwise in yaw)
    R_90 = compute_rotation_matrix(pitch=0.0, roll=0.0, yaw=np.deg2rad(90.0))
    engine_90 = NavigationEngine()
    engine_90.aligner.R_phone_to_vehicle = R_90.astype(np.float32)
    engine_90.aligner.state = AlignmentState.CALIBRATED
    engine_90.aligner.is_calibrated = True

    # Vehicle motion: forward accel 2.0 m/s^2, yaw rate 0.1 rad/s
    acc_v = np.array([2.0, 0.0, 9.81])
    gyro_v = np.array([0.0, 0.0, 0.1])
    # In phone coordinates: v_phone = R_90.T @ v_vehicle
    acc_p = R_90.T @ acc_v
    gyro_p = R_90.T @ gyro_v

    imu_90 = SensorInputFrame(
        timestamp=200.0,
        acc_x=float(acc_p[0]),
        acc_y=float(acc_p[1]),
        acc_z=float(acc_p[2]),
        gyro_x=float(gyro_p[0]),
        gyro_y=float(gyro_p[1]),
        gyro_z=float(gyro_p[2]),
    )
    engine_90.process_frame(imu_90, None)
    buf_90 = engine_90.imu_window_buffer[-1]

    # Reconstructed AI vector MUST be in vehicle frame: [2.0, 0.0, 9.81, 0.0, 0.0, 0.1]
    assert abs(buf_90[0] - 2.0) < 0.01   # a_v_x (Forward)
    assert abs(buf_90[1] - 0.0) < 0.01   # a_v_y (Lateral)
    assert abs(buf_90[2] - 9.81) < 0.01  # a_v_z (Vertical)
    assert abs(buf_90[5] - 0.1) < 0.01   # gyro_v_z (Yaw)

    # ── TEST C: Arbitrary 3D Mounting (Yaw=40°, Pitch=20°, Roll=15°) ──
    R_arb = compute_rotation_matrix(pitch=np.deg2rad(20.0), roll=np.deg2rad(15.0), yaw=np.deg2rad(40.0))
    engine_arb = NavigationEngine()
    engine_arb.aligner.R_phone_to_vehicle = R_arb.astype(np.float32)
    engine_arb.aligner.state = AlignmentState.CALIBRATED
    engine_arb.aligner.is_calibrated = True

    # True vehicle-frame test vector across all 6 axes
    acc_v_true = np.array([1.75, -0.45, 9.55])
    gyro_v_true = np.array([0.03, -0.02, 0.09])

    # Convert to phone-frame inputs
    acc_p_arb = R_arb.T @ acc_v_true
    gyro_p_arb = R_arb.T @ gyro_v_true

    imu_arb = SensorInputFrame(
        timestamp=300.0,
        acc_x=float(acc_p_arb[0]),
        acc_y=float(acc_p_arb[1]),
        acc_z=float(acc_p_arb[2]),
        gyro_x=float(gyro_p_arb[0]),
        gyro_y=float(gyro_p_arb[1]),
        gyro_z=float(gyro_p_arb[2]),
    )
    engine_arb.process_frame(imu_arb, None)
    buf_arb = engine_arb.imu_window_buffer[-1]

    # Verify all 6 axes are cleanly reconstructed into vehicle frame
    assert np.allclose(buf_arb[0:3], acc_v_true, atol=1e-3)
    assert np.allclose(buf_arb[3:6], gyro_v_true, atol=1e-3)

    # ── TEST D: Feature Ordering & Channel Contract ──
    # [0] = a_v_x, [1] = a_v_y, [2] = a_v_z, [3] = g_v_x, [4] = g_v_y, [5] = g_v_z
    assert len(buf_arb) == 6
    assert isinstance(buf_arb, np.ndarray)
    assert buf_arb.dtype == np.float32

    # ── TEST E: Double-Rotation Prevention ──
    # Verify that aligner.transform_imu applies R_phone_to_vehicle exactly once
    v_raw = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    v_transformed, _ = engine_arb.aligner.transform_imu(v_raw, np.zeros((1, 3)))
    expected_v = (R_arb.astype(np.float32) @ v_raw.T).T
    assert np.allclose(v_transformed, expected_v, atol=1e-4)

    # ── TEST F: Calibration Incomplete Handling ──
    engine_uncal = NavigationEngine()
    assert engine_uncal.aligner.is_calibrated is False
    imu_uncal = SensorInputFrame(1.0, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0)
    engine_uncal.process_frame(imu_uncal, None)
    # While uncalibrated and gravity not yet estimated, interim safe passthrough occurs
    assert len(engine_uncal.imu_window_buffer) == 1
    assert np.all(np.isfinite(engine_uncal.imu_window_buffer[0]))

    # ── TEST G: NaN/Inf Rejection ──
    engine_nan = NavigationEngine()
    imu_nan = SensorInputFrame(2.0, float("nan"), float("inf"), 9.81, 0.0, float("nan"), 0.0)
    engine_nan.process_frame(imu_nan, None)
    buf_nan = engine_nan.imu_window_buffer[-1]
    assert np.all(np.isfinite(buf_nan))

    # ── TEST H: Live VelocityEstimatorNet Neural Inference ──
    engine_nn = NavigationEngine()
    for i in range(55):
        imu_frame = SensorInputFrame(
            timestamp=10.0 + i * 0.1,
            acc_x=0.5 + 0.1 * np.sin(i * 0.3),
            acc_y=0.02,
            acc_z=9.81,
            gyro_x=0.001,
            gyro_y=0.001,
            gyro_z=0.005,
        )
        engine_nn.process_frame(imu_frame, None)

    # Receptive field full -> AI model evaluated
    assert np.isfinite(engine_nn.latest_ai_speed)
    assert engine_nn.latest_ai_speed >= 0.0


def test_ai_resampler_temporal_alignment_and_receptive_field():
    """Comprehensive test suite for AI IMU TimestampAwareAIResampler.

    Validates:
    - Test A: Perfect 10 Hz input preservation
    - Test B: 50 Hz input downsampled to ~10 Hz representation
    - Test C: 59 Hz Android Chrome input downsampled to ~10 Hz representation
    - Test D: Jittered timestamp resilience
    - Test E: Dropped packet gap interpolation
    - Test F: Known sinusoidal signal preservation without phase corruption
    - Test G: Constant acceleration preservation
    - Test H: Physical temporal context validation (50 samples == 5.0 seconds)
    - Test I: Vehicle-frame channel contract preservation [ax, ay, az, gx, gy, gz]
    - Test J: Live VelocityEstimatorNet inference with 50 Hz streaming input
    """
    from idr.engine.resampler import TimestampAwareAIResampler

    # ── TEST A: Perfect 10 Hz Input ──
    resampler_10 = TimestampAwareAIResampler(target_rate_hz=10.0, window_size=50)
    for i in range(50):
        t = i * 0.100
        vec = np.array([1.5, 0.2, 9.81, 0.01, 0.0, 0.05], dtype=np.float32)
        out = resampler_10.add_sample(t, vec)
        assert out is not None
        assert np.allclose(out, vec, atol=1e-5)
    assert len(resampler_10.ai_window_buffer) == 50
    assert resampler_10.is_ready is True

    # ── TEST B: 50 Hz Input (20 ms spacing) ──
    # 5.0 seconds of 50 Hz input = 250 frames -> must emit exactly 50 samples at 10 Hz
    resampler_50 = TimestampAwareAIResampler(target_rate_hz=10.0, window_size=50)
    emitted_50 = 0
    for i in range(250):
        t = i * 0.020
        vec = np.array([2.0, -0.1, 9.81, 0.0, 0.0, 0.1], dtype=np.float32)
        out = resampler_50.add_sample(t, vec)
        if out is not None:
            emitted_50 += 1
            assert np.allclose(out, vec, atol=1e-4)
    assert emitted_50 == 50
    assert len(resampler_50.ai_window_buffer) == 50

    # ── TEST C: 59 Hz Android Chrome Input (~16.95 ms spacing) ──
    # 5.0 seconds of 59 Hz input = ~295 frames -> must emit ~50 samples at 10 Hz
    resampler_59 = TimestampAwareAIResampler(target_rate_hz=10.0, window_size=50)
    emitted_59 = 0
    for i in range(295):
        t = i * (1.0 / 59.0)
        vec = np.array([1.8, 0.05, 9.81, 0.02, -0.01, 0.08], dtype=np.float32)
        out = resampler_59.add_sample(t, vec)
        if out is not None:
            emitted_59 += 1
    assert 48 <= emitted_59 <= 52
    assert len(resampler_59.ai_window_buffer) == 50

    # ── TEST D: Jittered Timestamps (15 ms to 25 ms random jitter around 50 Hz) ──
    resampler_jitter = TimestampAwareAIResampler(target_rate_hz=10.0, window_size=50)
    rng = np.random.RandomState(42)
    t_curr = 0.0
    emitted_jitter = 0
    while t_curr < 5.0:
        dt = rng.uniform(0.015, 0.025)
        t_curr += dt
        vec = np.array([1.0, 0.0, 9.81, 0.0, 0.0, 0.0], dtype=np.float32)
        out = resampler_jitter.add_sample(t_curr, vec)
        if out is not None:
            emitted_jitter += 1
    assert 48 <= emitted_jitter <= 52

    # ── TEST E: Dropped Samples / Packet Loss Gap ──
    resampler_drop = TimestampAwareAIResampler(target_rate_hz=10.0, window_size=50)
    # Feed 1.0s of data (10 emitted), then 0.5s drop gap, then resume
    for i in range(50):
        resampler_drop.add_sample(i * 0.02, np.array([1.0, 0, 9.81, 0, 0, 0], dtype=np.float32))
    # Jump by 0.5s (gap from 1.0s to 1.5s)
    resampler_drop.add_sample(1.50, np.array([2.0, 0, 9.81, 0, 0, 0], dtype=np.float32))
    # Gap interpolation should have filled missing steps
    assert len(resampler_drop.ai_window_buffer) >= 15

    # ── TEST F: Known Sinusoidal Signal Preservation ──
    # 0.5 Hz sine wave sampled at 50 Hz: anti-aliasing mean should closely preserve the waveform
    resampler_sine = TimestampAwareAIResampler(target_rate_hz=10.0, window_size=50)
    sine_out = []
    sine_t = []
    for i in range(250):
        t = i * 0.020
        val = np.sin(2.0 * np.pi * 0.5 * t)
        vec = np.array([val, 0.0, 9.81, 0.0, 0.0, 0.0], dtype=np.float32)
        out = resampler_sine.add_sample(t, vec)
        if out is not None:
            sine_out.append(out[0])
            sine_t.append(t)
    # Correlation with true 10 Hz sampled sine should be > 0.98
    true_sine = np.sin(2.0 * np.pi * 0.5 * np.array(sine_t))
    corr = np.corrcoef(np.array(sine_out), true_sine)[0, 1]
    assert corr > 0.98

    # ── TEST G: Constant Acceleration Value Preservation ──
    resampler_const = TimestampAwareAIResampler(target_rate_hz=10.0, window_size=50)
    for i in range(100):
        vec = np.array([3.14159, -1.414, 9.81, 0.123, -0.456, 0.789], dtype=np.float32)
        resampler_const.add_sample(i * 0.02, vec)
    latest_const = resampler_const.ai_window_buffer[-1]
    assert np.allclose(latest_const, [3.14159, -1.414, 9.81, 0.123, -0.456, 0.789], atol=1e-4)

    # ── TEST H: AI Receptive Field Validation (5.0 Seconds Physical Context) ──
    engine = NavigationEngine()
    # Feed 250 frames at 50 Hz (5.0 physical seconds)
    for i in range(250):
        t = i * 0.020
        imu = SensorInputFrame(
            timestamp=t,
            acc_x=1.5,
            acc_y=0.0,
            acc_z=9.81,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.05,
        )
        engine.process_frame(imu, None)

    # Check that 50 samples are in the AI buffer spanning 5.0 seconds
    assert len(engine.imu_window_buffer) == 50
    assert engine.ai_resampler.is_ready is True

    # ── TEST I: Vehicle-Frame Channel Contract Preservation ──
    buf = engine.imu_window_buffer[-1]
    assert len(buf) == 6
    assert isinstance(buf, np.ndarray)
    assert buf.dtype == np.float32

    # ── TEST J: Live VelocityEstimatorNet Inference with 50 Hz Streaming ──
    assert np.isfinite(engine.latest_ai_speed)
    assert engine.latest_ai_speed >= 0.0



