"""Tests for Live GNSS Trust Stability, Asynchronous Geolocation, and Stationary Invariance.

Validates:
1. Asynchronous mobile browser Geolocation (1 fix every 3.5s - 12s) while stationary
   does NOT flap to blackout or reacquiring mode.
2. Cold start before first GNSS fix remains in standby / acquiring mode without false blackout.
3. Moving vehicle in tunnel (>6.0s outage while moving) transitions cleanly to blackout.
4. Recovery from blackout restores trusted navigation.
"""

import numpy as np
import pytest
from idr.engine.navigation_engine import (
    NavigationEngine,
    SensorInputFrame,
    GNSSInputFix,
)
from idr.engine.gnss_trust import GNSSTrustStatus


def test_cold_start_before_first_gnss_no_false_blackout():
    """Verify that when navigation starts before the first GPS fix arrives (0-2s),
    the engine does NOT prematurely enter blackout outage mode or log false blackspots.
    """
    engine = NavigationEngine(ref_lat=12.9716, ref_lon=77.5946, vehicle_type="two_wheeler")

    # Stream 20 frames (2 seconds) of pure IMU before any GPS fix arrives
    for i in range(20):
        t = i * 0.1
        imu = SensorInputFrame(timestamp=t, acc_x=0.0, acc_y=0.0, acc_z=9.81, gyro_x=0.0, gyro_y=0.0, gyro_z=0.0)
        res = engine.process_frame(imu, None)

        assert res.is_in_blackout is False, f"False blackout declared at startup t={t}s"
        assert engine.in_blackout is False
        assert len(engine.blackspot_tracker.get_all_records()) == 0

    # First GPS fix arrives at t=2.0s
    gnss_first = GNSSInputFix(
        timestamp=2.0, latitude=12.9716, longitude=77.5946, altitude=920.0,
        accuracy_m=10.0, speed_mps=0.0, heading_deg=None
    )
    res_first = engine.process_frame(SensorInputFrame(2.0, 0, 0, 9.81, 0, 0, 0), gnss_first)
    assert res_first.gnss_status in (GNSSTrustStatus.TRUSTED.value, "RECOVERING")
    # Should not trigger reacquisition smoothing because we were never in blackout
    assert res_first.nav_mode.value != "REACQUISITION_SMOOTHING"


def test_asynchronous_browser_geolocation_stationary_no_flapping():
    """Verify that a stationary phone sending 10 Hz IMU with GNSS arriving every 3.5s
    remains continuously in TRUSTED / GNSS_INS_FULL without flapping to BLACKOUT.
    """
    engine = NavigationEngine(ref_lat=12.9716, ref_lon=77.5946, vehicle_type="two_wheeler")

    # Initial fix at t=0
    gnss_init = GNSSInputFix(
        timestamp=100.0,
        latitude=12.9716,
        longitude=77.5946,
        altitude=920.0,
        speed_mps=0.02,
        heading_deg=None,
        accuracy_m=12.0,
    )
    imu_0 = SensorInputFrame(
        timestamp=100.0,
        acc_x=0.0,
        acc_y=0.0,
        acc_z=9.81,
        gyro_x=0.0,
        gyro_y=0.0,
        gyro_z=0.0,
    )
    res_0 = engine.process_frame(imu_0, gnss_init)
    assert res_0.gnss_status in (GNSSTrustStatus.TRUSTED.value, "RECOVERING")

    # Simulate 30 seconds at 10 Hz (300 steps), with GNSS fix arriving only every 35 steps (3.5s)
    state_transitions = []
    prev_status = res_0.gnss_status

    current_time = 100.0
    for i in range(1, 300):
        current_time += 0.1
        is_gps_step = (i % 35 == 0)
        
        gnss_frame = None
        if is_gps_step:
            jitter_lat = 12.9716 + float(np.random.uniform(-0.00001, 0.00001))
            jitter_lon = 77.5946 + float(np.random.uniform(-0.00001, 0.00001))
            gnss_frame = GNSSInputFix(
                timestamp=current_time,
                latitude=jitter_lat,
                longitude=jitter_lon,
                altitude=920.0,
                speed_mps=float(np.random.uniform(0.0, 0.15)),
                heading_deg=None,
                accuracy_m=float(np.random.uniform(10.0, 28.0)),
            )

        frame = SensorInputFrame(
            timestamp=current_time,
            acc_x=float(np.random.normal(0, 0.02)),
            acc_y=float(np.random.normal(0, 0.02)),
            acc_z=float(9.81 + np.random.normal(0, 0.02)),
            gyro_x=float(np.random.normal(0, 0.002)),
            gyro_y=float(np.random.normal(0, 0.002)),
            gyro_z=float(np.random.normal(0, 0.002)),
        )

        res = engine.process_frame(frame, gnss_frame)
        
        if res.gnss_status != prev_status:
            state_transitions.append((current_time, prev_status, res.gnss_status))
            prev_status = res.gnss_status

    blackout_transitions = [t for t in state_transitions if t[2] == GNSSTrustStatus.BLACKOUT.value]
    assert len(blackout_transitions) == 0, f"Unexpected blackout transitions: {state_transitions}"
    assert res.forward_speed_mps < 0.2, f"Stationary speed drifted: {res.forward_speed_mps}"


def test_motion_gnss_loss_transitions_cleanly_after_timeout():
    """Verify that if GNSS stops arriving while the vehicle is in motion,
    the state machine transitions to BLACKOUT after gnss_stale_timeout_sec (6.0s).
    """
    engine = NavigationEngine(ref_lat=12.9716, ref_lon=77.5946, vehicle_type="two_wheeler", gnss_stale_timeout_sec=6.0)

    # Initial fix at t=100.0 with forward motion (speed = 10 m/s = 36 km/h)
    gnss_init = GNSSInputFix(
        timestamp=100.0, latitude=12.9716, longitude=77.5946, altitude=920.0,
        speed_mps=10.0, heading_deg=90.0, accuracy_m=5.0,
    )
    # Accelerometer shows dynamic motion (accel + vibration)
    frame_0 = SensorInputFrame(timestamp=100.0, acc_x=1.5, acc_y=0.0, acc_z=9.81, gyro_x=0.05, gyro_y=0.02, gyro_z=0.01)
    engine.process_frame(frame_0, gnss_init)

    current_time = 100.0
    blackout_detected_at = None

    # Run for 10 seconds of continuous motion without GNSS
    for i in range(1, 100):
        current_time += 0.1
        frame = SensorInputFrame(
            timestamp=current_time,
            acc_x=float(1.5 + np.random.normal(0, 0.2)), # In motion vibration
            acc_y=float(np.random.normal(0, 0.2)),
            acc_z=float(9.81 + np.random.normal(0, 0.2)),
            gyro_x=float(np.random.normal(0, 0.05)),
            gyro_y=float(np.random.normal(0, 0.05)),
            gyro_z=float(np.random.normal(0, 0.05)),
        )
        res = engine.process_frame(frame, None)

        if res.gnss_status == GNSSTrustStatus.BLACKOUT.value and blackout_detected_at is None:
            blackout_detected_at = current_time

    assert blackout_detected_at is not None, "Blackout was never detected during in-motion GNSS outage"
    elapsed_to_blackout = blackout_detected_at - 100.0
    assert 5.9 <= elapsed_to_blackout <= 6.2, f"Blackout trigger timing off: {elapsed_to_blackout}s"
    assert res.is_in_blackout is True


def test_motion_gnss_recovery_after_blackout():
    """Verify that when GNSS reappears after an in-motion blackout, it triggers reacquisition smoothing."""
    engine = NavigationEngine(ref_lat=12.9716, ref_lon=77.5946, vehicle_type="two_wheeler", gnss_stale_timeout_sec=6.0)

    gnss_init = GNSSInputFix(
        timestamp=100.0, latitude=12.9716, longitude=77.5946, altitude=920.0,
        speed_mps=10.0, heading_deg=90.0, accuracy_m=5.0,
    )
    engine.process_frame(SensorInputFrame(100.0, 1.5, 0, 9.81, 0.05, 0.02, 0.01), gnss_init)

    # 8 seconds of in-motion outage
    current_time = 100.0
    for _ in range(80):
        current_time += 0.1
        frame = SensorInputFrame(current_time, float(1.5 + np.random.normal(0, 0.2)), 0.0, 9.81, 0.05, 0.0, 0.0)
        engine.process_frame(frame, None)

    assert engine.in_blackout is True

    # Recovery fix arrives near the dead-reckoned position
    cur_lat, cur_lon = engine.fusion.enu_to_latlon(engine.fusion.position_enu[0] + 2.0, engine.fusion.position_enu[1])
    gnss_recovery = GNSSInputFix(
        timestamp=current_time, latitude=cur_lat, longitude=cur_lon, altitude=920.0,
        speed_mps=12.0, heading_deg=90.0, accuracy_m=10.0,
    )
    res_rec = engine.process_frame(SensorInputFrame(current_time, 0.0, 0, 9.81, 0, 0, 0), gnss_recovery)

    assert res_rec.gnss_status in (GNSSTrustStatus.TRUSTED.value, "RECOVERING")
    assert engine.in_blackout is False
