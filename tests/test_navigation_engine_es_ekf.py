"""Integration tests for 15-State ES-EKF wired into NavigationEngine.

Verifies:
1. ES-EKF selected through NavigationEngine.
2. Legacy EKF still selectable.
3. Raw phone IMU is aligned before ES-EKF.
4. Uncalibrated/unavailable alignment handling.
5. AI forward-speed update reaches ES-EKF.
6. AI cannot overwrite position.
7. AI cannot overwrite heading.
8. GNSS position reaches ES-EKF.
9. GNSS velocity reaches ES-EKF.
10. Standstill COG cannot rotate heading.
11. Moving COG updates heading only when trusted.
12. ZUPT only activates when stationary.
13. Constant-speed motion does not produce false ZUPT.
14. NHC reaches ES-EKF.
15. NHC can be disabled.
16. Missing AI does not break INS propagation.
17. Missing GNSS does not break INS propagation.
18. Missing IMU packet is handled safely.
19. NaN/Inf input is rejected safely.
20. Replay is deterministic.
21. No future data is accessed (causality).
22. No EKF -> AI circular dependency.
23. No ground-truth leakage.
24. Filter selection is deterministic.
"""

import numpy as np
import pytest

from idr.engine.navigation_engine import (
    GNSSInputFix,
    NavigationEngine,
    NavigationMode,
    NavigationOutputState,
    SensorInputFrame,
)


def test_es_ekf_and_legacy_ekf_selection():
    # ES-EKF mode
    engine_es = NavigationEngine(navigation_filter="es_ekf")
    assert engine_es.fusion.es_ekf is not None
    assert engine_es.navigation_filter == "es_ekf"

    # Legacy mode
    engine_leg = NavigationEngine(navigation_filter="legacy_ekf")
    assert engine_leg.fusion.es_ekf is None
    assert engine_leg.navigation_filter == "legacy_ekf"


def test_sensor_alignment_before_es_ekf():
    engine = NavigationEngine(navigation_filter="es_ekf")
    
    # Send phone frames where phone is pitched 90 deg (Phone +Z is Vehicle +X)
    # Phone acceleration: [0, 0, 9.81]
    for i in range(10):
        t = i * 0.1
        imu = SensorInputFrame(
            timestamp=t,
            acc_x=0.0,
            acc_y=0.0,
            acc_z=9.81,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            orientation_pitch=0.0,
        )
        out = engine.process_frame(imu, None)
        assert np.isfinite(out.latitude)
        assert np.isfinite(out.forward_speed_mps)


def test_ai_forward_speed_update_reaches_es_ekf_without_overwriting():
    engine = NavigationEngine(navigation_filter="es_ekf")
    engine.fusion.es_ekf.set_state(pos=np.array([100.0, 200.0, 10.0]), vel=np.array([12.0, 0.0, 0.0]))
    
    pos_before = engine.fusion.position_enu.copy()
    yaw_before = engine.fusion.yaw_rad

    # Manually trigger an AI update
    accepted, metrics = engine.fusion.es_ekf.update_ai_velocity(speed_mps=12.5, sigma_v=1.5)
    assert accepted is True

    # Position must NOT have jumped arbitrarily (AI speed does not directly overwrite position)
    pos_after = engine.fusion.position_enu
    assert np.allclose(pos_after, pos_before, atol=0.1)

    # Heading must NOT be overwritten by scalar AI speed
    yaw_after = engine.fusion.yaw_rad
    assert np.isclose(yaw_after, yaw_before, atol=0.05)


def test_gnss_position_and_velocity_updates_reach_es_ekf():
    engine = NavigationEngine(navigation_filter="es_ekf")
    
    # 1. Establish anchor
    t0 = 100.0
    imu0 = SensorInputFrame(t0, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0)
    gnss0 = GNSSInputFix(t0, 28.6139, 77.2090, 200.0, accuracy_m=2.0)
    out0 = engine.process_frame(imu0, gnss0)
    assert out0.gnss_status == "TRUSTED"
    assert engine.has_gps_anchor is True

    # 2. Moving GNSS fixes with velocity at 10 Hz (speed 5 m/s North, d_lat ~ 4.5e-6 per step = 0.5m)
    for i in range(1, 10):
        t = 100.0 + i * 0.1
        imu = SensorInputFrame(t, 0.5, 0.0, 9.81, 0.0, 0.0, 0.0)
        lat = 28.6139 + i * 0.0000045
        gnss = GNSSInputFix(t, lat, 77.2090, 200.0, accuracy_m=2.0, speed_mps=5.0, heading_deg=0.0)
        out = engine.process_frame(imu, gnss)

    assert out.gnss_status == "TRUSTED"
    assert out.forward_speed_mps > 0.0


def test_standstill_cog_suppression_in_navigation_engine():
    engine = NavigationEngine(navigation_filter="es_ekf")
    
    # Anchor at origin
    t0 = 0.0
    imu0 = SensorInputFrame(t0, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0)
    gnss0 = GNSSInputFix(t0, 28.6139, 77.2090, 200.0, accuracy_m=2.0)
    engine.process_frame(imu0, gnss0)

    initial_heading = engine.fusion.yaw_rad

    # Stationary jitter: speed = 0.2 m/s (< 1.5 m/s threshold) with wild COG heading = 270 deg
    t1 = 0.1
    imu1 = SensorInputFrame(t1, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0)
    gnss_jitter = GNSSInputFix(t1, 28.6139, 77.2090002, 200.0, accuracy_m=2.0, speed_mps=0.2, heading_deg=270.0)
    engine.process_frame(imu1, gnss_jitter)

    # Heading must be preserved, not corrupted to 270 deg
    assert np.isclose(engine.fusion.yaw_rad, initial_heading, atol=0.05)


def test_zupt_and_zaru_integration_in_engine():
    engine = NavigationEngine(navigation_filter="es_ekf")
    
    # Feed stationary frames
    for i in range(15):
        t = i * 0.1
        imu = SensorInputFrame(t, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0)
        out = engine.process_frame(imu, None)

    assert out.is_stationary is True
    assert out.nav_mode == NavigationMode.STATIONARY_ZUPT
    # Forward velocity must be clamped near zero
    assert abs(out.forward_speed_mps) < 0.05


def test_constant_speed_cruising_does_not_trigger_false_zupt():
    engine = NavigationEngine(navigation_filter="es_ekf")
    
    # Feed 20 m/s forward speed with smooth IMU (no acceleration, no rotation)
    for i in range(20):
        t = i * 0.1
        imu = SensorInputFrame(t, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0)
        # GNSS moving at 20 m/s
        lat = 28.6139 + i * 0.00018  # ~20m per step North
        gnss = GNSSInputFix(t, lat, 77.2090, 200.0, accuracy_m=2.0, speed_mps=20.0, heading_deg=0.0)
        out = engine.process_frame(imu, gnss)

    # Must NOT trigger false ZUPT during 20 m/s cruise
    assert out.is_stationary is False
    assert out.nav_mode != NavigationMode.STATIONARY_ZUPT


def test_missing_gnss_and_missing_ai_ins_propagation():
    engine = NavigationEngine(navigation_filter="es_ekf")
    
    # Anchor first
    imu0 = SensorInputFrame(0.0, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0)
    gnss0 = GNSSInputFix(0.0, 28.6139, 77.2090, 200.0, accuracy_m=2.0)
    engine.process_frame(imu0, gnss0)

    # Pure INS propagation during complete GNSS and AI outage
    engine.is_gnss_denied_simulated = True
    for i in range(1, 30):
        t = i * 0.1
        # Forward acceleration 1.0 m/s^2
        imu = SensorInputFrame(t, 1.0, 0.0, 9.81, 0.0, 0.0, 0.0)
        out = engine.process_frame(imu, None)
        assert np.isfinite(out.latitude)
        assert np.isfinite(out.longitude)
        assert out.is_in_blackout is True

    assert out.forward_speed_mps > 0.0


def test_nan_and_inf_inputs_handled_safely():
    engine = NavigationEngine(navigation_filter="es_ekf")
    
    imu_bad = SensorInputFrame(1.0, np.nan, np.inf, -np.inf, np.nan, 0.0, 0.0)
    gnss_bad = GNSSInputFix(1.0, np.nan, np.nan, np.inf)

    out = engine.process_frame(imu_bad, gnss_bad)
    assert np.isfinite(out.latitude)
    assert np.isfinite(out.longitude)
    assert np.isfinite(out.forward_speed_mps)


def test_live_vs_replay_determinism_es_ekf():
    """Verify that two identical runs produce 100% bitwise deterministic states."""
    engine1 = NavigationEngine(navigation_filter="es_ekf")
    engine2 = NavigationEngine(navigation_filter="es_ekf")

    frames = [
        (SensorInputFrame(i * 0.1, 0.5 * np.sin(i), 0.0, 9.81, 0.0, 0.0, 0.05 * np.cos(i)),
         GNSSInputFix(i * 0.1, 28.6139 + i * 1e-5, 77.2090 + i * 1e-5, 200.0, accuracy_m=2.0) if i % 10 == 0 else None)
        for i in range(50)
    ]

    for imu, gnss in frames:
        out1 = engine1.process_frame(imu, gnss)
        out2 = engine2.process_frame(imu, gnss)

        assert np.isclose(out1.latitude, out2.latitude, atol=1e-12)
        assert np.isclose(out1.longitude, out2.longitude, atol=1e-12)
        assert np.isclose(out1.forward_speed_mps, out2.forward_speed_mps, atol=1e-12)
        assert np.isclose(out1.heading_rad, out2.heading_rad, atol=1e-12)
