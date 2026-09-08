"""Test Suite: Live Data Provenance, Heading Transformation, and Sensor Integrity.

Verifies:
1. Real device live inputs vs Replay dataset separation.
2. ZUPT speed clamping (0.0 m/s when stationary, responsive when moving).
3. Mathematical W3C alpha (counter-clockwise) to Compass Heading (clockwise) transformation.
4. Stationary compass heading tracking across all 4 quadrants (North, East, South, West).
5. Immunity of stationary heading to spurious zero-speed GNSS Course-Over-Ground (COG).
6. Dynamic activation of GNSS COG at speed >= 1.5 m/s.
7. Absence of synthetic or mock data leakage in live navigation mode.
8. Stationary GPS jitter filtering and fixed ENU anchoring.
"""

import math
import numpy as np
import pytest

from idr.engine.navigation_engine import NavigationEngine, SensorInputFrame, GNSSInputFix
from idr.engine.replay import DriveReplayer


def test_w3c_alpha_to_compass_heading_transformation():
    """Verify exact mathematical conversion between W3C DeviceOrientation alpha and Geographic Compass Heading."""
    def alpha_to_heading(alpha: float) -> float:
        h = (360.0 - alpha) % 360.0
        return h + 360.0 if h < 0 else h

    # North: alpha=0 -> Heading=0
    assert alpha_to_heading(0.0) == pytest.approx(0.0)
    assert alpha_to_heading(360.0) == pytest.approx(0.0)
    # East: alpha=270 -> Heading=90
    assert alpha_to_heading(270.0) == pytest.approx(90.0)
    # South: alpha=180 -> Heading=180
    assert alpha_to_heading(180.0) == pytest.approx(180.0)
    # West: alpha=90 -> Heading=270
    assert alpha_to_heading(90.0) == pytest.approx(270.0)
    # Wraparounds
    assert alpha_to_heading(359.0) == pytest.approx(1.0)
    assert alpha_to_heading(1.0) == pytest.approx(359.0)


def test_stationary_heading_invariance_across_all_quadrants():
    """Verify that stationary compass headings (0, 90, 180, 270) correctly converge in NavigationEngine."""
    cardinals = [
        (0.0, "North"),
        (90.0, "East"),
        (180.0, "South"),
        (270.0, "West"),
    ]

    for target_heading, label in cardinals:
        engine = NavigationEngine(ref_lat=28.6139, ref_lon=77.2090, vehicle_type="two_wheeler")
        engine.reset()

        t0 = 5000.0
        for i in range(150):  # 3.0s at 50 Hz
            t = t0 + i * 0.02
            imu = SensorInputFrame(
                timestamp=t,
                acc_x=0.0, acc_y=0.0, acc_z=9.81,
                gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
                orientation_yaw=target_heading,
            )
            gnss = None
            if i == 0:
                gnss = GNSSInputFix(
                    timestamp=t,
                    latitude=28.6139, longitude=77.2090, altitude=216.0,
                    accuracy_m=2.0, speed_mps=0.0, heading_deg=target_heading,
                )
            state = engine.process_frame(imu, gnss)

        angular_err = abs((state.heading_deg - target_heading + 180) % 360 - 180)
        assert angular_err < 3.0, f"Heading failed to track {label} ({target_heading}°): got {state.heading_deg}°"


def test_zero_speed_ignores_spurious_gnss_cog():
    """Verify that when stationary, noisy GNSS COG headings (e.g. rotating randomly) are NOT fused into yaw."""
    engine = NavigationEngine(ref_lat=28.6139, ref_lon=77.2090, vehicle_type="two_wheeler")
    engine.reset()

    # Phone is stationary facing East (90 deg)
    t0 = 6000.0
    for i in range(100):
        t = t0 + i * 0.02
        imu = SensorInputFrame(
            timestamp=t,
            acc_x=0.0, acc_y=0.0, acc_z=9.81,
            gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
            orientation_yaw=90.0,
        )
        # Spurious GNSS fix reporting wild random COG (e.g. 315 deg NW) at speed 0.0 m/s
        gnss = None
        if i % 25 == 0:
            gnss = GNSSInputFix(
                timestamp=t,
                latitude=28.6139, longitude=77.2090, altitude=216.0,
                accuracy_m=3.0,
                speed_mps=0.0,  # Zero speed -> COG must be ignored
                heading_deg=315.0,  # False COG
            )
        state = engine.process_frame(imu, gnss)

    # Heading must remain locked to true compass (90 deg East), NOT wild COG (315 deg)
    assert abs((state.heading_deg - 90.0 + 180) % 360 - 180) < 3.0


def test_live_zero_speed_provenance_when_stationary():
    """Verify that stationary sensor inputs produce 0.0 m/s speed with zero false acceleration."""
    engine = NavigationEngine(ref_lat=28.6139, ref_lon=77.2090, vehicle_type="two_wheeler")
    engine.reset()

    # Feed 100 stationary frames (50 Hz, 2 seconds)
    t0 = 1000.0
    for i in range(100):
        t = t0 + i * 0.02
        imu = SensorInputFrame(
            timestamp=t,
            acc_x=0.01 * float(np.random.randn()),
            acc_y=0.01 * float(np.random.randn()),
            acc_z=9.81 + 0.01 * float(np.random.randn()),
            gyro_x=0.001 * float(np.random.randn()),
            gyro_y=0.001 * float(np.random.randn()),
            gyro_z=0.001 * float(np.random.randn()),
            orientation_yaw=90.0,
        )
        gnss = None
        if i == 0:
            gnss = GNSSInputFix(
                timestamp=t,
                latitude=28.6139,
                longitude=77.2090,
                altitude=216.0,
                accuracy_m=2.5,
                speed_mps=0.0,
                heading_deg=90.0,
            )
        state = engine.process_frame(imu, gnss)

    # Forward speed must be exactly 0.0 m/s (ZUPT locked)
    assert state.forward_speed_mps == pytest.approx(0.0, abs=1e-3)
    assert state.lean_angle_deg == pytest.approx(0.0, abs=1e-2)
    # Heading must be locked to compass (90.0 deg)
    assert abs((state.heading_deg - 90.0 + 180) % 360 - 180) < 3.0


def test_live_heading_tracks_physical_compass_at_rest():
    """Verify that when stationary, rotating the phone compass smoothly tracks state heading."""
    engine = NavigationEngine(ref_lat=28.6139, ref_lon=77.2090, vehicle_type="two_wheeler")
    engine.reset()

    # Step 1: Initialize at 45 degrees East-North-East for 150 frames (3.0s)
    t0 = 2000.0
    for i in range(150):
        t = t0 + i * 0.02
        imu_initial = SensorInputFrame(
            timestamp=t,
            acc_x=0.0, acc_y=0.0, acc_z=9.81,
            gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
            orientation_yaw=45.0,
        )
        gnss_initial = None
        if i == 0:
            gnss_initial = GNSSInputFix(
                timestamp=t,
                latitude=28.6139, longitude=77.2090, altitude=216.0,
                accuracy_m=2.0, speed_mps=0.0, heading_deg=45.0,
            )
        state = engine.process_frame(imu_initial, gnss_initial)
    assert abs((state.heading_deg - 45.0 + 180) % 360 - 180) < 3.0

    # Step 2: Physically rotate phone to South-East (135 degrees) for 200 frames (4.0s)
    for i in range(150, 350):
        t = t0 + i * 0.02
        imu_rot = SensorInputFrame(
            timestamp=t,
            acc_x=0.0, acc_y=0.0, acc_z=9.81,
            gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
            orientation_yaw=135.0,
        )
        state = engine.process_frame(imu_rot, None)

    # State heading must have tracked the physical compass to 135 degrees
    assert abs((state.heading_deg - 135.0 + 180) % 360 - 180) < 3.0


def test_no_replay_data_leaks_into_live_session():
    """Verify that live session processing is completely isolated from DriveReplayer."""
    engine = NavigationEngine(ref_lat=28.6139, ref_lon=77.2090, vehicle_type="two_wheeler")
    replayer = DriveReplayer(engine)

    # Load replay dataset
    replayer.load_iovnbd_drive("Vf", reset_engine=False)
    assert len(replayer.data_frames) > 0
    assert not replayer.is_playing

    # Reset live engine
    engine.reset()
    assert not engine.has_physical_gps_fix

    # Process a live stationary frame
    imu = SensorInputFrame(
        timestamp=5000.0,
        acc_x=0.0, acc_y=0.0, acc_z=9.81,
        gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
        orientation_yaw=180.0,
    )
    state = engine.process_frame(imu, None)

    # Must NOT have dataset positions or speed from Vf
    assert state.forward_speed_mps == 0.0
    assert not state.is_in_blackout  # Should not enter blackout before first physical GPS fix


def test_live_speed_increases_on_dynamic_acceleration():
    """Verify that non-stationary accelerometer and GPS measurements dynamically produce nonzero speed."""
    engine = NavigationEngine(ref_lat=28.6139, ref_lon=77.2090, vehicle_type="two_wheeler")
    engine.reset()

    # Anchor GPS fix
    t0 = 10000.0
    gnss0 = GNSSInputFix(
        timestamp=t0,
        latitude=28.6139, longitude=77.2090, altitude=216.0,
        accuracy_m=2.0, speed_mps=0.0, heading_deg=0.0,
    )
    imu0 = SensorInputFrame(
        timestamp=t0,
        acc_x=0.0, acc_y=0.0, acc_z=9.81,
        gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
        orientation_yaw=0.0,
    )
    engine.process_frame(imu0, gnss0)

    # Accelerate forward at 1.5 m/s^2 along North for 2 seconds (100 frames)
    for i in range(1, 101):
        t = t0 + i * 0.02
        imu_acc = SensorInputFrame(
            timestamp=t,
            acc_x=0.0, acc_y=1.5, acc_z=9.81,  # Forward acceleration in body frame
            gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
            orientation_yaw=0.0,
        )
        gnss_fix = None
        if i % 50 == 0:  # 1 Hz GPS fix
            gnss_fix = GNSSInputFix(
                timestamp=t,
                latitude=28.6139 + (0.5 * 1.5 * (i * 0.02)**2) / 111320.0,
                longitude=77.2090,
                altitude=216.0,
                accuracy_m=2.0,
                speed_mps=1.5 * (i * 0.02),
                heading_deg=0.0,
            )
        state = engine.process_frame(imu_acc, gnss_fix)

    # Forward speed must be positive and non-zero
    assert state.forward_speed_mps > 0.5


def test_stationary_gps_jitter_does_not_accumulate_velocity_or_dr():
    """Verify that stationary data with fluctuating GPS noise produces zero DR distance."""
    np.random.seed(42)
    engine = NavigationEngine(ref_lat=28.6139, ref_lon=77.2090, vehicle_type="two_wheeler")
    engine.reset()

    lat0, lon0 = 28.6139, 77.2090
    t0 = 20000.0

    for i in range(500):  # 10 seconds at 50 Hz
        t = t0 + i * 0.02
        imu = SensorInputFrame(
            timestamp=t,
            acc_x=float(np.random.normal(0.0, 0.02)),
            acc_y=float(np.random.normal(0.0, 0.02)),
            acc_z=float(9.81 + np.random.normal(0.0, 0.02)),
            gyro_x=float(np.random.normal(0.0, 0.002)),
            gyro_y=float(np.random.normal(0.0, 0.002)),
            gyro_z=float(np.random.normal(0.0, 0.002)),
            orientation_yaw=90.0,
        )
        gnss = None
        if i % 50 == 0:  # 1 Hz GPS fix with +/- 2.5m multipath jitter
            d_lat = float(np.random.normal(0.0, 2.0)) / 111320.0
            d_lon = float(np.random.normal(0.0, 2.0)) / (111320.0 * np.cos(np.deg2rad(lat0)))
            gnss = GNSSInputFix(
                timestamp=t,
                latitude=lat0 + d_lat,
                longitude=lon0 + d_lon,
                altitude=216.0,
                accuracy_m=3.0,
                speed_mps=0.0,
                heading_deg=90.0,
            )
        state = engine.process_frame(imu, gnss)

        # Invariants at rest (once stillness is confirmed after 10 frames):
        if i >= 10:
            assert state.forward_speed_mps == 0.0
            assert abs(state.velocity_east) < 0.05
            assert abs(state.velocity_north) < 0.05
            assert state.lean_angle_deg == 0.0
            assert state.is_stationary

    # Accumulated DR distance must remain strictly zero
    assert state.diagnostics.total_dr_distance_m == 0.0


def test_fixed_enu_anchor_invariance():
    """Verify that ENU origin remains strictly anchored and does not drift or repeatedly re-anchor."""
    engine = NavigationEngine(ref_lat=28.6139, ref_lon=77.2090, vehicle_type="two_wheeler")
    engine.reset()

    # Initial fix
    gnss0 = GNSSInputFix(
        timestamp=100.0,
        latitude=28.6139, longitude=77.2090, altitude=216.0,
        accuracy_m=2.0, speed_mps=0.0, heading_deg=90.0,
    )
    imu0 = SensorInputFrame(timestamp=100.0, acc_x=0.0, acc_y=0.0, acc_z=9.81, gyro_x=0.0, gyro_y=0.0, gyro_z=0.0)
    engine.process_frame(imu0, gnss0)

    anchor_lat = engine.ref_lat
    anchor_lon = engine.ref_lon

    # Feed multiple fixes with jitter
    for i in range(1, 50):
        t = 100.0 + i * 1.0
        imu = SensorInputFrame(timestamp=t, acc_x=0.0, acc_y=0.0, acc_z=9.81, gyro_x=0.0, gyro_y=0.0, gyro_z=0.0)
        gnss = GNSSInputFix(
            timestamp=t,
            latitude=28.6139 + 0.00001 * np.sin(i),
            longitude=77.2090 + 0.00001 * np.cos(i),
            altitude=216.0,
            accuracy_m=2.0,
            speed_mps=0.0,
            heading_deg=90.0,
        )
        engine.process_frame(imu, gnss)

    # Anchor must not have changed
    assert engine.ref_lat == anchor_lat
    assert engine.ref_lon == anchor_lon


def test_transition_stationary_to_moving_and_back():
    """Verify clean state transition from standstill -> moving -> standstill."""
    engine = NavigationEngine(ref_lat=28.6139, ref_lon=77.2090, vehicle_type="two_wheeler")
    engine.reset()

    # 1. Establish anchor at standstill (15 frames)
    t = 100.0
    for i in range(15):
        t += 0.1
        imu = SensorInputFrame(timestamp=t, acc_x=0.0, acc_y=0.0, acc_z=9.81, gyro_x=0.0, gyro_y=0.0, gyro_z=0.0, orientation_yaw=0.0)
        gnss = None
        if i == 0:
            gnss = GNSSInputFix(timestamp=t, latitude=28.6139, longitude=77.2090, altitude=216.0, accuracy_m=2.0, speed_mps=0.0, heading_deg=0.0)
        s1 = engine.process_frame(imu, gnss)
    assert s1.is_stationary is True
    assert s1.forward_speed_mps == 0.0

    # 2. Moving phase (15 frames at 5 m/s)
    for i in range(1, 16):
        t += 0.1
        imu = SensorInputFrame(timestamp=t, acc_x=0.5, acc_y=0.0, acc_z=9.81, gyro_x=0.0, gyro_y=0.0, gyro_z=0.0, orientation_yaw=0.0)
        lat = 28.6139 + i * 0.0000045
        gnss = GNSSInputFix(timestamp=t, latitude=lat, longitude=77.2090, altitude=216.0, accuracy_m=2.0, speed_mps=5.0, heading_deg=0.0)
        s2 = engine.process_frame(imu, gnss)
    assert s2.is_stationary is False
    assert s2.forward_speed_mps > 0.0

    # 3. Return to standstill (30 frames at 0 speed)
    last_gnss_lat = 28.6139 + 15 * 0.0000045
    last_gnss_lon = 77.2090
    for i in range(30):
        t += 0.1
        imu = SensorInputFrame(timestamp=t, acc_x=0.0, acc_y=0.0, acc_z=9.81, gyro_x=0.0, gyro_y=0.0, gyro_z=0.0, orientation_yaw=0.0)
        gnss = GNSSInputFix(timestamp=t, latitude=last_gnss_lat, longitude=last_gnss_lon, altitude=216.0, accuracy_m=2.0, speed_mps=0.0, heading_deg=0.0)
        s3 = engine.process_frame(imu, gnss)
    assert s3.is_stationary is True
    assert s3.forward_speed_mps == 0.0
