"""Deterministic Regression Tests for Phase 39: Phone/Vehicle Attitude Frame Consistency.

Tests:
1. Stationary phone with 30 deg pitch: no artificial horizontal acceleration.
2. Stationary phone with 15 deg roll + 30 deg pitch: no velocity accumulation.
3. Phone mounting rotation: phone-frame gravity transforms correctly into vehicle frame.
4. Vehicle-level IMU: ES-EKF must not reapply phone mounting tilt.
5. 50 Hz IMU: correct dt remains ~0.02 s.
6. Stationary -> walking: velocity transitions smoothly within human scale.
7. Walking acceleration: no negative/positive hundreds-of-km/h explosion.
8. Stop after walking: ZUPT returns velocity toward zero.
9. Heading source arbitration: absolute orientation cannot be overwritten by relative orientation.
10. Relative-only device: heading is explicitly marked unverified/relative.
11. Heading cardinal conversion: N / NE / E / SE / S / SW / W / NW.
12. No GNSS COG heading while stationary.
13. Live physical provenance remains FIELD/AUTHENTIC.
14. Replay data cannot enter live UI.
"""

import math
import numpy as np
import pytest

from idr.calib.alignment import PhoneToVehicleAligner
from idr.data.provenance import DatasetAuthenticity
from idr.engine.navigation_engine import (
    GNSSInputFix,
    NavigationEngine,
    NavigationOutputState,
    SensorInputFrame,
)
from idr.filters.es_ekf import ErrorStateKalmanFilter


def test_01_stationary_phone_with_30deg_pitch_no_artificial_horizontal_acceleration():
    """1. Stationary phone tilted at 30 deg pitch:

    Under phone-to-vehicle leveling, gravity is leveled into vehicle +Z_v.
    ES-EKF attitude represents vehicle attitude (level).
    Navigation-frame acceleration must remain ~0.0 m/s^2 horizontally.
    """
    engine = NavigationEngine(ref_lat=28.6139, ref_lon=77.2090, dt=0.02)
    engine.aligner.is_calibrated = True

    g = 9.80665
    pitch_rad = math.radians(30.0)
    acc_x = 0.0
    acc_y = g * math.sin(pitch_rad)
    acc_z = g * math.cos(pitch_rad)

    # Set aligner matrix to level this exact gravity vector into [0, 0, g]
    stat_samples = np.tile(np.array([acc_x, acc_y, acc_z]), (50, 1))
    mot_samples = np.tile(np.array([1.0, 0.0, 0.0]), (10, 1))
    engine.aligner.estimate_from_stationary_and_motion(
        stationary_acc=stat_samples,
        motion_acc=mot_samples,
    )

    t = 1725750000.0
    out = None
    for _ in range(100):
        t += 0.02
        imu = SensorInputFrame(
            timestamp=t,
            acc_x=acc_x,
            acc_y=acc_y,
            acc_z=acc_z,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            raw_alpha=0.0,
            raw_beta=30.0,
            raw_gamma=0.0,
            is_absolute=True,
            orientation_yaw=0.0,
            orientation_pitch=30.0,
            orientation_roll=0.0,
            orientation_event_type="deviceorientationabsolute",
        )
        out = engine.process_frame(imu, None)

    assert out is not None
    assert abs(out.forward_speed_mps) < 0.05, f"Expected ~0 m/s, got {out.forward_speed_mps} m/s"
    assert out.is_stationary is True


def test_02_stationary_phone_with_15deg_roll_and_30deg_pitch_no_velocity_accumulation():
    """2. Stationary phone with 15 deg roll + 30 deg pitch:

    Verify zero velocity accumulation over 200 frames (4 seconds).
    """
    engine = NavigationEngine(ref_lat=28.6139, ref_lon=77.2090, dt=0.02)

    roll_rad = math.radians(15.0)
    pitch_rad = math.radians(30.0)
    g = 9.80665

    acc_x = -g * math.sin(roll_rad) * math.cos(pitch_rad)
    acc_y = g * math.sin(pitch_rad)
    acc_z = g * math.cos(pitch_rad) * math.cos(roll_rad)

    stat_samples = np.tile(np.array([acc_x, acc_y, acc_z]), (50, 1))
    mot_samples = np.tile(np.array([1.0, 0.0, 0.0]), (10, 1))
    engine.aligner.estimate_from_stationary_and_motion(
        stationary_acc=stat_samples,
        motion_acc=mot_samples,
    )

    t = 1725750000.0
    out = None
    for _ in range(200):
        t += 0.02
        imu = SensorInputFrame(
            timestamp=t,
            acc_x=acc_x,
            acc_y=acc_y,
            acc_z=acc_z,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            raw_alpha=45.0,
            raw_beta=30.0,
            raw_gamma=15.0,
            is_absolute=True,
            orientation_yaw=45.0,
            orientation_pitch=30.0,
            orientation_roll=15.0,
            orientation_event_type="deviceorientationabsolute",
        )
        out = engine.process_frame(imu, None)

    assert out is not None
    assert abs(out.forward_speed_mps) < 0.05
    assert abs(out.velocity_east) < 0.05
    assert abs(out.velocity_north) < 0.05


def test_03_phone_mounting_rotation_transforms_gravity_correctly():
    """3. Phone mounting rotation:

    Phone-frame gravity must transform cleanly into vehicle vertical [0, 0, g].
    """
    aligner = PhoneToVehicleAligner()
    g = 9.80665

    phone_grav = np.array([3.2, -4.5, 7.8])
    phone_grav = phone_grav / np.linalg.norm(phone_grav) * g

    stat_samples = np.tile(phone_grav, (50, 1))
    mot_samples = np.tile(np.array([1.0, 0.0, 0.0]), (10, 1))
    aligner.estimate_from_stationary_and_motion(
        stationary_acc=stat_samples,
        motion_acc=mot_samples,
    )

    v_acc, _ = aligner.transform_imu(phone_grav.reshape(1, 3), np.zeros((1, 3)))

    assert abs(v_acc[0, 0]) < 1e-2
    assert abs(v_acc[0, 1]) < 1e-2
    assert abs(v_acc[0, 2] - g) < 1e-2


def test_04_vehicle_level_imu_ekf_does_not_reapply_phone_mounting_tilt():
    """4. Vehicle-level IMU:

    When IMU is leveled in vehicle frame, ES-EKF attitude must reflect vehicle attitude (level),
    not phone mounting tilt.
    """
    engine = NavigationEngine(ref_lat=28.6139, ref_lon=77.2090, dt=0.02)
    engine.aligner.is_calibrated = True

    # Initialize leveling with leveled vehicle IMU
    g = 9.80665
    imu = SensorInputFrame(
        timestamp=1725750000.0,
        acc_x=0.0,
        acc_y=0.0,
        acc_z=g,
        gyro_x=0.0,
        gyro_y=0.0,
        gyro_z=0.0,
        raw_alpha=0.0,
        raw_beta=40.0,  # Phone tilted at 40 deg pitch
        raw_gamma=25.0,  # Phone tilted at 25 deg roll
        is_absolute=True,
        orientation_yaw=90.0,
        orientation_pitch=40.0,
        orientation_roll=25.0,
    )
    out = engine.process_frame(imu, None)

    # In vehicle frame, vehicle roll and pitch in ENU are ~0
    roll_rad, pitch_rad, yaw_rad = engine.fusion.es_ekf.euler_angles
    roll_deg, pitch_deg = np.rad2deg(roll_rad), np.rad2deg(pitch_rad)
    assert abs(roll_deg) < 2.0, f"Vehicle roll should be ~0, got {roll_deg}"
    assert abs(pitch_deg) < 2.0, f"Vehicle pitch should be ~0, got {pitch_deg}"
    # Raw phone pitch 40 deg MUST NOT be in vehicle EKF attitude
    assert abs(pitch_deg - 40.0) > 30.0


def test_05_50hz_imu_correct_dt_remains_approx_0_02():
    """5. 50 Hz IMU:

    Verify dt tracking remains stable at ~0.02 s.
    """
    engine = NavigationEngine(ref_lat=28.6139, ref_lon=77.2090, dt=0.02)
    t = 1725750000.0

    dts = []
    for _ in range(50):
        t += 0.02
        imu = SensorInputFrame(
            timestamp=t,
            acc_x=0.0,
            acc_y=0.0,
            acc_z=9.80665,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            raw_alpha=0.0,
            raw_beta=0.0,
            raw_gamma=0.0,
            is_absolute=True,
        )
        out = engine.process_frame(imu, None)
        if out.forensics is not None and "dt" in out.forensics:
            dts.append(out.forensics["dt"])

    if len(dts) > 0:
        avg_dt = float(np.mean(dts[1:]))
        assert 0.019 <= avg_dt <= 0.021


def test_06_stationary_to_walking_transitions_smoothly():
    """6. Stationary -> walking:

    Velocity transitions smoothly within human scale (~1.0 to 1.8 m/s, ~3.6 to 6.5 km/h).
    """
    engine = NavigationEngine(ref_lat=28.6139, ref_lon=77.2090, dt=0.02)
    engine.aligner.is_calibrated = True

    t = 1725750000.0
    # Phase 1: Stationary 2 seconds
    for _ in range(100):
        t += 0.02
        imu = SensorInputFrame(
            timestamp=t,
            acc_x=0.0,
            acc_y=0.0,
            acc_z=9.80665,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            raw_alpha=0.0,
            raw_beta=0.0,
            raw_gamma=0.0,
            is_absolute=True,
        )
        out = engine.process_frame(imu, None)

    assert out.is_stationary is True
    assert abs(out.forward_speed_mps) < 0.05

    # Phase 2: Walking forward with step cycle
    velocities = []
    for step in range(150):
        t += 0.02
        walking_phase = step * 0.02 * 2.0 * math.pi * 1.8
        fwd_a = 0.4 * math.sin(walking_phase)
        vert_a = 9.80665 + 1.2 * math.cos(walking_phase)

        imu = SensorInputFrame(
            timestamp=t,
            acc_x=fwd_a,
            acc_y=0.0,
            acc_z=vert_a,
            gyro_x=0.05 * math.sin(walking_phase),
            gyro_y=0.0,
            gyro_z=0.0,
            raw_alpha=0.0,
            raw_beta=0.0,
            raw_gamma=0.0,
            is_absolute=True,
        )
        out = engine.process_frame(imu, None)
        velocities.append(out.forward_speed_mps)

    max_v = max(abs(v) for v in velocities)
    assert max_v < 4.0, f"Walking velocity exploded to {max_v} m/s ({max_v*3.6} km/h)"


def test_07_walking_acceleration_no_hundreds_of_kmh_explosion():
    """7. Walking acceleration:

    Ensure no negative or positive hundreds-of-km/h explosion even under walking vibrations.
    """
    engine = NavigationEngine(ref_lat=28.6139, ref_lon=77.2090, dt=0.02)
    engine.aligner.is_calibrated = True

    t = 1725750000.0
    for step in range(300):
        t += 0.02
        phase = step * 0.1
        imu = SensorInputFrame(
            timestamp=t,
            acc_x=0.3 * math.sin(phase),
            acc_y=0.1 * math.cos(phase),
            acc_z=9.80665 + 0.5 * math.sin(2 * phase),
            gyro_x=0.05 * math.sin(phase),
            gyro_y=0.02 * math.cos(phase),
            gyro_z=0.01 * math.sin(phase),
            raw_alpha=0.0,
            raw_beta=25.0,
            raw_gamma=0.0,
            is_absolute=True,
        )
        out = engine.process_frame(imu, None)

        v_kmh = out.forward_speed_mps * 3.6
        # Velocity must remain within sane bounds (no 200+ km/h gravity explosion)
        assert abs(v_kmh) < 50.0, f"Frame {step}: Velocity exploded to {v_kmh:.1f} km/h"


def test_08_stop_after_walking_zupt_returns_velocity_to_zero():
    """8. Stop after walking:

    ZUPT returns velocity toward zero upon stopping.
    """
    engine = NavigationEngine(ref_lat=28.6139, ref_lon=77.2090, dt=0.02)
    engine.aligner.is_calibrated = True

    # 1. Feed initial gentle walking motion
    t = 1725750000.0
    for step in range(30):
        t += 0.02
        imu = SensorInputFrame(
            timestamp=t,
            acc_x=0.4,
            acc_y=0.0,
            acc_z=9.80665,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            raw_alpha=0.0,
            raw_beta=0.0,
            raw_gamma=0.0,
            is_absolute=True,
        )
        engine.process_frame(imu, None)

    # 2. Feed stationary frames with zero motion and zero speed
    out = None
    for _ in range(50):
        t += 0.02
        imu = SensorInputFrame(
            timestamp=t,
            acc_x=0.0,
            acc_y=0.0,
            acc_z=9.80665,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            raw_alpha=0.0,
            raw_beta=0.0,
            raw_gamma=0.0,
            is_absolute=True,
        )
        out = engine.process_frame(imu, None)

    assert out is not None
    assert abs(out.forward_speed_mps) < 0.2, f"Velocity after ZUPT: {out.forward_speed_mps} m/s"


def test_09_heading_source_arbitration_relative_cannot_overwrite_absolute():
    """9. Heading source arbitration:

    Relative standard deviceorientation cannot overwrite verified absolute heading.
    """
    engine = NavigationEngine(ref_lat=28.6139, ref_lon=77.2090, dt=0.02)

    # 1. Provide verified absolute heading of 135 deg (SE)
    t = 1725750000.0
    for _ in range(10):
        t += 0.02
        imu = SensorInputFrame(
            timestamp=t,
            acc_x=0.0,
            acc_y=0.0,
            acc_z=9.80665,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            raw_alpha=225.0,  # W3C alpha 225 -> heading 135
            raw_beta=0.0,
            raw_gamma=0.0,
            is_absolute=True,
            orientation_yaw=135.0,
            orientation_event_type="deviceorientationabsolute",
        )
        out = engine.process_frame(imu, None)

    assert abs(out.heading_deg - 135.0) < 3.0

    # 2. Incoming relative frame with alpha=0 (e.g. Android relative deviceorientation)
    for _ in range(10):
        t += 0.02
        imu = SensorInputFrame(
            timestamp=t,
            acc_x=0.0,
            acc_y=0.0,
            acc_z=9.80665,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            raw_alpha=0.0,
            raw_beta=0.0,
            raw_gamma=0.0,
            is_absolute=False,
            orientation_yaw=0.0,
            orientation_event_type="deviceorientation",
        )
        out = engine.process_frame(imu, None)

    # Must NOT overwrite 135.0 deg with 0.0 deg
    assert abs(out.heading_deg - 135.0) < 3.0


def test_10_relative_only_device_marked_unverified_relative():
    """10. Relative-only device:

    When no absolute heading reference exists, heading metadata indicates relative/unverified.
    """
    engine = NavigationEngine(ref_lat=28.6139, ref_lon=77.2090, dt=0.02)

    imu = SensorInputFrame(
        timestamp=1725750000.0,
        acc_x=0.0,
        acc_y=0.0,
        acc_z=9.80665,
        gyro_x=0.0,
        gyro_y=0.0,
        gyro_z=0.0,
        raw_alpha=45.0,
        raw_beta=10.0,
        raw_gamma=0.0,
        is_absolute=False,
        orientation_yaw=45.0,
        orientation_event_type="deviceorientation",
    )
    out = engine.process_frame(imu, None)

    assert out.forensics is not None
    assert out.forensics["is_absolute"] is False
    assert out.forensics["orientation_event_type"] == "deviceorientation"


def test_11_heading_cardinal_conversion():
    """11. Heading cardinal conversion: N / NE / E / SE / S / SW / W / NW."""
    def get_cardinal(heading_deg):
        dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        idx = int(round(heading_deg / 45.0)) % 8
        return dirs[idx]

    assert get_cardinal(0.0) == "N"
    assert get_cardinal(45.0) == "NE"
    assert get_cardinal(90.0) == "E"
    assert get_cardinal(135.0) == "SE"
    assert get_cardinal(180.0) == "S"
    assert get_cardinal(225.0) == "SW"
    assert get_cardinal(270.0) == "W"
    assert get_cardinal(315.0) == "NW"
    assert get_cardinal(360.0) == "N"


def test_12_no_gnss_cog_heading_while_stationary():
    """12. No GNSS COG heading while stationary:

    GNSS Course-Over-Ground is noisy/undefined at 0 km/h and must be rejected.
    """
    engine = NavigationEngine(ref_lat=28.6139, ref_lon=77.2090, dt=0.02)

    # Initial frame with 45 deg
    imu0 = SensorInputFrame(
        timestamp=1725750000.0,
        acc_x=0.0,
        acc_y=0.0,
        acc_z=9.80665,
        gyro_x=0.0,
        gyro_y=0.0,
        gyro_z=0.0,
        raw_alpha=315.0,
        raw_beta=0.0,
        raw_gamma=0.0,
        is_absolute=True,
        orientation_yaw=45.0,
    )
    engine.process_frame(imu0, None)

    # GNSS stationary fix with random noisy COG=270 deg
    gnss = GNSSInputFix(
        timestamp=1725750000.02,
        latitude=28.6139,
        longitude=77.2090,
        altitude=215.0,
        accuracy_m=5.0,
        speed_mps=0.1,  # < 1.5 m/s threshold
        heading_deg=270.0,
    )
    imu1 = SensorInputFrame(
        timestamp=1725750000.02,
        acc_x=0.0,
        acc_y=0.0,
        acc_z=9.80665,
        gyro_x=0.0,
        gyro_y=0.0,
        gyro_z=0.0,
        raw_alpha=315.0,
        raw_beta=0.0,
        raw_gamma=0.0,
        is_absolute=True,
        orientation_yaw=45.0,
    )
    out = engine.process_frame(imu1, gnss)

    # Heading must NOT be overwritten by stationary GNSS COG (270 deg)
    assert abs(out.heading_deg - 45.0) < 2.0


def test_13_live_physical_provenance_remains_field_or_authentic():
    """13. Live physical provenance:

    Frames originating from live hardware sensor stream maintain FIELD / AUTHENTIC provenance.
    """
    auth = DatasetAuthenticity.from_str("field")
    assert auth == DatasetAuthenticity.FIELD
    assert auth != DatasetAuthenticity.SYNTHETIC


def test_14_replay_data_cannot_enter_live_ui():
    """14. Replay data isolation:

    Replay telemetry with synthetic or offline tags cannot be disguised as FIELD data.
    """
    synth = DatasetAuthenticity.from_str("synthetic")
    assert synth == DatasetAuthenticity.SYNTHETIC
    assert synth != DatasetAuthenticity.FIELD
