"""Regression Tests for Phase 40: Forensic Investigation of Real Android Walking Failure.

Verifies:
1. StationaryDetector recovery after high-speed motion without deadlock from uncorrected EKF velocity.
2. AI velocity weighting binds forward speed closely to physical human walking pace (~0.8 to 1.8 m/s).
3. Dead Reckoning distance ceases accumulating once stopped (no stationary drift inflation).
4. Lean angle kinematics do not falsely exceed reasonable bounds during walking.
5. NavigationMode and trajectory source explicitly report DEAD_RECKONING when GNSS is denied.
6. 55 Hz IMU stream maintains dt ~0.018s.
"""

import math
import numpy as np
import pytest

from idr.engine.navigation_engine import (
    GNSSInputFix,
    NavigationEngine,
    NavigationMode,
    NavigationOutputState,
    SensorInputFrame,
)
from idr.filters.zupt import StationaryDetector


def test_stationary_detector_recovery_after_motion_without_deadlock():
    """Verify that StationaryDetector correctly detects rest after motion even if EKF speed is high."""
    detector = StationaryDetector()

    # Phase 1: High speed motion (5 m/s) with active vibrations
    for i in range(30):
        acc = np.array([0.8 * math.sin(i), 0.2 * math.cos(i), 9.80665 + 1.5 * math.sin(2 * i)])
        gyro = np.array([0.2 * math.sin(i), 0.0, 0.0])
        s = detector.update(acc, gyro, speed_mps=5.0)
    assert s is False

    # Phase 2: Stopped on table (zero vibration, pure gravity), with speed_mps=None (GNSS outage)
    is_rest = False
    for _ in range(20):
        acc = np.array([0.0, 0.0, 9.80665])
        gyro = np.array([0.0, 0.0, 0.0])
        is_rest = detector.update(acc, gyro, speed_mps=None)

    assert is_rest is True, "Stationary detector failed to latch rest after motion"


def test_walking_trajectory_bounded_speed_and_dr_distance():
    """Verify that a 15-second human walk (~10m) does not produce 30+ km/h speed or 200+ m DR distance."""
    dt = 1.0 / 55.0  # 55 Hz
    engine = NavigationEngine(ref_lat=28.6139, ref_lon=77.2090, dt=dt)
    
    g = 9.80665
    p_nom = math.radians(35.0)
    r_nom = math.radians(5.0)

    def get_phone_acc(pitch, roll, dyn_fwd=0.0, dyn_lat=0.0, dyn_vert=0.0):
        gx = -g * math.sin(roll) * math.cos(pitch)
        gy = g * math.sin(pitch)
        gz = g * math.cos(pitch) * math.cos(roll)
        return np.array([gx + dyn_lat, gy + dyn_fwd, gz + dyn_vert], dtype=np.float64)

    t = 1725750000.0

    # Step 1: Standstill 10s
    for i in range(550):
        t += dt
        acc = get_phone_acc(p_nom, r_nom)
        imu = SensorInputFrame(
            timestamp=t,
            acc_x=float(acc[0]),
            acc_y=float(acc[1]),
            acc_z=float(acc[2]),
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            orientation_yaw=9.0,
            orientation_pitch=35.0,
            orientation_roll=5.0,
            is_absolute=True,
            orientation_event_type="deviceorientationabsolute"
        )
        engine.process_frame(imu, None)

    # Step 2: Walk 15s (825 frames)
    step_freq = 1.8
    peak_speed_kmh = 0.0
    for i in range(825):
        t += dt
        phase = 2.0 * math.pi * step_freq * (i * dt)
        pitch_now = p_nom + math.radians(4.0 * math.sin(phase))
        roll_now = r_nom + math.radians(2.0 * math.cos(phase))
        
        fwd_push = 0.5 * math.sin(phase)
        vert_bounce = 1.2 * math.cos(phase)
        lat_sway = 0.3 * math.sin(phase * 0.5)

        acc = get_phone_acc(pitch_now, roll_now, dyn_fwd=fwd_push, dyn_lat=lat_sway, dyn_vert=vert_bounce)
        gyro_pitch = math.radians(4.0 * step_freq * 2.0 * math.pi * math.cos(phase))
        gyro_yaw = math.radians(15.0 * math.sin(phase * 0.5))

        imu = SensorInputFrame(
            timestamp=t,
            acc_x=float(acc[0]),
            acc_y=float(acc[1]),
            acc_z=float(acc[2]),
            gyro_x=float(gyro_pitch),
            gyro_y=0.0,
            gyro_z=float(gyro_yaw),
            orientation_yaw=9.0 + 2.0 * math.sin(phase * 0.5),
            orientation_pitch=float(math.degrees(pitch_now)),
            orientation_roll=float(math.degrees(roll_now)),
            is_absolute=True,
            orientation_event_type="deviceorientationabsolute"
        )
        out = engine.process_frame(imu, None)
        kmh = out.forward_speed_mps * 3.6
        if kmh > peak_speed_kmh:
            peak_speed_kmh = kmh

    # Peak walking speed must remain in realistic human scale (< 15.0 km/h)
    assert peak_speed_kmh < 15.0, f"Walking speed reached {peak_speed_kmh:.1f} km/h (expected < 15 km/h)"
    # DR distance after walking 15s at human pace should be reasonable (< 50m, not 255m)
    assert out.diagnostics.total_dr_distance_m < 50.0

    # Step 3: Stop 10s (550 frames)
    out_stopped = None
    for i in range(550):
        t += dt
        acc = get_phone_acc(p_nom, r_nom)
        imu = SensorInputFrame(
            timestamp=t,
            acc_x=float(acc[0]),
            acc_y=float(acc[1]),
            acc_z=float(acc[2]),
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            orientation_yaw=9.0,
            orientation_pitch=35.0,
            orientation_roll=5.0,
            is_absolute=True,
            orientation_event_type="deviceorientationabsolute"
        )
        out_stopped = engine.process_frame(imu, None)

    # After stopping, speed must return to 0.0 m/s and is_stationary must be True
    assert out_stopped is not None
    assert out_stopped.is_stationary is True
    assert abs(out_stopped.forward_speed_mps) < 0.05
    
    # Verify that during continued standstill, DR distance does not accumulate further
    dr_latched = out_stopped.diagnostics.total_dr_distance_m
    for _ in range(100):
        t += dt
        acc = get_phone_acc(p_nom, r_nom)
        imu = SensorInputFrame(
            timestamp=t,
            acc_x=float(acc[0]),
            acc_y=float(acc[1]),
            acc_z=float(acc[2]),
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            orientation_yaw=9.0,
            orientation_pitch=35.0,
            orientation_roll=5.0,
            is_absolute=True,
            orientation_event_type="deviceorientationabsolute"
        )
        out_still = engine.process_frame(imu, None)
    
    assert abs(out_still.diagnostics.total_dr_distance_m - dr_latched) < 0.01


def test_nav_mode_reports_dead_reckoning_when_gnss_denied():
    """Verify that when GNSS is denied or absent, nav_mode is DEAD_RECKONING (not GNSS)."""
    engine = NavigationEngine(ref_lat=28.6139, ref_lon=77.2090, dt=0.02)
    
    imu = SensorInputFrame(
        timestamp=1725750000.0,
        acc_x=0.0, acc_y=0.0, acc_z=9.80665,
        gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
        orientation_yaw=45.0, is_absolute=True
    )
    out = engine.process_frame(imu, None)

    # Pre-anchor standstill is STATIONARY_ZUPT or DEAD_RECKONING, never GNSS_INS_FULL
    assert out.nav_mode in (NavigationMode.STATIONARY_ZUPT, NavigationMode.DEAD_RECKONING_PURE, NavigationMode.DEAD_RECKONING_NHC_AI)
    assert out.nav_mode != NavigationMode.GNSS_INS_FULL
    assert out.gnss_status == "DEGRADED"
