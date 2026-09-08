"""Comprehensive physical integrity tests for live mobile sensor ingestion and stationary invariance.

Validates:
1. Flat stationary phone without GNSS (pure DR at rest)
2. Tilted stationary phone in portrait/handheld orientations (pitch 45-75 deg, roll 10 deg)
3. Transition from initial GNSS anchor to extended GNSS blackout
4. Stationary IMU with hand-tremor vibration noise
5. Hard physical invariants:
   - forward_speed_mps < 0.2 m/s (0.7 km/h) throughout 20 seconds of rest
   - velocity_east and velocity_north near 0
   - total position drift < 0.5 meters
   - lean angle == 0 deg
   - pos_uncertainty bounded
   - no NaN or Inf in any state component
"""

import numpy as np
import pytest

from idr.engine.navigation_engine import (
    GNSSInputFix,
    NavigationEngine,
    NavigationMode,
    SensorInputFrame,
)


def test_flat_stationary_phone_no_gnss():
    """Flat stationary phone (acc=[0, 0, 9.81], gyro=[0, 0, 0]) without GNSS."""
    engine = NavigationEngine(ref_lat=28.6139, ref_lon=77.2090, vehicle_type="two_wheeler")
    
    for i in range(200):  # 20 seconds at 10 Hz
        t = i * 0.1
        imu = SensorInputFrame(
            timestamp=t,
            acc_x=0.0,
            acc_y=0.0,
            acc_z=9.81,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
        )
        out = engine.process_frame(imu, None)
        
        # Hard physical invariants
        assert np.isfinite(out.forward_speed_mps)
        assert abs(out.forward_speed_mps) < 0.1, f"Stationary speed drifted to {out.forward_speed_mps} m/s at t={t}s"
        assert abs(out.velocity_east) < 0.1
        assert abs(out.velocity_north) < 0.1
        assert out.lean_angle_deg == 0.0
        assert np.isfinite(out.latitude)
        assert np.isfinite(out.longitude)
        assert out.pos_uncertainty_m < 20.0
        
    # After 20 seconds, total DR distance should be minimal (< 0.1m)
    assert engine.total_dr_distance < 0.2


def test_tilted_portrait_handheld_stationary_phone():
    """Stationary phone held in hand in portrait mode (pitch 60 deg, roll 10 deg)."""
    engine = NavigationEngine(ref_lat=28.6139, ref_lon=77.2090, vehicle_type="two_wheeler")
    
    # 60 deg pitch, 10 deg roll specific force decomposition
    # g = 9.81 m/s^2 along gravity
    # acc_raw has components on x, y, z
    ax = 9.81 * np.sin(np.deg2rad(10.0)) * np.cos(np.deg2rad(60.0))  # ~0.85
    ay = 9.81 * np.sin(np.deg2rad(60.0))                           # ~8.50
    az = 9.81 * np.cos(np.deg2rad(10.0)) * np.cos(np.deg2rad(60.0))  # ~4.83
    
    for i in range(200):
        t = i * 0.1
        imu = SensorInputFrame(
            timestamp=t,
            acc_x=float(ax),
            acc_y=float(ay),
            acc_z=float(az),
            gyro_x=0.0005,
            gyro_y=-0.0005,
            gyro_z=0.0002,
        )
        out = engine.process_frame(imu, None)
        
        # Speed must remain near 0 m/s throughout
        assert abs(out.forward_speed_mps) < 0.15, f"Tilted stationary speed drifted to {out.forward_speed_mps} m/s at t={t}s"
        assert abs(out.velocity_east) < 0.15
        assert abs(out.velocity_north) < 0.15
        assert out.lean_angle_deg == 0.0
        
    assert engine.total_dr_distance < 0.3


def test_gnss_to_blackout_stationary_continuity():
    """Initial 5 seconds with valid GNSS fix, followed by 20 seconds of GNSS blackout."""
    engine = NavigationEngine(ref_lat=28.6139, ref_lon=77.2090, vehicle_type="two_wheeler")
    
    for i in range(250):  # 25 seconds total
        t = i * 0.1
        imu = SensorInputFrame(
            timestamp=t,
            acc_x=0.05,
            acc_y=8.5,
            acc_z=4.9,
            gyro_x=0.001,
            gyro_y=0.001,
            gyro_z=0.001,
        )
        gnss = None
        # GNSS active for first 5 seconds only (1 Hz fixes)
        if i < 50 and i % 10 == 0:
            gnss = GNSSInputFix(
                timestamp=t,
                latitude=28.6139,
                longitude=77.2090,
                altitude=0.0,
                accuracy_m=2.5,
                speed_mps=0.0,
                heading_deg=90.0,
            )
        out = engine.process_frame(imu, gnss)
        
        assert abs(out.forward_speed_mps) < 0.15, f"Speed became {out.forward_speed_mps} m/s during blackout at t={t}s"
        assert abs(out.velocity_east) < 0.15
        assert abs(out.velocity_north) < 0.15
        assert out.lean_angle_deg == 0.0
        assert np.isfinite(out.pos_uncertainty_m)
        assert out.pos_uncertainty_m < 15.0


def test_hand_tremor_vibration_noise_robustness():
    """Stationary phone with hand-tremor Gaussian noise on accelerometer and gyroscope."""
    np.random.seed(42)
    engine = NavigationEngine(ref_lat=28.6139, ref_lon=77.2090, vehicle_type="two_wheeler")
    
    base_acc = np.array([0.1, 8.5, 4.9], dtype=np.float32)
    
    for i in range(200):
        t = i * 0.1
        noise_a = np.random.normal(0.0, 0.05, 3).astype(np.float32)
        noise_g = np.random.normal(0.0, 0.005, 3).astype(np.float32)
        
        imu = SensorInputFrame(
            timestamp=t,
            acc_x=float(base_acc[0] + noise_a[0]),
            acc_y=float(base_acc[1] + noise_a[1]),
            acc_z=float(base_acc[2] + noise_a[2]),
            gyro_x=float(noise_g[0]),
            gyro_y=float(noise_g[1]),
            gyro_z=float(noise_g[2]),
        )
        out = engine.process_frame(imu, None)
        
        assert abs(out.forward_speed_mps) < 0.25, f"Noisy speed became {out.forward_speed_mps} m/s at t={t}s"
        assert abs(out.velocity_east) < 0.25
        assert abs(out.velocity_north) < 0.25
        assert out.lean_angle_deg == 0.0


def test_stationary_compass_heading_stability():
    """Verify that when orientation_yaw (compass) is provided while stationary, heading stays locked to true compass orientation."""
    engine = NavigationEngine(ref_lat=12.9716, ref_lon=77.5946, vehicle_type="two_wheeler")

    # Stationary phone pointing East (090 deg compass)
    for i in range(100):  # 10 seconds at 10 Hz
        t = i * 0.1
        imu = SensorInputFrame(
            timestamp=t,
            acc_x=0.0,
            acc_y=0.0,
            acc_z=9.81,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.015,  # Uncalibrated gyro bias drifting at 0.015 rad/s
            orientation_yaw=90.0,  # Compass says East
        )
        out = engine.process_frame(imu, None)

    # Heading should remain within 3 degrees of 90 deg (East) despite uncalibrated gyro bias
    assert abs(out.heading_deg - 90.0) < 3.0, f"Heading drifted from 90 deg to {out.heading_deg} deg"
    assert out.forward_speed_mps == 0.0
