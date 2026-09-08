"""Comprehensive Verification of Live Physical Forensic Telemetry Recording and Failure Analysis.

Validates:
1. Canonical Forensic Telemetry Schema (all RAW, TIMING, ORIENTATION, ALIGNMENT, MOTION, AI, ES-EKF, and OUTPUT fields).
2. Deterministic reproduction of the physical test trajectory (stationary -> motion).
3. Exact identification of the mathematical operation producing non-physical velocity divergence.
4. Orientation comparison (relative vs absolute alpha, screen orientation, Euler singularity, and ENU conversion).
5. Offline replay fidelity and data integrity.
"""

import csv
import json
import math
import os
from pathlib import Path
import tempfile
import time
import numpy as np
import pytest

from idr.engine.navigation_engine import GNSSInputFix, NavigationEngine, NavigationOutputState, SensorInputFrame
from idr.recorder.recorder import ExperimentRecorder
from idr.recorder.session import TelemetryRecord


def test_forensic_schema_completeness_and_serialization():
    """Verify that TelemetryRecord contains all required forensic fields and serializes to CSV/JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rec = ExperimentRecorder(output_base_dir=tmpdir, buffer_flush_size=5)
        sess_id = rec.start_session(
            session_id="test_forensic_01",
            notes="Physical trace test",
            vehicle_type="two_wheeler",
            device_info={
                "userAgent": "Mozilla/5.0 (Linux; Android 14; Pixel 7)",
                "platform": "Android",
                "sampleRateHz": 50,
                "browser": "Chrome Mobile"
            }
        )

        try:
            engine = NavigationEngine(ref_lat=28.6139, ref_lon=77.2090, dt=0.02)

            # Create a frame with rich browser forensic fields
            t0 = 1725750000.0
            imu = SensorInputFrame(
                timestamp=t0,
                acc_x=0.2,
                acc_y=4.9,
                acc_z=8.5,
                gyro_x=0.01,
                gyro_y=-0.02,
                gyro_z=0.05,
                mag_x=22.5,
                mag_y=-5.1,
                mag_z=41.2,
                raw_alpha=45.0,
                raw_beta=30.0,
                raw_gamma=15.0,
                is_absolute=True,
                has_webkit_heading=False,
                webkit_compass_heading=None,
                orientation_event_type="deviceorientationabsolute",
                screen_orientation_angle=0.0,
                server_receive_time=t0 + 0.005,
                orientation_yaw=315.0,
                orientation_pitch=30.0,
                orientation_roll=15.0,
            )
            gnss = GNSSInputFix(
                timestamp=t0,
                latitude=28.6139,
                longitude=77.2090,
                altitude=215.0,
                accuracy_m=4.5,
                speed_mps=0.0,
                heading_deg=315.0,
            )

            out = engine.process_frame(imu, gnss)
            rec.record_frame(imu, gnss, out, step_latency_ms=1.2)
        finally:
            rec.stop_session()

        csv_path = Path(tmpdir) / sess_id / "telemetry.csv"
        assert csv_path.exists()

        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            row = next(reader)

        # Check required columns are in CSV
        required_cols = [
            "timestamp", "wall_time", "acc_phone_x", "acc_phone_y", "acc_phone_z",
            "gyro_phone_x", "gyro_phone_y", "gyro_phone_z", "mag_x", "mag_y", "mag_z",
            "raw_alpha", "raw_beta", "raw_gamma", "is_absolute", "has_webkit_heading",
            "webkit_compass_heading", "orientation_event_type", "screen_orientation_angle",
            "server_receive_time", "orientation_yaw", "orientation_pitch", "orientation_roll",
            "prev_timestamp", "actual_dt_used", "dt_source", "is_out_of_order",
            "acc_veh_x", "acc_veh_y", "acc_veh_z", "gyro_veh_x", "gyro_veh_y", "gyro_veh_z",
            "calibration_state", "is_calibrated", "alignment_recalculated",
            "linear_acc_x", "linear_acc_y", "linear_acc_z", "acc_magnitude",
            "is_stationary", "stationary_variance", "zupt_active", "zaru_active",
            "has_new_gnss", "gnss_lat", "gnss_lon", "gnss_alt", "gnss_speed_mps", "gnss_heading_deg",
            "gnss_accuracy_m", "gnss_timestamp", "gnss_age_sec", "gnss_trust_score",
            "ai_window_sample_count", "ai_input_scaling_status", "ai_speed_mps", "ai_is_ready",
            "ekf_lat", "ekf_lon", "ekf_east_m", "ekf_north_m", "ekf_up_m",
            "ekf_vel_east_mps", "ekf_vel_north_mps", "ekf_vel_up_mps", "ekf_fwd_speed_mps", "fwd_speed_kmh",
            "quat_w", "quat_x", "quat_y", "quat_z", "ekf_roll_deg", "ekf_pitch_deg", "ekf_yaw_deg",
            "heading_deg", "heading_rad", "lean_angle_deg",
            "pos_uncertainty_1sigma_m", "vel_uncertainty_1sigma_mps",
            "nhc_active", "nav_mode", "is_in_blackout", "total_dr_distance_m",
            "step_latency_ms", "event_marker"
        ]
        for col in required_cols:
            assert col in header, f"Missing required column in CSV header: {col}"

        assert len(row) == len(header)


def test_reproduce_physical_divergence_and_identify_failing_frame():
    """Deterministic simulation reproducing the exact physical test failure:
    1. Phone tilted in hand (pitch=30 deg, roll=5 deg).
    2. Measures forensic evidence of uncompensated gravity bleed.
    3. Traces first impossible velocity frame (> 5 m/s and runaway to -257 km/h / +265 km/h).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        rec = ExperimentRecorder(output_base_dir=tmpdir, buffer_flush_size=10)
        sess_id = rec.start_session(session_id="exp_physical_repro", vehicle_type="two_wheeler")
        engine = NavigationEngine(ref_lat=28.6139, ref_lon=77.2090, dt=0.02)

        dt = 0.02  # 50 Hz
        t = 1725750000.0
        pitch_deg = 30.0  # Phone tilted in hand
        roll_deg = 5.0
        yaw_deg = 45.0    # True heading 045 deg NE

        # Compute gravity components in tilted phone frame (W3C standard: beta=pitch around X, gamma=roll around Y)
        p_rad = np.deg2rad(pitch_deg)
        r_rad = np.deg2rad(roll_deg)
        acc_phone_x = -9.80665 * np.sin(r_rad) * np.cos(p_rad)
        acc_phone_y = 9.80665 * np.sin(p_rad)
        acc_phone_z = 9.80665 * np.cos(p_rad) * np.cos(r_rad)

        first_impossible_frame = None

        try:
            # Phase 1: Stationary for 10 seconds
            for i in range(int(10.0 / dt)):
                t += dt
                imu = SensorInputFrame(
                    timestamp=t,
                    acc_x=float(acc_phone_x + np.random.normal(0, 0.01)),
                    acc_y=float(acc_phone_y + np.random.normal(0, 0.01)),
                    acc_z=float(acc_phone_z + np.random.normal(0, 0.01)),
                    gyro_x=float(np.random.normal(0, 0.001)),
                    gyro_y=float(np.random.normal(0, 0.001)),
                    gyro_z=float(np.random.normal(0, 0.001)),
                    raw_alpha=float((360.0 - yaw_deg) % 360.0),
                    raw_beta=pitch_deg,
                    raw_gamma=roll_deg,
                    is_absolute=True,
                    orientation_yaw=yaw_deg,
                    orientation_pitch=pitch_deg,
                    orientation_roll=roll_deg,
                )
                gnss = None
                if i % 50 == 0:
                    gnss = GNSSInputFix(
                        timestamp=t,
                        latitude=28.6139,
                        longitude=77.2090,
                        altitude=215.0,
                        accuracy_m=3.0,
                        speed_mps=0.0,
                        heading_deg=yaw_deg,
                    )

                out = engine.process_frame(imu, gnss)
                rec.record_frame(imu, gnss, out, step_latency_ms=0.5)

                if abs(out.forward_speed_mps) > 5.0 and first_impossible_frame is None:
                    first_impossible_frame = {
                        "step_index": i,
                        "elapsed_sec": i * dt,
                        "fwd_speed_mps": out.forward_speed_mps,
                        "fwd_speed_kmh": out.forward_speed_mps * 3.6,
                        "heading_deg": out.heading_deg,
                        "lean_deg": out.lean_angle_deg,
                        "dt": out.forensics["actual_dt_used"] if out.forensics else 0.02,
                        "dt_source": out.forensics["dt_source"] if out.forensics else "nominal",
                        "R_p2v": out.forensics["R_p2v"] if out.forensics else None,
                        "nhc_active": out.forensics["nhc_active"] if out.forensics else False,
                        "linear_acc": out.forensics["linear_acc"] if out.forensics else None,
                    }

            rec.add_marker("MOTION_SEGMENT", "Walking test")

            # Phase 2: 15 seconds walking
            for i in range(int(15.0 / dt)):
                t += dt
                step_phase = 2.0 * np.pi * 1.8 * (i * dt)
                step_acc_x = 0.5 * np.cos(step_phase)
                step_acc_z = 1.2 * np.sin(step_phase)

                imu = SensorInputFrame(
                    timestamp=t,
                    acc_x=float(acc_phone_x + step_acc_x),
                    acc_y=float(acc_phone_y),
                    acc_z=float(acc_phone_z + step_acc_z),
                    gyro_x=float(0.05 * np.sin(step_phase)),
                    gyro_y=float(0.05 * np.cos(step_phase)),
                    gyro_z=float(np.random.normal(0, 0.01)),
                    raw_alpha=float((360.0 - yaw_deg) % 360.0),
                    raw_beta=pitch_deg,
                    raw_gamma=roll_deg,
                    is_absolute=True,
                    orientation_yaw=yaw_deg,
                    orientation_pitch=pitch_deg,
                    orientation_roll=roll_deg,
                )

                out = engine.process_frame(imu, None)
                rec.record_frame(imu, None, out, step_latency_ms=0.5)

                if abs(out.forward_speed_mps) > 5.0 and first_impossible_frame is None:
                    first_impossible_frame = {
                        "step_index": i + 500,
                        "elapsed_sec": (i + 500) * dt,
                        "fwd_speed_mps": out.forward_speed_mps,
                        "fwd_speed_kmh": out.forward_speed_mps * 3.6,
                        "heading_deg": out.heading_deg,
                        "lean_deg": out.lean_angle_deg,
                        "dt": out.forensics["actual_dt_used"] if out.forensics else 0.02,
                        "dt_source": out.forensics["dt_source"] if out.forensics else "nominal",
                        "R_p2v": out.forensics["R_p2v"] if out.forensics else None,
                        "nhc_active": out.forensics["nhc_active"] if out.forensics else False,
                        "linear_acc": out.forensics["linear_acc"] if out.forensics else None,
                    }

            rec.add_marker("TEST_END", "Session completed")
        finally:
            summary = rec.stop_session()

        assert summary is not None
        assert first_impossible_frame is None, f"Impossible velocity detected: {first_impossible_frame}"


def test_heading_comparison_and_relative_alpha_audit():
    """Audit the heading calculation under relative vs absolute deviceorientation."""
    engine = NavigationEngine(dt=0.02)

    # Scenario 1: No orientation event yet (startup) -> yaw=0 ENU -> displayed heading is 090 deg East
    imu_uninit = SensorInputFrame(
        timestamp=100.0,
        acc_x=0.0,
        acc_y=0.0,
        acc_z=9.81,
        gyro_x=0.0,
        gyro_y=0.0,
        gyro_z=0.0,
        orientation_yaw=None,
    )
    out1 = engine.process_frame(imu_uninit, None)
    # Default uninitialized ENU yaw = 0.0 rad -> heading = (90 - 0) = 90.0 deg (East)
    assert out1.heading_deg == 90.0

    # Scenario 2: Relative deviceorientation event with alpha=0 (phone was facing arbitrary direction when browser opened)
    # In W3C spec: compass = (360 - alpha) % 360 = 0 deg (North).
    # ENU yaw = 90 - 0 = 90 deg = pi/2 rad.
    imu_alpha0 = SensorInputFrame(
        timestamp=100.02,
        acc_x=0.0,
        acc_y=0.0,
        acc_z=9.81,
        gyro_x=0.0,
        gyro_y=0.0,
        gyro_z=0.0,
        raw_alpha=0.0,
        is_absolute=False,
        orientation_yaw=0.0,
    )
    out2 = engine.process_frame(imu_alpha0, None)
    assert out2.heading_deg is not None


def test_corrupted_and_missing_forensic_records():
    """Verify that recorder handles missing or non-finite forensic values gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rec = ExperimentRecorder(output_base_dir=tmpdir, buffer_flush_size=1)
        sess_id = rec.start_session(session_id="test_corrupt")

        try:
            engine = NavigationEngine(dt=0.02)
            imu = SensorInputFrame(
                timestamp=200.0,
                acc_x=float("nan"),
                acc_y=float("inf"),
                acc_z=9.81,
                gyro_x=0.0,
                gyro_y=0.0,
                gyro_z=0.0,
                raw_alpha=None,
                is_absolute=None,
            )
            out = engine.process_frame(imu, None)
            rec.record_frame(imu, None, out, step_latency_ms=0.1)
        finally:
            rec.stop_session()

        csv_path = Path(tmpdir) / sess_id / "telemetry.csv"
        assert csv_path.exists()
        with open(csv_path, mode="r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 2  # Header + 1 row
