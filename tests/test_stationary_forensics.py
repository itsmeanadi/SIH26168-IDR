"""Forensic script for diagnosing stationary velocity leakage.
This script is designed to be run as a standalone test or integrated into the
live physical test suite to isolate the 'AI-induced ZUPT lockout' hypothesis.
"""

import numpy as np
import pytest
from idr.engine.navigation_engine import (
    GNSSInputFix,
    NavigationEngine,
    SensorInputFrame,
)

def test_forensic_stationary_velocity_leakage():
    """
    Forensic trace of a 30-second stationary period.
    Captures the interplay between AI speed, StationaryDetector, and ZUPT.
    """
    # Initialize engine
    engine = NavigationEngine(ref_lat=28.6139, ref_lon=77.2090, vehicle_type="two_wheeler")

    # Simulation parameters: 30 seconds at 10 Hz
    duration_sec = 30
    hz = 10
    total_frames = duration_sec * hz

    # Data collection containers
    history = []

    print("\n--- STARTING STATIONARY FORENSIC TRACE ---")
    print(f"Target: {duration_sec}s at {hz}Hz")

    for i in range(total_frames):
        t = i * 0.1
        # Stationary input (flat phone)
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

        # Extract forensic fields from the engine's comprehensive diagnostics
        forensics = out.forensics or {}

        frame_data = {
            "timestamp": t,
            "is_stationary": out.is_stationary,
            "ai_speed_mps": engine.latest_ai_speed,
            "stationary_variance": forensics.get("stationary_variance", 0.0),
            "dyn_acc_norm": forensics.get("acc_magnitude", 0.0), # Close enough for diagnostics
            "ekf_vel_enu": out.forensics.get("ekf_vel_enu", [0, 0, 0]) if out.forensics else [0, 0, 0],
            "zupt_active": out.nav_mode == "STATIONARY_ZUPT",
            "nhc_active": forensics.get("nhc_active", False),
            "total_dr_distance": engine.total_dr_distance,
        }
        history.append(frame_data)

    # Final Report Calculation
    timestamps = [f["timestamp"] for f in history]
    is_stationary_list = [f["is_stationary"] for f in history]
    ai_speeds = [f["ai_speed_mps"] for f in history]
    ekf_speeds = [np.linalg.norm(f["ekf_vel_enu"]) for f in history]

    stationary_start = None
    for i, stat in enumerate(is_stationary_list):
        if stat:
            stationary_start = timestamps[i]
            break

    # AI Accept/Reject from engine counters
    ai_accepted = engine.ai_accepted_count
    ai_rejected = engine.ai_rejected_count

    print("\n--- FORENSIC REPORT ---")
    print(f"First Stationary Latch (s): {stationary_start if stationary_start is not None else 'NEVER'}")
    print(f"AI Speed: Min={min(ai_speeds):.3f}, Max={max(ai_speeds):.3f}, Mean={np.mean(ai_speeds):.3f}")
    print(f"EKF Speed: Min={min(ekf_speeds):.3f}, Max={max(ekf_speeds):.3f}, Mean={np.mean(ekf_speeds):.3f}")
    print(f"Total DR Distance: {engine.total_dr_distance:.4f} m")
    print(f"Stationary Frame %: {(sum(is_stationary_list)/total_frames)*100:.1f}%")
    print(f"AI Events: Accepted={ai_accepted}, Rejected={ai_rejected}")

    # The Hypothesis Check
    # If AI speed > 0.8 and is_stationary is False, but variance is low -> Lockout confirmed.
    lockout_detected = False
    for f in history:
        if f["ai_speed_mps"] > 0.8 and not f["is_stationary"] and f["stationary_variance"] < 0.15:
            lockout_detected = True
            break

    print(f"AI-Induced ZUPT Lockout Hypothesis: {'CONFIRMED' if lockout_detected else 'DISPROVED'}")
    print("------------------------------------------\n")

    # Assertions for CI/CD
    assert engine.total_dr_distance < 0.1, f"Stationary DR distance leaked: {engine.total_dr_distance}m"
    assert stationary_start is not None, "System never entered stationary state"
