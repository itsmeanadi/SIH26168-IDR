"""Explainable Crash Detection Engine (USP 4).

Physics-grounded 3-stage heuristic:
1. Impact: Sudden deceleration/acceleration shock (|a| > threshold, e.g. 3.5g)
2. Dynamic Tumbling/Rotation: Abnormal angular velocity (|omega| > threshold, e.g. 250 deg/s or abnormal tilt)
3. Subsequent Stillness: Vehicle immobilization / speed drop to ~0 with low variance for confirmation window

Locks highest-accuracy fused/dead-reckoned coordinates and formats emergency payload.
"""

from dataclasses import asdict, dataclass
from enum import Enum
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class CrashState(str, Enum):
    NORMAL = "NORMAL"
    IMPACT_DETECTED = "IMPACT_DETECTED"
    CONFIRMING_STILLNESS = "CONFIRMING_STILLNESS"
    CRASH_CONFIRMED = "CRASH_CONFIRMED"
    CANCELLED_BY_USER = "CANCELLED_BY_USER"


@dataclass
class CrashAlert:
    id: str
    timestamp: float
    latitude: float
    longitude: float
    impact_g_force: float
    max_yaw_rate_deg_s: float
    vehicle_type: str
    confidence: float
    is_confirmed: bool
    google_maps_url: str
    emergency_message: str


class CrashDetector:
    """Explainable physics-based crash detection engine."""

    def __init__(
        self,
        impact_g_threshold: float = 3.5,        # 3.5g ~ 34.3 m/s^2 shock
        gyro_rate_threshold_deg_s: float = 250.0, # 250 deg/s ~ 4.36 rad/s
        stillness_duration_sec: float = 2.0,    # Time after impact to check for no movement
        speed_stillness_threshold_mps: float = 1.0, # Speed < 1.0 m/s
    ):
        self.impact_g_threshold = impact_g_threshold
        self.gyro_rate_threshold_deg_s = gyro_rate_threshold_deg_s
        self.stillness_duration_sec = stillness_duration_sec
        self.speed_stillness_threshold_mps = speed_stillness_threshold_mps

        self.state = CrashState.NORMAL
        self.impact_timestamp: Optional[float] = None
        self.peak_impact_g: float = 0.0
        self.peak_gyro_deg_s: float = 0.0
        self.locked_pos: Optional[Tuple[float, float]] = None
        self.active_alert: Optional[CrashAlert] = None
        self._alert_counter = 0

    def update(
        self,
        acc_3d: np.ndarray,      # [ax, ay, az] in m/s^2 (phone or vehicle body frame)
        gyro_3d: np.ndarray,     # [gx, gy, gz] in rad/s
        current_speed_mps: float,
        current_lat: float,
        current_lon: float,
        vehicle_type: str = "two_wheeler",
        timestamp: Optional[float] = None,
    ) -> Optional[CrashAlert]:
        """Process an IMU + navigation state frame for crash indicators."""
        t = timestamp if timestamp is not None else time.time()
        g_acc = float(np.linalg.norm(acc_3d) / 9.80665)
        gyro_deg_s = float(np.rad2deg(np.linalg.norm(gyro_3d)))

        # State 1: Normal monitoring
        if self.state == CrashState.NORMAL:
            if g_acc >= self.impact_g_threshold:
                # Stage 1 triggered: High G impact
                self.impact_timestamp = t
                self.peak_impact_g = g_acc
                self.peak_gyro_deg_s = gyro_deg_s
                self.locked_pos = (float(current_lat), float(current_lon))

                # If rotational surge is already present on impact frame
                if gyro_deg_s >= self.gyro_rate_threshold_deg_s or g_acc >= (self.impact_g_threshold * 1.5):
                    self.state = CrashState.CONFIRMING_STILLNESS
                else:
                    self.state = CrashState.IMPACT_DETECTED

        # State 2: Check for abnormal rotation surge within 0.5s of impact
        elif self.state == CrashState.IMPACT_DETECTED:
            self.peak_impact_g = max(self.peak_impact_g, g_acc)
            self.peak_gyro_deg_s = max(self.peak_gyro_deg_s, gyro_deg_s)
            
            # If within 0.5s of impact, verify rotational tumble
            dt_impact = t - (self.impact_timestamp or t)
            if self.peak_gyro_deg_s >= self.gyro_rate_threshold_deg_s or self.peak_impact_g >= (self.impact_g_threshold * 1.5):
                self.state = CrashState.CONFIRMING_STILLNESS
            elif dt_impact > 0.5:
                # Timed out without sufficient rotational surge
                if self.peak_impact_g < (self.impact_g_threshold * 1.8):
                    # Likely just a severe road pothole / speed bump
                    self.state = CrashState.NORMAL
                else:
                    self.state = CrashState.CONFIRMING_STILLNESS

        # State 3: Confirm stillness / immobilization
        if self.state == CrashState.CONFIRMING_STILLNESS:
            dt_impact = t - (self.impact_timestamp or t)
            
            # If vehicle continues moving at regular speed, it wasn't an incapacitating crash
            if current_speed_mps > (self.speed_stillness_threshold_mps * 2.0) and dt_impact > 1.0:
                # Resumed normal movement -> False alarm
                self.state = CrashState.NORMAL
                return None

            if dt_impact >= self.stillness_duration_sec:
                # Stillness confirmed!
                if current_speed_mps <= self.speed_stillness_threshold_mps:
                    self._alert_counter += 1
                    lat, lon = self.locked_pos if self.locked_pos else (current_lat, current_lon)
                    maps_url = f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"
                    sos_msg = (
                        f"EMERGENCY: Potential {vehicle_type.upper()} crash detected at "
                        f"{lat:.6f}, {lon:.6f} (Peak Impact: {self.peak_impact_g:.1f}g). "
                        f"Location: {maps_url}"
                    )

                    self.active_alert = CrashAlert(
                        id=f"CRASH-{self._alert_counter:04d}",
                        timestamp=self.impact_timestamp or t,
                        latitude=lat,
                        longitude=lon,
                        impact_g_force=round(self.peak_impact_g, 1),
                        max_yaw_rate_deg_s=round(self.peak_gyro_deg_s, 1),
                        vehicle_type=vehicle_type,
                        confidence=min(0.98, 0.70 + (self.peak_impact_g / 20.0)),
                        is_confirmed=True,
                        google_maps_url=maps_url,
                        emergency_message=sos_msg,
                    )
                    self.state = CrashState.CRASH_CONFIRMED
                    return self.active_alert
                else:
                    self.state = CrashState.NORMAL

        elif self.state == CrashState.CRASH_CONFIRMED:
            return self.active_alert

        return None

    def trigger_test_crash(
        self,
        current_lat: float,
        current_lon: float,
        vehicle_type: str = "two_wheeler",
    ) -> CrashAlert:
        """Manually trigger a crash alert simulation for demonstration."""
        self._alert_counter += 1
        t = time.time()
        maps_url = f"https://www.google.com/maps?q={current_lat:.6f},{current_lon:.6f}"
        sos_msg = (
            f"EMERGENCY TEST: Simulated {vehicle_type.upper()} crash at "
            f"{current_lat:.6f}, {current_lon:.6f} (Impact: 4.8g). Location: {maps_url}"
        )
        self.active_alert = CrashAlert(
            id=f"CRASH-TEST-{self._alert_counter:04d}",
            timestamp=t,
            latitude=float(current_lat),
            longitude=float(current_lon),
            impact_g_force=4.8,
            max_yaw_rate_deg_s=312.0,
            vehicle_type=vehicle_type,
            confidence=0.95,
            is_confirmed=True,
            google_maps_url=maps_url,
            emergency_message=sos_msg,
        )
        self.state = CrashState.CRASH_CONFIRMED
        return self.active_alert

    def cancel_alert(self):
        """User pressed cancel within SOS window."""
        self.state = CrashState.CANCELLED_BY_USER
        self.active_alert = None
        self.impact_timestamp = None
        self.peak_impact_g = 0.0
        self.peak_gyro_deg_s = 0.0

    def reset(self):
        self.state = CrashState.NORMAL
        self.active_alert = None
        self.impact_timestamp = None
        self.peak_impact_g = 0.0
        self.peak_gyro_deg_s = 0.0
        self.locked_pos = None
