"""Canonical Telemetry Schema and Session Definitions for Scientific Experiments."""

from dataclasses import asdict, dataclass, field
import datetime
import json
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class EventMarker:
    """Timestamped experimental observation or maneuver marker."""
    timestamp: float
    wall_time: str
    label: str
    notes: Optional[str] = None


@dataclass
class TelemetryRecord:
    """Canonical frame-by-frame telemetry record capturing raw, calibrated, AI, and EKF states."""
    timestamp: float
    wall_time: str

    # 1. Raw Smartphone Sensors (Phone Body Frame)
    acc_phone_x: float
    acc_phone_y: float
    acc_phone_z: float
    gyro_phone_x: float
    gyro_phone_y: float
    gyro_phone_z: float
    mag_x: Optional[float] = None
    mag_y: Optional[float] = None
    mag_z: Optional[float] = None
    orientation_yaw: Optional[float] = None
    orientation_pitch: Optional[float] = None
    orientation_roll: Optional[float] = None

    # 2. Calibrated Vehicle-Frame IMU
    acc_veh_x: float = 0.0
    acc_veh_y: float = 0.0
    acc_veh_z: float = 9.81
    gyro_veh_x: float = 0.0
    gyro_veh_y: float = 0.0
    gyro_veh_z: float = 0.0

    # 3. Dynamic Phone-to-Vehicle Calibration
    calibration_state: str = "NOT_CALIBRATED"
    is_calibrated: bool = False

    # 4. Mixed-Rate GNSS Observation
    has_new_gnss: bool = False
    gnss_lat: Optional[float] = None
    gnss_lon: Optional[float] = None
    gnss_alt: Optional[float] = None
    gnss_speed_mps: Optional[float] = None
    gnss_heading_deg: Optional[float] = None
    gnss_accuracy_m: Optional[float] = None
    gnss_age_sec: float = 0.0
    gnss_trust_score: float = 0.0
    gnss_trust_status: str = "BLACKOUT"
    gnss_is_trusted: bool = False

    # 5. AI Velocity Estimator
    ai_speed_mps: float = 0.0
    ai_is_ready: bool = False
    ai_has_new_inference: bool = False

    # 6. EKF & Navigation Engine Output
    ekf_lat: float = 0.0
    ekf_lon: float = 0.0
    ekf_east_m: float = 0.0
    ekf_north_m: float = 0.0
    ekf_vel_east_mps: float = 0.0
    ekf_vel_north_mps: float = 0.0
    ekf_fwd_speed_mps: float = 0.0
    heading_deg: float = 0.0
    lean_angle_deg: float = 0.0
    nav_mode: str = "DEAD_RECKONING_PURE"
    is_in_blackout: bool = False
    is_stationary: bool = False
    nhc_active: bool = False
    zupt_active: bool = False
    zaru_active: bool = False
    total_dr_distance_m: float = 0.0

    # 7. System Diagnostics & Annotations
    step_latency_ms: float = 0.0
    event_marker: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, (np.floating, float)):
                d[k] = float(v) if np.isfinite(v) else 0.0
            elif isinstance(v, (np.integer, int)):
                d[k] = int(v)
            elif isinstance(v, np.bool_):
                d[k] = bool(v)
        return d

    @classmethod
    def csv_header(cls) -> List[str]:
        """Get ordered CSV header columns."""
        return [
            "timestamp", "wall_time",
            "acc_phone_x", "acc_phone_y", "acc_phone_z",
            "gyro_phone_x", "gyro_phone_y", "gyro_phone_z",
            "mag_x", "mag_y", "mag_z",
            "orientation_yaw", "orientation_pitch", "orientation_roll",
            "acc_veh_x", "acc_veh_y", "acc_veh_z",
            "gyro_veh_x", "gyro_veh_y", "gyro_veh_z",
            "calibration_state", "is_calibrated",
            "has_new_gnss", "gnss_lat", "gnss_lon", "gnss_alt",
            "gnss_speed_mps", "gnss_heading_deg", "gnss_accuracy_m",
            "gnss_age_sec", "gnss_trust_score", "gnss_trust_status", "gnss_is_trusted",
            "ai_speed_mps", "ai_is_ready", "ai_has_new_inference",
            "ekf_lat", "ekf_lon", "ekf_east_m", "ekf_north_m",
            "ekf_vel_east_mps", "ekf_vel_north_mps", "ekf_fwd_speed_mps",
            "heading_deg", "lean_angle_deg", "nav_mode",
            "is_in_blackout", "is_stationary",
            "nhc_active", "zupt_active", "zaru_active", "total_dr_distance_m",
            "step_latency_ms", "event_marker"
        ]

    def to_csv_row(self) -> List[Any]:
        """Format as row for CSV writer."""
        def safe_val(v):
            if v is None:
                return ""
            if isinstance(v, (float, np.floating)):
                return f"{v:.6f}" if abs(v) < 10000 else f"{v:.4e}"
            return str(v)

        return [
            f"{self.timestamp:.4f}",
            self.wall_time,
            safe_val(self.acc_phone_x), safe_val(self.acc_phone_y), safe_val(self.acc_phone_z),
            safe_val(self.gyro_phone_x), safe_val(self.gyro_phone_y), safe_val(self.gyro_phone_z),
            safe_val(self.mag_x), safe_val(self.mag_y), safe_val(self.mag_z),
            safe_val(self.orientation_yaw), safe_val(self.orientation_pitch), safe_val(self.orientation_roll),
            safe_val(self.acc_veh_x), safe_val(self.acc_veh_y), safe_val(self.acc_veh_z),
            safe_val(self.gyro_veh_x), safe_val(self.gyro_veh_y), safe_val(self.gyro_veh_z),
            self.calibration_state, "1" if self.is_calibrated else "0",
            "1" if self.has_new_gnss else "0",
            safe_val(self.gnss_lat), safe_val(self.gnss_lon), safe_val(self.gnss_alt),
            safe_val(self.gnss_speed_mps), safe_val(self.gnss_heading_deg), safe_val(self.gnss_accuracy_m),
            safe_val(self.gnss_age_sec), safe_val(self.gnss_trust_score), self.gnss_trust_status, "1" if self.gnss_is_trusted else "0",
            safe_val(self.ai_speed_mps), "1" if self.ai_is_ready else "0", "1" if self.ai_has_new_inference else "0",
            safe_val(self.ekf_lat), safe_val(self.ekf_lon), safe_val(self.ekf_east_m), safe_val(self.ekf_north_m),
            safe_val(self.ekf_vel_east_mps), safe_val(self.ekf_vel_north_mps), safe_val(self.ekf_fwd_speed_mps),
            safe_val(self.heading_deg), safe_val(self.lean_angle_deg), self.nav_mode,
            "1" if self.is_in_blackout else "0", "1" if self.is_stationary else "0",
            "1" if self.nhc_active else "0", "1" if self.zupt_active else "0", "1" if self.zaru_active else "0",
            safe_val(self.total_dr_distance_m),
            safe_val(self.step_latency_ms),
            self.event_marker or ""
        ]


@dataclass
class SessionMetadata:
    """High-level metadata for an experiment session."""
    session_id: str
    start_time_iso: str
    end_time_iso: Optional[str] = None
    duration_sec: float = 0.0
    record_count: int = 0
    vehicle_type: str = "two_wheeler"
    notes: Optional[str] = None
    schema_version: str = "1.0.0"
    software_version: str = "1.0.0-idr"
    device_info: Dict[str, Any] = field(default_factory=dict)
    event_markers: List[Dict[str, Any]] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentSession:
    """In-memory wrapper for an active or completed experiment recording."""
    metadata: SessionMetadata
    records: List[TelemetryRecord] = field(default_factory=list)
