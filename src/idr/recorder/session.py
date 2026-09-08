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

    # Raw Orientation & Browser Provenance
    raw_alpha: Optional[float] = None
    raw_beta: Optional[float] = None
    raw_gamma: Optional[float] = None
    is_absolute: Optional[bool] = None
    has_webkit_heading: Optional[bool] = None
    webkit_compass_heading: Optional[float] = None
    orientation_event_type: Optional[str] = None
    screen_orientation_angle: Optional[float] = None
    server_receive_time: Optional[float] = None

    # Euler / Converted Angles
    orientation_yaw: Optional[float] = None
    orientation_pitch: Optional[float] = None
    orientation_roll: Optional[float] = None

    # 2. Timing
    prev_timestamp: Optional[float] = None
    actual_dt_used: float = 0.1
    dt_source: str = "nominal"
    is_out_of_order: bool = False

    # 3. Calibrated Vehicle-Frame IMU & Alignment
    acc_veh_x: float = 0.0
    acc_veh_y: float = 0.0
    acc_veh_z: float = 9.81
    gyro_veh_x: float = 0.0
    gyro_veh_y: float = 0.0
    gyro_veh_z: float = 0.0
    calibration_state: str = "NOT_CALIBRATED"
    is_calibrated: bool = False
    alignment_recalculated: bool = False
    R_p2v_00: float = 1.0
    R_p2v_01: float = 0.0
    R_p2v_02: float = 0.0
    R_p2v_10: float = 0.0
    R_p2v_11: float = 1.0
    R_p2v_12: float = 0.0
    R_p2v_20: float = 0.0
    R_p2v_21: float = 0.0
    R_p2v_22: float = 1.0

    # 4. Motion & Stationary Dynamics
    linear_acc_x: float = 0.0
    linear_acc_y: float = 0.0
    linear_acc_z: float = 0.0
    acc_magnitude: float = 9.81
    is_stationary: bool = False
    stationary_variance: float = 0.0
    zupt_active: bool = False
    zaru_active: bool = False

    # 5. Mixed-Rate GNSS Observation
    has_new_gnss: bool = False
    gnss_lat: Optional[float] = None
    gnss_lon: Optional[float] = None
    gnss_alt: Optional[float] = None
    gnss_speed_mps: Optional[float] = None
    gnss_heading_deg: Optional[float] = None
    gnss_accuracy_m: Optional[float] = None
    gnss_timestamp: Optional[float] = None
    gnss_age_sec: float = 0.0
    gnss_trust_score: float = 0.0
    gnss_trust_status: str = "BLACKOUT"
    gnss_is_trusted: bool = False

    # 6. AI Velocity Estimator
    ai_window_sample_count: int = 0
    ai_input_scaling_status: str = "raw_m_s2"
    ai_speed_mps: float = 0.0
    ai_is_ready: bool = False
    ai_has_new_inference: bool = False
    ai_confidence_sigma: float = 3.0
    ai_innovation: Optional[float] = None
    ai_accepted: bool = False

    # 7. ES-EKF States & Observability
    ekf_east_m: float = 0.0
    ekf_north_m: float = 0.0
    ekf_up_m: float = 0.0
    ekf_vel_east_mps: float = 0.0
    ekf_vel_north_mps: float = 0.0
    ekf_vel_up_mps: float = 0.0
    ekf_fwd_speed_mps: float = 0.0
    fwd_speed_kmh: float = 0.0
    quat_w: float = 1.0
    quat_x: float = 0.0
    quat_y: float = 0.0
    quat_z: float = 0.0
    ekf_roll_deg: float = 0.0
    ekf_pitch_deg: float = 0.0
    ekf_yaw_deg: float = 0.0
    heading_deg: float = 0.0
    heading_rad: float = 0.0
    lean_angle_deg: float = 0.0
    bias_acc_x: float = 0.0
    bias_acc_y: float = 0.0
    bias_acc_z: float = 0.0
    bias_gyro_x: float = 0.0
    bias_gyro_y: float = 0.0
    bias_gyro_z: float = 0.0
    pos_uncertainty_1sigma_m: float = 5.0
    vel_uncertainty_1sigma_mps: float = 0.5
    nhc_active: bool = False
    nhc_residual_lat: float = 0.0
    nhc_residual_vert: float = 0.0
    nhc_accepted: bool = False
    gnss_vel_update_accepted: bool = False
    attitude_update_accepted: bool = False
    self_healing_reanchored: bool = False

    # 8. Navigation Output & Track
    ekf_lat: float = 0.0
    ekf_lon: float = 0.0
    nav_mode: str = "DEAD_RECKONING_PURE"
    is_in_blackout: bool = False
    total_dr_distance_m: float = 0.0
    trajectory_point_accepted: bool = True

    # 9. System Diagnostics & Annotations
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
            # Raw Phone
            "acc_phone_x", "acc_phone_y", "acc_phone_z",
            "gyro_phone_x", "gyro_phone_y", "gyro_phone_z",
            "mag_x", "mag_y", "mag_z",
            "raw_alpha", "raw_beta", "raw_gamma", "is_absolute",
            "has_webkit_heading", "webkit_compass_heading", "orientation_event_type", "screen_orientation_angle",
            "server_receive_time",
            "orientation_yaw", "orientation_pitch", "orientation_roll",
            # Timing
            "prev_timestamp", "actual_dt_used", "dt_source", "is_out_of_order",
            # Vehicle Frame & Alignment
            "acc_veh_x", "acc_veh_y", "acc_veh_z",
            "gyro_veh_x", "gyro_veh_y", "gyro_veh_z",
            "calibration_state", "is_calibrated", "alignment_recalculated",
            "R_p2v_00", "R_p2v_01", "R_p2v_02",
            "R_p2v_10", "R_p2v_11", "R_p2v_12",
            "R_p2v_20", "R_p2v_21", "R_p2v_22",
            # Motion & Stationary
            "linear_acc_x", "linear_acc_y", "linear_acc_z", "acc_magnitude",
            "is_stationary", "stationary_variance", "zupt_active", "zaru_active",
            # GNSS
            "has_new_gnss", "gnss_lat", "gnss_lon", "gnss_alt",
            "gnss_speed_mps", "gnss_heading_deg", "gnss_accuracy_m", "gnss_timestamp",
            "gnss_age_sec", "gnss_trust_score", "gnss_trust_status", "gnss_is_trusted",
            # AI
            "ai_window_sample_count", "ai_input_scaling_status", "ai_speed_mps",
            "ai_is_ready", "ai_has_new_inference", "ai_confidence_sigma", "ai_innovation", "ai_accepted",
            # ES-EKF States
            "ekf_lat", "ekf_lon", "ekf_east_m", "ekf_north_m", "ekf_up_m",
            "ekf_vel_east_mps", "ekf_vel_north_mps", "ekf_vel_up_mps",
            "ekf_fwd_speed_mps", "fwd_speed_kmh",
            "quat_w", "quat_x", "quat_y", "quat_z",
            "ekf_roll_deg", "ekf_pitch_deg", "ekf_yaw_deg",
            "heading_deg", "heading_rad", "lean_angle_deg",
            "bias_acc_x", "bias_acc_y", "bias_acc_z",
            "bias_gyro_x", "bias_gyro_y", "bias_gyro_z",
            "pos_uncertainty_1sigma_m", "vel_uncertainty_1sigma_mps",
            "nhc_active", "nhc_residual_lat", "nhc_residual_vert", "nhc_accepted",
            "gnss_vel_update_accepted", "attitude_update_accepted", "self_healing_reanchored",
            # Navigation Output
            "nav_mode", "is_in_blackout", "total_dr_distance_m", "trajectory_point_accepted",
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
            # Raw Phone
            safe_val(self.acc_phone_x), safe_val(self.acc_phone_y), safe_val(self.acc_phone_z),
            safe_val(self.gyro_phone_x), safe_val(self.gyro_phone_y), safe_val(self.gyro_phone_z),
            safe_val(self.mag_x), safe_val(self.mag_y), safe_val(self.mag_z),
            safe_val(self.raw_alpha), safe_val(self.raw_beta), safe_val(self.raw_gamma),
            "1" if self.is_absolute else ("0" if self.is_absolute is not None else ""),
            "1" if self.has_webkit_heading else ("0" if self.has_webkit_heading is not None else ""),
            safe_val(self.webkit_compass_heading), self.orientation_event_type or "", safe_val(self.screen_orientation_angle),
            safe_val(self.server_receive_time),
            safe_val(self.orientation_yaw), safe_val(self.orientation_pitch), safe_val(self.orientation_roll),
            # Timing
            safe_val(self.prev_timestamp), safe_val(self.actual_dt_used), self.dt_source, "1" if self.is_out_of_order else "0",
            # Vehicle Frame & Alignment
            safe_val(self.acc_veh_x), safe_val(self.acc_veh_y), safe_val(self.acc_veh_z),
            safe_val(self.gyro_veh_x), safe_val(self.gyro_veh_y), safe_val(self.gyro_veh_z),
            self.calibration_state, "1" if self.is_calibrated else "0", "1" if self.alignment_recalculated else "0",
            safe_val(self.R_p2v_00), safe_val(self.R_p2v_01), safe_val(self.R_p2v_02),
            safe_val(self.R_p2v_10), safe_val(self.R_p2v_11), safe_val(self.R_p2v_12),
            safe_val(self.R_p2v_20), safe_val(self.R_p2v_21), safe_val(self.R_p2v_22),
            # Motion & Stationary
            safe_val(self.linear_acc_x), safe_val(self.linear_acc_y), safe_val(self.linear_acc_z), safe_val(self.acc_magnitude),
            "1" if self.is_stationary else "0", safe_val(self.stationary_variance),
            "1" if self.zupt_active else "0", "1" if self.zaru_active else "0",
            # GNSS
            "1" if self.has_new_gnss else "0",
            safe_val(self.gnss_lat), safe_val(self.gnss_lon), safe_val(self.gnss_alt),
            safe_val(self.gnss_speed_mps), safe_val(self.gnss_heading_deg), safe_val(self.gnss_accuracy_m), safe_val(self.gnss_timestamp),
            safe_val(self.gnss_age_sec), safe_val(self.gnss_trust_score), self.gnss_trust_status, "1" if self.gnss_is_trusted else "0",
            # AI
            str(self.ai_window_sample_count), self.ai_input_scaling_status, safe_val(self.ai_speed_mps),
            "1" if self.ai_is_ready else "0", "1" if self.ai_has_new_inference else "0",
            safe_val(self.ai_confidence_sigma), safe_val(self.ai_innovation), "1" if self.ai_accepted else "0",
            # ES-EKF States
            safe_val(self.ekf_lat), safe_val(self.ekf_lon), safe_val(self.ekf_east_m), safe_val(self.ekf_north_m), safe_val(self.ekf_up_m),
            safe_val(self.ekf_vel_east_mps), safe_val(self.ekf_vel_north_mps), safe_val(self.ekf_vel_up_mps),
            safe_val(self.ekf_fwd_speed_mps), safe_val(self.fwd_speed_kmh),
            safe_val(self.quat_w), safe_val(self.quat_x), safe_val(self.quat_y), safe_val(self.quat_z),
            safe_val(self.ekf_roll_deg), safe_val(self.ekf_pitch_deg), safe_val(self.ekf_yaw_deg),
            safe_val(self.heading_deg), safe_val(self.heading_rad), safe_val(self.lean_angle_deg),
            safe_val(self.bias_acc_x), safe_val(self.bias_acc_y), safe_val(self.bias_acc_z),
            safe_val(self.bias_gyro_x), safe_val(self.bias_gyro_y), safe_val(self.bias_gyro_z),
            safe_val(self.pos_uncertainty_1sigma_m), safe_val(self.vel_uncertainty_1sigma_mps),
            "1" if self.nhc_active else "0", safe_val(self.nhc_residual_lat), safe_val(self.nhc_residual_vert), "1" if self.nhc_accepted else "0",
            "1" if self.gnss_vel_update_accepted else "0", "1" if self.attitude_update_accepted else "0", "1" if self.self_healing_reanchored else "0",
            # Navigation Output
            self.nav_mode, "1" if self.is_in_blackout else "0",
            safe_val(self.total_dr_distance_m), "1" if self.trajectory_point_accepted else "0",
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
