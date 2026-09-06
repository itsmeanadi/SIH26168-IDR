"""High-Performance Streaming Experiment Recorder for Real-World Field Validation.

Streams synchronized mixed-rate telemetry (Raw IMU + GNSS + Calibrated IMU + AI + EKF)
to structured session archives without blocking the 50 Hz navigation loop.
"""

import csv
from dataclasses import asdict
import datetime
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional
import numpy as np

from idr.engine.navigation_engine import GNSSInputFix, NavigationOutputState, SensorInputFrame
from .session import EventMarker, ExperimentSession, SessionMetadata, TelemetryRecord

logger = logging.getLogger(__name__)


class ExperimentRecorder:
    """Manages active experimental recording sessions and buffered disk streaming."""

    def __init__(self, output_base_dir: str = "data/sessions", buffer_flush_size: int = 50):
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.buffer_flush_size = buffer_flush_size

        self.is_recording = False
        self.active_session: Optional[ExperimentSession] = None
        self.session_dir: Optional[Path] = None
        self.csv_file = None
        self.csv_writer = None

        self._record_buffer: List[TelemetryRecord] = []
        self._lock = threading.RLock()
        self._last_event_marker: Optional[str] = None
        self._start_perf_time: Optional[float] = None
        self._last_gnss_timestamp: Optional[float] = None
        self._last_ai_speed: float = 0.0

    def start_session(
        self,
        session_id: Optional[str] = None,
        notes: Optional[str] = None,
        vehicle_type: str = "two_wheeler",
        device_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Start a new timestamped telemetry recording session."""
        with self._lock:
            if self.is_recording:
                self.stop_session()

            now = datetime.datetime.now(datetime.timezone.utc)
            if not session_id:
                session_id = f"exp_{now.strftime('%Y%m%d_%H%M%S')}_{vehicle_type}"

            # Anonymize device info (no personal identifiers)
            clean_device_info = {}
            if device_info:
                allowed_keys = ["userAgent", "platform", "screenResolution", "sampleRateHz", "browser"]
                clean_device_info = {k: device_info[k] for k in allowed_keys if k in device_info}

            metadata = SessionMetadata(
                session_id=session_id,
                start_time_iso=now.isoformat(),
                vehicle_type=vehicle_type,
                notes=notes,
                device_info=clean_device_info,
            )

            self.active_session = ExperimentSession(metadata=metadata)
            self.session_dir = self.output_base_dir / session_id
            self.session_dir.mkdir(parents=True, exist_ok=True)

            # Initialize CSV file and write header
            csv_path = self.session_dir / "telemetry.csv"
            self.csv_file = open(csv_path, mode="w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(TelemetryRecord.csv_header())
            self.csv_file.flush()

            self._record_buffer.clear()
            self._last_event_marker = "SESSION_START"
            self._start_perf_time = time.perf_counter()
            self.is_recording = True

            logger.info(f"Started experiment recording session: {session_id} -> {self.session_dir}")
            return session_id

    def add_marker(self, label: str, notes: Optional[str] = None):
        """Annotate the active session with an event or maneuver marker."""
        with self._lock:
            if not self.is_recording or self.active_session is None:
                return

            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
            t = time.perf_counter() - (self._start_perf_time or 0.0)
            marker = EventMarker(timestamp=t, wall_time=now_iso, label=label, notes=notes)
            self.active_session.metadata.event_markers.append(asdict(marker))
            self._last_event_marker = label
            logger.info(f"Added experiment marker: [{label}] (t={t:.2f}s) - {notes or ''}")

    def record_frame(
        self,
        imu: SensorInputFrame,
        gnss: Optional[GNSSInputFix],
        nav_state: NavigationOutputState,
        step_latency_ms: float = 0.0,
    ):
        """Record a single high-frequency navigation frame."""
        if not self.is_recording or self.active_session is None:
            return

        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        t = float(imu.timestamp)

        # Detect if this frame contains a genuinely new GNSS fix
        has_new_gnss = False
        gnss_age = 0.0
        if gnss is not None:
            if self._last_gnss_timestamp is None or (gnss.timestamp is not None and gnss.timestamp != self._last_gnss_timestamp):
                has_new_gnss = True
                self._last_gnss_timestamp = gnss.timestamp
            gnss_age = max(0.0, t - float(gnss.timestamp)) if gnss.timestamp is not None else 0.0

        # Diagnostics data if available
        ai_speed = 0.0
        total_dr = 0.0
        calib_state = "CALIBRATED" if nav_state.is_aligned else "NOT_CALIBRATED"
        if nav_state.diagnostics:
            ai_speed = float(nav_state.diagnostics.ai_speed_mps)
            total_dr = float(nav_state.diagnostics.total_dr_distance_m)
            if hasattr(nav_state.diagnostics, "phone_alignment_state"):
                calib_state = str(nav_state.diagnostics.phone_alignment_state)

        # Detect AI inference progression
        ai_has_new = False
        if ai_speed != self._last_ai_speed:
            ai_has_new = True
            self._last_ai_speed = ai_speed

        marker_to_log = self._last_event_marker
        self._last_event_marker = None  # Consume single-frame marker

        record = TelemetryRecord(
            timestamp=t,
            wall_time=now_iso,
            # Raw Phone
            acc_phone_x=float(imu.acc_x),
            acc_phone_y=float(imu.acc_y),
            acc_phone_z=float(imu.acc_z),
            gyro_phone_x=float(imu.gyro_x),
            gyro_phone_y=float(imu.gyro_y),
            gyro_phone_z=float(imu.gyro_z),
            mag_x=imu.mag_x,
            mag_y=imu.mag_y,
            mag_z=imu.mag_z,
            orientation_yaw=imu.orientation_yaw,
            orientation_pitch=imu.orientation_pitch,
            orientation_roll=imu.orientation_roll,
            # Calibrated Vehicle Frame
            acc_veh_x=float(imu.acc_x),
            acc_veh_y=float(imu.acc_y),
            acc_veh_z=float(imu.acc_z),
            gyro_veh_x=float(imu.gyro_x),
            gyro_veh_y=float(imu.gyro_y),
            gyro_veh_z=float(imu.gyro_z),
            # Calibration State
            calibration_state=calib_state,
            is_calibrated=nav_state.is_aligned,
            # GNSS
            has_new_gnss=has_new_gnss,
            gnss_lat=gnss.latitude if gnss else None,
            gnss_lon=gnss.longitude if gnss else None,
            gnss_alt=gnss.altitude if gnss else None,
            gnss_speed_mps=gnss.speed_mps if gnss else None,
            gnss_heading_deg=gnss.heading_deg if gnss else None,
            gnss_accuracy_m=gnss.accuracy_m if gnss else None,
            gnss_age_sec=gnss_age,
            gnss_trust_score=float(nav_state.gnss_trust_score),
            gnss_trust_status=nav_state.gnss_status,
            gnss_is_trusted=(nav_state.gnss_trust_score > 0.7),
            # AI
            ai_speed_mps=ai_speed,
            ai_is_ready=True if ai_speed > 0.01 else False,
            ai_has_new_inference=ai_has_new,
            # EKF State
            ekf_lat=float(nav_state.latitude),
            ekf_lon=float(nav_state.longitude),
            ekf_east_m=0.0,
            ekf_north_m=0.0,
            ekf_vel_east_mps=float(nav_state.velocity_east),
            ekf_vel_north_mps=float(nav_state.velocity_north),
            ekf_fwd_speed_mps=float(nav_state.forward_speed_mps),
            heading_deg=float(nav_state.heading_deg),
            lean_angle_deg=float(nav_state.lean_angle_deg),
            nav_mode=nav_state.nav_mode.value if hasattr(nav_state.nav_mode, 'value') else str(nav_state.nav_mode),
            is_in_blackout=nav_state.is_in_blackout,
            is_stationary=nav_state.is_stationary,
            nhc_active=nav_state.is_in_blackout and not nav_state.is_stationary,
            zupt_active=nav_state.is_stationary,
            zaru_active=nav_state.is_stationary,
            total_dr_distance_m=total_dr,
            step_latency_ms=step_latency_ms,
            event_marker=marker_to_log,
        )

        with self._lock:
            self._record_buffer.append(record)
            self.active_session.metadata.record_count += 1
            if len(self._record_buffer) >= self.buffer_flush_size:
                self._flush_buffer_to_disk()

    def _flush_buffer_to_disk(self):
        """Flush in-memory record buffer to disk CSV."""
        if not self.csv_writer or not self._record_buffer:
            return
        for rec in self._record_buffer:
            self.csv_writer.writerow(rec.to_csv_row())
        self.csv_file.flush()
        self._record_buffer.clear()

    def stop_session(self) -> Optional[Dict[str, Any]]:
        """Stop active session and finalize metadata / summary stats."""
        with self._lock:
            if not self.is_recording or self.active_session is None:
                return None

            now = datetime.datetime.now(datetime.timezone.utc)
            self.active_session.metadata.end_time_iso = now.isoformat()

            # Compute session duration
            if self._start_perf_time is not None:
                self.active_session.metadata.duration_sec = round(time.perf_counter() - self._start_perf_time, 2)

            self.add_marker("SESSION_STOP")
            self._flush_buffer_to_disk()

            if self.csv_file:
                self.csv_file.close()
                self.csv_file = None
                self.csv_writer = None

            # Calculate summary stats from CSV
            summary = self._compute_and_save_metadata()
            session_id = self.active_session.metadata.session_id

            self.is_recording = False
            self.active_session = None
            self.session_dir = None
            self._start_perf_time = None
            self._last_gnss_timestamp = None

            logger.info(f"Stopped experiment session: {session_id} ({summary.get('duration_sec', 0)}s, {summary.get('record_count', 0)} records)")
            return summary

    def _compute_and_save_metadata(self) -> Dict[str, Any]:
        """Compute final session statistics and write metadata.json."""
        if not self.session_dir or not self.active_session:
            return {}

        meta = self.active_session.metadata
        csv_path = self.session_dir / "telemetry.csv"

        # Compute summary stats from written CSV if possible
        stats = {
            "total_records": meta.record_count,
            "duration_sec": meta.duration_sec,
            "average_rate_hz": round(meta.record_count / meta.duration_sec, 2) if meta.duration_sec > 0 else 0.0,
        }

        if csv_path.exists() and meta.record_count > 0:
            try:
                import pandas as pd
                df = pd.read_csv(csv_path)
                stats["max_speed_kmh"] = round(float(df["ekf_fwd_speed_mps"].max() * 3.6), 2) if "ekf_fwd_speed_mps" in df else 0.0
                stats["mean_speed_kmh"] = round(float(df["ekf_fwd_speed_mps"].mean() * 3.6), 2) if "ekf_fwd_speed_mps" in df else 0.0
                stats["max_ai_speed_kmh"] = round(float(df["ai_speed_mps"].max() * 3.6), 2) if "ai_speed_mps" in df else 0.0
                stats["max_lean_angle_deg"] = round(float(df["lean_angle_deg"].abs().max()), 2) if "lean_angle_deg" in df else 0.0
                stats["total_dr_distance_m"] = round(float(df["total_dr_distance_m"].max()), 2) if "total_dr_distance_m" in df else 0.0
                stats["gnss_fixes_count"] = int(df["has_new_gnss"].sum()) if "has_new_gnss" in df else 0
            except Exception as e:
                logger.warning(f"Could not compute extended stats: {e}")

        meta.stats = stats
        meta_path = self.session_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta.to_json_dict(), f, indent=2)

        return meta.to_json_dict()

    def get_status(self) -> Dict[str, Any]:
        """Get current recorder status for telemetry polling."""
        if not self.is_recording or self.active_session is None:
            return {"is_recording": False, "active_session": None}

        duration = time.perf_counter() - (self._start_perf_time or 0.0)
        return {
            "is_recording": True,
            "session_id": self.active_session.metadata.session_id,
            "duration_sec": round(duration, 1),
            "record_count": self.active_session.metadata.record_count,
            "vehicle_type": self.active_session.metadata.vehicle_type,
            "notes": self.active_session.metadata.notes,
        }

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all completed recording sessions stored in data/sessions/."""
        sessions = []
        if not self.output_base_dir.exists():
            return []

        for p in sorted(self.output_base_dir.glob("exp_*"), reverse=True):
            if p.is_dir():
                meta_file = p / "metadata.json"
                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                            sessions.append(meta)
                    except Exception:
                        pass
                else:
                    sessions.append({"session_id": p.name, "status": "INCOMPLETE"})
        return sessions
