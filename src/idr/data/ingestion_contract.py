"""Authentic IO-VNBD Ingestion Contract and Schema Validator.

Defines the mathematical and schema invariants required for any authentic IO-VNBD dataset
before ingestion into preprocessing or training.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np
import pandas as pd
from ..io.schema import detect_schema, SchemaMap

logger = logging.getLogger(__name__)


class IngestionContractError(Exception):
    """Raised when an ingested drive violates the physical or schema contract."""
    pass


@dataclass
class DriveValidationReport:
    drive_id: str
    is_valid: bool
    num_samples: int
    duration_sec: float
    imu_sampling_rate_hz: float
    gps_sampling_rate_hz: float
    has_vehicle_speed: bool
    has_vehicle_accel: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class IOVNBDDatasetIngestionContract:
    """Strict schema and physical range validator for authentic IO-VNBD drives."""

    # Physical range constants for two-wheeler / automotive motion
    MAX_ACCEL_MPS2 = 50.0       # > 5g is non-physical or catastrophic crash shock
    MAX_GYRO_RADS = 25.0        # > 1400 deg/s is non-physical sensor glitch
    MAX_SPEED_MPS = 80.0        # ~288 km/h upper bound for production vehicles
    MIN_IMU_RATE_HZ = 40.0      # Smartphone IMU should deliver at least 40 Hz
    MAX_IMU_RATE_HZ = 250.0     # Smartphone IMU cap
    MIN_SAMPLES_COUNT = 50      # At least 5s of data

    @classmethod
    def validate_drive_folder(cls, folder_path: Path, drive_id: str) -> DriveValidationReport:
        """Validate all files and sensor streams in a drive folder against the contract."""
        folder = Path(folder_path)
        errors: List[str] = []
        warnings: List[str] = []

        if not folder.exists() or not folder.is_dir():
            return DriveValidationReport(
                drive_id=drive_id,
                is_valid=False,
                num_samples=0,
                duration_sec=0.0,
                imu_sampling_rate_hz=0.0,
                gps_sampling_rate_hz=0.0,
                has_vehicle_speed=False,
                has_vehicle_accel=False,
                errors=[f"Drive folder does not exist: {folder}"],
            )

        csv_files = list(folder.glob("*.csv"))
        if not csv_files:
            return DriveValidationReport(
                drive_id=drive_id,
                is_valid=False,
                num_samples=0,
                duration_sec=0.0,
                imu_sampling_rate_hz=0.0,
                gps_sampling_rate_hz=0.0,
                has_vehicle_speed=False,
                has_vehicle_accel=False,
                errors=[f"No CSV files found in drive folder: {folder}"],
            )

        # Check for combined S-*.csv / V-*.csv files or individual channel files
        s_combined = [f for f in csv_files if (f.stem.startswith("S-") or f.stem.startswith("S_")) and ("acc" not in f.stem.lower() and "gyro" not in f.stem.lower() and "gps" not in f.stem.lower())]
        v_combined = [f for f in csv_files if (f.stem.startswith("V-") or f.stem.startswith("V_")) and ("speed" not in f.stem.lower() and "acc" not in f.stem.lower())]

        phone_df: Optional[pd.DataFrame] = None
        vehicle_df: Optional[pd.DataFrame] = None

        if s_combined:
            phone_df = pd.read_csv(s_combined[0])
            s_schema = detect_schema(phone_df, file_type_hint="smartphone")
            if not s_schema.has_required_imu():
                errors.append(f"Combined smartphone file {s_combined[0].name} missing required 6-DOF IMU channels")
            if not s_schema.has_required_gps():
                warnings.append(f"Combined smartphone file {s_combined[0].name} missing GPS lat/lon channels")
        else:
            # Check individual channel files
            file_map: Dict[str, Path] = {}
            for f in csv_files:
                lower = f.name.lower()
                if "phone_acc" in lower or lower.startswith("s-acc") or "acc" in lower:
                    file_map["phone_acc"] = f
                elif "phone_gyro" in lower or lower.startswith("s-gyro") or "gyro" in lower:
                    file_map["phone_gyro"] = f
                elif "phone_gps" in lower or lower.startswith("s-gps") or "gps" in lower:
                    file_map["phone_gps"] = f

            if "phone_acc" not in file_map:
                errors.append("Missing required phone stream: 'phone_acc'")
            if "phone_gyro" not in file_map:
                errors.append("Missing required phone stream: 'phone_gyro'")
            if "phone_gps" not in file_map:
                warnings.append("Missing phone stream: 'phone_gps'")

            if "phone_acc" in file_map:
                phone_df = pd.read_csv(file_map["phone_acc"])

        # Check vehicle ground-truth stream
        if v_combined:
            vehicle_df = pd.read_csv(v_combined[0])
            v_schema = detect_schema(vehicle_df, file_type_hint="vehicle")
            if "wheel_speed" not in v_schema.field_to_col and "speed" not in v_schema.field_to_col:
                errors.append(f"Combined vehicle file {v_combined[0].name} missing vehicle speed ground-truth column")
        else:
            v_speed_files = [f for f in csv_files if "vehicle_speed" in f.name.lower() or f.name.lower().startswith("v-speed") or "speed" in f.name.lower()]
            if not v_speed_files:
                # Check if phone file itself has vehicle speed
                if phone_df is not None and ("speed" in phone_df.columns or "wheel_speed" in phone_df.columns):
                    vehicle_df = phone_df
                else:
                    errors.append("Missing required vehicle ground-truth stream: 'vehicle_speed'")
            else:
                vehicle_df = pd.read_csv(v_speed_files[0])

        if phone_df is None or len(phone_df) < cls.MIN_SAMPLES_COUNT:
            errors.append(f"Phone stream has insufficient samples: {len(phone_df) if phone_df is not None else 0} < {cls.MIN_SAMPLES_COUNT}")

        # Check timestamps, NaNs, and physical bounds
        duration = 0.0
        imu_rate = 0.0
        if phone_df is not None and len(phone_df) > 0:
            t_col = [c for c in phone_df.columns if "time" in c.lower() or c in ["t", "timestamp"]]
            if not t_col:
                errors.append("No timestamp column found in phone stream")
            else:
                t_arr = phone_df[t_col[0]].to_numpy(dtype=np.float64)
                if np.any(np.diff(t_arr) <= 0):
                    errors.append("Phone stream timestamps are not strictly monotonic")
                if len(t_arr) > 1:
                    duration = float(t_arr[-1] - t_arr[0])
                    dt_mean = float(np.mean(np.diff(t_arr)))
                    if dt_mean > 0:
                        imu_rate = 1.0 / dt_mean

            # Numeric columns check
            num_cols = phone_df.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                vals = phone_df[col].to_numpy()
                if np.any(np.isnan(vals)) or np.any(np.isinf(vals)):
                    errors.append(f"Phone stream column '{col}' contains NaN or Inf values")
                if "acc" in col.lower():
                    max_a = float(np.max(np.abs(vals)))
                    if max_a > cls.MAX_ACCEL_MPS2:
                        errors.append(f"Phone accelerometer column '{col}' exceeds physical bound: {max_a:.2f} m/s^2 > {cls.MAX_ACCEL_MPS2} m/s^2")
                elif "gyro" in col.lower():
                    max_g = float(np.max(np.abs(vals)))
                    if max_g > cls.MAX_GYRO_RADS:
                        errors.append(f"Phone gyroscope column '{col}' exceeds physical bound: {max_g:.2f} rad/s > {cls.MAX_GYRO_RADS} rad/s")

        # Vehicle speed check
        has_v_speed = False
        if vehicle_df is not None:
            v_col = [c for c in vehicle_df.columns if "speed" in c.lower() or "vel" in c.lower() or "wspeed" in c.lower()]
            if v_col:
                has_v_speed = True
                speeds = vehicle_df[v_col[0]].to_numpy(dtype=np.float64)
                if np.any(speeds < -1.0) or np.any(speeds > cls.MAX_SPEED_MPS):
                    errors.append(f"Vehicle speed values out of physical bounds: min={np.min(speeds):.2f}, max={np.max(speeds):.2f} m/s")

        gps_rate = 1.0

        return DriveValidationReport(
            drive_id=drive_id,
            is_valid=(len(errors) == 0),
            num_samples=len(phone_df) if phone_df is not None else 0,
            duration_sec=duration,
            imu_sampling_rate_hz=imu_rate,
            gps_sampling_rate_hz=gps_rate,
            has_vehicle_speed=has_v_speed,
            has_vehicle_accel=(vehicle_df is not None and any("acc" in c.lower() for c in vehicle_df.columns)),
            errors=errors,
            warnings=warnings,
        )
