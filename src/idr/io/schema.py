"""Schema auto-detection and validation for IO-VNBD dataset CSV files."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Canonical field names
CANONICAL_FIELDS = [
    "timestamp",
    "acc_x", "acc_y", "acc_z",
    "gyro_x", "gyro_y", "gyro_z",
    "mag_x", "mag_y", "mag_z",
    "lat", "lon",
    "speed", "heading",
    "wheel_speed", "yaw_rate",
]

# Field alias dictionary for robust matching without single-character substring collisions
FIELD_ALIASES: Dict[str, List[str]] = {
    "timestamp": ["time_since_start_ms", "time_since_start_of_day_seconds", "time_since_start", "timestamp", "time_stamp", "time_secs", "time"],
    "acc_x": ["accelerometer_x", "linear_acceleration_x", "acc_x", "accx", "accel_x"],
    "acc_y": ["accelerometer_y", "linear_acceleration_y", "acc_y", "accy", "accel_y"],
    "acc_z": ["accelerometer_z", "linear_acceleration_z", "acc_z", "accz", "accel_z"],
    "gyro_x": ["gyroscope_roll", "angular_velocity_x", "gyro_x", "gyrox", "gyroscope_x"],
    "gyro_y": ["gyroscope_pitch", "angular_velocity_y", "gyro_y", "gyroy", "gyroscope_y"],
    "gyro_z": ["gyroscope_yaw", "angular_velocity_z", "gyro_z", "gyroz", "gyroscope_z"],
    "mag_x": ["magnetic_field_x", "magnetometer_x", "mag_x", "magx"],
    "mag_y": ["magnetic_field_y", "magnetometer_y", "mag_y", "magy"],
    "mag_z": ["magnetic_field_z", "magnetometer_z", "mag_z", "magz"],
    "lat": ["gps_latitude", "latitude"],
    "lon": ["gps_longitude", "longitude"],
    "speed": ["gps_speed", "ground_speed", "gps_vel"],
    "heading": ["gps_orientation", "gps_bearing", "bearing", "gps_heading", "course"],
    "wheel_speed": ["indicated_vehicle_speed", "velocity_km_hr", "wheel_speed", "wheelspeed", "vehicle_speed", "velocity"],
    "yaw_rate": ["yaw_rate", "yawrate", "yaw_speed"],
}

@dataclass
class SchemaMap:
    """Mapping from canonical field names to actual CSV column names."""
    field_to_col: Dict[str, str]
    file_type: str  # 'smartphone' or 'vehicle'

    def has_required_imu(self) -> bool:
        required = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
        return all(f in self.field_to_col for f in required)

    def has_required_gps(self) -> bool:
        return "lat" in self.field_to_col and "lon" in self.field_to_col

def normalize_col_name(col: str) -> str:
    """Normalize string for robust alias matching."""
    cleaned = col.strip().lower()
    for char in ["(", ")", "[", "]", "{", "}", "/", "\\", "-", ".", "°", "²", "μ", "µ"]:
        cleaned = cleaned.replace(char, " ")
    return "_".join(cleaned.split())

def detect_schema(csv_path_or_df, file_type_hint: Optional[str] = None) -> SchemaMap:
    """Auto-detect column mapping for an IO-VNBD CSV file.
    
    Fails loudly if essential IMU or GPS fields cannot be resolved.
    """
    if isinstance(csv_path_or_df, pd.DataFrame):
        columns = list(csv_path_or_df.columns)
    else:
        sample = pd.read_csv(csv_path_or_df, nrows=5, encoding="latin1")
        columns = list(sample.columns)

    normalized_cols = {normalize_col_name(c): c for c in columns}
    field_to_col: Dict[str, str] = {}
    used_cols = set()

    # Pass 1: Exact normalized match
    for field, aliases in FIELD_ALIASES.items():
        if field in field_to_col:
            continue
        for alias in aliases:
            norm_alias = normalize_col_name(alias)
            if norm_alias in normalized_cols and normalized_cols[norm_alias] not in used_cols:
                orig_col = normalized_cols[norm_alias]
                field_to_col[field] = orig_col
                used_cols.add(orig_col)
                break

    # Pass 2: Substring matching (longer aliases first to prevent partial collision)
    for field, aliases in FIELD_ALIASES.items():
        if field in field_to_col:
            continue
        sorted_aliases = sorted(aliases, key=len, reverse=True)
        for alias in sorted_aliases:
            norm_alias = normalize_col_name(alias)
            if len(norm_alias) < 3:
                continue
            matched = False
            for norm_c, orig_c in normalized_cols.items():
                if orig_c in used_cols:
                    continue
                # Word-level containment
                if norm_alias in norm_c.split("_") or f"_{norm_alias}_" in f"_{norm_c}_":
                    field_to_col[field] = orig_c
                    used_cols.add(orig_c)
                    matched = True
                    break
            if matched:
                break

    # Determine file type if not provided
    if file_type_hint:
        file_type = file_type_hint
    else:
        file_type = "vehicle" if ("wheel_speed" in field_to_col and "acc_x" not in field_to_col) else "smartphone"

    schema = SchemaMap(field_to_col=field_to_col, file_type=file_type)

    logger.info(f"Detected schema for [{file_type}]: mapped {len(field_to_col)}/{len(CANONICAL_FIELDS)} canonical fields.")
    for k, v in field_to_col.items():
        logger.debug(f"  {k} -> '{v}'")

    # Validate essential requirements
    if file_type == "smartphone":
        if not schema.has_required_imu():
            missing = [f for f in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"] if f not in field_to_col]
            raise ValueError(
                f"FATAL SCHEMA ERROR: Required smartphone IMU fields missing: {missing}. Available columns: {columns}"
            )
        if not schema.has_required_gps():
            logger.warning("GPS coordinates (lat/lon) missing or unmapped in this smartphone file.")
    elif file_type == "vehicle":
        if "wheel_speed" not in field_to_col and "speed" not in field_to_col:
            raise ValueError(
                f"FATAL SCHEMA ERROR: Required vehicle speed field missing. Available columns: {columns}"
            )

    return schema
