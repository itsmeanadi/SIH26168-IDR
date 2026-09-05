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

# Field alias dictionary for flexible regex/substring matching
FIELD_ALIASES: Dict[str, List[str]] = {
    "timestamp": ["time", "timestamp", "t", "epoch", "time_stamp", "secs"],
    "acc_x": ["acc_x", "accx", "acceleration_x", "linear_acceleration_x", "accel_x", "a_x"],
    "acc_y": ["acc_y", "accy", "acceleration_y", "linear_acceleration_y", "accel_y", "a_y"],
    "acc_z": ["acc_z", "accz", "acceleration_z", "linear_acceleration_z", "accel_z", "a_z"],
    "gyro_x": ["gyro_x", "gyrox", "angular_velocity_x", "gyroscope_x", "w_x", "omega_x"],
    "gyro_y": ["gyro_y", "gyroy", "angular_velocity_y", "gyroscope_y", "w_y", "omega_y"],
    "gyro_z": ["gyro_z", "gyroz", "angular_velocity_z", "gyroscope_z", "w_z", "omega_z", "yaw_rate", "yawrate"],
    "mag_x": ["mag_x", "magx", "magnetic_field_x", "magnetometer_x", "m_x"],
    "mag_y": ["mag_y", "magy", "magnetic_field_y", "magnetometer_y", "m_y"],
    "mag_z": ["mag_z", "magz", "magnetic_field_z", "magnetometer_z", "m_z"],
    "lat": ["latitude", "lat", "gps_latitude", "pos_lat"],
    "lon": ["longitude", "lon", "lng", "gps_longitude", "pos_lon"],
    "speed": ["speed", "gps_speed", "ground_speed", "velocity", "gps_vel"],
    "heading": ["heading", "bearing", "course", "gps_heading", "yaw"],
    "wheel_speed": ["wheelspeed", "wheel_speed", "wspeed", "v_wheel", "vehicle_speed", "ecu_speed"],
    "yaw_rate": ["yawrate", "yaw_rate", "yaw_speed", "gyro_z_vehicle"],
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
    return col.strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_")

def detect_schema(csv_path_or_df, file_type_hint: Optional[str] = None) -> SchemaMap:
    """Auto-detect column mapping for an IO-VNBD CSV file.
    
    Fails loudly if essential IMU or GPS fields cannot be resolved.
    """
    if isinstance(csv_path_or_df, pd.DataFrame):
        columns = list(csv_path_or_df.columns)
    else:
        sample = pd.read_csv(csv_path_or_df, nrows=5)
        columns = list(sample.columns)

    normalized_cols = {normalize_col_name(c): c for c in columns}
    field_to_col: Dict[str, str] = {}

    for field, aliases in FIELD_ALIASES.items():
        matched = False
        # Exact alias matching
        for alias in aliases:
            norm_alias = normalize_col_name(alias)
            if norm_alias in normalized_cols:
                field_to_col[field] = normalized_cols[norm_alias]
                matched = True
                break
        
        # Substring matching fallback if not matched
        if not matched:
            for norm_c, orig_c in normalized_cols.items():
                for alias in aliases:
                    if alias in norm_c:
                        field_to_col[field] = orig_c
                        matched = True
                        break
                if matched:
                    break

    # Determine file type if not provided
    if file_type_hint:
        file_type = file_type_hint
    else:
        file_type = "vehicle" if "wheel_speed" in field_to_col else "smartphone"

    schema = SchemaMap(field_to_col=field_to_col, file_type=file_type)

    logger.info(f"Detected schema for [{file_type}]: mapped {len(field_to_col)}/{len(CANONICAL_FIELDS)} canonical fields.")
    for k, v in field_to_col.items():
        logger.debug(f"  {k} -> '{v}'")

    # Validate essential requirements
    if not schema.has_required_imu():
        missing = [f for f in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"] if f not in field_to_col]
        raise ValueError(
            f"FATAL SCHEMA ERROR: Required IMU fields missing: {missing}. Available columns: {columns}"
        )
    
    if not schema.has_required_gps():
        logger.warning("GPS coordinates (lat/lon) missing or unmapped in this file.")

    return schema
