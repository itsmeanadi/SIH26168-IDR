from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
import numpy as np
import pandas as pd
from .schema import detect_schema, SchemaMap
from .sync import parse_phone_date_to_seconds, synchronize_and_interpolate_drive

logger = logging.getLogger(__name__)

@dataclass
class IOVNBDrive:
    """Encapsulates a synchronized drive recording (smartphone + vehicle ECU)."""
    drive_id: str
    phone_df: pd.DataFrame
    vehicle_df: Optional[pd.DataFrame] = None
    phone_schema: Optional[SchemaMap] = None
    vehicle_schema: Optional[SchemaMap] = None
    sync_metadata: Optional[Dict[str, float]] = None

    def get_synced_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract synchronized arrays within valid temporal overlap:
        - phone_imu: (N, 6) [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        - phone_gps: (N, 4) [lat, lon, speed, heading]
        - vehicle_speed: (N,) forward speed (m/s)
        - timestamps: (N,) relative seconds from start of aligned segment
        """
        # Align streams temporally and interpolate vehicle speeds
        aligned_phone_df, aligned_v_speed, meta = synchronize_and_interpolate_drive(
            self.phone_df, self.vehicle_df
        )
        self.sync_metadata = meta

        N = len(aligned_phone_df)
        phone_imu = aligned_phone_df[["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]].to_numpy(dtype=np.float32)
        
        # GPS coordinates
        gps_cols = ["lat", "lon"]
        if "speed" in aligned_phone_df.columns:
            gps_cols.append("speed")
        if "heading" in aligned_phone_df.columns:
            gps_cols.append("heading")
        phone_gps = aligned_phone_df[gps_cols].to_numpy(dtype=np.float64)

        t = aligned_phone_df["timestamp"].to_numpy(dtype=np.float64)
        return phone_imu, phone_gps, aligned_v_speed, t

    def get_synced_data_with_mag(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract synchronized arrays including 3D magnetometer readings:
        - phone_imu: (N, 6) [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        - phone_gps: (N, 4) [lat, lon, speed, heading]
        - vehicle_speed: (N,) forward speed (m/s)
        - timestamps: (N,) relative seconds from start of aligned segment
        - phone_mag: (N, 3) [mag_x, mag_y, mag_z] in uT
        """
        aligned_phone_df, aligned_v_speed, meta = synchronize_and_interpolate_drive(
            self.phone_df, self.vehicle_df
        )
        self.sync_metadata = meta

        phone_imu = aligned_phone_df[["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]].to_numpy(dtype=np.float32)
        
        gps_cols = ["lat", "lon"]
        if "speed" in aligned_phone_df.columns:
            gps_cols.append("speed")
        if "heading" in aligned_phone_df.columns:
            gps_cols.append("heading")
        phone_gps = aligned_phone_df[gps_cols].to_numpy(dtype=np.float64)

        t = aligned_phone_df["timestamp"].to_numpy(dtype=np.float64)
        
        if all(c in aligned_phone_df.columns for c in ["mag_x", "mag_y", "mag_z"]):
            phone_mag = aligned_phone_df[["mag_x", "mag_y", "mag_z"]].to_numpy(dtype=np.float64)
        else:
            phone_mag = np.zeros((len(phone_imu), 3), dtype=np.float64)
            
        return phone_imu, phone_gps, aligned_v_speed, t, phone_mag

def standardize_dataframe(df: pd.DataFrame, schema: SchemaMap) -> pd.DataFrame:
    """Rename columns to canonical names and ensure continuous timestamps."""
    inv_map = {col: field for field, col in schema.field_to_col.items()}
    renamed = df.rename(columns=inv_map)
    
    # Keep canonical names, not the original source names in the inverse map.
    keep_cols = [field for field in schema.field_to_col if field in renamed.columns]
    standardized = renamed[keep_cols].copy()

    # Parse continuous timestamp in seconds of day
    date_cols = [c for c in df.columns if "DATE" in c.upper()]
    if date_cols and schema.file_type == "smartphone":
        # Parse absolute date string to continuous seconds
        standardized["timestamp"] = parse_phone_date_to_seconds(df[date_cols[0]])
    elif "timestamp" in standardized.columns:
        t = standardized["timestamp"].to_numpy(dtype=np.float64)
        orig_col_name = schema.field_to_col.get("timestamp", "").lower()
        if "(ms)" in orig_col_name or "ms" in orig_col_name:
            t = t / 1000.0
        standardized["timestamp"] = t
    else:
        # Synthesize 10 Hz timestamp
        standardized["timestamp"] = np.arange(len(standardized)) * 0.1

    # Convert speed from km/h to m/s if recorded in km/h
    for spd_col in ["speed", "wheel_speed"]:
        if spd_col in standardized.columns:
            orig_spd_name = schema.field_to_col.get(spd_col, "").lower()
            if "km" in orig_spd_name or "km/h" in orig_spd_name or "kmh" in orig_spd_name or "velocity" in orig_spd_name:
                standardized[spd_col] = standardized[spd_col] / 3.6

    # Fill NaN / interpolate missing
    standardized = standardized.interpolate(method="linear").bfill().ffill()
    return standardized

def load_drive_pair(drive_dir: Path, drive_id: str) -> IOVNBDrive:
    """Locate and load S-*.csv and V-*.csv for a given drive identifier."""
    drive_dir = Path(drive_dir)
    phone_files = list(drive_dir.glob(f"S-{drive_id}.csv")) or list(drive_dir.glob("S-*.csv"))
    vehicle_files = list(drive_dir.glob(f"V-{drive_id}.csv")) or list(drive_dir.glob("V-*.csv"))

    if not phone_files:
        raise FileNotFoundError(f"No smartphone file (S-*.csv) found in {drive_dir}")

    # Load smartphone data
    phone_path = phone_files[0]
    logger.info(f"Loading smartphone data from: {phone_path}")
    raw_phone_df = pd.read_csv(phone_path, encoding="latin1")
    phone_schema = detect_schema(raw_phone_df, file_type_hint="smartphone")
    std_phone_df = standardize_dataframe(raw_phone_df, phone_schema)

    # Load vehicle data if available
    std_vehicle_df = None
    vehicle_schema = None
    if vehicle_files:
        vehicle_path = vehicle_files[0]
        logger.info(f"Loading vehicle ECU data from: {vehicle_path}")
        raw_vehicle_df = pd.read_csv(vehicle_path, encoding="latin1")
        try:
            vehicle_schema = detect_schema(raw_vehicle_df, file_type_hint="vehicle")
            std_vehicle_df = standardize_dataframe(raw_vehicle_df, vehicle_schema)
        except Exception as e:
            logger.warning(f"Could not fully parse vehicle file {vehicle_path}: {e}")

    return IOVNBDrive(
        drive_id=drive_id,
        phone_df=std_phone_df,
        vehicle_df=std_vehicle_df,
        phone_schema=phone_schema,
        vehicle_schema=vehicle_schema,
    )
