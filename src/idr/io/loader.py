"""Loader for IO-VNBD synchronized drive pairs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import logging
import numpy as np
import pandas as pd
from .schema import detect_schema, SchemaMap

logger = logging.getLogger(__name__)

@dataclass
class IOVNBDrive:
    """Encapsulates a synchronized drive recording (smartphone + vehicle ECU)."""
    drive_id: str
    phone_df: pd.DataFrame
    vehicle_df: Optional[pd.DataFrame] = None
    phone_schema: Optional[SchemaMap] = None
    vehicle_schema: Optional[SchemaMap] = None

    def get_synced_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract synchronized arrays:
        - phone_imu: (N, 6) [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        - phone_gps: (N, 4) [lat, lon, speed, heading]
        - vehicle_speed: (N,) forward speed (m/s)
        - timestamps: (N,) seconds
        """
        N = len(self.phone_df)
        phone_imu = self.phone_df[["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]].to_numpy(dtype=np.float32)
        
        # GPS coordinates
        gps_cols = ["lat", "lon"]
        if "speed" in self.phone_df.columns:
            gps_cols.append("speed")
        if "heading" in self.phone_df.columns:
            gps_cols.append("heading")
        phone_gps = self.phone_df[gps_cols].to_numpy(dtype=np.float64)

        # Vehicle speed (ground truth)
        if self.vehicle_df is not None and "wheel_speed" in self.vehicle_df.columns:
            # Ensure length alignment
            v_speed = self.vehicle_df["wheel_speed"].to_numpy(dtype=np.float32)[:N]
            if len(v_speed) < N:
                v_speed = np.pad(v_speed, (0, N - len(v_speed)), mode="edge")
        elif "speed" in self.phone_df.columns:
            v_speed = self.phone_df["speed"].to_numpy(dtype=np.float32)
        else:
            v_speed = np.zeros(N, dtype=np.float32)

        t = self.phone_df["timestamp"].to_numpy(dtype=np.float64)
        return phone_imu, phone_gps, v_speed, t

def standardize_dataframe(df: pd.DataFrame, schema: SchemaMap) -> pd.DataFrame:
    """Rename columns to canonical names and ensure monotonically increasing timestamps."""
    inv_map = {col: field for field, col in schema.field_to_col.items()}
    renamed = df.rename(columns=inv_map)
    
    # Keep canonical names, not the original source names in the inverse map.
    keep_cols = [field for field in schema.field_to_col if field in renamed.columns]
    standardized = renamed[keep_cols].copy()

    # Normalize timestamp to seconds starting at 0 if epoch or milliseconds
    if "timestamp" in standardized.columns:
        t = standardized["timestamp"].to_numpy(dtype=np.float64)
        if np.nanmean(t) > 1e11:  # epoch ms
            t = t / 1000.0
        t = t - t[0]  # zero-based relative time
        standardized["timestamp"] = t
    else:
        # Synthesize 10 Hz timestamp
        standardized["timestamp"] = np.arange(len(standardized)) * 0.1

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
    raw_phone_df = pd.read_csv(phone_path)
    phone_schema = detect_schema(raw_phone_df, file_type_hint="smartphone")
    std_phone_df = standardize_dataframe(raw_phone_df, phone_schema)

    # Load vehicle data if available
    std_vehicle_df = None
    vehicle_schema = None
    if vehicle_files:
        vehicle_path = vehicle_files[0]
        logger.info(f"Loading vehicle ECU data from: {vehicle_path}")
        raw_vehicle_df = pd.read_csv(vehicle_path)
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
