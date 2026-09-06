"""Dataset & Session Replayer for Simulation and Verification.

Loads IO-VNBD synchronized drive datasets (e.g. Vf, M, S, Y1) or custom traces
and streams IMU + GNSS frames sequentially into the NavigationEngine with configurable
replay rates (1x, 2x, 5x, 10x) and interactive blackout/jump injection.
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .navigation_engine import GNSSInputFix, NavigationEngine, NavigationOutputState, SensorInputFrame


class DriveReplayer:
    """Streams drive sessions through the NavigationEngine."""

    def __init__(self, engine: NavigationEngine):
        self.engine = engine
        self.is_playing = False
        self.playback_speed = 1.0
        self.current_index = 0
        self.data_frames: List[Tuple[SensorInputFrame, Optional[GNSSInputFix]]] = []
        self.drive_name = ""

    def load_iovnbd_drive(self, drive_name: str = "Vf", data_dir: str = "data/raw/categorised", reset_engine: bool = True) -> bool:
        """Load an IO-VNBD categorized drive (e.g. Vf, M, S, Y1, Vta, Vtb, Vw)."""
        s_path = os.path.join(data_dir, drive_name, f"S-{drive_name}.csv")
        v_path = os.path.join(data_dir, drive_name, f"V-{drive_name}.csv")

        # Fallback to local search
        if not os.path.exists(s_path):
            alt_dir = os.path.join(os.path.dirname(__file__), "../../../data/raw/categorised")
            s_path = os.path.join(alt_dir, drive_name, f"S-{drive_name}.csv")
            v_path = os.path.join(alt_dir, drive_name, f"V-{drive_name}.csv")

        if not os.path.exists(s_path):
            # Fallback to generating synthetic benchmark trajectory
            return self._generate_synthetic_drive(drive_name, reset_engine=reset_engine)

        try:
            df_s = pd.read_csv(s_path)
            self.drive_name = drive_name
            self.data_frames = []

            # Determine column names (IO-VNBD format)
            # Smartphone CSV: timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, lat, lon, etc.
            t_col = "timestamp" if "timestamp" in df_s.columns else df_s.columns[0]
            ax_col = [c for c in df_s.columns if "acc" in c.lower() and "x" in c.lower()]
            ay_col = [c for c in df_s.columns if "acc" in c.lower() and "y" in c.lower()]
            az_col = [c for c in df_s.columns if "acc" in c.lower() and "z" in c.lower()]

            gx_col = [c for c in df_s.columns if "gyr" in c.lower() and "x" in c.lower()]
            gy_col = [c for c in df_s.columns if "gyr" in c.lower() and "y" in c.lower()]
            gz_col = [c for c in df_s.columns if "gyr" in c.lower() and "z" in c.lower()]

            lat_col = [c for c in df_s.columns if "lat" in c.lower()]
            lon_col = [c for c in df_s.columns if "lon" in c.lower()]

            has_gps = len(lat_col) > 0 and len(lon_col) > 0

            # Reference position
            first_lat = float(df_s[lat_col[0]].dropna().iloc[0]) if has_gps and len(df_s[lat_col[0]].dropna()) > 0 else 28.6139
            first_lon = float(df_s[lon_col[0]].dropna().iloc[0]) if has_gps and len(df_s[lon_col[0]].dropna()) > 0 else 77.2090

            if reset_engine:
                self.engine.reset(ref_lat=first_lat, ref_lon=first_lon)

            # Downsample / stride to ~10 Hz if needed
            step_size = 1
            for i in range(0, len(df_s), step_size):
                row = df_s.iloc[i]
                t = float(row[t_col]) if t_col in row else float(i * 0.1)

                ax = float(row[ax_col[0]]) if ax_col else 0.0
                ay = float(row[ay_col[0]]) if ay_col else 0.0
                az = float(row[az_col[0]]) if az_col else 9.81

                gx = float(row[gx_col[0]]) if gx_col else 0.0
                gy = float(row[gy_col[0]]) if gy_col else 0.0
                gz = float(row[gz_col[0]]) if gz_col else 0.0

                imu_frame = SensorInputFrame(
                    timestamp=t,
                    acc_x=ax,
                    acc_y=ay,
                    acc_z=az,
                    gyro_x=gx,
                    gyro_y=gy,
                    gyro_z=gz,
                )

                gnss_fix = None
                if has_gps and pd.notna(row[lat_col[0]]) and pd.notna(row[lon_col[0]]):
                    gnss_fix = GNSSInputFix(
                        timestamp=t,
                        latitude=float(row[lat_col[0]]),
                        longitude=float(row[lon_col[0]]),
                        accuracy_m=3.0,
                    )

                self.data_frames.append((imu_frame, gnss_fix))

            self.current_index = 0
            return True
        except Exception as e:
            return self._generate_synthetic_drive(drive_name, reset_engine=reset_engine)

    def load_recorded_session(self, session_path_or_id: str, reset_engine: bool = True) -> bool:
        """Load a recorded field experiment session (telemetry.csv)."""
        from pathlib import Path
        p = Path(session_path_or_id)
        if not p.exists():
            p = Path("data/sessions") / session_path_or_id
            if p.is_dir():
                p = p / "telemetry.csv"
        elif p.is_dir():
            p = p / "telemetry.csv"

        if not p.exists():
            return False

        try:
            df = pd.read_csv(p)
            self.drive_name = p.parent.name if p.name == "telemetry.csv" else p.stem
            self.data_frames = []

            first_lat = 28.6139
            first_lon = 77.2090
            if "gnss_lat" in df.columns and len(df["gnss_lat"].dropna()) > 0:
                first_lat = float(df["gnss_lat"].dropna().iloc[0])
                first_lon = float(df["gnss_lon"].dropna().iloc[0])

            if reset_engine:
                self.engine.reset(ref_lat=first_lat, ref_lon=first_lon)

            for _, row in df.iterrows():
                t = float(row["timestamp"])
                ax = float(row["acc_phone_x"]) if "acc_phone_x" in row else 0.0
                ay = float(row["acc_phone_y"]) if "acc_phone_y" in row else 0.0
                az = float(row["acc_phone_z"]) if "acc_phone_z" in row else 9.81
                gx = float(row["gyro_phone_x"]) if "gyro_phone_x" in row else 0.0
                gy = float(row["gyro_phone_y"]) if "gyro_phone_y" in row else 0.0
                gz = float(row["gyro_phone_z"]) if "gyro_phone_z" in row else 0.0

                imu_frame = SensorInputFrame(
                    timestamp=t,
                    acc_x=ax,
                    acc_y=ay,
                    acc_z=az,
                    gyro_x=gx,
                    gyro_y=gy,
                    gyro_z=gz,
                    mag_x=float(row["mag_x"]) if ("mag_x" in row and pd.notna(row["mag_x"])) else None,
                    mag_y=float(row["mag_y"]) if ("mag_y" in row and pd.notna(row["mag_y"])) else None,
                    mag_z=float(row["mag_z"]) if ("mag_z" in row and pd.notna(row["mag_z"])) else None,
                    orientation_yaw=float(row["orientation_yaw"]) if ("orientation_yaw" in row and pd.notna(row["orientation_yaw"])) else None,
                    orientation_pitch=float(row["orientation_pitch"]) if ("orientation_pitch" in row and pd.notna(row["orientation_pitch"])) else None,
                    orientation_roll=float(row["orientation_roll"]) if ("orientation_roll" in row and pd.notna(row["orientation_roll"])) else None,
                )

                gnss_fix = None
                if "has_new_gnss" in row and int(row["has_new_gnss"]) == 1 and pd.notna(row.get("gnss_lat")):
                    gnss_fix = GNSSInputFix(
                        timestamp=t,
                        latitude=float(row["gnss_lat"]),
                        longitude=float(row["gnss_lon"]),
                        altitude=float(row["gnss_alt"]) if pd.notna(row.get("gnss_alt")) else 0.0,
                        accuracy_m=float(row["gnss_accuracy_m"]) if pd.notna(row.get("gnss_accuracy_m")) else 3.0,
                        speed_mps=float(row["gnss_speed_mps"]) if pd.notna(row.get("gnss_speed_mps")) else None,
                        heading_deg=float(row["gnss_heading_deg"]) if pd.notna(row.get("gnss_heading_deg")) else None,
                    )

                self.data_frames.append((imu_frame, gnss_fix))

            self.current_index = 0
            return True
        except Exception:
            return False

    def _generate_synthetic_drive(self, drive_name: str, reset_engine: bool = True) -> bool:
        """Generate physics-consistent 10-minute motorcycle test route with turns and tunnels."""
        self.drive_name = f"{drive_name} (Benchmark Synthetic)"
        self.data_frames = []
        ref_lat, ref_lon = 28.6139, 77.2090
        if reset_engine:
            self.engine.reset(ref_lat=ref_lat, ref_lon=ref_lon)

        dt = 0.1
        total_steps = 1500  # 150 seconds of 10 Hz navigation
        cur_east, cur_north = 0.0, 0.0
        cur_yaw = 0.0  # radians (East = 0, North = pi/2)
        cur_speed = 0.0

        for k in range(total_steps):
            t = k * dt

            # Drive profile: accelerate -> cruise -> 90 deg turn -> tunnel -> reacquisition
            if k < 50:
                acc = 1.2
                yaw_rate = 0.0
            elif k < 300:
                acc = 0.0
                yaw_rate = 0.0
            elif k < 400:
                acc = 0.0
                yaw_rate = float(np.deg2rad(15.0))  # Smooth turn
            elif k < 800:
                acc = 0.0
                yaw_rate = 0.0
            elif k < 900:
                acc = 0.0
                yaw_rate = -float(np.deg2rad(15.0))  # Reverse turn
            elif k < 1300:
                acc = 0.0
                yaw_rate = 0.0
            else:
                acc = -1.0  # Deceleration to stop
                yaw_rate = 0.0

            cur_speed = max(0.0, cur_speed + acc * dt)
            cur_yaw = (cur_yaw + yaw_rate * dt + np.pi) % (2 * np.pi) - np.pi

            cur_east += cur_speed * np.cos(cur_yaw) * dt
            cur_north += cur_speed * np.sin(cur_yaw) * dt

            # Compute lat/lon
            lat, lon = self.engine.fusion.enu_to_latlon(cur_east, cur_north)

            # IMU frame
            imu = SensorInputFrame(
                timestamp=t,
                acc_x=float(acc + np.random.normal(0, 0.05)),
                acc_y=float(cur_speed * yaw_rate + np.random.normal(0, 0.05)),
                acc_z=9.81 + float(np.random.normal(0, 0.05)),
                gyro_x=float(np.random.normal(0, 0.005)),
                gyro_y=float(np.random.normal(0, 0.005)),
                gyro_z=float(yaw_rate + np.random.normal(0, 0.005)),
            )

            # Simulated tunnel / blackout between step 450 and 750 (30 seconds)
            is_in_tunnel = 450 <= k <= 750
            gnss = None
            if not is_in_tunnel:
                gnss = GNSSInputFix(
                    timestamp=t,
                    latitude=lat + float(np.random.normal(0, 1e-5)),
                    longitude=lon + float(np.random.normal(0, 1e-5)),
                    accuracy_m=3.5,
                    speed_mps=cur_speed,
                    heading_deg=float((90.0 - np.rad2deg(cur_yaw)) % 360.0),
                )

            self.data_frames.append((imu, gnss))

        self.current_index = 0
        return True

    def step(self) -> Optional[NavigationOutputState]:
        """Step one frame forward in the replay."""
        if self.current_index >= len(self.data_frames):
            self.is_playing = False
            return None

        imu, gnss = self.data_frames[self.current_index]
        self.current_index += 1
        return self.engine.process_frame(imu, gnss)

    def seek(self, progress_fraction: float):
        """Seek to a relative position [0.0, 1.0]."""
        idx = int(np.clip(progress_fraction * len(self.data_frames), 0, len(self.data_frames) - 1))
        self.current_index = idx

    def get_status(self) -> Dict[str, Any]:
        total = len(self.data_frames)
        return {
            "drive_name": self.drive_name,
            "current_index": self.current_index,
            "total_frames": total,
            "progress_percent": round((self.current_index / total * 100.0) if total > 0 else 0.0, 1),
            "is_playing": self.is_playing,
            "playback_speed": self.playback_speed,
        }
