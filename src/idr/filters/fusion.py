"""High-level GNSS+INS Fusion Engine orchestrating EKF, NHC, and AI-velocity updates."""

from typing import List, Optional, Tuple
import numpy as np

try:
    import pyproj
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

from .ekf import ExtendedKalmanFilter
from .nhc import apply_nhc_update
from .zupt import StationaryDetector, apply_zupt, apply_zaru

class GNSSINSFusion:
    """Manages coordinate transformations, GNSS blackout transitions, and Dead Reckoning."""

    def __init__(self, ref_lat: float, ref_lon: float, dt: float = 0.1):
        self.dt = dt
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        self.stationary_detector = StationaryDetector(window_size=10, acc_var_threshold=0.15)
        
        # Local ENU projection centered at initial GPS point
        if HAS_PYPROJ:
            self.proj_enu = pyproj.Proj(
                proj="aeqd",
                lat_0=ref_lat,
                lon_0=ref_lon,
                datum="WGS84",
                units="m"
            )
        else:
            self.proj_enu = None

        self.ekf = ExtendedKalmanFilter(dt=dt)
        self.in_blackout = False
        self.trajectory_history: List[np.ndarray] = []
        self.prev_gnss_enu: Optional[Tuple[float, float]] = None

    def latlon_to_enu(self, lat: float, lon: float) -> Tuple[float, float, float]:
        """Convert WGS84 lat/lon to local East-North-Up (m)."""
        if self.proj_enu is not None:
            x, y = self.proj_enu(lon, lat)
            return float(x), float(y), 0.0
        # High-precision WGS84 flat-Earth geodesic projection
        lat_rad = np.deg2rad(self.ref_lat)
        R_m = 6378137.0
        d_lat = np.deg2rad(lat - self.ref_lat)
        d_lon = np.deg2rad(lon - self.ref_lon)
        north = d_lat * R_m
        east = d_lon * R_m * np.cos(lat_rad)
        return float(east), float(north), 0.0

    def enu_to_latlon(self, east: float, north: float) -> Tuple[float, float]:
        """Convert local East-North-Up to WGS84 lat/lon."""
        if self.proj_enu is not None:
            lon, lat = self.proj_enu(east, north, inverse=True)
            return float(lat), float(lon)
        lat_rad = np.deg2rad(self.ref_lat)
        R_m = 6378137.0
        d_lat = north / R_m
        d_lon = east / (R_m * np.cos(lat_rad))
        lat = self.ref_lat + np.rad2deg(d_lat)
        lon = self.ref_lon + np.rad2deg(d_lon)
        return float(lat), float(lon)

    def step(
        self,
        fwd_accel: float,
        yaw_rate: float,
        gnss_pos: Optional[Tuple[float, float]] = None,
        ai_velocity: Optional[object] = None,
        is_gnss_denied: bool = False,
        use_nhc: bool = True,
        acc_3d: Optional[np.ndarray] = None,
        gyro_3d: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Single 10 Hz filter integration step.
        
        Args:
            fwd_accel: Forward acceleration in vehicle body frame (m/s^2)
            yaw_rate: Yaw rate in vehicle body frame (rad/s)
            gnss_pos: (lat, lon) or None
            ai_velocity: Forward velocity (float) or 2D velocity [v_fwd, v_lat] from AI model
            is_gnss_denied: True if currently inside simulated blackout
            use_nhc: Whether to apply non-holonomic constraints
            acc_3d: Optional 3-axis accel for stationary ZUPT detection
            gyro_3d: Optional 3-axis gyro for stationary ZARU detection
        """
        # 1. IMU Prediction step
        self.ekf.predict(fwd_accel, yaw_rate)

        # 2. Check stationary detector for ZUPT and ZARU
        if acc_3d is not None and gyro_3d is not None:
            if self.stationary_detector.update(acc_3d, gyro_3d):
                apply_zupt(self.ekf, sigma_v=0.01)
                apply_zaru(self.ekf, gyro_z_raw=float(gyro_3d[2]) if len(gyro_3d) > 2 else yaw_rate)

        # 3. Measurement updates
        if not is_gnss_denied and gnss_pos is not None:
            if self.in_blackout:
                self.in_blackout = False

            east, north, up = self.latlon_to_enu(gnss_pos[0], gnss_pos[1])
            self.ekf.update_gnss_pos(np.array([east, north, up]))

            # Course Over Ground (COG) heading and GNSS velocity updates
            if self.prev_gnss_enu is not None:
                de = east - self.prev_gnss_enu[0]
                dn = north - self.prev_gnss_enu[1]
                dist = np.hypot(de, dn)
                if dist > 0.1:  # Moving threshold
                    cog = np.arctan2(dn, de)
                    self.ekf.update_heading(cog, R_yaw=0.03)
                    v_gnss = np.array([de / self.dt, dn / self.dt, 0.0])
                    self.ekf.update_gnss_vel(v_gnss, R_cov=np.eye(3, dtype=np.float64) * (0.1**2))
            self.prev_gnss_enu = (east, north)
        else:
            self.in_blackout = True
            self.prev_gnss_enu = None

            # 4. Apply Non-Holonomic Constraints (NHC) during blackout
            if use_nhc:
                apply_nhc_update(self.ekf, sigma_lat=0.05, sigma_vert=0.05)

            # 5. Apply AI Velocity update (scalar or 2D displacement/velocity)
            if ai_velocity is not None:
                if isinstance(ai_velocity, (tuple, list, np.ndarray)) and len(ai_velocity) >= 2:
                    self.ekf.update_velocity_2d(float(ai_velocity[0]), float(ai_velocity[1]))
                else:
                    self.ekf.update_velocity(float(ai_velocity), R_speed=0.5)

        current_state = self.ekf.x.copy()
        self.trajectory_history.append(current_state)
        return current_state
