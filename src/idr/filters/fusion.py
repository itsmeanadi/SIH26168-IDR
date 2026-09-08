"""High-level GNSS+INS Fusion Engine orchestrating EKF, NHC, and AI-velocity updates."""

from typing import List, Optional, Tuple
import numpy as np

try:
    import pyproj
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

from .ekf import ExtendedKalmanFilter
from .es_ekf import ErrorStateKalmanFilter, ESEKFConfig
from .nhc import apply_nhc_update
from .zupt import StationaryDetector, apply_zupt, apply_zaru

class GNSSINSFusion:
    """Manages coordinate transformations, GNSS blackout transitions, and Dead Reckoning."""

    def __init__(
        self,
        ref_lat: float,
        ref_lon: float,
        dt: float = 0.1,
        filter_type: str = "ekf",
        min_cog_speed_mps: float = 1.5,
    ):
        self.dt = dt
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        self.filter_type = filter_type
        self.min_cog_speed_mps = min_cog_speed_mps
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

        if filter_type in ("es_ekf", "15state"):
            self.filter_type = "es_ekf"
            self.es_ekf = ErrorStateKalmanFilter()
            self.ekf = self.es_ekf  # Pointer for unified references
        else:
            self.filter_type = "ekf"
            self.ekf = ExtendedKalmanFilter(dt=dt)
            self.es_ekf = None
        self.in_blackout = False
        self.trajectory_history: List[np.ndarray] = []
        self.prev_gnss_enu: Optional[Tuple[float, float]] = None

    @property
    def position_enu(self) -> np.ndarray:
        """Get 3D position [East, North, Up] in meters."""
        if self.es_ekf is not None:
            return self.es_ekf.p.copy()
        return self.ekf.x[0:3].copy()

    @property
    def velocity_enu(self) -> np.ndarray:
        """Get 3D velocity [vE, vN, vU] in m/s."""
        if self.es_ekf is not None:
            return self.es_ekf.v.copy()
        return self.ekf.x[3:6].copy()

    @property
    def yaw_rad(self) -> float:
        """Get mathematical ENU yaw angle in radians."""
        if self.es_ekf is not None:
            return float(self.es_ekf.euler_angles[2])
        return float(self.ekf.x[6])

    @property
    def pos_covariance_2d(self) -> np.ndarray:
        """Get 2x2 East-North position covariance matrix."""
        if self.es_ekf is not None:
            return self.es_ekf.P[0:2, 0:2].copy()
        return self.ekf.P[0:2, 0:2].copy()

    @property
    def pos_uncertainty_m(self) -> float:
        """Get 1-sigma horizontal position uncertainty in meters."""
        P2d = self.pos_covariance_2d
        return float(np.sqrt(P2d[0, 0] + P2d[1, 1]))

    def latlon_to_enu(self, lat: float, lon: float) -> Tuple[float, float, float]:
        """Convert WGS84 lat/lon to local East-North-Up (m)."""
        if self.proj_enu is not None:
            x, y = self.proj_enu(lon, lat)
            return float(x), float(y), 0.0
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
        pitch_rad: float = 0.0,
    ) -> np.ndarray:
        """Single filter integration step.
        
        Args:
            fwd_accel: Forward acceleration in vehicle body frame (m/s^2)
            yaw_rate: Yaw rate in vehicle body frame (rad/s)
            gnss_pos: (lat, lon) or None
            ai_velocity: Forward velocity (float) or 2D velocity [v_fwd, v_lat] from AI model
            is_gnss_denied: True if currently inside simulated blackout
            use_nhc: Whether to apply non-holonomic constraints
            acc_3d: Optional 3-axis accel for stationary ZUPT detection and 3D ES-EKF
            gyro_3d: Optional 3-axis gyro for stationary ZARU detection and 3D ES-EKF
            pitch_rad: Pitch angle in radians (for legacy 2D pitch slope compensation)
        """
        # Prepare 3D IMU measurements
        if acc_3d is not None:
            a_3d = np.array(acc_3d, dtype=np.float64).reshape(3)
        else:
            a_3d = np.array([fwd_accel, 0.0, 9.80665], dtype=np.float64)

        if gyro_3d is not None:
            g_3d = np.array(gyro_3d, dtype=np.float64).reshape(3)
        else:
            g_3d = np.array([0.0, 0.0, yaw_rate], dtype=np.float64)

        # 1. IMU Prediction step
        if self.es_ekf is not None:
            self.es_ekf.predict(a_3d, g_3d, dt=self.dt)
        else:
            self.ekf.predict(fwd_accel, yaw_rate, pitch_rad=pitch_rad)

        # 2. Stationary detection with velocity-aware gating
        vel = self.velocity_enu
        cur_speed = float(np.hypot(vel[0], vel[1]))
        if ai_velocity is not None:
            if isinstance(ai_velocity, (tuple, list, np.ndarray)) and len(ai_velocity) >= 1:
                cur_speed = max(cur_speed, float(ai_velocity[0]))
            elif isinstance(ai_velocity, (int, float)) and np.isfinite(ai_velocity):
                cur_speed = max(cur_speed, float(ai_velocity))

        is_stat = self.stationary_detector.update(a_3d, g_3d, speed_mps=cur_speed)
        if is_stat:
            if self.es_ekf is not None:
                self.es_ekf.update_zupt(sigma_v=0.01)
                self.es_ekf.update_zaru(g_3d, sigma_bg=0.001)
            else:
                apply_zupt(self.ekf, sigma_v=0.01)
                apply_zaru(self.ekf, gyro_z_raw=float(g_3d[2]))

        # 3. Measurement updates
        if not is_gnss_denied and gnss_pos is not None:
            if self.in_blackout:
                self.in_blackout = False

            east, north, up = self.latlon_to_enu(gnss_pos[0], gnss_pos[1])
            gnss_enu = np.array([east, north, up], dtype=np.float64)

            if self.es_ekf is not None:
                self.es_ekf.update_gnss_pos(gnss_enu)
            else:
                self.ekf.update_gnss_pos(gnss_enu)

            # Course Over Ground (COG) heading and GNSS velocity updates
            if self.prev_gnss_enu is not None:
                de = east - self.prev_gnss_enu[0]
                dn = north - self.prev_gnss_enu[1]
                dist = np.hypot(de, dn)
                gnss_speed = dist / max(1e-3, self.dt)
                if gnss_speed >= self.min_cog_speed_mps:  # Standstill GNSS COG suppression
                    cog = np.arctan2(dn, de)
                    if self.es_ekf is not None:
                        self.es_ekf.update_heading(cog, sigma_yaw=0.05)
                    else:
                        self.ekf.update_heading(cog, R_yaw=0.03)
                if dist > 0.1:
                    v_gnss = np.array([de / self.dt, dn / self.dt, 0.0], dtype=np.float64)
                    if self.es_ekf is not None:
                        self.es_ekf.update_gnss_vel(v_gnss)
                    else:
                        self.ekf.update_gnss_vel(v_gnss, R_cov=np.eye(3, dtype=np.float64) * (0.1**2))
            self.prev_gnss_enu = (east, north)
        else:
            self.in_blackout = True
            self.prev_gnss_enu = None

            # 4. Apply Non-Holonomic Constraints (NHC) during blackout
            if use_nhc:
                if self.es_ekf is not None:
                    self.es_ekf.update_nhc(sigma_lat=0.05, sigma_vert=0.05)
                else:
                    apply_nhc_update(self.ekf, sigma_lat=0.05, sigma_vert=0.05)

            # 5. Apply AI Velocity update (scalar forward speed or 2D velocity)
            if ai_velocity is not None:
                if self.es_ekf is not None:
                    ai_spd = float(ai_velocity[0]) if isinstance(ai_velocity, (tuple, list, np.ndarray)) else float(ai_velocity)
                    self.es_ekf.update_ai_velocity(ai_spd, sigma_v=3.0)
                else:
                    if isinstance(ai_velocity, (tuple, list, np.ndarray)) and len(ai_velocity) >= 2:
                        self.ekf.update_velocity_2d(float(ai_velocity[0]), float(ai_velocity[1]))
                    else:
                        self.ekf.update_ai_velocity(float(ai_velocity), sigma_v=3.0)

        pos = self.position_enu
        vel = self.velocity_enu
        yaw = self.yaw_rad
        current_state = np.array([pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], yaw], dtype=np.float64)
        self.trajectory_history.append(current_state)
        return current_state
