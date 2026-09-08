"""Zero-Velocity Update (ZUPT) and Zero Angular Rate Update (ZARU) with Motorcycle Vibration Robustness.

Detects stationary periods even in the presence of high-frequency single-cylinder
motorcycle engine idle vibration using dual-threshold statistical filtering and
temporal persistence hysteresis.
"""

from typing import List, Optional
import numpy as np
from .ekf import ExtendedKalmanFilter

GRAVITY = 9.80665


class StationaryDetector:
    """Statistical stationary detector robust to vehicle cruising and engine idle vibrations.

    Physical Principle:
    By the Equivalence Principle, an accelerometer in a smoothly cruising vehicle
    at constant velocity measures approximately the same specific force (gravity ~9.81 m/s^2)
    and low variance as a stationary vehicle.
    Therefore, a low-variance acceleration signal ALONE is insufficient evidence of rest.

    Multi-Modal Stationary Gate requires:
    1. Mean acceleration magnitude matches Earth gravity (| ||a|| - g | <= gravity_tolerance)
    2. Low acceleration magnitude variance (acc_var < acc_var_threshold)
    3. Low angular rate (gyro_mean_norm < gyro_norm_threshold)
    4. Kinematic / external velocity consistency (speed <= max_stationary_speed_mps if speed available)
    5. Multi-frame persistence hysteresis (persistence_frames consecutive quiet frames to latch)
    """

    def __init__(
        self,
        window_size: int = 10,                 # 1.0 s sliding window at 10 Hz
        acc_var_threshold: float = 0.25,       # Accommodates engine idle vibration
        gyro_norm_threshold: float = 0.05,     # rad/s max mean angular rate when stationary (~2.8 deg/s)
        gravity_tolerance_mps2: float = 1.2,   # | ||a|| - g | tolerance
        max_stationary_speed_mps: float = 0.8, # Max velocity considered stationary (0.8 m/s = ~2.9 km/h)
        persistence_frames: int = 2,           # 2 consecutive quiet frames required to confirm stop
    ):
        self.window_size = int(window_size)
        self.acc_var_threshold = float(acc_var_threshold)
        self.gyro_norm_threshold = float(gyro_norm_threshold)
        self.gravity_tolerance_mps2 = float(gravity_tolerance_mps2)
        self.max_stationary_speed_mps = float(max_stationary_speed_mps)
        self.persistence_frames = int(persistence_frames)

        self.acc_buffer: List[np.ndarray] = []
        self.gyro_buffer: List[np.ndarray] = []
        self._stationary_counter = 0
        self._is_stationary_latched = False

    def reset(self):
        self.acc_buffer.clear()
        self.gyro_buffer.clear()
        self._stationary_counter = 0
        self._is_stationary_latched = False

    def update(
        self,
        acc_3d: np.ndarray,
        gyro_3d: np.ndarray,
        speed_mps: Optional[float] = None,
    ) -> bool:
        """Push one IMU sample and evaluate stationary status.

        Args:
            acc_3d: 3-axis accelerometer specific force in m/s^2
            gyro_3d: 3-axis gyroscope angular rate in rad/s
            speed_mps: Optional current speed estimate (m/s) from EKF, GNSS, or AI
        """
        acc = np.asarray(acc_3d, dtype=np.float64).flatten()
        gyro = np.asarray(gyro_3d, dtype=np.float64).flatten()

        if len(acc) != 3 or len(gyro) != 3 or not np.all(np.isfinite(acc)) or not np.all(np.isfinite(gyro)):
            return self._is_stationary_latched

        # Immediate motion check: if speed is known and exceeds stationary threshold, unlatch instantly
        if speed_mps is not None and np.isfinite(speed_mps) and speed_mps > self.max_stationary_speed_mps:
            self._stationary_counter = 0
            self._is_stationary_latched = False
            return False

        self.acc_buffer.append(acc)
        self.gyro_buffer.append(gyro)

        if len(self.acc_buffer) > self.window_size:
            self.acc_buffer.pop(0)
            self.gyro_buffer.pop(0)

        if len(self.acc_buffer) < self.window_size:
            return False

        acc_arr = np.array(self.acc_buffer)
        gyro_arr = np.array(self.gyro_buffer)

        # 1. Check mean gravity norm consistency
        mean_acc = np.mean(acc_arr, axis=0)
        norm_mean_acc = float(np.linalg.norm(mean_acc))
        is_gravity_consistent = abs(norm_mean_acc - GRAVITY) <= self.gravity_tolerance_mps2

        # 2. Check low-pass acceleration variance
        acc_mags = np.linalg.norm(acc_arr, axis=1)
        acc_var = float(np.var(acc_mags))
        is_acc_quiet = acc_var < self.acc_var_threshold

        # 3. Check mean and max angular velocity
        gyro_mean_norm = float(np.linalg.norm(np.mean(gyro_arr, axis=0)))
        gyro_max_norm = float(np.max(np.linalg.norm(gyro_arr, axis=1)))
        is_gyro_quiet = (gyro_mean_norm < self.gyro_norm_threshold) and (gyro_max_norm < (self.gyro_norm_threshold * 2.5))

        # 4. Check speed consistency
        is_speed_quiet = True
        if speed_mps is not None and np.isfinite(speed_mps):
            is_speed_quiet = speed_mps <= self.max_stationary_speed_mps

        instantaneous_stationary = is_gravity_consistent and is_acc_quiet and is_gyro_quiet and is_speed_quiet

        # 5. Hysteresis logic
        if instantaneous_stationary:
            self._stationary_counter = min(self.persistence_frames, self._stationary_counter + 1)
            if self._stationary_counter >= self.persistence_frames:
                self._is_stationary_latched = True
        else:
            # Immediate unlatch on significant motion or consecutive motion frames
            self._stationary_counter = 0
            self._is_stationary_latched = False

        return self._is_stationary_latched


def apply_zupt(
    ekf: ExtendedKalmanFilter,
    sigma_v: float = 0.01,
):
    """Apply Zero-Velocity Update: velocity = [0, 0, 0]."""
    H = np.zeros((3, ekf.dim_x), dtype=np.float64)
    H[0, 3] = 1.0  # vE
    H[1, 4] = 1.0  # vN
    H[2, 5] = 1.0  # vU

    y = -ekf.x[3:6].copy()
    R = np.eye(3) * (sigma_v**2)

    S = H @ ekf.P @ H.T + R
    K = ekf.P @ H.T @ np.linalg.inv(S)

    ekf.x = ekf.x + K @ y
    I = np.eye(ekf.dim_x)
    ekf.P = (I - K @ H) @ ekf.P @ (I - K @ H).T + K @ R @ K.T


def apply_zaru(
    ekf: ExtendedKalmanFilter,
    gyro_z_raw: float,
    sigma_bias: float = 0.005,
):
    """Apply Zero Angular Rate Update: observed gyro_z = gyro bias."""
    H = np.zeros((1, ekf.dim_x), dtype=np.float64)
    H[0, 8] = 1.0

    y = np.array([float(gyro_z_raw) - ekf.x[8]])
    R = np.array([[sigma_bias**2]])

    S = H @ ekf.P @ H.T + R
    K = ekf.P @ H.T @ np.linalg.inv(S)

    self_x_update = (K @ y).flatten()
    ekf.x = ekf.x + self_x_update
    I = np.eye(ekf.dim_x)
    ekf.P = (I - K @ H) @ ekf.P @ (I - K @ H).T + K @ R @ K.T
