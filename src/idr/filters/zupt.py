"""Zero-Velocity Update (ZUPT) and Zero Angular Rate Update (ZARU).

Detects stationary periods from accelerometer variance and applies:
1. ZUPT: velocity = [0, 0, 0] pseudo-measurement
2. ZARU: gyro reading = gyro bias (direct bias observation)

These updates anchor the IMU bias estimates during stops, which is
critical for stop-and-go driving scenarios.
"""

import numpy as np
from .ekf import ExtendedKalmanFilter

# Gravity magnitude (m/s²)
GRAVITY = 9.80665


class StationaryDetector:
    """Statistical detector for vehicle stationary periods.

    Uses accelerometer magnitude variance over a sliding window.
    When the vehicle is stationary, the only acceleration is gravity,
    so |a| ≈ g with very low variance (dominated by sensor noise).
    When moving, road vibrations and dynamic accelerations increase variance.
    """

    def __init__(
        self,
        window_size: int = 10,  # 1.0 s at 10 Hz
        acc_var_threshold: float = 0.15,  # m²/s⁴ — tuned for smartphone MEMS
        gyro_norm_threshold: float = 0.05,  # rad/s — max gyro magnitude when stationary
    ):
        self.window_size = window_size
        self.acc_var_threshold = acc_var_threshold
        self.gyro_norm_threshold = gyro_norm_threshold
        self.acc_buffer = []
        self.gyro_buffer = []

    def update(self, acc_3d: np.ndarray, gyro_3d: np.ndarray) -> bool:
        """Push one IMU sample and return True if vehicle is stationary.

        Args:
            acc_3d: [ax, ay, az] in m/s² (any frame)
            gyro_3d: [gx, gy, gz] in rad/s
        """
        self.acc_buffer.append(acc_3d.copy())
        self.gyro_buffer.append(gyro_3d.copy())

        if len(self.acc_buffer) > self.window_size:
            self.acc_buffer.pop(0)
            self.gyro_buffer.pop(0)

        if len(self.acc_buffer) < self.window_size:
            return False

        acc_arr = np.array(self.acc_buffer)
        gyro_arr = np.array(self.gyro_buffer)

        # Accelerometer magnitude variance
        acc_mag = np.linalg.norm(acc_arr, axis=1)
        acc_var = np.var(acc_mag)

        # Gyro magnitude (mean over window)
        gyro_mean_norm = np.mean(np.linalg.norm(gyro_arr, axis=1))

        is_stationary = (
            acc_var < self.acc_var_threshold
            and gyro_mean_norm < self.gyro_norm_threshold
        )
        return is_stationary


def apply_zupt(
    ekf: ExtendedKalmanFilter,
    sigma_v: float = 0.01,
):
    """Apply Zero-Velocity Update: velocity = [0, 0, 0].

    When the vehicle is detected as stationary, its velocity in
    the world frame must be zero. This is an extremely tight constraint
    (σ = 0.01 m/s) that rapidly corrects velocity drift and, through
    cross-covariance, helps constrain position and bias states.
    """
    H = np.zeros((3, ekf.dim_x), dtype=np.float64)
    H[0, 3] = 1.0  # vE
    H[1, 4] = 1.0  # vN
    H[2, 5] = 1.0  # vU

    # Innovation: measurement (0) - predicted velocity
    y = -ekf.x[3:6].copy()

    R = np.eye(3) * sigma_v**2

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
    """Apply Zero Angular Rate Update: observed gyro_z = gyro bias.

    When stationary, the true angular rate is zero, so:
        gyro_z_raw = b_w + noise
    This directly observes the gyro bias state x[8].

    Args:
        gyro_z_raw: Raw z-axis gyroscope reading (rad/s)
        sigma_bias: Measurement noise for bias observation
    """
    H = np.zeros((1, ekf.dim_x), dtype=np.float64)
    H[0, 8] = 1.0  # gyro bias state

    # z = gyro_z_raw (this IS the bias when stationary)
    y = np.array([gyro_z_raw - ekf.x[8]])

    R = np.array([[sigma_bias**2]])
    S = H @ ekf.P @ H.T + R
    K = ekf.P @ H.T @ np.linalg.inv(S)

    ekf.x = ekf.x + K @ y
    I = np.eye(ekf.dim_x)
    ekf.P = (I - K @ H) @ ekf.P @ (I - K @ H).T + K @ R @ K.T
