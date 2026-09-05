"""Extended Kalman Filter (EKF) for GNSS/INS Integration."""

import numpy as np

class ExtendedKalmanFilter:
    """9-State Extended Kalman Filter in Local ENU Frame:
    
    State vector x:
      x[0:3]: Position [East, North, Up] (meters)
      x[3:6]: Velocity [v_East, v_North, v_Up] (m/s)
      x[6]:   Yaw / Heading angle psi (radians, counter-clockwise from East)
      x[7]:   Accelerometer bias b_a (m/s^2)
      x[8]:   Gyroscope bias b_w (rad/s)
    """

    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.dim_x = 9
        self.x = np.zeros(self.dim_x, dtype=np.float64)
        
        # State covariance matrix P
        self.P = np.eye(self.dim_x, dtype=np.float64)
        self.P[0:3, 0:3] *= 10.0   # Initial position uncertainty (m^2)
        self.P[3:6, 3:6] *= 1.0    # Initial velocity uncertainty ((m/s)^2)
        self.P[6, 6] *= np.deg2rad(5.0)**2  # Initial heading uncertainty (rad^2)
        self.P[7, 7] = 0.2**2     # Accel bias (m/s^2)^2
        self.P[8, 8] = 0.02**2    # Gyro bias (rad/s)^2

        # Process noise covariance Q (tuned for smartphone MEMS Allan variance)
        self.Q = np.eye(self.dim_x, dtype=np.float64)
        self.Q[0:3, 0:3] *= (0.05)**2
        self.Q[3:6, 3:6] *= (0.1)**2
        self.Q[6, 6] *= (0.01)**2
        self.Q[7, 7] = (5e-3)**2  # Accel bias random walk
        self.Q[8, 8] = (1e-4)**2  # Gyro bias random walk (Allan variance smartphone MEMS)

    def predict(self, fwd_accel: float, yaw_rate: float):
        """Mechanization step: integrate forward acceleration and yaw rate."""
        dt = self.dt
        psi = self.x[6]
        b_a = self.x[7]
        b_w = self.x[8]

        # Corrected inputs
        acc = fwd_accel - b_a
        omega = yaw_rate - b_w

        # Current forward speed along vehicle heading
        v_fwd = self.x[3] * np.cos(psi) + self.x[4] * np.sin(psi)

        # Centripetal acceleration in world frame due to turning: a_c = omega x v
        a_e = acc * np.cos(psi) - v_fwd * omega * np.sin(psi)
        a_n = acc * np.sin(psi) + v_fwd * omega * np.cos(psi)

        # State transition: x_k = f(x_{k-1}, u_k)
        self.x[0] += self.x[3] * dt + 0.5 * a_e * dt**2  # East pos
        self.x[1] += self.x[4] * dt + 0.5 * a_n * dt**2  # North pos
        self.x[2] += self.x[5] * dt                      # Up pos

        self.x[3] += a_e * dt                            # East vel
        self.x[4] += a_n * dt                            # North vel
        self.x[5] = 0.0                                  # Planar vehicle constraint

        self.x[6] = (self.x[6] + omega * dt + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]

        # Jacobian F = df/dx
        F = np.eye(self.dim_x, dtype=np.float64)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        F[3, 6] = -(acc * np.sin(psi) + v_fwd * omega * np.cos(psi)) * dt
        F[4, 6] = (acc * np.cos(psi) - v_fwd * omega * np.sin(psi)) * dt
        F[3, 7] = -np.cos(psi) * dt
        F[4, 7] = -np.sin(psi) * dt

        F[6, 8] = -dt

        # Covariance prediction
        self.P = F @ self.P @ F.T + self.Q

    def update_gnss_pos(self, pos_enu: np.ndarray, R_cov: np.ndarray = None):
        """Update with GNSS position [East, North, Up]."""
        if R_cov is None:
            R_cov = np.eye(3) * (3.0**2)

        H = np.zeros((3, self.dim_x), dtype=np.float64)
        H[0:3, 0:3] = np.eye(3)

        z = pos_enu
        z_pred = self.x[0:3]
        y = z - z_pred  # Innovation

        S = H @ self.P @ H.T + R_cov
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(self.dim_x)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R_cov @ K.T

    def update_heading(self, yaw_meas: float, R_yaw: float = 0.05):
        """Update with heading measurement (e.g. Course Over Ground from GNSS velocity)."""
        H = np.zeros((1, self.dim_x), dtype=np.float64)
        H[0, 6] = 1.0

        # Innovation with angular wrap-around to [-pi, pi]
        y = yaw_meas - self.x[6]
        y = (y + np.pi) % (2 * np.pi) - np.pi
        y = np.array([y])

        R = np.array([[R_yaw**2]])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ y).flatten()
        self.x[6] = (self.x[6] + np.pi) % (2 * np.pi) - np.pi
        I = np.eye(self.dim_x)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T

    def update_velocity(self, fwd_speed: float, R_speed: float = 0.5):
        """Update with deep learning estimated forward speed: z = sqrt(v_e^2 + v_n^2)."""
        ve, vn = self.x[3], self.x[4]
        v_norm = np.sqrt(ve**2 + vn**2) + 1e-6

        # H = [0, 0, 0, ve/v_norm, vn/v_norm, 0, 0, 0, 0]
        H = np.zeros((1, self.dim_x), dtype=np.float64)
        H[0, 3] = ve / v_norm
        H[0, 4] = vn / v_norm

        y = np.array([fwd_speed - v_norm])
        R = np.array([[R_speed**2]])

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ y).flatten()
        I = np.eye(self.dim_x)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T

    def update_gnss_vel(self, vel_enu: np.ndarray, R_cov: np.ndarray = None):
        """Update with GNSS velocity vector [v_East, v_North, v_Up].
        
        Provides direct observation of velocity states, which accelerates
        accelerometer bias convergence via Kalman cross-terms.
        """
        if R_cov is None:
            R_cov = np.eye(3, dtype=np.float64) * (0.1**2)

        H = np.zeros((3, self.dim_x), dtype=np.float64)
        H[0:3, 3:6] = np.eye(3, dtype=np.float64)

        z = vel_enu
        y = z - self.x[3:6]

        S = H @ self.P @ H.T + R_cov
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(self.dim_x)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R_cov @ K.T

    def update_velocity_2d(self, v_fwd: float, v_lat: float = 0.0, R_cov: np.ndarray = None):
        """Update with 2D vehicle body-frame velocities [v_fwd, v_lat].
        
        Transforms body velocity into world ENU frame and updates velocity states.
        """
        psi = self.x[6]
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)

        # World velocity measurement derived from body velocity
        v_e_meas = v_fwd * cos_psi - v_lat * sin_psi
        v_n_meas = v_fwd * sin_psi + v_lat * cos_psi

        if R_cov is None:
            R_cov = np.diag([0.2**2, 0.2**2])

        H = np.zeros((2, self.dim_x), dtype=np.float64)
        H[0, 3] = 1.0
        H[1, 4] = 1.0

        y = np.array([v_e_meas - self.x[3], v_n_meas - self.x[4]])

        S = H @ self.P @ H.T + R_cov
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(self.dim_x)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R_cov @ K.T

