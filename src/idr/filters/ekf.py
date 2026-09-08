"""Extended Kalman Filter (EKF) for GNSS/INS Integration with Exact Analytical Jacobian.

State vector x (dim_x = 9):
  x[0:3]: Position [East, North, Up] (meters) in local ENU
  x[3:6]: Velocity [v_East, v_North, v_Up] (m/s) in local ENU
  x[6]:   Yaw / Heading angle psi (radians, counter-clockwise from East)
  x[7]:   Accelerometer bias b_a (m/s^2)
  x[8]:   Gyroscope bias b_w (rad/s)
"""

from typing import Optional, Tuple
import numpy as np


class ExtendedKalmanFilter:
    """9-State Extended Kalman Filter with full kinematic cross-derivatives."""

    def __init__(self, dt: float = 0.1):
        self.dt = float(dt)
        self.dim_x = 9
        self.x = np.zeros(self.dim_x, dtype=np.float64)

        # State covariance matrix P
        self.P = np.eye(self.dim_x, dtype=np.float64)
        self.P[0:3, 0:3] *= 10.0                 # Initial position uncertainty (m^2)
        self.P[3:6, 3:6] *= 1.0                  # Initial velocity uncertainty ((m/s)^2)
        self.P[6, 6] = float(np.deg2rad(5.0)**2) # Initial heading uncertainty (rad^2)
        self.P[7, 7] = 0.2**2                    # Accel bias uncertainty (m/s^2)^2
        self.P[8, 8] = 0.02**2                   # Gyro bias uncertainty (rad/s)^2

        # Process noise covariance Q (tuned for smartphone MEMS Allan variance)
        self.Q = np.eye(self.dim_x, dtype=np.float64)
        self.Q[0:3, 0:3] *= (0.05)**2
        self.Q[3:6, 3:6] *= (0.10)**2
        self.Q[6, 6] = (0.01)**2
        self.Q[7, 7] = (5e-3)**2                 # Accel bias random walk
        self.Q[8, 8] = (1e-4)**2                 # Gyro bias random walk

    def state_transition_function(
        self,
        x_state: np.ndarray,
        fwd_accel: float,
        yaw_rate: float,
        pitch_rad: float = 0.0,
    ) -> np.ndarray:
        """Pure nonlinear state transition function f(x, u)."""
        dt = self.dt
        x_next = x_state.copy()

        psi = x_state[6]
        b_a = x_state[7]
        b_w = x_state[8]

        # Sensor bias and slope compensation
        g = 9.80665
        a_fwd_net = (fwd_accel - b_a) - g * np.sin(pitch_rad)
        omega_corr = yaw_rate - b_w

        # Current forward velocity along vehicle heading
        v_fwd = x_state[3] * np.cos(psi) + x_state[4] * np.sin(psi)

        # World accelerations including centripetal turning acceleration
        a_e = a_fwd_net * np.cos(psi) - v_fwd * omega_corr * np.sin(psi)
        a_n = a_fwd_net * np.sin(psi) + v_fwd * omega_corr * np.cos(psi)
        a_u = a_fwd_net * np.sin(pitch_rad) - 0.5 * x_state[5]  # Soft vertical damping

        # Position integration
        x_next[0] += x_state[3] * dt + 0.5 * a_e * dt**2
        x_next[1] += x_state[4] * dt + 0.5 * a_n * dt**2
        x_next[2] += x_state[5] * dt + 0.5 * a_u * dt**2

        # Velocity integration
        x_next[3] += a_e * dt
        x_next[4] += a_n * dt
        x_next[5] += a_u * dt

        # Heading integration
        x_next[6] = (x_state[6] + omega_corr * dt + np.pi) % (2 * np.pi) - np.pi

        # Biases modeled as random walk (identity transition)
        x_next[7] = b_a
        x_next[8] = b_w

        return x_next

    def compute_analytical_jacobian(
        self,
        x_state: np.ndarray,
        fwd_accel: float,
        yaw_rate: float,
        pitch_rad: float = 0.0,
    ) -> np.ndarray:
        """Derives exact analytical Jacobian matrix F = df/dx."""
        dt = self.dt
        psi = x_state[6]
        b_a = x_state[7]
        b_w = x_state[8]

        g = 9.80665
        a_fwd_net = (fwd_accel - b_a) - g * np.sin(pitch_rad)
        omega_corr = yaw_rate - b_w
        v_fwd = x_state[3] * np.cos(psi) + x_state[4] * np.sin(psi)

        # 1. Partial derivatives of acceleration wrt velocity states (x[3], x[4])
        da_e_dv_e = -omega_corr * np.sin(psi) * np.cos(psi)
        da_e_dv_n = -omega_corr * (np.sin(psi)**2)
        da_n_dv_e = omega_corr * (np.cos(psi)**2)
        da_n_dv_n = omega_corr * np.sin(psi) * np.cos(psi)

        # 2. Partial derivatives of acceleration wrt heading (x[6])
        # d/dpsi [v_fwd * sin(psi)] = x_3 * cos(2*psi) + x_4 * sin(2*psi)
        # d/dpsi [v_fwd * cos(psi)] = -x_3 * sin(2*psi) + x_4 * cos(2*psi)
        da_e_d_psi = -a_fwd_net * np.sin(psi) - omega_corr * (x_state[3] * np.cos(2 * psi) + x_state[4] * np.sin(2 * psi))
        da_n_d_psi = a_fwd_net * np.cos(psi) + omega_corr * (-x_state[3] * np.sin(2 * psi) + x_state[4] * np.cos(2 * psi))

        # 3. Partial derivatives wrt accelerometer bias (x[7])
        da_e_d_ba = -np.cos(psi)
        da_n_d_ba = -np.sin(psi)
        da_u_d_ba = -np.sin(pitch_rad)

        # 4. Partial derivatives wrt gyro bias (x[8])
        da_e_d_bw = v_fwd * np.sin(psi)
        da_n_d_bw = -v_fwd * np.cos(psi)

        F = np.eye(self.dim_x, dtype=np.float64)

        # Position row 0 (East)
        F[0, 3] = dt + 0.5 * dt**2 * da_e_dv_e
        F[0, 4] = 0.5 * dt**2 * da_e_dv_n
        F[0, 6] = 0.5 * dt**2 * da_e_d_psi
        F[0, 7] = 0.5 * dt**2 * da_e_d_ba
        F[0, 8] = 0.5 * dt**2 * da_e_d_bw

        # Position row 1 (North)
        F[1, 3] = 0.5 * dt**2 * da_n_dv_e
        F[1, 4] = dt + 0.5 * dt**2 * da_n_dv_n
        F[1, 6] = 0.5 * dt**2 * da_n_d_psi
        F[1, 7] = 0.5 * dt**2 * da_n_d_ba
        F[1, 8] = 0.5 * dt**2 * da_n_d_bw

        # Position row 2 (Up)
        F[2, 5] = dt - 0.25 * dt**2
        F[2, 7] = 0.5 * dt**2 * da_u_d_ba

        # Velocity row 3 (v_East)
        F[3, 3] = 1.0 + dt * da_e_dv_e
        F[3, 4] = dt * da_e_dv_n
        F[3, 6] = dt * da_e_d_psi
        F[3, 7] = dt * da_e_d_ba
        F[3, 8] = dt * da_e_d_bw

        # Velocity row 4 (v_North)
        F[4, 3] = dt * da_n_dv_e
        F[4, 4] = 1.0 + dt * da_n_dv_n
        F[4, 6] = dt * da_n_d_psi
        F[4, 7] = dt * da_n_d_ba
        F[4, 8] = dt * da_n_d_bw

        # Velocity row 5 (v_Up)
        F[5, 5] = 1.0 - 0.5 * dt
        F[5, 7] = dt * da_u_d_ba

        # Heading row 6
        F[6, 6] = 1.0
        F[6, 8] = -dt

        return F

    def compute_numerical_jacobian(
        self,
        x_state: np.ndarray,
        fwd_accel: float,
        yaw_rate: float,
        pitch_rad: float = 0.0,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """Computes finite-difference numerical Jacobian for mathematical validation."""
        F_num = np.zeros((self.dim_x, self.dim_x), dtype=np.float64)
        for i in range(self.dim_x):
            dx_plus = x_state.copy()
            dx_minus = x_state.copy()
            dx_plus[i] += eps
            dx_minus[i] -= eps

            f_plus = self.state_transition_function(dx_plus, fwd_accel, yaw_rate, pitch_rad)
            f_minus = self.state_transition_function(dx_minus, fwd_accel, yaw_rate, pitch_rad)

            # Handle angular wrap-around in heading difference
            d_heading = f_plus[6] - f_minus[6]
            d_heading = (d_heading + np.pi) % (2 * np.pi) - np.pi
            f_diff = f_plus - f_minus
            f_diff[6] = d_heading

            F_num[:, i] = f_diff / (2.0 * eps)
        return F_num

    def predict(self, fwd_accel: float, yaw_rate: float, pitch_rad: float = 0.0):
        """Mechanization step: integrate forward acceleration, yaw rate, and pitch slope."""
        # 1. State prediction
        self.x = self.state_transition_function(self.x, fwd_accel, yaw_rate, pitch_rad)

        # 2. Analytical Jacobian computation
        F = self.compute_analytical_jacobian(self.x, fwd_accel, yaw_rate, pitch_rad)

        # 3. Covariance propagation
        self.P = F @ self.P @ F.T + self.Q

    def update_gnss_pos(self, pos_enu: np.ndarray, R_cov: Optional[np.ndarray] = None):
        """Update with GNSS position [East, North, Up]."""
        if R_cov is None:
            R_cov = np.eye(3) * (3.0**2)

        H = np.zeros((3, self.dim_x), dtype=np.float64)
        H[0:3, 0:3] = np.eye(3)

        z = pos_enu
        z_pred = self.x[0:3]
        y = z - z_pred

        S = H @ self.P @ H.T + R_cov
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(self.dim_x)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R_cov @ K.T

    def update_heading(self, yaw_meas: float, R_yaw: float = 0.05):
        """Update with heading measurement."""
        H = np.zeros((1, self.dim_x), dtype=np.float64)
        H[0, 6] = 1.0

        y = yaw_meas - self.x[6]
        y = (y + np.pi) % (2 * np.pi) - np.pi
        y_arr = np.array([y])

        R = np.array([[R_yaw**2]])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ y_arr).flatten()
        self.x[6] = (self.x[6] + np.pi) % (2 * np.pi) - np.pi
        I = np.eye(self.dim_x)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T

    def update_ai_velocity(
        self,
        fwd_speed: float,
        sigma_v: float = 3.0,
        max_innovation_sigma: float = 3.0,
        min_sigma_v: float = 1.0,
        max_sigma_v: float = 10.0,
    ) -> Tuple[bool, dict]:
        """Update with AI-estimated forward vehicle speed as a controlled pseudo-measurement.

        Measurement model:
            z = v_fwd_est
            h(x) = cos(psi) * v_E + sin(psi) * v_N

        Jacobian H (1 x 9):
            dh/dv_E = cos(psi)
            dh/dv_N = sin(psi)
            dh/dpsi = -sin(psi) * v_E + cos(psi) * v_N = v_lat

        Innovation & Gating:
            nu = z - h(x)
            S = H P H^T + sigma_v^2
            NIS = nu^2 / S
            Gate check: If NIS > max_innovation_sigma^2, reject update.

        Returns:
            (accepted, metrics_dict)
        """
        if not np.isfinite(fwd_speed) or fwd_speed < 0.0 or fwd_speed > 60.0:
            return False, {
                "accepted": False,
                "rejection_reason": "OUT_OF_RANGE_OR_NON_FINITE",
                "measurement": float(fwd_speed) if np.isfinite(fwd_speed) else 0.0,
                "predicted": 0.0,
                "innovation": 0.0,
                "innovation_cov": 0.0,
                "nis": 0.0,
                "applied_sigma": float(sigma_v),
            }

        psi = float(self.x[6])
        ve = float(self.x[3])
        vn = float(self.x[4])

        cos_psi = float(np.cos(psi))
        sin_psi = float(np.sin(psi))

        # Predicted forward velocity in vehicle body frame
        v_fwd_pred = cos_psi * ve + sin_psi * vn
        v_lat_pred = -sin_psi * ve + cos_psi * vn

        # Innovation
        nu = float(fwd_speed - v_fwd_pred)

        # Measurement Jacobian H (1 x dim_x)
        H = np.zeros((1, self.dim_x), dtype=np.float64)
        H[0, 3] = cos_psi
        H[0, 4] = sin_psi
        H[0, 6] = v_lat_pred

        # Covariance bounds
        clamped_sigma = float(np.clip(sigma_v, min_sigma_v, max_sigma_v))
        R = np.array([[clamped_sigma**2]], dtype=np.float64)

        # Innovation covariance S
        S_mat = H @ self.P @ H.T + R
        S = float(S_mat[0, 0])

        if S <= 1e-9 or not np.isfinite(S):
            return False, {
                "accepted": False,
                "rejection_reason": "SINGULAR_INNOVATION_COVARIANCE",
                "measurement": float(fwd_speed),
                "predicted": v_fwd_pred,
                "innovation": nu,
                "innovation_cov": S,
                "nis": 0.0,
                "applied_sigma": clamped_sigma,
            }

        nis = (nu**2) / S
        gate_threshold = float(max_innovation_sigma**2)

        metrics = {
            "measurement": float(fwd_speed),
            "predicted": float(v_fwd_pred),
            "innovation": float(nu),
            "innovation_cov": float(S),
            "nis": float(nis),
            "applied_sigma": float(clamped_sigma),
            "gate_threshold": float(gate_threshold),
        }

        if nis > gate_threshold:
            metrics["accepted"] = False
            metrics["rejection_reason"] = f"INNOVATION_GATE_EXCEEDED (NIS {nis:.2f} > {gate_threshold:.2f})"
            return False, metrics

        # Kalman Gain K = P H^T / S
        K = (self.P @ H.T) / S  # shape (dim_x, 1)

        # State Update
        self.x = self.x + (K * nu).flatten()
        self.x[6] = (self.x[6] + np.pi) % (2 * np.pi) - np.pi

        # Joseph form Covariance Update
        I = np.eye(self.dim_x, dtype=np.float64)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + (K @ R @ K.T)
        self.P = 0.5 * (self.P + self.P.T)

        metrics["accepted"] = True
        metrics["rejection_reason"] = "NONE"
        return True, metrics

    def update_velocity(self, fwd_speed: float, R_speed: float = 3.0):
        """Update with AI or wheel speed using exact body-frame projection."""
        self.update_ai_velocity(fwd_speed=fwd_speed, sigma_v=R_speed)

    def update_gnss_vel(self, vel_enu: np.ndarray, R_cov: Optional[np.ndarray] = None):
        """Update with GNSS velocity vector [v_East, v_North, v_Up]."""
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

    def update_velocity_2d(self, v_fwd: float, v_lat: float = 0.0, R_cov: Optional[np.ndarray] = None):
        """Update with 2D vehicle body-frame velocities [v_fwd, v_lat]."""
        psi = self.x[6]
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)

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
