"""15-State 3D Error-State Extended Kalman Filter (ES-EKF) for GNSS/INS/AI Navigation.

State Representation:
    Nominal State (16):
        - p: Position in East-North-Up (ENU) frame (3,) [m]
        - v: Velocity in ENU frame (3,) [m/s]
        - q: Unit quaternion [qw, qx, qy, qz] representing Body -> ENU rotation (4,)
        - ba: Accelerometer bias in Body (FLU) frame (3,) [m/s^2]
        - bg: Gyroscope bias in Body (FLU) frame (3,) [rad/s]

    Error State (15):
        - delta_p: Position error in ENU (3,)
        - delta_v: Velocity error in ENU (3,)
        - delta_theta: Right-multiplicative (Body-frame) attitude error (3,)
        - delta_ba: Accelerometer bias error in Body frame (3,)
        - delta_bg: Gyroscope bias error in Body frame (3,)

Attitude Error Convention:
    q_true = q_nominal (x) delta_q
    delta_q ~= [1, 0.5 * delta_theta]^T
    R_true = R(q_nominal) * (I_3 + [delta_theta]_x)

Mathematical Properties:
    - Body-to-navigation IMU mechanization with ENU gravity g_n = [0, 0, -9.80665]^T.
    - Exact SO(3) Right Jacobian J_r(omega * dt) for gyro bias coupling F_theta_bg.
    - Joseph-form covariance updates for guaranteed positive semi-definiteness.
    - Multiplicative attitude error injection and post-update covariance reset G P G^T.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np


# ---------------------------------------------------------------------------
# Lie Algebra SO(3) & Quaternion Utilities
# ---------------------------------------------------------------------------

def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Compute 3x3 skew-symmetric matrix [v]_x from 3D vector v."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=np.float64)


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion [qw, qx, qy, qz] to 3x3 Direction Cosine Matrix R.
    
    R maps vectors from Body frame to Navigation (ENU) frame: v_nav = R @ v_body.
    """
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    return np.array([
        [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)],
        [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx)],
        [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)]
    ], dtype=np.float64)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamiltonian product of two quaternions q1 (x) q2 with [qw, qx, qy, qz]."""
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ], dtype=np.float64)


def exp_quaternion(v: np.ndarray) -> np.ndarray:
    """Quaternion exponential map from rotation vector v in R^3 to unit quaternion.
    
    Returns [qw, qx, qy, qz].
    """
    angle = np.linalg.norm(v)
    if angle < 1e-8:
        # Second-order Taylor expansion for numerical stability
        qw = 1.0 - 0.125 * angle * angle
        scale = 0.5 - (angle * angle) / 48.0
        q_vec = scale * v
        q = np.array([qw, q_vec[0], q_vec[1], q_vec[2]], dtype=np.float64)
        return q / np.linalg.norm(q)
    
    half_angle = 0.5 * angle
    qw = np.cos(half_angle)
    scale = np.sin(half_angle) / angle
    q_vec = scale * v
    return np.array([qw, q_vec[0], q_vec[1], q_vec[2]], dtype=np.float64)


def so3_right_jacobian(v: np.ndarray) -> np.ndarray:
    """Compute the 3x3 Right Jacobian of SO(3): J_r(v).
    
    Formula:
        J_r(v) = I_3 - (1 - cos(theta)) / theta^2 [v]_x + (theta - sin(theta)) / theta^3 [v]_x^2
    """
    theta = np.linalg.norm(v)
    v_skew = skew_symmetric(v)
    if theta < 1e-6:
        # Taylor expansion: I - 0.5 [v]_x + 1/6 [v]_x^2
        return np.eye(3, dtype=np.float64) - 0.5 * v_skew + (1.0 / 6.0) * (v_skew @ v_skew)
    
    theta2 = theta * theta
    theta3 = theta2 * theta
    term1 = (1.0 - np.cos(theta)) / theta2
    term2 = (theta - np.sin(theta)) / theta3
    return np.eye(3, dtype=np.float64) - term1 * v_skew + term2 * (v_skew @ v_skew)


def rot_to_euler_rpy(R: np.ndarray) -> Tuple[float, float, float]:
    """Extract Roll, Pitch, Yaw (radians) from DCM R (Body -> ENU).
    
    Convention:
        - Roll (phi): rotation about body +X (Forward)
        - Pitch (theta): rotation about body +Y (Lateral)
        - Yaw (psi): counter-clockwise angle from East in ENU plane
    """
    pitch = float(-np.arcsin(np.clip(R[2, 0], -1.0, 1.0)))
    if np.abs(np.cos(pitch)) > 1e-6:
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    else:
        # Gimbal lock singularity handling
        roll = float(np.arctan2(-R[1, 2], R[1, 1]))
        yaw = 0.0
    return roll, pitch, yaw


# ---------------------------------------------------------------------------
# Filter Configuration Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ESEKFConfig:
    """Process and measurement noise configuration for the 15-state ES-EKF."""
    # Continuous noise spectral densities
    accel_noise_density: float = 0.1       # m/s^2 / sqrt(Hz)
    gyro_noise_density: float = 0.01       # rad/s / sqrt(Hz)
    accel_bias_random_walk: float = 0.001  # m/s^3 / sqrt(Hz)
    gyro_bias_random_walk: float = 0.0001  # rad/s^2 / sqrt(Hz)

    # Initial standard deviations
    init_pos_std: float = 5.0              # meters
    init_vel_std: float = 0.5              # m/s
    init_att_std: float = 0.05             # radians (~2.8 deg)
    init_accel_bias_std: float = 0.05      # m/s^2
    init_gyro_bias_std: float = 0.005      # rad/s

    # Default measurement standard deviations
    default_gnss_pos_std: float = 3.0      # meters
    default_gnss_vel_std: float = 0.2      # m/s
    default_ai_speed_std: float = 1.5      # m/s
    default_nhc_lat_std: float = 0.05      # m/s
    default_nhc_vert_std: float = 0.05     # m/s
    default_zupt_std: float = 0.01         # m/s
    default_zaru_std: float = 0.001        # rad/s

    # Chi-square gating thresholds (99% confidence)
    chi2_gate_1d: float = 6.635            # 1-DOF
    chi2_gate_2d: float = 9.210            # 2-DOF
    chi2_gate_3d: float = 11.345           # 3-DOF


# ---------------------------------------------------------------------------
# 15-State Error-State Extended Kalman Filter
# ---------------------------------------------------------------------------

class ErrorStateKalmanFilter:
    """15-State 3D Error-State Extended Kalman Filter (ES-EKF)."""

    def __init__(self, config: Optional[ESEKFConfig] = None):
        self.config = config or ESEKFConfig()
        
        # Nominal state: [p(3), v(3), q(4), ba(3), bg(3)] (16 elements)
        self.p = np.zeros(3, dtype=np.float64)
        self.v = np.zeros(3, dtype=np.float64)
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # [qw, qx, qy, qz]
        self.ba = np.zeros(3, dtype=np.float64)
        self.bg = np.zeros(3, dtype=np.float64)

        # 15x15 Error-state covariance matrix P
        self.P = np.zeros((15, 15), dtype=np.float64)
        self._init_covariance()

        # Local gravity vector in ENU navigation frame (downward along -Z)
        self.g_nav = np.array([0.0, 0.0, -9.80665], dtype=np.float64)

        # Diagnostics & counters
        self.step_count = 0
        self.last_dt = 0.1
        self.last_acc = np.zeros(3, dtype=np.float64)
        self.last_gyro = np.zeros(3, dtype=np.float64)
        self.last_update_diagnostics: Dict[str, Any] = {}

    def _init_covariance(self):
        """Initialize covariance matrix P from configuration standard deviations."""
        cfg = self.config
        p_var = cfg.init_pos_std ** 2
        v_var = cfg.init_vel_std ** 2
        att_var = cfg.init_att_std ** 2
        ba_var = cfg.init_accel_bias_std ** 2
        bg_var = cfg.init_gyro_bias_std ** 2

        self.P = np.diag([
            p_var, p_var, p_var,
            v_var, v_var, v_var,
            att_var, att_var, att_var,
            ba_var, ba_var, ba_var,
            bg_var, bg_var, bg_var
        ]).astype(np.float64)

    @property
    def x(self) -> np.ndarray:
        """Backward-compatible 9D state vector [p(3), v(3), yaw, ba_x, ba_y]."""
        roll, pitch, yaw = self.euler_angles
        return np.array([
            self.p[0], self.p[1], self.p[2],
            self.v[0], self.v[1], self.v[2],
            yaw,
            self.ba[0], self.ba[1]
        ], dtype=np.float64)

    @x.setter
    def x(self, val: np.ndarray):
        """Backward-compatible setter for state vector."""
        arr = np.array(val, dtype=np.float64)
        if len(arr) >= 3:
            self.p = arr[0:3].copy()
        if len(arr) >= 6:
            self.v = arr[3:6].copy()
        if len(arr) >= 7:
            yaw = float(arr[6])
            roll, pitch, _ = self.euler_angles
            q_r = exp_quaternion(np.array([roll, 0.0, 0.0]))
            q_p = exp_quaternion(np.array([0.0, pitch, 0.0]))
            q_y = exp_quaternion(np.array([0.0, 0.0, yaw]))
            self.q = quat_multiply(q_y, quat_multiply(q_p, q_r))
        if len(arr) >= 9:
            self.ba[0] = arr[7]
            self.ba[1] = arr[8]

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Current Body -> ENU Direction Cosine Matrix R(q)."""
        return quat_to_rot(self.q)

    @property
    def euler_angles(self) -> Tuple[float, float, float]:
        """Current Euler angles (roll, pitch, yaw) in radians."""
        return rot_to_euler_rpy(self.rotation_matrix)

    @property
    def forward_speed(self) -> float:
        """Current forward speed along Body +X axis (m/s): v_body_x = e1^T R^T v."""
        v_body = self.rotation_matrix.T @ self.v
        return float(v_body[0])

    def set_state(
        self,
        pos: Optional[np.ndarray] = None,
        vel: Optional[np.ndarray] = None,
        quat: Optional[np.ndarray] = None,
        accel_bias: Optional[np.ndarray] = None,
        gyro_bias: Optional[np.ndarray] = None,
    ):
        """Manually override nominal state vector components with validation."""
        if pos is not None:
            self.p = np.array(pos, dtype=np.float64).reshape(3)
        if vel is not None:
            self.v = np.array(vel, dtype=np.float64).reshape(3)
        if quat is not None:
            q_arr = np.array(quat, dtype=np.float64).reshape(4)
            norm = np.linalg.norm(q_arr)
            if norm < 1e-6:
                raise ValueError("Quaternion norm cannot be near zero.")
            self.q = q_arr / norm
        if accel_bias is not None:
            self.ba = np.array(accel_bias, dtype=np.float64).reshape(3)
        if gyro_bias is not None:
            self.bg = np.array(gyro_bias, dtype=np.float64).reshape(3)

    def initialize_leveling(
        self,
        acc_meas: np.ndarray,
        yaw_rad: float = 0.0,
    ) -> None:
        """Deterministically initialize roll, pitch, and vertical bias from specific force.
        
        Aligns the initial quaternion q such that R(q) f_b + g_nav == 0 for stationary or cruising conditions.
        """
        a = np.array(acc_meas, dtype=np.float64).flatten()
        norm_a = float(np.linalg.norm(a))
        if norm_a < 1e-3 or not np.all(np.isfinite(a)):
            a = np.array([0.0, 0.0, 9.80665], dtype=np.float64)
            norm_a = 9.80665

        # Extract initial pitch and roll relative to the gravity vector
        u = a / norm_a
        pitch = float(np.arcsin(np.clip(-u[0], -1.0, 1.0)))
        roll = float(np.arctan2(u[1], u[2]))

        q_r = exp_quaternion(np.array([roll, 0.0, 0.0]))
        q_p = exp_quaternion(np.array([0.0, pitch, 0.0]))
        q_y = exp_quaternion(np.array([0.0, 0.0, yaw_rad]))
        self.q = quat_multiply(q_y, quat_multiply(q_p, q_r))
        self.q /= np.linalg.norm(self.q)

        # Initialize accelerometer bias along gravity axis if sensor scale differs from 9.80665 m/s^2
        scale_error = norm_a - 9.80665
        self.ba = scale_error * u

    # -----------------------------------------------------------------------
    # IMU Kinematic Propagation (Prediction Step)
    # -----------------------------------------------------------------------

    def predict(
        self,
        acc_meas: np.ndarray,
        gyro_meas: np.ndarray,
        dt: float,
    ) -> None:
        """Propagate nominal state and error covariance with 3-axis IMU measurements.
        
        Args:
            acc_meas: 3D specific force in Body frame [ax, ay, az] (m/s^2)
            gyro_meas: 3D angular velocity in Body frame [gx, gy, gz] (rad/s)
            dt: Integration time step (seconds)
        """
        # 1. Sanitize inputs and dt
        if not np.isfinite(dt) or dt <= 0.0 or dt > 1.0:
            dt = 0.1
        self.last_dt = dt

        acc = np.array(acc_meas, dtype=np.float64).reshape(3)
        gyro = np.array(gyro_meas, dtype=np.float64).reshape(3)

        if not np.all(np.isfinite(acc)):
            acc = np.nan_to_num(acc, nan=0.0, posinf=9.81, neginf=-9.81)
        if not np.all(np.isfinite(gyro)):
            gyro = np.nan_to_num(gyro, nan=0.0, posinf=0.0, neginf=0.0)

        self.last_acc = acc.copy()
        self.last_gyro = gyro.copy()

        # 2. Bias-corrected IMU measurements in Body frame
        f_b = acc - self.ba
        omega_b = gyro - self.bg

        # 3. Nominal attitude propagation
        delta_theta = omega_b * dt
        delta_q = exp_quaternion(delta_theta)
        q_next = quat_multiply(self.q, delta_q)
        q_norm = np.linalg.norm(q_next)
        if q_norm > 1e-6:
            self.q = q_next / q_norm
        else:
            self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # 4. Specific force rotation to navigation frame & acceleration
        R = quat_to_rot(self.q)
        a_nav = R @ f_b + self.g_nav

        # 5. Position & Velocity propagation (2nd-order discrete mechanization)
        dt2 = dt * dt
        self.p = self.p + self.v * dt + 0.5 * a_nav * dt2
        self.v = self.v + a_nav * dt

        # Physical ground-vehicle velocity bound: enforce max speed <= 80 m/s (288 km/h)
        v_norm = float(np.linalg.norm(self.v))
        if v_norm > 80.0:
            self.v = (self.v / v_norm) * 80.0

        # 6. Construct 15x15 Discrete Error Transition Matrix F_d
        F_d = self.compute_discrete_transition_matrix(f_b, omega_b, dt, R, delta_q)

        # 7. Construct 15x15 Discrete Process Noise Covariance Q_d
        Q_d = self.compute_process_noise_covariance(dt)

        # 8. Covariance propagation P_k+1 = F_d P_k F_d^T + Q_d
        self.P = F_d @ self.P @ F_d.T + Q_d
        self.P = 0.5 * (self.P + self.P.T)  # Enforce numerical symmetry

        self.step_count += 1

    def compute_discrete_transition_matrix(
        self,
        f_b: np.ndarray,
        omega_b: np.ndarray,
        dt: float,
        R: Optional[np.ndarray] = None,
        delta_q: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the 15x15 discrete state transition matrix F_d.
        
        Blocks:
            F_pv = I_3 * dt
            F_p_theta = -0.5 * R * [f_b]_x * dt^2
            F_p_ba = -0.5 * R * dt^2
            F_v_theta = -R * [f_b]_x * dt
            F_v_ba = -R * dt
            F_theta_theta = R(delta_q)^T
            F_theta_bg = -J_r(omega_b * dt) * dt   <-- Exact SO(3) Right Jacobian
        """
        if R is None:
            R = quat_to_rot(self.q)
        if delta_q is None:
            delta_q = exp_quaternion(omega_b * dt)

        dt2 = dt * dt
        f_b_skew = skew_symmetric(f_b)
        R_dq = quat_to_rot(delta_q)
        J_r = so3_right_jacobian(omega_b * dt)

        F_d = np.eye(15, dtype=np.float64)

        # Position coupling
        F_d[0:3, 3:6] = np.eye(3, dtype=np.float64) * dt
        F_d[0:3, 6:9] = -0.5 * (R @ f_b_skew) * dt2
        F_d[0:3, 9:12] = -0.5 * R * dt2

        # Velocity coupling
        F_d[3:6, 6:9] = -(R @ f_b_skew) * dt
        F_d[3:6, 9:12] = -R * dt

        # Attitude coupling
        F_d[6:9, 6:9] = R_dq.T
        F_d[6:9, 12:15] = -J_r * dt

        return F_d

    def compute_process_noise_covariance(self, dt: float) -> np.ndarray:
        """Compute the 15x15 discrete process noise covariance matrix Q_d."""
        cfg = self.config
        dt2 = dt * dt
        q_a = (cfg.accel_noise_density ** 2)
        q_g = (cfg.gyro_noise_density ** 2)
        q_ba = (cfg.accel_bias_random_walk ** 2)
        q_bg = (cfg.gyro_bias_random_walk ** 2)

        Q_d = np.zeros((15, 15), dtype=np.float64)
        # Position process noise contribution (integrated velocity random walk)
        Q_d[0:3, 0:3] = np.eye(3) * (0.25 * q_a * dt2 * dt2)
        # Velocity white noise
        Q_d[3:6, 3:6] = np.eye(3) * (q_a * dt)
        # Attitude white noise
        Q_d[6:9, 6:9] = np.eye(3) * (q_g * dt)
        # Accelerometer bias random walk
        Q_d[9:12, 9:12] = np.eye(3) * (q_ba * dt)
        # Gyroscope bias random walk
        Q_d[12:15, 12:15] = np.eye(3) * (q_bg * dt)

        return Q_d

    # -----------------------------------------------------------------------
    # Generic Measurement Update & Injection
    # -----------------------------------------------------------------------

    def update_measurement(
        self,
        z: np.ndarray,
        h_x: np.ndarray,
        H: np.ndarray,
        R_cov: np.ndarray,
        meas_name: str = "generic",
        gate_threshold: Optional[float] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Perform a generic linear or linearized error-state measurement update.
        
        Args:
            z: Measured vector (m,)
            h_x: Predicted measurement h(x) (m,)
            H: Observation Jacobian w.r.t 15-state error delta_x (m, 15)
            R_cov: Measurement noise covariance (m, m)
            meas_name: Identifier for diagnostic logging
            gate_threshold: Chi-square NIS rejection threshold
            
        Returns:
            (accepted, diagnostics_dict)
        """
        z = np.atleast_1d(np.array(z, dtype=np.float64))
        h_x = np.atleast_1d(np.array(h_x, dtype=np.float64))
        m = len(z)

        # 1. Innovation vector nu
        nu = z - h_x

        # 2. Innovation covariance S = H P H^T + R
        S = H @ self.P @ H.T + R_cov
        S = 0.5 * (S + S.T)

        # 3. Robust Inversion via Cholesky decomposition
        try:
            L = np.linalg.cholesky(S)
            # Solve L y = nu  =>  ||y||^2 = nu^T S^-1 nu (NIS)
            y = np.linalg.solve(L, nu)
            nis = float(np.sum(y * y))
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if S is near-singular
            S_inv = np.linalg.pinv(S)
            nis = float(nu.T @ S_inv @ nu)
            L = None

        # 4. Chi-square gating
        if gate_threshold is not None and nis > gate_threshold:
            diag = {
                "accepted": False,
                "measurement": meas_name,
                "dimension": m,
                "innovation": nu,
                "nis": nis,
                "gate_threshold": gate_threshold,
                "rejection_reason": f"NIS {nis:.2f} exceeded gate {gate_threshold:.2f}"
            }
            self.last_update_diagnostics = diag
            return False, diag

        # 5. Kalman Gain K = P H^T S^-1
        if L is not None:
            # K = (S^-1 H P)^T = (solve(S, H P))^T
            HP = H @ self.P
            K_t = np.linalg.solve(S, HP)
            K = K_t.T
        else:
            K = self.P @ H.T @ np.linalg.pinv(S)

        # 6. Error-state estimation delta_x_hat = K * nu
        delta_x = K @ nu

        # 7. Covariance update using Joseph stabilized formulation:
        #    P_post = (I - K H) P (I - K H)^T + K R K^T
        I_KH = np.eye(15, dtype=np.float64) - K @ H
        P_post = I_KH @ self.P @ I_KH.T + K @ R_cov @ K.T
        P_post = 0.5 * (P_post + P_post.T)

        # 8. Error-State Injection & Multiplicative Covariance Reset
        self.inject_error_and_reset(delta_x, P_post)

        diag = {
            "accepted": True,
            "measurement": meas_name,
            "dimension": m,
            "innovation": nu,
            "nis": nis,
            "gate_threshold": gate_threshold,
            "delta_x_norm": float(np.linalg.norm(delta_x)),
        }
        self.last_update_diagnostics = diag
        return True, diag

    def inject_error_and_reset(self, delta_x: np.ndarray, P_post: np.ndarray) -> None:
        """Inject 15-state estimated error into nominal state and reset covariance.
        
        Injections:
            p <- p + delta_p
            v <- v + delta_v
            q <- q (x) exp(delta_theta)  [Right-multiplicative body error]
            ba <- ba + delta_ba
            bg <- bg + delta_bg
            
        Covariance Reset:
            P <- G P_post G^T
            where G = diag(I_3, I_3, I_3 - 0.5 * [delta_theta]_x, I_3, I_3)
        """
        dp = delta_x[0:3]
        dv = delta_x[3:6]
        dtheta = delta_x[6:9]
        dba = delta_x[9:12]
        dbg = delta_x[12:15]

        # 1. State Injections
        self.p += dp
        self.v += dv
        
        dq = exp_quaternion(dtheta)
        q_new = quat_multiply(self.q, dq)
        self.q = q_new / np.linalg.norm(q_new)

        self.ba += dba
        self.bg += dbg

        # 2. Covariance Reset Transformation G
        G = np.eye(15, dtype=np.float64)
        G[6:9, 6:9] = np.eye(3, dtype=np.float64) - 0.5 * skew_symmetric(dtheta)

        self.P = G @ P_post @ G.T
        self.P = 0.5 * (self.P + self.P.T)

    # -----------------------------------------------------------------------
    # Specific Sensor Measurement Models
    # -----------------------------------------------------------------------

    def update_gnss_pos(
        self,
        pos_enu: np.ndarray,
        R_cov: Optional[np.ndarray] = None,
        gate_threshold: Optional[float] = None,
        is_trusted: bool = True,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Update with GNSS local ENU position measurement [East, North, Up].
        
        Args:
            pos_enu: Position measurement in local ENU frame (m)
            R_cov: Measurement noise covariance (3x3)
            gate_threshold: Chi-Square NIS threshold. If None and is_trusted is True,
                            no premature lockout is applied during normal fusion.
            is_trusted: Whether fix is passed from trusted GNSS engine
        """
        pos = np.array(pos_enu, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(pos)):
            return False, {"accepted": False, "reason": "Non-finite GNSS position"}

        if R_cov is None:
            std = self.config.default_gnss_pos_std
            R_cov = np.diag([std * std, std * std, (std * 2.0) ** 2])

        effective_gate = gate_threshold if gate_threshold is not None else (None if is_trusted else self.config.chi2_gate_3d)

        h_x = self.p.copy()
        H = np.zeros((3, 15), dtype=np.float64)
        H[0:3, 0:3] = np.eye(3, dtype=np.float64)

        return self.update_measurement(pos, h_x, H, R_cov, "gnss_pos", effective_gate)

    def update_gnss_vel(
        self,
        vel_enu: np.ndarray,
        R_cov: Optional[np.ndarray] = None,
        gate_threshold: Optional[float] = None,
        is_trusted: bool = True,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Update with GNSS local ENU velocity measurement [vE, vN, vU]."""
        vel = np.array(vel_enu, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(vel)):
            return False, {"accepted": False, "reason": "Non-finite GNSS velocity"}

        if R_cov is None:
            std = self.config.default_gnss_vel_std
            R_cov = np.eye(3, dtype=np.float64) * (std * std)

        effective_gate = gate_threshold if gate_threshold is not None else (None if is_trusted else self.config.chi2_gate_3d)

        h_x = self.v.copy()
        H = np.zeros((3, 15), dtype=np.float64)
        H[0:3, 3:6] = np.eye(3, dtype=np.float64)

        return self.update_measurement(vel, h_x, H, R_cov, "gnss_vel", effective_gate)

    def update_ai_velocity(
        self,
        speed_mps: float,
        sigma_v: Optional[float] = None,
        max_innovation_sigma: float = 3.0,
        min_sigma_v: float = 0.5,
        max_sigma_v: float = 10.0,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Update with AI-estimated forward vehicle speed (m/s).
        
        Observation:
            z_ai = e1^T R(q)^T v = v_body_x
        Jacobian:
            H_ai = [0_{1x3}, e1^T R^T, [0, -v_bz, v_by], 0_{1x3}, 0_{1x3}]
        """
        if not np.isfinite(speed_mps) or speed_mps < 0.0 or speed_mps > 80.0:
            return False, {"accepted": False, "reason": f"Invalid AI speed {speed_mps}"}

        sigma = float(np.clip(sigma_v or self.config.default_ai_speed_std, min_sigma_v, max_sigma_v))
        R_cov = np.array([[sigma * sigma]], dtype=np.float64)

        R = self.rotation_matrix
        v_body = R.T @ self.v
        h_x = np.array([v_body[0]], dtype=np.float64)

        H = np.zeros((1, 15), dtype=np.float64)
        H[0, 3:6] = R[:, 0]  # e1^T R^T
        H[0, 6:9] = np.array([0.0, -v_body[2], v_body[1]], dtype=np.float64)

        gate = (max_innovation_sigma ** 2)
        accepted, diag = self.update_measurement(np.array([speed_mps]), h_x, H, R_cov, "ai_velocity", gate)
        if not accepted:
            # If the filter has accumulated severe runaway velocity (> 15 m/s while AI reports low physical speed < 5 m/s),
            # re-anchor the forward velocity to the AI measurement to self-heal integration divergence
            if abs(float(h_x[0])) > 15.0 and speed_mps < 5.0:
                v_body_healed = v_body.copy()
                v_body_healed[0] = speed_mps
                self.v = R @ v_body_healed
                self.P[3:6, 3:6] = np.eye(3) * (sigma * sigma * 4.0)
                diag["accepted"] = True
                diag["reanchored"] = True
                return True, diag
        return accepted, diag

    def update_attitude(
        self,
        roll_rad: Optional[float] = None,
        pitch_rad: Optional[float] = None,
        yaw_rad: Optional[float] = None,
        sigma_att: float = 0.08,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Update filter attitude with 3D orientation (roll, pitch, yaw)."""
        current_roll, current_pitch, current_yaw = self.euler_angles
        innovations = []
        H_rows = []
        variances = []

        if roll_rad is not None and np.isfinite(roll_rad):
            d_roll = (roll_rad - current_roll + np.pi) % (2.0 * np.pi) - np.pi
            innovations.append(d_roll)
            h = np.zeros(15, dtype=np.float64)
            h[6] = 1.0  # delta_theta_x (roll)
            H_rows.append(h)
            variances.append(sigma_att * sigma_att)

        if pitch_rad is not None and np.isfinite(pitch_rad):
            d_pitch = (pitch_rad - current_pitch + np.pi) % (2.0 * np.pi) - np.pi
            innovations.append(d_pitch)
            h = np.zeros(15, dtype=np.float64)
            h[7] = 1.0  # delta_theta_y (pitch)
            H_rows.append(h)
            variances.append(sigma_att * sigma_att)

        if yaw_rad is not None and np.isfinite(yaw_rad):
            d_yaw = (yaw_rad - current_yaw + np.pi) % (2.0 * np.pi) - np.pi
            innovations.append(d_yaw)
            h = np.zeros(15, dtype=np.float64)
            h[8] = 1.0  # delta_theta_z (yaw)
            H_rows.append(h)
            variances.append(sigma_att * sigma_att)

        if not innovations:
            return False, {"accepted": False, "reason": "No valid attitude angles provided"}

        z = np.array(innovations, dtype=np.float64)
        h_x = np.zeros(len(innovations), dtype=np.float64)
        H = np.vstack(H_rows)
        R_cov = np.diag(variances)

        # Ensure attitude prior covariance is responsive to absolute physical orientation updates
        for row, var in zip(H_rows, variances):
            idx = int(np.argmax(row))
            if self.P[idx, idx] < var * 4.0:
                self.P[idx, idx] = var * 4.0

        return self.update_measurement(z, h_x, H, R_cov, "attitude_3d", gate_threshold=None)

    def update_nhc(
        self,
        sigma_lat: Optional[float] = None,
        sigma_vert: Optional[float] = None,
        gate_threshold: Optional[float] = None,
        yaw_rate: Optional[float] = None,
        forward_speed: Optional[float] = None,
        cornering_threshold_mps2: float = 0.5,
        turn_policy: str = "decouple",
    ) -> Tuple[bool, Dict[str, Any]]:
        """Update with Non-Holonomic Constraints (v_body_y = 0, v_body_z = 0).
        
        turn_policy options:
            'decouple': Decouple lateral velocity residual from yaw attitude error (Phase 30A).
            'disable_lateral': Completely disable lateral NHC during dynamic cornering, retaining vertical NHC (Phase 31A).
            'coupled': Full 3D attitude coupling (Phase 28 legacy).
        """
        s_lat = sigma_lat or self.config.default_nhc_lat_std
        s_vert = sigma_vert or self.config.default_nhc_vert_std

        R = self.rotation_matrix
        v_body = R.T @ self.v
        v_bx = float(forward_speed) if forward_speed is not None else float(v_body[0])
        v_by = float(v_body[1])
        v_bz = float(v_body[2])
        
        w_z = float(yaw_rate) if yaw_rate is not None else float(self.last_gyro[2] - self.bg[2])
        a_lat = abs(w_z * v_bx)
        is_dynamic_cornering = (a_lat > cornering_threshold_mps2)

        if is_dynamic_cornering and turn_policy == "disable_lateral":
            # Real vehicles experience lateral tire slip during cornering; disable lateral NHC, retain vertical NHC
            z = np.zeros(1, dtype=np.float64)
            h_x = np.array([v_bz], dtype=np.float64)
            H = np.zeros((1, 15), dtype=np.float64)
            H[0, 3:6] = R[:, 2]  # e3^T R^T
            H[0, 6:9] = np.array([-v_by, v_bx, 0.0], dtype=np.float64)
            R_cov = np.array([[s_vert * s_vert]], dtype=np.float64)
            gate = gate_threshold if gate_threshold is not None else self.config.chi2_gate_1d
            accepted, diag = self.update_measurement(z, h_x, H, R_cov, "nhc_vert_only", gate)
            diag["is_dynamic_cornering"] = is_dynamic_cornering
            diag["a_lat"] = a_lat
            diag["turn_policy"] = turn_policy
            return accepted, diag

        # 2-DOF NHC
        R_cov = np.diag([s_lat * s_lat, s_vert * s_vert]).astype(np.float64)
        h_x = np.array([v_body[1], v_body[2]], dtype=np.float64)
        z = np.zeros(2, dtype=np.float64)

        H = np.zeros((2, 15), dtype=np.float64)
        H[0, 3:6] = R[:, 1]  # e2^T R^T (lateral velocity)
        H[1, 3:6] = R[:, 2]  # e3^T R^T (vertical velocity)

        if is_dynamic_cornering and turn_policy == "decouple":
            # Decouple lateral velocity error from yaw attitude
            H[0, 6:9] = np.zeros(3, dtype=np.float64)
            H[1, 6:9] = np.array([-v_by, v_bx, 0.0], dtype=np.float64)
        else:
            # Standard full 3D attitude coupling
            H[0, 6:9] = np.array([v_bz, 0.0, -v_bx], dtype=np.float64)
            H[1, 6:9] = np.array([-v_by, v_bx, 0.0], dtype=np.float64)

        if gate_threshold is None:
            gate_threshold = self.config.chi2_gate_2d

        accepted, diag = self.update_measurement(z, h_x, H, R_cov, "nhc", gate_threshold)
        diag["is_dynamic_cornering"] = is_dynamic_cornering
        diag["a_lat"] = a_lat
        diag["turn_policy"] = turn_policy
        return accepted, diag

    def update_zupt(
        self,
        sigma_v: Optional[float] = None,
        gate_threshold: Optional[float] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Apply Zero Velocity Update (ZUPT): v_nav = 0."""
        std = sigma_v or self.config.default_zupt_std
        R_cov = np.eye(3, dtype=np.float64) * (std * std)

        z = np.zeros(3, dtype=np.float64)
        h_x = self.v.copy()

        H = np.zeros((3, 15), dtype=np.float64)
        H[0:3, 3:6] = np.eye(3, dtype=np.float64)

        # By default, ZUPT is an absolute physical constraint and should not be gated out
        effective_gate = gate_threshold

        return self.update_measurement(z, h_x, H, R_cov, "zupt", effective_gate)

    def update_zaru(
        self,
        gyro_meas: Optional[np.ndarray] = None,
        sigma_bg: Optional[float] = None,
        gate_threshold: Optional[float] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Apply Zero Angular Rate Update (ZARU) for stationary gyro bias estimation.
        
        Observation:
            z_zaru = omega_meas - bg ~= 0  =>  observes delta_bg
        """
        std = sigma_bg or self.config.default_zaru_std
        R_cov = np.eye(3, dtype=np.float64) * (std * std)

        z = np.zeros(3, dtype=np.float64)
        if gyro_meas is not None:
            g = np.array(gyro_meas, dtype=np.float64).reshape(3)
            h_x = g - self.bg
        else:
            h_x = -self.bg.copy()

        H = np.zeros((3, 15), dtype=np.float64)
        H[0:3, 12:15] = -np.eye(3, dtype=np.float64)

        effective_gate = gate_threshold

        return self.update_measurement(z, h_x, H, R_cov, "zaru", effective_gate)

    def update_heading(
        self,
        heading_rad: float,
        sigma_yaw: float = 0.05,
        gate_threshold: Optional[float] = None,
        is_trusted: bool = True,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Update filter with an absolute yaw measurement in ENU navigation frame.
        
        Observation:
            psi_nav = heading_rad
        """
        roll, pitch, current_yaw = self.euler_angles
        diff_yaw = (heading_rad - current_yaw + np.pi) % (2.0 * np.pi) - np.pi

        z = np.array([diff_yaw], dtype=np.float64)
        h_x = np.zeros(1, dtype=np.float64)

        # For small pitch/roll, delta_yaw ~= delta_theta_z
        H = np.zeros((1, 15), dtype=np.float64)
        H[0, 8] = 1.0  # delta_theta_z

        R_cov = np.array([[sigma_yaw * sigma_yaw]], dtype=np.float64)
        effective_gate = gate_threshold if gate_threshold is not None else (None if is_trusted else self.config.chi2_gate_1d)

        return self.update_measurement(z, h_x, H, R_cov, "heading", effective_gate)

    def update_magnetic_heading(
        self,
        heading_rad: float,
        sigma_yaw: float = 0.15,
        gate_threshold: Optional[float] = None,
        is_clean: bool = True,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Update filter with a tilt-compensated magnetic heading measurement.
        
        Requirements:
        - If not is_clean: rejects update immediately.
        - Innovation angle wrapped to [-pi, pi].
        - Chi-square 1-DOF gating.
        """
        if not is_clean:
            diag = {
                "accepted": False,
                "measurement": "magnetic_heading",
                "rejection_reason": "Magnetic disturbance detected / unclean sample",
            }
            self.last_update_diagnostics = diag
            return False, diag

        roll, pitch, current_yaw = self.euler_angles
        diff_yaw = (heading_rad - current_yaw + np.pi) % (2.0 * np.pi) - np.pi

        z = np.array([diff_yaw], dtype=np.float64)
        h_x = np.zeros(1, dtype=np.float64)

        # Yaw error Jacobian in body/nav
        H = np.zeros((1, 15), dtype=np.float64)
        H[0, 8] = 1.0  # delta_theta_z

        R_cov = np.array([[sigma_yaw * sigma_yaw]], dtype=np.float64)
        if gate_threshold is None:
            gate_threshold = self.config.chi2_gate_1d

        return self.update_measurement(z, h_x, H, R_cov, "magnetic_heading", gate_threshold)

    def update_gravity_leveling(
        self,
        acc_meas: np.ndarray,
        forward_speed: float = 0.0,
        yaw_rate: float = 0.0,
        fwd_acc: float = 0.0,
        sigma_level: float = 0.5,
        gate_threshold: Optional[float] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Apply dynamic gravity leveling update to constrain pitch and roll drift.
        
        Compensates kinematic accelerations (centripetal and longitudinal) from the accelerometer:
            z_level = acc_meas - a_kinematic
            h_x = -R(q)^T g_nav
        """
        a_m = np.array(acc_meas, dtype=np.float64).reshape(3)
        # Kinematic acceleration in body coordinates [fwd_acc, centripetal, 0]
        centripetal = float(yaw_rate * forward_speed)
        a_kin = np.array([fwd_acc, centripetal, 0.0], dtype=np.float64)

        z_g = a_m - a_kin - self.ba
        h_g = -self.rotation_matrix.T @ self.g_nav

        # Leveling acts primarily on the 2 horizontal body axes (roll and pitch)
        z = z_g[0:2]
        h_x = h_g[0:2]

        H = np.zeros((2, 15), dtype=np.float64)
        # Attitude error Jacobian: [h_g]_x
        H[0:2, 6:9] = skew_symmetric(h_g)[0:2, :]
        # Accelerometer bias Jacobian
        H[0:2, 9:12] = np.eye(3)[0:2, :]

        R_cov = np.diag([sigma_level * sigma_level, sigma_level * sigma_level]).astype(np.float64)
        if gate_threshold is None:
            gate_threshold = self.config.chi2_gate_2d

        return self.update_measurement(z, h_x, H, R_cov, "gravity_leveling", gate_threshold)

    # -----------------------------------------------------------------------
    # Diagnostics & Telemetry
    # -----------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        """Export structured nominal state and covariance diagnostics."""
        roll, pitch, yaw = self.euler_angles
        diag_P = np.diag(self.P)
        return {
            "position_enu": self.p.copy(),
            "velocity_enu": self.v.copy(),
            "quaternion": self.q.copy(),
            "accel_bias": self.ba.copy(),
            "gyro_bias": self.bg.copy(),
            "roll_rad": roll,
            "pitch_rad": pitch,
            "yaw_rad": yaw,
            "forward_speed_mps": self.forward_speed,
            "pos_std_m": float(np.sqrt(np.mean(diag_P[0:3]))),
            "vel_std_mps": float(np.sqrt(np.mean(diag_P[3:6]))),
            "att_std_rad": float(np.sqrt(np.mean(diag_P[6:9]))),
            "accel_bias_std": float(np.sqrt(np.mean(diag_P[9:12]))),
            "gyro_bias_std": float(np.sqrt(np.mean(diag_P[12:15]))),
            "step_count": self.step_count,
        }
