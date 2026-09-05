"""Unscented Kalman Filter (UKF) for non-linear vehicle dynamics."""

from typing import Callable
import numpy as np

class UnscentedKalmanFilter:
    """9-State Unscented Kalman Filter using standard Julier-Uhlmann sigma points."""

    def __init__(self, dt: float = 0.1, alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0):
        self.dt = dt
        self.dim_x = 9
        self.x = np.zeros(self.dim_x, dtype=np.float64)
        self.P = np.eye(self.dim_x, dtype=np.float64) * 5.0
        self.Q = np.eye(self.dim_x, dtype=np.float64) * 0.05
        
        # Scaling parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lam = alpha**2 * (self.dim_x + kappa) - self.dim_x
        self.gamma = np.sqrt(self.dim_x + self.lam)

        # Compute weights
        num_sigmas = 2 * self.dim_x + 1
        self.Wm = np.full(num_sigmas, 0.5 / (self.dim_x + self.lam))
        self.Wc = np.full(num_sigmas, 0.5 / (self.dim_x + self.lam))
        self.Wm[0] = self.lam / (self.dim_x + self.lam)
        self.Wc[0] = self.lam / (self.dim_x + self.lam) + (1.0 - alpha**2 + beta)

    def generate_sigma_points(self) -> np.ndarray:
        # Cholesky decomposition of P
        L = np.linalg.cholesky(self.P + 1e-8 * np.eye(self.dim_x))
        sigmas = np.zeros((2 * self.dim_x + 1, self.dim_x))
        sigmas[0] = self.x
        for i in range(self.dim_x):
            sigmas[i + 1] = self.x + self.gamma * L[:, i]
            sigmas[self.dim_x + i + 1] = self.x - self.gamma * L[:, i]
        return sigmas

    def predict(self, fwd_accel: float, yaw_rate: float):
        sigmas = self.generate_sigma_points()
        dt = self.dt
        pred_sigmas = np.zeros_like(sigmas)

        for i, s in enumerate(sigmas):
            psi = s[6]
            ba = s[7]
            bw = s[8]
            acc = fwd_accel - ba
            w = yaw_rate - bw

            s_next = s.copy()
            s_next[0] += s[3] * dt + 0.5 * (acc * np.cos(psi)) * dt**2
            s_next[1] += s[4] * dt + 0.5 * (acc * np.sin(psi)) * dt**2
            s_next[2] += s[5] * dt
            s_next[3] += acc * np.cos(psi) * dt
            s_next[4] += acc * np.sin(psi) * dt
            s_next[5] = 0.0
            s_next[6] = (s[6] + w * dt + np.pi) % (2 * np.pi) - np.pi
            pred_sigmas[i] = s_next

        # Recombine predicted state
        self.x = np.sum(self.Wm[:, None] * pred_sigmas, axis=0)
        y = pred_sigmas - self.x[None, :]
        self.P = np.sum(self.Wc[:, None, None] * (y[:, :, None] @ y[:, None, :]), axis=0) + self.Q

    def update_measurement(self, z: np.ndarray, h_func: Callable[[np.ndarray], np.ndarray], R: np.ndarray):
        sigmas = self.generate_sigma_points()
        z_sigmas = np.array([h_func(s) for s in sigmas])
        z_pred = np.sum(self.Wm[:, None] * z_sigmas, axis=0)

        P_zz = R.copy()
        P_xz = np.zeros((self.dim_x, len(z)))

        for i in range(len(sigmas)):
            dz = z_sigmas[i] - z_pred
            dx = sigmas[i] - self.x
            P_zz += self.Wc[i] * np.outer(dz, dz)
            P_xz += self.Wc[i] * np.outer(dx, dz)

        K = P_xz @ np.linalg.inv(P_zz)
        self.x = self.x + K @ (z - z_pred)
        self.P = self.P - K @ P_zz @ K.T
