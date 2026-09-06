"""Estimation of phone-to-vehicle rotation matrix (attitude alignment).

Transforms arbitrary smartphone mounting orientations into the vehicle reference frame:
- X_v: Vehicle forward
- Y_v: Vehicle lateral (left/right)
- Z_v: Vehicle vertical (upward)
"""

from typing import Optional, Tuple
import numpy as np

def compute_rotation_matrix(pitch: float, roll: float, yaw: float) -> np.ndarray:
    """Euler Z-Y-X rotation matrix."""
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx

class PhoneToVehicleAligner:
    """Estimates and tracks the 3D rotation matrix R_{phone -> vehicle}."""

    def __init__(self):
        self.R_phone_to_vehicle = np.eye(3, dtype=np.float32)
        self.is_calibrated = False

    def estimate_from_stationary_and_motion(
        self,
        stationary_acc: np.ndarray,
        motion_acc: np.ndarray,
        motion_vel: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Estimate attitude using gravity vector and forward acceleration direction.
        
        Args:
            stationary_acc: (M, 3) accel when vehicle is stationary (pure gravity + sensor bias)
            motion_acc: (K, 3) accel during forward acceleration / braking
            motion_vel: Optional (K, 3) velocity in local frame
        """
        # ── Step 0: Input Sanitization & Finite Validation ────────────────────
        stat = np.asarray(stationary_acc, dtype=np.float64)
        if stat.ndim == 1:
            stat = stat.reshape(1, -1)
        valid_stat = stat[np.all(np.isfinite(stat), axis=1)]
        if len(valid_stat) == 0:
            valid_stat = np.array([[0.0, 0.0, 9.81]])

        # ── Step 1: Vertical Axis (Z_v) from Mean Gravity Vector ─────────────
        mean_g = np.mean(valid_stat, axis=0)
        norm_g = float(np.linalg.norm(mean_g))
        if not np.isfinite(norm_g) or norm_g < 1e-3:
            z_phone = np.array([0.0, 0.0, 1.0])
        else:
            # Gravity points downwards; vehicle Z points upwards
            z_phone = mean_g / norm_g

        # ── Step 2: Forward Axis (X_v) from Longitudinal Acceleration ────────
        mot = np.asarray(motion_acc, dtype=np.float64)
        if mot.ndim == 1:
            mot = mot.reshape(1, -1)
        valid_mot = mot[np.all(np.isfinite(mot), axis=1)]

        x_phone: Optional[np.ndarray] = None

        # Only compute PCA if we have at least 2 distinct dynamic samples
        if len(valid_mot) >= 2:
            # Subtract gravity projection to obtain dynamic acceleration
            dyn_acc = valid_mot - np.outer(valid_mot @ z_phone, z_phone)
            if np.all(np.isfinite(dyn_acc)):
                cov = np.cov(dyn_acc, rowvar=False)
                # Ensure covariance is finite and has non-degenerate variance (trace > 1e-5)
                if np.all(np.isfinite(cov)) and float(np.trace(cov)) > 1e-5:
                    try:
                        eigenvalues, eigenvectors = np.linalg.eigh(cov)
                        if np.all(np.isfinite(eigenvalues)) and np.all(np.isfinite(eigenvectors)):
                            x_phone = eigenvectors[:, int(np.argmax(eigenvalues))]
                    except np.linalg.LinAlgError:
                        x_phone = None

        # Safe deterministic horizontal reference if PCA is degenerate or unavailable
        if x_phone is None or not np.all(np.isfinite(x_phone)):
            # Pick a canonical horizontal reference orthogonal to z_phone
            ref_vec = np.array([0.0, 1.0, 0.0]) if abs(z_phone[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
            x_cand = ref_vec - np.dot(ref_vec, z_phone) * z_phone
            norm_x = float(np.linalg.norm(x_cand))
            if norm_x > 1e-4:
                x_phone = x_cand / norm_x
            else:
                ref_vec2 = np.array([1.0, 0.0, 0.0])
                x_cand2 = ref_vec2 - np.dot(ref_vec2, z_phone) * z_phone
                x_phone = x_cand2 / (float(np.linalg.norm(x_cand2)) + 1e-8)

        # Determine forward direction sign using velocity change
        if motion_vel is not None and len(motion_vel) > 1:
            valid_vel = np.asarray(motion_vel, dtype=np.float64)
            if np.all(np.isfinite(valid_vel)):
                dv = valid_vel[-1] - valid_vel[0]
                if np.dot(x_phone, dv[:3]) < 0:
                    x_phone = -x_phone

        # ── Step 3: Lateral Axis (Y_v) & Gram-Schmidt Orthonormalization ───────
        y_phone = np.cross(z_phone, x_phone)
        norm_y = float(np.linalg.norm(y_phone))
        if norm_y > 1e-4:
            y_phone /= norm_y
        else:
            y_phone = np.array([0.0, 1.0, 0.0])

        # Re-orthogonalize X = Y x Z to guarantee exact right-handed orthonormal frame
        x_phone = np.cross(y_phone, z_phone)
        norm_final_x = float(np.linalg.norm(x_phone))
        if norm_final_x > 1e-4:
            x_phone /= norm_final_x

        # R maps phone coordinates to vehicle frame: v_vehicle = R @ v_phone
        R = np.vstack([x_phone, y_phone, z_phone]).astype(np.float32)
        if np.all(np.isfinite(R)):
            self.R_phone_to_vehicle = R
            self.is_calibrated = True
        return self.R_phone_to_vehicle

    def transform_imu(self, acc: np.ndarray, gyro: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Rotate IMU vectors into the vehicle frame."""
        # acc: (N, 3), gyro: (N, 3)
        acc_arr = np.asarray(acc, dtype=np.float32)
        gyro_arr = np.asarray(gyro, dtype=np.float32)
        if not np.all(np.isfinite(acc_arr)):
            acc_arr = np.nan_to_num(acc_arr, nan=0.0)
        if not np.all(np.isfinite(gyro_arr)):
            gyro_arr = np.nan_to_num(gyro_arr, nan=0.0)
        acc_v = (self.R_phone_to_vehicle @ acc_arr.T).T
        gyro_v = (self.R_phone_to_vehicle @ gyro_arr.T).T
        return acc_v, gyro_v
