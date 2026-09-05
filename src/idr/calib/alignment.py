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
        # Step 1: Vertical axis (Z_v) from mean gravity vector
        mean_g = np.mean(stationary_acc, axis=0)
        norm_g = np.linalg.norm(mean_g)
        if norm_g < 1e-3:
            z_phone = np.array([0.0, 0.0, 1.0])
        else:
            # Gravity points downwards; vehicle Z points upwards
            z_phone = mean_g / norm_g

        # Step 2: Forward axis (X_v) from longitudinal acceleration
        # Subtract gravity projection to obtain dynamic acceleration
        dyn_acc = motion_acc - np.outer(motion_acc @ z_phone, z_phone)
        
        # Principal axis of forward motion via PCA (first eigenvector)
        cov = np.cov(dyn_acc, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        x_phone = eigenvectors[:, np.argmax(eigenvalues)]

        # Determine forward direction sign using velocity change
        if motion_vel is not None and len(motion_vel) > 1:
            dv = motion_vel[-1] - motion_vel[0]
            if np.dot(x_phone, dv[:3]) < 0:
                x_phone = -x_phone

        # Step 3: Lateral axis (Y_v) = Z_v x X_v (orthonormal right-handed frame)
        y_phone = np.cross(z_phone, x_phone)
        norm_y = np.linalg.norm(y_phone)
        if norm_y > 1e-4:
            y_phone /= norm_y
        else:
            y_phone = np.array([0.0, 1.0, 0.0])

        # Re-orthogonalize X = Y x Z
        x_phone = np.cross(y_phone, z_phone)
        x_phone /= np.linalg.norm(x_phone)

        # R maps phone coordinates to vehicle frame: v_vehicle = R @ v_phone
        # R rows are [x_phone; y_phone; z_phone]
        R = np.vstack([x_phone, y_phone, z_phone]).astype(np.float32)
        self.R_phone_to_vehicle = R
        self.is_calibrated = True
        return R

    def transform_imu(self, acc: np.ndarray, gyro: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Rotate IMU vectors into the vehicle frame."""
        # acc: (N, 3), gyro: (N, 3)
        acc_v = (self.R_phone_to_vehicle @ acc.T).T
        gyro_v = (self.R_phone_to_vehicle @ gyro.T).T
        return acc_v, gyro_v
