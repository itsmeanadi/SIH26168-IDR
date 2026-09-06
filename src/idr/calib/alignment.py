"""Estimation of phone-to-vehicle rotation matrix (attitude alignment).

Transforms arbitrary smartphone mounting orientations into the vehicle reference frame:
- X_v: Vehicle forward (longitudinal)
- Y_v: Vehicle lateral (left/right)
- Z_v: Vehicle vertical (upward)

Implements a robust 3-Stage Stateful Calibration Architecture:
1. STAGE 1 (Gravity / Leveling): While stationary, accumulates valid accelerometer samples,
   computes the mean gravity vector, and locks the vertical axis (Z_v). Complete alignment
   remains in CALIBRATING_FORWARD until dynamic motion evidence is observed.
2. STAGE 2 (Forward Axis PCA): While in motion, extracts gravity-free dynamic acceleration,
   verifies sufficient sample count and eigenvalue separation (rejecting isotropic vibration),
   and extracts the principal longitudinal motion axis.
3. STAGE 3 (PCA Sign Ambiguity Resolution): Resolves the +X vs -X forward direction using
   velocity delta, positive acceleration impulse, or GNSS speed progression.
"""

from enum import Enum
from typing import List, Optional, Tuple
import numpy as np


class AlignmentState(str, Enum):
    """Phone-to-vehicle alignment state machine."""
    NOT_CALIBRATED = "NOT_CALIBRATED"
    CALIBRATING_GRAVITY = "CALIBRATING_GRAVITY"
    CALIBRATING_FORWARD = "CALIBRATING_FORWARD"
    CALIBRATED = "CALIBRATED"


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
    """Estimates, validates, and locks the 3D rotation matrix R_{phone -> vehicle}."""

    def __init__(
        self,
        min_stationary_samples: int = 15,
        max_stationary_variance: float = 0.25,
        min_motion_samples: int = 20,
        min_motion_energy: float = 0.05,
        min_eigenvalue_ratio: float = 2.0,
    ):
        self.min_stationary_samples = min_stationary_samples
        self.max_stationary_variance = max_stationary_variance
        self.min_motion_samples = min_motion_samples
        self.min_motion_energy = min_motion_energy
        self.min_eigenvalue_ratio = min_eigenvalue_ratio

        self.R_phone_to_vehicle = np.eye(3, dtype=np.float32)
        self.state = AlignmentState.NOT_CALIBRATED
        self.is_calibrated = False

        self.z_phone: Optional[np.ndarray] = None
        self.x_phone: Optional[np.ndarray] = None
        self.y_phone: Optional[np.ndarray] = None

        self.stationary_samples: List[np.ndarray] = []
        self.motion_samples: List[np.ndarray] = []
        self.motion_speeds: List[float] = []
        self.motion_raw_accs: List[np.ndarray] = []

    def reset(self):
        """Reset aligner state and buffers."""
        self.R_phone_to_vehicle = np.eye(3, dtype=np.float32)
        self.state = AlignmentState.NOT_CALIBRATED
        self.is_calibrated = False
        self.z_phone = None
        self.x_phone = None
        self.y_phone = None
        self.stationary_samples.clear()
        self.motion_samples.clear()
        self.motion_speeds.clear()
        self.motion_raw_accs.clear()

    def update(
        self,
        acc_raw: np.ndarray,
        gyro_raw: Optional[np.ndarray] = None,
        is_stationary: bool = True,
        speed_mps: Optional[float] = None,
        gnss_vel: Optional[np.ndarray] = None,
    ) -> AlignmentState:
        """Process streaming sensor frame to progress calibration state machine."""
        if self.is_calibrated:
            return self.state

        acc = np.asarray(acc_raw, dtype=np.float64).flatten()
        if len(acc) != 3 or not np.all(np.isfinite(acc)):
            return self.state

        # ── STAGE 1: Gravity / Vertical Axis Estimation (while stationary) ──
        if self.state in (AlignmentState.NOT_CALIBRATED, AlignmentState.CALIBRATING_GRAVITY):
            if is_stationary:
                self.state = AlignmentState.CALIBRATING_GRAVITY
                self.stationary_samples.append(acc)
                if len(self.stationary_samples) >= self.min_stationary_samples:
                    stat_arr = np.array(self.stationary_samples[-self.min_stationary_samples:])
                    # Verify low variance during stationary period (not being handled/shaken)
                    norms = np.linalg.norm(stat_arr, axis=1)
                    if float(np.var(norms)) <= self.max_stationary_variance:
                        mean_g = np.mean(stat_arr, axis=0)
                        norm_g = float(np.linalg.norm(mean_g))
                        if 7.0 <= norm_g <= 12.5:
                            self.z_phone = mean_g / norm_g
                            self.state = AlignmentState.CALIBRATING_FORWARD
                            self._build_interim_rotation()
            else:
                # If motion happens before gravity calibration completes, keep accumulating
                if len(self.stationary_samples) > 0:
                    self.stationary_samples.pop(0)

        # ── STAGE 2 & 3: Forward Axis Estimation & Sign Resolution (during motion) ──
        elif self.state == AlignmentState.CALIBRATING_FORWARD:
            if not is_stationary and self.z_phone is not None:
                # Remove gravity component
                dyn_acc = acc - np.dot(acc, self.z_phone) * self.z_phone
                self.motion_samples.append(dyn_acc)
                self.motion_raw_accs.append(acc)
                if speed_mps is not None and np.isfinite(speed_mps):
                    self.motion_speeds.append(float(speed_mps))

                # Keep a bounded sliding window of motion samples (up to 150 frames)
                if len(self.motion_samples) > 150:
                    self.motion_samples.pop(0)
                    self.motion_raw_accs.pop(0)
                    if len(self.motion_speeds) > 150:
                        self.motion_speeds.pop(0)

                if len(self.motion_samples) >= self.min_motion_samples:
                    mot_arr = np.array(self.motion_samples)
                    # Uncentered second-moment energy matrix S = (1/N) M^T M
                    cov = (mot_arr.T @ mot_arr) / len(mot_arr)
                    trace_cov = float(np.trace(cov))

                    # Check 1: Must have significant dynamic energy (not micro-vibration)
                    if np.all(np.isfinite(cov)) and trace_cov >= self.min_motion_energy:
                        try:
                            eigenvalues, eigenvectors = np.linalg.eigh(cov)
                            if np.all(np.isfinite(eigenvalues)) and np.all(np.isfinite(eigenvectors)):
                                # Sort eigenvalues descending
                                idxs = np.argsort(eigenvalues)[::-1]
                                eigvals = eigenvalues[idxs]
                                eigvecs = eigenvectors[:, idxs]

                                l1, l2 = eigvals[0], eigvals[1]
                                # Check 2: Strong longitudinal dominance over transverse/lateral noise
                                if (l2 <= 1e-5) or (l1 / (l2 + 1e-6) >= self.min_eigenvalue_ratio):
                                    x_cand = eigvecs[:, 0]
                                    # Project onto horizontal plane
                                    x_cand = x_cand - np.dot(x_cand, self.z_phone) * self.z_phone
                                    norm_x = float(np.linalg.norm(x_cand))
                                    if norm_x > 1e-4:
                                        x_cand /= norm_x

                                        # Stage 3: Resolve sign ambiguity
                                        sign_resolved, sign = self._resolve_forward_sign(
                                            x_cand, gnss_vel=gnss_vel
                                        )
                                        if sign_resolved:
                                            x_cand *= sign
                                            self._lock_calibration(x_cand)

                        except np.linalg.LinAlgError:
                            pass

        return self.state

    def _resolve_forward_sign(
        self,
        x_cand: np.ndarray,
        gnss_vel: Optional[np.ndarray] = None,
    ) -> Tuple[bool, float]:
        """Resolves whether forward direction is +x_cand or -x_cand using velocity or impulse evidence."""
        # Evidence A: GNSS velocity vector delta
        if gnss_vel is not None and len(gnss_vel) > 1:
            valid_vel = np.asarray(gnss_vel, dtype=np.float64)
            if np.all(np.isfinite(valid_vel)):
                dv = valid_vel[-1] - valid_vel[0]
                if float(np.linalg.norm(dv[:2])) > 0.5:
                    dot = float(np.dot(x_cand, dv[:3]))
                    if abs(dot) > 1e-3:
                        return True, (1.0 if dot > 0 else -1.0)

        # Evidence B: Speed progression during start-up acceleration
        if len(self.motion_speeds) >= self.min_motion_samples:
            v_start = self.motion_speeds[0]
            v_end = self.motion_speeds[-1]
            dv = v_end - v_start
            mot_arr = np.array(self.motion_samples)
            mean_acc = np.mean(mot_arr, axis=0)

            # Acceleration from stop (speed increasing): mean acceleration aligns with forward
            if dv > 0.8:
                dot = float(np.dot(mean_acc, x_cand))
                if abs(dot) > 1e-3:
                    return True, (1.0 if dot > 0 else -1.0)
            # Deceleration / braking (speed decreasing): mean deceleration aligns with backward (-forward)
            elif dv < -0.8:
                dot = float(np.dot(mean_acc, x_cand))
                if abs(dot) > 1e-3:
                    return True, (-1.0 if dot > 0 else 1.0)

        # Evidence C: Pure acceleration burst from standstill
        mot_arr = np.array(self.motion_samples)
        mean_acc = np.mean(mot_arr, axis=0)
        norm_mean = float(np.linalg.norm(mean_acc))
        if norm_mean > 0.4:
            dot = float(np.dot(mean_acc, x_cand))
            if abs(dot) > 0.1:
                return True, (1.0 if dot > 0 else -1.0)

        # Insufficient evidence to disambiguate forward from backward reliably
        return False, 1.0

    def _lock_calibration(self, x_phone: np.ndarray):
        """Constructs right-handed orthonormal frame and locks calibration."""
        z_phone = self.z_phone
        y_phone = np.cross(z_phone, x_phone)
        norm_y = float(np.linalg.norm(y_phone))
        if norm_y > 1e-4:
            y_phone /= norm_y
        else:
            y_phone = np.array([0.0, 1.0, 0.0])

        x_phone = np.cross(y_phone, z_phone)
        norm_x = float(np.linalg.norm(x_phone))
        if norm_x > 1e-4:
            x_phone /= norm_x

        R = np.vstack([x_phone, y_phone, z_phone]).astype(np.float32)
        if np.all(np.isfinite(R)) and abs(float(np.linalg.det(R)) - 1.0) < 1e-2:
            self.x_phone = x_phone
            self.y_phone = y_phone
            self.R_phone_to_vehicle = R
            self.state = AlignmentState.CALIBRATED
            self.is_calibrated = True

    def _build_interim_rotation(self):
        """Builds a temporary horizontal leveling matrix while waiting for dynamic forward calibration."""
        if self.z_phone is None:
            return
        z_phone = self.z_phone
        ref_vec = np.array([0.0, 1.0, 0.0]) if abs(z_phone[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        x_cand = ref_vec - np.dot(ref_vec, z_phone) * z_phone
        norm_x = float(np.linalg.norm(x_cand))
        if norm_x > 1e-4:
            x_cand /= norm_x
        else:
            x_cand = np.array([1.0, 0.0, 0.0])
        y_cand = np.cross(z_phone, x_cand)
        norm_y = float(np.linalg.norm(y_cand))
        if norm_y > 1e-4:
            y_cand /= norm_y
        R = np.vstack([x_cand, y_cand, z_phone]).astype(np.float32)
        if np.all(np.isfinite(R)):
            self.R_phone_to_vehicle = R

    def estimate_from_stationary_and_motion(
        self,
        stationary_acc: np.ndarray,
        motion_acc: np.ndarray,
        motion_vel: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Batch estimation of attitude from stationary and motion segments."""
        self.reset()
        stat = np.asarray(stationary_acc, dtype=np.float64)
        if stat.ndim == 1:
            stat = stat.reshape(1, -1)
        valid_stat = stat[np.all(np.isfinite(stat), axis=1)]
        if len(valid_stat) == 0:
            valid_stat = np.array([[0.0, 0.0, 9.81]])

        mean_g = np.mean(valid_stat, axis=0)
        norm_g = float(np.linalg.norm(mean_g))
        if not np.isfinite(norm_g) or norm_g < 1e-3:
            self.z_phone = np.array([0.0, 0.0, 1.0])
        else:
            self.z_phone = mean_g / norm_g

        mot = np.asarray(motion_acc, dtype=np.float64)
        if mot.ndim == 1:
            mot = mot.reshape(1, -1)
        valid_mot = mot[np.all(np.isfinite(mot), axis=1)]

        # If motion samples are valid and have sufficient sample count & variance
        if len(valid_mot) >= 2:
            dyn_acc = valid_mot - np.outer(valid_mot @ self.z_phone, self.z_phone)
            if np.all(np.isfinite(dyn_acc)):
                cov = (dyn_acc.T @ dyn_acc) / len(dyn_acc)
                trace_cov = float(np.trace(cov))
                if np.all(np.isfinite(cov)) and trace_cov > 1e-5:
                    try:
                        eigenvalues, eigenvectors = np.linalg.eigh(cov)
                        if np.all(np.isfinite(eigenvalues)) and np.all(np.isfinite(eigenvectors)):
                            idxs = np.argsort(eigenvalues)[::-1]
                            x_cand = eigenvectors[:, idxs[0]]
                            x_cand = x_cand - np.dot(x_cand, self.z_phone) * self.z_phone
                            norm_x = float(np.linalg.norm(x_cand))
                            if norm_x > 1e-4:
                                x_cand /= norm_x
                                # Sign disambiguation if velocity is provided
                                if motion_vel is not None and len(motion_vel) > 1:
                                    valid_vel = np.asarray(motion_vel, dtype=np.float64)
                                    if np.all(np.isfinite(valid_vel)):
                                        dv = valid_vel[-1] - valid_vel[0]
                                        if np.dot(x_cand, dv[:3]) < 0:
                                            x_cand = -x_cand
                                else:
                                    # Sign from mean dynamic acceleration impulse
                                    mean_dyn = np.mean(dyn_acc, axis=0)
                                    if np.dot(mean_dyn, x_cand) < 0 and float(np.linalg.norm(mean_dyn)) > 0.05:
                                        x_cand = -x_cand

                                self._lock_calibration(x_cand)
                                return self.R_phone_to_vehicle
                    except np.linalg.LinAlgError:
                        pass

        # If motion was degenerate, insufficient, or absent:
        # Build safe interim leveling rotation, but do NOT declare calibrated
        self.state = AlignmentState.CALIBRATING_FORWARD if self.z_phone is not None else AlignmentState.NOT_CALIBRATED
        self.is_calibrated = False
        self._build_interim_rotation()
        return self.R_phone_to_vehicle

    def transform_imu(self, acc: np.ndarray, gyro: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Rotate IMU vectors into the vehicle frame."""
        acc_arr = np.asarray(acc, dtype=np.float32)
        gyro_arr = np.asarray(gyro, dtype=np.float32)
        if not np.all(np.isfinite(acc_arr)):
            acc_arr = np.nan_to_num(acc_arr, nan=0.0)
        if not np.all(np.isfinite(gyro_arr)):
            gyro_arr = np.nan_to_num(gyro_arr, nan=0.0)
        acc_v = (self.R_phone_to_vehicle @ acc_arr.T).T
        gyro_v = (self.R_phone_to_vehicle @ gyro_arr.T).T
        return acc_v, gyro_v

