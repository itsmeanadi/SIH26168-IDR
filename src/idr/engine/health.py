"""Navigation Health & Environment Diagnostics Engine (USP 2).

Aggregates real-time health telemetry across all layers:
- GNSS quality & trust status
- IMU noise, Allan-variance proxy, stationary calibration
- Navigation mode state machine
- Position & heading covariance (1-sigma / 2-sigma uncertainty)
- Phone-to-vehicle dynamic alignment metrics
- Two-Wheeler lean angle & dynamics
"""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np


class NavigationMode(str, Enum):
    GNSS_INS_FULL = "GNSS_INS_FULL"
    GNSS_DEGRADED = "GNSS_DEGRADED"
    DEAD_RECKONING_PURE = "DEAD_RECKONING_PURE"
    DEAD_RECKONING_NHC_AI = "DEAD_RECKONING_NHC_AI"
    REACQUISITION_SMOOTHING = "REACQUISITION_SMOOTHING"
    STATIONARY_ZUPT = "STATIONARY_ZUPT"


@dataclass
class HealthDiagnostics:
    nav_mode: NavigationMode
    gnss_trust_score: float         # 0.0 to 1.0
    gnss_status: str                # TRUSTED, DEGRADED, BLACKOUT, REJECTED_SPOOFED
    pos_uncertainty_1sigma_m: float # sqrt(P[0,0] + P[1,1])
    heading_uncertainty_deg: float  # rad2deg(sqrt(P[6,6]))
    imu_acc_noise_mps2: float       # std dev of recent acc
    imu_gyro_noise_deg_s: float     # std dev of recent gyro
    is_stationary: bool
    is_phone_calibrated: bool
    vehicle_type: str               # "two_wheeler" | "car"
    lean_angle_deg: float           # Motorcycle roll lean angle
    ai_speed_mps: float
    ekf_forward_speed_mps: float
    total_dr_distance_m: float
    blackout_elapsed_sec: float
    reacquisition_blend_progress: float  # 0.0 to 1.0


class HealthDiagnosticEngine:
    """Computes real-time health telemetry for the live diagnostic dashboard."""

    def __init__(self, buffer_size: int = 50):
        self.buffer_size = buffer_size
        self.acc_mag_buffer: List[float] = []
        self.gyro_mag_buffer: List[float] = []

    def update_sensor_stats(self, acc_3d: np.ndarray, gyro_3d: np.ndarray):
        acc_mag = float(np.linalg.norm(acc_3d))
        gyro_mag = float(np.rad2deg(np.linalg.norm(gyro_3d)))

        self.acc_mag_buffer.append(acc_mag)
        self.gyro_mag_buffer.append(gyro_mag)

        if len(self.acc_mag_buffer) > self.buffer_size:
            self.acc_mag_buffer.pop(0)
            self.gyro_mag_buffer.pop(0)

    def compute_diagnostics(
        self,
        nav_mode: NavigationMode,
        gnss_trust_score: float,
        gnss_status: str,
        ekf_covariance: np.ndarray,
        is_stationary: bool,
        is_phone_calibrated: bool,
        vehicle_type: str,
        lean_angle_rad: float,
        ai_speed_mps: float,
        ekf_forward_speed_mps: float,
        total_dr_distance_m: float,
        blackout_elapsed_sec: float,
        reacquisition_blend_progress: float = 1.0,
    ) -> HealthDiagnostics:
        """Compute the full diagnostic state vector."""
        # 1-sigma horizontal position uncertainty
        p_east = max(0.0, float(ekf_covariance[0, 0]))
        p_north = max(0.0, float(ekf_covariance[1, 1]))
        pos_uncertainty = float(np.sqrt(p_east + p_north))

        # Heading uncertainty in degrees
        p_yaw = max(0.0, float(ekf_covariance[6, 6]))
        heading_uncertainty = float(np.rad2deg(np.sqrt(p_yaw)))

        # IMU noise estimation (Allan variance proxy)
        acc_noise = float(np.std(self.acc_mag_buffer)) if len(self.acc_mag_buffer) > 5 else 0.05
        gyro_noise = float(np.std(self.gyro_mag_buffer)) if len(self.gyro_mag_buffer) > 5 else 0.1

        return HealthDiagnostics(
            nav_mode=nav_mode,
            gnss_trust_score=round(float(gnss_trust_score), 2),
            gnss_status=str(gnss_status),
            pos_uncertainty_1sigma_m=round(pos_uncertainty, 2),
            heading_uncertainty_deg=round(heading_uncertainty, 2),
            imu_acc_noise_mps2=round(acc_noise, 3),
            imu_gyro_noise_deg_s=round(gyro_noise, 3),
            is_stationary=bool(is_stationary),
            is_phone_calibrated=bool(is_phone_calibrated),
            vehicle_type=str(vehicle_type),
            lean_angle_deg=round(float(np.rad2deg(lean_angle_rad)), 1),
            ai_speed_mps=round(float(ai_speed_mps), 2),
            ekf_forward_speed_mps=round(float(ekf_forward_speed_mps), 2),
            total_dr_distance_m=round(float(total_dr_distance_m), 1),
            blackout_elapsed_sec=round(float(blackout_elapsed_sec), 1),
            reacquisition_blend_progress=round(float(reacquisition_blend_progress), 2),
        )
