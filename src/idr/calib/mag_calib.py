"""Magnetometer Calibration Module.

Provides hard-iron offset correction, soft-iron scaling, and phone-to-vehicle alignment
for 3D magnetometer sensor streams in physical microTesla (uT) units.
"""

from typing import Optional, Tuple
import numpy as np


class MagnetometerCalibrator:
    """Estimates and applies hard-iron and soft-iron calibration to 3D magnetometer data."""

    def __init__(
        self,
        hard_iron_bias: Optional[np.ndarray] = None,
        scale_factors: Optional[np.ndarray] = None,
        R_p2v: Optional[np.ndarray] = None,
    ):
        self.hard_iron_bias = hard_iron_bias if hard_iron_bias is not None else np.zeros(3, dtype=np.float64)
        self.scale_factors = scale_factors if scale_factors is not None else np.ones(3, dtype=np.float64)
        self.R_p2v = R_p2v if R_p2v is not None else np.eye(3, dtype=np.float64)
        self.is_calibrated = False

    def fit_min_max_sphere(self, mag_raw: np.ndarray) -> None:
        """Estimate hard-iron bias and soft-iron scale from multi-orientation min-max envelope.
        
        Args:
            mag_raw: (N, 3) raw 3D magnetometer readings in uT.
        """
        if len(mag_raw) < 50:
            return

        min_vals = np.min(mag_raw, axis=0)
        max_vals = np.max(mag_raw, axis=0)

        # Hard-iron offset: center of the bounding ellipsoid
        self.hard_iron_bias = (max_vals + min_vals) / 2.0

        # Soft-iron scale: semi-axis lengths normalized to average radius
        semi_axes = (max_vals - min_vals) / 2.0
        semi_axes = np.where(semi_axes < 1.0, 1.0, semi_axes)  # Avoid division by zero
        avg_radius = np.mean(semi_axes)
        self.scale_factors = avg_radius / semi_axes

        self.is_calibrated = True

    def set_phone_to_vehicle_alignment(self, R_p2v: np.ndarray) -> None:
        """Set the 3x3 rotation matrix mapping Phone frame -> Vehicle Body frame."""
        self.R_p2v = np.array(R_p2v, dtype=np.float64)

    def transform_and_calibrate(self, mag_raw: np.ndarray) -> np.ndarray:
        """Calibrate and rotate magnetometer readings into vehicle body frame.
        
        Args:
            mag_raw: (N, 3) or (3,) raw phone magnetometer readings in uT.
            
        Returns:
            mag_vehicle: Calibrated and rotated magnetic field vector in vehicle frame.
        """
        raw = np.atleast_2d(np.array(mag_raw, dtype=np.float64))

        # 1. Hard-iron removal
        centered = raw - self.hard_iron_bias

        # 2. Soft-iron scale correction
        calibrated_phone = centered * self.scale_factors

        # 3. Rotate to Vehicle Body frame: m_v = R_p2v @ m_p
        mag_vehicle = (self.R_p2v @ calibrated_phone.T).T

        if np.ndim(mag_raw) == 1:
            return mag_vehicle[0]
        return mag_vehicle
