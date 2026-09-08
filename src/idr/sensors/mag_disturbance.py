"""Magnetic Disturbance Detector.

Performs causal multi-criteria magnetic anomaly detection:
1. Expected field norm consistency (|norm - B_ref| <= delta_norm)
2. Short-term temporal field variance stability (var <= var_threshold)
3. Rate of heading change consistency with gyroscope yaw rate (|dpsi/dt - w_z| <= threshold)
"""

from collections import deque
from typing import Dict, Optional, Tuple, Any
import numpy as np


class MagneticDisturbanceDetector:
    """Causally classifies magnetic environment into 'clean', 'suspicious', or 'severe'."""

    def __init__(
        self,
        ref_field_uT: float = 45.0,
        norm_tolerance_uT: float = 12.0,
        max_variance_uT2: float = 25.0,
        max_rate_discrepancy_deg_s: float = 45.0,
        window_size: int = 10,
    ):
        self.ref_field_uT = ref_field_uT
        self.norm_tolerance_uT = norm_tolerance_uT
        self.max_variance_uT2 = max_variance_uT2
        self.max_rate_discrepancy_deg_s = max_rate_discrepancy_deg_s
        self.window_size = window_size

        self.norm_history: deque = deque(maxlen=window_size)
        self.last_psi_mag: Optional[float] = None
        self.last_t: Optional[float] = None

    def update(
        self,
        mag_vector: np.ndarray,
        psi_mag: float,
        gyro_z_rad_s: float,
        t_sec: float,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Assess magnetic quality of the current sample.
        
        Returns:
            is_clean: True if magnetic update is safe for Kalman fusion.
            diagnostics: Dict containing quality metrics and classification.
        """
        norm = float(np.linalg.norm(mag_vector))
        self.norm_history.append(norm)

        # 1. Norm consistency check
        norm_diff = abs(norm - self.ref_field_uT)
        norm_ok = (norm_diff <= self.norm_tolerance_uT)

        # 2. Short-term variance check
        if len(self.norm_history) >= 5:
            current_var = float(np.var(self.norm_history))
        else:
            current_var = 0.0
        var_ok = (current_var <= self.max_variance_uT2)

        # 3. Rate of change consistency check
        rate_ok = True
        mag_rate_deg_s = 0.0
        gyro_rate_deg_s = np.rad2deg(gyro_z_rad_s)

        if self.last_psi_mag is not None and self.last_t is not None and t_sec > self.last_t:
            dt = t_sec - self.last_t
            if dt > 1e-4 and dt < 1.0:
                dpsi = float(np.arctan2(np.sin(psi_mag - self.last_psi_mag), np.cos(psi_mag - self.last_psi_mag)))
                mag_rate_deg_s = np.rad2deg(dpsi / dt)
                rate_diff = abs(mag_rate_deg_s - gyro_rate_deg_s)
                rate_ok = (rate_diff <= self.max_rate_discrepancy_deg_s)

        self.last_psi_mag = psi_mag
        self.last_t = t_sec

        # Classification
        fail_count = int(not norm_ok) + int(not var_ok) + int(not rate_ok)
        if fail_count == 0:
            level = "clean"
            is_clean = True
            quality = 1.0
        elif fail_count == 1 and norm_diff < self.norm_tolerance_uT * 1.5:
            level = "suspicious"
            is_clean = False
            quality = 0.5
        else:
            level = "severe"
            is_clean = False
            quality = 0.0

        diag = {
            "is_clean": is_clean,
            "level": level,
            "quality_score": quality,
            "norm_uT": norm,
            "norm_diff_uT": norm_diff,
            "variance_uT2": current_var,
            "mag_rate_deg_s": mag_rate_deg_s,
            "gyro_rate_deg_s": gyro_rate_deg_s,
        }
        return is_clean, diag
