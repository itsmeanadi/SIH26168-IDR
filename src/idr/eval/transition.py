"""GNSS-INS Transition Smoothing and Discontinuity Mitigation.

Implements exponential blend filtering during GNSS reacquisition to eliminate
position jumps (discrete leaps) when leaving a tunnel or underground structure.
"""

from typing import List, Tuple
import numpy as np


class ReacquisitionSmoother:
    """Blends dead-reckoned trajectory smoothly toward reacquired GNSS fixes."""

    def __init__(self, blend_duration_sec: float = 1.5, dt: float = 0.1):
        self.blend_duration_sec = blend_duration_sec
        self.dt = dt
        self.total_blend_steps = max(1, int(blend_duration_sec / dt))
        self.blend_counter = self.total_blend_steps
        self.offset = np.zeros(2, dtype=np.float64)

    def trigger_reacquisition(self, current_dr_enu: np.ndarray, gnss_enu: np.ndarray) -> float:
        """Trigger transition smoothing upon receiving first valid fix after blackout.
        
        Returns:
            jump_magnitude_m: Raw discontinuity magnitude before smoothing.
        """
        self.offset = current_dr_enu[:2] - gnss_enu[:2]
        jump_mag = float(np.linalg.norm(self.offset))
        self.blend_counter = 0
        return jump_mag

    def apply_smoothing(self, current_dr_enu: np.ndarray, raw_gnss_enu: np.ndarray) -> np.ndarray:
        """Blend current position toward GNSS fix.
        
        Args:
            current_dr_enu: (2,) current dead reckoning output
            raw_gnss_enu: (2,) new GNSS position
        Returns:
            smoothed_enu: (2,) smoothly continuous output
        """
        if self.blend_counter >= self.total_blend_steps:
            return raw_gnss_enu[:2].copy()

        # Fraction of transition completed [0.0 -> 1.0]
        progress = (self.blend_counter + 1) / self.total_blend_steps
        # Smooth cosine bell weighting (C1 continuous): 0 at t=0, 1 at t=T
        weight_gnss = 0.5 * (1.0 - np.cos(np.pi * progress))

        smoothed = (1.0 - weight_gnss) * current_dr_enu[:2] + weight_gnss * raw_gnss_enu[:2]
        self.blend_counter += 1
        return smoothed
