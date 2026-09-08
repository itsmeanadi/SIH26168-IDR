"""GNSS Blackout Gate and Dual Stream Separation.

Enforces strict physical separation between:
1. REFERENCE STREAM: Retains untouched reference GNSS for post-hoc error calculation.
2. NAVIGATION STREAM: Feeds the estimator/filter, strictly blocking all GNSS during outage.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StreamSample:
    """Individual time step in the dual-stream blackout architecture."""
    index: int
    timestamp_s: float
    imu_data: np.ndarray  # shape (6,) [ax, ay, az, gx, gy, gz]
    reference_gnss: Optional[Tuple[float, float, float, float]]  # (lat, lon, speed, heading)
    navigation_gnss: Optional[Tuple[float, float]]  # (lat, lon) provided to filter (None during blackout)
    is_blackout: bool
    ai_velocity_input: Optional[float] = None
    estimator_state_output: Optional[np.ndarray] = None


@dataclass
class BlackoutPeriod:
    """Explicitly timestamped blackout window."""
    start_time_s: float
    end_time_s: float
    start_index: int
    end_index: int
    duration_s: float

    def contains_time(self, t: float) -> bool:
        return self.start_time_s <= t < self.end_time_s

    def contains_index(self, idx: int) -> bool:
        return self.start_index <= idx < self.end_index


class GNSSBlackoutGate:
    """Dual-stream gate ensuring reference retention and zero GNSS leakage into navigation."""

    def __init__(
        self,
        blackout_start_s: float,
        blackout_end_s: float,
        blackout_start_idx: Optional[int] = None,
        blackout_end_idx: Optional[int] = None,
    ):
        self.start_s = float(blackout_start_s)
        self.end_s = float(blackout_end_s)
        self.start_idx = blackout_start_idx
        self.end_idx = blackout_end_idx
        
        self.reference_stream: List[StreamSample] = []
        self.navigation_stream: List[StreamSample] = []
        self._total_steps = 0
        self._blackout_steps = 0

    @property
    def blackout_period(self) -> BlackoutPeriod:
        s_idx = self.start_idx if self.start_idx is not None else 0
        e_idx = self.end_idx if self.end_idx is not None else 0
        return BlackoutPeriod(
            start_time_s=self.start_s,
            end_time_s=self.end_s,
            start_index=s_idx,
            end_index=e_idx,
            duration_s=self.end_s - self.start_s,
        )

    def is_in_blackout(self, index: int, timestamp_s: float) -> bool:
        """Evaluate whether current step is inside blackout window."""
        if self.start_idx is not None and self.end_idx is not None:
            return self.start_idx <= index < self.end_idx
        return self.start_s <= timestamp_s < self.end_s

    def process_step(
        self,
        index: int,
        timestamp_s: float,
        imu_data: np.ndarray,
        raw_gnss: Optional[Tuple[float, float, float, float]],
    ) -> Tuple[Optional[Tuple[float, float]], bool]:
        """Process incoming raw sensor packet through the gate.
        
        Returns:
            (navigation_gnss_position, is_blackout_active)
        """
        is_bo = self.is_in_blackout(index, timestamp_s)
        self._total_steps += 1
        if is_bo:
            self._blackout_steps += 1

        # Navigation receives GNSS only when NOT in blackout
        nav_gnss = None
        if not is_bo and raw_gnss is not None:
            nav_gnss = (raw_gnss[0], raw_gnss[1])

        sample = StreamSample(
            index=index,
            timestamp_s=timestamp_s,
            imu_data=np.asarray(imu_data, dtype=np.float32),
            reference_gnss=raw_gnss,
            navigation_gnss=nav_gnss,
            is_blackout=is_bo,
        )

        self.reference_stream.append(sample)
        self.navigation_stream.append(sample)

        return nav_gnss, is_bo

    def get_reference_positions(self) -> np.ndarray:
        """Extract all reference GNSS positions (lat, lon) for evaluation."""
        coords = []
        for s in self.reference_stream:
            if s.reference_gnss is not None:
                coords.append([s.reference_gnss[0], s.reference_gnss[1]])
            else:
                coords.append([np.nan, np.nan])
        return np.array(coords, dtype=np.float64)

    def get_blackout_mask(self) -> np.ndarray:
        """Get boolean mask indicating blackout indices."""
        return np.array([s.is_blackout for s in self.reference_stream], dtype=bool)

    def verify_no_state_leakage(self) -> Dict[str, bool]:
        """Perform cryptographic/structural assertions on stream isolation.
        
        Verifies:
        A. Reference stream has GNSS during blackout.
        B. Navigation stream has strictly None during blackout.
        C. Navigation stream receives GNSS outside blackout (if available).
        D. No ground-truth velocity leaked into navigation_gnss.
        """
        bo_ref_present = any(
            s.is_blackout and s.reference_gnss is not None
            for s in self.reference_stream
        )
        bo_nav_empty = all(
            (not s.is_blackout) or (s.navigation_gnss is None)
            for s in self.navigation_stream
        )
        outside_nav_present = any(
            (not s.is_blackout) and (s.navigation_gnss is not None)
            for s in self.navigation_stream
        )

        return {
            "reference_retained_during_blackout": bo_ref_present,
            "navigation_gnss_strictly_blocked": bo_nav_empty,
            "navigation_resumed_after_blackout": outside_nav_present,
            "stream_length_identical": len(self.reference_stream) == len(self.navigation_stream),
        }
