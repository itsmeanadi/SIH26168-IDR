"""Timestamp-Aware Anti-Aliased Resampler for AI Velocity Estimation.

Aligns arbitrary high-rate live smartphone IMU streams (e.g. 50–100 Hz) with the
exact 10 Hz (100 ms) temporal sampling semantics and 5.0-second receptive field
expected by VelocityEstimatorNet.

Contract:
- Target Frequency: 10.0 Hz (dt_target = 0.100 s)
- Receptive Field: 50 samples * 0.1 s = 5.0 seconds
- Channel Contract: [fwd_accel, lat_accel, vert_accel, roll_rate, pitch_rate, yaw_rate]
- Anti-Aliasing: Sub-interval boxcar averaging over high-rate intermediate frames
- Jitter & Drop Resilience: Linear interpolation across sensor packet loss / frame drops
"""

from typing import List, Optional
import numpy as np


class TimestampAwareAIResampler:
    """Resamples incoming vehicle-frame IMU frames into uniform 10 Hz representation."""

    def __init__(
        self,
        target_rate_hz: float = 10.0,
        window_size: int = 50,
        max_gap_interpolation_sec: float = 1.0,
    ):
        self.target_rate_hz = float(target_rate_hz)
        self.dt_target = 1.0 / self.target_rate_hz  # 0.1 s
        self.window_size = int(window_size)
        self.max_gap_interpolation_sec = float(max_gap_interpolation_sec)

        # Sliding 10 Hz AI buffer (stores up to window_size samples of shape (6,))
        self.ai_window_buffer: List[np.ndarray] = []

        # Internal accumulation state
        self._accumulated_samples: List[np.ndarray] = []
        self._accumulated_times: List[float] = []
        self._last_emitted_time: Optional[float] = None
        self._last_emitted_sample: Optional[np.ndarray] = None
        self._total_emitted_count: int = 0

    @property
    def is_ready(self) -> bool:
        """True when the sliding buffer contains the full 50 samples (5.0s)."""
        return len(self.ai_window_buffer) >= self.window_size

    def reset(self):
        """Clear all buffers and reset timestamp tracking."""
        self.ai_window_buffer.clear()
        self._accumulated_samples.clear()
        self._accumulated_times.clear()
        self._last_emitted_time = None
        self._last_emitted_sample = None
        self._total_emitted_count = 0

    def add_sample(self, timestamp: float, imu_vehicle_6d: np.ndarray) -> Optional[np.ndarray]:
        """Add a single high-rate vehicle-frame IMU frame.

        Args:
            timestamp: Monotonic sensor timestamp in seconds.
            imu_vehicle_6d: 6D array [fwd_acc, lat_acc, vert_acc, roll_rate, pitch_rate, yaw_rate].

        Returns:
            np.ndarray of shape (6,) if a new 10 Hz sample was emitted, else None.
        """
        sample = np.asarray(imu_vehicle_6d, dtype=np.float32)
        if sample.shape != (6,):
            raise ValueError(f"Expected 6D vehicle IMU sample, got shape {sample.shape}")

        # Sanitize any non-finite values before accumulation
        if not np.all(np.isfinite(sample)):
            sample = np.nan_to_num(sample, nan=0.0, posinf=0.0, neginf=0.0)

        # First sample initialization
        if self._last_emitted_time is None:
            self._last_emitted_time = float(timestamp)
            self._last_emitted_sample = sample.copy()
            self._push_to_buffer(sample)
            return sample

        # Check for non-monotonic / zero time progression
        dt = float(timestamp) - self._last_emitted_time
        if dt <= 0.0:
            # Out-of-order or duplicate timestamp -> accumulate without advancing
            self._accumulated_samples.append(sample)
            self._accumulated_times.append(timestamp)
            return None

        # Check if current time has reached the next 10 Hz boundary (with small tolerance)
        tolerance = self.dt_target * 0.10  # 10ms tolerance for jitter
        if dt < (self.dt_target - tolerance):
            # Still inside the current 100ms sub-interval -> accumulate for anti-aliasing
            self._accumulated_samples.append(sample)
            self._accumulated_times.append(timestamp)
            return None

        # Time to emit one or more 10 Hz samples
        self._accumulated_samples.append(sample)
        self._accumulated_times.append(timestamp)

        # Compute anti-aliased average over the collected high-rate frames
        mean_sample = np.mean(self._accumulated_samples, axis=0).astype(np.float32)

        # Handle dropped packets / large time gaps gracefully
        if dt >= (self.dt_target * 1.8) and dt <= self.max_gap_interpolation_sec:
            # Interpolate missing steps to maintain 5.0s physical time continuity
            num_steps = int(np.round(dt / self.dt_target))
            prev_sample = self._last_emitted_sample if self._last_emitted_sample is not None else mean_sample
            for step_idx in range(1, num_steps):
                alpha = step_idx / float(num_steps)
                interp_sample = (1.0 - alpha) * prev_sample + alpha * mean_sample
                self._push_to_buffer(interp_sample.astype(np.float32))

        # Push the primary anti-aliased sample
        self._push_to_buffer(mean_sample)
        self._last_emitted_time = float(timestamp)
        self._last_emitted_sample = mean_sample.copy()

        # Reset accumulation for next interval
        self._accumulated_samples.clear()
        self._accumulated_times.clear()

        return mean_sample

    def _push_to_buffer(self, sample: np.ndarray):
        """Append sample and maintain fixed window size."""
        self.ai_window_buffer.append(sample)
        if len(self.ai_window_buffer) > self.window_size:
            self.ai_window_buffer.pop(0)
        self._total_emitted_count += 1
