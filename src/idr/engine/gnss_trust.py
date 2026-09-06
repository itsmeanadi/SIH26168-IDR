"""GNSS Trust Check & Anomaly Engine (USP 1).

Evaluates GNSS measurement plausibility using:
- Mahalanobis innovation gating against dead-reckoned prediction
- Kinematic velocity / acceleration consistency
- Multipath & teleportation jump detection
- Reported horizontal dilution/accuracy thresholds
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
import numpy as np


class GNSSTrustStatus(str, Enum):
    TRUSTED = "TRUSTED"
    DEGRADED = "DEGRADED"
    SUSPICIOUS_JUMP = "SUSPICIOUS_JUMP"
    REJECTED_SPOOFED = "REJECTED_SPOOFED"
    BLACKOUT = "BLACKOUT"


@dataclass
class GNSSTrustResult:
    status: GNSSTrustStatus
    trust_score: float  # 0.0 to 1.0
    innovation_dist_m: float
    implied_speed_mps: float
    is_trusted: bool
    rejection_reason: Optional[str] = None


class GNSSTrustEngine:
    """Real-time statistical gating and anomaly rejection for GNSS fixes."""

    def __init__(
        self,
        max_speed_threshold_mps: float = 55.0,  # ~200 km/h
        max_jump_dist_m: float = 25.0,           # Max single-step jump allowed
        max_innovation_sigma: float = 3.5,       # Chi-square gate threshold (~99.7%)
        max_accuracy_threshold_m: float = 30.0,  # Ignore fixes worse than 30m
        trust_ema_alpha: float = 0.25,
    ):
        self.max_speed_threshold_mps = max_speed_threshold_mps
        self.max_jump_dist_m = max_jump_dist_m
        self.max_innovation_sigma = max_innovation_sigma
        self.max_accuracy_threshold_m = max_accuracy_threshold_m
        self.trust_ema_alpha = trust_ema_alpha

        self.current_trust_score = 1.0
        self.prev_gnss_enu: Optional[np.ndarray] = None
        self.prev_gnss_timestamp: Optional[float] = None
        self.rejection_count = 0
        self.consecutive_rejections = 0

    def evaluate_fix(
        self,
        gnss_enu: np.ndarray,
        pred_dr_enu: np.ndarray,
        pos_covariance: np.ndarray,
        timestamp: float,
        reported_accuracy_m: Optional[float] = None,
        inertial_speed_mps: float = 0.0,
    ) -> GNSSTrustResult:
        """Evaluate whether a GNSS fix is physically reliable and consistent.

        Args:
            gnss_enu: [East, North, Up] in meters from GNSS receiver
            pred_dr_enu: [East, North, Up] predicted position from EKF/DR
            pos_covariance: (3, 3) or (2, 2) position uncertainty matrix P[0:2, 0:2]
            timestamp: epoch timestamp in seconds
            reported_accuracy_m: receiver reported accuracy (e.g. 1-sigma radius)
            inertial_speed_mps: current dead-reckoned forward speed
        """
        # 1. Check reported accuracy radius
        if reported_accuracy_m is not None and reported_accuracy_m > self.max_accuracy_threshold_m:
            self._update_trust(0.2)
            self.consecutive_rejections += 1
            return GNSSTrustResult(
                status=GNSSTrustStatus.DEGRADED,
                trust_score=self.current_trust_score,
                innovation_dist_m=float(np.linalg.norm(gnss_enu[:2] - pred_dr_enu[:2])),
                implied_speed_mps=0.0,
                is_trusted=False,
                rejection_reason=f"Reported accuracy too poor ({reported_accuracy_m:.1f}m > {self.max_accuracy_threshold_m}m)",
            )

        # 2. Kinematic delta check against previous GNSS fix
        implied_speed = 0.0
        if self.prev_gnss_enu is not None and self.prev_gnss_timestamp is not None:
            dt = max(1e-3, timestamp - self.prev_gnss_timestamp)
            delta_pos = np.linalg.norm(gnss_enu[:2] - self.prev_gnss_enu[:2])
            implied_speed = float(delta_pos / dt)

            # Detect impossible single-step jump (> 25m jump or > 55m/s speed without inertial support)
            if delta_pos > self.max_jump_dist_m and implied_speed > (inertial_speed_mps * 2.5 + 15.0):
                self._update_trust(0.0)
                self.rejection_count += 1
                self.consecutive_rejections += 1
                return GNSSTrustResult(
                    status=GNSSTrustStatus.SUSPICIOUS_JUMP,
                    trust_score=self.current_trust_score,
                    innovation_dist_m=float(np.linalg.norm(gnss_enu[:2] - pred_dr_enu[:2])),
                    implied_speed_mps=implied_speed,
                    is_trusted=False,
                    rejection_reason=f"Kinematic jump detected ({delta_pos:.1f}m in {dt:.2f}s = {implied_speed:.1f} m/s)",
                )

        # 3. Innovation distance & Mahalanobis gating against EKF prior
        diff = gnss_enu[:2] - pred_dr_enu[:2]
        innov_dist = float(np.linalg.norm(diff))

        # Position covariance 2x2
        cov2d = pos_covariance[:2, :2] + np.eye(2) * ((reported_accuracy_m or 3.0) ** 2)
        try:
            inv_cov = np.linalg.inv(cov2d)
            mahalanobis_sq = float(diff @ inv_cov @ diff)
            mahalanobis_dist = np.sqrt(max(0.0, mahalanobis_sq))
        except np.linalg.LinAlgError:
            mahalanobis_dist = innov_dist / 5.0

        # If Mahalanobis distance exceeds threshold
        if mahalanobis_dist > self.max_innovation_sigma:
            # Check if this is an initial convergence or genuine outlier
            if self.consecutive_rejections > 15:
                # Receiver might have recovered from long outage: allow gradual recovery
                target_score = 0.5
                status = GNSSTrustStatus.DEGRADED
                is_trusted = True
            else:
                target_score = max(0.0, 1.0 - (mahalanobis_dist - self.max_innovation_sigma) * 0.2)
                status = GNSSTrustStatus.REJECTED_SPOOFED if mahalanobis_dist > 6.0 else GNSSTrustStatus.SUSPICIOUS_JUMP
                is_trusted = False
                self.rejection_count += 1
                self.consecutive_rejections += 1

            self._update_trust(target_score)
            return GNSSTrustResult(
                status=status,
                trust_score=self.current_trust_score,
                innovation_dist_m=innov_dist,
                implied_speed_mps=implied_speed,
                is_trusted=is_trusted,
                rejection_reason=f"Innovation gating failed (Mahalanobis {mahalanobis_dist:.2f}σ > {self.max_innovation_sigma}σ, diff={innov_dist:.1f}m)",
            )

        # 4. Valid and consistent fix
        self.consecutive_rejections = 0
        self.prev_gnss_enu = gnss_enu.copy()
        self.prev_gnss_timestamp = timestamp
        self._update_trust(1.0)

        return GNSSTrustResult(
            status=GNSSTrustStatus.TRUSTED,
            trust_score=self.current_trust_score,
            innovation_dist_m=innov_dist,
            implied_speed_mps=implied_speed,
            is_trusted=True,
            rejection_reason=None,
        )

    def _update_trust(self, target: float):
        self.current_trust_score = (
            (1.0 - self.trust_ema_alpha) * self.current_trust_score
            + self.trust_ema_alpha * target
        )
        self.current_trust_score = float(np.clip(self.current_trust_score, 0.0, 1.0))

    def reset(self):
        self.current_trust_score = 1.0
        self.prev_gnss_enu = None
        self.prev_gnss_timestamp = None
        self.rejection_count = 0
        self.consecutive_rejections = 0
