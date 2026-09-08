"""Dead reckoning navigation accuracy and drift metrics."""

from dataclasses import dataclass
from typing import Dict
import numpy as np

@dataclass
class NavigationMetrics:
    total_distance_m: float
    final_drift_m: float
    drift_percent: float  # (final_drift / total_distance) * 100
    rmse_position_m: float
    cep_50_m: float
    drms_95_m: float
    mean_speed_m_s: float
    mae_position_m: float = 0.0
    max_error_m: float = 0.0

def compute_navigation_metrics(
    pred_enu: np.ndarray,
    gt_enu: np.ndarray,
) -> NavigationMetrics:
    """Compute standard GNSS/INS accuracy figures.
    
    Args:
        pred_enu: (N, 2) or (N, 3) predicted positions [East, North]
        gt_enu:   (N, 2) or (N, 3) ground-truth positions [East, North]
    """
    diff = pred_enu[:, :2] - gt_enu[:, :2]
    horizontal_errors = np.linalg.norm(diff, axis=1)

    # Incremental distance travelled along ground truth
    gt_steps = np.diff(gt_enu[:, :2], axis=0)
    step_dists = np.linalg.norm(gt_steps, axis=1)
    total_dist = float(np.sum(step_dists))

    final_drift = float(horizontal_errors[-1]) if len(horizontal_errors) > 0 else 0.0
    drift_pct = (final_drift / total_dist * 100.0) if total_dist > 0 else 0.0

    rmse = float(np.sqrt(np.mean(horizontal_errors**2))) if len(horizontal_errors) > 0 else 0.0
    mae = float(np.mean(horizontal_errors)) if len(horizontal_errors) > 0 else 0.0
    max_err = float(np.max(horizontal_errors)) if len(horizontal_errors) > 0 else 0.0
    cep_50 = float(np.percentile(horizontal_errors, 50)) if len(horizontal_errors) > 0 else 0.0
    drms_95 = float(np.percentile(horizontal_errors, 95)) if len(horizontal_errors) > 0 else 0.0

    return NavigationMetrics(
        total_distance_m=total_dist,
        final_drift_m=final_drift,
        drift_percent=drift_pct,
        rmse_position_m=rmse,
        cep_50_m=cep_50,
        drms_95_m=drms_95,
        mean_speed_m_s=total_dist / (len(pred_enu) * 0.1) if len(pred_enu) > 0 else 0.0,
        mae_position_m=mae,
        max_error_m=max_err,
    )
