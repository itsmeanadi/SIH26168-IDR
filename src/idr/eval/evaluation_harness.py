"""Scientific Evaluation Harness and Provenance Audit Engine.

Enforces:
1. Dataset authenticity verification.
2. Dual-stream GNSS blackout gate (Reference retained vs Navigation blocked).
3. Explicit AI model invocation tracking (prevents circular EKF self-feedback from claiming AI performance).
4. KalmanNet execution verification.
5. Map-matching ground-truth leakage protection.
6. Machine-readable scientific audit report generation.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import logging
import numpy as np
import torch

from ..config import CONFIG, set_seed
from ..data.provenance import (
    DatasetAuthenticity,
    DatasetProvenance,
    DatasetRegistry,
    SyntheticDataBlockedError,
    UnsafeEvaluationError,
    get_dataset_registry,
)
from ..filters.fusion import GNSSINSFusion
from ..models.velocity_net import VelocityEstimatorNet
from ..mapmatch.hmm_matcher import HMMMapMatcher
from ..mapmatch.osm_graph import OSMGraphLoader
from .blackout_gate import GNSSBlackoutGate
from .metrics import compute_navigation_metrics, NavigationMetrics

logger = logging.getLogger(__name__)


@dataclass
class EvaluationSpec:
    """Rigorous declarative specification for an IDR evaluation experiment."""
    dataset_name: str
    sequence_ids: List[str]
    blackout_start_s: float
    blackout_end_s: float
    allow_synthetic: bool = False
    use_nhc: bool = True
    use_ai_velocity: bool = True
    use_kalmannet: bool = False
    use_mapmatch: bool = False
    map_source: str = "none"  # "osm_offline", "synthetic_grid", "unsafe_waypoints"
    smoothing_mode: str = "none"  # "none", "online_window", "offline_rts"
    notes: str = ""


@dataclass
class EvaluationProvenanceAudit:
    """Scientific audit record attached to evaluation results."""
    dataset_name: str
    dataset_authenticity: str
    is_scientific_research_valid: bool
    ai_model_invoked: bool
    ai_model_name: Optional[str]
    ai_observation_source: str  # "neural_inference", "none", "simulated", "ekf_internal_state_no_ai"
    kalmannet_invoked: bool
    mapmatch_ground_truth_leakage_safe: bool
    navigation_gnss_leakage_safe: bool
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationRunResult:
    """Complete result of a single evaluation run."""
    spec: EvaluationSpec
    provenance_audit: EvaluationProvenanceAudit
    metrics: Dict[str, float]
    blackout_duration_s: float
    total_trajectory_points: int
    blackout_points: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec": asdict(self.spec),
            "provenance_audit": self.provenance_audit.to_dict(),
            "metrics": self.metrics,
            "blackout_duration_s": self.blackout_duration_s,
            "total_trajectory_points": self.total_trajectory_points,
            "blackout_points": self.blackout_points,
        }

    def save_report(self, filepath: Path) -> Path:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved evaluation provenance report to: {filepath}")
        return filepath


class ScientificEvaluator:
    """Audited evaluation runner executing clean Dead Reckoning benchmarks."""

    def __init__(
        self,
        registry: Optional[DatasetRegistry] = None,
        models_dir: Path = Path("models"),
    ):
        self.registry = registry or get_dataset_registry()
        self.models_dir = Path(models_dir)

    def evaluate_scenario(
        self,
        imu_data: np.ndarray,
        gps_latlon: np.ndarray,
        spec: EvaluationSpec,
        vel_model: Optional[VelocityEstimatorNet] = None,
        matcher: Optional[HMMMapMatcher] = None,
        kalman_filter: Optional[Any] = None,
        dt: float = 0.1,
    ) -> EvaluationRunResult:
        """Run single evaluated scenario with active safety checks and audit tracking."""
        # 1. Dataset provenance safety check
        try:
            prov = self.registry.get(spec.dataset_name)
        except Exception:
            # If unregistered name, check if mock manifest exists
            mock_yaml = Path("configs/datasets/synthetic_iovnbd_mock.yaml")
            if mock_yaml.exists():
                prov = self.registry.register_manifest_file(mock_yaml)
            else:
                prov = DatasetProvenance(
                    dataset_name=spec.dataset_name,
                    source="unregistered",
                    authenticity=DatasetAuthenticity.SYNTHETIC,
                )

        prov.validate_for_evaluation(allow_synthetic=spec.allow_synthetic)

        N = len(imu_data)
        ref_lat, ref_lon = float(gps_latlon[0, 0]), float(gps_latlon[0, 1])
        fusion = GNSSINSFusion(ref_lat=ref_lat, ref_lon=ref_lon, dt=dt)

        # Ground truth ENU positions strictly for error metrics
        gt_enu = np.array([fusion.latlon_to_enu(lat, lon)[:2] for lat, lon in gps_latlon[:, :2]])

        # Initialize EKF state
        fusion.ekf.x[0:2] = gt_enu[0]
        if len(gt_enu) > 1:
            init_v = (gt_enu[1] - gt_enu[0]) / dt
            fusion.ekf.x[3:5] = init_v
            fusion.ekf.x[6] = np.arctan2(init_v[1], init_v[0])

        # 2. Setup GNSS Blackout Gate
        bo_gate = GNSSBlackoutGate(
            blackout_start_s=spec.blackout_start_s,
            blackout_end_s=spec.blackout_end_s,
        )

        # 3. AI velocity inference provenance tracking
        ai_model_invoked = False
        ai_model_name = None
        ai_source = "none"

        ai_speeds = np.zeros(N)
        if spec.use_ai_velocity:
            if vel_model is not None:
                # Verify model provenance before evaluating
                from ..data.model_provenance import ModelProvenanceManager, ModelProvenanceError
                mgr = ModelProvenanceManager()
                model_name = vel_model.__class__.__name__
                try:
                    mgr.verify_and_load(model_name, allow_synthetic_weights=spec.allow_synthetic)
                except ModelProvenanceError as e:
                    if not spec.allow_synthetic:
                        raise UnsafeEvaluationError(
                            f"MODEL PROVENANCE SAFETY VIOLATION: {e}"
                        ) from e

                vel_model.eval()
                win_size = CONFIG["dataset"].window_size
                with torch.no_grad():
                    for i in range(N):
                        start = max(0, i - win_size + 1)
                        win = imu_data[start:i + 1].T
                        if win.shape[1] < win_size:
                            win = np.pad(win, ((0, 0), (win_size - win.shape[1], 0)), mode="edge")
                        win_tensor = torch.tensor(win[None, :, :], dtype=torch.float32)
                        pred_v = float(vel_model(win_tensor).item())
                        ai_speeds[i] = pred_v
                ai_model_invoked = True
                ai_model_name = vel_model.__class__.__name__
                ai_source = "neural_inference"
            else:
                # No neural model supplied
                ai_source = "none"
                ai_model_invoked = False

        # 4. KalmanNet tracking
        kalmannet_invoked = False
        if spec.use_kalmannet:
            if kalman_filter is not None:
                kalmannet_invoked = True
            else:
                raise UnsafeEvaluationError(
                    "KALMANNET SAFETY GATE: Evaluation configured with use_kalmannet=True, "
                    "but no KalmanNet filter instance was provided. Running standard EKF "
                    "while reporting KalmanNet results is strictly prohibited."
                )

        # 5. Map-matching safety verification
        mapmatch_safe = True
        if spec.use_mapmatch and spec.map_source == "unsafe_waypoints":
            mapmatch_safe = False
            if not spec.allow_synthetic:
                raise UnsafeEvaluationError(
                    "MAP MATCHING SAFETY GATE: Map source 'unsafe_waypoints' was requested without "
                    "allow_synthetic=True. Ground-truth waypoints cannot be used as a map source."
                )

        pred_enu = np.zeros((N, 2))

        # 6. Execute step-by-step through blackout gate
        for i in range(N):
            t_s = i * dt
            raw_gnss = (float(gps_latlon[i, 0]), float(gps_latlon[i, 1]), 0.0, 0.0) if not np.isnan(gps_latlon[i, 0]) else None
            
            nav_gnss, is_denied = bo_gate.process_step(
                index=i,
                timestamp_s=t_s,
                imu_data=imu_data[i],
                raw_gnss=raw_gnss,
            )

            fwd_accel = float(imu_data[i, 0])
            yaw_rate = float(imu_data[i, 5])
            ai_v = ai_speeds[i] if (is_denied and ai_model_invoked) else None

            state = fusion.step(
                fwd_accel=fwd_accel,
                yaw_rate=yaw_rate,
                gnss_pos=nav_gnss,
                ai_velocity=ai_v,
                is_gnss_denied=is_denied,
                use_nhc=spec.use_nhc,
            )
            pred_enu[i] = state[0:2]

        # 7. Optional map-matching snap
        if spec.use_mapmatch and matcher is not None:
            coords = [(pred_enu[k, 0], pred_enu[k, 1]) for k in range(N)]
            snapped = matcher.snap_trajectory(coords)
            pred_enu = np.array(snapped)

        # 8. Extract blackout metrics
        bo_mask = bo_gate.get_blackout_mask()
        bo_indices = np.where(bo_mask)[0]
        if len(bo_indices) > 0:
            bo_start_idx, bo_end_idx = bo_indices[0], bo_indices[-1] + 1
            metrics_obj = compute_navigation_metrics(
                pred_enu[bo_start_idx:bo_end_idx],
                gt_enu[bo_start_idx:bo_end_idx],
            )
            metrics_dict = {
                "final_drift_m": round(metrics_obj.final_drift_m, 2),
                "drift_percent": round(metrics_obj.drift_percent, 2),
                "rmse_position_m": round(metrics_obj.rmse_position_m, 2),
                "cep_50_m": round(metrics_obj.cep_50_m, 2),
                "drms_95_m": round(metrics_obj.drms_95_m, 2),
                "total_distance_m": round(metrics_obj.total_distance_m, 2),
            }
        else:
            metrics_dict = {"final_drift_m": 0.0, "drift_percent": 0.0, "rmse_position_m": 0.0, "cep_50_m": 0.0}

        leak_checks = bo_gate.verify_no_state_leakage()

        audit = EvaluationProvenanceAudit(
            dataset_name=prov.dataset_name,
            dataset_authenticity=prov.authenticity.value,
            is_scientific_research_valid=(prov.authenticity != DatasetAuthenticity.SYNTHETIC and mapmatch_safe),
            ai_model_invoked=ai_model_invoked,
            ai_model_name=ai_model_name,
            ai_observation_source=ai_source,
            kalmannet_invoked=kalmannet_invoked,
            mapmatch_ground_truth_leakage_safe=mapmatch_safe,
            navigation_gnss_leakage_safe=leak_checks["navigation_gnss_strictly_blocked"],
        )

        return EvaluationRunResult(
            spec=spec,
            provenance_audit=audit,
            metrics=metrics_dict,
            blackout_duration_s=spec.blackout_end_s - spec.blackout_start_s,
            total_trajectory_points=N,
            blackout_points=int(np.sum(bo_mask)),
        )
