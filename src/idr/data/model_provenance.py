"""Model Provenance and Authenticity Classification System.

Tracks, audits, and enforces the distinction between:
- HISTORICAL SYNTHETIC BASELINE CHECKPOINTS (mock / simulation trained)
- AUTHENTIC RESEARCH CHECKPOINTS (trained on genuine IO-VNBD datasets)
- FIELD-TRAINED / FINE-TUNED CHECKPOINTS
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import logging

logger = logging.getLogger(__name__)


class ModelWeightsAuthenticity(str, Enum):
    HISTORICAL_SYNTHETIC = "historical_synthetic"
    AUTHENTIC_RESEARCH = "authentic_research"
    FIELD_CALIBRATED = "field_calibrated"


class ModelProvenanceError(Exception):
    """Raised when unvalidated or synthetic model weights are loaded in scientific workflows."""
    pass


@dataclass
class ModelProvenanceManifest:
    model_name: str
    checkpoint_file: str
    architecture: str
    authenticity: ModelWeightsAuthenticity
    is_scientific_research_valid: bool
    training_dataset_name: str
    training_drives: List[str]
    input_channels: int
    window_size: int
    target_description: str
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["authenticity"] = self.authenticity.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelProvenanceManifest:
        d = dict(data)
        if "authenticity" in d and isinstance(d["authenticity"], str):
            d["authenticity"] = ModelWeightsAuthenticity(d["authenticity"])
        return cls(**d)


class ModelProvenanceManager:
    """Manages model checkpoint metadata and verifies scientific authenticity before loading."""

    def __init__(self, manifest_path: Optional[Path] = None):
        self.manifest_path = Path(manifest_path) if manifest_path else Path("models/model_manifest.json")
        self._manifests: Dict[str, ModelProvenanceManifest] = {}
        self._load_or_init_default()

    def _load_or_init_default(self):
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item in data.get("models", []):
                    m = ModelProvenanceManifest.from_dict(item)
                    self._manifests[m.model_name.lower()] = m
                return
            except Exception as e:
                logger.warning(f"Failed to read model manifest from {self.manifest_path}: {e}")

        # Default registered checkpoints currently on disk (all marked historical synthetic)
        self._manifests = {
            "velocityestimatornet": ModelProvenanceManifest(
                model_name="VelocityEstimatorNet",
                checkpoint_file="models/velocity_net.pt",
                architecture="1D-CNN + GRU",
                authenticity=ModelWeightsAuthenticity.HISTORICAL_SYNTHETIC,
                is_scientific_research_valid=False,
                training_dataset_name="IO-VNBD (Synthetic Mock Baseline)",
                training_drives=["M", "S", "Vta", "Vtb", "Vw"],
                input_channels=6,
                window_size=50,
                target_description="Vehicle forward speed (m/s)",
                notes="Historical synthetic baseline. Must be retrained on authentic IO-VNBD for scientific claims.",
            ),
            "imudenoisenet": ModelProvenanceManifest(
                model_name="IMUDenoiseNet",
                checkpoint_file="models/imu_denoise_net.pt",
                architecture="1D Dilated Residual CNN",
                authenticity=ModelWeightsAuthenticity.HISTORICAL_SYNTHETIC,
                is_scientific_research_valid=False,
                training_dataset_name="IO-VNBD (Synthetic Mock Baseline)",
                training_drives=["M", "S", "Vta", "Vtb", "Vw"],
                input_channels=6,
                window_size=50,
                target_description="IMU noise/bias residual offset (m/s^2, rad/s)",
                notes="Experimental prototype. Requires authentic 6-DOF reference IMU for supervised training.",
            ),
            "kalmannetgainestimator": ModelProvenanceManifest(
                model_name="KalmanNetGainEstimator",
                checkpoint_file="models/kalmannet.pt",
                architecture="GRU Neural Kalman Gain Estimator",
                authenticity=ModelWeightsAuthenticity.HISTORICAL_SYNTHETIC,
                is_scientific_research_valid=False,
                training_dataset_name="Synthetic EKF Simulation",
                training_drives=["Simulation"],
                input_channels=6,
                window_size=1,
                target_description="Analytical Kalman gain approximation Kt",
                notes="Experimental research prototype.",
            ),
        }

    def save_manifest(self, path: Optional[Path] = None):
        target = Path(path) if path else self.manifest_path
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"models": [m.to_dict() for m in self._manifests.values()]}
        with open(target, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Saved model provenance manifest to {target}")

    def verify_and_load(
        self,
        model_name: str,
        allow_synthetic_weights: bool = False,
    ) -> ModelProvenanceManifest:
        """Verify model authenticity before permitting evaluation."""
        key = model_name.strip().lower()
        if key not in self._manifests:
            raise ModelProvenanceError(f"Model '{model_name}' has no registered provenance manifest.")

        manifest = self._manifests[key]
        if not manifest.is_scientific_research_valid and not allow_synthetic_weights:
            raise ModelProvenanceError(
                f"SCIENTIFIC SAFETY VIOLATION: Model '{manifest.model_name}' has authenticity "
                f"'{manifest.authenticity.value}' and is marked NOT scientifically valid for authentic benchmarking. "
                f"To use historical synthetic weights for pipeline/regression tests, pass allow_synthetic_weights=True."
            )
        return manifest
