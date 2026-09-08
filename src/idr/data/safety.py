"""Training and Evaluation Safety Gates.

Enforces strict provenance verification, blocking accidental training or scientific
claims derived from synthetic / mock data unless explicitly permitted for software tests.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import logging
from .provenance import (
    DatasetAuthenticity,
    DatasetProvenance,
    DatasetRegistry,
    SyntheticDataBlockedError,
    DatasetNotFoundError,
    get_dataset_registry,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingProvenanceRecord:
    """Audit record written alongside model weights upon training completion."""
    model_name: str
    dataset_name: str
    authenticity: str
    allow_synthetic_flag: bool
    train_drives: List[str]
    val_drives: List[str]
    num_train_samples: int
    num_val_samples: int
    epochs_trained: int
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    git_commit: Optional[str] = None
    notes: str = ""
    is_scientific_research_valid: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "authenticity": self.authenticity,
            "is_scientific_research_valid": self.is_scientific_research_valid,
            "allow_synthetic_flag": self.allow_synthetic_flag,
            "train_drives": self.train_drives,
            "val_drives": self.val_drives,
            "num_train_samples": self.num_train_samples,
            "num_val_samples": self.num_val_samples,
            "epochs_trained": self.epochs_trained,
            "timestamp_utc": self.timestamp_utc,
            "git_commit": self.git_commit,
            "notes": self.notes,
        }

    def save(self, filepath: Path) -> Path:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved training provenance audit record to: {filepath}")
        return filepath


class TrainingSafetyGate:
    """Guards training workflows against unverified, synthetic, or unmanifested data."""

    @classmethod
    def enforce(
        cls,
        dataset_name: Optional[str] = None,
        manifest_path: Optional[Path] = None,
        allow_synthetic: bool = False,
        registry: Optional[DatasetRegistry] = None,
    ) -> DatasetProvenance:
        """Verify that training dataset satisfies authenticity constraints.
        
        Raises:
            SyntheticDataBlockedError: If dataset is synthetic and allow_synthetic is False.
            DatasetNotFoundError: If dataset is not found.
        """
        if registry is None:
            registry = get_dataset_registry()
            # Scan configs/datasets directory if present
            default_manifest_dir = Path("configs/datasets")
            if default_manifest_dir.exists():
                registry.scan_directory(default_manifest_dir)

        if manifest_path is not None:
            prov = DatasetProvenance.from_file(Path(manifest_path))
            registry.register(prov)
            dataset_name = prov.dataset_name

        if dataset_name is None:
            dataset_name = "synthetic-iovnbd-mock"

        try:
            prov = registry.get(dataset_name)
        except DatasetNotFoundError:
            # Fallback check if default synthetic manifest exists
            mock_yaml = Path("configs/datasets/synthetic_iovnbd_mock.yaml")
            if mock_yaml.exists():
                prov = registry.register_manifest_file(mock_yaml)
            else:
                raise DatasetNotFoundError(
                    f"No registered dataset or manifest found for '{dataset_name}'."
                )

        # Enforce safety check
        if prov.is_synthetic() and not allow_synthetic:
            raise SyntheticDataBlockedError(
                f"\n{'='*75}\n"
                f"TRAINING SAFETY GATE VIOLATION:\n"
                f"Dataset '{prov.dataset_name}' is classified as SYNTHETIC.\n"
                f"Silent training on synthetic/mock data for research is strictly prohibited.\n"
                f"To train for software development/testing only, pass '--allow-synthetic' or 'allow_synthetic=True'.\n"
                f"To train research models, configure an authentic dataset manifest in configs/datasets/.\n"
                f"{'='*75}"
            )

        if prov.is_synthetic():
            logger.warning(
                "DEVELOPMENT WARNING: Training with allow_synthetic=True on mock data. "
                "Output weights must NOT be used for scientific navigation benchmarks."
            )
        else:
            logger.info(
                f"TRAINING SAFETY PASSED: Authentic research dataset '{prov.dataset_name}' verified."
            )

        return prov
