"""Dataset Provenance and Authenticity Classification System.

Explicitly tracks and enforces the distinction between:
- SYNTHETIC DEVELOPMENT DATA (mock / simulation signals for software testing)
- AUTHENTIC RESEARCH DATA (genuine peer-reviewed or benchmark open datasets)
- REAL MOTORCYCLE FIELD DATA (collected via synchronized onboard sensors / app)
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import logging
import yaml

logger = logging.getLogger(__name__)


class DatasetAuthenticity(str, Enum):
    """Rigorous classification of dataset authenticity."""
    SYNTHETIC = "synthetic"      # Mock / formula-generated / simulated signals
    AUTHENTIC = "authentic"      # Genuine public / benchmark datasets (e.g. real IO-VNBD)
    FIELD = "field"              # Real hardware field telemetry recorded on vehicle

    @classmethod
    def from_str(cls, value: str) -> DatasetAuthenticity:
        val_lower = str(value).strip().lower()
        for item in cls:
            if item.value == val_lower:
                return item
        raise ValueError(
            f"Invalid authenticity value '{value}'. Allowed values: {[e.value for e in cls]}"
        )


class DatasetAuthenticityError(Exception):
    """Base exception for dataset authenticity and safety violations."""
    pass


class SyntheticDataBlockedError(DatasetAuthenticityError):
    """Raised when synthetic data is supplied to a research or scientific workflow without explicit override."""
    pass


class DatasetNotFoundError(Exception):
    """Raised when a requested dataset manifest or path cannot be resolved."""
    pass


class DataLeakageError(Exception):
    """Raised when sequence overlap or target leakage is detected across train/val/test splits."""
    pass


class UnsafeEvaluationError(Exception):
    """Raised when an evaluation protocol violates scientific integrity (e.g. GT leakage into map matching)."""
    pass


@dataclass
class DatasetProvenance:
    """Comprehensive, machine-readable dataset provenance record."""
    dataset_name: str
    source: str
    authenticity: DatasetAuthenticity
    source_url: Optional[str] = None
    vehicle_type: str = "unknown"
    sensor_type: str = "unknown"
    sampling_rate_hz: float = 10.0
    ground_truth_type: str = "unknown"
    license_status: str = "unknown"
    acquisition_date: Optional[str] = None
    preprocessing_version: str = "1.0.0"
    drive_ids: List[str] = field(default_factory=list)
    notes: str = ""
    file_hashes: Dict[str, str] = field(default_factory=dict)
    data_path: Optional[str] = None
    metadata_extra: Dict[str, Any] = field(default_factory=dict)

    def is_synthetic(self) -> bool:
        return self.authenticity == DatasetAuthenticity.SYNTHETIC

    def is_authentic(self) -> bool:
        return self.authenticity == DatasetAuthenticity.AUTHENTIC

    def is_field(self) -> bool:
        return self.authenticity == DatasetAuthenticity.FIELD

    def validate_for_training(self, allow_synthetic: bool = False) -> None:
        """Validate whether this dataset is permitted for training."""
        if self.is_synthetic() and not allow_synthetic:
            raise SyntheticDataBlockedError(
                f"TRAINING SAFETY GATE TRIGGERED: Dataset '{self.dataset_name}' is classified as "
                f"SYNTHETIC. Training AI models for scientific research on synthetic data is strictly "
                f"blocked to prevent false performance claims. To train for software development or "
                f"regression tests only, explicitly pass '--allow-synthetic' or 'allow_synthetic=True'."
            )

    def validate_for_evaluation(self, allow_synthetic: bool = False) -> None:
        """Validate whether this dataset is permitted for scientific evaluation."""
        if self.is_synthetic() and not allow_synthetic:
            raise SyntheticDataBlockedError(
                f"EVALUATION SAFETY GATE TRIGGERED: Dataset '{self.dataset_name}' is classified as "
                f"SYNTHETIC. Scientific benchmarks cannot be computed on synthetic mock data. "
                f"Pass 'allow_synthetic=True' only for software integration testing."
            )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["authenticity"] = self.authenticity.value
        return data

    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict(), sort_keys=False)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save_manifest(self, filepath: Path) -> Path:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if filepath.suffix in [".yaml", ".yml"]:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(self.to_yaml())
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(self.to_json())
        logger.info(f"Saved dataset provenance manifest to: {filepath}")
        return filepath

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DatasetProvenance:
        d = dict(data)
        if "authenticity" in d:
            if isinstance(d["authenticity"], str):
                d["authenticity"] = DatasetAuthenticity.from_str(d["authenticity"])
        return cls(**d)

    @classmethod
    def from_file(cls, filepath: Path) -> DatasetProvenance:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset manifest file not found: {filepath}")
        
        with open(filepath, "r", encoding="utf-8") as f:
            if filepath.suffix in [".yaml", ".yml"]:
                content = yaml.safe_load(f)
            else:
                content = json.load(f)
        return cls.from_dict(content)


class DatasetRegistry:
    """Central registry of known datasets and their provenance manifests."""

    def __init__(self):
        self._datasets: Dict[str, DatasetProvenance] = {}
        self._paths: Dict[str, Path] = {}

    def register(self, provenance: DatasetProvenance, data_path: Optional[Path] = None) -> None:
        name = provenance.dataset_name.strip().lower()
        self._datasets[name] = provenance
        if data_path is not None:
            self._paths[name] = Path(data_path)
        elif provenance.data_path:
            self._paths[name] = Path(provenance.data_path)
        logger.debug(f"Registered dataset '{name}' [{provenance.authenticity.value}]")

    def register_manifest_file(self, manifest_path: Path, data_path_override: Optional[Path] = None) -> DatasetProvenance:
        prov = DatasetProvenance.from_file(manifest_path)
        self.register(prov, data_path=data_path_override)
        return prov

    def scan_directory(self, directory: Path) -> int:
        """Scan a directory for .yaml and .json dataset manifests."""
        directory = Path(directory)
        if not directory.exists():
            return 0
        count = 0
        for p in directory.glob("*.*"):
            if p.suffix in [".yaml", ".yml", ".json"]:
                try:
                    self.register_manifest_file(p)
                    count += 1
                except Exception as e:
                    logger.debug(f"Could not load manifest from {p}: {e}")
        return count

    def get(self, dataset_name: str) -> DatasetProvenance:
        key = dataset_name.strip().lower()
        if key not in self._datasets:
            raise DatasetNotFoundError(
                f"Dataset '{dataset_name}' is not registered in the DatasetRegistry. "
                f"Known datasets: {list(self._datasets.keys())}"
            )
        return self._datasets[key]

    def resolve(
        self,
        dataset_name: str,
        allow_synthetic: bool = False,
    ) -> Tuple[Path, DatasetProvenance]:
        """Resolve dataset path and validate safety permissions.
        
        Returns:
            (dataset_root_path, DatasetProvenance)
        """
        prov = self.get(dataset_name)
        prov.validate_for_training(allow_synthetic=allow_synthetic)

        key = dataset_name.strip().lower()
        path = self._paths.get(key)
        if path is None:
            raise DatasetNotFoundError(
                f"No data path configured for dataset '{dataset_name}'."
            )
        if not path.exists():
            raise DatasetNotFoundError(
                f"Data path for dataset '{dataset_name}' does not exist on disk: {path}"
            )
        return path, prov

    def list_datasets(self) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in self._datasets.values()]


# Default singleton instance
_GLOBAL_REGISTRY = DatasetRegistry()


def get_dataset_registry() -> DatasetRegistry:
    return _GLOBAL_REGISTRY
