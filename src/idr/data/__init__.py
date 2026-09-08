"""Data provenance, manifest management, group-aware dataset splitting, and safety gates."""

from .provenance import (
    DatasetAuthenticity,
    DatasetAuthenticityError,
    SyntheticDataBlockedError,
    DatasetNotFoundError,
    DataLeakageError,
    UnsafeEvaluationError,
    DatasetProvenance,
    DatasetRegistry,
    get_dataset_registry,
)
from .split import (
    DriveSplitter,
    SplitResult,
)
from .safety import (
    TrainingSafetyGate,
    TrainingProvenanceRecord,
)

__all__ = [
    "DatasetAuthenticity",
    "DatasetAuthenticityError",
    "SyntheticDataBlockedError",
    "DatasetNotFoundError",
    "DataLeakageError",
    "UnsafeEvaluationError",
    "DatasetProvenance",
    "DatasetRegistry",
    "get_dataset_registry",
    "DriveSplitter",
    "SplitResult",
    "TrainingSafetyGate",
    "TrainingProvenanceRecord",
]
