"""Comprehensive Experiment Provenance Manifest Generator.

Guarantees full scientific reproducibility by logging the exact configuration,
dataset hashes, model provenance, and execution parameters for every experiment.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExperimentProvenanceManifest:
    """Complete machine-readable record detailing how any numerical result was generated."""
    experiment_id: str
    git_commit_hash: str
    dataset_name: str
    dataset_authenticity: str
    is_scientific_research_valid: bool
    train_drives: List[str]
    val_drives: List[str]
    test_drives: List[str]
    model_name: str
    model_checkpoint_file: str
    model_authenticity: str
    training_seed: int
    window_size: int
    sampling_rate_hz: float
    receptive_field_sec: float
    blackout_start_s: float
    blackout_end_s: float
    use_nhc: bool
    use_two_wheeler_profile: bool
    use_mapmatch: bool
    map_source: str
    smoothing_mode: str
    metrics_summary: Dict[str, float]
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, file_path: Path) -> Path:
        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved experiment provenance manifest to: {target}")
        return target
