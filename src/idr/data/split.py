"""Drive / Group-Level Train/Val/Test Splitting Utility.

Prevents data leakage across contiguous temporal windows by strictly partitioning
at the physical DRIVE / SEQUENCE level rather than individual sliding windows.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Set, Tuple
import logging
import random
from .provenance import DataLeakageError

logger = logging.getLogger(__name__)


@dataclass
class SplitResult:
    """Encapsulates the output and audit metadata of a drive-level split."""
    train_drives: List[str]
    val_drives: List[str]
    test_drives: List[str]
    ratios: Tuple[float, float, float]
    seed: int
    total_drives: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate_no_overlap(self) -> None:
        """Strictly verify zero overlap between train, val, and test drive partitions."""
        train_set = set(self.train_drives)
        val_set = set(self.val_drives)
        test_set = set(self.test_drives)

        train_val_overlap = train_set & val_set
        train_test_overlap = train_set & test_set
        val_test_overlap = val_set & test_set

        if train_val_overlap or train_test_overlap or val_test_overlap:
            leakage_info = {
                "train_val_leakage": list(train_val_overlap),
                "train_test_leakage": list(train_test_overlap),
                "val_test_leakage": list(val_test_overlap),
            }
            raise DataLeakageError(
                f"DATA LEAKAGE DETECTED: Drive partitions share identical sequence IDs! {leakage_info}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_drives": self.train_drives,
            "val_drives": self.val_drives,
            "test_drives": self.test_drives,
            "ratios": self.ratios,
            "seed": self.seed,
            "total_drives": self.total_drives,
            "metadata": self.metadata,
        }


class DriveSplitter:
    """Deterministic, group-aware drive splitter."""

    @staticmethod
    def split_by_drive(
        sequence_ids: Sequence[str],
        ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
        seed: int = 42,
        shuffle: bool = True,
        min_per_split: int = 1,
    ) -> SplitResult:
        """Partition drive/sequence IDs into disjoint (train, val, test) groups.
        
        Args:
            sequence_ids: Unique identifiers of physical drive recordings.
            ratios: (train_ratio, val_ratio, test_ratio) summing to 1.0.
            seed: Deterministic random seed.
            shuffle: Whether to permute sequence order before splitting.
            min_per_split: Minimum number of drives per split when sequence count allows.
            
        Returns:
            SplitResult with verified disjoint partitions.
        """
        unique_ids = list(dict.fromkeys(sequence_ids))  # preserve order while deduplicating
        N = len(unique_ids)
        if N == 0:
            raise ValueError("Cannot split empty sequence list.")

        train_r, val_r, test_r = ratios
        ratio_sum = train_r + val_r + test_r
        if not (0.999 <= ratio_sum <= 1.001):
            raise ValueError(f"Split ratios must sum to 1.0, got {ratios} (sum={ratio_sum})")

        items = list(unique_ids)
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(items)

        if N < 3:
            logger.warning(
                f"DriveSplitter received only {N} drive(s). All assigned to train to maintain execution."
            )
            train_drives = items
            val_drives = []
            test_drives = []
        else:
            # Group-based integer allocation
            n_val = max(min_per_split, int(round(N * val_r)))
            n_test = max(min_per_split, int(round(N * test_r)))
            
            # Ensure train has at least min_per_split
            if N - (n_val + n_test) < min_per_split:
                n_val = max(1, n_val - 1)
                if N - (n_val + n_test) < min_per_split:
                    n_test = max(1, n_test - 1)

            n_train = N - (n_val + n_test)
            if n_train < 1:
                n_train = 1
                if n_test > 1:
                    n_test -= 1
                elif n_val > 1:
                    n_val -= 1

            train_drives = items[:n_train]
            val_drives = items[n_train:n_train + n_val]
            test_drives = items[n_train + n_val:]

        result = SplitResult(
            train_drives=train_drives,
            val_drives=val_drives,
            test_drives=test_drives,
            ratios=ratios,
            seed=seed,
            total_drives=N,
            metadata={
                "train_count": len(train_drives),
                "val_count": len(val_drives),
                "test_count": len(test_drives),
            },
        )

        result.validate_no_overlap()
        logger.info(
            f"DriveSplitter complete: {len(train_drives)} train ({train_drives}), "
            f"{len(val_drives)} val ({val_drives}), {len(test_drives)} test ({test_drives})"
        )
        return result
