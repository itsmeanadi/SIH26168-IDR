"""Evaluation submodule for blackout simulation, metrics, and report plotting."""

from .metrics import compute_navigation_metrics, NavigationMetrics
from .plotting import generate_evaluation_plots
from .blackout_gate import GNSSBlackoutGate, StreamSample, BlackoutPeriod
from .evaluation_harness import (
    ScientificEvaluator,
    EvaluationSpec,
    EvaluationProvenanceAudit,
    EvaluationRunResult,
)

__all__ = [
    "compute_navigation_metrics",
    "NavigationMetrics",
    "simulate_blackout_benchmark",
    "generate_evaluation_plots",
    "GNSSBlackoutGate",
    "StreamSample",
    "BlackoutPeriod",
    "ScientificEvaluator",
    "EvaluationSpec",
    "EvaluationProvenanceAudit",
    "EvaluationRunResult",
    "ExperimentHarness",
    "ExperimentResult",
    "ExperimentMetadata",
    "TimingAndSamplingStats",
    "SpeedAccuracyMetrics",
    "DeadReckoningMetrics",
]

def __getattr__(name):
    """Load optional or circular-dependent modules dynamically on access."""
    if name == "simulate_blackout_benchmark":
        from .blackout import simulate_blackout_benchmark
        return simulate_blackout_benchmark
    if name in (
        "ExperimentHarness",
        "ExperimentResult",
        "ExperimentMetadata",
        "TimingAndSamplingStats",
        "SpeedAccuracyMetrics",
        "DeadReckoningMetrics",
    ):
        from . import experiment_runner
        return getattr(experiment_runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

