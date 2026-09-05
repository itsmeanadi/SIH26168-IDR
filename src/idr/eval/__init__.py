"""Evaluation submodule for blackout simulation, metrics, and report plotting."""

from .metrics import compute_navigation_metrics, NavigationMetrics
from .plotting import generate_evaluation_plots

__all__ = [
    "compute_navigation_metrics",
    "NavigationMetrics",
    "simulate_blackout_benchmark",
    "generate_evaluation_plots",
]

def __getattr__(name):
    """Load the optional torch-backed benchmark only when it is requested."""
    if name == "simulate_blackout_benchmark":
        from .blackout import simulate_blackout_benchmark
        return simulate_blackout_benchmark
    raise AttributeError(name)
