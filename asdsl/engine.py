"""High-level engine entrypoints.

Compatibility module that exposes benchmark-oriented calls from the
inference engine package.
"""

from asdsl.inference.engine import DualModelBenchmarkResult, run_dual_model_speculative_benchmark

__all__ = ["DualModelBenchmarkResult", "run_dual_model_speculative_benchmark"]
