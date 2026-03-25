"""High-level engine entrypoints.

Compatibility module that exposes benchmark-oriented calls from the
inference engine package.
"""

from asdsl.inference.engine import (
	DualModelBenchmarkResult,
	NativePerplexityResult,
	evaluate_perplexity_phase8_native,
	run_dual_model_speculative_benchmark,
)

__all__ = [
	"DualModelBenchmarkResult",
	"NativePerplexityResult",
	"evaluate_perplexity_phase8_native",
	"run_dual_model_speculative_benchmark",
]
