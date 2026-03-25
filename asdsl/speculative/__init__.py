"""Speculative decoding algorithms for ASDSL."""

from asdsl.speculative.dual_model import (
	DualModelSpeculativeDecoder,
	SimulatedDualModel,
	SpeculativeRunResult,
	run_target_only_baseline,
)

__all__ = [
	"DualModelSpeculativeDecoder",
	"SimulatedDualModel",
	"SpeculativeRunResult",
	"run_target_only_baseline",
]
