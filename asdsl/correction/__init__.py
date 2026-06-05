"""Per-layer MLP hidden-state correction for quantized Phi-4 inference (Phase 4)."""

from asdsl.correction.apply_correction import (
    CorrectionMLP,
    CorrectionModel,
    apply_correction,
    apply_layer_correction,
    load_correction,
)
from asdsl.correction.collect_training_data import collect_training_data
from asdsl.correction.train_corrections import train_corrections

# Legacy bias-based API (deprecated; kept for older scripts)
try:
    from asdsl.correction.collect import collect_residuals
    from asdsl.correction.train import train_correction
except ImportError:
    collect_residuals = None  # type: ignore[misc, assignment]
    train_correction = None  # type: ignore[misc, assignment]

__all__ = [
    "CorrectionMLP",
    "CorrectionModel",
    "apply_correction",
    "apply_layer_correction",
    "load_correction",
    "collect_training_data",
    "train_corrections",
    "collect_residuals",
    "train_correction",
]
