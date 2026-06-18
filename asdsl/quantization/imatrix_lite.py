"""Importance-matrix-lite: per-channel activation variance for weighted RTN quant."""

from __future__ import annotations

import numpy as np


def collect_channel_importance(
    activation_samples: np.ndarray,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    """activation_samples: [n_samples, n_channels] float32 -> importance [n_channels]."""
    if activation_samples.ndim != 2:
        raise ValueError("activation_samples must be 2-D")
    var = np.var(activation_samples.astype(np.float64), axis=0)
    imp = np.sqrt(var + eps).astype(np.float32)
    imp /= float(np.max(imp) + eps)
    return imp


def importance_weighted_round(
    weights: np.ndarray,
    importance: np.ndarray,
    scale: float,
    zero: float,
    *,
    bits: int = 4,
) -> np.ndarray:
    """Asymmetric quant with per-column importance weighting on rounding."""
    qmax = (1 << bits) - 1
    imp = importance.reshape(1, -1) if importance.ndim == 1 else importance
    w_adj = weights.astype(np.float64) * imp
    q = np.round((w_adj - zero) / scale).astype(np.int32)
    q = np.clip(q, 0, qmax)
    return q.astype(np.uint8)
