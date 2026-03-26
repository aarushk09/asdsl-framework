"""Importance matrix (imatrix) helpers for mixed-precision quantization.

Typical use: run a short calibration forward pass, accumulate activation
magnitudes or gradient proxies, then derive per-group importance scores.

WikiText / HF example (outline)::

    # hidden: list of (batch, seq, hidden) tensors from selected layers
    im = accumulate_activation_importance(hidden_states, group_size=128)
"""

from __future__ import annotations

import numpy as np


def importance_from_activation_columns(
    x: np.ndarray,
    *,
    group_size: int,
) -> np.ndarray:
    """Per-input-column importance from calibration activations ``x``.

    Args:
        x: Shape ``(n_samples, K)`` or ``(batch, seq, K)`` (flattened internally).
        group_size: Quantization group width along K.

    Returns:
        Vector of shape ``(K,)`` with non-negative importance scores.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1, x.shape[-1])
    # Mean absolute activation as a simple saliency proxy (GGUF-style imatrix).
    return np.mean(np.abs(x), axis=0)


def group_importance_from_imatrix(
    imatrix_k: np.ndarray,
    k: int,
    group_size: int,
) -> np.ndarray:
    """Collapse per-column importance to one score per group along K."""
    imatrix_k = np.asarray(imatrix_k, dtype=np.float32).reshape(-1)
    if imatrix_k.size != k:
        raise ValueError(f"imatrix length {imatrix_k.size} != K {k}")
    if k % group_size != 0:
        raise ValueError("K must be divisible by group_size")
    g = k // group_size
    v = imatrix_k.reshape(g, group_size)
    return np.max(v, axis=1).astype(np.float32)


def assign_mixed_bits_from_groups(
    group_scores: np.ndarray,
    *,
    q4_group_fraction: float,
) -> np.ndarray:
    """Return uint8 array of 3 or 4: top ``q4_group_fraction`` groups use 4 bits."""
    if not 0.0 <= q4_group_fraction <= 1.0:
        raise ValueError("q4_group_fraction must be in [0, 1]")
    s = np.asarray(group_scores, dtype=np.float32).reshape(-1)
    n = s.size
    n4 = int(round(q4_group_fraction * n))
    bits = np.full(n, 3, dtype=np.uint8)
    if n4 <= 0:
        return bits
    thr_idx = np.argsort(-s)[:n4]
    bits[thr_idx] = 4
    return bits
