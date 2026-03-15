"""Activation-sparse GEMV dispatch and bitmask utilities (Tier 3).

Provides Python-level bitmask computation and dispatch to either the
native C++/AVX2 sparse kernel or a NumPy fallback.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_native_sparse = None
_native_available = False

try:
    from asdsl.kernels import _native_sparse_gemv as _native_sparse
    _native_available = True
    logger.info("Native sparse GEMV kernel loaded")
except ImportError:
    logger.info("Native sparse GEMV not built — using NumPy fallback")


def has_native_sparse_kernel() -> bool:
    return _native_available


def _ensure_f32(arr) -> np.ndarray:
    try:
        import torch
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().float().contiguous().numpy()
    except ImportError:
        pass
    return np.ascontiguousarray(arr, dtype=np.float32)


def _ensure_u8(arr) -> np.ndarray:
    try:
        import torch
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().contiguous().numpy().astype(np.uint8)
    except ImportError:
        pass
    return np.ascontiguousarray(arr, dtype=np.uint8)


def compute_activation_bitmask(act: np.ndarray, threshold: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Compute a uint32 bitmask indicating non-negligible activations.

    Returns:
        (bitmask, active_indices): bitmask is uint32 array of length ceil(K/32),
        active_indices is int32 array of column indices where |act| >= threshold.
    """
    act_flat = act.ravel().astype(np.float32)

    if _native_available:
        bitmask = np.asarray(_native_sparse.compute_bitmask(act_flat, threshold))
    else:
        n = len(act_flat)
        n_words = (n + 31) // 32
        active = (np.abs(act_flat) >= threshold)
        bitmask = np.zeros(n_words, dtype=np.uint32)
        for wi in range(n_words):
            start = wi * 32
            end = min(start + 32, n)
            bits = active[start:end]
            for bp, b in enumerate(bits):
                if b:
                    bitmask[wi] |= np.uint32(1 << bp)

    active_indices = np.where(np.abs(act_flat) >= threshold)[0].astype(np.int32)

    return bitmask, active_indices


def gemv_sparse_unpacked(
    w: np.ndarray,
    x: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    bitmask: np.ndarray,
    M: int,
    K: int,
    group_size: int,
) -> np.ndarray:
    """Sparse GEMV skipping near-zero activation columns."""
    x = _ensure_f32(x)
    scales = _ensure_f32(scales)
    biases = _ensure_f32(biases)
    w = _ensure_u8(w)
    bitmask = np.ascontiguousarray(bitmask, dtype=np.uint32)

    if _native_available:
        return np.asarray(
            _native_sparse.gemv_sparse_unpacked(
                w, x, scales, biases, bitmask, M, K, group_size
            )
        )

    return _gemv_sparse_numpy(w, x, scales, biases, bitmask, M, K, group_size)


def gemv_sparse_with_indices(
    w: np.ndarray,
    x: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    active_indices: np.ndarray,
    M: int,
    K: int,
    group_size: int,
) -> np.ndarray:
    """Sparse GEMV using pre-computed active column indices."""
    x = _ensure_f32(x)
    scales = _ensure_f32(scales)
    biases = _ensure_f32(biases)
    w = _ensure_u8(w)
    active_indices = np.ascontiguousarray(active_indices, dtype=np.int32)

    if _native_available:
        return np.asarray(
            _native_sparse.gemv_sparse_with_indices(
                w, x, scales, biases, active_indices, M, K, group_size
            )
        )

    return _gemv_sparse_indices_numpy(w, x, scales, biases, active_indices, M, K, group_size)


def _gemv_sparse_numpy(
    w: np.ndarray,
    x: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    bitmask: np.ndarray,
    M: int,
    K: int,
    group_size: int,
) -> np.ndarray:
    """NumPy fallback: build active index list from bitmask, then scatter."""
    n_words = len(bitmask)
    active_cols = []
    for wi in range(n_words):
        mask = int(bitmask[wi])
        base = wi * 32
        while mask:
            bit = (mask & -mask).bit_length() - 1
            k = base + bit
            if k < K:
                active_cols.append(k)
            mask &= mask - 1
    active_cols = np.array(active_cols, dtype=np.int32)
    return _gemv_sparse_indices_numpy(w, x, scales, biases, active_cols, M, K, group_size)


def _gemv_sparse_indices_numpy(
    w: np.ndarray,
    x: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    active_indices: np.ndarray,
    M: int,
    K: int,
    group_size: int,
) -> np.ndarray:
    """NumPy fallback for index-based sparse GEMV."""
    groups_per_row = K // group_size
    y = np.zeros(M, dtype=np.float32)

    if len(active_indices) == 0:
        return y

    w_mat = w.reshape(M, K).astype(np.float32)
    x_active = x[active_indices]
    w_active = w_mat[:, active_indices]

    # Group assignment for each active column
    col_groups = active_indices // group_size

    for g in range(groups_per_row):
        mask = col_groups == g
        if not np.any(mask):
            continue
        x_g = x_active[mask]
        w_g = w_active[:, mask]
        gs = np.arange(M) * groups_per_row + g
        s = scales[gs].astype(np.float32)
        b = biases[gs].astype(np.float32)
        dot = w_g @ x_g
        sum_x = x_g.sum()
        y += s * dot + b * sum_x

    return y
