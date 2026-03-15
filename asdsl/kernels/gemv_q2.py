"""Python bindings and dispatch for the native 2-bit GEMV kernel.

Provides a pure-Python/NumPy fallback and an optional native C++/AVX2
fast path for 2-bit quantized weights.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from asdsl.quantization.core import QuantizedTensor

logger = logging.getLogger(__name__)

_native_q2 = None
_native_available = False

try:
    from asdsl.kernels import _native_gemv_q2 as _native_q2

    if _native_q2.check_avx2() and _native_q2.check_fma():
        _native_available = True
        _omp = "OpenMP" if _native_q2.has_openmp else "single-threaded"
        logger.info("Native 2-bit GEMV kernel loaded (AVX2+FMA, %s)", _omp)
    else:
        logger.warning(
            "Native 2-bit GEMV module loaded but CPU lacks AVX2/FMA — using NumPy"
        )
except ImportError:
    logger.info("Native 2-bit GEMV extension not built — using NumPy fallback")


def has_native_kernel() -> bool:
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


def _gemv_q2_numpy_unpacked(
    w: np.ndarray,
    x: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    M: int,
    K: int,
    group_size: int,
) -> np.ndarray:
    """Vectorized NumPy fallback for unpacked uint8 2-bit weights."""
    groups_per_row = K // group_size
    x_grouped = x.reshape(groups_per_row, group_size)
    sum_x = x_grouped.sum(axis=1)

    y = np.empty(M, dtype=np.float32)
    CHUNK = 256

    for start in range(0, M, CHUNK):
        end = min(start + CHUNK, M)
        n = end - start
        chunk = w[start * K : end * K].reshape(n, K).astype(np.float32)
        ug = chunk.reshape(n, groups_per_row, group_size)
        int_dots = np.sum(ug * x_grouped[np.newaxis, :, :], axis=2)

        gs = start * groups_per_row
        ge = end * groups_per_row
        s = scales[gs:ge].reshape(n, groups_per_row)
        b = biases[gs:ge].reshape(n, groups_per_row)

        y[start:end] = np.sum(int_dots * s + b * sum_x[np.newaxis, :], axis=1)

    return y


def gemv_q2_unpacked(
    w: np.ndarray,
    x: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    M: int,
    K: int,
    group_size: int,
) -> np.ndarray:
    """Fused 2-bit GEMV on pre-unpacked uint8 weights."""
    x = _ensure_f32(x)
    scales = _ensure_f32(scales)
    biases = _ensure_f32(biases)
    w = _ensure_u8(w)

    if _native_available:
        return np.asarray(
            _native_q2.gemv_q2_unpacked(w, x, scales, biases, M, K, group_size)
        )

    return _gemv_q2_numpy_unpacked(w, x, scales, biases, M, K, group_size)


def gemv_q2_packed(
    w_packed: np.ndarray,
    x: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    M: int,
    K: int,
    group_size: int,
) -> np.ndarray:
    """Fused 2-bit GEMV on packed weights (4 values per byte).

    Loads 4x less data from DRAM than the unpacked path. Used for
    QCSD draft tokens where bandwidth is the bottleneck.
    """
    x = _ensure_f32(x)
    scales = _ensure_f32(scales)
    biases = _ensure_f32(biases)
    w_packed = _ensure_u8(w_packed)

    if _native_available:
        return np.asarray(
            _native_q2.gemv_q2_packed(w_packed, x, scales, biases, M, K, group_size)
        )

    # NumPy fallback: unpack inline per chunk
    groups_per_row = K // group_size
    packed_per_row = K // 4
    x_grouped = x.reshape(groups_per_row, group_size)
    sum_x = x_grouped.sum(axis=1)

    y = np.empty(M, dtype=np.float32)
    CHUNK = 256

    for start in range(0, M, CHUNK):
        end = min(start + CHUNK, M)
        n = end - start
        chunk = w_packed[start * packed_per_row : end * packed_per_row].reshape(n, packed_per_row)

        unpacked = np.empty((n, K), dtype=np.float32)
        unpacked[:, 0::4] = (chunk & 0x03).astype(np.float32)
        unpacked[:, 1::4] = ((chunk >> 2) & 0x03).astype(np.float32)
        unpacked[:, 2::4] = ((chunk >> 4) & 0x03).astype(np.float32)
        unpacked[:, 3::4] = ((chunk >> 6) & 0x03).astype(np.float32)

        ug = unpacked.reshape(n, groups_per_row, group_size)
        int_dots = np.sum(ug * x_grouped[np.newaxis, :, :], axis=2)

        gs = start * groups_per_row
        ge = end * groups_per_row
        s = scales[gs:ge].reshape(n, groups_per_row)
        b = biases[gs:ge].reshape(n, groups_per_row)
        y[start:end] = np.sum(int_dots * s + b * sum_x[np.newaxis, :], axis=1)

    return y
