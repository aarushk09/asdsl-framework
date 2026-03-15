"""Native AVX2 LUT build + matvec dispatch layer.

Thin wrapper around the compiled ``asdsl.kernels._native_lut`` extension.
Falls back gracefully if the extension is not compiled.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import the native C++ extension (optional)
# ---------------------------------------------------------------------------

_native_lut = None
_native_lut_available = False

try:
    from asdsl.kernels import _native_lut as _native_lut

    _native_lut_available = True
    _omp = "OpenMP" if _native_lut.has_openmp else "single-threaded"
    logger.info("Native LUT kernel loaded (AVX2, %s)", _omp)
except ImportError:
    logger.info("Native LUT extension not built — using Python fallback")


def has_native_lut() -> bool:
    """Return True if the compiled AVX2 LUT kernel is available."""
    return _native_lut_available


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _ensure_f32_contiguous(arr) -> np.ndarray:
    """Convert torch.Tensor or any array to contiguous float32 numpy."""
    try:
        import torch
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().float().contiguous().numpy()
    except ImportError:
        pass
    return np.ascontiguousarray(arr, dtype=np.float32)


def _ensure_u8_contiguous(arr) -> np.ndarray:
    """Convert to contiguous uint8 numpy array."""
    try:
        import torch
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().contiguous().numpy().astype(np.uint8)
    except ImportError:
        pass
    return np.ascontiguousarray(arr, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_lut_tables_native(
    unpacked_weights: np.ndarray,
    activation: np.ndarray,
    scales: np.ndarray,
    bits: int,
    group_width: int,
    output_size: int,
    input_size: int,
    quant_group_size: int,
) -> np.ndarray:
    """Build all LUT tables using the native AVX2 kernel.

    Returns a flat float32 array of all tables concatenated.
    Raises RuntimeError if native extension is not available.
    """
    if not _native_lut_available:
        raise RuntimeError("Native LUT extension not available")

    weights = _ensure_u8_contiguous(unpacked_weights.ravel())
    act = _ensure_f32_contiguous(activation.ravel())
    sc = _ensure_f32_contiguous(scales.ravel())

    return np.asarray(_native_lut.lut_build_tables(
        weights, act, sc,
        bits, group_width, output_size, input_size, quant_group_size,
    ))


def lut_matvec_native(
    flat_tables: np.ndarray,
    unpacked_weights: np.ndarray,
    bits: int,
    group_width: int,
    output_size: int,
    input_size: int,
) -> np.ndarray:
    """LUT-based matrix-vector multiply using the native AVX2 kernel.

    Raises RuntimeError if native extension is not available.
    """
    if not _native_lut_available:
        raise RuntimeError("Native LUT extension not available")

    tables = _ensure_f32_contiguous(flat_tables.ravel())
    weights = _ensure_u8_contiguous(unpacked_weights.ravel())

    return np.asarray(_native_lut.lut_matvec(
        tables, weights,
        bits, group_width, output_size, input_size,
    ))
