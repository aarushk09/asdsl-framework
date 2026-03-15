"""Python bindings and dispatch for the native 8-bit GEMV kernel.

This module provides a pure-Python fallback (using PyTorch dequantization)
and an optional native C++/AVX2 fast path if compiled.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Native AVX2 extension load
# ---------------------------------------------------------------------------

_native_gemv_q8 = None
_native_available = False

try:
    from asdsl.kernels import _native_gemv_q8 as _native_gemv_q8
    _native_available = True
    _omp = "OpenMP" if _native_gemv_q8.has_openmp else "single-threaded"
    logger.info("Native 8-bit GEMV kernel loaded (AVX2, %s)", _omp)
except ImportError:
    logger.info(
        "Native 8-bit GEMV kernel not built — using Python/Torch fallback (slow)"
    )


def has_native_kernel() -> bool:
    """Return True if the C++/AVX2 8-bit GEMV extension is available."""
    return _native_available


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def gemv_q8_unpacked(
    w_u8: torch.Tensor | "numpy.ndarray",
    x: torch.Tensor | "numpy.ndarray",
    scales: torch.Tensor | "numpy.ndarray",
    biases: torch.Tensor | "numpy.ndarray",
    m: int,
    k: int,
    group_size: int = 128,
) -> "numpy.ndarray":
    """
    Computes y = dequant(W_q8) @ x.

    This uses the fast AVX2 C++ kernel if available, otherwise it falls back
    to PyTorch. The fast path expects unpacked 1-byte-per-value uint8 weights.

    Args:
        w_u8: Flat array of uint8 weights.
        x: Flat array of float32 input features.
        scales: Flat array of float32 per-group scales.
        biases: Flat array of float32 per-group biases.
        m: Number of output features (rows of W).
        k: Number of input features (cols of W).
        group_size: Quantization group size.

    Returns:
        Flat float32 output array of size `m`.
    """
    if _native_available:
        import numpy as np
        
        # Ensure fast C-contiguous numpy arrays
        def _ensure_np(arr, dtype):
            if isinstance(arr, torch.Tensor):
                return arr.detach().cpu().contiguous().numpy()
            return np.ascontiguousarray(arr, dtype=dtype)

        w_np = _ensure_np(w_u8, np.uint8)
        x_np = _ensure_np(x, np.float32)
        s_np = _ensure_np(scales, np.float32)
        b_np = _ensure_np(biases, np.float32)

        return _native_gemv_q8.gemv_q8_unpacked(
            w_np, x_np, s_np, b_np, m, k, group_size
        )

    # -----------------------------------------------------------------------
    # PyTorch Fallback (SLOW)
    # -----------------------------------------------------------------------
    with torch.no_grad():
        if not isinstance(w_u8, torch.Tensor):
            w_u8 = torch.from_numpy(w_u8)
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)
        if not isinstance(scales, torch.Tensor):
            scales = torch.from_numpy(scales)
        if not isinstance(biases, torch.Tensor):
            biases = torch.from_numpy(biases)
            
        w_u8 = w_u8.to(torch.uint8)
        x = x.to(torch.float32)
        scales = scales.to(torch.float32)
        biases = biases.to(torch.float32)

        groups_per_row = k // group_size
        w_view = w_u8.view(m, groups_per_row, group_size).float()
        sc_view = scales.view(m, groups_per_row, 1)
        bi_view = biases.view(m, groups_per_row, 1)

        w_dequant = w_view * sc_view + bi_view
        w_dequant = w_dequant.view(m, k)

        return torch.mv(w_dequant, x).cpu().numpy()
