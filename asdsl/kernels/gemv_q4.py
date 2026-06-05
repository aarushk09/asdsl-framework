"""Fused 4-bit GEMV: y = dequant(W_q4) @ x  with AVX2/FMA acceleration.

Provides a high-level Python API that dispatches to the native C++ AVX2
kernel when available, falling back to a vectorized NumPy implementation.

The "fused" aspect: instead of first dequantizing the full weight matrix
to float32 (which doubles memory traffic), we compute the integer
dot-product in registers and apply the per-group affine correction
(scale, bias) as a scalar post-processing step per group.

Typical speedup vs the PyTorch dequant+mv path:
    Native AVX2:  50-200x  (memory-bandwidth limited, ~20 GB/s)
    NumPy fallback:  3-10x (avoids intermediate allocations)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from asdsl.quantization.core import QuantizedTensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import the native C++ extension (optional)
# ---------------------------------------------------------------------------

_native = None
_native_available = False

try:
    from asdsl.kernels import _native_gemv as _native

    if _native.check_avx2() and _native.check_fma():
        _native_available = True
        _omp = "OpenMP" if _native.has_openmp else "single-threaded"
        logger.info("Native GEMV kernel loaded (AVX2+FMA, %s)", _omp)
    else:
        logger.warning(
            "Native GEMV module loaded but CPU lacks AVX2/FMA — using NumPy"
        )
        _native_available = False
except ImportError:
    logger.info("Native GEMV extension not built — using NumPy fallback")

# ---------------------------------------------------------------------------
# LUT kernel (Phase 1): vpshufb-based Q4 GEMV
# ---------------------------------------------------------------------------

_native_lut = None
_lut_import_attempted = False


def _try_import_lut():
    """Lazy import of _native_lut extension. Returns module or None."""
    global _native_lut, _lut_import_attempted
    if not _lut_import_attempted:
        _lut_import_attempted = True
        try:
            from asdsl.kernels import _native_lut as _nl
            _native_lut = _nl
            logger.info(
                "Native LUT GEMV kernel loaded (vpshufb=%s)",
                getattr(_nl, "lut_use_shuffle", "unknown"),
            )
        except ImportError:
            logger.info("Native LUT extension not built — LUT path unavailable")
    return _native_lut


def has_native_kernel() -> bool:
    """Return True if the compiled AVX2+FMA GEMV kernel is available."""
    return _native_available


# ---------------------------------------------------------------------------
# Vectorized NumPy fallback
# ---------------------------------------------------------------------------

def _gemv_q4_numpy_packed(
    w_packed: np.ndarray,
    x: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    M: int,
    K: int,
    group_size: int,
) -> np.ndarray:
    """Vectorized NumPy implementation of fused 4-bit packed GEMV.

    Processes rows in L3-friendly chunks to avoid allocating the full
    (M, K) dequantized matrix at once.
    """
    groups_per_row = K // group_size
    packed_per_row = K // 2

    x_grouped = x.reshape(groups_per_row, group_size)
    sum_x = x_grouped.sum(axis=1)  # (groups_per_row,)

    y = np.empty(M, dtype=np.float32)
    CHUNK = 256

    for start in range(0, M, CHUNK):
        end = min(start + CHUNK, M)
        n = end - start

        chunk = w_packed[
            start * packed_per_row : end * packed_per_row
        ].reshape(n, packed_per_row)

        unpacked = np.empty((n, K), dtype=np.float32)
        unpacked[:, 0::2] = (chunk & 0x0F).astype(np.float32)
        unpacked[:, 1::2] = ((chunk >> 4) & 0x0F).astype(np.float32)

        ug = unpacked.reshape(n, groups_per_row, group_size)
        int_dots = np.sum(ug * x_grouped[np.newaxis, :, :], axis=2)

        gs = start * groups_per_row
        ge = end * groups_per_row
        s = scales[gs:ge].reshape(n, groups_per_row)
        b = biases[gs:ge].reshape(n, groups_per_row)

        y[start:end] = np.sum(
            int_dots * s + b * sum_x[np.newaxis, :], axis=1
        )

    return y


def _gemv_q4_numpy_unpacked(
    w: np.ndarray,
    x: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    M: int,
    K: int,
    group_size: int,
) -> np.ndarray:
    """Vectorized NumPy fallback for unpacked uint8 weights."""
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

        y[start:end] = np.sum(
            int_dots * s + b * sum_x[np.newaxis, :], axis=1
        )

    return y


# ---------------------------------------------------------------------------
# Tensor conversion helpers
# ---------------------------------------------------------------------------

def _ensure_f32_contiguous(arr) -> np.ndarray:
    """Convert torch.Tensor or any array to contiguous float32 numpy."""
    if isinstance(arr, np.ndarray):
        if arr.dtype == np.float32 and arr.flags["C_CONTIGUOUS"]:
            return arr
        if arr.dtype == np.float32:
            return np.ascontiguousarray(arr)
    try:
        import torch

        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().float().contiguous().numpy()
    except ImportError:
        pass
    return np.ascontiguousarray(arr, dtype=np.float32)


def _ensure_u8_contiguous_fast(arr) -> np.ndarray:
    if isinstance(arr, np.ndarray) and arr.dtype == np.uint8 and arr.flags["C_CONTIGUOUS"]:
        return arr
    return _ensure_u8_contiguous(arr)


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

def gemv_q4_packed(
    w_packed: np.ndarray,
    x: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    M: int,
    K: int,
    group_size: int,
    out: np.ndarray | None = None,
    use_lut: bool = False,
    use_q8: bool = False,
    lut_cache=None,
    bits: int = 4,
) -> np.ndarray:
    """Fused 4-bit packed GEMV: y = dequant(W_q4) @ x.

    Args:
        w_packed:   Packed uint8 array, shape (M*K/2,).
                    Two 4-bit values per byte (low nibble first).
        x:          Input vector, shape (K,), float32, or batch (B, K) C-contiguous
                    for B independent GEMVs (native path avoids per-row pybind).
        scales:     Per-group scale factors, shape (total_groups,), float32.
        biases:     Per-group biases (= -zero*scale), shape (total_groups,), float32.
        M:          Output dimension (rows).
        K:          Input dimension (columns).
        group_size: Elements per quantization group.
        use_lut:    If True, use Phase 1 LUT-native path when lut_cache is provided,
                    else legacy vpshufb LUT, else FMA.
        use_q8:     If True, use dynamic Q8 activation quantization + madd_epi16 (Phase B).
        lut_cache:  Optional LUTProjectionCache from warm_cache (prebuilt T tables).
        bits:       Weight bit-width for LUT dispatch policy (default 4).

    Returns:
        Output float32, shape (M,) if x is 1-D, else (B, M).
    """
    x = _ensure_f32_contiguous(x)
    scales = _ensure_f32_contiguous(scales)
    biases = _ensure_f32_contiguous(biases)
    w_packed = _ensure_u8_contiguous_fast(w_packed)
    if out is not None and (out.shape[0] != M or out.dtype != np.float32):
        raise ValueError(f"out must be float32 shape ({M},), got {out.shape} {out.dtype}")

    if x.ndim == 1:
        if x.shape[0] != K:
            raise ValueError(f"x length {x.shape[0]} does not match K={K}")
    elif x.ndim == 2:
        if x.shape[1] != K:
            raise ValueError(f"x.shape[1]={x.shape[1]} does not match K={K}")
    else:
        raise ValueError("x must be 1-D (K,) or 2-D (batch, K)")

    # Q8 path (Phase B): dynamic Q8 activation quantization + integer GEMV
    # Use Q8 for ALL matrices — the activation quantization is done ONCE per
    # matrix call (shared across all output rows), so the overhead is amortized.
    if use_q8 and x.ndim == 1:
        if _native_available and hasattr(_native, "gemv_q4_q8_avx2"):
            y = np.zeros(M, dtype=np.float32)
            _native.gemv_q4_q8_avx2(w_packed, scales, x, y, M, K, group_size)
            return y

    # LUT path (Phase 1): prebuilt-table LUT or legacy vpshufb
    if use_lut and x.ndim == 1:
        from asdsl.lut.lut_dispatcher import LUTKernelDispatcher

        return LUTKernelDispatcher.dispatch(
            w_packed,
            x,
            scales,
            biases,
            M,
            K,
            group_size,
            lut_cache=lut_cache,
            bits=bits,
            use_lut=True,
            _try_import_lut=_try_import_lut,
            _native_available=_native_available,
            _native=_native,
            _gemv_numpy=_gemv_q4_numpy_packed,
        )

    if _native_available:
        if out is not None and x.ndim == 1:
            if hasattr(_native, "gemv_q4_packed_into"):
                _native.gemv_q4_packed_into(
                    w_packed, x, scales, biases, out, M, K, group_size
                )
                return out
            np.copyto(
                out,
                np.asarray(
                    _native.gemv_q4_packed(
                        w_packed, x, scales, biases, M, K, group_size
                    )
                ),
            )
            return out
        return np.asarray(
            _native.gemv_q4_packed(w_packed, x, scales, biases, M, K, group_size)
        )

    if x.ndim == 1:
        return _gemv_q4_numpy_packed(w_packed, x, scales, biases, M, K, group_size)
    b = int(x.shape[0])
    out = np.empty((b, M), dtype=np.float32)
    for i in range(b):
        out[i] = _gemv_q4_numpy_packed(
            w_packed, x[i], scales, biases, M, K, group_size
        )
    return out


def gemv_q4_unpacked(
    w: np.ndarray,
    x: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    M: int,
    K: int,
    group_size: int,
) -> np.ndarray:
    """Fused 4-bit GEMV on pre-unpacked uint8 weights.

    Drop-in replacement for WeightStore._matvec_quant: same data layout
    (1 byte per quantized value) but with fused dequant+dot product.

    Args:
        w:          Unpacked uint8 array, shape (M*K,). One value per byte.
        x:          Input vector, shape (K,), float32.
        scales:     Per-group scale factors, shape (total_groups,), float32.
        biases:     Per-group biases (= -zero*scale), shape (total_groups,), float32.
        M:          Output dimension (rows).
        K:          Input dimension (columns).
        group_size: Quantization group size.

    Returns:
        Output vector, shape (M,), float32.
    """
    x = _ensure_f32_contiguous(x)
    scales = _ensure_f32_contiguous(scales)
    biases = _ensure_f32_contiguous(biases)
    w = _ensure_u8_contiguous(w)

    if _native_available:
        return np.asarray(
            _native.gemv_q4_unpacked(w, x, scales, biases, M, K, group_size)
        )

    return _gemv_q4_numpy_unpacked(w, x, scales, biases, M, K, group_size)


def gemv_q4km_q8(
    weights_q4km: np.ndarray,
    x: np.ndarray,
    out_features: int,
    in_features: int,
) -> np.ndarray:
    """Q4_K_M superblock GEMV using native Q8 activation quantization path."""
    if not _native_available or not hasattr(_native, "gemv_q4km_q8_avx2"):
        raise RuntimeError("Native Q4_K_M GEMV is unavailable in this build")

    w = _ensure_u8_contiguous(weights_q4km).reshape(-1)
    xv = _ensure_f32_contiguous(x).reshape(-1)
    if xv.size != in_features:
        raise ValueError(f"x length {xv.size} does not match in_features={in_features}")

    y = np.zeros(out_features, dtype=np.float32)
    _native.gemv_q4km_q8_avx2(w, xv, y, out_features, in_features)
    return y


def gemv_q4(qtensor: QuantizedTensor, x: np.ndarray) -> np.ndarray:
    """High-level fused 4-bit GEMV from a QuantizedTensor.

    Computes y = dequant(qtensor) @ x without materializing the full
    dequantized weight matrix.

    Args:
        qtensor: A 4-bit QuantizedTensor (from asdsl.quantization.core).
        x:       Input vector, shape (K,), float32.

    Returns:
        Output vector, shape (M,), float32.
    """
    if qtensor.bits != 4:
        raise ValueError(
            f"gemv_q4 only supports 4-bit tensors, got {qtensor.bits}-bit"
        )

    M, K = qtensor.shape
    gs = qtensor.group_size

    scales = qtensor.scales.astype(np.float32)

    if qtensor.is_symmetric:
        half_range = 7.5  # (2^4 - 1) / 2
        biases = np.full_like(scales, -half_range) * scales
    else:
        zeros = qtensor.zeros.astype(np.float32)
        biases = -zeros * scales

    return gemv_q4_packed(qtensor.data, x, scales, biases, M, K, gs)
