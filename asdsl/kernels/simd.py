"""C kernel interface stubs for low-level SIMD operations.

Provides Python bindings to the platform-specific SIMD kernels
(AVX2 VPSHUFB, AVX-512 VNNI, ARM NEON TBL) used by the LUT engine.
Falls back to numpy for platforms without compiled kernels.
"""

from __future__ import annotations

import logging
import platform
from enum import IntEnum

import numpy as np

from asdsl.config import CPUFeature, detect_cpu_features

logger = logging.getLogger(__name__)


class KernelBackend(IntEnum):
    """Available compute kernel backends."""

    NUMPY = 0    # Pure numpy fallback
    AVX2 = 1     # x86 AVX2 + VPSHUFB
    AVX512 = 2   # x86 AVX-512
    VNNI = 3     # x86 AVX-512 VNNI (INT8 acceleration)
    AMX = 4      # Intel AMX (tile-based matrix ops)
    NEON = 5     # ARM NEON + TBL


def select_backend() -> KernelBackend:
    """Auto-detect and select the best available kernel backend."""
    features = detect_cpu_features()

    if CPUFeature.AVX512_VNNI in features:
        logger.info("Selected kernel backend: AVX-512 VNNI")
        return KernelBackend.VNNI
    elif CPUFeature.AVX512 in features:
        logger.info("Selected kernel backend: AVX-512")
        return KernelBackend.AVX512
    elif CPUFeature.AVX2 in features:
        logger.info("Selected kernel backend: AVX2")
        return KernelBackend.AVX2
    elif CPUFeature.NEON in features:
        logger.info("Selected kernel backend: ARM NEON")
        return KernelBackend.NEON

    logger.info("Selected kernel backend: NumPy (fallback)")
    return KernelBackend.NUMPY


def lut_shuffle_avx2(
    table: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Emulate AVX2 VPSHUFB (packed shuffle bytes) operation.

    In the real C kernel, this maps to a single VPSHUFB instruction
    that performs 32 parallel table lookups per cycle. This Python
    implementation provides identical semantics for testing.

    Args:
        table: Lookup table, shape (16,) per 128-bit lane.
        indices: Byte indices to look up.

    Returns:
        Looked-up values.
    """
    # VPSHUFB semantics: for each byte in indices, if high bit is set
    # the result is 0, otherwise result = table[index & 0x0F]
    result = np.zeros_like(indices)
    mask = (indices & 0x80) == 0
    result[mask] = table[indices[mask] & 0x0F]
    return result


def lut_tbl_neon(
    table: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Emulate ARM NEON TBL (table vector lookup) operation.

    In ARM NEON, TBL performs up to 16 parallel byte lookups from
    a 16-byte or 32-byte table per cycle.

    Args:
        table: Lookup table, up to 64 bytes.
        indices: Byte indices to look up.

    Returns:
        Looked-up values (0 for out-of-range indices).
    """
    result = np.zeros_like(indices)
    valid = indices < len(table)
    result[valid] = table[indices[valid]]
    return result


def fma_vnni_int8(
    a: np.ndarray,
    b: np.ndarray,
    accumulator: np.ndarray,
) -> np.ndarray:
    """Emulate AVX-512 VNNI VPDPBUSD (dot product with accumulation).

    Computes: acc += a_uint8 * b_int8 (4-element dot product per lane).
    Used during the prefill phase for INT8 compute acceleration.

    Args:
        a: Unsigned 8-bit activations.
        b: Signed 8-bit weights.
        accumulator: 32-bit accumulator.

    Returns:
        Updated accumulator.
    """
    # VNNI computes 4-element dot products and accumulates into int32
    a_i32 = a.astype(np.int32)
    b_i32 = b.astype(np.int32)

    # Process in groups of 4
    result = accumulator.copy()
    num_groups = min(len(a_i32), len(b_i32)) // 4

    for i in range(num_groups):
        start = i * 4
        dot = np.sum(a_i32[start : start + 4] * b_i32[start : start + 4])
        if i < len(result):
            result[i] += dot

    return result


def prefill_matmul_int8(
    activations: np.ndarray,
    weights: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """INT8 matrix multiplication for the prefill phase.

    During prefill, data reuse is high (processing entire prompt at once),
    making compute-bound INT8 multiplication more efficient than LUT-based
    approaches. This kernel is used for prefill, while LUT handles decode.

    Args:
        activations: Input activations, shape (seq_len, hidden_dim), int8.
        weights: Weight matrix, shape (out_dim, hidden_dim), int8.
        scales: Dequantization scales, shape (num_groups,).

    Returns:
        Output matrix, shape (seq_len, out_dim), float32.
    """
    # INT8 matmul with scale correction
    result_i32 = activations.astype(np.int32) @ weights.astype(np.int32).T

    # Apply per-group scales
    if len(scales) == 1:
        result = result_i32.astype(np.float32) * float(scales[0])
    else:
        # Broadcast scales across output dimension
        result = result_i32.astype(np.float32)
        groups = len(scales)
        per_group = max(result.shape[-1] // groups, 1)
        for g in range(groups):
            start = g * per_group
            end = min(start + per_group, result.shape[-1])
            result[:, start:end] *= float(scales[g])

    return result
