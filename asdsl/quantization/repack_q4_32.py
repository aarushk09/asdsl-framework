"""Repack ASDSL Q4 weights into 20-byte Q4_32 blocks for gemv_q4_32_preq_avx2.

Layout per group (group_size=32):
  bytes 0-1: fp16 scale
  bytes 2-3: fp16 zero-point (asymmetric quant), matches packed GEMV bias = -zero*scale
  bytes 4-19: 16 packed nibbles (low nibble = even index), copied from packed weights
"""

from __future__ import annotations

import numpy as np

from asdsl.quantization.core import _unpack_bits


BLOCK_SIZE = 20


def _float32_to_fp16_bits(scale: float) -> int:
    return int(np.float16(scale).view(np.uint16))


def repack_symmetric_from_packed(
    packed: np.ndarray,
    scales: np.ndarray,
    rows: int,
    cols: int,
    group_size: int = 32,
    *,
    bits: int = 4,
) -> np.ndarray:
    """Build (rows, n_groups, 18) uint8 blocks from symmetric Q4 packed weights."""
    if group_size != 32:
        raise ValueError("repack_symmetric_from_packed requires group_size=32")
    n_groups_per_row = cols // group_size
    blocks = np.zeros((rows, n_groups_per_row, BLOCK_SIZE), dtype=np.uint8)
    packed_2d = packed.reshape(rows, cols // 2)
    sc = scales.astype(np.float32).ravel()
    half = float((1 << bits) - 1) / 2.0

    for row in range(rows):
        for g in range(n_groups_per_row):
            gidx = row * n_groups_per_row + g
            scale = float(max(sc[gidx], 1e-8))
            sf16 = _float32_to_fp16_bits(scale)
            blocks[row, g, 0] = sf16 & 0xFF
            blocks[row, g, 1] = (sf16 >> 8) & 0xFF
            zf16 = _float32_to_fp16_bits(8.0)
            blocks[row, g, 2] = zf16 & 0xFF
            blocks[row, g, 3] = (zf16 >> 8) & 0xFF
            w_u8 = _unpack_bits(packed_2d[row], bits)
            base = g * group_size
            wg = w_u8[base : base + group_size].astype(np.float32)
            w_f = (wg - half) * scale
            inv = half / scale if scale > 0 else 0.0
            q = np.clip(np.round(w_f * inv + half), 0, 15).astype(np.uint8)
            nibbles = (q[1::2] << 4) | q[0::2]
            blocks[row, g, 4:] = nibbles

    return np.ascontiguousarray(blocks)


def repack_asymmetric_to_q4_32_blocks(
    packed: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    rows: int,
    cols: int,
    group_size: int = 32,
    *,
    bits: int = 4,
) -> np.ndarray:
    """Convert asymmetric packed + scale/bias into symmetric Q4_32 blocks."""
    if group_size != 32:
        raise ValueError("repack_asymmetric_to_q4_32_blocks requires group_size=32")
    n_groups_per_row = cols // group_size
    n_groups_total = rows * n_groups_per_row
    packed_2d = packed.reshape(rows, cols // 2)
    sc = scales.astype(np.float32).ravel()
    bi = biases.astype(np.float32).ravel()
    qmax = (1 << bits) - 1
    blocks = np.zeros((rows, n_groups_per_row, BLOCK_SIZE), dtype=np.uint8)

    for row in range(rows):
        for g in range(n_groups_per_row):
            gidx = row * n_groups_per_row + g
            scale = float(max(sc[gidx], 1e-8))
            bias = float(bi[gidx])
            zero = -bias / scale
            w_u8 = _unpack_bits(packed_2d[row], bits)
            base = g * group_size
            q = w_u8[base : base + group_size]
            sf16 = _float32_to_fp16_bits(scale)
            blocks[row, g, 0] = sf16 & 0xFF
            blocks[row, g, 1] = (sf16 >> 8) & 0xFF
            zf16 = _float32_to_fp16_bits(zero)
            blocks[row, g, 2] = zf16 & 0xFF
            blocks[row, g, 3] = (zf16 >> 8) & 0xFF
            nibbles = (q[1::2] << 4) | q[0::2]
            blocks[row, g, 4:] = nibbles

    return np.ascontiguousarray(blocks)


def blocks_to_flat(blocks: np.ndarray) -> np.ndarray:
    """Flatten (rows, n_groups, 18) to row-major bytes for C++ kernels."""
    rows, n_groups, _ = blocks.shape
    return np.ascontiguousarray(blocks.reshape(rows, n_groups * BLOCK_SIZE))
