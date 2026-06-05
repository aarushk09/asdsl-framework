"""Repack ASDSL Q4 weights into 66-byte Q4_128 blocks for gemv_q4_128_preq_avx2.

Layout per group (group_size=128), matches gemv_q4_128.cpp:
  bytes 0-1: fp16 scale (symmetric, amax/7)
  bytes 2-65: 64 packed nibbles (128 weights, zero-point implicit at 8)
"""

from __future__ import annotations

import numpy as np

from asdsl.quantization.core import _unpack_bits

BLOCK_SIZE = 66
GROUP_SIZE = 128


def _float32_to_fp16_bits(scale: float) -> int:
    return int(np.float16(scale).view(np.uint16))


def _symmetric_quantize_group(w_f: np.ndarray) -> tuple[float, np.ndarray]:
    """Return (scale, 64 nibbles) for 128 float weights."""
    amax = float(np.max(np.abs(w_f)))
    if amax < 1e-12:
        return 0.0, np.full(64, 0x88, dtype=np.uint8)
    scale = amax / 7.0
    inv = 7.0 / amax
    q = np.clip(np.round(w_f * inv) + 8.0, 0, 15).astype(np.uint8)
    nibbles = (q[1::2] << 4) | q[0::2]
    return scale, nibbles


def repack_asymmetric_to_q4_128_blocks(
    packed: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    rows: int,
    cols: int,
    group_size: int = GROUP_SIZE,
    *,
    bits: int = 4,
) -> np.ndarray:
    """Convert asymmetric packed weights to symmetric Q4_128 blocks (re-quantize per group)."""
    if group_size != GROUP_SIZE:
        raise ValueError(f"repack_asymmetric_to_q4_128_blocks requires group_size={GROUP_SIZE}")
    n_groups_per_row = cols // group_size
    packed_2d = packed.reshape(rows, cols // 2)
    sc = scales.astype(np.float32).ravel()
    bi = biases.astype(np.float32).ravel()
    blocks = np.zeros((rows, n_groups_per_row, BLOCK_SIZE), dtype=np.uint8)

    for row in range(rows):
        w_u8 = _unpack_bits(packed_2d[row], bits)
        for g in range(n_groups_per_row):
            gidx = row * n_groups_per_row + g
            scale_a = float(max(sc[gidx], 1e-8))
            bias = float(bi[gidx])
            zero = -bias / scale_a
            base = g * group_size
            q = w_u8[base : base + group_size].astype(np.float32)
            w_f = (q - zero) * scale_a
            scale_s, nibbles = _symmetric_quantize_group(w_f)
            sf16 = _float32_to_fp16_bits(scale_s)
            blocks[row, g, 0] = sf16 & 0xFF
            blocks[row, g, 1] = (sf16 >> 8) & 0xFF
            blocks[row, g, 2:] = nibbles

    return np.ascontiguousarray(blocks)


def repack_fp32_to_q4_128_blocks(
    weights: np.ndarray,
    rows: int,
    cols: int,
    group_size: int = GROUP_SIZE,
) -> np.ndarray:
    """Pack fp32 row-major weights into Q4_128 blocks."""
    if group_size != GROUP_SIZE:
        raise ValueError(f"repack_fp32_to_q4_128_blocks requires group_size={GROUP_SIZE}")
    n_groups_per_row = cols // group_size
    w2d = weights.reshape(rows, cols)
    blocks = np.zeros((rows, n_groups_per_row, BLOCK_SIZE), dtype=np.uint8)
    for row in range(rows):
        for g in range(n_groups_per_row):
            w_f = w2d[row, g * group_size : (g + 1) * group_size].astype(np.float32)
            scale_s, nibbles = _symmetric_quantize_group(w_f)
            sf16 = _float32_to_fp16_bits(scale_s)
            blocks[row, g, 0] = sf16 & 0xFF
            blocks[row, g, 1] = (sf16 >> 8) & 0xFF
            blocks[row, g, 2:] = nibbles
    return np.ascontiguousarray(blocks)


def blocks_to_flat(blocks: np.ndarray) -> np.ndarray:
    rows, n_groups, _ = blocks.shape
    return np.ascontiguousarray(blocks.reshape(rows, n_groups * BLOCK_SIZE))
