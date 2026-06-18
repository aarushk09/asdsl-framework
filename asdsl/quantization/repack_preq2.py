"""Repack Q4_32 preq blocks into preq2 layout (aligned quant + sidecar meta).

preq2 per row:
  meta: n_groups × 4 bytes (fp16 scale + fp16 zero), row-major
  quant_interleaved: organized in 4-row bands; per band per group 64 bytes =
    row0 nibbles[16] || row1[16] || row2[16] || row3[16] (cache-line aligned)

Bit-identical dequant values vs Q4_32 preq blocks.
"""

from __future__ import annotations

import numpy as np

from asdsl.quantization.repack_q4_32 import BLOCK_SIZE as PREQ_BLOCK_SIZE

PREQ2_META_BYTES = 4
PREQ2_GROUP_QUANT_BYTES = 64
PREQ2_ROW_BAND = 4


def repack_preq_blocks_to_preq2(flat_preq: np.ndarray, rows: int, cols: int, group_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Convert flat 20-byte preq row to (meta, quant_interleaved) arrays."""
    if group_size != 32:
        raise ValueError("preq2 repack requires group_size=32")
    flat_preq = np.ravel(flat_preq)
    n_groups = cols // group_size
    row_stride = n_groups * PREQ_BLOCK_SIZE
    if flat_preq.size < rows * row_stride:
        raise ValueError("flat_preq too small for shape")

    meta = np.zeros((rows, n_groups, PREQ2_META_BYTES), dtype=np.uint8)
    n_bands = (rows + PREQ2_ROW_BAND - 1) // PREQ2_ROW_BAND
    quant = np.zeros((n_bands, n_groups, PREQ2_GROUP_QUANT_BYTES), dtype=np.uint8)

    for row in range(rows):
        row_base = row * row_stride
        for g in range(n_groups):
            blk = flat_preq[row_base + g * PREQ_BLOCK_SIZE : row_base + (g + 1) * PREQ_BLOCK_SIZE]
            meta[row, g, :] = blk[0:4]
            band = row // PREQ2_ROW_BAND
            slot = row % PREQ2_ROW_BAND
            off = slot * 16
            quant[band, g, off : off + 16] = blk[4:20]

    return np.ascontiguousarray(meta.reshape(rows, n_groups * PREQ2_META_BYTES)), np.ascontiguousarray(quant)


def preq2_flat_sizes(rows: int, cols: int, group_size: int = 32) -> tuple[int, int]:
    n_groups = cols // group_size
    n_bands = (rows + PREQ2_ROW_BAND - 1) // PREQ2_ROW_BAND
    meta_bytes = rows * n_groups * PREQ2_META_BYTES
    quant_bytes = n_bands * n_groups * PREQ2_GROUP_QUANT_BYTES
    return meta_bytes, quant_bytes


def meta_to_flat(meta: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(meta.reshape(-1))


def quant_to_flat(quant: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(quant.reshape(-1))
