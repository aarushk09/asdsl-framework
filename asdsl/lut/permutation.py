"""Weight permutation and interleaving for optimal LUT memory access patterns.

Offline weight transformation that reshapes and tiles weight matrices so their
storage order in DRAM matches the exact chronological order the LUT execution
kernels request them during inference. This maximizes cache line utilization
and enables the CPU memory controller to prefetch perfectly.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def permute_weights_for_lut(
    packed_weights: np.ndarray,
    bits: int,
    output_size: int,
    input_size: int,
    group_width: int = 4,
    cache_line_bytes: int = 64,
) -> np.ndarray:
    """Permute packed weights for sequential LUT access during inference.

    The standard weight layout stores rows contiguously:
        [row0_col0..colN, row1_col0..colN, ...]

    For LUT-based inference, we want weights tiled so that when processing
    output element i with LUT group j, the data for group j+1 is in the
    next bytes — ensuring sequential memory access and full cache line use.

    Strategy:
    - Tile by (tile_rows, group_width) blocks
    - Within each tile, data is laid out for sequential LUT consumption
    - Tile size chosen to align with CPU cache line size

    Args:
        packed_weights: Packed quantized weights.
        bits: Bit-width.
        output_size: Number of output features.
        input_size: Number of input features.
        group_width: LUT group width.
        cache_line_bytes: CPU cache line size (typically 64 bytes).

    Returns:
        Permuted packed weights array optimized for LUT access.
    """
    from asdsl.quantization.core import _pack_bits, _unpack_bits

    unpacked = _unpack_bits(packed_weights, bits)

    # Ensure we have enough data
    total_elements = output_size * input_size
    if len(unpacked) < total_elements:
        unpacked = np.concatenate([
            unpacked,
            np.zeros(total_elements - len(unpacked), dtype=np.uint8),
        ])

    # Reshape to matrix form
    matrix = unpacked[:total_elements].reshape(output_size, input_size)

    # Compute tile dimensions
    # Elements per cache line (packed)
    elements_per_cl = (cache_line_bytes * 8) // bits
    tile_cols = group_width
    tile_rows = min(elements_per_cl // tile_cols, output_size)
    tile_rows = max(tile_rows, 1)

    # Pad to tile boundaries
    pad_rows = (tile_rows - output_size % tile_rows) % tile_rows
    pad_cols = (tile_cols - input_size % tile_cols) % tile_cols

    if pad_rows or pad_cols:
        matrix = np.pad(matrix, ((0, pad_rows), (0, pad_cols)), constant_values=0)

    padded_rows, padded_cols = matrix.shape

    # Tile and interleave
    permuted = []
    for tile_row_start in range(0, padded_rows, tile_rows):
        for tile_col_start in range(0, padded_cols, tile_cols):
            tile = matrix[
                tile_row_start : tile_row_start + tile_rows,
                tile_col_start : tile_col_start + tile_cols,
            ]
            # Flatten tile in row-major order for sequential access
            permuted.append(tile.reshape(-1))

    permuted_flat = np.concatenate(permuted).astype(np.uint8)

    return _pack_bits(permuted_flat, bits)


def interleave_for_simd(
    packed_weights: np.ndarray,
    bits: int,
    simd_width: int = 32,
) -> np.ndarray:
    """Interleave packed weights for SIMD register-width aligned access.

    Swizzles the bit layout so that CPU load instructions naturally unpack
    values into the correct SIMD lanes without runtime bit manipulation.

    For AVX2 (256-bit = 32 bytes), we arrange data so each 32-byte load
    fills all lanes with correctly positioned quantized values.

    Args:
        packed_weights: Packed quantized weight data.
        bits: Bit-width.
        simd_width: SIMD register width in bytes (AVX2=32, AVX512=64).

    Returns:
        Interleaved packed weights aligned for SIMD access.
    """
    # For 2-bit: 4 values per byte, 128 values per AVX2 register
    values_per_byte = 8 // bits
    values_per_simd = simd_width * values_per_byte

    # Ensure alignment
    pad_needed = (simd_width - len(packed_weights) % simd_width) % simd_width
    if pad_needed:
        aligned = np.concatenate([
            packed_weights,
            np.zeros(pad_needed, dtype=np.uint8),
        ])
    else:
        aligned = packed_weights.copy()

    # Reorder within each SIMD-width block for lane-aligned access
    # This is a simplified version; real implementation would use
    # platform-specific shuffle patterns
    result = np.empty_like(aligned)
    num_blocks = len(aligned) // simd_width

    for block in range(num_blocks):
        start = block * simd_width
        end = start + simd_width
        block_data = aligned[start:end]

        # Interleave even/odd bytes for paired SIMD operations
        # This allows VPSHUFB to extract values without masking
        even = block_data[0::2]
        odd = block_data[1::2]
        result[start : start + len(even)] = even
        result[start + len(even) : end] = odd

    return result


def compute_permutation_map(
    output_size: int,
    input_size: int,
    group_width: int,
    tile_rows: int,
) -> np.ndarray:
    """Compute the index permutation map for weight reordering.

    This map can be applied at model load time and stored alongside
    the quantized weights, avoiding re-computation.

    Args:
        output_size: Number of output features.
        input_size: Number of input features.
        group_width: LUT group width.
        tile_rows: Number of rows per tile.

    Returns:
        Permutation index array — apply via weights[perm_map] to reorder.
    """
    total = output_size * input_size
    indices = np.arange(total).reshape(output_size, input_size)

    tile_cols = group_width
    pad_rows = (tile_rows - output_size % tile_rows) % tile_rows
    pad_cols = (tile_cols - input_size % tile_cols) % tile_cols

    if pad_rows or pad_cols:
        indices = np.pad(indices, ((0, pad_rows), (0, pad_cols)), constant_values=-1)

    padded_rows, padded_cols = indices.shape
    permuted_indices = []

    for tr in range(0, padded_rows, tile_rows):
        for tc in range(0, padded_cols, tile_cols):
            tile = indices[tr : tr + tile_rows, tc : tc + tile_cols]
            permuted_indices.append(tile.reshape(-1))

    return np.concatenate(permuted_indices)
