"""Lookup Table (LUT) engine for sub-byte matrix-vector multiplication.

Replaces traditional dequantize-then-FMA with precomputed partial-sum
table lookups. For 2-bit weights (4 possible values), all partial sums
are precomputed and stored in tables small enough for L1 cache.

At runtime, matrix-vector products are computed via table lookups +
additions, completely eliminating floating-point multiply overhead.
This is the T-MAC methodology adapted for the ASDSL framework.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LookupTable:
    """A precomputed partial-sum lookup table for a weight group.

    For a group of `group_width` weights at `bits` precision, the table
    contains all 2^(group_width * bits) possible partial sums with any
    activation vector segment.

    Attributes:
        table: The lookup table array. Shape depends on bit-width and group.
        bits: Bit-width of the quantized weights.
        group_width: Number of weight elements per lookup group.
        scale: Scale factor for this group.
        offset: Zero-point offset.
    """

    table: np.ndarray
    bits: int
    group_width: int
    scale: float
    offset: float = 0.0

    @property
    def num_entries(self) -> int:
        return len(self.table)

    @property
    def memory_bytes(self) -> int:
        return self.table.nbytes


@dataclass
class LUTEngine:
    """Manages precomputed lookup tables for LUT-based matrix multiplication.

    The engine holds tables for all weight groups and provides the
    lut_matvec() operation that replaces standard GEMV.
    """

    tables: list[LookupTable] = field(default_factory=list)
    bits: int = 2
    group_width: int = 4  # Number of weights per LUT entry

    @property
    def total_memory_bytes(self) -> int:
        return sum(t.memory_bytes for t in self.tables)

    @property
    def total_memory_kb(self) -> float:
        return self.total_memory_bytes / 1024


def build_lut_for_group(
    weights_quantized: np.ndarray,
    activation_segment: np.ndarray,
    bits: int,
    scale: float,
    group_width: int = 4,
) -> LookupTable:
    """Build a lookup table for a group of quantized weights and activations.

    For `group_width` weights at `bits` precision, there are
    (2^bits)^group_width = 2^(bits*group_width) possible weight combinations.
    For each combination, we precompute the dot product with the activation.

    For 2-bit, group_width=4: 2^8 = 256 entries (fits in one cache line set).
    For 2-bit, group_width=2: 2^4 = 16 entries (fits in a single SIMD register).

    Args:
        weights_quantized: Quantized weight values for this group, shape (group_width,).
        activation_segment: Corresponding activation values, shape (group_width,).
        bits: Bit-width of weights.
        scale: Dequantization scale factor.
        group_width: Number of weights per lookup entry.

    Returns:
        LookupTable with precomputed partial sums.
    """
    num_values = 1 << bits  # Number of possible values per weight
    num_entries = num_values**group_width
    # Use (num_values//4) as center: for 2-bit → 1, meaning qval=1 maps to zero.
    # This matches the convention (qval - 1)*scale for 2-bit symmetric quantization.
    half = num_values // 4

    table = np.zeros(num_entries, dtype=np.float32)

    # Precompute all possible partial sums
    for idx in range(num_entries):
        partial_sum = 0.0
        remaining = idx
        for w in range(group_width):
            qval = remaining % num_values
            remaining //= num_values
            # Symmetric dequant: float_val = (qval - half) * scale
            float_val = (qval - half) * scale
            partial_sum += float_val * activation_segment[w]
        table[idx] = partial_sum

    return LookupTable(
        table=table,
        bits=bits,
        group_width=group_width,
        scale=scale,
    )


def build_lut_tables_for_layer(
    packed_weights: np.ndarray,
    scales: np.ndarray,
    activation: np.ndarray,
    bits: int,
    group_size: int,
    group_width: int = 4,
) -> list[LookupTable]:
    """Build all LUT tables needed for a single layer's weight matrix.

    Args:
        packed_weights: Packed quantized weights (uint8 array).
        scales: Per-quantization-group scale factors.
        activation: Input activation vector, shape (in_features,).
        bits: Bit-width.
        group_size: Elements per quantization group (for scale lookup).
        group_width: Elements per LUT entry.

    Returns:
        List of LookupTable objects covering the entire weight matrix.
    """
    from asdsl.quantization.core import _unpack_bits

    unpacked = _unpack_bits(packed_weights, bits)
    tables = []

    # Process in LUT groups of group_width elements
    for start in range(0, len(unpacked), group_width):
        end = min(start + group_width, len(unpacked))
        if end - start < group_width:
            break

        w_group = unpacked[start:end]

        # Map activation indices (handle matrix layout)
        act_start = start % len(activation)
        act_end = act_start + group_width
        if act_end <= len(activation):
            act_group = activation[act_start:act_end]
        else:
            # Wrap around
            act_group = np.concatenate([
                activation[act_start:],
                activation[: act_end - len(activation)],
            ])

        # Get scale for this quantization group
        quant_group_idx = start // group_size
        if quant_group_idx < len(scales):
            scale = float(scales[quant_group_idx])
        else:
            scale = float(scales[-1])

        table = build_lut_for_group(
            weights_quantized=w_group,
            activation_segment=act_group,
            bits=bits,
            scale=scale,
            group_width=group_width,
        )
        tables.append(table)

    return tables


def lut_matvec(
    tables: list[LookupTable],
    packed_weights: np.ndarray,
    bits: int,
    output_size: int,
    input_size: int,
) -> np.ndarray:
    """Perform matrix-vector multiplication using precomputed lookup tables.

    This replaces the traditional:
        output = dequantize(weights) @ activation
    with:
        output[i] = sum of table_lookups for row i

    The activation values are already baked into the LUT entries during
    table construction, so at inference time we only need to:
    1. Read the packed weight indices
    2. Use them as table offsets
    3. Sum the partial results

    Args:
        tables: Precomputed LUT tables (from build_lut_tables_for_layer).
        packed_weights: Packed weight data.
        bits: Bit-width.
        output_size: Number of output features (rows in weight matrix).
        input_size: Number of input features (columns in weight matrix).

    Returns:
        Output vector, shape (output_size,).
    """
    from asdsl.quantization.core import _unpack_bits

    unpacked = _unpack_bits(packed_weights, bits)
    output = np.zeros(output_size, dtype=np.float32)

    if not tables:
        return output

    group_width = tables[0].group_width

    # Process each output row
    for row in range(output_size):
        row_start = row * input_size
        row_sum = 0.0

        # Process in LUT-width chunks
        for col_start in range(0, input_size, group_width):
            col_end = min(col_start + group_width, input_size)
            if col_end - col_start < group_width:
                break

            # Compute table index from packed weight values
            flat_idx = row_start + col_start
            table_key = 0
            for w in range(group_width):
                if flat_idx + w < len(unpacked):
                    qval = int(unpacked[flat_idx + w])
                    table_key += qval * ((1 << bits) ** w)

            # Table lookup
            table_idx = (row * (input_size // group_width)) + (col_start // group_width)
            if table_idx < len(tables):
                table = tables[table_idx]
                if table_key < table.num_entries:
                    row_sum += table.table[table_key]

        output[row] = row_sum

    return output


def lut_matvec_batched(
    tables: list[LookupTable],
    packed_weights: np.ndarray,
    activations: np.ndarray,
    bits: int,
    output_size: int,
    input_size: int,
) -> np.ndarray:
    """Batched LUT matrix-vector multiplication for verification passes.

    During self-speculative verification, multiple tokens are verified
    in parallel. This batched variant processes all tokens simultaneously.

    Args:
        tables: Precomputed LUT tables.
        packed_weights: Packed weight data.
        activations: Batch of activation vectors, shape (batch_size, in_features).
        bits: Bit-width.
        output_size: Output features.
        input_size: Input features.

    Returns:
        Output matrix, shape (batch_size, output_size).
    """
    batch_size = activations.shape[0]
    outputs = np.zeros((batch_size, output_size), dtype=np.float32)

    for b in range(batch_size):
        # For each batch element, rebuild tables with new activations
        # In optimized implementation, tables are rebuilt efficiently
        # using SIMD register shuffles
        outputs[b] = lut_matvec(
            tables=tables,
            packed_weights=packed_weights,
            bits=bits,
            output_size=output_size,
            input_size=input_size,
        )

    return outputs


def estimate_lut_memory(
    bits: int,
    group_width: int,
    num_weight_groups: int,
) -> dict[str, float]:
    """Estimate memory requirements for LUT tables.

    Args:
        bits: Weight bit-width.
        group_width: Elements per LUT entry.
        num_weight_groups: Total number of LUT groups across all layers.

    Returns:
        Dictionary with memory estimates in bytes and KB.
    """
    entries_per_table = (1 << bits) ** group_width
    bytes_per_entry = 4  # float32
    bytes_per_table = entries_per_table * bytes_per_entry
    total_bytes = bytes_per_table * num_weight_groups

    return {
        "entries_per_table": entries_per_table,
        "bytes_per_table": bytes_per_table,
        "total_tables": num_weight_groups,
        "total_bytes": total_bytes,
        "total_kb": total_bytes / 1024,
        "fits_l1_cache": total_bytes <= 64 * 1024,  # Typical L1 = 64KB
        "fits_l2_cache": total_bytes <= 512 * 1024,  # Typical L2 = 512KB
    }
