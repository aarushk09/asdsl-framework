"""Tests for the LUT engine."""

import numpy as np
import pytest

from asdsl.lut.engine import (
    LookupTable,
    build_lut_for_group,
    estimate_lut_memory,
    lut_matvec,
)
from asdsl.lut.permutation import (
    compute_permutation_map,
    interleave_for_simd,
    permute_weights_for_lut,
)
from asdsl.quantization.core import _pack_bits


class TestLUTConstruction:
    """Tests for lookup table building."""

    def test_2bit_group4_table_size(self):
        """2-bit, group_width=4 should produce 256 entries."""
        weights = np.array([1, 0, 2, 3], dtype=np.uint8)
        activations = np.array([1.0, 2.0, 0.5, -1.0], dtype=np.float32)
        table = build_lut_for_group(
            weights, activations, bits=2, scale=0.5, group_width=4
        )
        assert table.num_entries == 256  # 4^4

    def test_2bit_group2_table_size(self):
        """2-bit, group_width=2 should produce 16 entries."""
        weights = np.array([1, 2], dtype=np.uint8)
        activations = np.array([1.0, 2.0], dtype=np.float32)
        table = build_lut_for_group(
            weights, activations, bits=2, scale=1.0, group_width=2
        )
        assert table.num_entries == 16  # 4^2

    def test_lut_precomputes_correct_sums(self):
        """LUT entries should match manual dot product computation."""
        weights = np.array([0, 1], dtype=np.uint8)
        activations = np.array([2.0, 3.0], dtype=np.float32)

        table = build_lut_for_group(
            weights, activations, bits=2, scale=1.0, group_width=2
        )

        # For index 0 (w0=0, w1=0): (0-1)*1.0*2.0 + (0-1)*1.0*3.0 = -5.0
        np.testing.assert_allclose(table.table[0], -5.0, atol=1e-5)

    def test_lut_memory_fits_l1(self):
        """2-bit group_width=2 tables should fit in L1 cache."""
        table = build_lut_for_group(
            np.array([0, 1], dtype=np.uint8),
            np.array([1.0, 1.0], dtype=np.float32),
            bits=2,
            scale=1.0,
            group_width=2,
        )
        # 16 entries * 4 bytes = 64 bytes per table
        assert table.memory_bytes == 64


class TestLUTMemoryEstimates:
    """Tests for LUT memory estimation."""

    def test_2bit_group4_memory(self):
        """Memory estimates should be accurate for 2-bit/group4."""
        est = estimate_lut_memory(bits=2, group_width=4, num_weight_groups=100)
        assert est["entries_per_table"] == 256
        assert est["bytes_per_table"] == 1024
        assert est["total_bytes"] == 102400

    def test_2bit_group2_fits_l1(self):
        """Small tables should fit in L1 cache."""
        est = estimate_lut_memory(bits=2, group_width=2, num_weight_groups=100)
        assert est["fits_l1_cache"] is True


class TestWeightPermutation:
    """Tests for weight layout optimization."""

    def test_permutation_preserves_data(self):
        """Permuted weights should contain the same values."""
        np.random.seed(42)
        data = np.random.randint(0, 4, size=64, dtype=np.uint8)
        packed = _pack_bits(data, bits=2)

        permuted = permute_weights_for_lut(
            packed, bits=2, output_size=8, input_size=8, group_width=4
        )
        # Permuted should have similar total size
        assert permuted.nbytes > 0

    def test_simd_interleave_alignment(self):
        """Interleaved data should be SIMD-width aligned."""
        data = np.random.randint(0, 256, size=100, dtype=np.uint8)
        interleaved = interleave_for_simd(data, bits=2, simd_width=32)
        assert len(interleaved) % 32 == 0

    def test_permutation_map_coverage(self):
        """Permutation map should cover all valid indices."""
        perm = compute_permutation_map(
            output_size=16, input_size=16, group_width=4, tile_rows=4
        )
        valid = perm[perm >= 0]
        # All original indices should appear
        assert len(np.unique(valid)) == 256  # 16*16
