"""Tests for SIMD kernel backends."""

import numpy as np
import pytest

from asdsl.kernels.simd import (
    KernelBackend,
    lut_shuffle_avx2,
    lut_tbl_neon,
    fma_vnni_int8,
    prefill_matmul_int8,
    select_backend,
)


class TestKernelBackendSelection:
    """Tests for automatic backend selection."""

    def test_select_backend_returns_valid(self):
        backend = select_backend()
        assert isinstance(backend, KernelBackend)

    def test_fallback_is_always_available(self):
        """SCALAR fallback should always be available."""
        assert KernelBackend.SCALAR is not None


class TestLUTShuffleAVX2:
    """Tests for the AVX2 VPSHUFB emulation kernel."""

    def test_output_shape(self):
        lut = np.random.randn(16).astype(np.float32)
        indices = np.array([0, 3, 7, 15, 1, 2], dtype=np.uint8)
        result = lut_shuffle_avx2(lut, indices)
        assert result.shape == indices.shape

    def test_correct_lookup(self):
        lut = np.arange(16, dtype=np.float32) * 10
        indices = np.array([0, 5, 15], dtype=np.uint8)
        result = lut_shuffle_avx2(lut, indices)
        np.testing.assert_array_equal(result, [0.0, 50.0, 150.0])

    def test_rejects_out_of_range_indices(self):
        """Indices must be in [0, 15] for 4-bit LUT."""
        lut = np.zeros(16, dtype=np.float32)
        indices = np.array([16], dtype=np.uint8)  # Out of range
        with pytest.raises((IndexError, ValueError)):
            lut_shuffle_avx2(lut, indices)


class TestLUTNEON:
    """Tests for the ARM NEON TBL emulation kernel."""

    def test_output_shape(self):
        lut = np.random.randn(16).astype(np.float32)
        indices = np.array([0, 1, 2, 3], dtype=np.uint8)
        result = lut_tbl_neon(lut, indices)
        assert result.shape == (4,)

    def test_correct_lookup(self):
        lut = np.arange(16, dtype=np.float32) + 1
        indices = np.array([0, 7, 15], dtype=np.uint8)
        result = lut_tbl_neon(lut, indices)
        np.testing.assert_array_equal(result, [1.0, 8.0, 16.0])


class TestVNNIInt8:
    """Tests for INT8 FMA (VPDPBUSD emulation)."""

    def test_output_shape(self):
        a = np.random.randint(-128, 127, size=(4, 8), dtype=np.int8)
        b = np.random.randint(-128, 127, size=(8, 6), dtype=np.int8)
        result = fma_vnni_int8(a, b)
        assert result.shape == (4, 6)

    def test_matches_reference(self):
        """INT8 matmul should match float reference within rounding."""
        a = np.array([[1, 2], [3, 4]], dtype=np.int8)
        b = np.array([[5, 6], [7, 8]], dtype=np.int8)
        result = fma_vnni_int8(a, b)
        expected = a.astype(np.int32) @ b.astype(np.int32)
        np.testing.assert_array_equal(result, expected)


class TestPrefillMatmul:
    """Tests for the prefill-phase INT8 matmul."""

    def test_output_shape(self):
        a = np.random.randint(-128, 127, size=(16, 32), dtype=np.int8)
        b = np.random.randint(-128, 127, size=(32, 64), dtype=np.int8)
        result = prefill_matmul_int8(a, b)
        assert result.shape == (16, 64)

    def test_deterministic(self):
        """Same inputs should produce same outputs."""
        a = np.random.randint(-128, 127, size=(8, 16), dtype=np.int8)
        b = np.random.randint(-128, 127, size=(16, 4), dtype=np.int8)
        r1 = prefill_matmul_int8(a, b)
        r2 = prefill_matmul_int8(a, b)
        np.testing.assert_array_equal(r1, r2)
