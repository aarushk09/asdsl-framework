"""Tests for the fused 4-bit GEMV kernel (AVX2 + NumPy fallback).

Validates correctness against a scalar reference, tests edge cases,
and verifies integration with QuantizedTensor.
"""

import numpy as np
import pytest

from asdsl.kernels.gemv_q4 import (
    gemv_q4,
    gemv_q4_packed,
    gemv_q4_unpacked,
    has_native_kernel,
    _gemv_q4_numpy_packed,
    _gemv_q4_numpy_unpacked,
)
from asdsl.quantization.core import (
    QuantizedTensor,
    _pack_bits,
    _unpack_bits,
    quantize_weights,
    dequantize_weights,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scalar_reference(w_int_2d, x, scales, biases, group_size):
    """Dead-simple scalar reference for verification (no vectorization)."""
    M, K = w_int_2d.shape
    groups_per_row = K // group_size
    y = np.zeros(M, dtype=np.float64)

    for m in range(M):
        for g in range(groups_per_row):
            gidx = m * groups_per_row + g
            k0 = g * group_size
            k1 = k0 + group_size
            w_g = w_int_2d[m, k0:k1].astype(np.float64)
            x_g = x[k0:k1].astype(np.float64)
            y[m] += np.dot(w_g, x_g) * scales[gidx] + biases[gidx] * np.sum(x_g)

    return y.astype(np.float32)


def _make_test_data(rng, M, K, group_size):
    """Generate random 4-bit quantized test data."""
    groups_per_row = K // group_size
    total_groups = M * groups_per_row

    w_int = rng.integers(0, 16, size=(M, K), dtype=np.uint8)
    w_packed = _pack_bits(w_int.ravel(), 4)
    x = rng.standard_normal(K).astype(np.float32)
    scales = rng.uniform(0.001, 0.1, total_groups).astype(np.float32)
    biases = rng.uniform(-0.5, 0.5, total_groups).astype(np.float32)

    return w_int, w_packed, x, scales, biases


# ---------------------------------------------------------------------------
# NumPy fallback correctness
# ---------------------------------------------------------------------------

class TestGemvQ4NumpyPacked:
    """Tests for the vectorized NumPy packed implementation."""

    def test_small_gs16(self):
        rng = np.random.default_rng(42)
        M, K, gs = 4, 32, 16
        w_int, w_packed, x, scales, biases = _make_test_data(rng, M, K, gs)
        expected = _scalar_reference(w_int, x, scales, biases, gs)
        actual = _gemv_q4_numpy_packed(w_packed, x, scales, biases, M, K, gs)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

    def test_medium_gs32(self):
        rng = np.random.default_rng(123)
        M, K, gs = 128, 256, 32
        w_int, w_packed, x, scales, biases = _make_test_data(rng, M, K, gs)
        expected = _scalar_reference(w_int, x, scales, biases, gs)
        actual = _gemv_q4_numpy_packed(w_packed, x, scales, biases, M, K, gs)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

    def test_large_gs128(self):
        rng = np.random.default_rng(99)
        M, K, gs = 64, 512, 128
        w_int, w_packed, x, scales, biases = _make_test_data(rng, M, K, gs)
        expected = _scalar_reference(w_int, x, scales, biases, gs)
        actual = _gemv_q4_numpy_packed(w_packed, x, scales, biases, M, K, gs)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

    def test_single_row(self):
        rng = np.random.default_rng(7)
        M, K, gs = 1, 64, 32
        w_int, w_packed, x, scales, biases = _make_test_data(rng, M, K, gs)
        expected = _scalar_reference(w_int, x, scales, biases, gs)
        actual = _gemv_q4_numpy_packed(w_packed, x, scales, biases, M, K, gs)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

    def test_single_group(self):
        rng = np.random.default_rng(13)
        M, K, gs = 8, 32, 32
        w_int, w_packed, x, scales, biases = _make_test_data(rng, M, K, gs)
        expected = _scalar_reference(w_int, x, scales, biases, gs)
        actual = _gemv_q4_numpy_packed(w_packed, x, scales, biases, M, K, gs)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


class TestGemvQ4NumpyUnpacked:
    """Tests for the vectorized NumPy unpacked implementation."""

    def test_matches_packed(self):
        """Unpacked path must produce the same result as packed path."""
        rng = np.random.default_rng(55)
        M, K, gs = 32, 128, 32
        w_int, w_packed, x, scales, biases = _make_test_data(rng, M, K, gs)

        w_flat = w_int.ravel().astype(np.uint8)
        packed_result = _gemv_q4_numpy_packed(w_packed, x, scales, biases, M, K, gs)
        unpacked_result = _gemv_q4_numpy_unpacked(w_flat, x, scales, biases, M, K, gs)
        np.testing.assert_allclose(unpacked_result, packed_result, rtol=1e-5)


# ---------------------------------------------------------------------------
# Public API dispatch (auto-selects native or numpy)
# ---------------------------------------------------------------------------

class TestGemvQ4Dispatch:
    """Tests for the public API that auto-dispatches to native or numpy."""

    def test_packed_api(self):
        rng = np.random.default_rng(42)
        M, K, gs = 16, 64, 32
        w_int, w_packed, x, scales, biases = _make_test_data(rng, M, K, gs)
        expected = _scalar_reference(w_int, x, scales, biases, gs)
        actual = gemv_q4_packed(w_packed, x, scales, biases, M, K, gs)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

    def test_unpacked_api(self):
        rng = np.random.default_rng(42)
        M, K, gs = 16, 64, 32
        w_int, w_packed, x, scales, biases = _make_test_data(rng, M, K, gs)
        w_flat = w_int.ravel().astype(np.uint8)
        expected = _scalar_reference(w_int, x, scales, biases, gs)
        actual = gemv_q4_unpacked(w_flat, x, scales, biases, M, K, gs)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

    def test_known_values(self):
        """Hand-computable example: all weights = 8, x = 1.0."""
        M, K, gs = 2, 32, 32
        w_int = np.full((M, K), 8, dtype=np.uint8)
        w_packed = _pack_bits(w_int.ravel(), 4)
        x = np.ones(K, dtype=np.float32)
        scales = np.array([0.1, 0.1], dtype=np.float32)
        biases = np.array([-0.75, -0.75], dtype=np.float32)

        # dot(8*ones, ones) = 8*32 = 256; sum(ones) = 32
        # y = 256 * 0.1 + (-0.75) * 32 = 25.6 - 24.0 = 1.6
        expected = np.array([1.6, 1.6], dtype=np.float32)
        actual = gemv_q4_packed(w_packed, x, scales, biases, M, K, gs)
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_zero_weights(self):
        """All-zero weights should produce bias*sum(x) output."""
        M, K, gs = 4, 64, 32
        w_int = np.zeros((M, K), dtype=np.uint8)
        w_packed = _pack_bits(w_int.ravel(), 4)
        rng = np.random.default_rng(77)
        x = rng.standard_normal(K).astype(np.float32)
        groups_per_row = K // gs
        total_groups = M * groups_per_row
        scales = np.ones(total_groups, dtype=np.float32) * 0.05
        biases = np.ones(total_groups, dtype=np.float32) * (-0.3)

        expected = _scalar_reference(w_int, x, scales, biases, gs)
        actual = gemv_q4_packed(w_packed, x, scales, biases, M, K, gs)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

    def test_deterministic(self):
        """Same inputs must produce identical outputs."""
        rng = np.random.default_rng(100)
        M, K, gs = 32, 128, 32
        w_int, w_packed, x, scales, biases = _make_test_data(rng, M, K, gs)
        r1 = gemv_q4_packed(w_packed, x, scales, biases, M, K, gs)
        r2 = gemv_q4_packed(w_packed, x, scales, biases, M, K, gs)
        np.testing.assert_array_equal(r1, r2)


# ---------------------------------------------------------------------------
# Integration with QuantizedTensor
# ---------------------------------------------------------------------------

class TestGemvQ4QuantizedTensor:
    """Tests for the high-level gemv_q4(qtensor, x) interface."""

    def test_matches_dequant_matvec_asymmetric(self):
        """Native GEMV must match the reference dequant-then-multiply path."""
        rng = np.random.default_rng(42)
        W = rng.standard_normal((64, 128)).astype(np.float32)
        x = rng.standard_normal(128).astype(np.float32)

        qt = quantize_weights(W, bits=4, group_size=32, symmetric=False, optimize_clips=True)
        W_deq = dequantize_weights(qt)
        expected = W_deq @ x

        actual = gemv_q4(qt, x)
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)

    def test_matches_dequant_matvec_symmetric(self):
        rng = np.random.default_rng(99)
        W = rng.standard_normal((32, 64)).astype(np.float32)
        x = rng.standard_normal(64).astype(np.float32)

        qt = quantize_weights(W, bits=4, group_size=32, symmetric=True)
        W_deq = dequantize_weights(qt)
        expected = W_deq @ x

        actual = gemv_q4(qt, x)
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)

    def test_rejects_non_4bit(self):
        rng = np.random.default_rng(42)
        W = rng.standard_normal((32, 128)).astype(np.float32)
        qt = quantize_weights(W, bits=8, group_size=128, symmetric=True)
        x = rng.standard_normal(128).astype(np.float32)

        with pytest.raises(ValueError, match="4-bit"):
            gemv_q4(qt, x)


# ---------------------------------------------------------------------------
# Native kernel availability check
# ---------------------------------------------------------------------------

class TestNativeKernel:
    """Tests for native kernel availability and info."""

    def test_has_native_kernel_returns_bool(self):
        result = has_native_kernel()
        assert isinstance(result, bool)

    @pytest.mark.skipif(
        not has_native_kernel(),
        reason="Native GEMV kernel not built",
    )
    def test_native_matches_numpy_packed(self):
        """When native is available, it must match NumPy exactly."""
        rng = np.random.default_rng(42)
        M, K, gs = 64, 256, 32
        w_int, w_packed, x, scales, biases = _make_test_data(rng, M, K, gs)
        numpy_result = _gemv_q4_numpy_packed(w_packed, x, scales, biases, M, K, gs)
        native_result = gemv_q4_packed(w_packed, x, scales, biases, M, K, gs)
        np.testing.assert_allclose(native_result, numpy_result, rtol=1e-5, atol=1e-6)

    @pytest.mark.skipif(
        not has_native_kernel(),
        reason="Native GEMV kernel not built",
    )
    def test_native_matches_numpy_unpacked(self):
        rng = np.random.default_rng(42)
        M, K, gs = 64, 256, 32
        w_int, w_packed, x, scales, biases = _make_test_data(rng, M, K, gs)
        w_flat = w_int.ravel().astype(np.uint8)
        numpy_result = _gemv_q4_numpy_unpacked(w_flat, x, scales, biases, M, K, gs)
        native_result = gemv_q4_unpacked(w_flat, x, scales, biases, M, K, gs)
        np.testing.assert_allclose(native_result, numpy_result, rtol=1e-5, atol=1e-6)

    @pytest.mark.skipif(
        not has_native_kernel(),
        reason="Native GEMV kernel not built",
    )
    def test_native_large_matrix(self):
        """Stress test with realistic dimensions (similar to Phi-4 projections)."""
        rng = np.random.default_rng(42)
        M, K, gs = 512, 1024, 32
        w_int, w_packed, x, scales, biases = _make_test_data(rng, M, K, gs)
        expected = _scalar_reference(w_int, x, scales, biases, gs)
        actual = gemv_q4_packed(w_packed, x, scales, biases, M, K, gs)
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
