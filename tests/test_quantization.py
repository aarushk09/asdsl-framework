"""Tests for the core quantization engine."""

import numpy as np
import pytest

from asdsl.quantization.core import (
    QuantizedTensor,
    _pack_bits,
    _unpack_bits,
    compute_quantization_error,
    compute_scale_zero,
    dequantize_weights,
    quantize_weights,
)


class TestBitPacking:
    """Tests for sub-byte bit packing and unpacking."""

    def test_pack_unpack_2bit_roundtrip(self):
        """2-bit values should survive pack → unpack roundtrip."""
        values = np.array([0, 1, 2, 3, 1, 0, 3, 2], dtype=np.uint8)
        packed = _pack_bits(values, bits=2)
        unpacked = _unpack_bits(packed, bits=2)
        np.testing.assert_array_equal(unpacked[: len(values)], values)

    def test_pack_unpack_4bit_roundtrip(self):
        """4-bit values should survive pack → unpack roundtrip."""
        values = np.array([0, 15, 7, 8, 1, 14, 3, 12], dtype=np.uint8)
        packed = _pack_bits(values, bits=4)
        unpacked = _unpack_bits(packed, bits=4)
        np.testing.assert_array_equal(unpacked[: len(values)], values)

    def test_pack_unpack_8bit_passthrough(self):
        """8-bit packing should be identity."""
        values = np.array([0, 128, 255, 1, 64], dtype=np.uint8)
        packed = _pack_bits(values, bits=8)
        unpacked = _unpack_bits(packed, bits=8)
        np.testing.assert_array_equal(unpacked[: len(values)], values)

    def test_pack_unpack_3bit_roundtrip(self):
        """3-bit values should survive pack → unpack roundtrip."""
        values = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.uint8)
        packed = _pack_bits(values, bits=3)
        unpacked = _unpack_bits(packed, bits=3)
        np.testing.assert_array_equal(unpacked[: len(values)], values)

    def test_2bit_packing_density(self):
        """2-bit packing should achieve 4:1 compression."""
        values = np.zeros(256, dtype=np.uint8)
        packed = _pack_bits(values, bits=2)
        assert len(packed) == 64  # 256 values / 4 per byte

    def test_4bit_packing_density(self):
        """4-bit packing should achieve 2:1 compression."""
        values = np.zeros(256, dtype=np.uint8)
        packed = _pack_bits(values, bits=4)
        assert len(packed) == 128  # 256 values / 2 per byte


class TestScaleComputation:
    """Tests for quantization scale and zero-point computation."""

    def test_symmetric_scale(self):
        """Symmetric scale should be based on max absolute value."""
        weights = np.array([[1.0, -2.0, 0.5, 1.5]], dtype=np.float32)
        scales, zeros = compute_scale_zero(weights, bits=4, symmetric=True)
        assert zeros is None
        assert scales.shape == (1, 1)
        # Scale = max_abs / (qmax/2) = 2.0 / 7.5
        expected_scale = 2.0 / 7.5
        np.testing.assert_allclose(float(scales[0, 0]), expected_scale, rtol=0.01)

    def test_asymmetric_scale_and_zero(self):
        """Asymmetric should compute both scale and zero-point."""
        weights = np.array([[1.0, 3.0, 5.0, 7.0]], dtype=np.float32)
        scales, zeros = compute_scale_zero(weights, bits=4, symmetric=False)
        assert zeros is not None
        assert scales.shape == (1, 1)
        assert zeros.shape == (1, 1)


class TestQuantization:
    """Tests for full quantize → dequantize pipeline."""

    @pytest.mark.parametrize("bits", [2, 4, 8])
    def test_quantize_dequantize_shape_preserved(self, bits):
        """Output shape should match input shape after round-trip."""
        weights = np.random.randn(64, 128).astype(np.float32)
        qtensor = quantize_weights(weights, bits=bits, group_size=32)
        reconstructed = dequantize_weights(qtensor)
        assert reconstructed.shape == weights.shape

    @pytest.mark.parametrize("bits", [2, 4, 8])
    def test_quantization_reduces_memory(self, bits):
        """Quantized tensor should use less memory than FP32."""
        weights = np.random.randn(256, 256).astype(np.float32)
        qtensor = quantize_weights(weights, bits=bits, group_size=128)
        assert qtensor.memory_bytes < weights.nbytes

    def test_8bit_high_fidelity(self):
        """8-bit quantization should have very low error."""
        weights = np.random.randn(64, 64).astype(np.float32)
        qtensor = quantize_weights(weights, bits=8, group_size=64)
        metrics = compute_quantization_error(weights, qtensor)
        assert metrics["snr_db"] > 30  # High SNR = low error

    def test_2bit_has_more_error_than_8bit(self):
        """2-bit should have higher error than 8-bit."""
        weights = np.random.randn(64, 128).astype(np.float32)
        q2 = quantize_weights(weights, bits=2, group_size=128)
        q8 = quantize_weights(weights, bits=8, group_size=128)
        err2 = compute_quantization_error(weights, q2)
        err8 = compute_quantization_error(weights, q8)
        assert err2["mse"] > err8["mse"]

    def test_quantized_tensor_metadata(self):
        """QuantizedTensor should track correct metadata."""
        weights = np.random.randn(32, 64).astype(np.float32)
        qtensor = quantize_weights(weights, bits=4, group_size=64)
        assert qtensor.bits == 4
        assert qtensor.group_size == 64
        assert qtensor.shape == (32, 64)
        assert qtensor.numel == 32 * 64
        assert qtensor.is_symmetric is True
