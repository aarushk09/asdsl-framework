"""Core quantization primitives for uniform and mixed-precision weight compression.

Supports 2-bit through 8-bit symmetric and asymmetric quantization with
group-wise granularity. Provides the base layer that salience-driven
bit allocation builds upon.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class QuantizedTensor:
    """A quantized weight tensor with metadata for reconstruction.

    Attributes:
        data: Packed integer data (uint8 backing store, sub-byte packed).
        scales: Per-group scale factors (float16).
        zeros: Per-group zero points (float16 for asymmetric, None for symmetric).
        bits: Bit-width used for this tensor.
        group_size: Number of elements per quantization group.
        shape: Original tensor shape before quantization.
        is_symmetric: Whether symmetric quantization was used.
    """

    data: np.ndarray
    scales: np.ndarray
    zeros: np.ndarray | None
    bits: int
    group_size: int
    shape: tuple[int, ...]
    is_symmetric: bool = True

    @property
    def numel(self) -> int:
        """Number of elements in the original tensor."""
        result = 1
        for s in self.shape:
            result *= s
        return result

    @property
    def memory_bytes(self) -> int:
        """Approximate memory footprint in bytes."""
        data_bytes = self.data.nbytes
        scale_bytes = self.scales.nbytes
        zero_bytes = self.zeros.nbytes if self.zeros is not None else 0
        return data_bytes + scale_bytes + zero_bytes


def compute_scale_zero(
    weights: np.ndarray,
    bits: int,
    symmetric: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Compute quantization scale and zero-point for a group of weights.

    Args:
        weights: Float weight values, shape (num_groups, group_size).
        bits: Target bit-width.
        symmetric: If True, uses symmetric quantization (zero-centered).

    Returns:
        Tuple of (scales, zeros). zeros is None for symmetric quantization.
    """
    qmin = 0
    qmax = (1 << bits) - 1

    if symmetric:
        # Symmetric: scale based on max absolute value
        abs_max = np.maximum(np.abs(weights).max(axis=-1, keepdims=True), 1e-10)
        half_range = (qmax - qmin) / 2
        scales = abs_max / half_range
        return scales.astype(np.float16), None
    else:
        # Asymmetric: use full range [min, max]
        w_min = weights.min(axis=-1, keepdims=True)
        w_max = weights.max(axis=-1, keepdims=True)
        w_range = np.maximum(w_max - w_min, 1e-10)
        scales = w_range / qmax
        zeros = -w_min / scales
        return scales.astype(np.float16), zeros.astype(np.float16)


def _find_optimal_scales(
    grouped: np.ndarray,
    bits: int,
    symmetric: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Find per-group scales that minimize reconstruction MSE via grid search.

    Tests several clipping ratios and picks the best per group.
    Fully vectorized across groups — loops only over the small ratio grid.
    MSE is evaluated at float16 precision (matching storage) so the chosen
    ratio is truly optimal after quantization parameter rounding.
    """
    qmin = 0
    qmax = (1 << bits) - 1
    n_groups = grouped.shape[0]

    ratios = np.array([0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 1.0],
                      dtype=np.float32)
    best_mse = np.full(n_groups, np.inf, dtype=np.float32)
    best_scales = np.ones((n_groups, 1), dtype=np.float32) * 1e-10
    best_zeros = None if symmetric else np.zeros((n_groups, 1), dtype=np.float32)

    if symmetric:
        abs_max = np.maximum(np.abs(grouped).max(axis=1, keepdims=True), 1e-10)
        half_range = (qmax - qmin) / 2.0
        for r in ratios:
            clip_val = abs_max * r
            # Round scale to float16 to match actual storage precision
            scale = np.maximum(clip_val / half_range, 1e-10)
            scale_f16 = scale.astype(np.float16).astype(np.float32)
            quantized = np.clip(np.round(grouped / scale_f16 + half_range), qmin, qmax)
            dequantized = (quantized - half_range) * scale_f16
            mse = np.mean((grouped - dequantized) ** 2, axis=1)
            improved = mse < best_mse
            best_mse = np.where(improved, mse, best_mse)
            best_scales = np.where(improved[:, None], scale, best_scales)
    else:
        w_min = grouped.min(axis=1, keepdims=True)
        w_max = grouped.max(axis=1, keepdims=True)
        w_range = np.maximum(w_max - w_min, 1e-10)
        for r in ratios:
            margin = w_range * (1.0 - r) / 2.0
            clip_min = w_min + margin
            clip_max = w_max - margin
            clip_range = np.maximum(clip_max - clip_min, 1e-10)
            scale = clip_range / qmax
            zero = np.clip(-clip_min / np.maximum(scale, 1e-10), 0, qmax)
            # Evaluate MSE at float16 precision (matching storage)
            scale_f16 = scale.astype(np.float16).astype(np.float32)
            zero_f16 = zero.astype(np.float16).astype(np.float32)
            quantized = np.clip(np.round(grouped / scale_f16 + zero_f16), qmin, qmax)
            dequantized = (quantized - zero_f16) * scale_f16
            mse = np.mean((grouped - dequantized) ** 2, axis=1)
            improved = mse < best_mse
            best_mse = np.where(improved, mse, best_mse)
            best_scales = np.where(improved[:, None], scale, best_scales)
            best_zeros = np.where(improved[:, None], zero, best_zeros)

    if best_zeros is not None:
        return best_scales.astype(np.float16), best_zeros.astype(np.float16)
    return best_scales.astype(np.float16), None


def quantize_weights(
    weights: torch.Tensor | np.ndarray,
    bits: int = 4,
    group_size: int = 128,
    symmetric: bool = True,
    optimize_clips: bool = False,
) -> QuantizedTensor:
    """Quantize a weight tensor to the specified bit-width with group-wise granularity.

    Args:
        weights: Full-precision weight tensor. Shape (out_features, in_features).
        bits: Target quantization bit-width (2, 3, 4, or 8).
        group_size: Number of contiguous elements sharing scale/zero-point.
        symmetric: Whether to use symmetric quantization.
        optimize_clips: If True, search for MSE-optimal clipping ratio per group.

    Returns:
        A QuantizedTensor containing packed data, scales, and metadata.
    """
    if isinstance(weights, torch.Tensor):
        weights_np = weights.detach().cpu().float().numpy()
    else:
        weights_np = weights.astype(np.float32)

    original_shape = weights_np.shape
    flat = weights_np.reshape(-1)

    # Pad to group_size boundary
    remainder = flat.shape[0] % group_size
    if remainder != 0:
        pad_size = group_size - remainder
        flat = np.concatenate([flat, np.zeros(pad_size, dtype=np.float32)])

    num_groups = flat.shape[0] // group_size
    grouped = flat.reshape(num_groups, group_size)

    # Compute scales and zeros
    if optimize_clips:
        scales, zeros = _find_optimal_scales(grouped, bits, symmetric)
    else:
        scales, zeros = compute_scale_zero(grouped, bits, symmetric)

    # Quantize
    qmin = 0
    qmax = (1 << bits) - 1

    if symmetric:
        half_range = (qmax - qmin) / 2
        offset = half_range
        quantized = np.round(grouped / scales.astype(np.float32) + offset)
    else:
        quantized = np.round(grouped / scales.astype(np.float32) + zeros.astype(np.float32))

    quantized = np.clip(quantized, qmin, qmax).astype(np.uint8)

    # Pack sub-byte values into uint8 array
    packed = _pack_bits(quantized.reshape(-1), bits)

    return QuantizedTensor(
        data=packed,
        scales=scales.reshape(-1),
        zeros=zeros.reshape(-1) if zeros is not None else None,
        bits=bits,
        group_size=group_size,
        shape=original_shape,
        is_symmetric=symmetric,
    )


def dequantize_weights(qtensor: QuantizedTensor) -> np.ndarray:
    """Dequantize a QuantizedTensor back to float32.

    This is the traditional dequantization path — used for validation
    and accuracy testing. The LUT engine bypasses this entirely at runtime.

    Args:
        qtensor: A previously quantized tensor.

    Returns:
        Reconstructed float32 weight array in the original shape.
    """
    bits = qtensor.bits
    group_size = qtensor.group_size

    # Unpack
    unpacked = _unpack_bits(qtensor.data, bits)

    # Figure out total padded length
    numel = qtensor.numel
    remainder = numel % group_size
    padded_len = numel + (group_size - remainder if remainder else 0)
    unpacked = unpacked[:padded_len]

    num_groups = padded_len // group_size
    grouped = unpacked.reshape(num_groups, group_size).astype(np.float32)

    scales = qtensor.scales.reshape(num_groups, 1).astype(np.float32)

    qmax = (1 << bits) - 1

    if qtensor.is_symmetric:
        half_range = qmax / 2
        dequantized = (grouped - half_range) * scales
    else:
        zeros = qtensor.zeros.reshape(num_groups, 1).astype(np.float32)
        dequantized = (grouped - zeros) * scales

    return dequantized.reshape(-1)[:numel].reshape(qtensor.shape)


def _pack_bits(data: np.ndarray, bits: int) -> np.ndarray:
    """Pack quantized integers into a compact uint8 array.

    For 2-bit: 4 values per byte.
    For 4-bit: 2 values per byte.
    For 8-bit: 1 value per byte (no packing needed).
    For 3-bit: Uses a 3-byte group for 8 values.
    """
    if bits == 8:
        return data.astype(np.uint8)

    if bits == 4:
        # Pack pairs into bytes: low nibble first
        if len(data) % 2 != 0:
            data = np.append(data, np.uint8(0))
        low = data[0::2] & 0x0F
        high = (data[1::2] & 0x0F) << 4
        return (low | high).astype(np.uint8)

    if bits == 2:
        # Pack 4 values per byte
        pad_needed = (4 - len(data) % 4) % 4
        if pad_needed:
            data = np.concatenate([data, np.zeros(pad_needed, dtype=np.uint8)])
        b0 = data[0::4] & 0x03
        b1 = (data[1::4] & 0x03) << 2
        b2 = (data[2::4] & 0x03) << 4
        b3 = (data[3::4] & 0x03) << 6
        return (b0 | b1 | b2 | b3).astype(np.uint8)

    if bits == 3:
        # 10-in-32 packing: 10 × 3-bit values → one uint32 (30/32 bits used).
        # 2 MSBs are wasted for alignment, giving fast 32-bit aligned reads.
        pad_needed = (10 - len(data) % 10) % 10
        if pad_needed:
            data = np.concatenate([data, np.zeros(pad_needed, dtype=np.uint8)])
        data = data.astype(np.uint32).reshape(-1, 10)
        packed = np.zeros(data.shape[0], dtype=np.uint32)
        for j in range(10):
            packed |= (data[:, j] & 0x07) << (j * 3)
        return packed.view(np.uint8)

    raise ValueError(f"Unsupported bit-width: {bits}")


def _unpack_bits(packed: np.ndarray, bits: int) -> np.ndarray:
    """Unpack a packed uint8 array back to individual quantized values."""
    if bits == 8:
        return packed.astype(np.uint8)

    if bits == 4:
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        return np.stack([low, high], axis=-1).reshape(-1).astype(np.uint8)

    if bits == 2:
        b0 = packed & 0x03
        b1 = (packed >> 2) & 0x03
        b2 = (packed >> 4) & 0x03
        b3 = (packed >> 6) & 0x03
        return np.stack([b0, b1, b2, b3], axis=-1).reshape(-1).astype(np.uint8)

    if bits == 3:
        # 10-in-32 unpacking: every 4 bytes → one uint32 → 10 × 3-bit values.
        words = packed.view(np.uint32)
        out = np.empty((len(words), 10), dtype=np.uint8)
        for j in range(10):
            out[:, j] = ((words >> (j * 3)) & 0x07).astype(np.uint8)
        return out.reshape(-1)

    raise ValueError(f"Unsupported bit-width: {bits}")


def compute_quantization_error(
    original: torch.Tensor | np.ndarray,
    qtensor: QuantizedTensor,
) -> dict[str, float]:
    """Compute quantization error metrics between original and quantized weights.

    Returns:
        Dictionary with MSE, MAE, max_error, and SNR metrics.
    """
    if isinstance(original, torch.Tensor):
        original_np = original.detach().cpu().float().numpy()
    else:
        original_np = original.astype(np.float32)

    reconstructed = dequantize_weights(qtensor)
    error = original_np - reconstructed

    mse = float(np.mean(error**2))
    mae = float(np.mean(np.abs(error)))
    max_err = float(np.max(np.abs(error)))

    signal_power = float(np.mean(original_np**2))
    snr = 10 * math.log10(signal_power / max(mse, 1e-20))

    return {
        "mse": mse,
        "mae": mae,
        "max_error": max_err,
        "snr_db": snr,
        "compression_ratio": (original_np.nbytes) / max(qtensor.memory_bytes, 1),
    }
