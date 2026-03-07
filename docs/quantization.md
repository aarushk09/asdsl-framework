# Quantization Guide

This guide explains how to use the ASDSL mixed-precision quantization pipeline to
compress a language model for efficient CPU inference.

## Quick Start

```python
import numpy as np
from asdsl.quantization.core import quantize_weights, dequantize_weights
from asdsl.quantization.pipeline import quantize_model_mixed_precision

# Basic uniform quantization
weights = np.random.randn(3072, 3072).astype(np.float32)
qt = quantize_weights(weights, bits=4, group_size=128)
reconstructed = dequantize_weights(qt)

print(f"Original:    {weights.nbytes / 1e6:.1f} MB")
print(f"Compressed:  {qt.data.nbytes / 1e6:.1f} MB")
print(f"Ratio:       {weights.nbytes / qt.data.nbytes:.1f}x")
```

## Bit Width Options

| Bits | Compression | Quality Impact | Use Case |
|------|-------------|---------------|----------|
| 8    | 4x          | Negligible    | Baseline, high-fidelity layers |
| 4    | 8x          | Minimal       | Default for most layers |
| 3    | ~10.7x      | Small         | Non-critical attention projections |
| 2    | 16x         | Moderate      | Low-salience feed-forward groups |

## Salience-Driven Allocation

Instead of uniform bit widths, ASDSL analyzes weight importance:

```python
from asdsl.quantization.salience import (
    compute_gradient_salience,
    allocate_bits_by_salience,
)

# Requires calibration data (gradients from a few samples)
salience = compute_gradient_salience(weights, gradients, group_size=128)
allocation = allocate_bits_by_salience(salience, target_avg_bits=3.5)

print(f"Average bits: {allocation.avg_bits:.2f}")
print(f"Bit distribution: {dict(zip(*np.unique(allocation.bits_per_group, return_counts=True)))}")
```

### How Salience Scoring Works

Each weight group receives a score:
$$S_g = \|W_g \odot \nabla_W \mathcal{L}\|_F$$

Groups with high salience (important for model output) receive more bits.
Groups with low salience can be aggressively quantized.

## Group Size

The `group_size` parameter controls the granularity of quantization:

- **128** (default): Good balance of compression and quality. Each 128-element
  group gets its own scale factor.
- **64**: Better quality, slightly less compression.
- **256**: More compression, may lose quality on heterogeneous weight distributions.

## Error Analysis

```python
from asdsl.quantization.core import compute_quantization_error

errors = compute_quantization_error(weights, qt)
print(f"MSE:              {errors['mse']:.6f}")
print(f"MAE:              {errors['mae']:.6f}")
print(f"SNR (dB):         {errors['snr_db']:.1f}")
print(f"Compression Ratio: {errors['compression_ratio']:.1f}x")
```

## Full Pipeline

The `quantize_model_mixed_precision()` function handles an entire model:

```python
from asdsl.quantization.pipeline import quantize_model_mixed_precision
from asdsl.config import PHI3_MINI_CONFIG, QuantizationConfig

config = QuantizationConfig(
    target_avg_bits=3.5,
    group_size=128,
    bit_tiers=[2, 3, 4, 8],
)

# model_weights: dict mapping layer names to numpy arrays
quantized_model = quantize_model_mixed_precision(
    model_weights=model_weights,
    model_config=PHI3_MINI_CONFIG,
    quant_config=config,
    calibration_gradients=gradients,  # Optional, enables salience
)
```

## Saving and Loading

```python
from asdsl.quantization.pipeline import save_quantized_model

save_quantized_model(quantized_model, "model_quantized.npz")
# Load with: np.load("model_quantized.npz", allow_pickle=True)
```
