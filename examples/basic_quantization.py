"""Example: Basic weight quantization with error analysis.

Demonstrates how to quantize a weight matrix at different bit widths
and compare the resulting compression ratio and reconstruction error.
"""

import numpy as np

from asdsl.quantization.core import (
    quantize_weights,
    dequantize_weights,
    compute_quantization_error,
)


def main() -> None:
    # Simulate a feed-forward projection weight matrix
    np.random.seed(42)
    weights = np.random.randn(3072, 3072).astype(np.float32)
    print(f"Original weights: shape={weights.shape}, size={weights.nbytes / 1e6:.1f} MB\n")

    for bits in [8, 4, 3, 2]:
        qt = quantize_weights(weights, bits=bits, group_size=128)
        recon = dequantize_weights(qt)
        errors = compute_quantization_error(weights, qt)

        print(f"--- {bits}-bit quantization ---")
        print(f"  Compressed size:   {qt.data.nbytes / 1e6:.2f} MB")
        print(f"  Compression ratio: {errors['compression_ratio']:.1f}x")
        print(f"  MSE:               {errors['mse']:.6f}")
        print(f"  SNR:               {errors['snr_db']:.1f} dB")
        print(f"  Max abs error:     {np.max(np.abs(weights - recon)):.4f}")
        print()


if __name__ == "__main__":
    main()
