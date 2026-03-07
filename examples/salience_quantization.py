"""Example: Salience-driven mixed-precision quantization.

Shows how to compute weight salience scores and use them for
intelligent bit allocation across weight groups.
"""

import numpy as np

from asdsl.quantization.core import quantize_weights, compute_quantization_error
from asdsl.quantization.salience import (
    compute_gradient_salience,
    allocate_bits_by_salience,
)


def main() -> None:
    np.random.seed(42)

    # Simulate weights and calibration gradients
    shape = (3072, 3072)
    weights = np.random.randn(*shape).astype(np.float32)
    gradients = np.random.randn(*shape).astype(np.float32) * 0.01

    # Compute salience scores
    salience = compute_gradient_salience(weights, gradients, group_size=128)
    print(f"Salience scores: {salience.scores.shape[0]} groups")
    print(f"  Min salience:  {salience.scores.min():.4f}")
    print(f"  Max salience:  {salience.scores.max():.4f}")
    print(f"  Mean salience: {salience.scores.mean():.4f}")
    print()

    # Allocate bits based on salience
    allocation = allocate_bits_by_salience(salience, target_avg_bits=3.5)
    unique, counts = np.unique(allocation.bits_per_group, return_counts=True)
    print(f"Bit allocation (target avg = 3.5):")
    print(f"  Average bits: {allocation.avg_bits:.3f}")
    for b, c in zip(unique, counts):
        pct = c / len(allocation.bits_per_group) * 100
        print(f"  {b}-bit: {c} groups ({pct:.1f}%)")
    print()

    # Compare uniform vs. salience-driven quantization
    print("--- Comparison: Uniform vs. Salience-Driven ---")

    # Uniform 4-bit (same average)
    qt_uniform = quantize_weights(weights, bits=4, group_size=128)
    err_uniform = compute_quantization_error(weights, qt_uniform)
    print(f"Uniform 4-bit:    MSE={err_uniform['mse']:.6f}, SNR={err_uniform['snr_db']:.1f} dB")

    # Salience-driven: quantize high-salience groups at 4/8-bit, low at 2/3-bit
    # (Simplified demo — full pipeline uses quantize_model_mixed_precision)
    mse_mixed = 0.0
    n_groups = 0
    for i, bits in enumerate(allocation.bits_per_group):
        start = i * 128
        end = min(start + 128, weights.shape[1])
        for row_start in range(0, weights.shape[0], 128):
            row_end = min(row_start + 128, weights.shape[0])
            group = weights[row_start:row_end, start:end].flatten()
            if len(group) == 0:
                continue
            qt = quantize_weights(group.reshape(1, -1), bits=int(bits), group_size=len(group))
            from asdsl.quantization.core import dequantize_weights
            recon = dequantize_weights(qt).flatten()
            mse_mixed += np.mean((group - recon) ** 2) * len(group)
            n_groups += len(group)

    mse_mixed /= n_groups
    snr_mixed = 10 * np.log10(np.mean(weights ** 2) / (mse_mixed + 1e-10))
    print(f"Salience-driven:  MSE={mse_mixed:.6f}, SNR={snr_mixed:.1f} dB")
    print(f"  → {'Better' if mse_mixed < err_uniform['mse'] else 'Worse'} quality at similar avg bits")


if __name__ == "__main__":
    main()
