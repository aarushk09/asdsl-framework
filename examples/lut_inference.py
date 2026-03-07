"""Example: LUT-based matrix-vector multiplication.

Demonstrates building lookup tables for quantized weights and using
them for fast matrix-vector products.
"""

import time

import numpy as np

from asdsl.quantization.core import quantize_weights
from asdsl.lut.engine import (
    build_lut_tables_for_layer,
    lut_matvec,
    lut_matvec_batched,
    estimate_lut_memory,
)


def main() -> None:
    np.random.seed(42)

    out_features, in_features = 3072, 3072
    bits = 4
    group_size = 128

    print(f"Matrix: {out_features}x{in_features}, {bits}-bit, group_size={group_size}")
    print()

    # Quantize weights
    weights = np.random.randn(out_features, in_features).astype(np.float32)
    qt = quantize_weights(weights, bits=bits, group_size=group_size)

    # Build LUT tables
    t0 = time.perf_counter()
    tables = build_lut_tables_for_layer(qt)
    build_time = time.perf_counter() - t0
    print(f"Built {len(tables)} lookup tables in {build_time*1000:.1f} ms")

    # Memory estimate
    mem = estimate_lut_memory(out_features, in_features, bits, group_size)
    print(f"LUT memory: {mem['total_bytes'] / 1024:.1f} KB")
    print(f"  Fits L1 cache: {'Yes' if mem['fits_l1'] else 'No'}")
    print(f"  Fits L2 cache: {'Yes' if mem['fits_l2'] else 'No'}")
    print()

    # LUT-based matvec
    x = np.random.randn(in_features).astype(np.float32)

    t0 = time.perf_counter()
    y_lut = lut_matvec(tables, x, out_features=out_features, group_size=group_size)
    lut_time = time.perf_counter() - t0

    # Reference: standard float matmul
    t0 = time.perf_counter()
    y_ref = weights @ x
    ref_time = time.perf_counter() - t0

    error = np.mean((y_lut - y_ref) ** 2)
    print(f"LUT matvec time:      {lut_time*1000:.2f} ms")
    print(f"Reference matmul time: {ref_time*1000:.2f} ms")
    print(f"MSE vs reference:      {error:.6f}")
    print(f"Cosine similarity:     {np.dot(y_lut, y_ref) / (np.linalg.norm(y_lut) * np.linalg.norm(y_ref)):.6f}")
    print()

    # Batched verification
    y_batched = lut_matvec_batched(tables, x, out_features=out_features, group_size=group_size)
    match = np.allclose(y_lut, y_batched, atol=1e-5)
    print(f"Batched vs single match: {match}")


if __name__ == "__main__":
    main()
