"""Benchmark: 3-bit 10-in-32 packing throughput vs other bit widths.

Verifies that the aligned uint32 packing brings 3-bit quantization
throughput in line with 4-bit (goal: matching MB/s).
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asdsl.quantization.core import quantize_weights, dequantize_weights


def bench(shapes, bits_list, group_size=128, repeats=5):
    print("=" * 78)
    print("  3-bit Optimised Packing Throughput Benchmark  (10-in-32 uint32)")
    print("=" * 78)

    # --- Quantization throughput ---
    print("\n[1] Quantization Throughput")
    print(f"{'Shape':>18s}  {'Bits':>4s}  {'Time (ms)':>10s}  {'MB/s':>10s}  {'Comp.Ratio':>11s}")
    print("-" * 68)
    for shape in shapes:
        weights = np.random.randn(*shape).astype(np.float32)
        size_mb = weights.nbytes / (1024 * 1024)
        for bits in bits_list:
            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                qt = quantize_weights(weights, bits=bits, group_size=group_size)
                t1 = time.perf_counter()
                times.append(t1 - t0)
            median = sorted(times)[len(times) // 2]
            mbps = size_mb / median if median > 0 else 0
            cr = weights.nbytes / max(qt.memory_bytes, 1)
            print(f"{str(shape):>18s}  {bits:>4d}  {median*1000:>10.2f}  {mbps:>10.1f}  {cr:>10.2f}x")

    # --- Dequantization throughput ---
    print(f"\n[2] Dequantization Throughput (shape (3072, 8192))")
    print(f"{'Bits':>4s}  {'Time (ms)':>10s}  {'MB/s (out)':>12s}")
    print("-" * 32)
    shape = (3072, 8192)
    weights = np.random.randn(*shape).astype(np.float32)
    out_mb = weights.nbytes / (1024 * 1024)
    for bits in bits_list:
        qt = quantize_weights(weights, bits=bits, group_size=group_size)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = dequantize_weights(qt)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        median = sorted(times)[len(times) // 2]
        mbps = out_mb / median if median > 0 else 0
        print(f"{bits:>4d}  {median*1000:>10.2f}  {mbps:>12.1f}")

    print("\n" + "=" * 78)
    print("Done.")


if __name__ == "__main__":
    shapes = [(768, 768), (3072, 3072), (3072, 8192)]
    bits_list = [2, 3, 4, 8]
    bench(shapes, bits_list)
