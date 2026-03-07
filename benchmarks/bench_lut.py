"""LUT engine benchmarks: table build time, lookup throughput, memory footprint."""

import time
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asdsl.lut.engine import (
    build_lut_for_group,
    build_lut_tables_for_layer,
    lut_matvec,
    estimate_lut_memory,
)
from asdsl.lut.permutation import permute_weights_for_lut, interleave_for_simd
from asdsl.quantization.core import quantize_weights


def bench_lut_build(
    shape: tuple[int, int] = (3072, 3072),
    bits: int = 4,
    group_size: int = 128,
    repeats: int = 3,
) -> None:
    """Benchmark LUT table construction time."""
    print("=" * 72)
    print("LUT Table Build Benchmark")
    print("=" * 72)

    weights = np.random.randn(*shape).astype(np.float32)
    qt = quantize_weights(weights, bits=bits, group_size=group_size)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        tables = build_lut_tables_for_layer(qt)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    median_ms = sorted(times)[len(times) // 2] * 1000
    num_tables = len(tables)
    print(f"Shape: {shape}, Bits: {bits}, Group Size: {group_size}")
    print(f"Tables built: {num_tables}")
    print(f"Build time: {median_ms:.2f} ms")
    print()


def bench_lut_matvec(
    out_features: int = 3072,
    in_features: int = 3072,
    bits: int = 4,
    group_size: int = 128,
    repeats: int = 10,
) -> None:
    """Benchmark LUT-based matrix-vector multiply throughput."""
    print("=" * 72)
    print("LUT MatVec Throughput Benchmark")
    print("=" * 72)

    print(f"{'Shape':>18s}  {'Bits':>4s}  {'Time (ms)':>10s}  {'GOPS':>10s}")
    print("-" * 50)

    for bits in [2, 3, 4]:
        shape = (out_features, in_features)
        weights = np.random.randn(*shape).astype(np.float32)
        qt = quantize_weights(weights, bits=bits, group_size=group_size)
        tables = build_lut_tables_for_layer(qt)
        x = np.random.randn(in_features).astype(np.float32)

        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            result = lut_matvec(tables, x, out_features=out_features, group_size=group_size)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        median_ms = sorted(times)[len(times) // 2] * 1000
        ops = 2 * out_features * in_features  # multiply-add pairs
        gops = (ops / 1e9) / (median_ms / 1000) if median_ms > 0 else 0
        print(f"{str(shape):>18s}  {bits:>4d}  {median_ms:>10.2f}  {gops:>10.2f}")

    print()


def bench_lut_memory(
    shapes: list[tuple[int, int]] | None = None,
    bits_list: list[int] | None = None,
) -> None:
    """Report expected LUT memory for various configurations."""
    if shapes is None:
        shapes = [(3072, 3072), (3072, 8192), (8192, 3072)]
    if bits_list is None:
        bits_list = [2, 3, 4]

    print("=" * 72)
    print("LUT Memory Footprint Estimate")
    print("=" * 72)
    print(f"{'Shape':>18s}  {'Bits':>4s}  {'LUT Memory':>12s}  {'Fits L1?':>8s}  {'Fits L2?':>8s}")
    print("-" * 60)

    for shape in shapes:
        for bits in bits_list:
            est = estimate_lut_memory(
                out_features=shape[0],
                in_features=shape[1],
                bits=bits,
                group_size=128,
            )
            lut_kb = est["total_bytes"] / 1024
            unit = "KB" if lut_kb < 1024 else "MB"
            val = lut_kb if lut_kb < 1024 else lut_kb / 1024
            print(
                f"{str(shape):>18s}  {bits:>4d}  {val:>9.1f} {unit:>2s}  "
                f"{'Yes' if est['fits_l1'] else 'No':>8s}  "
                f"{'Yes' if est['fits_l2'] else 'No':>8s}"
            )

    print()


def bench_permutation(
    shape: tuple[int, int] = (3072, 3072),
    repeats: int = 5,
) -> None:
    """Benchmark weight permutation for LUT alignment."""
    print("=" * 72)
    print("Weight Permutation Benchmark")
    print("=" * 72)

    weights = np.random.randn(*shape).astype(np.float32)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        permuted = permute_weights_for_lut(weights, group_size=128, tile_size=16)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    print(f"permute_weights_for_lut ({shape}): {sorted(times)[len(times)//2]*1000:.2f} ms")

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        interleaved = interleave_for_simd(weights, lane_width=8)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    print(f"interleave_for_simd ({shape}): {sorted(times)[len(times)//2]*1000:.2f} ms")
    print()


def main() -> None:
    bench_lut_build()
    bench_lut_matvec()
    bench_lut_memory()
    bench_permutation()
    print("All LUT benchmarks complete.")


if __name__ == "__main__":
    main()
