"""Quantization benchmarks: throughput, compression ratio, and error across bit widths."""

import time
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asdsl.quantization.core import quantize_weights, dequantize_weights, compute_quantization_error
from asdsl.quantization.salience import compute_gradient_salience, allocate_bits_by_salience


def bench_quantize_throughput(
    shapes: list[tuple[int, int]],
    bits_list: list[int],
    group_size: int = 128,
    repeats: int = 5,
) -> None:
    """Benchmark quantization throughput in MB/s across shapes and bit widths."""
    print("=" * 72)
    print("Quantization Throughput Benchmark")
    print("=" * 72)
    print(f"{'Shape':>18s}  {'Bits':>4s}  {'Time (ms)':>10s}  {'MB/s':>10s}  {'Comp. Ratio':>12s}")
    print("-" * 72)

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

            median_ms = sorted(times)[len(times) // 2] * 1000
            throughput = size_mb / (median_ms / 1000) if median_ms > 0 else float("inf")

            compressed_bytes = qt.data.nbytes + qt.scales.nbytes
            if qt.zero_points is not None:
                compressed_bytes += qt.zero_points.nbytes
            ratio = weights.nbytes / compressed_bytes

            print(f"{str(shape):>18s}  {bits:>4d}  {median_ms:>10.2f}  {throughput:>10.1f}  {ratio:>12.2f}x")

    print()


def bench_dequantize_throughput(
    shape: tuple[int, int] = (3072, 3072),
    bits_list: list[int] = [2, 3, 4, 8],
    group_size: int = 128,
    repeats: int = 10,
) -> None:
    """Benchmark dequantization speed."""
    print("=" * 72)
    print("Dequantization Throughput Benchmark")
    print("=" * 72)
    print(f"{'Bits':>4s}  {'Time (ms)':>10s}  {'MB/s (output)':>14s}")
    print("-" * 40)

    weights = np.random.randn(*shape).astype(np.float32)
    output_mb = weights.nbytes / (1024 * 1024)

    for bits in bits_list:
        qt = quantize_weights(weights, bits=bits, group_size=group_size)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            dequantize_weights(qt)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        median_ms = sorted(times)[len(times) // 2] * 1000
        throughput = output_mb / (median_ms / 1000) if median_ms > 0 else float("inf")
        print(f"{bits:>4d}  {median_ms:>10.2f}  {throughput:>14.1f}")

    print()


def bench_quantization_error(
    shape: tuple[int, int] = (3072, 3072),
    bits_list: list[int] = [2, 3, 4, 8],
    group_size: int = 128,
) -> None:
    """Measure quantization error across bit widths."""
    print("=" * 72)
    print("Quantization Error Analysis")
    print("=" * 72)
    print(f"{'Bits':>4s}  {'MSE':>12s}  {'MAE':>12s}  {'SNR (dB)':>10s}  {'Comp. Ratio':>12s}")
    print("-" * 60)

    weights = np.random.randn(*shape).astype(np.float32)

    for bits in bits_list:
        qt = quantize_weights(weights, bits=bits, group_size=group_size)
        errors = compute_quantization_error(weights, qt)
        print(
            f"{bits:>4d}  {errors['mse']:>12.6f}  {errors['mae']:>12.6f}  "
            f"{errors['snr_db']:>10.2f}  {errors['compression_ratio']:>12.2f}x"
        )

    print()


def bench_salience_analysis(
    shape: tuple[int, int] = (3072, 3072),
    repeats: int = 3,
) -> None:
    """Benchmark salience computation and bit allocation."""
    print("=" * 72)
    print("Salience Analysis Benchmark")
    print("=" * 72)

    weights = np.random.randn(*shape).astype(np.float32)
    gradients = np.random.randn(*shape).astype(np.float32) * 0.01

    # Gradient salience
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        salience = compute_gradient_salience(weights, gradients, group_size=128)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    print(f"Gradient salience ({shape}): {sorted(times)[len(times)//2]*1000:.2f} ms")

    # Bit allocation
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        allocation = allocate_bits_by_salience(salience, target_avg_bits=3.5)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    print(f"Bit allocation: {sorted(times)[len(times)//2]*1000:.2f} ms")
    print(f"Avg bits assigned: {allocation.avg_bits:.3f}")
    print()


def main() -> None:
    shapes = [(768, 768), (3072, 3072), (3072, 8192)]
    bits_list = [2, 3, 4, 8]

    bench_quantize_throughput(shapes, bits_list)
    bench_dequantize_throughput()
    bench_quantization_error()
    bench_salience_analysis()

    print("All quantization benchmarks complete.")


if __name__ == "__main__":
    main()
