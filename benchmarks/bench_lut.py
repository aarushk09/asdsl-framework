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
    shape: tuple[int, int] = (256, 256),
    bits: int = 4,
    group_size: int = 128,
    group_width: int = 2,
    repeats: int = 3,
) -> None:
    """Benchmark LUT table construction time (Python reference kernels).

    Note: This is the Python reference implementation.  A native AVX2/VNNI
    backend would process each LUT group in a handful of vector instructions
    rather than a Python for-loop, giving ~100-1000x speedup.
    """
    print("="* 72)
    print("LUT Table Build Benchmark")
    print("=" * 72)

    weights = np.random.randn(*shape).astype(np.float32)
    qt = quantize_weights(weights, bits=bits, group_size=group_size)
    x = np.random.randn(shape[1]).astype(np.float32)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        tables = build_lut_tables_for_layer(
            qt.data, qt.scales, x, qt.bits, qt.group_size, group_width=group_width
        )
        t1 = time.perf_counter()
        times.append(t1 - t0)

    median_ms = sorted(times)[len(times) // 2] * 1000
    num_tables = len(tables)
    entries_per_table = (1 << bits) ** group_width
    print(f"Shape: {shape}, Bits: {bits}, group_width: {group_width}")
    print(f"Tables built: {num_tables}  ({entries_per_table} entries each)")
    print(f"Build time (Python): {median_ms:.2f} ms")

    # Native AVX2 path
    try:
        from asdsl.lut.lut_native import has_native_lut, build_lut_tables_native
        from asdsl.quantization.core import _unpack_bits
        if has_native_lut():
            unpacked = _unpack_bits(qt.data, bits)
            native_times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                flat = build_lut_tables_native(
                    unpacked[:shape[0] * shape[1]], x,
                    np.asarray(qt.scales, dtype=np.float32),
                    bits, group_width, shape[0], shape[1], group_size,
                )
                t1 = time.perf_counter()
                native_times.append(t1 - t0)
            native_ms = sorted(native_times)[len(native_times) // 2] * 1000
            speedup = median_ms / native_ms if native_ms > 0 else 0
            print(f"Build time (Native AVX2): {native_ms:.2f} ms  ({speedup:.1f}x faster)")
    except Exception:
        pass

    print()


def bench_lut_matvec(
    out_features: int = 256,
    in_features: int = 256,
    group_size: int = 128,
    group_width: int = 2,
    repeats: int = 5,
) -> None:
    """Benchmark LUT-based matrix-vector multiply throughput (Python reference).

    Note: The Python kernel is loop-based and ~100-1000x slower than what a
    native AVX2/VNNI implementation would achieve.
    """
    print("=" * 72)
    print("LUT MatVec Throughput Benchmark")
    print("=" * 72)

    print(f"{'Shape':>18s}  {'Bits':>4s}  {'Python (ms)':>12s}  {'Native (ms)':>12s}  {'Speedup':>8s}  {'GOPS':>10s}")
    print("-" * 70)

    for bits in [2, 4]:
        shape = (out_features, in_features)
        weights = np.random.randn(*shape).astype(np.float32)
        qt = quantize_weights(weights, bits=bits, group_size=group_size)
        x = np.random.randn(in_features).astype(np.float32)
        tables = build_lut_tables_for_layer(
            qt.data, qt.scales, x, qt.bits, qt.group_size, group_width=group_width
        )

        # Python timing
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            result = lut_matvec(tables, qt.data, qt.bits, out_features, in_features)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        py_ms = sorted(times)[len(times) // 2] * 1000

        # Native timing
        native_ms_str = "N/A"
        speedup_str = "-"
        gops_ms = py_ms
        try:
            from asdsl.lut.lut_native import has_native_lut, build_lut_tables_native, lut_matvec_native
            from asdsl.quantization.core import _unpack_bits
            if has_native_lut():
                unpacked = _unpack_bits(qt.data, bits)
                flat = build_lut_tables_native(
                    unpacked[:out_features * in_features], x,
                    np.asarray(qt.scales, dtype=np.float32),
                    bits, group_width, out_features, in_features, group_size,
                )
                native_times = []
                for _ in range(repeats):
                    t0 = time.perf_counter()
                    native_result = lut_matvec_native(
                        flat, unpacked[:out_features * in_features],
                        bits, group_width, out_features, in_features,
                    )
                    t1 = time.perf_counter()
                    native_times.append(t1 - t0)
                n_ms = sorted(native_times)[len(native_times) // 2] * 1000
                native_ms_str = f"{n_ms:.4f}"
                speedup_str = f"{py_ms / n_ms:.1f}x" if n_ms > 0 else "-"
                gops_ms = n_ms
        except Exception:
            pass

        ops = 2 * out_features * in_features
        gops = (ops / 1e9) / (gops_ms / 1000) if gops_ms > 0 else 0
        print(f"{str(shape):>18s}  {bits:>4d}  {py_ms:>12.2f}  {native_ms_str:>12s}  {speedup_str:>8s}  {gops:>10.4f}")

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

    group_width = 2  # 2 weights per LUT entry → 2^(bits*2) entries
    print("=" * 72)
    print("LUT Memory Footprint Estimate  (group_width=2, one forward pass)")
    print("=" * 72)
    print(f"{'Shape':>18s}  {'Bits':>4s}  {'Entries/LUT':>12s}  {'LUT Mem':>10s}  "
          f"{'In L1?':>6s}  {'In L2?':>6s}")
    print("-" * 66)

    for shape in shapes:
        in_features = shape[1]
        for bits in bits_list:
            # num_weight_groups = number of LUT tables for one forward pass
            # Each group covers group_width elements of the activation vector
            num_weight_groups = in_features // group_width
            est = estimate_lut_memory(
                bits=bits,
                group_width=group_width,
                num_weight_groups=num_weight_groups,
            )
            total_kb = est["total_kb"]
            unit = "KB" if total_kb < 1024 else "MB"
            val = total_kb if total_kb < 1024 else total_kb / 1024
            entries = est["entries_per_table"]
            print(
                f"{str(shape):>18s}  {bits:>4d}  {entries:>12d}  "
                f"{val:>7.1f} {unit:>2s}  "
                f"{'Yes' if est['fits_l1_cache'] else 'No':>6s}  "
                f"{'Yes' if est['fits_l2_cache'] else 'No':>6s}"
            )

    print()


def bench_permutation(
    shape: tuple[int, int] = (3072, 3072),
    bits: int = 4,
    group_size: int = 128,
    repeats: int = 5,
) -> None:
    """Benchmark weight permutation for LUT alignment."""
    print("=" * 72)
    print("Weight Permutation Benchmark")
    print("=" * 72)

    weights = np.random.randn(*shape).astype(np.float32)
    qt = quantize_weights(weights, bits=bits, group_size=group_size)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        permuted = permute_weights_for_lut(qt.data, qt.bits, shape[0], shape[1])
        t1 = time.perf_counter()
        times.append(t1 - t0)
    print(f"permute_weights_for_lut ({shape}, {bits}-bit): {sorted(times)[len(times)//2]*1000:.2f} ms")

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        interleaved = interleave_for_simd(qt.data, qt.bits)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    print(f"interleave_for_simd ({shape}, {bits}-bit): {sorted(times)[len(times)//2]*1000:.2f} ms")
    print()


def main() -> None:
    bench_lut_build()
    bench_lut_matvec()
    bench_lut_memory()
    bench_permutation()
    print("All LUT benchmarks complete.")


if __name__ == "__main__":
    main()
