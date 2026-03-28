#!/usr/bin/env python3
"""
phase 4 prerequisite a: benchmark tiled lut vs single-row lut vs reference gemv.

usage:
    python scripts/profile_lut_fix.py           # single-row vs reference
    python scripts/profile_lut_fix.py --tiled   # also benchmark tiled kernel
"""
from __future__ import annotations

import argparse
import gc
import time

import numpy as np


def make_synthetic_q4(out_f: int, in_f: int, gs: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    weights_q4 = rng.integers(0, 16, (out_f, in_f), dtype=np.uint8)
    weights_packed = (weights_q4[:, 0::2] | (weights_q4[:, 1::2] << 4)).astype(np.uint8).ravel()
    num_groups = in_f // gs
    scales = rng.uniform(0.01, 0.1, out_f * num_groups).astype(np.float32)
    biases = (-8.0 * scales).astype(np.float32)
    x = rng.standard_normal(in_f).astype(np.float32)
    return weights_packed, scales, biases, x


def bench(fn, *args, n_warmup: int = 2, n_iters: int = 5) -> float:
    for _ in range(n_warmup):
        fn(*args)
    gc.collect()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn(*args)
    return (time.perf_counter() - t0) * 1000 / n_iters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiled", action="store_true", help="Also benchmark tiled kernel")
    ap.add_argument("--n-iters", type=int, default=5)
    ap.add_argument("--out-f", type=int, default=14336)
    ap.add_argument("--in-f",  type=int, default=3072)
    ap.add_argument("--gs",    type=int, default=64)
    args = ap.parse_args()

    print(f"\n[profile_lut_fix] Phi-4 FFN dimensions: {args.out_f}×{args.in_f}, gs={args.gs}")
    print(f"[profile_lut_fix] iterations: {args.n_iters}\n")

    try:
        from asdsl.kernels._native_lut import gemv_lut_q4_avx2
        from asdsl.kernels._native_gemv import gemv_q4_packed
    except ImportError as e:
        print(f"ERROR: native kernels not built: {e}")
        print("Run: python setup.py build_ext --inplace")
        return

    wp, sc, bi, x = make_synthetic_q4(args.out_f, args.in_f, args.gs)

    # Reference: scatter-gather LUT (single-row path)
    t_ref = bench(gemv_q4_packed, wp, x, sc, bi, args.out_f, args.in_f, args.gs,
                  n_iters=args.n_iters)

    # Single-row vpshufb LUT
    t_lut = bench(gemv_lut_q4_avx2, wp, sc, bi, x, args.out_f, args.in_f, args.gs,
                  n_iters=args.n_iters)

    flops = 2 * args.out_f * args.in_f
    gops_ref = flops / (t_ref / 1000) / 1e9
    gops_lut = flops / (t_lut / 1000) / 1e9

    print(f"{'Kernel':<35} {'ms/call':>10} {'GOPS':>10} {'vs ref':>10}")
    print("-" * 70)
    print(f"{'Reference FMA (gemv_q4_packed)':<35} {t_ref:>10.2f} {gops_ref:>10.2f} {'1.00x':>10}")
    print(f"{'Single-row LUT (gemv_lut_q4_avx2)':<35} {t_lut:>10.2f} {gops_lut:>10.2f} {f'{t_ref/t_lut:.2f}x':>10}")

    if args.tiled:
        try:
            from asdsl.kernels._native_lut import gemv_lut_q4_tiled, lut_tile_rows
            print(f"\n[Tiled kernel: TILE_ROWS={lut_tile_rows}]")

            t_tiled = bench(gemv_lut_q4_tiled, wp, sc, bi, x, args.out_f, args.in_f, args.gs,
                            n_iters=args.n_iters)
            gops_tiled = flops / (t_tiled / 1000) / 1e9
            speedup = t_ref / t_tiled

            print(f"{'Tiled LUT (gemv_lut_q4_tiled)':<35} {t_tiled:>10.2f} {gops_tiled:>10.2f} {f'{speedup:.2f}x':>10}")

            # Correctness check
            from tests.test_lut_gemv_correctness import make_test_case, dequant_q4_ref
            wp_s, sc_s, bi_s, x_s = make_test_case(32, 3072, 64, seed=1)
            y_tiled_s = gemv_lut_q4_tiled(wp_s, sc_s, bi_s, x_s, 32, 3072, 64)
            y_ref_s   = dequant_q4_ref(wp_s, sc_s, bi_s, x_s, 32, 3072, 64)
            max_rel = float(np.max(np.abs(y_tiled_s - y_ref_s) / (np.abs(y_ref_s) + 1e-6)))
            print(f"\nTiled correctness check (32×3072): max_rel_err = {max_rel:.4f} "
                  f"({'PASS' if max_rel < 0.05 else 'FAIL'})")

            # Requirement: tiled GOPS > gather GOPS
            if gops_tiled > gops_lut:
                print(f"\n✓ PREREQUISITE A MET: tiled ({gops_tiled:.2f} GOPS) > single-row ({gops_lut:.2f} GOPS)")
            else:
                print(f"\n⚠ Tiled ({gops_tiled:.2f} GOPS) not faster than single-row ({gops_lut:.2f} GOPS)")
                print("  (May indicate cache pressure — try with fresh RAM state)")
        except ImportError as e:
            print(f"ERROR loading tiled kernel: {e}")


if __name__ == "__main__":
    main()
