#!/usr/bin/env python3
"""
Phase 0 — Dequantization overhead measurement for ASDSL benchmark baseline.

Differential benchmark:
  1. Time gemv_q4_packed on a synthetic 14336×3072 weight matrix (20 iters)
  2. Time numpy.dot on equivalent float32 matrix (20 iters)
  3. Estimate dequant fraction = max(0, (t_q4 - t_fp32) / t_q4)

--quick flag: reduces to 4096×1024, 5 iterations.

Writes results to benchmark_baseline.json under key "dequant_profile".
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ── constants ──────────────────────────────────────────────────────────────────
FULL_OUT, FULL_IN = 14336, 3072
QUICK_OUT, QUICK_IN = 4096, 1024
FULL_ITERS  = 20
QUICK_ITERS = 5


# ── helpers ────────────────────────────────────────────────────────────────────

def _time_calls(fn, n_iters: int) -> float:
    """Return total wall-clock ms for n_iters calls to fn()."""
    # Warm up
    fn()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    return (time.perf_counter() - t0) * 1000.0


def _dequant_q4_to_fp32(
    W_packed: np.ndarray,
    scales: np.ndarray,
    zeros: np.ndarray,
    out_rows: int,
    in_cols: int,
    group_size: int = 64,
) -> np.ndarray:
    """
    Dequantize packed Q4 uint8 matrix to float32.
    W_packed shape: (out_rows, in_cols // 2)
    Returns float32 (out_rows, in_cols)
    """
    # Unpack nibbles
    lo = (W_packed & 0x0F).astype(np.float32)
    hi = ((W_packed >> 4) & 0x0F).astype(np.float32)
    # Interleave: even cols = lo, odd cols = hi
    W_unpacked = np.empty((out_rows, in_cols), dtype=np.float32)
    W_unpacked[:, 0::2] = lo
    W_unpacked[:, 1::2] = hi

    # Apply per-group scale and zero-point
    n_groups = in_cols // group_size
    for g in range(n_groups):
        col_start = g * group_size
        col_end   = col_start + group_size
        s = scales[:, g].reshape(-1, 1)   # (out_rows, 1)
        z = zeros[:, g].reshape(-1, 1)
        W_unpacked[:, col_start:col_end] = (
            W_unpacked[:, col_start:col_end] - z
        ) * s

    return W_unpacked


def run_dequant_profile(quick: bool = False) -> dict:
    out_rows = QUICK_OUT if quick else FULL_OUT
    in_cols  = QUICK_IN  if quick else FULL_IN
    n_iters  = QUICK_ITERS if quick else FULL_ITERS
    group_size = 64

    rng = np.random.default_rng(42)
    # Native API expects flat (M*K/2,) uint8
    W_packed = rng.integers(0, 256, size=(out_rows * in_cols // 2,), dtype=np.uint8)
    W_packed_2d = W_packed.reshape(out_rows, in_cols // 2)  # for dequant helper
    x = rng.standard_normal(in_cols).astype(np.float32)

    n_groups = in_cols // group_size
    scales_flat = np.ones(out_rows * n_groups, dtype=np.float32)
    biases_flat = np.zeros(out_rows * n_groups, dtype=np.float32)
    scales_2d   = scales_flat.reshape(out_rows, n_groups)
    zeros_2d    = np.full((out_rows, n_groups), 8.0, dtype=np.float32)  # Q4 midpoint

    result: dict = {
        "quick_mode": quick,
        "out_rows": out_rows,
        "in_cols": in_cols,
        "n_iters": n_iters,
        "used_native": False,
        "t_q4_packed_ms": None,
        "t_fp32_mv_ms": None,
        "dequant_fraction_estimate": None,
        "q4_throughput_gops": None,
        "note": "",
    }

    # ── Step 1: time gemv_q4_packed ───────────────────────────────────────────
    native_ok = False
    try:
        from asdsl.kernels._native_gemv import gemv_q4_packed
        native_ok = True
        result["used_native"] = True

        def _q4_call():
            return gemv_q4_packed(W_packed, x, scales_flat, biases_flat,
                                  out_rows, in_cols, group_size)

        # Verify it runs
        _ = _q4_call()
        t_q4_ms = _time_calls(_q4_call, n_iters)
        result["t_q4_packed_ms"] = round(t_q4_ms, 3)

    except ImportError:
        result["note"] += "_native_gemv not built; using numpy fallback for Q4 timing. "
    except Exception as e:
        result["note"] += f"gemv_q4_packed failed: {e}. "
        native_ok = False

    if not native_ok:
        # Fallback: time the Python dequant + numpy.dot path
        def _q4_fallback():
            W_fp32 = _dequant_q4_to_fp32(W_packed_2d, scales_2d, zeros_2d,
                                          out_rows, in_cols, group_size)
            return W_fp32 @ x

        t_q4_ms = _time_calls(_q4_fallback, n_iters)
        result["t_q4_packed_ms"] = round(t_q4_ms, 3)
        result["used_native"] = False

    # ── Step 2: time float32 numpy.dot ────────────────────────────────────────
    # Dequantize once upfront
    W_fp32 = _dequant_q4_to_fp32(W_packed_2d, scales_2d, zeros_2d,
                                  out_rows, in_cols, group_size)

    def _fp32_call():
        return np.dot(W_fp32, x)

    t_fp32_ms = _time_calls(_fp32_call, n_iters)
    result["t_fp32_mv_ms"] = round(t_fp32_ms, 3)

    # ── Step 3: dequant fraction estimate ─────────────────────────────────────
    if result["t_q4_packed_ms"] is not None and result["t_q4_packed_ms"] > 0:
        frac = max(0.0, (result["t_q4_packed_ms"] - t_fp32_ms) / result["t_q4_packed_ms"])
        result["dequant_fraction_estimate"] = round(frac, 4)

    # ── Step 4: Q4 throughput in GOPS ─────────────────────────────────────────
    if result["t_q4_packed_ms"] is not None and result["t_q4_packed_ms"] > 0:
        total_ops = 2 * out_rows * in_cols * n_iters
        gops = total_ops / (result["t_q4_packed_ms"] / 1000.0) / 1e9
        result["q4_throughput_gops"] = round(gops, 4)

    if not result["note"]:
        result["note"] = (
            f"{'native' if result['used_native'] else 'numpy-fallback'} Q4 GEMV "
            f"vs FP32 numpy.dot, {out_rows}×{in_cols}, {n_iters} iters"
        )

    return result


def update_baseline_json(dequant: dict, baseline_path: Path) -> None:
    if baseline_path.exists():
        try:
            data = json.loads(baseline_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    else:
        data = {}
    data["dequant_profile"] = dequant
    baseline_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile Q4 dequantization overhead")
    parser.add_argument("--quick", action="store_true",
                        help="Use 4096×1024 matrix and 5 iterations")
    args = parser.parse_args()

    baseline_path = Path(__file__).parent.parent / "benchmark_baseline.json"
    result = run_dequant_profile(quick=args.quick)
    print(json.dumps(result, indent=2))
    update_baseline_json(result, baseline_path)
    print(f"\n[profile_dequant] Written to {baseline_path}")
