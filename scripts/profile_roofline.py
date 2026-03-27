#!/usr/bin/env python3
"""
Phase 0 — Roofline position measurement for ASDSL benchmark baseline.

Measures:
  1. Theoretical peak GFLOPS (AVX2 FMA formula)
  2. Achieved DRAM bandwidth (streaming benchmark)
  3. Arithmetic intensity of gemv_q4_packed kernel
  4. Roofline state: memory_bound vs compute_bound

Writes results to benchmark_baseline.json under key "roofline".
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import psutil

# ── constants ──────────────────────────────────────────────────────────────────
# Phi-4 largest projection dimensions (14336 × 3072)
PHI4_OUT = 14336
PHI4_IN  = 3072

CONSERVATIVE_CLOCK_GHZ = 2.5  # fallback if clock undetectable


# ── helpers ────────────────────────────────────────────────────────────────────

def _detect_base_clock_ghz() -> tuple[float, bool]:
    """Return (base_clock_ghz, is_estimated)."""
    # Try psutil cpu_freq
    try:
        freq = psutil.cpu_freq()
        if freq and freq.min > 0:
            return round(freq.min / 1000.0, 2), False
        if freq and freq.current > 0:
            return round(freq.current / 1000.0, 2), True
    except Exception:
        pass
    # Linux: /proc/cpuinfo "cpu MHz"
    if sys.platform == "linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "cpu mhz" in line.lower():
                        mhz = float(line.split(":")[1].strip())
                        return round(mhz / 1000.0, 2), True
        except Exception:
            pass
    return CONSERVATIVE_CLOCK_GHZ, True


def _detect_avx2() -> bool:
    if sys.platform == "linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("flags") and "avx2" in line:
                        return True
        except Exception:
            pass
    # On Windows, assume AVX2 if native extension loaded successfully
    try:
        import asdsl.kernels._native_gemv  # noqa: F401
        return True
    except ImportError:
        pass
    return False


def _physical_cores() -> int:
    try:
        c = psutil.cpu_count(logical=False)
        return c if c else 1
    except Exception:
        return max(1, (os.cpu_count() or 2) // 2)


def compute_peak_gflops(
    physical_cores: int,
    base_clock_ghz: float,
    avx2: bool,
) -> dict:
    """
    AVX2 FMA: 8 float32 × 2 (FMA) = 16 FLOPS/cycle/core.
    Scalar fallback: 1 FLOP/cycle/core.
    """
    flops_per_cycle = 16 if avx2 else 8
    peak = physical_cores * base_clock_ghz * flops_per_cycle
    return {
        "peak_gflops": round(peak, 2),
        "physical_cores_used": physical_cores,
        "base_clock_ghz": base_clock_ghz,
        "flops_per_cycle": flops_per_cycle,
        "avx2_assumed": avx2,
    }


def measure_dram_bandwidth() -> dict:
    """
    Streaming benchmark: b = a + 1.0 for 2 seconds.
    bytes_transferred = 3 × array_bytes × iterations
    (1 read a, 1 read a again for add, 1 write b — numpy fuses to 2 reads + 1 write)
    We use the conservative 3× factor as specified.
    """
    vm = psutil.virtual_memory()
    available_gb = vm.available / 1e9

    # Choose array size based on available RAM
    if available_gb >= 1.0:
        array_bytes = 256 * 1024 * 1024  # 256 MB
    else:
        array_bytes = 64 * 1024 * 1024   # 64 MB fallback

    n_elements = array_bytes // 4  # float32
    note = f"array_size_mb={array_bytes // (1024*1024)}, available_ram_gb={available_gb:.1f}"

    a = np.ones(n_elements, dtype=np.float32)
    b = np.empty(n_elements, dtype=np.float32)

    # Warm up
    np.add(a, 1.0, out=b)

    # Timed loop for 2 seconds
    iterations = 0
    t_start = time.perf_counter()
    deadline = t_start + 2.0
    while time.perf_counter() < deadline:
        np.add(a, 1.0, out=b)
        iterations += 1

    elapsed = time.perf_counter() - t_start
    bytes_transferred = 3 * array_bytes * iterations
    bandwidth_gbps = bytes_transferred / elapsed / 1e9

    return {
        "bandwidth_gbps": round(bandwidth_gbps, 2),
        "iterations": iterations,
        "elapsed_sec": round(elapsed, 3),
        "array_size_mb": array_bytes // (1024 * 1024),
        "note": note,
    }


def measure_gemv_arithmetic_intensity() -> dict:
    """
    Measure arithmetic intensity of gemv_q4_packed on a synthetic
    14336 × 3072 weight matrix (packed as 14336 × 1536 uint8).
    """
    result: dict = {
        "used_native": False,
        "out_rows": PHI4_OUT,
        "in_cols": PHI4_IN,
        "arithmetic_intensity": None,
        "gemv_time_ms_per_call": None,
        "note": "",
    }

    try:
        from asdsl.kernels._native_gemv import gemv_q4_packed
        result["used_native"] = True
    except ImportError:
        result["note"] = "_native_gemv not built; skipping intensity measurement"
        return result

    # Synthetic packed weight matrix: flat (M*K/2,) uint8 as required by native API
    GROUP_SIZE = 64
    rng = np.random.default_rng(42)
    W_packed = rng.integers(0, 256, size=(PHI4_OUT * PHI4_IN // 2,), dtype=np.uint8)
    x = rng.standard_normal(PHI4_IN).astype(np.float32)
    n_groups = PHI4_OUT * (PHI4_IN // GROUP_SIZE)
    scales = np.ones(n_groups, dtype=np.float32)
    biases = np.zeros(n_groups, dtype=np.float32)

    # Warm up
    try:
        _ = gemv_q4_packed(W_packed, x, scales, biases, PHI4_OUT, PHI4_IN, GROUP_SIZE)
    except Exception as e:
        result["note"] = f"gemv_q4_packed call failed: {e}"
        result["used_native"] = False
        return result

    # Time 10 calls
    N_CALLS = 10
    t0 = time.perf_counter()
    for _ in range(N_CALLS):
        _ = gemv_q4_packed(W_packed, x, scales, biases, PHI4_OUT, PHI4_IN, GROUP_SIZE)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # FLOP count: 2 × out_rows × in_cols (one multiply + one add per element)
    flops = 2 * PHI4_OUT * PHI4_IN * N_CALLS
    # Bytes: weight matrix (packed) + activation + output
    weight_bytes = W_packed.nbytes
    act_bytes    = x.nbytes
    out_bytes    = PHI4_OUT * 4  # float32 output
    total_bytes  = (weight_bytes + act_bytes + out_bytes) * N_CALLS

    ai = flops / total_bytes

    result["arithmetic_intensity"] = round(ai, 4)
    result["gemv_time_ms_per_call"] = round(elapsed_ms / N_CALLS, 3)
    result["flops_per_call"] = 2 * PHI4_OUT * PHI4_IN
    result["bytes_per_call"] = weight_bytes + act_bytes + out_bytes
    result["note"] = "synthetic 14336×3072 Q4 packed matrix, group_size=64"

    return result


def compute_roofline_state(
    peak_gflops: float,
    bandwidth_gbps: float,
    arithmetic_intensity: float | None,
) -> str:
    if arithmetic_intensity is None:
        return "unknown"
    # Ridge point: peak_gflops / bandwidth_gbps (in FLOPS/byte)
    ridge = (peak_gflops * 1e9) / (bandwidth_gbps * 1e9)
    return "memory_bound" if arithmetic_intensity < ridge else "compute_bound"


def run_roofline() -> dict:
    avx2 = _detect_avx2()
    physical_cores = _physical_cores()
    base_clock_ghz, clock_estimated = _detect_base_clock_ghz()

    peak_info = compute_peak_gflops(physical_cores, base_clock_ghz, avx2)
    bw_info   = measure_dram_bandwidth()
    gemv_info = measure_gemv_arithmetic_intensity()

    ai = gemv_info.get("arithmetic_intensity")
    state = compute_roofline_state(
        peak_info["peak_gflops"],
        bw_info["bandwidth_gbps"],
        ai,
    )

    ridge_point = None
    if bw_info["bandwidth_gbps"] > 0:
        ridge_point = round(
            (peak_info["peak_gflops"] * 1e9) / (bw_info["bandwidth_gbps"] * 1e9), 4
        )

    return {
        "peak_gflops": peak_info["peak_gflops"],
        "peak_gflops_detail": peak_info,
        "clock_estimated": clock_estimated,
        "bandwidth_gbps": bw_info["bandwidth_gbps"],
        "bandwidth_detail": bw_info,
        "gemv_arithmetic_intensity": ai,
        "gemv_detail": gemv_info,
        "ridge_point_flops_per_byte": ridge_point,
        "roofline_state": state,
        "prediction_confirmed": state == "memory_bound",
    }


def update_baseline_json(roofline: dict, baseline_path: Path) -> None:
    if baseline_path.exists():
        try:
            data = json.loads(baseline_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    else:
        data = {}
    data["roofline"] = roofline
    baseline_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


if __name__ == "__main__":
    baseline_path = Path(__file__).parent.parent / "benchmark_baseline.json"
    result = run_roofline()
    print(json.dumps(result, indent=2))
    update_baseline_json(result, baseline_path)
    print(f"\n[profile_roofline] Written to {baseline_path}")
