#!/usr/bin/env python3
"""
Phase 0 — Assemble benchmark_baseline.json.

Orchestrates all Phase 0 measurement steps:
  1. Kernel audit (Step 1)
  2. Hardware profile (already written by profile_hardware.py)
  3. Roofline profile (already written by profile_roofline.py)
  4. Dequant profile (already written by profile_dequant.py)
  5. Token throughput (Step 5) — runs phi4_cpu_run.py if model present
  6. Cache profile (Step 6) — Linux + perf only
  7. Assemble final JSON with schema_version, phase_0_summary, etc.

Run after the three profiling scripts have already been executed.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
BASELINE_PATH = ROOT / "benchmark_baseline.json"
MODEL_INDEX = ROOT / "models" / "phi4-multimodal-instruct" / "model.safetensors.index.json"


# ── helpers ────────────────────────────────────────────────────────────────────

def _load_baseline() -> dict:
    if BASELINE_PATH.exists():
        try:
            return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_baseline(data: dict) -> None:
    BASELINE_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


# ── Step 1: Kernel audit ───────────────────────────────────────────────────────

def build_kernel_audit() -> dict:
    audit: dict = {}

    # gemv_q4_avx2.cpp line count
    gemv_path = ROOT / "asdsl/kernels/native/gemv_q4_avx2.cpp"
    if gemv_path.exists():
        audit["gemv_q4_avx2_cpp_line_count"] = len(gemv_path.read_text(encoding="utf-8", errors="replace").splitlines())
    else:
        audit["gemv_q4_avx2_cpp_line_count"] = None

    # lut_avx2.cpp state
    lut_path = ROOT / "asdsl/kernels/native/lut_avx2.cpp"
    if not lut_path.exists():
        audit["lut_avx2_state"] = "missing"
        audit["lut_avx2_line_count"] = None
    else:
        lut_text = lut_path.read_text(encoding="utf-8", errors="replace")
        lut_lines = lut_text.splitlines()
        audit["lut_avx2_line_count"] = len(lut_lines)

        # Classify state
        has_pybind = "PYBIND11_MODULE" in lut_text
        has_impl = "lut_build_tables_impl" in lut_text or "lut_matvec_impl" in lut_text
        has_avx2 = "_mm256" in lut_text or "vpshufb" in lut_text or "vgatherdps" in lut_text.lower() or "_mm256_i32gather" in lut_text
        has_gemv_lut_q4 = "gemv_lut_q4" in lut_text  # Phase 1 target function

        if has_pybind and has_impl and has_avx2:
            audit["lut_avx2_state"] = "functional"
        elif has_pybind and has_impl:
            audit["lut_avx2_state"] = "partial"
        elif has_pybind:
            audit["lut_avx2_state"] = "stub"
        else:
            audit["lut_avx2_state"] = "partial"

    # setup.py flags
    setup_path = ROOT / "setup.py"
    setup_text = setup_path.read_text(encoding="utf-8") if setup_path.exists() else ""
    audit["openmp_enabled"] = "/openmp" in setup_text or "-fopenmp" in setup_text
    audit["avx512_enabled"] = "-mavx512f" in setup_text or "/arch:AVX-512" in setup_text or "avx512" in setup_text.lower()

    # Native extension import test
    native_ok = False
    try:
        import asdsl.kernels._native_gemv  # noqa: F401
        native_ok = True
    except ImportError:
        pass
    audit["native_extensions_built"] = native_ok

    # Notes
    notes = []
    if audit["lut_avx2_state"] == "functional":
        notes.append("lut_avx2.cpp is functional with AVX2 gather-based LUT matvec; "
                     "Phase 1 target is vpshufb-based gemv_lut_q4_avx2 (not yet present).")
    elif audit["lut_avx2_state"] == "partial":
        notes.append("lut_avx2.cpp has partial implementation; needs gemv_lut_q4_avx2 for Phase 1.")
    if not native_ok:
        notes.append("_native_gemv not importable — build with: python setup.py build_ext --inplace")
    if not notes:
        notes.append("Codebase is in good shape for Phase 1; gemv_q4_packed_impl_v2 present as correctness baseline.")
    audit["notes"] = " ".join(notes)

    return audit


# ── Step 5: Token throughput ───────────────────────────────────────────────────

def measure_token_throughput() -> dict:
    if not MODEL_INDEX.exists():
        return {
            "model_present": False,
            "tok_per_sec": None,
            "note": "model weights not found at models/phi4-multimodal-instruct/",
        }

    prompt = "The fundamental theorem of calculus states that"
    max_new_tokens = 32
    threads = 0  # auto (cpu_count // 2)

    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "phi4_cpu_run.py"),
        "--bits", "4",
        "--prompt", prompt,
        "--max-new-tokens", str(max_new_tokens),
        "--threads", str(threads),
    ]

    print(f"[token_throughput] Running: {' '.join(cmd)}", flush=True)
    print("[token_throughput] This may take several minutes...", flush=True)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min max
            cwd=str(ROOT),
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return {
            "model_present": True,
            "tok_per_sec": None,
            "note": "phi4_cpu_run.py timed out after 600s",
        }
    except Exception as e:
        return {
            "model_present": True,
            "tok_per_sec": None,
            "note": f"phi4_cpu_run.py failed: {e}",
        }

    # Parse tok/s from output.
    # Pattern 1: "Generated : 32 tokens  |  1.20 tok/s  |  decode 26.6s"
    tps_match = re.search(
        r"Generated\s*:\s*(\d+)\s+tokens?\s*\|\s*([\d.]+)\s*tok/s",
        output, re.IGNORECASE
    )
    if tps_match:
        tok_count = int(tps_match.group(1))
        tps = float(tps_match.group(2))
    else:
        # Pattern 2: "[N tokens | X.XX tok/s]"
        tps_match2 = re.search(r"\[(\d+)\s+tokens?\s*\|\s*([\d.]+)\s*tok/s\]", output)
        if tps_match2:
            tok_count = int(tps_match2.group(1))
            tps = float(tps_match2.group(2))
        else:
            # Pattern 3: "tokens_per_second": X.XX in JSON output
            tps_match3 = re.search(r'"tokens_per_second"\s*:\s*([\d.]+)', output)
            if tps_match3:
                tps = float(tps_match3.group(1))
                tok_count = max_new_tokens
            else:
                # Pattern 4: bare "X.XX tok/s"
                tps_match4 = re.search(r'([\d.]+)\s*tok/s', output, re.IGNORECASE)
                if tps_match4:
                    tps = float(tps_match4.group(1))
                    tok_count = None
                else:
                    tps = None
                    tok_count = None

    # Detect thread count used
    thread_match = re.search(r"threads?\s*[=:]\s*(\d+)", output, re.IGNORECASE)
    thread_count = int(thread_match.group(1)) if thread_match else None

    return {
        "model_present": True,
        "tok_per_sec": tps,
        "tokens_generated": tok_count,
        "thread_count_used": thread_count,
        "prompt_used": prompt,
        "max_new_tokens": max_new_tokens,
        "exit_code": result.returncode,
        "note": "phi4_cpu_run.py --bits 4 greedy decode" if tps else f"could not parse tok/s from output; exit={result.returncode}",
    }


# ── Step 6: Cache profile (Linux + perf only) ─────────────────────────────────

def measure_cache_profile() -> dict:
    if sys.platform != "linux":
        return {
            "available": False,
            "reason": "perf not available or not linux",
        }

    # Check if perf is available
    try:
        r = subprocess.run(["which", "perf"], capture_output=True, timeout=5)
        if r.returncode != 0:
            return {"available": False, "reason": "perf not available or not linux"}
    except Exception:
        return {"available": False, "reason": "perf not available or not linux"}

    cmd = [
        "perf", "stat",
        "-e", "cache-misses,cache-references,LLC-load-misses,LLC-loads,instructions,cycles",
        sys.executable, str(ROOT / "scripts" / "profile_dequant.py"), "--quick",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, cwd=str(ROOT)
        )
        perf_output = result.stderr  # perf stat writes to stderr
    except Exception as e:
        return {"available": False, "reason": f"perf run failed: {e}"}

    def _parse_perf_int(pattern: str, text: str) -> int | None:
        m = re.search(pattern, text)
        if m:
            return int(m.group(1).replace(",", "").replace(".", ""))
        return None

    cache_misses = _parse_perf_int(r"([\d,]+)\s+cache-misses", perf_output)
    cache_refs   = _parse_perf_int(r"([\d,]+)\s+cache-references", perf_output)
    llc_miss     = _parse_perf_int(r"([\d,]+)\s+LLC-load-misses", perf_output)
    llc_loads    = _parse_perf_int(r"([\d,]+)\s+LLC-loads", perf_output)
    instructions = _parse_perf_int(r"([\d,]+)\s+instructions", perf_output)
    cycles       = _parse_perf_int(r"([\d,]+)\s+cycles", perf_output)

    cache_miss_rate = None
    if cache_misses is not None and cache_refs and cache_refs > 0:
        cache_miss_rate = round(cache_misses / cache_refs, 4)

    llc_miss_rate = None
    if llc_miss is not None and llc_loads and llc_loads > 0:
        llc_miss_rate = round(llc_miss / llc_loads, 4)

    ipc = None
    if instructions is not None and cycles and cycles > 0:
        ipc = round(instructions / cycles, 4)

    return {
        "available": True,
        "cache_misses": cache_misses,
        "cache_references": cache_refs,
        "cache_miss_rate": cache_miss_rate,
        "llc_load_misses": llc_miss,
        "llc_loads": llc_loads,
        "llc_miss_rate": llc_miss_rate,
        "instructions": instructions,
        "cycles": cycles,
        "ipc": ipc,
    }


# ── Step 7: Assemble final JSON ────────────────────────────────────────────────

def build_phase0_summary(data: dict) -> dict:
    roofline = data.get("roofline", {})
    dequant  = data.get("dequant_profile", {})
    cache    = data.get("cache_profile", {})
    throughput = data.get("token_throughput", {})
    audit    = data.get("kernel_audit", {})

    bottleneck_confirmed = roofline.get("roofline_state") == "memory_bound"

    dequant_frac = dequant.get("dequant_fraction_estimate")
    dequant_pct  = round(dequant_frac * 100, 1) if dequant_frac is not None else None

    llc_miss_rate = cache.get("llc_miss_rate") if cache.get("available") else None

    baseline_tps = throughput.get("tok_per_sec")

    # lut_avx2 ready for Phase 1: functional or partial with pybind boilerplate
    lut_state = audit.get("lut_avx2_state", "missing")
    lut_ready = lut_state in ("functional", "partial", "stub")

    # Build summary note
    if bottleneck_confirmed:
        note = (
            f"Kernel is memory-bound (AI={roofline.get('gemv_arithmetic_intensity', '?')} "
            f"vs ridge={roofline.get('ridge_point_flops_per_byte', '?')} FLOPS/byte); "
            f"dequant overhead ~{dequant_pct}%; LUT vpshufb rewrite is the highest-leverage Phase 1 intervention."
        )
    else:
        note = "Roofline state unclear; re-run with larger matrix for accurate measurement."

    return {
        "bottleneck_confirmed_memory_bound": bottleneck_confirmed,
        "estimated_dequant_overhead_pct": dequant_pct,
        "llc_miss_rate": llc_miss_rate,
        "baseline_tok_per_sec": baseline_tps,
        "lut_avx2_ready_for_phase1": lut_ready,
        "notes": note,
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-phi4", action="store_true",
                        help="Skip phi4_cpu_run.py (use if already measured)")
    parser.add_argument("--tok-per-sec", type=float, default=None,
                        help="Manually supply tok/s result (used with --skip-phi4)")
    args = parser.parse_args()

    print("[assemble_baseline] Loading existing profiling data...", flush=True)
    data = _load_baseline()

    # Step 1: Kernel audit
    print("[assemble_baseline] Step 1: Kernel audit...", flush=True)
    data["kernel_audit"] = build_kernel_audit()

    # Step 5: Token throughput
    if args.skip_phi4:
        print("[assemble_baseline] Step 5: Token throughput (using supplied value)...", flush=True)
        tps = args.tok_per_sec
        data["token_throughput"] = {
            "model_present": MODEL_INDEX.exists(),
            "tok_per_sec": tps,
            "tokens_generated": 32,
            "thread_count_used": None,
            "prompt_used": "The fundamental theorem of calculus states that",
            "max_new_tokens": 32,
            "note": "phi4_cpu_run.py --bits 4 greedy decode (result supplied via --tok-per-sec)" if tps else "skipped",
        }
    else:
        print("[assemble_baseline] Step 5: Token throughput...", flush=True)
        data["token_throughput"] = measure_token_throughput()

    # Step 6: Cache profile
    print("[assemble_baseline] Step 6: Cache profile...", flush=True)
    data["cache_profile"] = measure_cache_profile()

    # Step 7: Assemble final structure
    data["schema_version"] = "1.0"
    data["generated_at"] = datetime.now(timezone.utc).isoformat()
    data["phase"] = 0
    data["phase_0_summary"] = build_phase0_summary(data)

    # Ensure all required top-level keys are present
    for key in ["hardware", "roofline", "dequant_profile"]:
        if key not in data:
            data[key] = {"error": f"{key} not yet measured — run profile_{key.replace('_profile','')}.py first"}

    _save_baseline(data)
    print(f"[assemble_baseline] Written to {BASELINE_PATH}", flush=True)

    # Validate JSON
    try:
        json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        print("[assemble_baseline] JSON validation: PASSED", flush=True)
    except json.JSONDecodeError as e:
        print(f"[assemble_baseline] JSON validation: FAILED — {e}", flush=True)
        sys.exit(1)

    # Print summary
    s = data["phase_0_summary"]
    print("\n=== PHASE 0 COMPLETE ===")
    print(f"Baseline JSON: {BASELINE_PATH.name}")
    print("Key findings:")
    print(f"  - Roofline state: {data.get('roofline', {}).get('roofline_state', 'unknown')}")
    print(f"  - Dequant overhead estimate: {s['estimated_dequant_overhead_pct']}% or unknown" if s['estimated_dequant_overhead_pct'] is not None else "  - Dequant overhead estimate: unknown")
    print(f"  - LLC miss rate: {s['llc_miss_rate']} or unknown" if s['llc_miss_rate'] is not None else "  - LLC miss rate: unknown")
    tps = s['baseline_tok_per_sec']
    print(f"  - Baseline tok/s: {tps}" if tps is not None else "  - Baseline tok/s: model not present")
    print(f"  - lut_avx2 ready for Phase 1: {'yes' if s['lut_avx2_ready_for_phase1'] else 'no'}")
    print("Next step: hand benchmark_baseline.json to the engineer for Phase 1 planning.")
    print("=== STOPPING ===")


if __name__ == "__main__":
    main()
