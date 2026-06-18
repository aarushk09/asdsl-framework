"""Phase 6: offline sweep of ASDSL_GEMV_CHUNK_DIV for preq2 gate_up microbench."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = ROOT / "benchmarks" / "results" / "phase6_chunk_cache.json"
DIVISORS = (2, 3, 4, 6, 8)
THREADS = 12


def _run_divisor(div: int) -> float | None:
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": str(THREADS),
            "MKL_NUM_THREADS": str(THREADS),
            "ASDSL_PREQ2": "1",
            "ASDSL_CHUNKED_GEMV": "1",
            "ASDSL_AFFINITY": "physical",
            "ASDSL_GEMV_CHUNK_DIV": str(div),
        }
    )
    proc = subprocess.run(
        [sys.executable, str(ROOT / "benchmarks" / "kernel_bench_compare.py")],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )
    if proc.returncode != 0:
        print(f"div={div} failed rc={proc.returncode}", flush=True)
        return None
    bench_path = ROOT / "benchmarks" / "results" / "phaseG_kernel_bench.json"
    if not bench_path.is_file():
        return None
    doc = json.loads(bench_path.read_text(encoding="utf-8"))
    try:
        return float(doc["projections"]["gate_up"]["kernels"]["gemv_preq2_fused_avx2"]["gb_s"])
    except (KeyError, TypeError):
        return None


def run_autotune(*, force: bool = False) -> dict:
    if CACHE_PATH.is_file() and not force:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))

    results: dict[str, float] = {}
    for div in DIVISORS:
        print(f"Sweep chunk_div={div} ...", flush=True)
        gbps = _run_divisor(div)
        if gbps is not None:
            results[str(div)] = gbps
            print(f"  {gbps:.2f} GB/s", flush=True)

    if not results:
        raise SystemExit("chunk autotune: no successful runs")

    best_div = max(results, key=lambda k: results[k])
    doc = {
        "phase": 6,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "threads": THREADS,
        "projection": "gate_up",
        "kernel": "gemv_preq2_fused_avx2",
        "divisors": results,
        "best_div": int(best_div),
        "best_gb_per_s": results[best_div],
        "default_div": 4,
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"Best ASDSL_GEMV_CHUNK_DIV={best_div} ({results[best_div]:.2f} GB/s)")
    print(f"Wrote {CACHE_PATH}")
    return doc


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    run_autotune(force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
