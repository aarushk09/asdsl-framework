"""Phase 1: thread × affinity × chunked GEMV sweep (kernel proxy).

Writes gate_up preq2 GB/s per combo to ``phase1_thread_sweep.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH_JSON = ROOT / "benchmarks" / "results" / "phaseG_kernel_bench.json"


def _read_gate_up_preq2() -> dict | None:
    if not BENCH_JSON.is_file():
        return None
    doc = json.loads(BENCH_JSON.read_text(encoding="utf-8"))
    gu = doc.get("projections", {}).get("gate_up", {})
    k = gu.get("kernels", {}).get("gemv_preq2_fused_avx2")
    return k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-o",
        type=Path,
        default=ROOT / "benchmarks" / "results" / "phase1_thread_sweep.json",
    )
    ap.add_argument("--quick", action="store_true", help="physical affinity + chunked only")
    args = ap.parse_args()

    threads = [8, 12, 16] if args.quick else [8, 10, 12, 14, 16]
    affinities = ["physical"] if args.quick else ["physical", "spread", "legacy"]
    chunked = ["1"] if args.quick else ["0", "1"]
    results: list[dict] = []

    for t in threads:
        for aff in affinities:
            for ch in chunked:
                env = os.environ.copy()
                env.update(
                    {
                        "OMP_NUM_THREADS": str(t),
                        "MKL_NUM_THREADS": str(t),
                        "OPENBLAS_NUM_THREADS": str(t),
                        "ASDSL_AFFINITY": aff,
                        "ASDSL_CHUNKED_GEMV": ch,
                        "ASDSL_USE_UNIFIED": "1",
                        "ASDSL_PREQ2": "1",
                        "ASDSL_FUSED_GEMV": "1",
                    }
                )
                cmd = [sys.executable, str(ROOT / "benchmarks" / "kernel_bench_compare.py")]
                proc = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=ROOT)
                gate = _read_gate_up_preq2()
                entry = {
                    "threads": t,
                    "affinity": aff,
                    "chunked": ch == "1",
                    "returncode": proc.returncode,
                    "gate_up_preq2": gate,
                }
                results.append(entry)
                gb = (gate or {}).get("gb_s", "?")
                print(f"{t}t {aff} chunked={ch}: rc={proc.returncode} gate_up={gb} GB/s")

    best = max(
        (r for r in results if r.get("gate_up_preq2")),
        key=lambda r: r["gate_up_preq2"]["gb_s"],
        default=None,
    )
    doc = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": 1,
        "quick_mode": args.quick,
        "best_gate_up_preq2": best,
        "results": results,
    }
    args.o.parent.mkdir(parents=True, exist_ok=True)
    args.o.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"Wrote {args.o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
