"""Measure dual-stream DRAM bandwidth (Gate A2 probe)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    from asdsl.profiler import measure_dual_stream_bandwidth, measure_stream_triad_bandwidth

    ap = argparse.ArgumentParser(description="Dual-stream STREAM triad probe")
    ap.add_argument("--size-a-mb", type=int, default=2400, help="Q4-sized array (MB)")
    ap.add_argument("--size-b-mb", type=int, default=750, help="Q2 draft-sized array (MB)")
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=ROOT / "benchmarks" / "results" / "dual_stream_probe.json",
    )
    args = ap.parse_args()

    single_a = measure_stream_triad_bandwidth(
        array_mb=min(args.size_a_mb, 512), runs=args.runs, warmup_runs=args.warmup
    )
    dual = measure_dual_stream_bandwidth(
        size_a_mb=args.size_a_mb,
        size_b_mb=args.size_b_mb,
        runs=args.runs,
        warmup_runs=args.warmup,
    )

    gate_pass = (
        dual.combined_bandwidth_gb_s >= 35.0
        and dual.retention_a_pct >= 80.0
        and dual.retention_b_pct >= 80.0
    )

    doc = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "size_a_mb": args.size_a_mb,
        "size_b_mb": args.size_b_mb,
        "runs": args.runs,
        "single_stream_a_gb_s": single_a.bandwidth_gb_s,
        "dual_stream": {
            "bandwidth_a_gb_s": dual.bandwidth_a_gb_s,
            "bandwidth_b_gb_s": dual.bandwidth_b_gb_s,
            "combined_bandwidth_gb_s": dual.combined_bandwidth_gb_s,
            "retention_a_pct": dual.retention_a_pct,
            "retention_b_pct": dual.retention_b_pct,
            "elapsed_sec": dual.elapsed_sec,
        },
        "gate_a2": {
            "combined_min_gb_s": 35.0,
            "retention_min_pct": 80.0,
            "pass": gate_pass,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    print(f"combined={dual.combined_bandwidth_gb_s:.2f} GB/s")
    print(f"retention_a={dual.retention_a_pct:.1f}% retention_b={dual.retention_b_pct:.1f}%")
    print(f"gate_a2={'PASS' if gate_pass else 'FAIL'}")
    print(f"Wrote {args.output}")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
