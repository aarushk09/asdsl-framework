"""N-thread parallel DRAM read bandwidth probe (Phase 0 ceiling measurement)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _parallel_read_gbps(array_mb: int, threads: int, runs: int = 6, warmup: int = 2) -> float:
    nbytes = array_mb * 1024 * 1024
    buf = np.empty(nbytes, dtype=np.uint8)
    buf.fill(0xA5)
    indices = np.arange(0, nbytes, 4096, dtype=np.int64)

    old_omp = os.environ.get("OMP_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = str(threads)
    try:
        from asdsl.kernels import _native_forward as nf

        if hasattr(nf, "stream_read_bandwidth"):
            raw = nf.stream_read_bandwidth(int(array_mb), int(threads), int(runs), int(warmup))
            return float(raw["bandwidth_gb_s"])
    except Exception:
        pass
    finally:
        if old_omp is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = old_omp

    # NumPy fallback: sum strided reads (approximate)
    t0 = time.perf_counter()
    for _ in range(warmup):
        _ = int(buf[indices].sum())
    t0 = time.perf_counter()
    for _ in range(runs):
        _ = int(buf[indices].sum())
    elapsed = time.perf_counter() - t0
    moved = nbytes * runs
    return moved / elapsed / 1e9


def main() -> int:
    ap = argparse.ArgumentParser(description="Memory bandwidth ceiling probe")
    ap.add_argument("--array-mb", type=int, default=2048)
    ap.add_argument("-o", type=Path, default=ROOT / "benchmarks" / "results" / "membw_probe.json")
    args = ap.parse_args()

    placements = {
        "4P_only": 4,
        "8E_only": 8,
        "4P_plus_8E": 12,
        "16_logical": 16,
    }
    results = {}
    for label, threads in placements.items():
        gbps = _parallel_read_gbps(args.array_mb, threads)
        results[label] = {"threads": threads, "bandwidth_gb_s": round(gbps, 2)}
        print(f"{label} ({threads}t): {gbps:.2f} GB/s")

    bw_ceiling = max(r["bandwidth_gb_s"] for r in results.values())
    doc = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "array_mb": args.array_mb,
        "placements": results,
        "bw_ceiling_gb_s": bw_ceiling,
    }
    args.o.parent.mkdir(parents=True, exist_ok=True)
    args.o.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"BW_ceiling={bw_ceiling:.2f} GB/s -> {args.o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
