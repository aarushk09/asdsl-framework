"""Microbench: preq baseline vs g4fused on gate_up-sized GEMV."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def bench_preq(fn_name: str, rows: int = 16384, cols: int = 3072, n: int = 25) -> dict:
    from asdsl.kernels import _native_gemv as ng

    gs = 32
    n_groups = cols // gs
    rng = np.random.default_rng(1)
    blocks = rng.integers(0, 256, (rows, n_groups, 20), dtype=np.uint8)
    for g in range(n_groups):
        blocks[:, g, 0:2] = np.array([0x00, 0x3C], dtype=np.uint8)
        blocks[:, g, 2:4] = np.array([0x00, 0x00], dtype=np.uint8)
    flat = np.ascontiguousarray(blocks.reshape(-1))
    x = rng.standard_normal(cols).astype(np.float32)
    y = np.empty(rows, dtype=np.float32)
    fn = getattr(ng, fn_name)
    for _ in range(5):
        fn(flat, x, y, rows, cols, gs)
    t0 = time.perf_counter()
    for _ in range(n):
        fn(flat, x, y, rows, cols, gs)
    ms = (time.perf_counter() - t0) / n * 1000.0
    bytes_moved = flat.nbytes + x.nbytes
    gb_s = bytes_moved / (ms / 1000.0) / 1e9
    return {"fn": fn_name, "rows": rows, "ms": round(ms, 2), "gb_s": round(gb_s, 2)}


def main() -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "12")
    os.environ.setdefault("MKL_NUM_THREADS", "12")
    os.environ["ASDSL_PREQ_G4FUSED"] = "0"
    base = bench_preq("gemv_q4_32_preq_4row_avx2")
    g4 = bench_preq("gemv_q4_32_preq_g4fused_4row_avx2")
    out = {"preq_4row": base, "preq_g4fused_4row": g4}
    pct = 100.0 * (g4["gb_s"] - base["gb_s"]) / base["gb_s"] if base["gb_s"] else 0
    out["g4_vs_base_pct"] = round(pct, 1)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
