"""Kernel-level GEMV throughput for Phi-4 projection shapes (Phase G)."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Phi-4 Q4_32 preq projection slices
PROJECTIONS = {
    "gate_up": {"out_features": 16384, "in_features": 3072, "group_size": 32},
    "down_proj": {"out_features": 3072, "in_features": 8192, "group_size": 32},
    "o_proj": {"out_features": 3072, "in_features": 3072, "group_size": 32},
    "lm_head": {"out_features": 200064, "in_features": 3072, "group_size": 32},
}

KERNELS = (
    "gemv_q4_32_preq_4row_avx2",
    "gemv_q4_32_preq_8row_avx2",
    "gemv_q4_32_preq_fused_avx2",
    "gemv_preq2_fused_avx2",
)


def _make_synthetic_blocks(rows: int, cols: int, gs: int, rng: np.random.Generator) -> np.ndarray:
    from asdsl.quantization.repack_q4_32 import BLOCK_SIZE, blocks_to_flat, repack_asymmetric_to_q4_32_blocks

    n_groups = cols // gs
    w_q = rng.integers(0, 16, (rows, cols), dtype=np.uint8)
    packed = ((w_q[:, 1::2] << 4) | w_q[:, 0::2]).astype(np.uint8)
    scales = rng.uniform(0.01, 0.5, (rows, n_groups)).astype(np.float32)
    biases = rng.uniform(-0.1, 0.1, (rows, n_groups)).astype(np.float32)
    blocks = repack_asymmetric_to_q4_32_blocks(packed, scales, biases, rows, cols, gs, bits=4)
    flat = blocks_to_flat(blocks)
    nbytes = rows * n_groups * BLOCK_SIZE
    return flat[:nbytes].copy()


def _bytes_moved(out_f: int, in_f: int, gs: int) -> int:
    n_groups = in_f // gs
    block_size = 20
    weight_bytes = out_f * n_groups * block_size
    act_bytes = in_f * 4 + n_groups * (32 + 4)  # fp32 in + q8 quant overhead
    out_bytes = out_f * 4
    return weight_bytes + act_bytes + out_bytes


def bench_kernel(fn, blocks: np.ndarray, x: np.ndarray, y: np.ndarray, out_f: int, in_f: int, gs: int,
                 warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn(blocks, x, y, out_f, in_f, gs)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(blocks, x, y, out_f, in_f, gs)
    elapsed = time.perf_counter() - t0
    return elapsed / iters


def main() -> None:
    ng = __import__("asdsl.kernels._native_gemv", fromlist=["*"])
    threads = int(os.environ.get("OMP_NUM_THREADS", "12"))
    if hasattr(ng, "set_num_threads"):
        ng.set_num_threads(threads)

    rng = np.random.default_rng(42)
    results: dict = {
        "phase": "G",
        "threads": threads,
        "env": {
            "ASDSL_GEMV_UNROLL": os.environ.get("ASDSL_GEMV_UNROLL", "4"),
            "ASDSL_PREQ_PREFETCH_GROUPS": os.environ.get("ASDSL_PREQ_PREFETCH_GROUPS", "8"),
            "ASDSL_PREQ2": os.environ.get("ASDSL_PREQ2", ""),
            "ASDSL_AFFINITY": os.environ.get("ASDSL_AFFINITY", ""),
            "ASDSL_CHUNKED_GEMV": os.environ.get("ASDSL_CHUNKED_GEMV", ""),
        },
        "projections": {},
    }

    for name, shape in PROJECTIONS.items():
        out_f = shape["out_features"]
        in_f = shape["in_features"]
        gs = shape["group_size"]
        blocks = _make_synthetic_blocks(out_f, in_f, gs, rng)
        x = rng.standard_normal(in_f).astype(np.float32)
        y = np.zeros(out_f, dtype=np.float32)
        moved = _bytes_moved(out_f, in_f, gs)
        proj: dict = {"shape": shape, "bytes_per_call": moved, "kernels": {}}

        for kname in KERNELS:
            fn = getattr(ng, kname, None)
            if fn is None:
                continue
            if kname == "gemv_preq2_fused_avx2":
                from asdsl.quantization.repack_preq2 import meta_to_flat, quant_to_flat, repack_preq_blocks_to_preq2

                meta, quant = repack_preq_blocks_to_preq2(blocks, out_f, in_f, gs)
                meta_f = meta_to_flat(meta)
                quant_f = quant_to_flat(quant)

                def _preq2_fn(_b, _x, _y, of, inf, g):
                    fn(meta_f, quant_f, _x, _y, of, inf, g)

                sec = bench_kernel(_preq2_fn, blocks, x, y, out_f, in_f, gs)
            else:
                sec = bench_kernel(fn, blocks, x, y, out_f, in_f, gs)
            gbps = (moved / sec) / 1e9
            proj["kernels"][kname] = {"ms": round(sec * 1000, 3), "gb_s": round(gbps, 2)}

        fused = proj["kernels"].get("gemv_q4_32_preq_fused_avx2", {})
        row4 = proj["kernels"].get("gemv_q4_32_preq_4row_avx2", {})
        if fused and row4:
            proj["speedup_vs_4row"] = round(row4["ms"] / fused["ms"], 3)
        results["projections"][name] = proj
        use_preq2 = os.environ.get("ASDSL_PREQ2", "0").strip().lower() not in ("0", "false", "no")
        prod = proj["kernels"].get("gemv_preq2_fused_avx2", {}) if use_preq2 else fused
        prod_label = "preq2" if use_preq2 else "preq_fused"
        print(f"{name}: {prod_label} {prod.get('gb_s', '?')} GB/s ({prod.get('ms', '?')} ms)")

    out_path = ROOT / "benchmarks" / "results" / "phaseG_kernel_bench.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
