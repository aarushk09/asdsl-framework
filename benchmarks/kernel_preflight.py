"""Pre-session kernel health check for parity runs (Phase 7).

Compares gate_up preq2 microbench against frozen reference in
``benchmarks/results/kernel_bench_reference.json`` (±15% ms tolerance).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REF_PATH = ROOT / "benchmarks" / "results" / "kernel_bench_reference.json"


def _gate_up_preq2_ms(threads: int = 12) -> tuple[float, float]:
    """Return (ms, gb_s) for one gate_up preq2 GEMV call."""
    sys.path.insert(0, str(ROOT))
    ng = __import__("asdsl.kernels._native_gemv", fromlist=["*"])
    if hasattr(ng, "set_num_threads"):
        ng.set_num_threads(threads)

    from asdsl.quantization.repack_preq2 import meta_to_flat, quant_to_flat, repack_preq_blocks_to_preq2
    from asdsl.quantization.repack_q4_32 import BLOCK_SIZE, blocks_to_flat, repack_asymmetric_to_q4_32_blocks

    out_f, in_f, gs = 16384, 3072, 32
    rng = np.random.default_rng(42)
    n_groups = in_f // gs
    w_q = rng.integers(0, 16, (out_f, in_f), dtype=np.uint8)
    packed = ((w_q[:, 1::2] << 4) | w_q[:, 0::2]).astype(np.uint8)
    scales = rng.uniform(0.01, 0.5, (out_f, n_groups)).astype(np.float32)
    biases = rng.uniform(-0.1, 0.1, (out_f, n_groups)).astype(np.float32)
    blocks = repack_asymmetric_to_q4_32_blocks(packed, scales, biases, out_f, in_f, gs, bits=4)
    flat = blocks_to_flat(blocks)
    nbytes = out_f * n_groups * BLOCK_SIZE
    preq = flat[:nbytes].copy()

    meta, quant = repack_preq_blocks_to_preq2(preq, out_f, in_f, gs)
    meta_f = meta_to_flat(meta)
    quant_f = quant_to_flat(quant)
    x = rng.standard_normal(in_f).astype(np.float32)
    y = np.zeros(out_f, dtype=np.float32)

    block_size = 20
    moved = out_f * n_groups * block_size + in_f * 4 + n_groups * 36 + out_f * 4

    for _ in range(3):
        ng.gemv_preq2_fused_avx2(meta_f, quant_f, x, y, out_f, in_f, gs)
    t0 = time.perf_counter()
    for _ in range(10):
        ng.gemv_preq2_fused_avx2(meta_f, quant_f, x, y, out_f, in_f, gs)
    sec = (time.perf_counter() - t0) / 10.0
    return sec * 1000.0, (moved / sec) / 1e9


def run_kernel_preflight(
    threads: int | None = None,
    reference_path: Path = REF_PATH,
    *,
    skip: bool = False,
) -> dict:
    """Return preflight result; raises RuntimeError on hard fail."""
    if skip or os.environ.get("ASDSL_SKIP_KERNEL_PREFLIGHT", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return {"skipped": True, "ok": True}

    if not reference_path.is_file():
        return {"ok": True, "warning": f"missing reference {reference_path}"}

    ref_doc = json.loads(reference_path.read_text(encoding="utf-8"))
    ref = ref_doc["gate_up_preq2"]
    ref_ms = float(ref["ms"])
    tol = float(ref.get("tolerance_pct", 15.0))
    t = threads or int(os.environ.get("OMP_NUM_THREADS", "12"))

    for k, v in (ref_doc.get("env") or {}).items():
        os.environ.setdefault(k, str(v))

    measured_ms, measured_gb_s = _gate_up_preq2_ms(t)
    lo = ref_ms * (1.0 - tol / 100.0)
    hi = ref_ms * (1.0 + tol / 100.0)
    # Fail only when slower than reference (throttle / regression). Faster is OK.
    ok = measured_ms <= hi
    result = {
        "ok": ok,
        "threads": t,
        "reference_ms": ref_ms,
        "measured_ms": round(measured_ms, 3),
        "measured_gb_s": round(measured_gb_s, 2),
        "tolerance_pct": tol,
        "allowed_ms_max": round(hi, 3),
    }
    if measured_ms < lo:
        result["faster_than_reference"] = True
    if not ok:
        result["error"] = (
            f"gate_up preq2 {measured_ms:.3f} ms > {hi:.3f} ms "
            f"(ref {ref_ms} ms +{tol}% — possible throttle or regression)"
        )
    return result


if __name__ == "__main__":
    r = run_kernel_preflight()
    print(json.dumps(r, indent=2))
    if not r.get("ok"):
        raise SystemExit(1)
