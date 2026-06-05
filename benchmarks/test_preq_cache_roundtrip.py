#!/usr/bin/env python3
"""Quick round-trip test for preq safetensors cache (no full model)."""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from asdsl.quantization.core import quantize_weights
from asdsl.quantization.repack_q4_32 import (
    BLOCK_SIZE,
    blocks_to_flat,
    repack_asymmetric_to_q4_32_blocks,
)
from phi4_cpu_run import (
    NUM_LAYERS,
    WeightStore,
    preq_cache_path_for_store,
    save_preq_cache,
    try_restore_preq_cache,
)


def main() -> int:
    rows, cols, gs = 64, 3072, 32
    rng = np.random.default_rng(0)
    w = rng.standard_normal((rows, cols)).astype(np.float32) * 0.02
    qt = quantize_weights(w, bits=4, group_size=gs, symmetric=False, optimize_clips=True)
    packed = np.ascontiguousarray(qt.data.ravel(), dtype=np.uint8)
    n_groups = rows * cols // gs
    sc = qt.scales[:n_groups].astype(np.float32)
    zr = qt.zeros[:n_groups].astype(np.float32)
    bi = (-zr * sc).astype(np.float32)
    blocks = blocks_to_flat(
        repack_asymmetric_to_q4_32_blocks(packed, sc, bi, rows, cols, gs, bits=4)
    )

    store = WeightStore(bits=4, group_size=gs)
    store._quant_shapes = {(0, "gate_up_proj"): (rows, cols)}
    store._preq_blocks_np = {(0, "gate_up_proj"): blocks}
    store._preq_built = True
    store._preq_block_size = BLOCK_SIZE

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "test_preq.safetensors"
        save_preq_cache(store, path)
        store2 = WeightStore(bits=4, group_size=gs)
        store2._quant_shapes = store._quant_shapes.copy()
        store2._preq_blocks_np.clear()
        store2._preq_built = False
        t0 = time.perf_counter()
        ok = try_restore_preq_cache(store2, path)
        dt = time.perf_counter() - t0
    if not ok:
        print("FAIL: restore returned False")
        return 1
    got = store2._preq_blocks_np[(0, "gate_up_proj")]
    if not np.array_equal(got, blocks):
        print("FAIL: data mismatch after round-trip")
        return 1
    print(f"PASS: round-trip ok, restore {dt*1000:.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
