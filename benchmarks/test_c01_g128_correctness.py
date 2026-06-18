"""C0.1 g128 gate_up repack must match imatrix requant reference GEMV."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _requant_g128_row(w_row: np.ndarray, imp: np.ndarray, gs: int = 128) -> np.ndarray:
    from asdsl.quantization.imatrix_lite import importance_weighted_round

    out = np.zeros_like(w_row, dtype=np.float32)
    zero = 8.0
    for g in range(w_row.size // gs):
        xg = w_row[g * gs : (g + 1) * gs]
        ig = imp[g * gs : (g + 1) * gs]
        amax = float(np.max(np.abs(xg)))
        if amax < 1e-12:
            continue
        scale = amax / 7.0
        q = importance_weighted_round(xg.reshape(1, -1), ig, scale, zero, bits=4).ravel()
        out[g * gs : (g + 1) * gs] = (q.astype(np.float32) - zero) * scale
    return out


def test_c01_g128_repack_matches_reference_gemv() -> None:
    ng = pytest.importorskip("asdsl.kernels._native_gemv")
    from asdsl.quantization.repack_q4_128 import blocks_to_flat, repack_fp32_to_q4_128_blocks

    rng = np.random.default_rng(12)
    rows, cols, gs = 64, 3072, 128
    w = rng.standard_normal((rows, cols)).astype(np.float32) * 0.02
    imp = rng.uniform(0.5, 1.0, cols).astype(np.float32)
    w_rq = np.stack([_requant_g128_row(w[r], imp) for r in range(rows)])
    blocks = repack_fp32_to_q4_128_blocks(w_rq, rows, cols, gs)
    flat = blocks_to_flat(blocks)

    x = rng.standard_normal(cols).astype(np.float32)
    y_ref = w_rq @ x
    x_q8 = np.zeros(cols, dtype=np.int8)
    x_sc = np.zeros(cols // gs, dtype=np.float32)
    ng.quantize_activation_avx2(x, x_q8, x_sc, cols, gs)
    y_g128 = np.zeros(rows, dtype=np.float32)
    ng.gemv_q4_128_preq_avx2(flat, x_q8, x_sc, y_g128, rows, cols, gs)
    maxdiff = float(np.max(np.abs(y_ref - y_g128)))
    assert maxdiff < 1.5, f"maxdiff={maxdiff}"


@pytest.mark.slow
def test_c01_gate_up_g128_on_real_weights_layer0() -> None:
    os.environ["ASDSL_C01"] = "1"
    os.environ["ASDSL_GATEUP_GS"] = "128"
    ng = pytest.importorskip("asdsl.kernels._native_gemv")
    from asdsl.quantization.repack_q4_128 import blocks_to_flat, repack_fp32_to_q4_128_blocks
    from experiments.phi4_cpu_run import WeightStore

    store = WeightStore(bits=4)
    store.load()
    store.build_preq_blocks()

    key = (0, "gate_up_proj")
    rows, cols = store._quant_shapes[key]
    r = 128
    w_f32 = store._dequant_from_preq_blocks(key)[:r]
    imp = np.ones(cols, dtype=np.float32)
    w_rq = np.stack([_requant_g128_row(w_f32[i], imp) for i in range(r)])
    blocks = repack_fp32_to_q4_128_blocks(w_rq, r, cols, 128)
    flat = blocks_to_flat(blocks)

    rng = np.random.default_rng(11)
    x = rng.standard_normal(cols).astype(np.float32)
    y_ref = w_rq @ x
    x_q8 = np.zeros(cols, dtype=np.int8)
    x_sc = np.zeros(cols // 128, dtype=np.float32)
    ng.quantize_activation_avx2(x, x_q8, x_sc, cols, 128)
    y_g128 = np.zeros(r, dtype=np.float32)
    ng.gemv_q4_128_preq_avx2(flat, x_q8, x_sc, y_g128, r, cols, 128)
    maxdiff = float(np.max(np.abs(y_ref - y_g128)))
    print(f"real gate_up g128 maxdiff={maxdiff:.4f}")
    assert maxdiff < 2.0
