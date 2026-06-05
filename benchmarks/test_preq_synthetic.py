#!/usr/bin/env python3
"""Synthetic gate_up-sized GEMV: packed vs preq without full model load."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ["OMP_NUM_THREADS"] = "8"

from asdsl.kernels.gemv_q4 import gemv_q4_packed
from asdsl.quantization.core import quantize_weights
from asdsl.quantization.repack_q4_32 import repack_asymmetric_to_q4_32_blocks, blocks_to_flat


def main() -> None:
    rows, cols, gs = 16384, 3072, 32
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
    x = rng.standard_normal(cols).astype(np.float32)
    out_a = np.empty(rows, dtype=np.float32)
    out_b = np.empty(rows, dtype=np.float32)

    from asdsl.kernels import _native_gemv as ng

    for _ in range(3):
        gemv_q4_packed(packed, x, sc, bi, rows, cols, gs, out=out_a)
    t0 = time.perf_counter()
    for _ in range(10):
        gemv_q4_packed(packed, x, sc, bi, rows, cols, gs, out=out_a)
    packed_ms = (time.perf_counter() - t0) / 10 * 1000

    x_q8 = np.empty(cols, dtype=np.int8)
    x_sc = np.empty(cols // gs, dtype=np.float32)
    ng.quantize_activation_avx2(x, x_q8, x_sc, cols, gs)
    for _ in range(3):
        ng.gemv_q4_32_preq_avx2(blocks, x_q8, x_sc, out_b, rows, cols, gs)
    t0 = time.perf_counter()
    for _ in range(10):
        ng.gemv_q4_32_preq_avx2(blocks, x_q8, x_sc, out_b, rows, cols, gs)
    preq_ms = (time.perf_counter() - t0) / 10 * 1000

    bytes_p = packed.nbytes + sc.nbytes + bi.nbytes + x.nbytes
    bytes_r = blocks.nbytes + x_q8.nbytes + x_sc.nbytes
    gb_p = bytes_p / (packed_ms / 1000.0) / 1e9
    gb_r = bytes_r / (preq_ms / 1000.0) / 1e9
    print(f"packed {packed_ms:.1f} ms  {gb_p:.2f} GB/s")
    print(f"preq   {preq_ms:.1f} ms  {gb_r:.2f} GB/s")
    from asdsl.quantization.core import _unpack_bits

    ng_per_row = cols // gs
    packed2d = packed.reshape(rows, cols // 2)
    ref_q8 = np.zeros(rows, dtype=np.float32)
    for m in range(rows):
        acc = 0.0
        for g in range(ng_per_row):
            gidx = m * ng_per_row + g
            scale = float(sc[gidx])
            bias = float(bi[gidx])
            q = _unpack_bits(packed2d[m], 4)[g * gs : (g + 1) * gs].astype(np.float32)
            xg = x_q8[g * gs : (g + 1) * gs].astype(np.float32) * x_sc[g]
            acc += scale * float(np.dot(q, xg)) + bias * float(xg.sum())
        ref_q8[m] = acc
    rel = np.abs(ref_q8 - out_b).max() / (np.abs(ref_q8).max() + 1e-6)
    rel_packed = np.abs(out_a - out_b).max() / (np.abs(out_a).max() + 1e-6)
    print(f"max rel err vs packed+Q8 ref: {rel:.6f}")
    print(f"max rel err vs packed float: {rel_packed:.4f}")
    gate = gb_r >= 18
    print(f"preq >= 18 GB/s: {gate}")
    return 0 if rel < 0.001 else 1


if __name__ == "__main__":
    raise SystemExit(main())
