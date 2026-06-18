"""Numerical parity: preq2 GEMV vs preq Q4_32 on synthetic weights."""

from __future__ import annotations

import numpy as np
import pytest

from asdsl.quantization.repack_preq2 import meta_to_flat, quant_to_flat, repack_preq_blocks_to_preq2
from asdsl.quantization.repack_q4_32 import blocks_to_flat, repack_asymmetric_to_q4_32_blocks


def _synthetic(rows: int, cols: int, gs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_groups = cols // gs
    w_q = rng.integers(0, 16, (rows, cols), dtype=np.uint8)
    packed = ((w_q[:, 1::2] << 4) | w_q[:, 0::2]).astype(np.uint8)
    scales = rng.uniform(0.01, 0.5, (rows, n_groups)).astype(np.float32)
    biases = rng.uniform(-0.1, 0.1, (rows, n_groups)).astype(np.float32)
    blocks = repack_asymmetric_to_q4_32_blocks(packed, scales, biases, rows, cols, gs, bits=4)
    flat = blocks_to_flat(blocks)
    return flat[: rows * n_groups * 20], rng.standard_normal(cols).astype(np.float32)


@pytest.mark.parametrize("shape", [(512, 3072), (3072, 8192), (16384, 3072)])
def test_preq2_matches_preq(shape: tuple[int, int]) -> None:
    ng = pytest.importorskip("asdsl.kernels._native_gemv")
    rows, cols = shape
    gs = 32
    preq, x = _synthetic(rows, cols, gs, 42)
    meta, quant = repack_preq_blocks_to_preq2(preq, rows, cols, gs)
    meta_f = meta_to_flat(meta)
    quant_f = quant_to_flat(quant)
    y_preq = np.zeros(rows, dtype=np.float32)
    y_p2 = np.zeros(rows, dtype=np.float32)
    ng.gemv_q4_32_preq_fused_avx2(preq, x, y_preq, rows, cols, gs)
    ng.gemv_preq2_fused_avx2(meta_f, quant_f, x, y_p2, rows, cols, gs)
    np.testing.assert_allclose(y_p2, y_preq, rtol=1e-4, atol=1e-3)
