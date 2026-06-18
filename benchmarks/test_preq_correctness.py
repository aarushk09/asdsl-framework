"""Preq Q4_32 GEMV numerical parity across kernel variants (Phase G)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _make_case(rows: int, cols: int, gs: int, seed: int):
    from asdsl.quantization.repack_q4_32 import BLOCK_SIZE, blocks_to_flat, repack_asymmetric_to_q4_32_blocks

    rng = np.random.default_rng(seed)
    n_groups = cols // gs
    w_q = rng.integers(0, 16, (rows, cols), dtype=np.uint8)
    packed = ((w_q[:, 1::2] << 4) | w_q[:, 0::2]).astype(np.uint8)
    scales = rng.uniform(0.01, 0.5, (rows, n_groups)).astype(np.float32)
    biases = rng.uniform(-0.1, 0.1, (rows, n_groups)).astype(np.float32)
    blocks = repack_asymmetric_to_q4_32_blocks(packed, scales, biases, rows, cols, gs, bits=4)
    flat = blocks_to_flat(blocks)[: rows * n_groups * BLOCK_SIZE].copy()
    x = rng.standard_normal(cols).astype(np.float32)
    return flat, x, rows, cols, gs


@pytest.fixture
def native_gemv():
    return pytest.importorskip("asdsl.kernels._native_gemv")


PHI_SHAPES = [
    ("gate_up", 16384, 3072),
    ("down_proj", 3072, 8192),
    ("o_proj", 3072, 3072),
    ("small", 64, 256),
]


@pytest.mark.parametrize("label,rows,cols", PHI_SHAPES)
def test_preq_fused_matches_q8_reference(native_gemv, label, rows, cols):
    """Fused and 4-row paths match pre-quantized reference within tolerance."""
    ng = native_gemv
    gs = 32
    blocks, x, out_f, in_f, group_size = _make_case(rows, cols, gs, seed=hash(label) % 10000)

    x_q8 = np.zeros(in_f, dtype=np.int8)
    x_scales = np.zeros(in_f // gs, dtype=np.float32)
    ng.quantize_activation_avx2(x, x_q8, x_scales, in_f, group_size)

    y_ref = np.zeros(out_f, dtype=np.float32)
    ng.gemv_q4_32_preq_avx2(blocks, x_q8, x_scales, y_ref, out_f, in_f, group_size)

    for kname in ("gemv_q4_32_preq_4row_avx2", "gemv_q4_32_preq_8row_avx2", "gemv_q4_32_preq_fused_avx2"):
        fn = getattr(ng, kname)
        y = np.zeros(out_f, dtype=np.float32)
        fn(blocks, x, y, out_f, in_f, group_size)
        np.testing.assert_allclose(y, y_ref, rtol=1e-4, atol=1e-3, err_msg=kname)


def test_quantize_once_per_fused_call(native_gemv):
    """Fused path quantizes activation inside GEMV (no external x_q8)."""
    ng = native_gemv
    blocks, x, out_f, in_f, gs = _make_case(128, 512, 32, seed=7)
    y = np.zeros(out_f, dtype=np.float32)
    ng.gemv_q4_32_preq_fused_avx2(blocks, x, y, out_f, in_f, gs)
    assert np.all(np.isfinite(y))
