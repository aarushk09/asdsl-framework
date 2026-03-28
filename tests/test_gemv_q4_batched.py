"""Batched packed Q4 GEMV: native (B, K) path matches B sequential (K,) calls."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_gemv_q4_packed_batched_matches_loop() -> None:
    pytest.importorskip("asdsl.kernels._native_gemv")
    from asdsl.kernels.gemv_q4 import gemv_q4_packed, has_native_kernel

    if not has_native_kernel():
        pytest.skip("Native Q4 GEMV not available")

    rng = np.random.default_rng(2026)
    M, K, group_size = 128, 256, 32
    assert K % group_size == 0
    groups_per_row = K // group_size

    w_packed = rng.integers(0, 256, size=M * K // 2, dtype=np.uint8)
    B = 5
    x_batch = rng.standard_normal((B, K)).astype(np.float32)
    scales = rng.uniform(0.01, 0.1, size=M * groups_per_row).astype(np.float32)
    biases = rng.standard_normal(M * groups_per_row).astype(np.float32)

    y_batched = gemv_q4_packed(w_packed, x_batch, scales, biases, M, K, group_size)
    assert y_batched.shape == (B, M)

    y_ref = np.empty((B, M), dtype=np.float32)
    for i in range(B):
        y_ref[i] = gemv_q4_packed(
            w_packed, x_batch[i], scales, biases, M, K, group_size
        )

    assert_allclose(y_batched, y_ref, rtol=1e-4, atol=1e-4)


def test_gemv_q4_avx2_gs64_batched_matches_loop() -> None:
    mod = pytest.importorskip("asdsl.kernels._native_gemv")
    if not (mod.check_avx2() and mod.check_fma()):
        pytest.skip("AVX2/FMA required")

    rng = np.random.default_rng(7)
    rows, cols = 64, 128
    packed_elems = rows * (cols // 2)
    groups = rows * (cols // 64)

    w = rng.integers(0, 256, size=packed_elems, dtype=np.uint8)
    B = 4
    xb = rng.standard_normal((B, cols)).astype(np.float32)
    scales = rng.uniform(0.02, 0.2, size=groups).astype(np.float32)

    y_b = mod.gemv_q4_avx2_gs64(w, xb, scales, rows, cols)
    assert y_b.shape == (B, rows)

    y_ref = np.empty((B, rows), dtype=np.float32)
    for i in range(B):
        y_ref[i] = np.asarray(
            mod.gemv_q4_avx2_gs64(w, xb[i], scales, rows, cols)
        ).ravel()

    assert_allclose(y_b, y_ref, rtol=1e-4, atol=1e-4)
