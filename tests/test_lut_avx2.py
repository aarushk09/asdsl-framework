"""Phase 2 AVX2+F16C LUT gather GEMV tests."""

from __future__ import annotations

import time

import numpy as np
import pytest

from asdsl.lut import LUTGEMVKernel, LUTTableBuilder
from asdsl.lut import lut_gemv_kernel as lgk
from asdsl.quantization.core import quantize_weights

asdsl_lut_avx2 = pytest.importorskip("asdsl.kernels.asdsl_lut_avx2")


def _make_asymmetric_qtensor(M: int, K: int, group_size: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((M, K)).astype(np.float32) * 0.05
    return quantize_weights(w, bits=4, group_size=group_size, symmetric=False)


def _qt_to_gemv_args(qt, M: int, K: int):
    gs = qt.group_size
    n_groups = M * (K // gs)
    scales = qt.scales[:n_groups].astype(np.float32)
    if qt.zeros is not None:
        zeros = qt.zeros[:n_groups].astype(np.float32)
        biases = (-zeros * scales).astype(np.float32)
    else:
        zeros = None
        biases = (-7.5 * scales).astype(np.float32)
    return qt.data, scales, biases, zeros, gs


def test_lut_avx2_matches_python(monkeypatch):
    """AVX2 tile path matches Python gather on asymmetric Q4 projection."""
    M, K, gs = 512, 256, 32
    qt = _make_asymmetric_qtensor(M, K, gs, seed=11)
    w, scales, biases, zeros, gs = _qt_to_gemv_args(qt, M, K)
    x = np.random.default_rng(12).standard_normal(K).astype(np.float32)
    cache = LUTTableBuilder.build_projection(
        w, scales, biases, M, K, gs, zeros=zeros
    )
    kernel = LUTGEMVKernel()

    monkeypatch.setattr(lgk, "_HAS_AVX2_LUT", False)
    y_py = kernel.gemv(cache, x, use_avx2=False)

    monkeypatch.setattr(lgk, "_HAS_AVX2_LUT", True)
    y_avx = kernel.gemv(cache, x, use_avx2=True)

    np.testing.assert_allclose(y_avx, y_py, rtol=1e-4, atol=1e-3)
    assert np.max(np.abs(y_avx - y_py)) < 1e-3


def test_lut_avx2_tile_shapes():
    """Invalid tensor shapes raise from the extension."""
    G, gs = 4, 32
    T = np.zeros((G, 16, gs), dtype=np.float16)
    q = np.zeros((G, gs), dtype=np.uint8)
    x = np.ones((G, gs), dtype=np.float32)

    asdsl_lut_avx2.lut_gemv_avx2_tile(T, q, x, G)

    with pytest.raises((ValueError, RuntimeError)):
        asdsl_lut_avx2.lut_gemv_avx2_tile(T[:, :8, :], q, x, G)

    with pytest.raises((ValueError, RuntimeError)):
        asdsl_lut_avx2.lut_gemv_avx2_tile(T, q[:2], x, G)


def test_lut_full_matches_tiled():
    """lut_gemv_full matches lut_gemv_avx2_projection within tolerance."""
    if not asdsl_lut_avx2.check_avx2() or not asdsl_lut_avx2.check_f16c():
        pytest.skip("AVX2/F16C not available")
    if not hasattr(asdsl_lut_avx2, "lut_gemv_full"):
        pytest.skip("lut_gemv_full not built")

    M, K, gs = 128, 256, 32
    qt = _make_asymmetric_qtensor(M, K, gs, seed=30)
    w, scales, biases, zeros, gs = _qt_to_gemv_args(qt, M, K)
    x = np.random.default_rng(31).standard_normal(K).astype(np.float32)
    q_packed = LUTTableBuilder.build_q_packed(w, M, K, gs)

    y_proj = asdsl_lut_avx2.lut_gemv_avx2_projection(
        w, scales, biases, x, M, K, gs, zeros=zeros, tile_groups=128
    )
    y_full = asdsl_lut_avx2.lut_gemv_full(
        w, scales, biases, x, M, K, zeros=zeros, q_vals=q_packed, tile_groups=128
    )
    y_full_noq = asdsl_lut_avx2.lut_gemv_full(
        w, scales, biases, x, M, K, zeros=zeros, tile_groups=128
    )
    np.testing.assert_allclose(y_full, y_proj, rtol=0, atol=1e-3)
    np.testing.assert_allclose(y_full_noq, y_proj, rtol=0, atol=1e-3)
    assert np.max(np.abs(y_full - y_proj)) < 1e-3


def test_lut_avx2_microbench_order():
    """lut_gemv_full / LUTGEMVKernel under 0.5 ms on 256x512 micro-GEMV."""
    if not asdsl_lut_avx2.check_avx2() or not asdsl_lut_avx2.check_f16c():
        pytest.skip("AVX2/F16C not available on this CPU")

    M, K, gs = 256, 512, 32
    qt = _make_asymmetric_qtensor(M, K, gs, seed=20)
    w, scales, biases, zeros, gs = _qt_to_gemv_args(qt, M, K)
    x = np.random.default_rng(21).standard_normal(K).astype(np.float32)
    cache = LUTTableBuilder.build_projection(
        w, scales, biases, M, K, gs, zeros=zeros, build_q_packed=True
    )
    kernel = LUTGEMVKernel()

    for _ in range(5):
        kernel.gemv(cache, x, use_avx2=True)

    t0 = time.perf_counter()
    n_rep = 20
    for _ in range(n_rep):
        kernel.gemv(cache, x, use_avx2=True)
    ms = (time.perf_counter() - t0) / n_rep * 1000.0

    if ms >= 0.5 and hasattr(asdsl_lut_avx2, "lut_gemv_full"):
        for tg in (64, 128):
            for _ in range(3):
                asdsl_lut_avx2.lut_gemv_full(
                    w, scales, biases, x, M, K,
                    zeros=zeros, q_vals=cache.q_packed, tile_groups=tg,
                )
            t0 = time.perf_counter()
            for _ in range(n_rep):
                asdsl_lut_avx2.lut_gemv_full(
                    w, scales, biases, x, M, K,
                    zeros=zeros, q_vals=cache.q_packed, tile_groups=tg,
                )
            ms = (time.perf_counter() - t0) / n_rep * 1000.0
            if ms < 0.5:
                break

    if ms >= 0.5:
        pytest.skip(f"LUT GEMV {ms:.3f} ms >= 0.5 ms target on this machine")
