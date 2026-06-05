"""Phase 1 LUT-native GEMV: table build, kernel, tiling, residency, dispatch."""

from __future__ import annotations

import numpy as np
import pytest

from asdsl.kernels.gemv_q4 import gemv_q4_packed
from asdsl.lut import (
    LUTGEMVKernel,
    LUTKernelDispatcher,
    LUTTableBuilder,
    should_use_lut,
)
from asdsl.lut.lut_table_builder import LUTProjectionCache
from asdsl.quantization.core import dequantize_weights, quantize_weights

from test_lut_gemv_correctness import dequant_q4_ref, make_test_case


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
        half = 7.5
        biases = (-half * scales).astype(np.float32)
    return qt.data, scales, biases, zeros, gs


class TestLUTTableConstruction:
    def test_lut_table_construction_matches_dequant(self):
        M, K, gs = 8, 128, 32
        qt = _make_asymmetric_qtensor(M, K, gs, seed=1)
        w, scales, biases, zeros, gs = _qt_to_gemv_args(qt, M, K)
        dequant_full = dequantize_weights(qt).reshape(M, K)

        groups_per_row = K // gs
        for row in range(M):
            for tile_idx in range(LUTTableBuilder.num_k_tiles(groups_per_row)):
                T = LUTTableBuilder.build_row_tile(
                    row,
                    tile_idx,
                    scales,
                    biases,
                    groups_per_row,
                    gs,
                    zeros=zeros,
                )
                g0 = tile_idx * LUTTableBuilder.DEFAULT_TILE_GROUPS
                for g_local in range(min(LUTTableBuilder.DEFAULT_TILE_GROUPS, groups_per_row - g0)):
                    g = g0 + g_local
                    z = float(zeros[row * groups_per_row + g]) if zeros is not None else -float(
                        biases[row * groups_per_row + g]
                    ) / float(scales[row * groups_per_row + g])
                    s = float(scales[row * groups_per_row + g])
                    for q in range(16):
                        expected = (q - z) * s
                        np.testing.assert_allclose(
                            T[g_local, q, :],
                            expected,
                            rtol=1e-3,
                            atol=1e-3,
                        )
                    # T is constant along i for fixed (g, q)
                    np.testing.assert_allclose(
                        T[g_local, q, 0],
                        T[g_local, q, -1],
                        rtol=0,
                        atol=0,
                    )


class TestLUTGEMV:
    def test_lut_gemv_matches_reference(self):
        M, K, gs = 64, 256, 32
        qt = _make_asymmetric_qtensor(M, K, gs, seed=2)
        w, scales, biases, zeros, gs = _qt_to_gemv_args(qt, M, K)
        x = np.random.default_rng(3).standard_normal(K).astype(np.float32)

        cache = LUTTableBuilder.build_projection(
            w, scales, biases, M, K, gs, zeros=zeros
        )
        y_lut = LUTGEMVKernel().gemv(cache, x)
        y_ref = dequant_q4_ref(w, scales, biases, x, M, K, gs)
        np.testing.assert_allclose(y_lut, y_ref, rtol=1e-4, atol=1e-3)

    def test_lut_tiling_consistency(self):
        M, K, gs = 32, 512, 32
        qt = _make_asymmetric_qtensor(M, K, gs, seed=4)
        w, scales, biases, zeros, gs = _qt_to_gemv_args(qt, M, K)
        x = np.random.default_rng(5).standard_normal(K).astype(np.float32)

        cache = LUTTableBuilder.build_projection(
            w, scales, biases, M, K, gs, zeros=zeros
        )
        y128 = LUTGEMVKernel(tile_groups=128).gemv(cache, x, tile_groups=128)
        y64 = LUTGEMVKernel(tile_groups=64).gemv(cache, x, tile_groups=64)
        np.testing.assert_allclose(y128, y64, rtol=1e-5, atol=1e-4)


class TestLUTResidency:
    def test_lut_l2_residency(self):
        assert LUTTableBuilder.footprint_bytes(128, 32) == 131072
        assert LUTTableBuilder.footprint_bytes(64, 32) == 65536


class TestLutAvx2Integration:
    def test_lut_avx2_matches_python(self):
        pytest.importorskip("asdsl.kernels.asdsl_lut_avx2")
        from asdsl.lut.lut_gemv_kernel import _HAS_AVX2_LUT

        if not _HAS_AVX2_LUT:
            pytest.skip("AVX2 LUT not available on this CPU/build")

        M, K, gs = 512, 256, 32
        qt = _make_asymmetric_qtensor(M, K, gs, seed=20)
        w, scales, biases, zeros, gs = _qt_to_gemv_args(qt, M, K)
        x = np.random.default_rng(21).standard_normal(K).astype(np.float32)
        cache = LUTTableBuilder.build_projection(
            w, scales, biases, M, K, gs, zeros=zeros
        )
        kernel = LUTGEMVKernel()
        y_py = kernel.gemv(cache, x, use_avx2=False)
        y_avx = kernel.gemv(cache, x, use_avx2=True)
        np.testing.assert_allclose(y_avx, y_py, rtol=0, atol=1e-3)


class TestLUTDispatcher:
    def test_lut_dispatcher_routing(self):
        assert should_use_lut(4, 32, 64, 128) is True
        assert should_use_lut(4, 64, 64, 128) is False
        assert should_use_lut(8, 32, 64, 128) is False
        assert should_use_lut(4, 32, 0, 128) is False

        M, K, gs = 16, 64, 32
        wp, sc, bi, x = make_test_case(M, K, gs, seed=10)
        cache = LUTTableBuilder.build_projection(wp, sc, bi, M, K, gs)

        y_cached = gemv_q4_packed(
            wp, x, sc, bi, M, K, gs, use_lut=True, lut_cache=cache, bits=4
        )
        y_ref = gemv_q4_packed(wp, x, sc, bi, M, K, gs, use_lut=False)
        np.testing.assert_allclose(y_cached, y_ref, rtol=1e-3, atol=2e-3)

        # Without cache: should still return valid output (FMA or legacy LUT)
        y_no_cache = gemv_q4_packed(
            wp, x, sc, bi, M, K, gs, use_lut=True, lut_cache=None, bits=4
        )
        assert y_no_cache.shape == (M,)
        assert not np.any(np.isnan(y_no_cache))

        y_policy_off = LUTKernelDispatcher.dispatch(
            wp, x, sc, bi, M, K, gs,
            lut_cache=cache,
            bits=8,
            use_lut=True,
            _native_available=False,
            _gemv_numpy=lambda *a, **k: np.zeros(M, np.float32),
        )
        np.testing.assert_array_equal(y_policy_off, np.zeros(M))
