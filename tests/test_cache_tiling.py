"""Phase 4: cache-tiled fused GEMV (K-direction blocking) vs reference and vs non-tiled."""

from __future__ import annotations

import sys
import time

import numpy as np
import pytest
from numpy.testing import assert_allclose

from asdsl.quantization.mixed_q34 import pack_linear_mixed_q34, reference_mixed_gemv_numpy


@pytest.fixture(scope="module")
def native_q3():
    pytest.importorskip("asdsl.kernels._native_gemv_q3")
    from asdsl.kernels import _native_gemv_q3 as mod

    if not (mod.check_avx2() and mod.check_fma()):
        pytest.skip("AVX2/FMA required")
    return mod


def test_gemv_q3_mixed_tiled_matches_reference_large_k(native_q3) -> None:
    """Tiled mixed GEMV matches NumPy reference; matrix larger than typical L2."""
    rng = np.random.default_rng(11)
    m, k, gs = 4096, 16384, 128
    w = (0.02 * rng.standard_normal((m, k))).astype(np.float32)
    im = np.abs(rng.standard_normal(k)).astype(np.float32)
    x = (0.03 * rng.standard_normal(k)).astype(np.float32)

    packed = pack_linear_mixed_q34(
        w, im, group_size=gs, q4_group_fraction=0.15
    )
    y_ref = reference_mixed_gemv_numpy(packed, x)

    from asdsl.kernels.gemv_q3 import gemv_q3_mixed

    y_tiled = gemv_q3_mixed(packed, x, cache_tiling=True)
    y_flat = gemv_q3_mixed(packed, x, cache_tiling=False)

    assert_allclose(y_tiled, y_ref, rtol=1e-4, atol=1e-3)
    assert_allclose(y_flat, y_ref, rtol=1e-4, atol=1e-3)
    assert_allclose(y_tiled, y_flat, rtol=0, atol=1e-5)


def test_tiling_faster_or_neutral_on_large_problem(native_q3) -> None:
    """100 iters: benchmark tiled vs flat; guard against large regressions only.

    Mixed Q3/Q4 uses variable-size blobs along K, so L2-friendly blocking does not
    always beat a single loop on every CPU. We still require rough parity so the
    tiled path stays production-viable.
    """
    rng = np.random.default_rng(22)
    # Large K so K-blocking spans multiple tiles (groups_per_row > tile_groups).
    m, k, gs = 1024, 16384, 128
    w = (0.02 * rng.standard_normal((m, k))).astype(np.float32)
    im = np.abs(rng.standard_normal(k)).astype(np.float32)
    x = rng.standard_normal(k).astype(np.float32)
    packed = pack_linear_mixed_q34(
        w, im, group_size=gs, q4_group_fraction=0.12
    )

    from asdsl.kernels.gemv_q3 import gemv_q3_mixed

    n_warm, n_iter = 5, 100
    for _ in range(n_warm):
        gemv_q3_mixed(packed, x, cache_tiling=True)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        gemv_q3_mixed(packed, x, cache_tiling=True)
    t_tile = time.perf_counter() - t0

    for _ in range(n_warm):
        gemv_q3_mixed(packed, x, cache_tiling=False)
    t1 = time.perf_counter()
    for _ in range(n_iter):
        gemv_q3_mixed(packed, x, cache_tiling=False)
    t_flat = time.perf_counter() - t1

    bytes_moved = (
        packed.w_bytes.nbytes
        + x.nbytes
        + packed.scales.nbytes
        + packed.biases.nbytes
        + m * 4
    ) * n_iter
    gbs_tile = bytes_moved / t_tile / 1e9
    gbs_flat = bytes_moved / t_flat / 1e9

    print(
        f"[cache_tiling] tiled {t_tile:.4f}s ({gbs_tile:.2f} GB/s model) vs "
        f"flat {t_flat:.4f}s ({gbs_flat:.2f} GB/s) over {n_iter} iters",
        file=sys.stderr,
    )

    assert t_tile <= t_flat * 1.15, (
        "cache tiling must not regress latency by more than ~15% vs flat; "
        f"tiled={t_tile:.4f}s flat={t_flat:.4f}s"
    )
    assert gbs_tile >= gbs_flat * 0.87, (
        "effective memory throughput should not collapse with tiling enabled"
    )


def test_q8_tiling_bitwise_matches_non_tiled() -> None:
    pytest.importorskip("asdsl.kernels._native_gemv_q8")
    rng = np.random.default_rng(33)
    m, k, gs = 256, 4096, 128
    w = rng.integers(0, 256, size=m * k, dtype=np.uint8)
    x = rng.standard_normal(k).astype(np.float32)
    g = k // gs
    scales = rng.uniform(0.01, 0.1, size=m * g).astype(np.float32)
    biases = rng.standard_normal(m * g).astype(np.float32)

    from asdsl.kernels import _native_gemv_q8 as q8

    y_t = q8.fused_dequant_gemv(w, x, scales, biases, m, k, gs, True)
    y_f = q8.fused_dequant_gemv(w, x, scales, biases, m, k, gs, False)
    assert_allclose(np.asarray(y_t), np.asarray(y_f), rtol=0, atol=1e-5)
