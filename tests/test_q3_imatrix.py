"""Mixed Q3/Q4 imatrix packer + native ``gemv_q3_mixed`` tests."""

from __future__ import annotations

import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose

from asdsl.quantization.mixed_q34 import (
    pack_linear_mixed_q34,
    q4_baseline_weight_bytes,
    reference_mixed_gemv_numpy,
)


@pytest.fixture(scope="module")
def native_q3():
    pytest.importorskip("asdsl.kernels._native_gemv_q3")
    from asdsl.kernels import _native_gemv_q3 as mod

    if not (mod.check_avx2() and mod.check_fma()):
        pytest.skip("CPU lacks AVX2/FMA")
    return mod


def test_footprint_reduction_vs_q4_baseline() -> None:
    rng = np.random.default_rng(1)
    m, k, gs = 128, 1024, 128
    w = (0.02 * rng.standard_normal((m, k))).astype(np.float32)
    im = np.abs(rng.standard_normal(k)).astype(np.float32)

    base_bytes = q4_baseline_weight_bytes(m, k)
    # ~8% of groups stay Q4; rest Q3. With gs=128, all-Q3 is ~52 B/group vs 64 for Q4
    # (~18.75% best case); a small Q4 tail keeps the mix realistic.
    packed = pack_linear_mixed_q34(
        w, im, group_size=gs, q4_group_fraction=0.08
    )
    mixed_bytes = packed.weight_bytes()
    ratio = mixed_bytes / float(base_bytes)
    print(
        f"[footprint] q4_baseline={base_bytes} B mixed={mixed_bytes} B ratio={ratio:.3f}",
        file=sys.stderr,
    )
    assert ratio < 1.0
    assert ratio <= 0.84, (
        f"expected measurable reduction vs packed Q4 baseline (theoretical max ~19% at gs=128); "
        f"got ratio {ratio}"
    )


def test_logit_quality_vs_uniform_q4(native_q3) -> None:
    """Mixed Q3/Q4 should stay close to uniform Q4 on the same float weights."""
    rng = np.random.default_rng(2)
    m, k, gs = 32, 512, 64
    w = (0.03 * rng.standard_normal((m, k))).astype(np.float32)
    im = np.abs(rng.standard_normal(k)).astype(np.float32)
    x = (0.05 * rng.standard_normal(k)).astype(np.float32)

    p_all4 = pack_linear_mixed_q34(w, im, group_size=gs, q4_group_fraction=1.0)
    p_mix = pack_linear_mixed_q34(w, im, group_size=gs, q4_group_fraction=0.15)

    y4 = reference_mixed_gemv_numpy(p_all4, x)
    ym = reference_mixed_gemv_numpy(p_mix, x)
    max_diff = float(np.max(np.abs(y4 - ym)))
    print(f"[quality] max|logits_q4_uniform - logits_mixed| = {max_diff:.4f}", file=sys.stderr)
    assert max_diff < 0.5, (
        f"expected max abs logit delta < 0.5 vs uniform Q4 reference; got {max_diff}"
    )

    from asdsl.kernels.gemv_q3 import gemv_q3_mixed

    yn = gemv_q3_mixed(p_mix, x)
    assert_allclose(yn, ym, rtol=1e-4, atol=1e-3)


def test_execution_hygiene_pinning(native_q3) -> None:
    rng = np.random.default_rng(3)
    m, k, gs = 24, 256, 32
    w = (0.02 * rng.standard_normal((m, k))).astype(np.float32)
    im = np.abs(rng.standard_normal(k)).astype(np.float32)
    x = rng.standard_normal(k).astype(np.float32)
    packed = pack_linear_mixed_q34(
        w, im, group_size=gs, q4_group_fraction=0.2
    )

    native_q3.set_pin_openmp_pcores(True)
    from asdsl.kernels.gemv_q3 import gemv_q3_mixed

    _ = gemv_q3_mixed(packed, x)
    assert native_q3.has_openmp
    omp_threads = int(native_q3.get_num_threads())
    detected = int(native_q3.detected_pcore_count())
    print(
        f"[hygiene] omp_max_threads={omp_threads} detected_pcores={detected} "
        f"(pinning shared via omp_pcore_pinning.hpp)",
        file=sys.stderr,
    )
    if sys.platform == "win32":
        assert detected > 0
        # omp_threads may be set higher than detected_pcores when the benchmark
        # uses additional E-core threads (e.g. OMP_NUM_THREADS=8 on a 4P+8E system).
        assert omp_threads >= detected
