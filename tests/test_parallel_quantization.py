"""Phase 2 parallel (batched) clipping grid search tests."""

from __future__ import annotations

import time

import numpy as np
import pytest

from asdsl.quantization import core as qcore
from asdsl.quantization.core import (
    CLIP_RATIOS,
    _find_optimal_scales,
    _find_optimal_scales_sequential,
    compute_scale_zero_batched,
    quantize_weights,
)


class TestParallelQuantization:
    def test_batched_scales_match_sequential(self):
        rng = np.random.default_rng(0)
        grouped = rng.standard_normal((128, 32)).astype(np.float32)
        bits = 4

        seq_s, seq_z = _find_optimal_scales_sequential(grouped, bits, symmetric=False)
        par_s, par_z = _find_optimal_scales(
            grouped, bits, symmetric=False, use_parallel=True
        )

        np.testing.assert_allclose(
            par_s.astype(np.float32),
            seq_s.astype(np.float32),
            rtol=0,
            atol=1e-6,
        )
        assert seq_z is not None and par_z is not None
        np.testing.assert_array_equal(par_z, seq_z)

        bat_s, bat_z = compute_scale_zero_batched(
            grouped, bits, CLIP_RATIOS, symmetric=False
        )
        assert bat_s.shape == (len(CLIP_RATIOS), 128, 1)
        assert bat_z is not None

    def test_batched_does_not_change_quantized_weights(self, monkeypatch):
        rng = np.random.default_rng(1)
        w = rng.standard_normal((64, 128)).astype(np.float32) * 0.05

        qt_par = quantize_weights(
            w, bits=4, group_size=32, symmetric=False, optimize_clips=True
        )

        def _sequential_only(g, b, s, **kw):
            return _find_optimal_scales(g, b, s, use_parallel=False)

        monkeypatch.setattr(qcore, "_find_optimal_scales", _sequential_only)
        qt_seq = quantize_weights(
            w, bits=4, group_size=32, symmetric=False, optimize_clips=True
        )

        np.testing.assert_array_equal(qt_par.data, qt_seq.data)
        np.testing.assert_array_equal(qt_par.scales, qt_seq.scales)
        if qt_par.zeros is not None:
            np.testing.assert_array_equal(qt_par.zeros, qt_seq.zeros)

    def test_parallel_quantization_speedup(self):
        rng = np.random.default_rng(2)
        grouped = rng.standard_normal((4096 * 128, 32)).astype(np.float32)
        bits = 4

        def _time(fn, repeats: int = 3) -> float:
            for _ in range(1):
                fn()
            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                fn()
                times.append(time.perf_counter() - t0)
            return float(np.mean(times))

        t_seq = _time(
            lambda: _find_optimal_scales_sequential(grouped, bits, symmetric=False)
        )
        t_par = _time(
            lambda: _find_optimal_scales(
                grouped, bits, symmetric=False, use_parallel=True
            )
        )
        assert t_par < t_seq
