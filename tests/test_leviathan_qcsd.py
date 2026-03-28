"""Tests for Leviathan et al. (2023) QCSD break-even in ``run_full_benchmark``."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _bench_mod():
    root = Path(__file__).resolve().parent.parent
    path = root / "scripts" / "run_full_benchmark.py"
    spec = importlib.util.spec_from_file_location("_rfbench", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_leviathan_speedup_high_cost_low_alpha_below_one() -> None:
    m = _bench_mod()
    s = m._leviathan_speedup(0.10, 7, 0.90)
    assert s < 1.0
    assert 0.14 < s < 0.16


def test_qcsd_break_even_fails_high_cost_low_alpha() -> None:
    m = _bench_mod()
    ok, msg, s = m._qcsd_break_even_ok(0.10, 7, 5220.0, 5800.0)
    assert ok is False
    assert "QCSD break-even FAIL" in msg
    assert "enable gate" in msg
    assert s < 1.0


def test_min_alpha_binary_search_bracket() -> None:
    m = _bench_mod()
    # Cheap draft: should need modest alpha for 1.05x
    a = m._min_alpha_for_leviathan_speedup(7, 128.0 / 5800.0, 1.05)
    assert a is not None
    assert 0.0 <= a <= 1.0
    assert m._leviathan_speedup(a, 7, 128.0 / 5800.0) >= 1.045


def test_phi4_acceptance_estimate_optimistic_raises_leviathan_s() -> None:
    """Higher alpha => higher S; at fixed c, 0.95 can pass break-even while 0.10 fails."""
    m = _bench_mod()
    g = 7
    t_mb = 5800.0
    d_mb = 0.3 * t_mb
    assert m._leviathan_speedup(0.95, g, 0.3) > m._leviathan_speedup(0.70, g, 0.3)
    ok_opt, _, so = m._qcsd_break_even_ok(0.95, g, d_mb, t_mb)
    ok_pess, _, sp = m._qcsd_break_even_ok(0.10, g, d_mb, t_mb)
    assert ok_opt and so >= 1.01
    assert not ok_pess and sp < 1.01


def test_phi4_acceptance_estimate_optimistic_enables_marginal_qcsd() -> None:
    """Higher --phi4-acceptance-estimate raises S; can flip a marginal config from FAIL to OK."""
    m = _bench_mod()
    g = 7
    t_mb = 5800.0
    d_mb = 0.35 * t_mb
    ok_lo, _, s_lo = m._qcsd_break_even_ok(0.62, g, d_mb, t_mb)
    ok_hi, _, s_hi = m._qcsd_break_even_ok(0.95, g, d_mb, t_mb)
    assert s_hi > s_lo
    assert not ok_lo and ok_hi


def test_greedy_verify_uses_reduced_batch_factors() -> None:
    from asdsl.speculative import dual_model as dm

    assert dm.GREEDY_VERIFY_BATCH_FACTOR_BASE == 0.34
    assert dm.GREEDY_VERIFY_BATCH_FACTOR_SLOPE == 0.055
    assert dm.GREEDY_VERIFY_BATCH_FACTOR_BASE < 0.55
    assert dm.GREEDY_VERIFY_BATCH_FACTOR_SLOPE < 0.08
