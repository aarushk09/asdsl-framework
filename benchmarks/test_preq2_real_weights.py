"""preq2 vs legacy preq on real cached gate_up weights (Phase 3/4 regression)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.mark.slow
def test_real_gate_up_preq2_drift_bounded() -> None:
    """Production gate_up slice: preq2 must track legacy preq within documented tolerance."""
    os.environ.setdefault("ASDSL_PREQ2", "1")
    ng = pytest.importorskip("asdsl.kernels._native_gemv")
    from asdsl.quantization.repack_preq2 import meta_to_flat, quant_to_flat, repack_preq_blocks_to_preq2
    from experiments.phi4_cpu_run import WeightStore

    store = WeightStore(bits=4)
    store.load()
    store.build_preq_blocks()

    rows, cols, gs = 512, 3072, 32
    preq_full = store._preq_blocks_np[(0, "gate_up_proj")]
    n_groups = cols // gs
    row_bytes = n_groups * 20
    preq = preq_full[: rows * row_bytes].copy()

    meta, quant = repack_preq_blocks_to_preq2(preq, rows, cols, gs)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(cols).astype(np.float32)
    y_legacy = np.zeros(rows, dtype=np.float32)
    y_p2 = np.zeros(rows, dtype=np.float32)
    ng.gemv_q4_32_preq_fused_avx2(preq, x, y_legacy, rows, cols, gs)
    ng.gemv_preq2_fused_avx2(meta_to_flat(meta), quant_to_flat(quant), x, y_p2, rows, cols, gs)
    diff = np.abs(y_legacy - y_p2)
    maxdiff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    p99 = float(np.percentile(diff, 99))
    print(f"real gate_up maxdiff={maxdiff:.4f} mean={mean_diff:.4f} p99={p99:.4f}")
    # Tightened from 20.0: production blocks show ~7–11 max on layer-0 slice (PHASE_WALKTHROUGH P1).
    assert maxdiff < 12.0, f"preq2 drift regressed: maxdiff={maxdiff}"
    assert mean_diff < 2.0, f"mean drift too high: {mean_diff}"
