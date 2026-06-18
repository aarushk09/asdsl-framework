"""PLD losslessness: greedy vs PLD trajectories must match under C0 preq2."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CACHE_DIR = ROOT / "models" / "phi4_weight_cache"


def _c0_env() -> None:
    os.environ["ASDSL_USE_UNIFIED"] = "1"
    os.environ["ASDSL_PREQ2"] = "1"
    os.environ["ASDSL_FUSED_GEMV"] = "1"
    os.environ["ASDSL_CHUNKED_GEMV"] = "1"
    os.environ["ASDSL_AFFINITY"] = "physical"
    os.environ["ASDSL_GROUP_SIZE"] = "32"
    os.environ.setdefault("OMP_NUM_THREADS", "12")


def _load_store():
    pytest.importorskip("asdsl.kernels._native_unified")
    if not CACHE_DIR.is_dir() or not any(CACHE_DIR.glob("*.safetensors")):
        pytest.skip("phi4 weight cache missing")
    from experiments.phi4_cpu_run import WeightStore, set_thread_count

    set_thread_count(12)
    store = WeightStore(bits=4, group_size=32, enable_qcsd=False)
    store.load()
    store.warm_cache()
    return store


@pytest.mark.slow
def test_pld_lookup_finds_repeat() -> None:
    from asdsl.speculative.pld import PLDConfig, PromptLookupDecoder

    dec = PromptLookupDecoder(PLDConfig(max_draft_k=3))
    ctx = [1, 2, 3, 10, 20, 30, 1, 2, 3]
    draft = dec.lookup(ctx)
    assert draft == [10, 20, 30]


@pytest.mark.slow
def test_verify_serial_matches_forward_token() -> None:
    """Oracle verify rows must match sequential forward_token logits."""
    _c0_env()
    store = _load_store()
    from asdsl.inference.unified_bridge import get_or_build_unified_engine

    eng = get_or_build_unified_engine(store)
    eng.reset_session()
    pre = [100, 200, 300]
    for i, tid in enumerate(pre):
        eng.forward_token(int(tid), i)
    start = len(pre)
    toks = [400, 500, 600]
    from experiments.phi4_cpu_run import VOCAB

    serial = np.asarray(
        eng.forward_verify_serial_all_logits(toks, start), dtype=np.float32
    ).reshape(len(toks), VOCAB)
    eng.reset_session()
    for i, tid in enumerate(pre):
        eng.forward_token(int(tid), i)
    for i, tid in enumerate(toks):
        row = np.asarray(eng.forward_token(int(tid), start + i), dtype=np.float32)
        np.testing.assert_allclose(serial[i], row, rtol=0, atol=0)


@pytest.mark.slow
@pytest.mark.parametrize(
    "prompt_ids",
    [
        [100, 200, 300],
        [1, 2, 3, 1, 2, 3, 4, 5],
    ],
)
def test_pld_matches_greedy(prompt_ids: list[int]) -> None:
    _c0_env()
    store = _load_store()
    from asdsl.inference.unified_bridge import greedy_generate, pld_generate

    n = 24
    greedy = greedy_generate(store, prompt_ids, n)
    pld = pld_generate(store, prompt_ids, n)
    assert greedy == pld, f"greedy={greedy}\npld={pld}"
