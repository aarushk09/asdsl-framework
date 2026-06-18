"""generate_stream must match greedy_generate (unified prefill commits all prompt tokens)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

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


@pytest.mark.slow
def test_generate_stream_matches_greedy_on_long_prompt() -> None:
    pytest.importorskip("asdsl.kernels._native_unified")
    if not CACHE_DIR.is_dir() or not any(CACHE_DIR.glob("*.safetensors")):
        pytest.skip("phi4 weight cache missing")

    from experiments.phi4_cpu_run import (
        WeightStore,
        _normalize_input_ids,
        generate_stream,
        load_tokenizer,
        set_thread_count,
    )
    from asdsl.inference.unified_bridge import greedy_generate, reset_unified_session

    _c0_env()
    set_thread_count(12)
    store = WeightStore(bits=4, group_size=32, enable_qcsd=False)
    store.load()
    store.warm_cache()
    tok = load_tokenizer()

    prompt = "Explain gravity in simple terms."
    input_ids = _normalize_input_ids(
        tok.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    assert len(input_ids) >= 8, "need multi-token prompt to exercise prefill"

    n_new = 32
    reset_unified_session(store)
    greedy = greedy_generate(store, input_ids, n_new)

    reset_unified_session(store)
    stream_ids: list[int] = []
    for st in generate_stream(
        prompt=prompt,
        store=store,
        tokenizer=tok,
        max_new_tokens=n_new,
        system_prompt="You are a helpful assistant.",
    ):
        stream_ids.append(int(st.token_id))

    assert stream_ids == greedy[len(input_ids) :], (
        f"stream/greedy mismatch on {len(input_ids)}-token prompt\n"
        f"greedy={greedy[len(input_ids):]}\nstream={stream_ids}"
    )
