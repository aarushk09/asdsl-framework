"""C++ SWIFT generate must match greedy AR token sequence."""

from __future__ import annotations

import os

import pytest

from experiments.phi4_cpu_run import _normalize_input_ids

def _greedy_ar_tokens(store, prompt_ids: list[int], max_new: int) -> list[int]:
    from asdsl.inference.unified_bridge import get_or_build_unified_engine

    eng = get_or_build_unified_engine(store)
    eng.reset_session()
    out: list[int] = []
    logits = None
    for pos, tid in enumerate(prompt_ids):
        logits = eng.forward_token(int(tid), int(pos))
    pos = len(prompt_ids)
    for _ in range(max_new):
        nxt = int(logits.argmax())
        out.append(nxt)
        logits = eng.forward_token(int(nxt), int(pos))
        pos += 1
    return out


def test_swift_matches_greedy_ar(
    phi4_store, ahsd_native_ok, fast_decode_tokens: int
) -> None:
    from asdsl.inference.unified_bridge import get_or_build_unified_engine
    from transformers import AutoTokenizer

    os.environ["ASDSL_IGNORE_EOS"] = "1"
    tok = AutoTokenizer.from_pretrained(
        "microsoft/Phi-4-multimodal-instruct", trust_remote_code=True
    )
    prompt_ids = _normalize_input_ids(
        tok.apply_chat_template(
            [{"role": "user", "content": "The"}],
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    ar = _greedy_ar_tokens(phi4_store, prompt_ids, fast_decode_tokens)

    eng = get_or_build_unified_engine(phi4_store)
    eng.reset_session()
    swift = eng.generate_swift(
        [int(t) for t in prompt_ids], int(fast_decode_tokens), 3
    )
    swift_gen = swift[len(prompt_ids) : len(prompt_ids) + fast_decode_tokens]
    assert swift_gen == ar
