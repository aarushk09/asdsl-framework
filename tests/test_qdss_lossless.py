"""QDSS: Q2 draft + Q4 verify must match Q4 greedy AR."""

from __future__ import annotations

import os

import pytest


def _greedy_ar(store, prompt_ids: list[int], max_new: int) -> list[int]:
    from asdsl.inference.unified_bridge import get_or_build_unified_engine

    eng = get_or_build_unified_engine(store)
    eng.reset_session()
    logits = None
    for pos, tid in enumerate(prompt_ids):
        logits = eng.forward_token(int(tid), int(pos))
    pos = len(prompt_ids)
    out: list[int] = []
    for _ in range(max_new):
        nxt = int(logits.argmax())
        out.append(nxt)
        logits = eng.forward_token(int(nxt), int(pos))
        pos += 1
    return out


def test_qdss_matches_ar(phi4_store_qcsd, ahsd_native_ok, fast_decode_tokens: int) -> None:
    from asdsl.inference.unified_bridge import ahsd_generate
    from transformers import AutoTokenizer

    if not getattr(phi4_store_qcsd, "_enable_qcsd", False):
        pytest.skip("QCSD draft bank not enabled")

    os.environ["ASDSL_IGNORE_EOS"] = "1"
    tok = AutoTokenizer.from_pretrained(
        "microsoft/Phi-4-multimodal-instruct", trust_remote_code=True
    )
    prompt_ids = tok.encode("The", add_special_tokens=True)
    ar = _greedy_ar(phi4_store_qcsd, prompt_ids, fast_decode_tokens)

    # AHSD with Q2 draft weights but not SDQS dual-stream scheduling.
    result = ahsd_generate(
        phi4_store_qcsd,
        prompt_ids,
        max_new_tokens=fast_decode_tokens,
        draft_k=3,
        use_sdqs=False,
    )
    gen = result.token_ids[len(prompt_ids) : len(prompt_ids) + fast_decode_tokens]
    assert gen == ar
