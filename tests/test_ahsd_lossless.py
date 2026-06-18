"""AHSD UnifiedEngine output must match greedy AR (lossless)."""

from __future__ import annotations

import os

import pytest

PARITY_PROMPTS = ["The", "In 2024,", "def fibonacci(n):"]


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


@pytest.mark.parametrize("prompt", PARITY_PROMPTS)
def test_ahsd_lossless_single_prompt(
    phi4_store, ahsd_native_ok, prompt: str, fast_decode_tokens: int
) -> None:
    from asdsl.inference.unified_bridge import ahsd_generate
    from transformers import AutoTokenizer

    os.environ["ASDSL_IGNORE_EOS"] = "1"
    tok = AutoTokenizer.from_pretrained(
        "microsoft/Phi-4-multimodal-instruct", trust_remote_code=True
    )
    prompt_ids = tok.encode(prompt, add_special_tokens=True)
    ar = _greedy_ar(phi4_store, prompt_ids, fast_decode_tokens)

    result = ahsd_generate(
        phi4_store,
        prompt_ids,
        max_new_tokens=fast_decode_tokens,
        draft_k=int(os.environ.get("ASDSL_AHSD_DRAFT_K", "3")),
        use_sdqs=False,
    )
    gen = result.token_ids[len(prompt_ids) : len(prompt_ids) + fast_decode_tokens]
    assert gen == ar


@pytest.mark.parametrize("prompt", PARITY_PROMPTS[:2])
def test_ahsd_extended_prompts(phi4_store, ahsd_native_ok, prompt: str) -> None:
    """Five prompts × 64 tokens in full runs; 2 prompts here for CI speed."""
    from asdsl.inference.unified_bridge import ahsd_generate
    from transformers import AutoTokenizer

    n_tok = int(os.environ.get("ASDSL_TEST_DECODE_TOKENS", "2"))
    if os.environ.get("ASDSL_TEST_FULL_AHSD", "0") == "1":
        n_tok = 64

    os.environ["ASDSL_IGNORE_EOS"] = "1"
    tok = AutoTokenizer.from_pretrained(
        "microsoft/Phi-4-multimodal-instruct", trust_remote_code=True
    )
    prompt_ids = tok.encode(prompt, add_special_tokens=True)
    ar = _greedy_ar(phi4_store, prompt_ids, n_tok)
    result = ahsd_generate(phi4_store, prompt_ids, max_new_tokens=n_tok, draft_k=3)
    gen = result.token_ids[len(prompt_ids) : len(prompt_ids) + n_tok]
    assert gen == ar
