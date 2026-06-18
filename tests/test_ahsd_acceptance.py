"""AHSD acceptance rate must exceed break-even for configured draft_k."""

from __future__ import annotations

import os

import pytest


def _min_alpha_for_k(draft_k: int) -> float:
    return {2: 0.78, 3: 0.72, 4: 0.68, 5: 0.65}.get(draft_k, 0.65)


def test_ahsd_acceptance_logged(phi4_store, ahsd_native_ok, fast_decode_tokens: int) -> None:
    from asdsl.inference.unified_bridge import ahsd_generate
    from transformers import AutoTokenizer

    os.environ["ASDSL_IGNORE_EOS"] = "1"
    draft_k = int(os.environ.get("ASDSL_AHSD_DRAFT_K", "3"))
    tok = AutoTokenizer.from_pretrained(
        "microsoft/Phi-4-multimodal-instruct", trust_remote_code=True
    )
    prompt_ids = tok.encode("The", add_special_tokens=True)
    result = ahsd_generate(
        phi4_store,
        prompt_ids,
        max_new_tokens=max(fast_decode_tokens, 8),
        draft_k=draft_k,
    )
    assert result.draft_tokens >= 0
    assert 0.0 <= result.acceptance_rate <= 1.0
    assert result.speculative_cycles >= 1


@pytest.mark.skipif(
    os.environ.get("ASDSL_ENFORCE_ACCEPTANCE_GATE", "0") != "1",
    reason="Set ASDSL_ENFORCE_ACCEPTANCE_GATE=1 to enforce alpha gate",
)
def test_ahsd_acceptance_gate(phi4_store, ahsd_native_ok) -> None:
    from asdsl.inference.unified_bridge import ahsd_generate
    from transformers import AutoTokenizer

    draft_k = int(os.environ.get("ASDSL_AHSD_DRAFT_K", "3"))
    tok = AutoTokenizer.from_pretrained(
        "microsoft/Phi-4-multimodal-instruct", trust_remote_code=True
    )
    prompt_ids = tok.encode("The", add_special_tokens=True)
    result = ahsd_generate(phi4_store, prompt_ids, max_new_tokens=32, draft_k=draft_k)
    assert result.acceptance_rate >= _min_alpha_for_k(draft_k)
