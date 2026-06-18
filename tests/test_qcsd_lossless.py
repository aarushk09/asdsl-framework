"""QCSD Python path must match greedy autoregressive token IDs."""

from __future__ import annotations

import os

import pytest


@pytest.mark.parametrize("prompt", ["The", "In 2024,", "def fibonacci(n):"])
def test_qcsd_matches_ar(phi4_store_qcsd, prompt: str, fast_decode_tokens: int) -> None:
    """Three prompts; skip if no weights (via conftest)."""
    from transformers import AutoTokenizer

    from experiments.phi4_cpu_run import generate_qcsd, generate_stream

    os.environ["ASDSL_IGNORE_EOS"] = "1"
    tok = AutoTokenizer.from_pretrained(
        "microsoft/Phi-4-multimodal-instruct", trust_remote_code=True
    )

    ar_ids: list[int] = []
    for st in generate_stream(
        prompt=prompt,
        store=phi4_store_qcsd,
        tokenizer=tok,
        max_new_tokens=fast_decode_tokens,
        system_prompt="",
    ):
        ar_ids.append(int(st.token_id))

    qcsd_ids: list[int] = []
    generate_qcsd(
        prompt=prompt,
        store=phi4_store_qcsd,
        tokenizer=tok,
        max_new_tokens=fast_decode_tokens,
        system_prompt="",
        draft_k=3,
        generated_ids_out=qcsd_ids,
    )
    assert qcsd_ids == ar_ids
