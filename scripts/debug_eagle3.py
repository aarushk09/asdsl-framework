#!/usr/bin/env python3
"""One-step diagnostic: MTP input vs greedy next token (Profile G alignment)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np
import torch
from transformers import AutoTokenizer


def main() -> None:
    from experiments.phi4_cpu_run import (
        WeightStore,
        KVHistory,
        build_rope_cache,
        forward_layer,
        rms_norm,
        NUM_LAYERS,
        ROTARY_DIM,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/Phi-4-multimodal-instruct", trust_remote_code=True
    )
    store = WeightStore(bits=4, enable_qcsd=False)
    store.load()
    store.warm_cache()
    fr = ROOT / "phi4_fatrelu_thresholds.json"
    if fr.exists():
        store.load_fatrelu(str(fr))
    mtp_p = ROOT / "models" / "mtp_head.pt"
    if not mtp_p.exists():
        print("ERROR: models/mtp_head.pt missing — run scripts/train_mtp_head.py")
        sys.exit(1)
    store.load_mtp_head(str(mtp_p))
    store._use_native_gemv = True
    store._use_lut_gemv = True
    store._enable_sparse = True
    store._sparsity_threshold = 0.0

    messages = [{"role": "user", "content": "The capital of France is"}]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    max_seq = len(input_ids) + 8
    rope_cos, rope_sin = build_rope_cache(max_seq, ROTARY_DIM)
    kv = KVHistory(max_seq=max_seq)

    with torch.inference_mode():
        hidden = None
        for pos, tid in enumerate(input_ids):
            hidden = store.embed_f16[tid].float().unsqueeze(0)
            for li in range(NUM_LAYERS):
                hidden = forward_layer(hidden, li, store, kv, rope_cos, rope_sin, pos)
        prev = rms_norm(hidden, store.final_norm)
        logits = store.lm_head_matvec(prev)
        actual_next = int(logits.argmax())

    # Match generate_eagle3: one warm step with greedy token, then MTP predicts the following token.
    pos = len(input_ids)
    hidden = store.embed_f16[actual_next].float().unsqueeze(0)
    for li in range(NUM_LAYERS):
        hidden = forward_layer(hidden, li, store, kv, rope_cos, rope_sin, pos)
    prev2 = rms_norm(hidden, store.final_norm)
    h = prev2.detach().cpu().float().numpy().ravel()
    print(
        f"_last_final_hidden (after warm step tok={actual_next}): shape={h.shape}, "
        f"mean={h.mean():.4f}, std={h.std():.4f}, norm={float(np.linalg.norm(h)):.2f}"
    )

    store._last_final_hidden = h.copy()
    mtp_ctx = int(actual_next)
    tok_emb = store._get_token_embedding(mtp_ctx)
    print(
        f"token_embed(MTP ctx id={mtp_ctx}): norm={float(np.linalg.norm(tok_emb)):.2f}"
    )

    from experiments.phi4_cpu_run import _run_mtp_draft

    logits2 = store.lm_head_matvec(prev2)
    actual_after = int(logits2.argmax())

    store._use_eagle3 = True
    drafts = _run_mtp_draft(store, mtp_ctx, k=1)
    predicted = drafts[0] if drafts else -1
    print(f"MTP 1-step predicted token: {predicted}")
    print(f"Greedy AR token after warm:   {actual_after}")
    print(f"Match: {predicted == actual_after}")


if __name__ == "__main__":
    main()
