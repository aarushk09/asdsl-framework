"""WikiText-64 perplexity for Phi-4 CPU path."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))

import phi4_cpu_run as p4  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--gguf-path", type=str, default=None)
    args = parser.parse_args()

    gs_env = os.environ.get("ASDSL_GROUP_SIZE", "").strip()
    group_size = int(gs_env) if gs_env else None

    os.environ.setdefault("ASDSL_USE_UNIFIED", "1")
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/Phi-4-multimodal-instruct", trust_remote_code=True
    )
    try:
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])
        tokens = tokenizer.encode(text)[: args.max_tokens + 1]
    except Exception:
        tokens = tokenizer.encode("The quick brown fox " * 20)[: args.max_tokens + 1]

    store = p4.WeightStore(bits=args.bits, group_size=group_size)
    store.load()
    store.warm_cache()
    if args.gguf_path:
        store.load_from_gguf(args.gguf_path)

    if os.environ.get("ASDSL_USE_UNIFIED", "0").strip() in ("1", "true", "yes"):
        from asdsl.inference.unified_bridge import build_unified_engine

        engine = build_unified_engine(store)
        engine.reset_kv()
        nll = 0.0
        for i in range(1, len(tokens)):
            logits = engine.forward_token(tokens[i - 1])
            log_probs = logits - np.logaddexp.reduce(logits)
            nll -= float(log_probs[tokens[i]])
        ppl = math.exp(nll / max(len(tokens) - 1, 1))
    else:
        raise RuntimeError("perplexity eval requires ASDSL_USE_UNIFIED=1")

    print(f"Perplexity: {ppl:.2f}  (tokens={len(tokens)-1}, gs={store.group_size})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
