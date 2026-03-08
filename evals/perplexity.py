"""
Perplexity evaluation for ASDSL-quantized Phi-4 on WikiText-2.

Computes token-level cross-entropy loss (perplexity) to measure how
quantization affects the model's core predictive distribution.

Usage:
  python evals/perplexity.py                      # default: bits=16 (baseline)
  python evals/perplexity.py --bits 8             # ASDSL 8-bit
  python evals/perplexity.py --bits 4             # ASDSL 4-bit
  python evals/perplexity.py --bits 3             # ASDSL 3-bit (10-in-32)
  python evals/perplexity.py --max-tokens 512     # evaluate fewer tokens (faster)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np
import torch

# --- import the inference engine (same as phi4_cpu_run.py) ---
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from phi4_cpu_run import (
    WeightStore, KVHistory, ASDSLKVTracker,
    forward_layer, rms_norm, build_rope_cache,
    NUM_LAYERS, ROTARY_DIM, EOS_TOKEN_IDS,
)
from transformers import AutoTokenizer

MODEL_DIR = ROOT / "models" / "phi4-multimodal-instruct"


# ---------------------------------------------------------------------------
# WikiText-2 loading
# ---------------------------------------------------------------------------

def load_wikitext2(tokenizer, max_tokens: int | None = None) -> list[int]:
    """Load WikiText-2 test split and tokenize it.

    Falls back to downloading from HuggingFace datasets if available,
    otherwise fetches the raw text from the standard URL.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])
    except Exception:
        # Minimal fallback: use a local file or a small built-in sample
        wt_path = ROOT / "data" / "wikitext2_test.txt"
        if wt_path.exists():
            text = wt_path.read_text(encoding="utf-8")
        else:
            print("WARNING: wikitext-2 not available. Using built-in sample.")
            text = _BUILTIN_SAMPLE

    tokens = tokenizer.encode(text)
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    print(f"  Dataset tokens: {len(tokens)}")
    return tokens


# Small fallback corpus (~200 tokens) for environments without internet/datasets
_BUILTIN_SAMPLE = (
    "The tower is 324 metres tall, about the same height as an 81-storey building, "
    "and the tallest structure in Paris. Its base is square, measuring 125 metres on "
    "each side. During its construction, the Eiffel Tower surpassed the Washington "
    "Monument to become the tallest man-made structure in the world, a title it held "
    "for 41 years until the Chrysler Building in New York City was finished in 1930. "
    "It was the first structure to reach a height of 300 metres. Due to the addition "
    "of a broadcasting aerial at the top of the tower in 1957, it is now taller than "
    "the Chrysler Building by 5.2 metres. Excluding transmitters, the Eiffel Tower "
    "is the second tallest free-standing structure in France after the Millau Viaduct.\n\n"
    "Robert Oppenheimer was an American theoretical physicist and professor of physics "
    "at the University of California, Berkeley. He was the wartime head of the Los "
    "Alamos Laboratory and is among those who are credited with being the father of "
    "the atomic bomb for their role in the Manhattan Project."
)


# ---------------------------------------------------------------------------
# Perplexity computation
# ---------------------------------------------------------------------------

def compute_perplexity(
    tokens: list[int],
    store: WeightStore,
    stride: int = 512,
) -> dict[str, float]:
    """Compute perplexity on a token sequence using a sliding window.

    To avoid reprocessing the entire sequence for each window, we use
    a stride-based approach: process `stride` tokens at a time, resetting
    the KV cache for each window.  This matches the standard approach used
    by HuggingFace/EleutherAI for causal LM perplexity evaluation.

    Returns dict with {ppl, nll_sum, num_tokens, tokens_per_sec}.
    """
    max_seq = stride + 64
    rope_cos, rope_sin = build_rope_cache(max_seq, ROTARY_DIM)

    nll_sum = 0.0
    n_scored = 0
    t_start = time.perf_counter()

    num_windows = max(1, (len(tokens) - 1) // stride)

    for win_idx in range(num_windows):
        begin = win_idx * stride
        end = min(begin + stride + 1, len(tokens))
        window = tokens[begin:end]
        if len(window) < 2:
            break

        # Fresh KV cache per window
        kv_hist = KVHistory()

        # Run forward pass for each token in the window
        logits = None
        for i, tid in enumerate(window[:-1]):
            hidden = store.embed_f16[tid].float().unsqueeze(0)
            for layer in range(NUM_LAYERS):
                hidden = forward_layer(
                    hidden, layer, store, kv_hist, rope_cos, rope_sin, pos=i,
                )
            # LM head (final norm + projection)
            hidden = rms_norm(hidden, store.final_norm)
            logits = store.lm_head_matvec(hidden)

            # Compute cross-entropy loss for the NEXT token
            target = window[i + 1]
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            nll = -log_probs[target].item()
            nll_sum += nll
            n_scored += 1

        elapsed = time.perf_counter() - t_start
        avg_nll = nll_sum / max(n_scored, 1)
        ppl = math.exp(min(avg_nll, 50))  # cap to avoid overflow
        tps = n_scored / elapsed if elapsed > 0 else 0

        print(f"  Window {win_idx+1}/{num_windows}: "
              f"tokens={n_scored}, NLL={avg_nll:.4f}, PPL={ppl:.2f}, "
              f"{tps:.2f} eval-tok/s",
              flush=True)

    elapsed = time.perf_counter() - t_start
    avg_nll = nll_sum / max(n_scored, 1)
    ppl = math.exp(min(avg_nll, 50))
    tps = n_scored / elapsed if elapsed > 0 else 0

    return {
        "ppl": ppl,
        "nll_sum": nll_sum,
        "avg_nll": avg_nll,
        "num_tokens": n_scored,
        "tokens_per_sec": tps,
        "elapsed_sec": elapsed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ASDSL Perplexity Evaluation")
    parser.add_argument("--bits", type=int, default=16,
                        choices=[2, 3, 4, 8, 16],
                        help="Quantization bit-width (16 = float16 baseline)")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max dataset tokens to evaluate (controls runtime)")
    parser.add_argument("--stride", type=int, default=512,
                        help="Sliding window stride for PPL computation")
    args = parser.parse_args()

    print("=" * 66)
    print(f"  ASDSL Perplexity Evaluation - bits={args.bits}")
    print("=" * 66)

    # Load tokenizer
    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/Phi-4-multimodal-instruct", trust_remote_code=True
    )

    # Load WikiText
    print("Loading WikiText-2 ...")
    tokens = load_wikitext2(tokenizer, max_tokens=args.max_tokens)

    # Load model
    print(f"Loading model (bits={args.bits}) ...")
    store = WeightStore(bits=args.bits)
    store.load()
    store.warm_cache()

    # Evaluate
    print("\nComputing perplexity ...")
    results = compute_perplexity(tokens, store, stride=args.stride)

    # Report
    quant_label = "float16 (baseline)" if args.bits == 16 else f"ASDSL {args.bits}-bit"
    print("\n" + "=" * 66)
    print(f"  Results: {quant_label}")
    print("=" * 66)
    print(f"  Perplexity (PPL):    {results['ppl']:.2f}")
    print(f"  Avg NLL / token:     {results['avg_nll']:.4f}")
    print(f"  Tokens evaluated:    {results['num_tokens']}")
    print(f"  Eval throughput:     {results['tokens_per_sec']:.2f} tok/s")
    print(f"  Total time:          {results['elapsed_sec']:.1f}s")
    print("=" * 66)


if __name__ == "__main__":
    main()
