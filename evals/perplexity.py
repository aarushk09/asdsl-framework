"""Phase-8 native perplexity-style evaluation for ASDSL.

This script routes evaluation through `asdsl.engine` high-level APIs and
native C++ kernels (`prefill_prompt_tokens` + native token generation),
avoiding legacy Python layer loops and chunked f16 matvec paths.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

try:
    import torch

    if hasattr(torch, "set_flush_denormal"):
        try:
            torch.set_flush_denormal(True)
        except Exception:
            pass
    _t = os.environ.get("ASDSL_TORCH_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS")
    _n = max(1, int(_t)) if _t is not None else min(8, max(1, os.cpu_count() or 4))
    torch.set_num_threads(_n)
    os.environ.setdefault("NUMEXPR_MAX_THREADS", str(_n))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(_n))
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from asdsl.engine import evaluate_perplexity_phase8_native, resolve_hf_ppl_model_id
from transformers import AutoTokenizer


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

    if max_tokens is not None:
        tokens = tokenizer.encode(
            text,
            max_length=max_tokens,
            truncation=True,
            add_special_tokens=False,
        )
    else:
        tokens = tokenizer.encode(text)
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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ASDSL Perplexity Evaluation")
    parser.add_argument("--bits", type=int, default=8,
                        choices=[2, 3, 4, 8, 16],
                        help="Quantization bit-width (16 = float16 baseline)")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max dataset tokens to evaluate (controls runtime)")
    parser.add_argument("--stride", type=int, default=512,
                        help="Sliding window stride for PPL computation")
    parser.add_argument(
        "--hf-model-id",
        type=str,
        default=None,
        help=(
            "HuggingFace model id for perplexity and dataset tokenization "
            "(default: $ASDSL_PPL_MODEL_ID or microsoft/phi-4)."
        ),
    )
    args = parser.parse_args()

    ppl_model_id = resolve_hf_ppl_model_id(args.hf_model_id)

    print("=" * 66)
    print(f"  ASDSL Perplexity Evaluation - bits={args.bits}")
    print("=" * 66)
    print(f"  HF model / tokenizer: {ppl_model_id}")
    print("Routing: HuggingFace causal LM (summed NLL); mmap paths for roofline metadata only")

    # Load tokenizer (must match hf_model_id used in evaluate_perplexity_phase8_native)
    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(ppl_model_id, trust_remote_code=True)

    # Load WikiText
    print("Loading WikiText-2 ...")
    tokens = load_wikitext2(tokenizer, max_tokens=args.max_tokens)

    # Evaluate
    print("\nComputing perplexity (Phase-8 native route) ...")
    results = evaluate_perplexity_phase8_native(
        tokens=tokens,
        bits=args.bits,
        stride=args.stride,
        hf_model_id=ppl_model_id,
    )

    # Report
    quant_label = "float16-compatible route" if args.bits == 16 else f"ASDSL {args.bits}-bit"
    print("\n" + "=" * 66)
    print(f"  Results: {quant_label}")
    print("=" * 66)
    print(f"  Perplexity (proxy):  {results.ppl:.2f}")
    print(f"  Avg NLL / token:     {results.avg_nll:.4f}")
    print(f"  Windows scored:      {results.windows}")
    print(f"  Targets scored:      {results.num_tokens}")
    print(f"  Eval throughput:     {results.tokens_per_second:.2f} tok/s")
    print(f"  Total time:          {results.elapsed_sec:.1f}s")
    print(f"  Backend weights:     {results.backend_model_bin}")
    print(f"  Backend metadata:    {results.backend_model_metadata}")
    print("=" * 66)


if __name__ == "__main__":
    main()
