"""Sliding-window perplexity for Phi-4 Q4 unified path (Phase 4 PPL gate)."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _nll_from_logits(logits: np.ndarray, target_id: int) -> float:
    x = logits.astype(np.float64)
    x = x - np.max(x)
    log_z = np.log(np.sum(np.exp(x)))
    return float(-(x[target_id] - log_z))


def _prompt_token_ids(
    tokenizer,
    text: str,
    *,
    use_chat_template: bool,
    system_prompt: str = "",
) -> list[int]:
    from experiments.phi4_cpu_run import _normalize_input_ids

    if use_chat_template:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        return _normalize_input_ids(ids)
    return tokenizer.encode(text, add_special_tokens=False)


def compute_perplexity(
    store,
    tokenizer,
    text: str,
    *,
    max_tokens: int = 64,
    stride: int | None = None,
    use_chat_template: bool = True,
    system_prompt: str = "",
) -> dict:
    from asdsl.inference.unified_bridge import get_or_build_unified_engine, reset_unified_session, unified_forward_token

    ids = _prompt_token_ids(
        tokenizer,
        text,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
    )
    if len(ids) < 2:
        raise ValueError("text too short for perplexity")
    ids = ids[:max_tokens]
    stride = stride or max(len(ids) // 2, 1)

    # Ensure engine is built after preq2/C01 blocks exist on the store.
    get_or_build_unified_engine(store)
    reset_unified_session(store)
    nll_sum = 0.0
    n_tok = 0
    top1_hits = 0
    for pos in range(len(ids) - 1):
        tid = int(ids[pos])
        target = int(ids[pos + 1])
        logits = unified_forward_token(store, tid, pos)
        if target < 0 or target >= logits.size:
            raise ValueError(f"target token {target} out of vocab range")
        nll_sum += _nll_from_logits(logits, target)
        if int(np.argmax(logits)) == target:
            top1_hits += 1
        n_tok += 1

    mean_nll = nll_sum / max(n_tok, 1)
    ppl = math.exp(mean_nll)
    return {
        "n_tokens": n_tok,
        "mean_nll": mean_nll,
        "perplexity": ppl,
        "top1_acc": top1_hits / max(n_tok, 1),
        "prompt_len": len(ids),
        "stride": stride,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phi-4 Q4 perplexity (unified engine)")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Calibration text (default: built-in paragraph)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="C0",
        choices=("C0", "C0.1", "C0.3"),
        help="Apply parity manifest env knobs (default C0)",
    )
    parser.add_argument(
        "--chat-template",
        action="store_true",
        help="Wrap text in Phi-4 chat template (default: raw encode for PPL gate)",
    )
    args = parser.parse_args()

    os.environ["ASDSL_USE_UNIFIED"] = "1"
    os.environ.setdefault("ASDSL_AFFINITY", "physical")
    os.environ.setdefault("ASDSL_CHUNKED_GEMV", "1")
    os.environ.setdefault("ASDSL_PREQ2", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "12")

    manifest_path = ROOT / "benchmarks" / "results" / "parity_manifest.json"
    manifest: dict = {}
    if manifest_path.is_file():
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for k, v in manifest.get("configs", {}).get(args.config, {}).get("env", {}).items():
            os.environ[k] = str(v)
        for k, v in manifest.get("stacks", {}).get("asdsl", {}).get("env_required", {}).items():
            os.environ.setdefault(k, str(v))

    from experiments.phi4_cpu_run import WeightStore, load_tokenizer

    manifest_prompt = manifest.get("prompt", {}) if manifest_path.is_file() else {}
    default_text = (
        "The theory of relativity usually encompasses two interrelated theories by Albert Einstein: "
        "special relativity and general relativity. " * 4
    )
    text = args.text or default_text
    system_prompt = manifest_prompt.get("asdsl_system_prompt", "") if args.chat_template else ""
    store = WeightStore(bits=args.bits)
    store._use_unified = True
    store.load()
    store._unified_engine = None
    tokenizer = load_tokenizer()

    out = compute_perplexity(
        store,
        tokenizer,
        text,
        max_tokens=args.max_tokens,
        use_chat_template=args.chat_template,
        system_prompt=system_prompt,
    )
    mode = "chat" if args.chat_template else "raw"
    print(
        f"config={args.config} mode={mode} prompt_len={out['prompt_len']} "
        f"n_tokens={out['n_tokens']} top1_acc={out['top1_acc']:.3f} "
        f"mean_nll={out['mean_nll']:.4f} perplexity={out['perplexity']:.2f}"
    )
    if args.chat_template and out["prompt_len"] < 8:
        print(
            "  note: short chat-template prompts score special tokens; "
            "use raw mode (default) for PPL gate."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
