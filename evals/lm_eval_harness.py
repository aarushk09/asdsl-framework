"""
ASDSL model wrapper for EleutherAI's lm-evaluation-harness.

Allows running standard 0-shot benchmarks (HellaSwag, PIQA, ARC, etc.)
against ASDSL-quantized Phi-4 inference.

Installation (one-time):
  pip install lm-eval

Usage:
  # Run HellaSwag 0-shot through asdsl (float16 baseline):
  python evals/lm_eval_harness.py --tasks hellaswag --bits 16

  # Run PIQA and HellaSwag with ASDSL 4-bit:
  python evals/lm_eval_harness.py --tasks hellaswag,piqa --bits 4

  # You can also register this as a custom model directly:
  lm_eval --model asdsl --model_args bits=8 --tasks piqa --num_fewshot 0
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from phi4_cpu_run import (
    WeightStore, KVHistory,
    forward_layer, rms_norm, build_rope_cache,
    NUM_LAYERS, ROTARY_DIM, EOS_TOKEN_IDS,
)

MODEL_DIR = ROOT / "models" / "phi4-multimodal-instruct"

try:
    from lm_eval.api.model import LM
    from lm_eval.api.instance import Instance
    from lm_eval.api.registry import register_model
    _HAS_LM_EVAL = True
except ImportError:
    _HAS_LM_EVAL = False
    # Stub so the class definition doesn't fail
    class LM:  # type: ignore[no-redef]
        pass
    def register_model(name):  # type: ignore[no-redef]
        return lambda cls: cls
    class Instance:  # type: ignore[no-redef]
        pass


# ---------------------------------------------------------------------------
# Core inference helper (shared by all harness methods)
# ---------------------------------------------------------------------------

def _run_forward_sequence(
    token_ids: list[int],
    store: WeightStore,
    return_all_logits: bool = False,
) -> Union[torch.Tensor, list[torch.Tensor]]:
    """Run a full forward pass over a sequence of token IDs.

    Args:
        token_ids: Token IDs to process.
        store: Loaded WeightStore with warm cache.
        return_all_logits: If True, return logits for every position.
            If False (default), return only the last position's logits.

    Returns:
        If return_all_logits: list of (vocab,) tensors, one per position.
        Otherwise: single (vocab,) tensor for the final position.
    """
    max_seq = len(token_ids) + 64
    rope_cos, rope_sin = build_rope_cache(max_seq, ROTARY_DIM)
    kv_hist = KVHistory()
    all_logits = [] if return_all_logits else None

    logits = None
    for pos, tid in enumerate(token_ids):
        hidden = store.embed_f32[tid].unsqueeze(0)
        for layer in range(NUM_LAYERS):
            hidden = forward_layer(
                hidden, layer, store, kv_hist, rope_cos, rope_sin, pos=pos,
            )
        hidden = rms_norm(hidden, store.final_norm)
        logits = (hidden @ store.lm_head.t()).squeeze(0)
        if return_all_logits:
            all_logits.append(logits)

    if return_all_logits:
        return all_logits
    return logits


# ---------------------------------------------------------------------------
# lm-evaluation-harness model class
# ---------------------------------------------------------------------------

@register_model("asdsl")
class ASDSLHarnessModel(LM):
    """EleutherAI lm-eval-harness compatible wrapper for ASDSL inference."""

    def __init__(
        self,
        bits: int = 16,
        batch_size: int = 1,
        max_length: int = 2048,
        **kwargs,
    ):
        if _HAS_LM_EVAL:
            super().__init__()
        self._bits = bits
        self._batch_size = batch_size
        self._max_length = max_length

        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-4-multimodal-instruct", trust_remote_code=True
        )

        self._store = WeightStore(bits=bits)
        self._store.load()
        self._store.warm_cache()

    # ---- Required by lm-eval API ----

    @property
    def eot_id(self) -> int:
        return 199999  # <|endoftext|>

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> str:
        return "cpu"

    def tok_encode(self, string: str, **kwargs) -> list[int]:
        return self._tokenizer.encode(string)

    def tok_decode(self, tokens: list[int], **kwargs) -> str:
        return self._tokenizer.decode(tokens, skip_special_tokens=True)

    def _logits_for_context(self, context_ids: list[int]) -> list[torch.Tensor]:
        """Return logits at every position for a given context."""
        return _run_forward_sequence(context_ids, self._store, return_all_logits=True)

    def loglikelihood(
        self, requests: list,
    ) -> list[tuple[float, bool]]:
        """Compute log-likelihood of continuations given contexts.

        Each request is an Instance with .args = (context_str, continuation_str).
        Returns list of (log_likelihood, is_greedy) tuples.
        """
        results = []
        for req in requests:
            if hasattr(req, 'args'):
                context, continuation = req.args
            else:
                context, continuation = req

            ctx_ids = self.tok_encode(context)
            cont_ids = self.tok_encode(continuation)
            full_ids = (ctx_ids + cont_ids)[-self._max_length:]

            # Recompute where continuation starts after potential truncation
            cont_start = len(full_ids) - len(cont_ids)

            all_logits = self._logits_for_context(full_ids)

            # Sum log-probs for continuation tokens
            log_likelihood = 0.0
            is_greedy = True
            for i, target_id in enumerate(cont_ids):
                pos = cont_start + i  # position whose logits predict target_id
                if pos == 0:
                    continue  # no logits before the first token
                logit_vec = all_logits[pos - 1].float()
                log_probs = torch.log_softmax(logit_vec, dim=-1)
                log_likelihood += log_probs[target_id].item()
                if logit_vec.argmax().item() != target_id:
                    is_greedy = False

            results.append((log_likelihood, is_greedy))
        return results

    def loglikelihood_rolling(
        self, requests: list,
    ) -> list[tuple[float, bool]]:
        """Rolling (unconditional) log-likelihood over full strings."""
        results = []
        for req in requests:
            if hasattr(req, 'args'):
                (text,) = req.args
            else:
                text = req[0] if isinstance(req, (tuple, list)) else req

            token_ids = self.tok_encode(text)[-self._max_length:]
            if len(token_ids) < 2:
                results.append((0.0, True))
                continue

            all_logits = self._logits_for_context(token_ids)

            log_likelihood = 0.0
            is_greedy = True
            for i in range(1, len(token_ids)):
                logit_vec = all_logits[i - 1].float()
                log_probs = torch.log_softmax(logit_vec, dim=-1)
                target = token_ids[i]
                log_likelihood += log_probs[target].item()
                if logit_vec.argmax().item() != target:
                    is_greedy = False

            results.append((log_likelihood, is_greedy))
        return results

    def generate_until(
        self, requests: list,
    ) -> list[str]:
        """Generate text until a stop condition for each request."""
        results = []
        for req in requests:
            if hasattr(req, 'args'):
                context, gen_kwargs = req.args
            else:
                context, gen_kwargs = req

            until = gen_kwargs.get("until", [])
            max_gen = gen_kwargs.get("max_gen_toks", self.max_gen_toks)

            ctx_ids = self.tok_encode(context)[-self._max_length:]

            max_seq = len(ctx_ids) + max_gen + 64
            rope_cos, rope_sin = build_rope_cache(max_seq, ROTARY_DIM)
            kv_hist = KVHistory()

            # Prefill
            logits = None
            for pos, tid in enumerate(ctx_ids):
                hidden = self._store.embed_f32[tid].unsqueeze(0)
                for layer in range(NUM_LAYERS):
                    hidden = forward_layer(
                        hidden, layer, self._store, kv_hist,
                        rope_cos, rope_sin, pos=pos,
                    )
                hidden = rms_norm(hidden, self._store.final_norm)
                logits = (hidden @ self._store.lm_head.t()).squeeze(0)

            # Decode
            generated = []
            pos = len(ctx_ids)
            for _ in range(max_gen):
                next_token = int(logits.argmax())
                generated.append(next_token)

                if next_token in EOS_TOKEN_IDS:
                    break

                text_so_far = self.tok_decode(generated)
                if any(s in text_so_far for s in until):
                    break

                hidden = self._store.embed_f32[next_token].unsqueeze(0)
                for layer in range(NUM_LAYERS):
                    hidden = forward_layer(
                        hidden, layer, self._store, kv_hist,
                        rope_cos, rope_sin, pos=pos,
                    )
                hidden = rms_norm(hidden, self._store.final_norm)
                logits = (hidden @ self._store.lm_head.t()).squeeze(0)
                pos += 1

            gen_text = self.tok_decode(generated)
            # Trim at stop strings
            for s in until:
                idx = gen_text.find(s)
                if idx != -1:
                    gen_text = gen_text[:idx]
            results.append(gen_text)

        return results


# ---------------------------------------------------------------------------
# CLI for standalone use
# ---------------------------------------------------------------------------

def main():
    if not _HAS_LM_EVAL:
        print("ERROR: lm-eval not installed. Run: pip install lm-eval")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Run EleutherAI lm-eval-harness benchmarks through ASDSL"
    )
    parser.add_argument("--bits", type=int, default=16, choices=[2, 3, 4, 8, 16])
    parser.add_argument("--tasks", type=str, default="hellaswag",
                        help="Comma-separated task names (e.g. hellaswag,piqa)")
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None,
                        help="Max examples per task (for quick testing)")
    args = parser.parse_args()

    import lm_eval

    model = ASDSLHarnessModel(bits=args.bits)

    task_names = [t.strip() for t in args.tasks.split(",")]

    print(f"\n{'='*66}")
    print(f"  lm-eval-harness: tasks={task_names}, bits={args.bits}, "
          f"fewshot={args.num_fewshot}")
    print(f"{'='*66}\n")

    results = lm_eval.simple_evaluate(
        model=model,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
    )

    # Print results table
    print(f"\n{'='*66}")
    print(f"  Results (ASDSL {args.bits}-bit)")
    print(f"{'='*66}")
    if "results" in results:
        for task_name, task_results in results["results"].items():
            print(f"\n  {task_name}:")
            for metric, value in sorted(task_results.items()):
                if isinstance(value, float):
                    print(f"    {metric}: {value:.4f}")
                else:
                    print(f"    {metric}: {value}")
    print(f"{'='*66}")


if __name__ == "__main__":
    main()
