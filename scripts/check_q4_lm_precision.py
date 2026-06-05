"""
Q4 LM Head Precision Check
===========================
Compares argmax token agreement between ASDSL's Q4 LM head (lm_head_q4_blocks_)
and the FP32 fallback path over N decode steps.

Usage:
  python scripts/check_q4_lm_precision.py [--steps 100] [--verbose]

Acceptance criterion: agreement rate >= 95% (i.e. the Q4 quantization does not
materially change which token is selected under greedy decoding).

If agreement drops below 95%, the Q4 LM head precision is hurting token quality;
revert to Q8 (or FP32) for the LM head while keeping Q4 for layer projections.

Requires: models/phi4_14b_q4_32_metadata.json and models/phi4_14b_q4_32.bin
to be present (same as benchmark_runner.py).
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import time
import traceback

import numpy as np

try:
    from asdsl.kernels._native_unified import EngineConfig, UnifiedEngine
except ImportError as e:
    raise SystemExit(
        "Could not import _native_unified. "
        "Build the C++ extension first: python setup.py build_ext --inplace\n"
        f"Original error: {e}"
    )


def build_engine(c: EngineConfig, embed_fp32, final_norm, lm_head,
                 cos_t, sin_t, layers, *, use_q4_lm_head: bool):
    """
    Build a UnifiedEngine. When use_q4_lm_head=False, pass None for lm_head
    so the C++ constructor skips Q4 quantization and falls back to FP32 GEMV.
    """
    proj = lm_head if use_q4_lm_head else np.zeros((0,), dtype=np.float32)
    return UnifiedEngine(c, embed_fp32, final_norm, proj, cos_t, sin_t, layers)


def single_token_argmax(engine, token_id: int, pos: int) -> int:
    """Run one forward pass and return argmax of logits."""
    logits = engine.forward_token(token_id, pos)
    return int(np.argmax(logits))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=100,
                        help="Number of decode steps to compare (default: 100)")
    parser.add_argument("--prompt", type=str,
                        default="The fundamental theorem of calculus states that",
                        help="Prompt to use for the comparison")
    parser.add_argument("--verbose", action="store_true",
                        help="Print each token comparison")
    parser.add_argument("--meta", default="models/phi4_14b_q4_32_metadata.json")
    parser.add_argument("--bin",  default="models/phi4_14b_q4_32.bin")
    args = parser.parse_args()

    print("Q4 LM Head Precision Check", flush=True)
    print(f"Steps: {args.steps}, Prompt: {args.prompt!r}", flush=True)
    print(flush=True)

    # ── Load weights ──────────────────────────────────────────────────────────
    print("Loading metadata…", flush=True)
    meta = json.load(open(args.meta))
    mm = np.memmap(args.bin, dtype=np.uint8, mode="r")

    def get_fp32(key):
        info = meta[key]
        raw = mm[info["offset"]: info["offset"] + info["size_bytes"]]
        return raw.view(np.float16).astype(np.float32).reshape(info["shape"])

    def get_q4(key):
        info = meta[key]
        return mm[info["offset"]: info["offset"] + info["size_bytes"]]

    # ── EngineConfig ──────────────────────────────────────────────────────────
    c = EngineConfig()
    c.num_layers        = 40
    c.hidden_size       = 5120
    c.num_heads         = 40
    c.num_kv_heads      = 10
    c.head_dim          = 128
    c.rotary_dim        = 128
    c.intermediate_size = 17920
    c.vocab_size        = 100352
    c.rms_norm_eps      = 1e-5
    c.group_size        = 32
    c.max_seq_len       = 512

    pos_idx = np.arange(c.max_seq_len, dtype=np.float32)
    dim_idx = np.arange(c.rotary_dim // 2, dtype=np.float32)
    inv_freq = 1.0 / (250000.0 ** (2.0 * dim_idx / c.head_dim))
    t_mat = pos_idx[:, None] * inv_freq[None, :]
    cos_t = np.cos(t_mat).astype(np.float32)
    sin_t = np.sin(t_mat).astype(np.float32)

    embed_fp32  = get_fp32("model.embed_tokens.weight")
    lm_head_fp32 = get_fp32("lm_head.weight")
    final_norm  = get_fp32("model.norm.weight").flatten()

    layers = {}
    print("Loading layer weights…", flush=True)
    for i in range(c.num_layers):
        layers[i] = {
            "rms_att":      get_fp32(f"model.layers.{i}.input_layernorm.weight").flatten(),
            "rms_ffn":      get_fp32(f"model.layers.{i}.post_attention_layernorm.weight").flatten(),
            "qkv_proj":     get_q4(f"model.layers.{i}.self_attn.qkv_proj.weight"),
            "o_proj":       get_q4(f"model.layers.{i}.self_attn.o_proj.weight"),
            "gate_up_proj": get_q4(f"model.layers.{i}.mlp.gate_up_proj.weight"),
            "down_proj":    get_q4(f"model.layers.{i}.mlp.down_proj.weight"),
        }

    # ── Build single engine (Q4 LM head) ─────────────────────────────────────
    # We use ONE engine instance to avoid sharing the singleton ThreadPool
    # across two concurrent engine states.  The FP32 reference logits are
    # computed via forward_token_fp32_lmhead(), which runs the same transformer
    # layers but applies the FP32 LM-head projection instead of the Q4 path.
    print("Building Q4 LM head engine…", flush=True)
    t0 = time.perf_counter()
    engine_q4 = UnifiedEngine(c, embed_fp32, final_norm, lm_head_fp32, cos_t, sin_t, layers)
    print(f"  Q4 engine built in {time.perf_counter()-t0:.1f}s", flush=True)
    print("  FP32 reference path: forward_token_fp32_lmhead() on same engine instance.", flush=True)

    # ── Tokenise prompt ───────────────────────────────────────────────────────
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-4", trust_remote_code=True)
        prompt_tokens = tokenizer.encode(args.prompt)
        print(f"Prompt tokens: {prompt_tokens} ({len(prompt_tokens)} tokens)", flush=True)
    except Exception:
        # Fallback: just use a few fixed token IDs as a seed
        print("transformers not available — using fixed seed tokens [1, 2, 3]", flush=True)
        prompt_tokens = [1, 2, 3]

    # ── Prefill the engine's KV cache with the prompt ────────────────────────
    print("Prefilling engine with prompt tokens…", flush=True)
    for idx, tok_id in enumerate(prompt_tokens[:-1]):
        engine_q4.forward_token(tok_id, idx)

    # ── Run N decode steps, comparing argmax ──────────────────────────────────
    print(f"\nComparing {args.steps} decode steps…", flush=True)
    matches = 0
    mismatches = 0
    current_tok = prompt_tokens[-1] if prompt_tokens else 1
    pos = len(prompt_tokens) - 1

    if args.verbose:
        print(f"{'Step':>5}  {'Token':>6}  {'Q4':>6}  {'FP32':>6}  {'Match':>5}")

    for step in range(args.steps):
        if pos >= c.max_seq_len:
            raise RuntimeError(
                f"Decode position {pos} reached max_seq_len={c.max_seq_len}. "
                "Increase EngineConfig.max_seq_len or reduce --steps/prompt length."
            )
        # Q4 path: uses internal Q4-quantised LM head
        logits_q4   = engine_q4.forward_token(current_tok, pos)
        # FP32 reference path: same transformer layers, FP32 LM-head projection.
        # forward_token_fp32_lmhead re-runs the full forward pass (including
        # advancing pos in the KV cache) so we must pass the same current_tok
        # and pos.  We call it AFTER forward_token so the KV cache position is
        # already populated; forward_token_fp32_lmhead with out_logits=nullptr
        # runs all layers again at the same pos — the KV cache write is
        # idempotent for the same (layer, pos) slot.
        logits_fp32 = engine_q4.forward_token_fp32_lmhead(current_tok, pos, lm_head_fp32)

        tok_q4   = int(np.argmax(logits_q4))
        tok_fp32 = int(np.argmax(logits_fp32))
        match    = tok_q4 == tok_fp32

        if match:
            matches += 1
        else:
            mismatches += 1

        if args.verbose:
            flag = "✓" if match else "✗"
            print(f"{step:>5}  {current_tok:>6}  {tok_q4:>6}  {tok_fp32:>6}  {flag:>5}")

        # Advance using the Q4 engine's prediction (primary path)
        current_tok = tok_q4
        pos += 1

    # ── Report ────────────────────────────────────────────────────────────────
    total       = matches + mismatches
    agree_pct   = 100.0 * matches / total if total > 0 else 0.0
    threshold   = 95.0

    print()
    print("=" * 50)
    print(f"Q4 LM Head Precision Report ({total} steps)")
    print(f"  Matches:    {matches:>4} / {total}")
    print(f"  Mismatches: {mismatches:>4} / {total}")
    print(f"  Agreement:  {agree_pct:.1f}%  (threshold: {threshold:.0f}%)")
    print()
    if agree_pct >= threshold:
        print(f"PASS  Q4 LM head is safe to use (agreement {agree_pct:.1f}% ≥ {threshold:.0f}%)")
    else:
        print(f"FAIL  Q4 LM head degrades quality (agreement {agree_pct:.1f}% < {threshold:.0f}%)")
        print("      Revert lm_head to Q8 or FP32 to restore accuracy.")
    print("=" * 50)


if __name__ == "__main__":
    # Ensure native crashes (e.g., segfault/access violation in extension code)
    # emit diagnostics instead of appearing as a silent exit.
    faulthandler.enable()
    try:
        main()
    except BaseException:
        print("\nERROR: check_q4_lm_precision.py failed with exception:", flush=True)
        traceback.print_exc()
        raise
