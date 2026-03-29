"""Validate ASDSL Phi-4 profiles vs PyTorch baseline (Profile A) via KL(P||Q)."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from transformers import AutoTokenizer

from experiments.phi4_cpu_run import (
    WeightStore,
    generate,
    generate_eagle3,
)


def compute_kl_divergence(
    p_logits: np.ndarray,
    q_logits: np.ndarray,
    temperature: float = 1.0,
) -> float:
    """KL(P || Q) in nats, full vocabulary."""
    p_log_probs = torch.log_softmax(torch.tensor(p_logits, dtype=torch.float32) / temperature, dim=-1)
    q_log_probs = torch.log_softmax(torch.tensor(q_logits, dtype=torch.float32) / temperature, dim=-1)
    p_probs = torch.exp(p_log_probs)
    kl_div = torch.sum(p_probs * (p_log_probs - q_log_probs), dim=-1)
    return float(kl_div.item())


def _collect_baseline_logits(
    prompt: str,
    max_new_tokens: int,
    tokenizer: AutoTokenizer,
) -> list[np.ndarray]:
    store = WeightStore(bits=4, group_size=None, enable_qcsd=False, draft_bits=2, enable_sparse=False)
    store.load()
    store.warm_cache()
    store._use_native_gemv = False
    store._use_lut_gemv = False
    baseline: list[np.ndarray] = []

    def hook(x: np.ndarray) -> None:
        baseline.append(np.array(x, dtype=np.float32, copy=True))

    buf_out = __import__("io").StringIO()
    with __import__("contextlib").redirect_stdout(buf_out):
        generate(
            prompt,
            store,
            tokenizer,
            max_new_tokens=max_new_tokens,
            logits_hook=hook,
        )
    del store
    gc.collect()
    return baseline


def _run_profile_logits(
    profile: str,
    prompt: str,
    max_new_tokens: int,
    tokenizer: AutoTokenizer,
    root: Path,
) -> list[np.ndarray]:
    store = WeightStore(bits=4, group_size=None, enable_qcsd=False, draft_bits=2, enable_sparse=False)
    store.load()
    store.warm_cache()

    collected: list[np.ndarray] = []

    def hook(x: np.ndarray) -> None:
        collected.append(np.array(x, dtype=np.float32, copy=True))

    buf_out = __import__("io").StringIO()
    with __import__("contextlib").redirect_stdout(buf_out):
        if profile == "C":
            store._use_native_gemv = True
            store._use_lut_gemv = False
            generate(prompt, store, tokenizer, max_new_tokens=max_new_tokens, logits_hook=hook)
        elif profile == "D":
            store._use_native_gemv = True
            store._use_lut_gemv = True
            generate(prompt, store, tokenizer, max_new_tokens=max_new_tokens, logits_hook=hook)
        elif profile == "E":
            slim_p = root / "phi4_slim_meta.json"
            if not slim_p.exists():
                del store
                gc.collect()
                return []
            store.load_slim(str(slim_p))
            store._use_native_gemv = True
            store._use_lut_gemv = True
            generate(prompt, store, tokenizer, max_new_tokens=max_new_tokens, logits_hook=hook)
        elif profile == "F":
            frp = root / "phi4_fatrelu_thresholds.json"
            if not frp.exists():
                del store
                gc.collect()
                return []
            store.load_fatrelu(str(frp))
            store._use_native_gemv = True
            store._use_lut_gemv = False
            store._enable_sparse = True
            store._sparsity_threshold = 0.0
            generate(prompt, store, tokenizer, max_new_tokens=max_new_tokens, logits_hook=hook)
        elif profile == "G":
            mtp = root / "models" / "mtp_head.pt"
            frp = root / "phi4_fatrelu_thresholds.json"
            if not mtp.exists() or not frp.exists():
                del store
                gc.collect()
                return []
            store.load_fatrelu(str(frp))
            store.load_mtp_head(str(mtp))
            store._use_native_gemv = True
            store._use_lut_gemv = True
            store._enable_sparse = True
            store._sparsity_threshold = 0.0
            generate_eagle3(
                prompt,
                store,
                tokenizer,
                max_new_tokens=max_new_tokens,
                logits_hook=hook,
            )
        else:
            raise ValueError(f"unknown profile {profile!r}")

    del store
    gc.collect()
    return collected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="KL validation: baseline A vs profile C/D/E/F/G (per decode step)",
    )
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Single profile to compare to A (C, D, E, F, G). Default: run C,D,E,F,G if present.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Cap decode steps at 4",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    max_new = min(args.max_new_tokens, 4) if args.quick else args.max_new_tokens

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/Phi-4-multimodal-instruct",
        trust_remote_code=True,
    )

    print("Collecting baseline (Profile A) logits...")
    baseline_logits = _collect_baseline_logits(args.prompt, max_new, tokenizer)
    print(f"  Baseline steps with logits: {len(baseline_logits)}")

    profiles = (
        [args.profile.upper()]
        if args.profile
        else ["C", "D", "E", "F", "G"]
    )

    summary: dict[str, dict] = {}

    for prof in profiles:
        print(f"\n--- Profile {prof} vs A ---")
        q_logits = _run_profile_logits(prof, args.prompt, max_new, tokenizer, root)
        if not q_logits:
            print(f"  Skipped (missing weights/meta for profile {prof})")
            summary[prof] = {"skipped": True, "kl_mean": None}
            continue

        n = min(len(baseline_logits), len(q_logits))
        if n == 0:
            print("  ERROR: no overlapping logits")
            sys.exit(1)
        if len(baseline_logits) != len(q_logits):
            print(
                f"  WARNING: length mismatch baseline={len(baseline_logits)} "
                f"profile={len(q_logits)} — using first {n} steps",
            )

        kls: list[float] = []
        for step in range(n):
            kl = compute_kl_divergence(
                baseline_logits[step], q_logits[step], temperature=args.temperature
            )
            kls.append(kl)
            print(f"  Step {step}: KL(A || {prof}) = {kl:.6f} nats")

        mean_kl = float(sum(kls) / len(kls))
        max_kl = float(max(kls))
        print(f"  Mean KL: {mean_kl:.6f}  |  Max KL: {max_kl:.6f}")
        summary[prof] = {
            "skipped": False,
            "kl_mean": mean_kl,
            "kl_max": max_kl,
            "steps": n,
        }

    d_mean = summary.get("D", {}).get("kl_mean")
    validation_realistic = bool(d_mean is not None and d_mean > 1e-6)
    print(f"\nvalidation_realistic (Profile D mean KL > 0 vs LUT path): {validation_realistic}")

    out_path = root / "validate_outputs_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"profiles": summary, "validation_realistic": validation_realistic}, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
