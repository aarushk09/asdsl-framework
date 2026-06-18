"""Per-layer draft skip impact profile (logit cosine vs full forward).

Uses the same probe as ``asdsl.speculative.ahsd.compute_skip_mask`` but prints
a full table and writes JSON for the AHSD diagnostic pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from asdsl.inference.unified_bridge import (  # noqa: E402
    clear_skip_mask,
    get_or_build_unified_engine,
    set_skip_mask,
)
from asdsl.speculative.ahsd import _logit_cosine, _prefill_prefix  # noqa: E402
from experiments.phi4_cpu_run import (  # noqa: E402
    WeightStore,
    _normalize_input_ids,
    set_thread_count,
)


def _build_prefix_ids(
    store: WeightStore,
    tokenizer,
    prompt: str,
    system_prompt: str,
    target_len: int,
) -> list[int]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    ids = list(
        _normalize_input_ids(
            tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        )
    )
    if len(ids) >= target_len:
        return ids[:target_len]

    eng = get_or_build_unified_engine(store)
    eng.reset_session()
    for pos, tid in enumerate(ids):
        logits = eng.forward_token(int(tid), int(pos))
    pos = len(ids)
    while len(ids) < target_len:
        tid = int(np.argmax(np.asarray(logits, dtype=np.float32)))
        ids.append(tid)
        pos = len(ids) - 1
        logits = eng.forward_token(int(tid), int(pos))
    return ids


def profile_layer_skip_impact(
    store: WeightStore,
    token_ids: list[int],
    *,
    threshold: float = 0.97,
    skip_first: int = 8,
    skip_last: int = 8,
    probe_positions: int = 3,
) -> dict:
    """Measure per-layer logit cosine when skipping exactly that layer."""
    from experiments.phi4_cpu_run import NUM_LAYERS

    num_layers = int(os.environ.get("ASDSL_NUM_LAYERS", str(NUM_LAYERS)))
    eng = get_or_build_unified_engine(store)

    if not hasattr(eng, "forward_token_draft"):
        raise RuntimeError("UnifiedEngine missing forward_token_draft; rebuild native extension")

    if len(token_ids) < 4:
        raise ValueError(f"need at least 4 prefix tokens, got {len(token_ids)}")

    probe_positions = max(1, min(probe_positions, len(token_ids) - 1))
    probe_indices = list(range(len(token_ids) - probe_positions, len(token_ids)))

    layer_scores: dict[int, list[float]] = {layer: [] for layer in range(num_layers)}

    for probe_pos in probe_indices:
        probe_tid = int(token_ids[probe_pos])
        prefix_len = probe_pos

        eng.reset_session()
        _prefill_prefix(eng, token_ids, prefix_len)
        full_logits = np.asarray(
            eng.forward_token(int(probe_tid), int(probe_pos)), dtype=np.float32
        ).copy()

        for layer in range(num_layers):
            layer_mask = np.zeros(num_layers, dtype=bool)
            layer_mask[layer] = True

            eng.reset_session()
            _prefill_prefix(eng, token_ids, prefix_len)
            set_skip_mask(eng, layer_mask)
            draft_logits = np.asarray(
                eng.forward_token_draft(int(probe_tid), int(probe_pos)), dtype=np.float32
            ).copy()
            clear_skip_mask(eng)
            layer_scores[layer].append(_logit_cosine(full_logits, draft_logits))

    rows = []
    high_sim_count = 0
    for layer in range(num_layers):
        mean_cos = float(np.mean(layer_scores[layer])) if layer_scores[layer] else 0.0
        in_band = skip_first <= layer < num_layers - skip_last
        skip_candidate = in_band and mean_cos > threshold
        if mean_cos > threshold:
            high_sim_count += 1
        rows.append(
            {
                "layer": layer,
                "cosine_sim": round(mean_cos, 6),
                "skip_candidate": bool(skip_candidate),
                "in_middle_band": bool(in_band),
            }
        )

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "metric": "logit_cosine_full_vs_single_layer_skip",
        "threshold": threshold,
        "skip_first": skip_first,
        "skip_last": skip_last,
        "prefix_len": len(token_ids),
        "probe_positions": probe_indices,
        "layers_above_threshold": high_sim_count,
        "middle_band_skip_candidates": sum(1 for r in rows if r["skip_candidate"]),
        "layers": rows,
    }


def _print_table(profile: dict) -> None:
    thresh = profile["threshold"]
    print(f"\nLayer skip impact (logit cosine, threshold={thresh:.3f})")
    print(f"{'Layer':>5} | {'Cosine Sim':>10} | {'Skip candidate?':>15}")
    print("-" * 36)
    for row in profile["layers"]:
        flag = "yes" if row["skip_candidate"] else "no"
        print(f"{row['layer']:5d} | {row['cosine_sim']:10.4f} | {flag:>15}")
    print("-" * 36)
    print(
        f"Layers with cos_sim > {thresh:.3f}: {profile['layers_above_threshold']}/{len(profile['layers'])}"
    )
    print(f"Middle-band skip candidates: {profile['middle_band_skip_candidates']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AHSD per-layer skip impact profile")
    parser.add_argument("--prompt", default="The")
    parser.add_argument("--system-prompt", default="", help='Parity default is ""')
    parser.add_argument("--prefix-len", type=int, default=30)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=0.97)
    parser.add_argument("--skip-first", type=int, default=8)
    parser.add_argument("--skip-last", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "benchmarks" / "results" / "layer_similarity_profile.json",
    )
    args = parser.parse_args()

    os.environ.setdefault("ASDSL_USE_UNIFIED", "1")
    os.environ.setdefault("ASDSL_FUSED_GEMV", "1")
    os.environ.setdefault("ASDSL_AFFINITY", "legacy")
    os.environ.setdefault("ASDSL_PREQ_PREFETCH_GROUPS", "0")

    set_thread_count(args.threads)

    from transformers import AutoTokenizer

    print("Loading tokenizer and weights ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/Phi-4-multimodal-instruct", trust_remote_code=True
    )
    store = WeightStore(bits=4, group_size=32, enable_qcsd=False)
    store.load()
    store.warm_cache()

    token_ids = _build_prefix_ids(
        store, tokenizer, args.prompt, args.system_prompt, args.prefix_len
    )
    print(f"Prefix: {len(token_ids)} tokens from prompt {args.prompt!r}", flush=True)

    profile = profile_layer_skip_impact(
        store,
        token_ids,
        threshold=args.threshold,
        skip_first=args.skip_first,
        skip_last=args.skip_last,
    )
    _print_table(profile)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
