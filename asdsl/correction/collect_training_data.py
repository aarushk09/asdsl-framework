"""Collect per-layer hidden-state residuals (fp32 reference vs 4-bit dispatch)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from asdsl.correction.apply_correction import HIDDEN, NUM_LAYERS

if TYPE_CHECKING:
    from experiments.phi4_cpu_run import WeightStore

DEFAULT_OUT = Path(__file__).resolve().parent / "training_data"


def _run_collect_pass(
    store: WeightStore,
    tokens: list[int],
    *,
    max_tokens: int,
    hook_layers: dict[int, list[np.ndarray]],
) -> None:
    from experiments.phi4_cpu_run import (
        KVHistory,
        ROTARY_DIM,
        build_rope_cache,
        forward_layer,
    )

    window = tokens[:max_tokens]
    if len(window) < 1:
        raise ValueError("need at least one token")

    max_seq = len(window) + 4
    rope_cos, rope_sin = build_rope_cache(max_seq, ROTARY_DIM)
    kv = KVHistory()

    for pos, tid in enumerate(window):
        hidden = store.embed_f16[tid].float().unsqueeze(0)
        for layer in range(NUM_LAYERS):
            hidden = forward_layer(
                hidden, layer, store, kv, rope_cos, rope_sin, pos=pos
            )
            hook_layers[layer].append(
                hidden.detach().cpu().float().numpy().ravel().astype(np.float32)
            )


def collect_training_data(
    ref_store: WeightStore,
    quant_store: WeightStore,
    tokens: list[int],
    output_dir: Path | str = DEFAULT_OUT,
    *,
    max_tokens: int = 512,
) -> Path:
    """Two forward passes; save layer hidden states and residuals delta = fp32 - quant."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fp32_layers: dict[int, list[np.ndarray]] = {i: [] for i in range(NUM_LAYERS)}
    quant_layers: dict[int, list[np.ndarray]] = {i: [] for i in range(NUM_LAYERS)}

    saved_correction = quant_store._correction
    quant_store._correction = None
    try:
        _run_collect_pass(ref_store, tokens, max_tokens=max_tokens, hook_layers=fp32_layers)
        _run_collect_pass(quant_store, tokens, max_tokens=max_tokens, hook_layers=quant_layers)
    finally:
        quant_store._correction = saved_correction

    arrays: dict[str, np.ndarray] = {}
    for layer in range(NUM_LAYERS):
        hf = np.stack(fp32_layers[layer], axis=0)
        hq = np.stack(quant_layers[layer], axis=0)
        arrays[f"layer_{layer}_hidden_fp32"] = hf
        arrays[f"layer_{layer}_hidden_quant"] = hq
        arrays[f"layer_{layer}_residual"] = (hf - hq).astype(np.float32)

    out_npz = output_dir / "layer_residuals.npz"
    np.savez_compressed(out_npz, **arrays)

    meta = {
        "num_tokens": min(max_tokens, len(tokens)),
        "num_layers": NUM_LAYERS,
        "hidden_size": HIDDEN,
        "ref_bits": ref_store.bits,
        "quant_bits": quant_store.bits,
        "npz": out_npz.name,
    }
    (output_dir / "collection_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return output_dir
