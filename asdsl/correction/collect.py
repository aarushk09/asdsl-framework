"""Collect per-layer hidden residuals (quant vs reference) for correction training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from asdsl.correction.apply import HIDDEN, NUM_LAYERS

if TYPE_CHECKING:
    from experiments.phi4_cpu_run import WeightStore


def collect_residuals(
    quant_store: WeightStore,
    tokens: list[int],
    output_dir: Path | str,
    *,
    ref_store: WeightStore | None = None,
    max_tokens: int = 64,
) -> Path:
    """Run forward passes and save per-layer hidden states and residuals.

    When ``ref_store`` is provided (typically bits=16), residuals are
    ``h_ref - h_quant`` after each layer. Otherwise only quantized hiddens
    are stored and residuals are zero (train will fit biases from paired run).
    """
    from experiments.phi4_cpu_run import (
        KVHistory,
        ROTARY_DIM,
        build_rope_cache,
        forward_layer,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    window = tokens[:max_tokens]
    if len(window) < 1:
        raise ValueError("need at least one token for collection")

    max_seq = len(window) + 4
    rope_cos, rope_sin = build_rope_cache(max_seq, ROTARY_DIM)

    h_q: list[np.ndarray] = []
    residuals: dict[int, list[np.ndarray]] = {i: [] for i in range(NUM_LAYERS)}

    for pos, tid in enumerate(window):
        hidden_q = quant_store.embed_f16[tid].float().unsqueeze(0)
        hidden_r = (
            ref_store.embed_f16[tid].float().unsqueeze(0)
            if ref_store is not None
            else None
        )
        kv_q = KVHistory()
        kv_r = KVHistory() if ref_store is not None else None

        for layer in range(NUM_LAYERS):
            hidden_q = forward_layer(
                hidden_q, layer, quant_store, kv_q, rope_cos, rope_sin, pos=pos
            )
            if ref_store is not None and hidden_r is not None and kv_r is not None:
                hidden_r = forward_layer(
                    hidden_r, layer, ref_store, kv_r, rope_cos, rope_sin, pos=pos
                )
                hq = hidden_q.detach().cpu().float().numpy().ravel()
                hr = hidden_r.detach().cpu().float().numpy().ravel()
                residuals[layer].append((hr - hq).astype(np.float32))
            else:
                residuals[layer].append(
                    np.zeros(HIDDEN, dtype=np.float32)
                )

        h_q.append(hidden_q.detach().cpu().float().numpy().ravel())

    stack_res = np.stack(
        [np.mean(residuals[l], axis=0) if residuals[l] else np.zeros(HIDDEN, np.float32)
         for l in range(NUM_LAYERS)],
        axis=0,
    )
    np.savez_compressed(
        output_dir / "correction_samples.npz",
        residuals=stack_res,
        num_tokens=len(window),
        hidden_size=HIDDEN,
        num_layers=NUM_LAYERS,
    )
    meta = {
        "num_tokens": len(window),
        "has_reference": ref_store is not None,
        "hidden_size": HIDDEN,
        "num_layers": NUM_LAYERS,
    }
    (output_dir / "correction_samples.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return output_dir
