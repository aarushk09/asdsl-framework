"""Adaptive Hidden-State Self-Speculative Decoding (AHSD) skip-mask calibration."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from experiments.phi4_cpu_run import WeightStore


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return float(raw)


def _static_skip_mask(num_layers: int, skip_first: int, skip_last: int) -> np.ndarray:
    """Default middle-band skip mask (layers 10–21 on 32-layer Phi-4)."""
    mask = np.zeros(num_layers, dtype=bool)
    for layer in range(10, min(22, num_layers)):
        if skip_first <= layer < num_layers - skip_last:
            mask[layer] = True
    return mask


def _logit_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b) / denom)


def _prefill_prefix(eng, token_ids: list[int], upto: int) -> None:
    for pos in range(upto):
        eng.forward_token(int(token_ids[pos]), int(pos))


def compute_skip_mask(
    store: "WeightStore",
    calibration_tokens: int = 50,
    threshold: float = 0.97,
    skip_first: int = 8,
    skip_last: int = 8,
) -> np.ndarray:
    """Build per-layer draft skip mask from full-vs-single-layer-skip logit cosine.

    For each candidate layer, prefill a short calibration prefix, compare
    ``forward_token`` vs ``forward_token_draft`` (with only that layer skipped),
    and keep layers whose mean cosine similarity exceeds ``threshold``.
    """
    from asdsl.inference.unified_bridge import (
        clear_skip_mask,
        get_or_build_unified_engine,
        set_skip_mask,
    )

    num_layers = int(os.environ.get("ASDSL_NUM_LAYERS", "32"))
    skip_first = _env_int("ASDSL_AHSD_SKIP_FIRST", skip_first)
    skip_last = _env_int("ASDSL_AHSD_SKIP_LAST", skip_last)
    threshold = _env_float("ASDSL_AHSD_THRESH", threshold)
    calibration_tokens = max(4, int(calibration_tokens))

    eng = get_or_build_unified_engine(store)
    cal_ids = list(range(min(calibration_tokens, 128)))

    if not hasattr(eng, "forward_token_draft"):
        return _static_skip_mask(num_layers, skip_first, skip_last)

    probe_pos = len(cal_ids) - 1
    probe_tid = cal_ids[probe_pos]
    prefix_len = probe_pos

    mask = np.zeros(num_layers, dtype=bool)
    for layer in range(skip_first, num_layers - skip_last):
        layer_mask = np.zeros(num_layers, dtype=bool)
        layer_mask[layer] = True

        eng.reset_session()
        _prefill_prefix(eng, cal_ids, prefix_len)
        full_logits = np.asarray(
            eng.forward_token(int(probe_tid), int(probe_pos)), dtype=np.float32
        ).copy()

        eng.reset_session()
        _prefill_prefix(eng, cal_ids, prefix_len)
        set_skip_mask(eng, layer_mask)
        draft_logits = np.asarray(
            eng.forward_token_draft(int(probe_tid), int(probe_pos)), dtype=np.float32
        ).copy()
        clear_skip_mask(eng)

        if _logit_cosine(full_logits, draft_logits) >= threshold:
            mask[layer] = True

    if not mask.any():
        mask = _static_skip_mask(num_layers, skip_first, skip_last)

    n_skip = int(mask.sum())
    print(
        f"  AHSD skip mask: {n_skip}/{num_layers} layers "
        f"(thresh={threshold:.3f}, first={skip_first}, last={skip_last})",
        flush=True,
    )
    return mask


def calibrate_and_store_skip_mask(store: "WeightStore") -> np.ndarray:
    """Run calibration and cache mask on ``store._ahsd_skip_mask``."""
    mask = compute_skip_mask(store)
    store._ahsd_skip_mask = mask
    return mask
