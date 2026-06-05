"""Bridge WeightStore to C++ UnifiedEngine (preq Q4_32 path)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from experiments.phi4_cpu_run import WeightStore


def _import_unified():
    try:
        from asdsl.kernels import _native_unified as nu

        return nu
    except ImportError as exc:
        raise RuntimeError(
            "UnifiedEngine requires _native_unified extension; run: python setup.py build_ext --inplace"
        ) from exc


def _phi4_constants():
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    exp = root / "experiments"
    if str(exp) not in sys.path:
        sys.path.insert(0, str(exp))
    import phi4_cpu_run as p4

    return p4


def build_unified_engine(store: "WeightStore") -> Any:
    """Construct UnifiedEngine from a loaded WeightStore with preq blocks."""
    nu = _import_unified()
    p4 = _phi4_constants()
    HIDDEN = p4.HIDDEN
    INTER = p4.INTER
    NUM_HEADS = p4.NUM_HEADS
    NUM_KV_HEADS = p4.NUM_KV_HEADS
    HEAD_DIM = p4.HEAD_DIM
    NUM_LAYERS = p4.NUM_LAYERS
    ROTARY_DIM = p4.ROTARY_DIM
    RMS_EPS = p4.RMS_EPS
    VOCAB = p4.VOCAB

    use_q4km = os.environ.get("ASDSL_USE_Q4KM_GGUF", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    q4km = getattr(store, "_q4km_weights", None) or {}
    mixed_q4km = bool(use_q4km and q4km)
    expected_preq = NUM_LAYERS * 4
    preq_ready = getattr(store, "_preq_built", False) and len(
        getattr(store, "_preq_blocks_np", None) or {}
    ) >= expected_preq
    if mixed_q4km:
        weight_src = "q4_k_gguf"
    else:
        weight_src = "preq"
    if not preq_ready:
        store.build_preq_blocks()

    cfg = nu.EngineConfig()
    cfg.num_layers = NUM_LAYERS
    cfg.hidden_size = HIDDEN
    cfg.intermediate_size = INTER
    cfg.num_heads = NUM_HEADS
    cfg.num_kv_heads = NUM_KV_HEADS
    cfg.head_dim = HEAD_DIM
    cfg.vocab_size = int(store.lm_head.shape[0]) if store.lm_head is not None else VOCAB
    cfg.group_size = store.group_size
    cfg.max_seq_len = 2048
    cfg.rotary_dim = ROTARY_DIM
    cfg.rms_norm_eps = RMS_EPS
    # weight_format=2 legacy (all layers q4km); mixed path uses per-projection flags.
    cfg.weight_format = 0

    embed = store.embed_f16.detach().cpu().float().numpy().ravel()
    out_norm = store.final_norm.detach().cpu().float().numpy().ravel()
    lm_head = store.lm_head.detach().cpu().float().numpy()
    cos = store._cos_table
    sin = store._sin_table
    if cos is None or sin is None:
        store._build_rope_native_tables()
        cos = store._cos_table
        sin = store._sin_table

    gguf_types = getattr(store, "_gguf_proj_types", None) or {}

    def _proj_weights(layer: int, name: str) -> tuple[np.ndarray, bool]:
        key = (layer, name)
        if mixed_q4km and key in q4km:
            return q4km[key], True
        return store._preq_blocks_np[key], False

    layers_dict: dict[int, dict] = {}
    for i in range(NUM_LAYERS):
        qkv_w, qkv_km = _proj_weights(i, "qkv_proj")
        o_w, o_km = _proj_weights(i, "o_proj")
        gu_w, gu_km = _proj_weights(i, "gate_up_proj")
        dn_w, dn_km = _proj_weights(i, "down_proj")
        layers_dict[i] = {
            "rms_att": store._norm_np.get(
                (i, "input_layernorm"),
                store.get_norm(i, "input_layernorm").detach().cpu().float().numpy(),
            ),
            "rms_ffn": store._norm_np.get(
                (i, "post_attention_layernorm"),
                store.get_norm(i, "post_attention_layernorm").detach().cpu().float().numpy(),
            ),
            "qkv_proj": qkv_w,
            "o_proj": o_w,
            "gate_up_proj": gu_w,
            "down_proj": dn_w,
            "qkv_q4km": qkv_km,
            "o_q4km": o_km,
            "gate_up_q4km": gu_km,
            "down_q4km": dn_km,
            "qkv_q5km": False,
            "down_q6km": False,
        }

    return nu.UnifiedEngine(
        cfg,
        embed,
        out_norm,
        lm_head,
        cos,
        sin,
        layers_dict,
    )


def get_or_build_unified_engine(store: "WeightStore") -> Any:
    eng = getattr(store, "_unified_engine", None)
    if eng is None:
        use_q4km = os.environ.get("ASDSL_USE_Q4KM_GGUF", "0").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        n_km = len(getattr(store, "_q4km_weights", None) or {})
        label = (
            f"mixed preq + {n_km} Q4_K GGUF projections"
            if use_q4km and n_km
            else "preq Q4_32"
        )
        print(f"  Building UnifiedEngine ({label}, C++ forward) ...")
        eng = build_unified_engine(store)
        store._unified_engine = eng
    return eng


def reset_unified_session(store: "WeightStore") -> None:
    """Clear UnifiedEngine KV state (call once per generation / PPL window)."""
    if getattr(store, "_use_unified", False) and store.bits == 4:
        get_or_build_unified_engine(store).reset_session()


def unified_forward_token(store: "WeightStore", token_id: int, pos: int) -> np.ndarray:
    """Single decode step; returns logits (vocab,)."""
    eng = get_or_build_unified_engine(store)
    logits = eng.forward_token(int(token_id), int(pos))
    return np.asarray(logits, dtype=np.float32)

