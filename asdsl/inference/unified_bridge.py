"""Bridge WeightStore to C++ UnifiedEngine (preq Q4_32 path)."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from experiments.phi4_cpu_run import WeightStore


@dataclass
class AhsdResult:
    """Result of AHSD or SDQS generation via UnifiedEngine."""

    token_ids: list[int]
    acceptance_rate: float
    draft_tokens: int
    accepted_tokens: int
    speculative_cycles: int
    draft_ms: float
    verify_ms: float
    prompt_len: int
    decode_tokens: int
    decode_s: float
    tokens_per_second: float


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


def default_ahsd_skip_mask(num_layers: int = 32) -> np.ndarray:
    """Static AHSD skip mask: layers 10–21 (inclusive) skipped during draft."""
    mask = np.zeros(num_layers, dtype=bool)
    for layer in range(10, min(22, num_layers)):
        mask[layer] = True
    return mask


def snapshot_kv(engine: Any) -> None:
    """Push current KV sequence length onto the engine snapshot stack."""
    engine.snapshot_kv()


def restore_kv(engine: Any) -> None:
    """Restore KV cache to the most recent snapshot."""
    engine.restore_kv()


def truncate_kv(engine: Any, n: int) -> None:
    """Truncate committed KV length to ``n`` positions."""
    engine.truncate_kv(int(n))


def get_kv_seq_len(engine: Any) -> int:
    """Return committed KV sequence length."""
    return int(engine.get_kv_seq_len())


def set_skip_mask(engine: Any, mask: np.ndarray) -> None:
    """Set per-layer draft skip mask (True = skip layer during draft)."""
    m = np.asarray(mask, dtype=bool).ravel()
    engine.set_skip_mask(m)


def clear_skip_mask(engine: Any) -> None:
    """Clear draft skip mask (no layers skipped)."""
    engine.clear_skip_mask()


def _attach_q2_weights(
    layers_dict: dict[int, dict],
    store: "WeightStore",
    num_layers: int,
) -> None:
    """Load Q2 draft bank into layers_dict when QCSD draft weights are available."""
    if not getattr(store, "_enable_qcsd", False):
        return
    packed = getattr(store, "_draft_quant_packed_np", None) or {}
    u8 = getattr(store, "_draft_quant_u8_np", None) or {}
    sc_np = getattr(store, "_draft_quant_sc_np", None) or {}
    bi_np = getattr(store, "_draft_quant_bi_np", None) or {}
    if not packed and not u8:
        return

    proj_map = {
        "qkv_proj": ("q2_qkv_proj", "q2_qkv_scales", "q2_qkv_biases"),
        "o_proj": ("q2_o_proj", "q2_o_scales", "q2_o_biases"),
        "gate_up_proj": ("q2_gate_up_proj", "q2_gate_up_scales", "q2_gate_up_biases"),
        "down_proj": ("q2_down_proj", "q2_down_scales", "q2_down_biases"),
    }

    for i in range(num_layers):
        lw = layers_dict[i]
        has_q2 = False
        for pname, (wkey, skey, bkey) in proj_map.items():
            key = (i, pname)
            w_arr = packed.get(key) or u8.get(key)
            if w_arr is None:
                continue
            sc = sc_np.get(key)
            bi = bi_np.get(key)
            if sc is None or bi is None:
                continue
            lw[wkey] = np.ascontiguousarray(w_arr, dtype=np.uint8)
            lw[skey] = np.ascontiguousarray(sc, dtype=np.float32)
            lw[bkey] = np.ascontiguousarray(bi, dtype=np.float32)
            has_q2 = True
        lw["has_q2"] = has_q2


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
    if hasattr(store, "build_preq2_blocks"):
        store.build_preq2_blocks()
    if hasattr(store, "build_c01_gs128_blocks"):
        store.build_c01_gs128_blocks()
    if hasattr(store, "_free_packed_after_unified_repack"):
        store._free_packed_after_unified_repack()

    cfg = nu.EngineConfig()
    cfg.num_layers = NUM_LAYERS
    cfg.hidden_size = HIDDEN
    cfg.intermediate_size = INTER
    cfg.num_heads = NUM_HEADS
    cfg.num_kv_heads = NUM_KV_HEADS
    cfg.head_dim = HEAD_DIM
    cfg.vocab_size = int(store.lm_head.shape[0]) if store.lm_head is not None else VOCAB
    cfg.group_size = store.group_size
    cfg.lm_head_group_size = int(os.environ.get("ASDSL_LMHEAD_GS", str(store.group_size)))
    cfg.max_seq_len = 2048
    cfg.rotary_dim = ROTARY_DIM
    cfg.rms_norm_eps = RMS_EPS
    cfg.weight_format = 0

    # Pass fp16 embed/lm_head directly — engine converts per-token / quantizes lm_head at init.
    embed_np = store.embed_f16.detach().cpu().numpy()
    if embed_np.dtype != np.float16:
        embed_np = embed_np.astype(np.float16, copy=False)
    embed = embed_np if embed_np.flags["C_CONTIGUOUS"] else np.ascontiguousarray(embed_np)
    out_norm = store.final_norm.detach().cpu().float().numpy().ravel()
    preq2_on = os.environ.get("ASDSL_PREQ2", "0").strip().lower() not in (
        "0",
        "false",
        "no",
    )
    lm_gs = int(os.environ.get("ASDSL_LMHEAD_GS", str(store.group_size)))
    # preq2 lm_head cache applies only to g32 layout; g128 quantizes fp16 at init.
    use_lmhead_preq2_cache = preq2_on and lm_gs == 32
    lm_head_meta: np.ndarray | None = None
    lm_head_quant: np.ndarray | None = None
    lm_head_fp16: np.ndarray | None = None
    if use_lmhead_preq2_cache:
        from asdsl.quantization.lmhead_preq2_cache import (
            lmhead_preq2_cache_path_for_store,
            save_lmhead_preq2_cache,
            try_restore_lmhead_preq2_cache,
        )

        cache_path = lmhead_preq2_cache_path_for_store(store)
        cached = try_restore_lmhead_preq2_cache(store, cache_path)
        if cached is not None:
            lm_head_meta, lm_head_quant = cached
        else:
            lm_head_fp16 = embed  # tied embedding; quantize at engine init once
    else:
        lm_head_fp16 = embed
    cos = store._cos_table
    sin = store._sin_table
    if cos is None or sin is None:
        store._build_rope_native_tables()
        cos = store._cos_table
        sin = store._sin_table

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
        layer_entry = {
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
        preq2_ready = getattr(store, "_preq2_built", False)
        if preq2_ready:
            for nm in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"):
                key = (i, nm)
                layer_entry[f"{nm}_meta"] = store._preq2_meta_np[key]
                layer_entry[f"{nm}_quant"] = store._preq2_quant_np[key]
        gs128 = getattr(store, "_preq_gs128_np", None) or {}
        for nm in ("gate_up_proj", "down_proj", "qkv_proj", "o_proj"):
            key = (i, nm)
            if key in gs128:
                layer_entry[f"{nm}_g128"] = gs128[key]
        layers_dict[i] = layer_entry

    _attach_q2_weights(layers_dict, store, NUM_LAYERS)

    q2_layers = sum(1 for i in range(NUM_LAYERS) if layers_dict[i].get("has_q2"))
    if q2_layers:
        print(f"  UnifiedEngine Q2 draft bank: {q2_layers}/{NUM_LAYERS} layers ({weight_src})")

    eng = nu.UnifiedEngine(
        cfg,
        embed,
        out_norm,
        lm_head_fp16,
        cos,
        sin,
        layers_dict,
        lm_head_meta if lm_head_meta is not None else None,
        lm_head_quant if lm_head_quant is not None else None,
    )

    if use_lmhead_preq2_cache and lm_head_meta is None and hasattr(eng, "export_lm_head_preq2"):
        from asdsl.quantization.lmhead_preq2_cache import (
            lmhead_preq2_cache_path_for_store,
            save_lmhead_preq2_cache,
        )

        meta, quant = eng.export_lm_head_preq2()
        save_lmhead_preq2_cache(store, lmhead_preq2_cache_path_for_store(store), meta, quant)

    return eng


def get_or_build_unified_engine(store: "WeightStore") -> Any:
    eng = getattr(store, "_unified_engine", None)
    if eng is None:
        use_q4km = os.environ.get("ASDSL_USE_Q4KM_GGUF", "0").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        n_km = len(getattr(store, "_q4km_weights", None) or {})
        q2 = bool(getattr(store, "_enable_qcsd", False))
        preq2_on = os.environ.get("ASDSL_PREQ2", "0").strip().lower() not in (
            "0",
            "false",
            "no",
        )
        preq2_built = getattr(store, "_preq2_built", False)
        if use_q4km and n_km:
            label = f"mixed preq + {n_km} Q4_K GGUF projections"
        elif preq2_on and preq2_built:
            label = "preq2+VNNI Q4_32"
        else:
            label = "preq Q4_32"
        if q2:
            label += " + Q2 draft"
        n_p2 = len(getattr(store, "_preq2_meta_np", None) or {})
        print(
            f"  Building UnifiedEngine ({label}, {n_p2} preq2 tensors, C++ forward) ...",
            flush=True,
        )
        eng = build_unified_engine(store)
        store._unified_engine = eng
    return eng


def reset_unified_session(store: "WeightStore") -> None:
    """Clear UnifiedEngine KV state (call once per generation / PPL window)."""
    if getattr(store, "_use_unified", False) and store.bits == 4:
        get_or_build_unified_engine(store).reset_session()


def unified_forward_token(
    store: "WeightStore",
    token_id: int,
    pos: int,
    *,
    need_logits: bool = True,
) -> np.ndarray | None:
    """Single decode step. When ``need_logits=False``, updates KV only (prefill body)."""
    eng = get_or_build_unified_engine(store)
    if need_logits:
        logits = eng.forward_token(int(token_id), int(pos))
        return np.asarray(logits, dtype=np.float32)
    eng.forward_token_prefill(int(token_id), int(pos))
    return None


def cpp_generate(
    store: "WeightStore",
    prompt_ids: list[int],
    max_new_tokens: int,
    stop_ids: list[int] | None = None,
) -> list[int]:
    """C++ decode loop — GIL released for full generation. Returns new token ids only."""
    import os

    eng = get_or_build_unified_engine(store)
    eng.reset_session()
    ignore_eos = os.environ.get("ASDSL_IGNORE_EOS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if ignore_eos:
        stop: list[int] = []
    else:
        stop = [int(s) for s in (stop_ids or [199999, 200020])]
    prompt = [int(t) for t in prompt_ids]
    all_tokens = eng.generate_with_stops(prompt, int(max_new_tokens), stop)
    return [int(t) for t in all_tokens[len(prompt) :]]


@dataclass
class PldResult:
    """Result of PLD generation via UnifiedEngine."""

    token_ids: list[int]
    prompt_len: int
    decode_tokens: int
    decode_s: float
    tokens_per_second: float


def greedy_generate(
    store: "WeightStore",
    prompt_ids: list[int],
    max_new_tokens: int,
) -> list[int]:
    """Serial greedy decode (reference for PLD losslessness)."""
    eng = get_or_build_unified_engine(store)
    eng.reset_session()
    p4 = _phi4_constants()
    eos = {199999, 200020}
    tokens = [int(t) for t in prompt_ids]
    prompt_len = len(prompt_ids)

    logits = np.zeros(p4.VOCAB, dtype=np.float32)
    for i, tid in enumerate(tokens):
        logits = np.asarray(eng.forward_token(tid, i), dtype=np.float32)

    pos = len(tokens)
    while len(tokens) - prompt_len < max_new_tokens:
        cur = int(logits.argmax())
        if cur in eos:
            break
        tokens.append(cur)
        logits = np.asarray(eng.forward_token(cur, pos), dtype=np.float32)
        pos += 1
    return tokens


def pld_generate(
    store: "WeightStore",
    prompt_ids: list[int],
    max_new_tokens: int,
    max_draft_k: int = 6,
) -> list[int]:
    """Lossless Prompt Lookup Decoding with batched verify."""
    from asdsl.speculative.pld import PLDConfig, PromptLookupDecoder

    eng = get_or_build_unified_engine(store)
    eng.reset_session()
    p4 = _phi4_constants()
    vocab = p4.VOCAB
    eos = {199999, 200020}
    decoder = PromptLookupDecoder(PLDConfig(max_draft_k=max_draft_k))
    tokens = [int(t) for t in prompt_ids]
    prompt_len = len(prompt_ids)

    logits = np.zeros(vocab, dtype=np.float32)
    for i, tid in enumerate(tokens):
        logits = np.asarray(eng.forward_token(tid, i), dtype=np.float32)

    pos = len(tokens)
    while len(tokens) - prompt_len < max_new_tokens:
        cur = int(logits.argmax())
        if cur in eos:
            break

        start_pos = pos
        draft = decoder.lookup(tokens)[:max_draft_k]
        if not draft:
            tokens.append(cur)
            logits = np.asarray(eng.forward_token(cur, start_pos), dtype=np.float32)
            pos = start_pos + 1
            continue

        n_draft = len(draft)
        verify_tokens = [cur] + draft[:-1]
        n_verify = len(verify_tokens)

        snapshot_kv(eng)
        vlog = np.asarray(
            eng.forward_batch_all_logits(verify_tokens, start_pos),
            dtype=np.float32,
        ).reshape(n_verify, vocab)
        restore_kv(eng)

        accepted = 0
        correction = None
        for i in range(n_draft):
            ref = int(vlog[i].argmax())
            if ref == draft[i]:
                accepted += 1
            else:
                correction = ref
                break

        if correction is not None:
            pending = [cur] + draft[:accepted] + [correction]
        else:
            pending = [cur] + draft[:accepted]

        commit = []
        for t in pending:
            if len(tokens) - prompt_len >= max_new_tokens:
                break
            commit.append(int(t))
            tokens.append(int(t))

        if not commit:
            break

        truncate_kv(eng, start_pos)
        for j, t in enumerate(commit):
            logits = np.asarray(
                eng.forward_token(int(t), start_pos + j), dtype=np.float32
            )
        pos = start_pos + len(commit)

        if len(tokens) - prompt_len >= max_new_tokens:
            break
        if any(int(t) in eos for t in commit):
            break

    return tokens


def pld_generate_timed(
    store: "WeightStore",
    prompt_ids: list[int],
    max_new_tokens: int,
    max_draft_k: int = 6,
) -> PldResult:
    """PLD generation with wall-clock timing."""
    t0 = time.perf_counter()
    out = pld_generate(store, prompt_ids, max_new_tokens, max_draft_k=max_draft_k)
    decode_s = time.perf_counter() - t0
    prompt_len = len(prompt_ids)
    decode_tokens = max(0, len(out) - prompt_len)
    tps = decode_tokens / decode_s if decode_s > 0 else 0.0
    return PldResult(
        token_ids=out,
        prompt_len=prompt_len,
        decode_tokens=decode_tokens,
        decode_s=float(decode_s),
        tokens_per_second=float(tps),
    )


def measure_verify_overhead(
    store: "WeightStore",
    *,
    k_values: tuple[int, ...] = (2, 4, 6),
    repeats: int = 5,
) -> dict[str, float]:
    """Ratio T_verify(K) / T_single for prefill+decode probe tokens."""
    eng = get_or_build_unified_engine(store)
    eng.reset_session()
    p4 = _phi4_constants()
    vocab = p4.VOCAB
    probe = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    for i, tid in enumerate(probe[:3]):
        eng.forward_token(int(tid), i)

    start_pos = 3
    single_ms: list[float] = []
    for _ in range(repeats):
        snapshot_kv(eng)
        t0 = time.perf_counter()
        eng.forward_token(int(probe[3]), start_pos)
        single_ms.append((time.perf_counter() - t0) * 1000.0)
        restore_kv(eng)

    t_single = float(np.median(single_ms))
    out: dict[str, float] = {"t_single_ms": round(t_single, 3)}
    for k in k_values:
        toks = [int(probe[3 + j]) for j in range(k)]
        batch_ms: list[float] = []
        for _ in range(repeats):
            snapshot_kv(eng)
            t0 = time.perf_counter()
            eng.forward_batch_all_logits(toks, start_pos)
            batch_ms.append((time.perf_counter() - t0) * 1000.0)
            restore_kv(eng)
        t_k = float(np.median(batch_ms))
        out[f"t_verify_k{k}_ms"] = round(t_k, 3)
        out[f"ratio_k{k}"] = round(t_k / t_single if t_single > 0 else 0.0, 3)
    out["vocab"] = float(vocab)
    return out


def _resolve_skip_mask(
    store: "WeightStore",
    skip_mask: np.ndarray | None,
    num_layers: int,
) -> np.ndarray:
    if skip_mask is not None:
        return np.asarray(skip_mask, dtype=bool).ravel()
    cached = getattr(store, "_ahsd_skip_mask", None)
    if cached is not None:
        return np.asarray(cached, dtype=bool).ravel()
    return default_ahsd_skip_mask(num_layers)


def ahsd_generate(
    store: "WeightStore",
    prompt_ids: list[int],
    max_new_tokens: int,
    draft_k: int = 1,
    skip_mask: np.ndarray | None = None,
    use_sdqs: bool = False,
) -> AhsdResult:
    """Run AHSD or SDQS generation through UnifiedEngine.

    When ``use_sdqs`` is True and the engine has Q2 weights, uses
    ``generate_sdqs_stats``; otherwise ``generate_ahsd_stats``.
    """
    p4 = _phi4_constants()
    num_layers = p4.NUM_LAYERS

    eng = get_or_build_unified_engine(store)
    eng.reset_session()

    mask = _resolve_skip_mask(store, skip_mask, num_layers)
    set_skip_mask(eng, mask)

    prompt = [int(t) for t in prompt_ids]
    t0 = time.perf_counter()
    try:
        if use_sdqs and getattr(store, "_enable_qcsd", False):
            out = eng.generate_sdqs_stats(prompt, int(max_new_tokens), int(draft_k))
        else:
            out = eng.generate_ahsd_stats(prompt, int(max_new_tokens), int(draft_k))
    finally:
        clear_skip_mask(eng)

    decode_s = time.perf_counter() - t0
    tokens = [int(t) for t in out["tokens"]]
    prompt_len = len(prompt)
    decode_tokens = max(0, len(tokens) - prompt_len)
    tps = decode_tokens / decode_s if decode_s > 0 else 0.0

    profile = os.environ.get("ASDSL_SPECULATIVE_PROFILE", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if profile:
        print(
            f"acceptance_rate={float(out['acceptance_rate']):.4f} "
            f"draft_tokens={int(out['draft_tokens'])} "
            f"verify_ms={float(out['verify_ms']):.2f} "
            f"draft_ms={float(out['draft_ms']):.2f} "
            f"speculative_cycles={int(out['speculative_cycles'])}",
            flush=True,
        )

    return AhsdResult(
        token_ids=tokens,
        acceptance_rate=float(out["acceptance_rate"]),
        draft_tokens=int(out["draft_tokens"]),
        accepted_tokens=int(out["accepted_tokens"]),
        speculative_cycles=int(out["speculative_cycles"]),
        draft_ms=float(out["draft_ms"]),
        verify_ms=float(out["verify_ms"]),
        prompt_len=prompt_len,
        decode_tokens=decode_tokens,
        decode_s=float(decode_s),
        tokens_per_second=float(tps),
    )
