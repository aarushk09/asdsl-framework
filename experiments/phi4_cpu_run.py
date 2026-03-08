"""
Phi-4 Multimodal Instruct – CPU inference through the ASDSL pipeline.

What this does:
  - Loads Phi-4 backbone weights from the local safetensors shards
  - Quantizes every projection weight to 4-bit with ASDSL
  - Runs a proper Phi-4 text forward pass (RMSNorm, RoPE, GQA, SiLU MLP)
  - Uses ASDSL's BlockSparseKVCache alongside inference for tracking
  - Generates tokens greedily and streams them to the terminal

Speed note: this is a Python reference implementation using PyTorch float32
matmul on CPU. On a modern desktop, expect ~0.1-0.5 tokens/second. The ASDSL
LUT kernel path (sub-byte matmul without dequantize) would require the native
C++/SIMD backend to be compiled – this script demonstrates correctness.

Usage:
  python experiments/phi4_cpu_run.py
  python experiments/phi4_cpu_run.py --prompt "Explain gravity in one sentence"
  python experiments/phi4_cpu_run.py --max-new-tokens 60 --bits 4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Suppress TensorFlow / JAX import attempts in transformers (they're incompatible
# with NumPy 2.x on this machine and aren't needed for text-only inference).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer


def set_thread_count(n: int) -> None:
    """Limit CPU threads for NumPy/BLAS/PyTorch to *n* cores."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n)
    torch.set_num_threads(n)


ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models" / "phi4-multimodal-instruct"
INDEX_FILE = MODEL_DIR / "model.safetensors.index.json"

# ---------------------------------------------------------------------------
# Architecture constants (from microsoft/Phi-4-multimodal-instruct config)
# ---------------------------------------------------------------------------
HIDDEN = 3072
NUM_LAYERS = 32
NUM_HEADS = 24
NUM_KV_HEADS = 8
HEAD_DIM = HIDDEN // NUM_HEADS          # 128
Q_DIM = NUM_HEADS * HEAD_DIM            # 3072
KV_DIM = NUM_KV_HEADS * HEAD_DIM        # 1024
QKV_DIM = Q_DIM + 2 * KV_DIM           # 5120
INTER = 8192                            # gate/up each; combined weight is 2*INTER
VOCAB = 200064
RMS_EPS = 1e-5
ROPE_THETA = 10000.0
# Phi-4 uses partial RoPE: only 75% of head_dim is rotated (config: partial_rotary_factor=0.75)
ROTARY_DIM = int(0.75 * HEAD_DIM)      # 96 — only these dims get RoPE applied

# Special token IDs
# 200020 = <|end|> (end of turn),  199999 = <|endoftext|> / </s> (end of text)
# 200019 = <|assistant|> (start of assistant turn — NOT an EOS token)
EOS_TOKEN_IDS = {200020, 199999}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Root-mean-square layer normalisation."""
    rms = x.pow(2).mean(-1, keepdim=True).add(RMS_EPS).sqrt()
    return (x / rms) * weight


def build_rope_cache(seq_len: int, head_dim: int, theta: float = ROPE_THETA,
                     dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute RoPE cos/sin tables up to seq_len."""
    pos = torch.arange(seq_len, dtype=torch.float32)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = torch.outer(pos, freqs)          # (seq_len, head_dim/2)
    cos = torch.cos(angles).to(dtype)
    sin = torch.sin(angles).to(dtype)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Partial rotate-half RoPE.

    x: (seq, heads, head_dim)  — full head vector
    cos/sin: (seq, ROTARY_DIM//2)  — tables for the rotated portion only

    Phi-4 sets partial_rotary_factor=0.75, so only the first ROTARY_DIM=96 dims
    of each head are rotated; the remaining 32 dims pass through unchanged.
    """
    d = cos.shape[-1]              # = ROTARY_DIM // 2 = 48
    rotary_dim = 2 * d             # = ROTARY_DIM = 96
    x_rot  = x[..., :rotary_dim]   # (seq, heads, 96) — will be rotated
    x_pass = x[..., rotary_dim:]   # (seq, heads, 32) — untouched
    x1 = x_rot[..., :d]            # first half:  dims 0..47
    x2 = x_rot[..., d:]            # second half: dims 48..95
    c  = cos.unsqueeze(-2)         # (seq, 1, 48) — broadcasts over heads
    s  = sin.unsqueeze(-2)
    rotated = torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
    return torch.cat([rotated, x_pass], dim=-1)


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


# ---------------------------------------------------------------------------
# Weight loading & quantization
# ---------------------------------------------------------------------------

class WeightStore:
    """
    Loads backbone weights from safetensors shards, quantizes them to N-bit
    with ASDSL, and stores the compressed tensors in a nested dict.

    Memory layout (4-bit, group_size=128):
      - ~50 MB per layer (vs ~402 MB float32)
      - ~1.7 GB total backbone (vs ~6.4 GB float32)
      - Embedding kept as bfloat16 torch tensor (~1.2 GB)
    """

    def __init__(self, bits: int = 4, group_size: int | None = None):
        from asdsl.quantization.core import quantize_weights, dequantize_weights
        self._quantize = quantize_weights
        self._dequantize = dequantize_weights
        self.bits = bits
        # Smart defaults: smaller groups + asymmetric + MSE-optimal clipping for low bits
        if group_size is None:
            if bits <= 3:
                self.group_size = 16  # Very small groups needed for 3-bit fidelity
            elif bits <= 4:
                self.group_size = 32  # Optimal balance for 4-bit
            else:
                self.group_size = 128
        else:
            self.group_size = group_size
        self._symmetric = bits > 4
        self._optimize_clips = bits <= 4

        self.layers: dict[int, dict[str, object]] = {}   # layer_idx → {name: QuantizedTensor}
        self.layer_norms: dict[int, dict[str, torch.Tensor]] = {}  # layer_idx → {name: tensor}
        self.embed: torch.Tensor | None = None            # (vocab, hidden) bfloat16, freed after caching
        self.embed_f32: torch.Tensor | None = None        # (vocab, hidden) float32 — for token lookup
        self.lm_head: torch.Tensor | None = None           # same object as embed_f32, for LM head
        self.final_norm: torch.Tensor | None = None       # (hidden,)
        self._weight_cache: dict[tuple, torch.Tensor] = {}   # (layer, name) → float16 tensor
        self._f32_bufs: dict[str, torch.Tensor] | None = None  # reusable float32 scratch buffers

    # ------------------------------------------------------------------
    def load(self) -> None:
        idx = json.loads(INDEX_FILE.read_text())["weight_map"]

        # Which shard holds which key
        shard_keys: dict[str, list[str]] = {}
        needed = set()
        needed.add("model.embed_tokens.weight")
        needed.add("model.norm.weight")
        for i in range(NUM_LAYERS):
            needed.add(f"model.layers.{i}.input_layernorm.weight")
            needed.add(f"model.layers.{i}.post_attention_layernorm.weight")
            needed.add(f"model.layers.{i}.self_attn.qkv_proj.base_layer.weight")
            needed.add(f"model.layers.{i}.self_attn.o_proj.base_layer.weight")
            needed.add(f"model.layers.{i}.mlp.gate_up_proj.base_layer.weight")
            needed.add(f"model.layers.{i}.mlp.down_proj.base_layer.weight")

        for k in needed:
            if k in idx:
                shard = idx[k]
                shard_keys.setdefault(shard, []).append(k)

        total_proj = NUM_LAYERS * 4
        done = 0

        if self.bits == 16:
            print(f"  Loading {total_proj} projection weights directly as float16 (no quantization) …")
        else:
            print(f"  Loading & quantizing {total_proj} projection weights to {self.bits}-bit …")
        print(f"  (embedding + norms loaded as bfloat16)")

        for shard_name, keys in sorted(shard_keys.items()):
            shard_path = MODEL_DIR / shard_name
            with safe_open(str(shard_path), framework="pt") as f:
                for key in keys:
                    tensor = f.get_tensor(key)   # bfloat16

                    if key == "model.embed_tokens.weight":
                        self.embed = tensor          # keep bfloat16
                        continue
                    if key == "model.norm.weight":
                        self.final_norm = tensor.to(torch.float32)
                        continue

                    parts = key.split(".")
                    layer_idx = int(parts[2])

                    if "layernorm" in key:
                        self.layer_norms.setdefault(layer_idx, {})[parts[-2]] = \
                            tensor.to(torch.float32)
                        continue

                    # Derive friendly key like "qkv_proj" / "o_proj" etc.
                    if "qkv_proj" in key:
                        friendly = "qkv_proj"
                    elif "o_proj" in key:
                        friendly = "o_proj"
                    elif "gate_up_proj" in key:
                        friendly = "gate_up_proj"
                    elif "down_proj" in key:
                        friendly = "down_proj"
                    else:
                        friendly = key.split(".")[-1]

                    if self.bits == 16:
                        # Skip ASDSL quantization — store directly in float16 cache.
                        # All forward-pass matmuls use float16→float32 scratch buffers
                        # anyway, so bits=16 gives best quality at no extra memory cost.
                        self._weight_cache[(layer_idx, friendly)] = tensor.to(torch.float16)
                    else:
                        # ASDSL N-bit quantization
                        w_f32 = tensor.to(torch.float32).numpy()
                        qt = self._quantize(w_f32, bits=self.bits,
                                            group_size=self.group_size,
                                            symmetric=self._symmetric,
                                            optimize_clips=self._optimize_clips)
                        self.layers.setdefault(layer_idx, {})[friendly] = qt

                    done += 1
                    if done % 16 == 0 or done == total_proj:
                        pct = done / total_proj * 100
                        print(f"    {done}/{total_proj} ({pct:.0f}%)  ", end="\r", flush=True)

        print(f"    {total_proj}/{total_proj} (100%)  done.               ")
        # Two float32 views of the embedding
        print("  Caching embed float32 views … ", end="", flush=True)
        embed32 = self.embed.to(torch.float32).clone()   # aligned PT allocation
        self.embed_f32 = embed32                          # (vocab, hidden) for lookup
        self.lm_head = embed32                            # same tensor, used as h @ embed.t()
        self.embed = None                                 # free bfloat16 copy (~1.2 GB)
        print("done")

    def get_weight(self, layer_idx: int, name: str) -> torch.Tensor:
        """Return the cached float16 weight tensor (out, in). Zero allocation."""
        cache_key = (layer_idx, name)
        if cache_key not in self._weight_cache:
            qt = self.layers[layer_idx][name]
            w_f32 = self._dequantize(qt)          # numpy float32, shape (out, in)
            self._weight_cache[cache_key] = torch.from_numpy(w_f32).to(torch.float16)
        return self._weight_cache[cache_key]     # float16 (out, in)

    def get_weight_f32(self, layer_idx: int, name: str) -> torch.Tensor:
        """Return a float32 view of the weight for BLAS matmul.
        Copies float16 cache → pre-allocated float32 buffer in-place.
        The returned tensor is valid until the NEXT call with the same name.
        This keeps weight memory as float16 (~6.4 GB) while giving float32 matmul
        precision — the only extra cost is a float16→float32 copy of the weight."""
        w_f16 = self.get_weight(layer_idx, name)
        buf = self._f32_bufs[name]
        buf.copy_(w_f16)   # in-place: float16 → float32, no allocation
        return buf         # (out, in) float32, contiguous, reused each call

    def warm_cache(self) -> None:
        """Populate float16 weight cache, allocate float32 scratch buffers,
        then free quantized and raw data.

        When bits=16: weights are already in the cache from load().
        When bits=2/3/4/8: dequantize ASDSL quantized tensors → float16.

        Memory layout after this call:
          float16 weight cache : ~6.4 GB   (128 tensors, always float16 regardless of bits)
          float32 buffers × 4  : ~0.4 GB   (reused in-place, never re-allocated)
          embed_f32            : ~2.4 GB
          Total                : ~9.2 GB   -- fits in 16 GB RAM with headroom
        """
        total = NUM_LAYERS * 4
        if self.bits == 16:
            print(f"  Float16 weight cache already populated ({total} tensors)")
        else:
            done = 0
            print(f"  Warming weight cache ({total} tensors) … ", end="", flush=True)
            for i in range(NUM_LAYERS):
                for name in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"):
                    _ = self.get_weight(i, name)
                    done += 1
            self.layers.clear()   # free quantized data
            print(f"done ({done}/{total})  |  {self.bits}-bit buffers freed")

        # Pre-allocate float32 scratch buffers (one per projection shape).
        # Reused in-place via get_weight_f32() — zero per-token allocations.
        print("  Allocating float32 scratch buffers (4 × ~100 MB) … ", end="", flush=True)
        self._f32_bufs = {
            "qkv_proj":    torch.empty(QKV_DIM, HIDDEN, dtype=torch.float32),
            "o_proj":      torch.empty(HIDDEN,   HIDDEN, dtype=torch.float32),
            "gate_up_proj": torch.empty(2 * INTER, HIDDEN, dtype=torch.float32),
            "down_proj":   torch.empty(HIDDEN,   INTER,  dtype=torch.float32),
        }
        print("done")

    def get_norm(self, layer_idx: int, name: str) -> torch.Tensor:
        return self.layer_norms[layer_idx][name]


# ---------------------------------------------------------------------------
# Per-layer KV history (simple list-based, one entry per token per layer)
# ---------------------------------------------------------------------------

class KVHistory:
    """
    Stores key/value tensors per layer across all generated positions.

    Layout: self.k[layer] = list of shape-(kv_heads, head_dim) numpy arrays.
    After each full forward pass, get_k(layer) returns a (seq_len, kv_heads, head_dim)
    tensor including the current token's contribution.
    """
    def __init__(self):
        self.k: dict[int, list[np.ndarray]] = {i: [] for i in range(NUM_LAYERS)}
        self.v: dict[int, list[np.ndarray]] = {i: [] for i in range(NUM_LAYERS)}

    def append(self, layer: int, k_vec: np.ndarray, v_vec: np.ndarray) -> None:
        self.k[layer].append(k_vec)
        self.v[layer].append(v_vec)

    def get(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        k = torch.from_numpy(np.stack(self.k[layer]))   # (seq, kv_heads, head_dim)
        v = torch.from_numpy(np.stack(self.v[layer]))
        return k, v

    @property
    def num_tokens(self) -> int:
        return len(self.k[0])


class ASDSLKVTracker:
    """
    Feeds the same KV data into ASDSL's BlockSparseKVCache for block-sparse
    tracking, eviction analytics, and memory budget calculations.
    Updated once per token (after all layers have been processed).
    """
    def __init__(self):
        from asdsl.inference.kv_cache import BlockSparseKVCache, KVCacheConfig
        cfg = KVCacheConfig(
            num_layers=NUM_LAYERS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            max_blocks=256,
            block_size=16,
        )
        self._cache = BlockSparseKVCache(cfg)

    def record_token(self, k_per_layer: list[np.ndarray],
                     v_per_layer: list[np.ndarray]) -> None:
        """k_per_layer[i] shape: (kv_heads, head_dim) for layer i."""
        k_all = np.stack(k_per_layer)   # (num_layers, kv_heads, head_dim)
        v_all = np.stack(v_per_layer)
        self._cache.append(k_all, v_all)

    def stats(self) -> dict:
        n = self._cache.length
        block_size = self._cache.config.block_size
        max_blocks = self._cache.config.max_blocks
        used_blocks = math.ceil(n / block_size) if n > 0 else 0
        bytes_used = (n * NUM_KV_HEADS * HEAD_DIM * 4 * 2 * NUM_LAYERS)  # float32 kv
        return {
            "tokens": n,
            "blocks_used": used_blocks,
            "blocks_capacity": max_blocks,
            "memory_mb": bytes_used / 1e6,
        }


# ---------------------------------------------------------------------------
# Transformer forward pass
# ---------------------------------------------------------------------------

def forward_layer(
    hidden: torch.Tensor,          # (1, hidden)
    layer_idx: int,
    store: WeightStore,
    kv_hist: KVHistory,            # mutable: this call appends to it
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    pos: int,
) -> torch.Tensor:
    """Single Phi-4 transformer layer. Updates kv_hist with the new K/V."""

    # — Self-attention —
    residual = hidden
    h = rms_norm(hidden, store.get_norm(layer_idx, "input_layernorm"))

    # get_weight_f32 copies float16 cache → pre-allocated float32 buf in-place.
    # No allocation per call; BLAS uses float32 for correct accumulation.
    w_qkv = store.get_weight_f32(layer_idx, "qkv_proj")   # (qkv_dim, hidden) f32 buf
    qkv = h @ w_qkv.t()                                    # (1, qkv_dim)

    q = qkv[:, :Q_DIM].view(1, NUM_HEADS, HEAD_DIM)
    k = qkv[:, Q_DIM:Q_DIM + KV_DIM].view(1, NUM_KV_HEADS, HEAD_DIM)
    v = qkv[:, Q_DIM + KV_DIM:].view(1, NUM_KV_HEADS, HEAD_DIM)

    cos_pos = rope_cos[pos:pos + 1]
    sin_pos = rope_sin[pos:pos + 1]
    q = apply_rope(q, cos_pos, sin_pos)
    k = apply_rope(k, cos_pos, sin_pos)

    # Add this token's K/V to the running history for this layer
    k_np = k.squeeze(0).detach().numpy().astype(np.float32)
    v_np = v.squeeze(0).detach().numpy().astype(np.float32)
    kv_hist.append(layer_idx, k_np, v_np)

    # Retrieve full history (includes current token just appended)
    k_hist, v_hist = kv_hist.get(layer_idx)   # (seq, kv_heads, head_dim)
    seq_len = k_hist.shape[0]

    # GQA: expand KV heads  →  (1, num_heads, seq, head_dim)
    expand = NUM_HEADS // NUM_KV_HEADS
    k_full = (k_hist.unsqueeze(2)
                    .expand(-1, -1, expand, -1)
                    .reshape(seq_len, NUM_HEADS, HEAD_DIM)
                    .permute(1, 0, 2)
                    .unsqueeze(0))
    v_full = (v_hist.unsqueeze(2)
                    .expand(-1, -1, expand, -1)
                    .reshape(seq_len, NUM_HEADS, HEAD_DIM)
                    .permute(1, 0, 2)
                    .unsqueeze(0))

    q_attn = q.unsqueeze(2)   # (1, num_heads, 1, head_dim)
    scale = HEAD_DIM ** -0.5
    scores = torch.matmul(q_attn, k_full.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    attn_out = torch.matmul(attn, v_full).permute(0, 2, 1, 3).reshape(1, Q_DIM)

    w_o = store.get_weight_f32(layer_idx, "o_proj")    # (hidden, hidden) f32 buf
    hidden = residual + attn_out @ w_o.t()

    # — Feed-forward —
    residual = hidden
    h = rms_norm(hidden, store.get_norm(layer_idx, "post_attention_layernorm"))

    w_gu = store.get_weight_f32(layer_idx, "gate_up_proj")     # (2*inter, hidden) f32 buf
    gu = h @ w_gu.t()                                           # (1, 2*inter)
    h = silu(gu[:, :INTER]) * gu[:, INTER:]               # (1, inter)

    w_d = store.get_weight_f32(layer_idx, "down_proj")         # (hidden, inter) f32 buf
    hidden = residual + h @ w_d.t()
    return hidden


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(
    prompt: str,
    store: WeightStore,
    tokenizer,
    max_new_tokens: int = 50,
    system_prompt: str = "You are a helpful AI assistant.",
) -> str:
    print("\n" + "=" * 66)
    print("ASDSL × Phi-4 — CPU Inference")
    print("=" * 66)
    print(f"Prompt : {prompt!r}")
    print("-" * 66)

    # Format prompt using the tokenizer's built-in chat template.
    # Phi-4 was fine-tuned with a system turn; including one markedly
    # improves instruction-following quality.
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )

    # Pre-compute RoPE tables (generous max length).
    # Pass ROTARY_DIM (96) — only the rotated portion of each head needs tables.
    max_seq = len(input_ids) + max_new_tokens + 64
    rope_cos, rope_sin = build_rope_cache(max_seq, ROTARY_DIM)

    # Per-layer KV history (used during forward pass)
    kv_hist = KVHistory()

    # ASDSL tracker — updated once per generated token for block-sparse analytics
    asdsl_tracker = ASDSLKVTracker()

    def run_forward(token_id: int, pos: int, need_logits: bool = True) -> torch.Tensor | None:
        """Full 32-layer forward pass for a single token at position pos.
        When need_logits=False (prefill body), skips the expensive LM-head matmul."""
        hidden = store.embed_f32[token_id].unsqueeze(0)

        k_new: list[np.ndarray] = []
        v_new: list[np.ndarray] = []

        for i in range(NUM_LAYERS):
            hidden = forward_layer(hidden, i, store, kv_hist, rope_cos, rope_sin, pos)
            k_new.append(kv_hist.k[i][-1])
            v_new.append(kv_hist.v[i][-1])

        # Feed this token's (all-layer) K/V into the ASDSL block-sparse tracker
        asdsl_tracker.record_token(k_new, v_new)

        if not need_logits:
            return None

        # Final norm + LM head (embed weights tied)
        hidden = rms_norm(hidden, store.final_norm)
        logits = (hidden @ store.lm_head.t()).squeeze(0)
        return logits

    # ------------------------------------------------------------------
    # Prefill: run every prompt token through the model sequentially.
    # We only keep the logits from the LAST token.
    # ------------------------------------------------------------------
    print("Prefill: ", end="", flush=True)
    t_prefill_start = time.perf_counter()

    logits = None
    for pos, tid in enumerate(input_ids):
        is_last = (pos == len(input_ids) - 1)
        logits = run_forward(tid, pos, need_logits=is_last)

    t_prefill = time.perf_counter() - t_prefill_start
    print(f"done ({len(input_ids)} tokens in {t_prefill:.1f}s)")

    # ------------------------------------------------------------------
    # Decode loop
    # ------------------------------------------------------------------
    print("\nAssistant: ", end="", flush=True)
    generated: list[int] = []
    t_decode_start = time.perf_counter()

    pos = len(input_ids)
    for _step in range(max_new_tokens):
        next_token = int(logits.argmax())
        generated.append(next_token)

        tok_text = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens([next_token])
        )
        print(tok_text, end="", flush=True)

        if next_token in EOS_TOKEN_IDS:
            break

        logits = run_forward(next_token, pos)
        pos += 1

    t_decode = time.perf_counter() - t_decode_start
    n_tokens = len(generated)
    tps = n_tokens / t_decode if t_decode > 0 else 0

    response_text = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(generated)
    )

    kv_stats = asdsl_tracker.stats()
    backbone_gb = (
        NUM_LAYERS
        * (QKV_DIM * HIDDEN + HIDDEN * HIDDEN + 2 * INTER * HIDDEN + HIDDEN * INTER)
        * 2   # float16 = 2 bytes per element; inference always uses float16 cache
        / 1e9
    )
    quant_label = "float16 (no quant)" if store.bits == 16 else f"{store.bits}-bit ASDSL"
    print(f"Generated : {n_tokens} tokens  |  {tps:.2f} tok/s  |  decode {t_decode:.1f}s")
    print(f"ASDSL KV  : {kv_stats['tokens']} tokens tracked  "
          f"| {kv_stats['blocks_used']}/{kv_stats['blocks_capacity']} blocks  "
          f"| {kv_stats['memory_mb']:.1f} MB")
    print(f"Weights   : {quant_label}  |  backbone f16 cache ≈ {backbone_gb:.2f} GB")
    print("=" * 66)

    return response_text


# ---------------------------------------------------------------------------
# Interactive chat session
# ---------------------------------------------------------------------------

def chat(
    store: WeightStore,
    tokenizer,
    max_new_tokens: int = 200,
    system_prompt: str = "You are a helpful AI assistant.",
) -> None:
    """Interactive multi-turn chat loop.

    The KV cache (kv_hist) and position counter are kept alive across turns,
    so the model only processes NEW tokens each round — previous context is
    already materialised in the K/V tensors.
    """
    print("\n" + "=" * 66)
    print("ASDSL × Phi-4 — Interactive Chat  (type 'quit' to exit)")
    print("=" * 66)

    # Persistent state across all turns
    kv_hist = KVHistory()
    asdsl_tracker = ASDSLKVTracker()
    max_seq = 4096
    rope_cos, rope_sin = build_rope_cache(max_seq, ROTARY_DIM)
    pos = 0  # global position counter

    def run_forward(token_id: int, need_logits: bool = True):
        nonlocal pos
        hidden = store.embed_f32[token_id].unsqueeze(0)
        k_new, v_new = [], []
        for i in range(NUM_LAYERS):
            hidden = forward_layer(hidden, i, store, kv_hist, rope_cos, rope_sin, pos)
            k_new.append(kv_hist.k[i][-1])
            v_new.append(kv_hist.v[i][-1])
        asdsl_tracker.record_token(k_new, v_new)
        pos += 1
        if not need_logits:
            return None
        hidden = rms_norm(hidden, store.final_norm)
        return (hidden @ store.lm_head.t()).squeeze(0)

    # Feed system prompt first
    system_tokens = tokenizer.encode(f"<|system|>\n{system_prompt}<|end|>\n")
    print(f"Feeding system prompt ({len(system_tokens)} tokens) … ", end="", flush=True)
    for tid in system_tokens:
        run_forward(tid, need_logits=False)
    print("done\n")

    turn = 0
    while True:
        turn += 1
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Session ended]")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("[Goodbye]")
            break
        if not user_input:
            continue

        # Tokenize this user turn (without re-feeding history)
        turn_text = f"<|user|>\n{user_input}<|end|>\n<|assistant|>\n"
        turn_ids = tokenizer.encode(turn_text)

        # Prefill new tokens (skip LM head for all but the last)
        t_pre = time.perf_counter()
        logits = None
        for idx, tid in enumerate(turn_ids):
            is_last = (idx == len(turn_ids) - 1)
            logits = run_forward(tid, need_logits=is_last)
        t_pre = time.perf_counter() - t_pre

        # Decode
        print(f"Assistant: ", end="", flush=True)
        generated = []
        t_dec = time.perf_counter()

        for _ in range(max_new_tokens):
            next_token = int(logits.argmax())
            generated.append(next_token)
            tok_text = tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens([next_token])
            )
            print(tok_text, end="", flush=True)
            if next_token in EOS_TOKEN_IDS:
                break
            logits = run_forward(next_token, need_logits=True)

        t_dec = time.perf_counter() - t_dec
        n = len(generated)
        tps = n / t_dec if t_dec > 0 else 0
        kv_stats = asdsl_tracker.stats()
        print(f"\n  [{n} tok | {tps:.2f} tok/s | KV: {kv_stats['tokens']} tokens "
              f"| {kv_stats['blocks_used']}/{kv_stats['blocks_capacity']} blocks]\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phi-4 CPU inference via ASDSL")
    parser.add_argument("--prompt", default="What is 2+2?",
                        help="Single-turn prompt (ignored when --chat is used)")
    parser.add_argument("--chat", action="store_true",
                        help="Start an interactive multi-turn chat session")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--bits", type=int, default=16, choices=[2, 3, 4, 8, 16],
                        help="Weight precision: 16=float16 (best quality, default), "
                             "8/4/3/2=ASDSL N-bit quantization (demo, lower quality)")
    parser.add_argument("--group-size", type=int, default=0,
                        help="Quantization group size (0=auto: 32 for ≤4-bit, 128 for 8-bit)")
    parser.add_argument("--threads", type=int, default=4,
                        help="CPU threads for BLAS/OMP (default: 4 for efficiency)")
    args = parser.parse_args()

    # Thread control — fewer cores = less power, often same throughput for memory-bound ops
    set_thread_count(args.threads)

    # Verify model is present
    if not INDEX_FILE.exists():
        print("ERROR: Model index not found at", INDEX_FILE)
        print("Run the download step first: python experiments/phi4_integration.py")
        sys.exit(1)

    print("=" * 66)
    print("ASDSL Phi-4 CPU Inference Setup")
    print("=" * 66)

    # ── Tokenizer ──────────────────────────────────────────────────────────
    print("Loading tokenizer …", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/Phi-4-multimodal-instruct",
        trust_remote_code=True,
    )
    print(f"  Vocabulary size: {tokenizer.vocab_size:,}")

    # ── Weights ────────────────────────────────────────────────────────────
    if args.bits == 16:
        print(f"Loading weights as float16 (no quantization) …")
    else:
        print(f"\nLoading weights (ASDSL {args.bits}-bit quantization) …")
    t0 = time.perf_counter()
    store = WeightStore(bits=args.bits,
                        group_size=args.group_size if args.group_size > 0 else None)
    store.load()
    t_load = time.perf_counter() - t0
    verb = "Load" if args.bits == 16 else "Load + quantize"
    print(f"  {verb} complete in {t_load / 60:.1f} minutes")
    if args.bits != 16:
        print(f"  Layers ready: {len(store.layers)}/32  "
              f"| Norms ready: {len(store.layer_norms)}/32")
    store.warm_cache()
    print(f"  Memory: f16 weight cache (~6.4 GB) + f32 scratch (~0.4 GB) + embed_f32 (~2.4 GB) ≈ 9.2 GB")

    # ── Chat or single-turn generate ───────────────────────────────────────
    if args.chat:
        chat(
            store=store,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        generate(
            prompt=args.prompt,
            store=store,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
        )


if __name__ == "__main__":
    main()
