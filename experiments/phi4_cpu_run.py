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
    """Set CPU threads for NumPy/BLAS/PyTorch. 0 = auto-detect."""
    if n <= 0:
        n = max(1, os.cpu_count() or 4)
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

# Target chunk size for streaming dequant+BLAS matvec.
# Each chunk must fit in CPU L2/L3 cache to avoid redundant DRAM traffic.
_TARGET_CHUNK_BYTES = 8 * 1024 * 1024  # 8 MB – fits L3 comfortably


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
        from asdsl.quantization.core import quantize_weights
        self._quantize = quantize_weights
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
        self.embed_f16: torch.Tensor | None = None        # (vocab, hidden) float16 — for token lookup
        self.lm_head: torch.Tensor | None = None           # same object as embed_f16, for LM head
        self.final_norm: torch.Tensor | None = None       # (hidden,)
        self._weight_cache: dict[tuple, torch.Tensor] = {}   # (layer, name) → float16 tensor (bits=16 only)
        # Quantized path (bits≤8): uint8 values + f16 scales/biases per group
        self._quant_u8: dict[tuple, torch.Tensor] = {}    # (layer, name) → uint8 (rows*cols,)
        self._quant_sc: dict[tuple, torch.Tensor] = {}    # (layer, name) → float16 scales (n_groups,)
        self._quant_bi: dict[tuple, torch.Tensor] = {}    # (layer, name) → float16 biases (n_groups,)
        self._quant_shapes: dict[tuple, tuple] = {}       # (layer, name) → (rows, cols)

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
        # Embedding: keep as float16 to save ~1.2 GB RAM.
        # Token lookup casts one row to float32 (12 KB, negligible).
        # LM head reads f16 in L2-sized chunks, halving DRAM traffic.
        print("  Caching embed float16 … ", end="", flush=True)
        self.embed_f16 = self.embed.to(torch.float16).clone()
        self.lm_head = self.embed_f16              # tied weight, float16
        self.embed = None                          # free bfloat16 copy (~1.2 GB)
        print("done")

        # Pre-allocate a flat float32 pool for chunked matvec.
        self._pool = torch.empty(_TARGET_CHUNK_BYTES // 4, dtype=torch.float32)

    # --- float16 path (bits=16) helpers --------------------------------

    def _matvec_f16(self, layer_idx: int, name: str, x: torch.Tensor) -> torch.Tensor:
        """Chunked f16→f32 matvec for unquantized weights."""
        w_f16 = self._weight_cache[(layer_idx, name)]
        rows, cols = w_f16.shape
        x_flat = x.view(-1)
        chunk_rows = max(1, _TARGET_CHUNK_BYTES // (cols * 4))
        result = torch.empty(rows, dtype=torch.float32)
        buf = self._pool[:chunk_rows * cols].view(chunk_rows, cols)
        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            n = end - start
            buf[:n].copy_(w_f16[start:end])
            torch.mv(buf[:n], x_flat, out=result[start:end])
        return result.unsqueeze(0)

    # --- quantized path (bits≤8) helpers --------------------------------

    def _matvec_quant(self, layer_idx: int, name: str, x: torch.Tensor) -> torch.Tensor:
        """Chunked uint8→dequant→matvec for quantized weights.

        Reads uint8 pre-unpacked values in L3-sized chunks, applies
        scale/bias in-place (no intermediate allocations) and does BLAS
        gemv from cache.  Keeps weights compressed in memory.
        """
        key = (layer_idx, name)
        u8 = self._quant_u8[key]
        sc = self._quant_sc[key]
        bi = self._quant_bi[key]
        rows, cols = self._quant_shapes[key]
        groups_per_row = cols // self.group_size
        x_flat = x.view(-1)
        chunk_rows = max(1, _TARGET_CHUNK_BYTES // (cols * 4))
        result = torch.empty(rows, dtype=torch.float32)
        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            n = end - start
            flat_len = n * cols
            # Flat view of pool — copy uint8→f32 in-place
            buf = self._pool[:flat_len]
            buf.copy_(u8[start * cols:end * cols])
            vals = buf.view(n, groups_per_row, self.group_size)
            gs = start * groups_per_row
            ge = end * groups_per_row
            # In-place dequantize:  val = val * scale + bias
            vals.mul_(sc[gs:ge].float().view(n, groups_per_row, 1))
            vals.add_(bi[gs:ge].float().view(n, groups_per_row, 1))
            torch.mv(vals.view(n, cols), x_flat, out=result[start:end])
        return result.unsqueeze(0)

    def matvec(self, layer_idx: int, name: str, x: torch.Tensor) -> torch.Tensor:
        """Bandwidth-efficient matrix-vector product: y = W @ x."""
        if self.bits == 16:
            return self._matvec_f16(layer_idx, name, x)
        return self._matvec_quant(layer_idx, name, x)

    def lm_head_matvec(self, hidden: torch.Tensor) -> torch.Tensor:
        """Compute logits = hidden @ lm_head.T with chunked f16 reads."""
        x = hidden.view(-1)
        lm = self.lm_head   # (vocab, hidden) float16
        rows, cols = lm.shape
        chunk_rows = max(1, _TARGET_CHUNK_BYTES // (cols * 4))
        result = torch.empty(rows, dtype=torch.float32)
        buf = self._pool[:chunk_rows * cols].view(chunk_rows, cols)
        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            n = end - start
            buf[:n].copy_(lm[start:end])
            torch.mv(buf[:n], x, out=result[start:end])
        return result

    def warm_cache(self) -> None:
        """Prepare weight cache for streaming inference.

        bits=16: float16 weight cache is already populated from load().
        bits≤8:  pre-unpack quantized weights to uint8 tensors with f16
                 scales/biases.  Keeps weights compressed — no f16 cache.
        """
        from asdsl.quantization.core import _unpack_bits
        total = NUM_LAYERS * 4
        if self.bits == 16:
            print(f"  Float16 weight cache already populated ({total} tensors)")
            print("  Inference: chunked f16 matvec")
        else:
            done = 0
            qmax = (1 << self.bits) - 1
            print(f"  Pre-unpacking {total} tensors to uint8 … ", end="", flush=True)
            for i in range(NUM_LAYERS):
                for name in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"):
                    qt = self.layers[i][name]
                    rows, cols = qt.shape
                    key = (i, name)
                    self._quant_shapes[key] = (rows, cols)

                    # Unpack to uint8 (1 byte per value vs 0.5 for 4-bit packed)
                    numel = rows * cols
                    unpacked = _unpack_bits(qt.data, qt.bits)[:numel]
                    self._quant_u8[key] = torch.from_numpy(unpacked.astype(np.uint8))

                    # Compute per-group scales and biases (bias = -zero * scale)
                    n_groups = numel // qt.group_size
                    sc_f16 = torch.from_numpy(qt.scales[:n_groups].copy()).to(torch.float16)
                    if qt.is_symmetric:
                        half_range = qmax / 2.0
                        bi_f16 = (-half_range * sc_f16.float()).half()
                    else:
                        zr = torch.from_numpy(qt.zeros[:n_groups].copy()).to(torch.float16)
                        bi_f16 = (-zr.float() * sc_f16.float()).half()
                    self._quant_sc[key] = sc_f16
                    self._quant_bi[key] = bi_f16

                    done += 1
            self.layers.clear()           # free packed QuantizedTensor data
            self._weight_cache.clear()    # ensure no stale f16 cache
            u8_bytes = sum(t.nbytes for t in self._quant_u8.values())
            sc_bytes = sum(t.nbytes for t in self._quant_sc.values())
            bi_bytes = sum(t.nbytes for t in self._quant_bi.values())
            total_mb = (u8_bytes + sc_bytes + bi_bytes) / 1e6
            print(f"done ({done}/{total}) | {total_mb:.0f} MB")
            print(f"  Inference: chunked uint8 dequant+matvec (in-place, no f16 cache)")

    def get_norm(self, layer_idx: int, name: str) -> torch.Tensor:
        return self.layer_norms[layer_idx][name]


# ---------------------------------------------------------------------------
# Per-layer KV history (simple list-based, one entry per token per layer)
# ---------------------------------------------------------------------------

class KVHistory:
    """
    Pre-allocated KV cache stored as contiguous torch tensors.

    Each layer gets a (max_seq, kv_heads, head_dim) buffer.
    `get()` returns zero-copy views — no np.stack / torch.from_numpy per call.
    """
    def __init__(self, max_seq: int = 2048):
        self.k_buf: dict[int, torch.Tensor] = {
            i: torch.zeros(max_seq, NUM_KV_HEADS, HEAD_DIM) for i in range(NUM_LAYERS)
        }
        self.v_buf: dict[int, torch.Tensor] = {
            i: torch.zeros(max_seq, NUM_KV_HEADS, HEAD_DIM) for i in range(NUM_LAYERS)
        }
        self._len: dict[int, int] = {i: 0 for i in range(NUM_LAYERS)}

    def append(self, layer: int, k_vec: torch.Tensor, v_vec: torch.Tensor) -> None:
        n = self._len[layer]
        self.k_buf[layer][n] = k_vec
        self.v_buf[layer][n] = v_vec
        self._len[layer] = n + 1

    def get(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return views into the pre-allocated buffers (zero-copy)."""
        n = self._len[layer]
        return self.k_buf[layer][:n], self.v_buf[layer][:n]

    def get_last_np(self, layer: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the most recently appended K/V as numpy (zero-copy view)."""
        n = self._len[layer] - 1
        return self.k_buf[layer][n].numpy(), self.v_buf[layer][n].numpy()

    @property
    def num_tokens(self) -> int:
        return self._len[0]


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
    """Single Phi-4 transformer layer. Updates kv_hist with the new K/V.

    Uses chunked matvec (streaming dequant for quantized, chunked f16 read
    for float16) to minimise DRAM bandwidth.  KV history is pre-allocated
    torch tensors — no per-token np.stack / allocations.
    """

    # — Self-attention —
    residual = hidden
    h = rms_norm(hidden, store.get_norm(layer_idx, "input_layernorm"))

    qkv = store.matvec(layer_idx, "qkv_proj", h)           # (1, qkv_dim)

    q = qkv[:, :Q_DIM].view(1, NUM_HEADS, HEAD_DIM)
    k = qkv[:, Q_DIM:Q_DIM + KV_DIM].view(1, NUM_KV_HEADS, HEAD_DIM)
    v = qkv[:, Q_DIM + KV_DIM:].view(1, NUM_KV_HEADS, HEAD_DIM)

    cos_pos = rope_cos[pos:pos + 1]
    sin_pos = rope_sin[pos:pos + 1]
    q = apply_rope(q, cos_pos, sin_pos)
    k = apply_rope(k, cos_pos, sin_pos)

    # Append torch tensors directly (no numpy conversion)
    kv_hist.append(layer_idx, k.squeeze(0), v.squeeze(0))

    # Retrieve views into pre-allocated buffers (zero-copy)
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
    attn_out = torch.nn.functional.scaled_dot_product_attention(
        q_attn, k_full, v_full,
    )  # (1, num_heads, 1, head_dim)
    attn_out = attn_out.permute(0, 2, 1, 3).reshape(1, Q_DIM)

    hidden = residual + store.matvec(layer_idx, "o_proj", attn_out)

    # — Feed-forward —
    residual = hidden
    h = rms_norm(hidden, store.get_norm(layer_idx, "post_attention_layernorm"))

    gu = store.matvec(layer_idx, "gate_up_proj", h)         # (1, 2*inter)
    h = silu(gu[:, :INTER]) * gu[:, INTER:]                 # (1, inter)

    hidden = residual + store.matvec(layer_idx, "down_proj", h)
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

    # Per-layer KV history (pre-allocated torch tensors)
    kv_hist = KVHistory(max_seq=max_seq)

    # ASDSL tracker — updated once per generated token for block-sparse analytics
    asdsl_tracker = ASDSLKVTracker()

    def run_forward(token_id: int, pos: int, need_logits: bool = True) -> torch.Tensor | None:
        """Full 32-layer forward pass for a single token at position pos.
        When need_logits=False (prefill body), skips the expensive LM-head matmul."""
        hidden = store.embed_f16[token_id].float().unsqueeze(0)

        k_new: list[np.ndarray] = []
        v_new: list[np.ndarray] = []

        for i in range(NUM_LAYERS):
            hidden = forward_layer(hidden, i, store, kv_hist, rope_cos, rope_sin, pos)
            k_np, v_np = kv_hist.get_last_np(i)
            k_new.append(k_np)
            v_new.append(v_np)

        # Feed this token's (all-layer) K/V into the ASDSL block-sparse tracker
        asdsl_tracker.record_token(k_new, v_new)

        if not need_logits:
            return None

        # Final norm + LM head (embed weights tied)
        hidden = rms_norm(hidden, store.final_norm)
        logits = store.lm_head_matvec(hidden)
        return logits

    # ------------------------------------------------------------------
    # Prefill: run every prompt token through the model sequentially.
    # We only keep the logits from the LAST token.
    # ------------------------------------------------------------------
    print("Prefill: ", end="", flush=True)
    t_prefill_start = time.perf_counter()

    with torch.inference_mode():
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
    with torch.inference_mode():
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
    quant_label = "float16 (no quant)" if store.bits == 16 else f"{store.bits}-bit ASDSL"
    print(f"Generated : {n_tokens} tokens  |  {tps:.2f} tok/s  |  decode {t_decode:.1f}s")
    print(f"ASDSL KV  : {kv_stats['tokens']} tokens tracked  "
          f"| {kv_stats['blocks_used']}/{kv_stats['blocks_capacity']} blocks  "
          f"| {kv_stats['memory_mb']:.1f} MB")
    print(f"Weights   : {quant_label}")
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
    max_seq = 4096
    kv_hist = KVHistory(max_seq=max_seq)
    asdsl_tracker = ASDSLKVTracker()
    rope_cos, rope_sin = build_rope_cache(max_seq, ROTARY_DIM)
    pos = 0  # global position counter

    def run_forward(token_id: int, need_logits: bool = True):
        nonlocal pos
        hidden = store.embed_f16[token_id].float().unsqueeze(0)
        k_new, v_new = [], []
        for i in range(NUM_LAYERS):
            hidden = forward_layer(hidden, i, store, kv_hist, rope_cos, rope_sin, pos)
            k_np, v_np = kv_hist.get_last_np(i)
            k_new.append(k_np)
            v_new.append(v_np)
        asdsl_tracker.record_token(k_new, v_new)
        pos += 1
        if not need_logits:
            return None
        hidden = rms_norm(hidden, store.final_norm)
        return store.lm_head_matvec(hidden)

    # Feed system prompt first
    system_tokens = tokenizer.encode(f"<|system|>\n{system_prompt}<|end|>\n")
    print(f"Feeding system prompt ({len(system_tokens)} tokens) … ", end="", flush=True)
    with torch.inference_mode():
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
        with torch.inference_mode():
            for idx, tid in enumerate(turn_ids):
                is_last = (idx == len(turn_ids) - 1)
                logits = run_forward(tid, need_logits=is_last)
        t_pre = time.perf_counter() - t_pre

        # Decode
        print(f"Assistant: ", end="", flush=True)
        generated = []
        t_dec = time.perf_counter()

        with torch.inference_mode():
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
    parser.add_argument("--threads", type=int, default=0,
                        help="CPU threads for BLAS/OMP (0=auto: all cores)")
    args = parser.parse_args()

    # Thread control — use all cores by default for memory-bound inference
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
    if args.bits == 16:
        print(f"  Memory: f16 weight cache (~6.4 GB) + embed_f16 (~1.2 GB) ≈ 7.6 GB")
    else:
        u8_mb = sum(t.nbytes for t in store._quant_u8.values()) / 1e6
        sc_mb = sum(t.nbytes for t in store._quant_sc.values()) / 1e6
        bi_mb = sum(t.nbytes for t in store._quant_bi.values()) / 1e6
        embed_mb = store.embed_f16.nbytes / 1e6
        total_mb = u8_mb + sc_mb + bi_mb + embed_mb
        print(f"  Memory: uint8 weights ({u8_mb:.0f} MB) + scales/biases ({sc_mb + bi_mb:.0f} MB)"
              f" + embed_f16 ({embed_mb:.0f} MB) ≈ {total_mb / 1e3:.1f} GB")

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
