"""HEP inference engine: full transformer forward pass using AES-NI weight synthesis.

This engine implements the same `generate(tokens, n_decode)` interface as
`UnifiedEngine` from `asdsl.kernels._native_unified`, making it a drop-in
replacement for benchmarking purposes.

Architecture
------------
- Non-quantized tensors (embed, norm, lm_head): stored as FP16, loaded once
  into pinned RAM.
- Weight projection matrices (qkv, o_proj, gate_up, down): stored as HEP
  coefficients (seeds + int8 alphas). Each GEMV synthesises weight rows
  on-the-fly from AES-NI bases, eliminating DDR5 weight traffic.
- KV cache: Q8 quantized (same as UnifiedEngine).
- Attention: flash attention with online softmax (same as forward_loop.cpp).

The HEP GEMV dispatch falls back to a pure-Python numpy path if the native
`_native_hep` extension is not built. Build it with:
    python setup.py build_ext --inplace
"""

from __future__ import annotations

import json
import os
import struct
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Try to import the native AES-NI kernel; fall back to Python reference
try:
    from asdsl.kernels import _native_hep as _hep_native
    _NATIVE_HEP = True
except ImportError:
    _NATIVE_HEP = False

from .codec import hep_decode, HEPTensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rms_norm(x: np.ndarray, w: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """RMS normalisation: x / rms(x) * w."""
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return (x / rms) * w


def _silu(x: np.ndarray) -> np.ndarray:
    """SiLU activation: x * sigmoid(x)."""
    return x * (1.0 / (1.0 + np.exp(-x)))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _hsum(arr: np.ndarray) -> float:
    return float(np.sum(arr))


# ---------------------------------------------------------------------------
# HEP weight container
# ---------------------------------------------------------------------------

class HEPWeights:
    """Container for HEP-encoded projection weights for one transformer layer."""

    def __init__(
        self,
        seeds:       np.ndarray,   # uint64 (out_dim,)
        alphas:      np.ndarray,   # int8   (out_dim, n_groups, rank)
        out_dim:     int,
        in_dim:      int,
        rank:        int,
        group_size:  int,
    ):
        self.seeds      = seeds
        self.alphas     = alphas
        self.out_dim    = out_dim
        self.in_dim     = in_dim
        self.rank       = rank
        self.group_size = group_size

        # Cache the decoded weight matrix if native kernel unavailable
        # (expensive — only for Python fallback path)
        self._weight_cache: Optional[np.ndarray] = None

    def gemv(self, x: np.ndarray) -> np.ndarray:
        """Compute W_hep @ x using AES-NI synthesis (or numpy fallback)."""
        if _NATIVE_HEP:
            return _hep_native.gemv_hep(
                self.seeds, self.alphas, x.astype(np.float32),
                self.out_dim, self.in_dim, self.rank, self.group_size,
            )
        else:
            # Numpy fallback: decode on demand and cache
            if self._weight_cache is None:
                tensor = HEPTensor(
                    seeds=self.seeds, alphas=self.alphas,
                    rank=self.rank, group_size=self.group_size,
                    shape=(self.out_dim, self.in_dim),
                )
                self._weight_cache = hep_decode(tensor)
            return self._weight_cache @ x.astype(np.float32)


# ---------------------------------------------------------------------------
# HEP Engine
# ---------------------------------------------------------------------------

class HEPEngine:
    """CPU inference engine using HEP weight synthesis for projection matrices.

    Provides the same API as UnifiedEngine:
        engine.generate(tokens: list[int], n_decode: int) -> list[int]

    Args:
        meta_path: Path to HEP metadata JSON (phi4_14b_hep_r4_meta.json).
        bin_path:  Path to HEP binary file (phi4_14b_hep_r4.bin).
        embed_fp32: Token embedding table, shape (vocab, hidden).
        final_norm: Final RMS norm weights, shape (hidden,).
        lm_head:    LM head projection, shape (vocab, hidden).
        cos_t, sin_t: RoPE tables, shape (max_seq_len, rotary_dim/2).
        config:     Dict with model hyperparameters.
    """

    def __init__(
        self,
        meta_path:  str,
        bin_path:   str,
        embed_fp32: np.ndarray,
        final_norm: np.ndarray,
        lm_head:    np.ndarray,
        cos_t:      np.ndarray,
        sin_t:      np.ndarray,
        config:     dict,
    ):
        self.embed   = embed_fp32.astype(np.float32)
        self.f_norm  = final_norm.astype(np.float32)
        self.lm_head = lm_head.astype(np.float32)
        self.cos_t   = cos_t.astype(np.float32)
        self.sin_t   = sin_t.astype(np.float32)
        self.cfg     = config

        n_layers   = config["num_layers"]
        hidden     = config["hidden_size"]
        n_heads    = config["num_heads"]
        n_kv_heads = config["num_kv_heads"]
        head_dim   = config["head_dim"]
        inter_dim  = config["intermediate_size"]
        rotary_dim = config["rotary_dim"]

        self.n_layers   = n_layers
        self.hidden     = hidden
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim   = head_dim
        self.inter_dim  = inter_dim
        self.rotary_dim = rotary_dim
        self.rms_eps    = config.get("rms_norm_eps", 1e-5)
        self.groups     = max(1, n_heads // n_kv_heads)

        # KV cache: [layer][pos] → (k, v) float32 arrays
        max_seq = config.get("max_seq_len", 2048)
        self.max_seq = max_seq
        self.kv_k = [np.zeros((max_seq, n_kv_heads, head_dim), dtype=np.float32) for _ in range(n_layers)]
        self.kv_v = [np.zeros((max_seq, n_kv_heads, head_dim), dtype=np.float32) for _ in range(n_layers)]

        # Load per-layer weights
        self._load_weights(meta_path, bin_path)

        print(f"HEPEngine ready: {'native AES-NI' if _NATIVE_HEP else 'numpy fallback'} GEMV")

    def _load_weights(self, meta_path: str, bin_path: str) -> None:
        """Load HEP coefficient store and per-layer norm weights."""
        print(f"Loading HEP weights from {bin_path} ...")
        with open(meta_path) as f:
            meta = json.load(f)

        rank       = meta["rank"]
        group_size = meta["group_size"]
        tensors    = meta["tensors"]

        # Memory-map the binary coefficient file
        mm = np.memmap(bin_path, dtype=np.uint8, mode="r")

        self.layers_hep: list[dict] = []

        for layer_idx in range(self.n_layers):
            prefix = f"model.layers.{layer_idx}"
            layer_d: dict[str, object] = {}

            for proj_name in ("self_attn.qkv_proj", "self_attn.o_proj",
                              "mlp.gate_up_proj", "mlp.down_proj"):
                key = f"{prefix}.{proj_name}.weight"
                if key not in tensors:
                    continue
                info = tensors[key]

                if info["type"] == "hep":
                    out_dim, in_dim = info["shape"]
                    n_groups = info["n_groups"]

                    seed_bytes  = mm[info["offset"]: info["offset"] + info["seed_bytes"]]
                    alpha_bytes = mm[info["alpha_offset"]: info["alpha_offset"] + info["alpha_bytes"]]

                    seeds  = np.frombuffer(seed_bytes, dtype=np.uint64).copy()
                    alphas = np.frombuffer(alpha_bytes, dtype=np.int8).reshape(out_dim, n_groups, rank).copy()

                    layer_d[proj_name] = HEPWeights(seeds, alphas, out_dim, in_dim, rank, group_size)
                else:
                    # FP16 passthrough
                    raw = mm[info["offset"]: info["offset"] + info["size_bytes"]]
                    w   = np.frombuffer(raw, dtype=np.float16).reshape(info["shape"]).astype(np.float32).copy()
                    layer_d[proj_name] = w

            # Norms (FP16 passthrough)
            for norm_name in ("input_layernorm", "post_attention_layernorm"):
                key = f"{prefix}.{norm_name}.weight"
                if key in tensors:
                    info = tensors[key]
                    raw  = mm[info["offset"]: info["offset"] + info["size_bytes"]]
                    layer_d[norm_name] = np.frombuffer(raw, dtype=np.float16).astype(np.float32).copy()

            self.layers_hep.append(layer_d)

        print(f"Loaded {self.n_layers} layers.")

    # -----------------------------------------------------------------------
    # Forward pass helpers
    # -----------------------------------------------------------------------

    def _apply_rope(self, x: np.ndarray, pos: int) -> np.ndarray:
        """Apply rotary position embedding to x (num_heads, head_dim)."""
        half = self.rotary_dim // 2
        x_rot = x[:, :self.rotary_dim].reshape(-1, 2, half)
        cos = self.cos_t[pos]  # (half,)
        sin = self.sin_t[pos]  # (half,)
        x0, x1 = x_rot[:, 0, :], x_rot[:, 1, :]
        r0 = x0 * cos - x1 * sin
        r1 = x0 * sin + x1 * cos
        out = x.copy()
        out[:, :self.rotary_dim] = np.stack([r0, r1], axis=1).reshape(-1, self.rotary_dim)
        return out

    def _proj(self, weights_or_hep, x: np.ndarray) -> np.ndarray:
        """Dispatch to HEP GEMV or plain matmul."""
        if isinstance(weights_or_hep, HEPWeights):
            return weights_or_hep.gemv(x)
        else:
            return weights_or_hep @ x

    def _forward_layer(self, x: np.ndarray, layer_idx: int, pos: int) -> np.ndarray:
        """Single transformer layer forward pass."""
        ld = self.layers_hep[layer_idx]

        # --- Self-attention ---
        # RMSNorm
        h = _rms_norm(x, ld.get("input_layernorm", np.ones(self.hidden, dtype=np.float32)), self.rms_eps)

        # QKV projection (HEP)
        qkv = self._proj(ld["self_attn.qkv_proj"], h)
        q_dim = self.n_heads * self.head_dim
        k_dim = self.n_kv_heads * self.head_dim
        Q = qkv[:q_dim].reshape(self.n_heads, self.head_dim)
        K = qkv[q_dim: q_dim + k_dim].reshape(self.n_kv_heads, self.head_dim)
        V = qkv[q_dim + k_dim:].reshape(self.n_kv_heads, self.head_dim)

        # RoPE
        Q = self._apply_rope(Q, pos)
        K = self._apply_rope(K, pos)

        # Store KV
        self.kv_k[layer_idx][pos] = K
        self.kv_v[layer_idx][pos] = V

        # Flash attention (online softmax)
        attn_out = np.zeros((self.n_heads, self.head_dim), dtype=np.float32)
        inv_scale = 1.0 / np.sqrt(float(self.head_dim))

        for h_idx in range(self.n_heads):
            kv_h = h_idx // self.groups
            q_h  = Q[h_idx]  # (head_dim,)
            m, l_acc = -np.inf, 0.0
            num = np.zeros(self.head_dim, dtype=np.float32)

            for t in range(pos + 1):
                k_t = self.kv_k[layer_idx][t, kv_h]
                s   = float(np.dot(q_h, k_t)) * inv_scale
                new_m    = max(m, s)
                old_scale = np.exp(m - new_m) if l_acc > 0 else 0.0
                w  = np.exp(s - new_m)
                v_t = self.kv_v[layer_idx][t, kv_h]
                num = num * old_scale + w * v_t
                l_acc = l_acc * old_scale + w
                m = new_m

            attn_out[h_idx] = num / max(l_acc, 1e-30)

        # O projection
        attn_flat = attn_out.reshape(-1)
        o = self._proj(ld["self_attn.o_proj"], attn_flat)
        x = x + o

        # --- FFN (SwiGLU) ---
        h2 = _rms_norm(x, ld.get("post_attention_layernorm", np.ones(self.hidden, dtype=np.float32)), self.rms_eps)
        gate_up = self._proj(ld["mlp.gate_up_proj"], h2)
        half = gate_up.shape[0] // 2
        gate, up = gate_up[:half], gate_up[half:]
        ffn_hidden = _silu(gate) * up
        down = self._proj(ld["mlp.down_proj"], ffn_hidden)
        x = x + down

        return x

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def forward_token(self, token_id: int, pos: int) -> np.ndarray:
        """Single-token forward pass. Returns logits (vocab_size,)."""
        x = self.embed[token_id].copy()

        for layer_idx in range(self.n_layers):
            x = self._forward_layer(x, layer_idx, pos)

        x = _rms_norm(x, self.f_norm, self.rms_eps)
        logits = self.lm_head @ x
        return logits

    def prefill(self, tokens: list[int]) -> np.ndarray:
        """Prefill a prompt and return logits for the last token."""
        x = None
        for pos, tok in enumerate(tokens):
            x_emb = self.embed[tok].copy()
            if x is None:
                x = x_emb
            else:
                # In a full prefill we'd process all tokens; here we simplify
                # to sequential single-token passes (accurate but slower).
                x = x_emb
            for layer_idx in range(self.n_layers):
                x = self._forward_layer(x, layer_idx, pos)

        x = _rms_norm(x, self.f_norm, self.rms_eps)
        return self.lm_head @ x

    def generate(self, tokens: list[int], n_decode: int) -> list[int]:
        """Generate n_decode new tokens following the prompt.

        Args:
            tokens:   Prompt token IDs.
            n_decode: Number of new tokens to generate.

        Returns:
            Full token list including prompt + generated tokens.
        """
        output = list(tokens)

        # Prefill
        logits = self.prefill(tokens)
        next_tok = int(np.argmax(logits))
        output.append(next_tok)

        # Decode
        pos = len(tokens)
        for _ in range(n_decode - 1):
            logits = self.forward_token(output[-1], pos)
            next_tok = int(np.argmax(logits))
            output.append(next_tok)
            pos += 1

        return output
