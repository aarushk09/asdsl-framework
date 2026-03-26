"""Phase 6: Q4 KV cache memory, accuracy, and GQA-native attention."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from asdsl.inference.kv_cache import (
    KVCacheConfig,
    BlockSparseKVCache,
    ContiguousKVCache,
    dequantize_kv_q4_blocks,
    quantize_kv_q4_blocks,
    KV_QBLOCK,
)


def _fp32_kv_bytes(num_layers: int, seq: int, num_kv_heads: int, head_dim: int) -> int:
    cells = num_layers * seq * num_kv_heads * head_dim
    return 2 * cells * 4


def _q4_kv_bytes(num_layers: int, seq: int, num_kv_heads: int, head_dim: int) -> int:
    cells = num_layers * seq * num_kv_heads * head_dim
    pack = (cells + 1) // 2
    nb = (head_dim + KV_QBLOCK - 1) // KV_QBLOCK
    scale_bytes = num_layers * seq * num_kv_heads * nb * 4 * 2 * 2  # K+V, float32
    return 2 * pack + scale_bytes


def test_memory_footprint_q4_vs_fp32_8k_context() -> None:
    """Q4 KV payload + scales uses far less memory than FP32 for large contexts."""
    nl, seq, nkv, hd = 2, 8192, 8, 128
    fp32 = _fp32_kv_bytes(nl, seq, nkv, hd)
    q4 = _q4_kv_bytes(nl, seq, nkv, hd)
    assert q4 < fp32 * 0.35, (
        f"Q4 footprint should be well below FP32 (incl. scales); got q4={q4} fp32={fp32}"
    )
    ratio = q4 / fp32
    assert ratio < 0.5, f"expect <50% of FP32 total bytes; ratio={ratio:.3f}"

    cfg_fp = KVCacheConfig(
        num_layers=nl,
        num_kv_heads=nkv,
        head_dim=hd,
        max_blocks=max(1, seq // 64),
        block_size=64,
        quantize_kv=False,
    )
    cfg_q4 = KVCacheConfig(
        num_layers=nl,
        num_kv_heads=nkv,
        head_dim=hd,
        max_blocks=max(1, seq // 64),
        block_size=64,
        quantize_kv=True,
        kv_bits=4,
    )
    assert cfg_q4.memory_budget_bytes() < cfg_fp.memory_budget_bytes() * 0.5


def test_contiguous_cache_fp32_vs_q4_outputs() -> None:
    """Round-trip Q4 storage stays close to FP32 references for attention inputs."""
    rng = np.random.default_rng(0)
    nl, nh, hd, sl = 1, 4, 64, 12
    k = rng.standard_normal((nl, sl, nh, hd)).astype(np.float32)
    v = rng.standard_normal((nl, sl, nh, hd)).astype(np.float32)

    c_fp = ContiguousKVCache(nl, nh, hd, max_seq_len=32, quantize=False)
    c_q4 = ContiguousKVCache(nl, nh, hd, max_seq_len=32, quantize=True, kv_bits=4)
    c_fp.append(k, v)
    c_q4.append(k, v)

    kf, vf = c_fp.get_keys_values(0)
    kq, vq = c_q4.get_keys_values(0)
    # Expect agreement with a Python Q4 round-trip (same as native block scheme).
    for t in range(sl):
        ke = k[:, t, :, :]
        ve = v[:, t, :, :]
        pk, sk = quantize_kv_q4_blocks(ke)
        pv, sv = quantize_kv_q4_blocks(ve)
        kt = dequantize_kv_q4_blocks(pk, sk, hd)
        vt = dequantize_kv_q4_blocks(pv, sv, hd)
        assert_allclose(kq[t], kt.reshape(nh, hd), rtol=1e-5, atol=1e-5)
        assert_allclose(vq[t], vt.reshape(nh, hd), rtol=1e-5, atol=1e-5)
    assert c_q4.memory_bytes < c_fp.memory_bytes * 0.55


def test_block_sparse_q4_append_and_layer_slice() -> None:
    cfg = KVCacheConfig(
        num_layers=2,
        num_kv_heads=4,
        head_dim=32,
        max_blocks=4,
        block_size=8,
        quantize_kv=True,
        kv_bits=4,
    )
    cache = BlockSparseKVCache(cfg)
    rng = np.random.default_rng(1)
    for _ in range(5):
        keys = rng.standard_normal((2, 4, 32)).astype(np.float32)
        vals = rng.standard_normal((2, 4, 32)).astype(np.float32)
        cache.append(keys, vals)
    k0, v0 = cache.get_attention_keys_values(0)
    assert k0.shape == (5, 4, 32)
    assert v0.shape == (5, 4, 32)


def _numpy_gqa_attention(
    q: np.ndarray,
    k_rows: list[np.ndarray],
    v_rows: list[np.ndarray],
    head_dim: int,
) -> np.ndarray:
    """Reference softmax attention; k_rows[p] shape (num_kv_heads, head_dim)."""
    num_heads = q.shape[0]
    num_kv = k_rows[0].shape[0]
    g = num_heads // num_kv
    scale = 1.0 / np.sqrt(head_dim, dtype=np.float32)
    out = np.zeros_like(q)
    for h in range(num_heads):
        kv_h = h // g
        logits = np.array(
            [np.dot(q[h], k_rows[p][kv_h]) * scale for p in range(len(k_rows))],
            dtype=np.float32,
        )
        logits -= np.max(logits)
        w = np.exp(logits)
        w /= np.sum(w) + 1e-30
        acc = np.zeros(head_dim, dtype=np.float32)
        for p in range(len(v_rows)):
            acc += w[p] * v_rows[p][kv_h]
        out[h] = acc
    return out


def test_native_attention_q4_gqa_40_10_matches_reference() -> None:
    pytest.importorskip("asdsl.kernels._native_forward")
    from asdsl.kernels import _native_forward as nf

    num_heads = 40
    num_kv_heads = 10
    assert num_heads % num_kv_heads == 0
    head_dim = 32
    layers = 1
    max_seq = 64
    rng = np.random.default_rng(42)

    cache_q4 = nf.KVCacheQ4(layers, max_seq, num_kv_heads, head_dim)
    k_hist: list[np.ndarray] = []
    v_hist: list[np.ndarray] = []

    last_out_ref = None
    last_out_q4 = None
    for seq_pos in range(0, 9):
        q = rng.standard_normal((num_heads, head_dim)).astype(np.float32)
        k_new = rng.standard_normal((num_kv_heads, head_dim)).astype(np.float32)
        v_new = rng.standard_normal((num_kv_heads, head_dim)).astype(np.float32)
        pk, sk = quantize_kv_q4_blocks(k_new)
        pv, sv = quantize_kv_q4_blocks(v_new)
        k_hist.append(dequantize_kv_q4_blocks(pk, sk, head_dim))
        v_hist.append(dequantize_kv_q4_blocks(pv, sv, head_dim))

        out_q4 = nf.compute_attention_q4(
            q, k_new, v_new, 0, seq_pos, num_heads, cache_q4
        )
        last_out_q4 = np.asarray(out_q4)
        last_out_ref = _numpy_gqa_attention(q, k_hist, v_hist, head_dim)

        assert last_out_q4.shape == (num_heads, head_dim)
        assert_allclose(last_out_q4, last_out_ref, rtol=0.02, atol=0.02)

    assert last_out_ref is not None


def test_native_q4_matches_python_reference_small() -> None:
    """Native Q4 attention matches NumPy reference when K/V history uses Q4 dequant."""
    pytest.importorskip("asdsl.kernels._native_forward")
    from asdsl.kernels import _native_forward as nf

    num_heads = 16
    num_kv_heads = 4
    head_dim = 64
    layers = 1
    max_seq = 32
    rng = np.random.default_rng(7)

    k4 = nf.KVCacheQ4(layers, max_seq, num_kv_heads, head_dim)
    k_hist: list[np.ndarray] = []
    v_hist: list[np.ndarray] = []

    for seq_pos in range(4):
        q = rng.standard_normal((num_heads, head_dim)).astype(np.float32)
        kn = rng.standard_normal((num_kv_heads, head_dim)).astype(np.float32)
        vn = rng.standard_normal((num_kv_heads, head_dim)).astype(np.float32)
        pk, sk = quantize_kv_q4_blocks(kn)
        pv, sv = quantize_kv_q4_blocks(vn)
        k_hist.append(dequantize_kv_q4_blocks(pk, sk, head_dim))
        v_hist.append(dequantize_kv_q4_blocks(pv, sv, head_dim))
        o4 = nf.compute_attention_q4(q, kn, vn, 0, seq_pos, num_heads, k4)
        ref = _numpy_gqa_attention(q, k_hist, v_hist, head_dim)
        assert_allclose(np.asarray(o4), ref, rtol=0.02, atol=0.02)


def test_quantize_q4_roundtrip_small() -> None:
    x = np.linspace(-1.0, 1.0, 128).astype(np.float32).reshape(2, 64)
    p, s = quantize_kv_q4_blocks(x)
    y = dequantize_kv_q4_blocks(p, s, 64)
    assert_allclose(y, x, rtol=0.2, atol=0.06)
