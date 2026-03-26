"""Block-wise sparse attention for KV cache management.

Implements structured sparsity for attention computation to prevent
KV cache from ballooning on long contexts. Uses query-centric
similarity to prune less important KV pages dynamically.

Quantized KV storage:
  * **Q4** (default when ``quantize_kv``): packed nibbles + 64-dim block scales
    (~50% vs FP16, ~75% vs FP32 for payload).
  * **INT8**: per-row min/max quantization (legacy ``kv_bits=8``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# INT8 KV Quantization helpers
# ---------------------------------------------------------------------------

def quantize_kv_per_head(
    tensor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize a float32 tensor to INT8 with per-head scale/zero.

    Args:
        tensor: Float32 tensor of shape (..., head_dim).

    Returns:
        (int8_data, scales, zeros) where:
          - int8_data: shape same as tensor, dtype int8
          - scales: shape (..., 1), dtype float32
          - zeros: shape (..., 1), dtype float32
    """
    # Compute per-row (per-head) statistics
    leading_shape = tensor.shape[:-1]
    head_dim = tensor.shape[-1]
    flat = tensor.reshape(-1, head_dim)

    row_min = flat.min(axis=-1, keepdims=True)
    row_max = flat.max(axis=-1, keepdims=True)
    row_range = np.maximum(row_max - row_min, 1e-7)

    scales = row_range / 255.0
    zeros = row_min

    quantized = np.clip(
        np.round((flat - zeros) / scales), 0, 255
    ).astype(np.uint8).view(np.int8)

    return (
        quantized.reshape(tensor.shape),
        scales.reshape(leading_shape + (1,)).astype(np.float32),
        zeros.reshape(leading_shape + (1,)).astype(np.float32),
    )


def dequantize_kv_per_head(
    int8_data: np.ndarray,
    scales: np.ndarray,
    zeros: np.ndarray,
) -> np.ndarray:
    """Dequantize INT8 tensor back to float32.

    Args:
        int8_data: INT8 quantized data.
        scales: Per-head scale factors.
        zeros: Per-head zero points.

    Returns:
        Float32 tensor.
    """
    return int8_data.view(np.uint8).astype(np.float32) * scales + zeros


# Matches C++ KVCache::KV_QBLOCK / KVCacheQ4
KV_QBLOCK: int = 64


def quantize_kv_q4_blocks(
    tensor: np.ndarray,
    block_size: int = KV_QBLOCK,
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric Q4 per block along the last dim (same scheme as native KVCacheQ4).

    Returns:
        packed: uint8, shape (..., (head_dim + 1) // 2), 2 nibbles per byte.
        scales: float32, shape (..., n_blocks) where n_blocks = ceil(head_dim / block_size).
    """
    x = np.asarray(tensor, dtype=np.float32)
    *lead, head_dim = x.shape
    nb = (head_dim + block_size - 1) // block_size
    flat = x.reshape(-1, head_dim)
    nrows = flat.shape[0]
    scales = np.zeros((nrows, nb), dtype=np.float32)
    packed = np.zeros((nrows, (head_dim + 1) // 2), dtype=np.uint8)

    for r in range(nrows):
        for b in range(nb):
            off = b * block_size
            sl = flat[r, off : off + block_size]
            if sl.size == 0:
                continue
            amax = float(np.max(np.abs(sl)))
            amax = max(amax, 1e-12)
            sc = amax / 8.0
            scales[r, b] = sc
            inv = 1.0 / sc
            for i in range(min(block_size, head_dim - off)):
                qi = int(np.round(flat[r, off + i] * inv + 8.0))
                qi = max(0, min(15, qi))
                virt = off + i
                bi = virt // 2
                if virt % 2 == 0:
                    packed[r, bi] = (packed[r, bi] & 0xF0) | qi
                else:
                    packed[r, bi] = (packed[r, bi] & 0x0F) | (qi << 4)

    return packed.reshape(*lead, (head_dim + 1) // 2), scales.reshape(*lead, nb)


def dequantize_kv_q4_blocks(
    packed: np.ndarray,
    scales: np.ndarray,
    head_dim: int,
    block_size: int = KV_QBLOCK,
) -> np.ndarray:
    """Reconstruct float32 K or V from Q4 packed storage."""
    *lead, ph = packed.shape
    nb = (head_dim + block_size - 1) // block_size
    flat_p = packed.reshape(-1, ph)
    flat_s = scales.reshape(-1, nb)
    nrows = flat_p.shape[0]
    out = np.zeros((nrows, head_dim), dtype=np.float32)

    for r in range(nrows):
        for d in range(head_dim):
            b = d // block_size
            sc = flat_s[r, b]
            bi = d // 2
            q = int(flat_p[r, bi] & 0xF) if d % 2 == 0 else int((flat_p[r, bi] >> 4) & 0xF)
            out[r, d] = (float(q) - 8.0) * sc

    return out.reshape(*lead, head_dim)


@dataclass
class KVCacheConfig:
    """Configuration for the block-sparse KV cache."""

    num_layers: int = 1
    num_kv_heads: int = 8
    head_dim: int = 96
    max_blocks: int = 32
    block_size: int = 64
    quantize_kv: bool = True
    #: When ``quantize_kv`` is True: ``4`` = Q4 packed (default), ``8`` = INT8.
    kv_bits: int = 4
    max_context_length: int = 4096

    def memory_budget_bytes(self) -> int:
        """Upper bound on memory for the configured KV layout."""
        tokens = self.max_blocks * self.block_size
        cells = self.num_layers * tokens * self.num_kv_heads * self.head_dim
        if not self.quantize_kv:
            return int(2 * cells * 4)

        if self.kv_bits == 4:
            pack = (cells + 1) // 2
            nb = (self.head_dim + KV_QBLOCK - 1) // KV_QBLOCK
            scale_el = self.num_layers * tokens * self.num_kv_heads * nb
            return int(2 * pack + 2 * scale_el * 4)

        # INT8 path (+ per-head scale + zero ~ float32 each)
        dtype_bytes = 1
        scale_overhead = 2
        return (
            self.num_layers
            * self.num_kv_heads
            * self.head_dim
            * self.max_blocks
            * self.block_size
            * 2
            * dtype_bytes
            * scale_overhead
        )


@dataclass
class KVBlock:
    """A block of key-value cache entries."""

    block_id: int
    keys: np.ndarray
    values: np.ndarray
    importance_score: float = 0.0
    token_positions: list[int] = field(default_factory=list)
    is_pinned: bool = False


class BlockSparseKVCache:
    """Block-wise sparse KV cache with Q4 (default), INT8, or FP32 storage."""

    def __init__(self, config: KVCacheConfig):
        self.config = config
        self._max_tokens: int = config.max_blocks * config.block_size
        self._quantize = config.quantize_kv
        self._kv_bits = config.kv_bits if config.quantize_kv else 32

        if not self._quantize:
            self._storage = "fp32"
            self._keys: list[np.ndarray] = []
            self._values: list[np.ndarray] = []
        elif self._kv_bits == 4:
            self._storage = "q4"
            self._keys_q4: list[tuple[np.ndarray, np.ndarray]] = []
            self._values_q4: list[tuple[np.ndarray, np.ndarray]] = []
        elif self._kv_bits == 8:
            self._storage = "int8"
            self._keys_q: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            self._values_q: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        else:
            raise ValueError(f"Unsupported kv_bits={self._kv_bits} (use 4 or 8)")

        self._importance: list[float] = []

    @property
    def length(self) -> int:
        if self._storage == "fp32":
            return len(self._keys)
        if self._storage == "q4":
            return len(self._keys_q4)
        return len(self._keys_q)

    @property
    def memory_bytes(self) -> int:
        if self._storage == "fp32":
            if not self._keys:
                return 0
            return sum(k.nbytes + v.nbytes for k, v in zip(self._keys, self._values))
        if self._storage == "q4":
            if not self._keys_q4:
                return 0
            return sum(
                kp[0].nbytes + kp[1].nbytes + vp[0].nbytes + vp[1].nbytes
                for kp, vp in zip(self._keys_q4, self._values_q4)
            )
        if not self._keys_q:
            return 0
        return sum(
            kq[0].nbytes + kq[1].nbytes + kq[2].nbytes +
            vq[0].nbytes + vq[1].nbytes + vq[2].nbytes
            for kq, vq in zip(self._keys_q, self._values_q)
        )

    def _evict_if_full(self) -> None:
        if self.length >= self._max_tokens:
            evict_idx = int(np.argmin(self._importance))
            if self._storage == "fp32":
                del self._keys[evict_idx]
                del self._values[evict_idx]
            elif self._storage == "q4":
                del self._keys_q4[evict_idx]
                del self._values_q4[evict_idx]
            else:
                del self._keys_q[evict_idx]
                del self._values_q[evict_idx]
            del self._importance[evict_idx]

    def append(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        is_pivot: bool = False,
    ) -> None:
        """Append a single token's KV state for all layers."""
        self._evict_if_full()
        keys = keys.astype(np.float32, copy=False)
        values = values.astype(np.float32, copy=False)

        if self._storage == "fp32":
            self._keys.append(keys.copy())
            self._values.append(values.copy())
        elif self._storage == "q4":
            kp, ks = quantize_kv_q4_blocks(keys)
            vp, vs = quantize_kv_q4_blocks(values)
            self._keys_q4.append((kp, ks))
            self._values_q4.append((vp, vs))
        else:
            k_q, k_s, k_z = quantize_kv_per_head(keys)
            v_q, v_s, v_z = quantize_kv_per_head(values)
            self._keys_q.append((k_q, k_s, k_z))
            self._values_q.append((v_q, v_s, v_z))

        self._importance.append(float("inf") if is_pivot else 0.0)

    def get_attention_keys_values(
        self,
        layer_idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        hd = self.config.head_dim
        nh = self.config.num_kv_heads
        if self._storage == "fp32":
            if not self._keys:
                z = np.empty((0, nh, hd), dtype=np.float32)
                return z, z.copy()
            return (
                np.stack([k[layer_idx] for k in self._keys], axis=0),
                np.stack([v[layer_idx] for v in self._values], axis=0),
            )
        if self._storage == "q4":
            if not self._keys_q4:
                z = np.empty((0, nh, hd), dtype=np.float32)
                return z, z.copy()
            k_list = []
            v_list = []
            for kp, vp in zip(self._keys_q4, self._values_q4):
                k_list.append(dequantize_kv_q4_blocks(kp[0][layer_idx], kp[1][layer_idx], hd))
                v_list.append(dequantize_kv_q4_blocks(vp[0][layer_idx], vp[1][layer_idx], hd))
            return np.stack(k_list, axis=0), np.stack(v_list, axis=0)

        if not self._keys_q:
            z = np.empty((0, nh, hd), dtype=np.float32)
            return z, z.copy()
        k_list = []
        v_list = []
        for kq, vq in zip(self._keys_q, self._values_q):
            k_list.append(dequantize_kv_per_head(kq[0][layer_idx], kq[1][layer_idx], kq[2][layer_idx]))
            v_list.append(dequantize_kv_per_head(vq[0][layer_idx], vq[1][layer_idx], vq[2][layer_idx]))
        return np.stack(k_list, axis=0), np.stack(v_list, axis=0)

    def update_importance(
        self,
        layer_idx: int,
        query_state: np.ndarray,
    ) -> None:
        q = query_state.reshape(-1).astype(np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0:
            return
        hd = self.config.head_dim

        if self._storage == "fp32":
            rows = list(enumerate(self._keys))
        elif self._storage == "q4":
            rows = list(enumerate(self._keys_q4))
        else:
            rows = list(enumerate(self._keys_q))

        for i, row in rows:
            if self._importance[i] == float("inf"):
                continue
            if self._storage == "fp32":
                k_deq = row[layer_idx]
            elif self._storage == "q4":
                kp, ks = row
                k_deq = dequantize_kv_q4_blocks(kp[layer_idx], ks[layer_idx], hd)
            else:
                kq = row
                k_deq = dequantize_kv_per_head(kq[0][layer_idx], kq[1][layer_idx], kq[2][layer_idx])
            k_flat = k_deq.reshape(-1).astype(np.float32)
            k_norm = float(np.linalg.norm(k_flat))
            if k_norm == 0:
                continue
            score = float(np.dot(q, k_flat) / (q_norm * k_norm))
            self._importance[i] = max(self._importance[i], score)

    def clear(self) -> None:
        if self._storage == "fp32":
            self._keys.clear()
            self._values.clear()
        elif self._storage == "q4":
            self._keys_q4.clear()
            self._values_q4.clear()
        else:
            self._keys_q.clear()
            self._values_q.clear()
        self._importance.clear()
        logger.debug("KV cache cleared")


class ContiguousKVCache:
    """Pre-allocated contiguous KV: FP32, INT8, or Q4 packed + block scales."""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int,
        quantize: bool = True,
        kv_bits: int = 4,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.quantize = quantize
        self.kv_bits = kv_bits if quantize else 32
        hd = head_dim
        ph = (hd + 1) // 2
        nb = (hd + KV_QBLOCK - 1) // KV_QBLOCK

        if not quantize:
            self._mode = "fp32"
            self.k_cache = np.zeros(
                (num_layers, max_seq_len, num_heads, hd), dtype=np.float32
            )
            self.v_cache = np.zeros(
                (num_layers, max_seq_len, num_heads, hd), dtype=np.float32
            )
        elif self.kv_bits == 4:
            self._mode = "q4"
            self.k_pack = np.zeros(
                (num_layers, max_seq_len, num_heads, ph), dtype=np.uint8
            )
            self.v_pack = np.zeros(
                (num_layers, max_seq_len, num_heads, ph), dtype=np.uint8
            )
            self.k_scales_b = np.zeros(
                (num_layers, max_seq_len, num_heads, nb), dtype=np.float32
            )
            self.v_scales_b = np.zeros(
                (num_layers, max_seq_len, num_heads, nb), dtype=np.float32
            )
        elif self.kv_bits == 8:
            self._mode = "int8"
            self.k_cache = np.zeros(
                (num_layers, max_seq_len, num_heads, hd), dtype=np.int8
            )
            self.v_cache = np.zeros(
                (num_layers, max_seq_len, num_heads, hd), dtype=np.int8
            )
            self.k_scales = np.zeros(
                (num_layers, max_seq_len, num_heads, 1), dtype=np.float32
            )
            self.k_zeros = np.zeros(
                (num_layers, max_seq_len, num_heads, 1), dtype=np.float32
            )
            self.v_scales = np.zeros(
                (num_layers, max_seq_len, num_heads, 1), dtype=np.float32
            )
            self.v_zeros = np.zeros(
                (num_layers, max_seq_len, num_heads, 1), dtype=np.float32
            )
        else:
            raise ValueError("kv_bits must be 4 or 8 when quantize=True")

        self.seq_len = 0

    def append(self, k: np.ndarray, v: np.ndarray) -> None:
        batch_seq_len = k.shape[1]
        assert self.seq_len + batch_seq_len <= self.max_seq_len, "KV cache capacity exceeded"

        if self._mode == "fp32":
            self.k_cache[:, self.seq_len:self.seq_len + batch_seq_len, :, :] = k
            self.v_cache[:, self.seq_len:self.seq_len + batch_seq_len, :, :] = v
        elif self._mode == "q4":
            for t in range(batch_seq_len):
                pos = self.seq_len + t
                k_slice = k[:, t, :, :].astype(np.float32)
                v_slice = v[:, t, :, :].astype(np.float32)
                for li in range(self.num_layers):
                    kp, ks = quantize_kv_q4_blocks(k_slice[li])
                    vp, vs = quantize_kv_q4_blocks(v_slice[li])
                    self.k_pack[li, pos, :, :] = kp
                    self.v_pack[li, pos, :, :] = vp
                    self.k_scales_b[li, pos, :, :] = ks
                    self.v_scales_b[li, pos, :, :] = vs
        else:
            for t in range(batch_seq_len):
                pos = self.seq_len + t
                k_slice = k[:, t, :, :]
                v_slice = v[:, t, :, :]
                k_q, k_s, k_z = quantize_kv_per_head(k_slice.astype(np.float32))
                v_q, v_s, v_z = quantize_kv_per_head(v_slice.astype(np.float32))
                self.k_cache[:, pos, :, :] = k_q
                self.v_cache[:, pos, :, :] = v_q
                self.k_scales[:, pos, :, :] = k_s
                self.k_zeros[:, pos, :, :] = k_z
                self.v_scales[:, pos, :, :] = v_s
                self.v_zeros[:, pos, :, :] = v_z

        self.seq_len += batch_seq_len

    def get_keys_values(
        self, layer_idx: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.seq_len == 0:
            shape = (0, self.num_heads, self.head_dim)
            empty = np.empty(shape, dtype=np.float32)
            return empty, empty.copy()

        if self._mode == "fp32":
            return (
                self.k_cache[layer_idx, :self.seq_len, :, :].copy(),
                self.v_cache[layer_idx, :self.seq_len, :, :].copy(),
            )
        if self._mode == "q4":
            sl = self.seq_len
            hd = self.head_dim
            k_rows = [
                dequantize_kv_q4_blocks(
                    self.k_pack[layer_idx, ti], self.k_scales_b[layer_idx, ti], hd
                )
                for ti in range(sl)
            ]
            v_rows = [
                dequantize_kv_q4_blocks(
                    self.v_pack[layer_idx, ti], self.v_scales_b[layer_idx, ti], hd
                )
                for ti in range(sl)
            ]
            return np.stack(k_rows, axis=0), np.stack(v_rows, axis=0)

        k_int8 = self.k_cache[layer_idx, :self.seq_len, :, :]
        v_int8 = self.v_cache[layer_idx, :self.seq_len, :, :]
        k_fp32 = dequantize_kv_per_head(
            k_int8, self.k_scales[layer_idx, :self.seq_len, :, :],
            self.k_zeros[layer_idx, :self.seq_len, :, :],
        )
        v_fp32 = dequantize_kv_per_head(
            v_int8, self.v_scales[layer_idx, :self.seq_len, :, :],
            self.v_zeros[layer_idx, :self.seq_len, :, :],
        )
        return k_fp32, v_fp32

    def clear(self) -> None:
        self.seq_len = 0
        logger.debug("ContiguousKVCache cleared")

    @property
    def memory_bytes(self) -> int:
        if self._mode == "fp32":
            return self.k_cache.nbytes + self.v_cache.nbytes
        if self._mode == "q4":
            return (
                self.k_pack.nbytes + self.v_pack.nbytes +
                self.k_scales_b.nbytes + self.v_scales_b.nbytes
            )
        return (
            self.k_cache.nbytes + self.v_cache.nbytes +
            self.k_scales.nbytes + self.k_zeros.nbytes +
            self.v_scales.nbytes + self.v_zeros.nbytes
        )
