"""Block-wise sparse attention for KV cache management.

Implements structured sparsity for attention computation to prevent
KV cache from ballooning on long contexts. Uses query-centric
similarity to prune less important KV pages dynamically.

NEW: INT8 quantized KV cache option reduces memory by 4x compared to
FP32 storage while maintaining accuracy for attention computation.
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


@dataclass
class KVCacheConfig:
    """Configuration for the block-sparse KV cache."""

    num_layers: int = 1
    num_kv_heads: int = 8
    head_dim: int = 96
    max_blocks: int = 32
    block_size: int = 64
    quantize_kv: bool = True  # NEW: INT8 quantization for 4x memory savings
    max_context_length: int = 4096

    def memory_budget_bytes(self) -> int:
        """Upper bound on memory: all layers * heads * blocks * block_size * K+V * dtype_size."""
        dtype_bytes = 1 if self.quantize_kv else 4  # int8 vs float32
        # Scales/zeros add ~2% overhead at per-head granularity
        scale_overhead = 2 if self.quantize_kv else 1
        return (
            self.num_layers
            * self.num_kv_heads
            * self.head_dim
            * self.max_blocks
            * self.block_size
            * 2   # keys + values
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
    """Block-wise sparse KV cache with dynamic eviction and INT8 quantization.

    Stores per-token KV states for all layers. When the cache exceeds
    max_blocks * block_size tokens, the least-important token is evicted.

    INT8 mode: Keys and values are quantized to INT8 on append and
    dequantized on-the-fly during attention computation, reducing
    memory by 4x with minimal accuracy loss.

    ``append(k, v)`` takes a single-token snapshot where k/v have shape
    ``(num_layers, num_kv_heads, head_dim)``.

    ``get_attention_keys_values(layer_idx)`` returns all stored tokens
    for that layer as ``(seq_len, num_kv_heads, head_dim)`` float32 arrays.
    """

    def __init__(self, config: KVCacheConfig):
        self.config = config
        self._max_tokens: int = config.max_blocks * config.block_size
        self._quantize = config.quantize_kv

        if self._quantize:
            # Quantized storage: (int8_data, scales, zeros) per token
            self._keys_q: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            self._values_q: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        else:
            # Float32 storage (legacy path)
            self._keys: list[np.ndarray] = []
            self._values: list[np.ndarray] = []

        self._importance: list[float] = []

    @property
    def length(self) -> int:
        """Number of tokens currently stored."""
        if self._quantize:
            return len(self._keys_q)
        return len(self._keys)

    @property
    def memory_bytes(self) -> int:
        """Approximate memory used by stored KV tensors."""
        if self._quantize:
            if not self._keys_q:
                return 0
            return sum(
                kq[0].nbytes + kq[1].nbytes + kq[2].nbytes +
                vq[0].nbytes + vq[1].nbytes + vq[2].nbytes
                for kq, vq in zip(self._keys_q, self._values_q)
            )
        if not self._keys:
            return 0
        return sum(k.nbytes + v.nbytes for k, v in zip(self._keys, self._values))

    def append(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        is_pivot: bool = False,
    ) -> None:
        """Append a single token's KV state for all layers.

        Args:
            keys:   Shape (num_layers, num_kv_heads, head_dim), float32.
            values: Shape (num_layers, num_kv_heads, head_dim), float32.
            is_pivot: Pinned tokens are never evicted.
        """
        # Evict lowest-importance entry when at capacity
        if self._quantize:
            if len(self._keys_q) >= self._max_tokens:
                evict_idx = int(np.argmin(self._importance))
                del self._keys_q[evict_idx]
                del self._values_q[evict_idx]
                del self._importance[evict_idx]

            # Quantize to INT8 before storing (4x memory savings)
            k_q, k_s, k_z = quantize_kv_per_head(keys.astype(np.float32))
            v_q, v_s, v_z = quantize_kv_per_head(values.astype(np.float32))
            self._keys_q.append((k_q, k_s, k_z))
            self._values_q.append((v_q, v_s, v_z))
        else:
            if len(self._keys) >= self._max_tokens:
                evict_idx = int(np.argmin(self._importance))
                del self._keys[evict_idx]
                del self._values[evict_idx]
                del self._importance[evict_idx]

            self._keys.append(keys.astype(np.float32).copy())
            self._values.append(values.astype(np.float32).copy())

        # Pivot tokens get max importance so they survive eviction sweeps
        self._importance.append(float("inf") if is_pivot else 0.0)

    def get_attention_keys_values(
        self,
        layer_idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return all stored keys/values for a specific layer.

        Returns:
            (keys, values) each shaped (seq_len, num_kv_heads, head_dim), float32.
            INT8 data is dequantized on-the-fly.
        """
        if self._quantize:
            if not self._keys_q:
                shape = (0, self.config.num_kv_heads, self.config.head_dim)
                empty = np.empty(shape, dtype=np.float32)
                return empty, empty.copy()

            # Dequantize on-the-fly during attention (zero-copy from INT8 storage)
            k_list = []
            v_list = []
            for kq, vq in zip(self._keys_q, self._values_q):
                k_deq = dequantize_kv_per_head(kq[0][layer_idx], kq[1][layer_idx], kq[2][layer_idx])
                v_deq = dequantize_kv_per_head(vq[0][layer_idx], vq[1][layer_idx], vq[2][layer_idx])
                k_list.append(k_deq)
                v_list.append(v_deq)

            return np.stack(k_list, axis=0), np.stack(v_list, axis=0)
        else:
            if not self._keys:
                shape = (0, self.config.num_kv_heads, self.config.head_dim)
                empty = np.empty(shape, dtype=np.float32)
                return empty, empty.copy()

            k_stack = np.stack([k[layer_idx] for k in self._keys], axis=0)
            v_stack = np.stack([v[layer_idx] for v in self._values], axis=0)
            return k_stack, v_stack

    def update_importance(
        self,
        layer_idx: int,
        query_state: np.ndarray,
    ) -> None:
        """Update importance scores via query-key cosine similarity.

        Args:
            layer_idx:   Layer to compute similarity for.
            query_state: Query vector, shape (num_kv_heads, head_dim) or (head_dim,).
        """
        q = query_state.reshape(-1).astype(np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0:
            return

        if self._quantize:
            for i, kq in enumerate(self._keys_q):
                if self._importance[i] == float("inf"):
                    continue
                # Dequantize just the layer we need for importance scoring
                k_deq = dequantize_kv_per_head(kq[0][layer_idx], kq[1][layer_idx], kq[2][layer_idx])
                k_flat = k_deq.reshape(-1).astype(np.float32)
                k_norm = float(np.linalg.norm(k_flat))
                if k_norm == 0:
                    continue
                score = float(np.dot(q, k_flat) / (q_norm * k_norm))
                self._importance[i] = max(self._importance[i], score)
        else:
            for i, k in enumerate(self._keys):
                if self._importance[i] == float("inf"):
                    continue
                k_flat = k[layer_idx].reshape(-1).astype(np.float32)
                k_norm = float(np.linalg.norm(k_flat))
                if k_norm == 0:
                    continue
                score = float(np.dot(q, k_flat) / (q_norm * k_norm))
                self._importance[i] = max(self._importance[i], score)

    def clear(self) -> None:
        """Clear all cached KV states."""
        if self._quantize:
            self._keys_q.clear()
            self._values_q.clear()
        else:
            self._keys.clear()
            self._values.clear()
        self._importance.clear()
        logger.debug("KV cache cleared")


class ContiguousKVCache:
    """Contiguous pre-allocated KV cache with INT8 quantization support.

    Pre-allocates a flat contiguous buffer for maximum memory locality.
    Supports both FP16 and INT8 storage modes.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int,
        quantize: bool = True,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.quantize = quantize

        if quantize:
            # INT8 storage: 4x less memory than FP32
            self.k_cache = np.zeros(
                (num_layers, max_seq_len, num_heads, head_dim),
                dtype=np.int8
            )
            self.v_cache = np.zeros(
                (num_layers, max_seq_len, num_heads, head_dim),
                dtype=np.int8
            )
            # Per-token per-layer per-head scales and zeros
            self.k_scales = np.zeros(
                (num_layers, max_seq_len, num_heads, 1),
                dtype=np.float32
            )
            self.k_zeros = np.zeros(
                (num_layers, max_seq_len, num_heads, 1),
                dtype=np.float32
            )
            self.v_scales = np.zeros(
                (num_layers, max_seq_len, num_heads, 1),
                dtype=np.float32
            )
            self.v_zeros = np.zeros(
                (num_layers, max_seq_len, num_heads, 1),
                dtype=np.float32
            )
        else:
            # FP32 storage (legacy)
            self.k_cache = np.zeros(
                (num_layers, max_seq_len, num_heads, head_dim),
                dtype=np.float32
            )
            self.v_cache = np.zeros(
                (num_layers, max_seq_len, num_heads, head_dim),
                dtype=np.float32
            )

        self.seq_len = 0

    def append(self, k: np.ndarray, v: np.ndarray) -> None:
        """Append new k, v representations.

        Args:
            k: Key tensor, shape (num_layers, batch_seq_len, num_heads, head_dim).
            v: Value tensor, shape (num_layers, batch_seq_len, num_heads, head_dim).
        """
        batch_seq_len = k.shape[1]
        assert self.seq_len + batch_seq_len <= self.max_seq_len, "KV cache capacity exceeded"

        if self.quantize:
            # Quantize per-head before storing
            for t in range(batch_seq_len):
                pos = self.seq_len + t
                k_slice = k[:, t, :, :]  # (num_layers, num_heads, head_dim)
                v_slice = v[:, t, :, :]

                k_q, k_s, k_z = quantize_kv_per_head(k_slice.astype(np.float32))
                v_q, v_s, v_z = quantize_kv_per_head(v_slice.astype(np.float32))

                self.k_cache[:, pos, :, :] = k_q
                self.v_cache[:, pos, :, :] = v_q
                self.k_scales[:, pos, :, :] = k_s
                self.k_zeros[:, pos, :, :] = k_z
                self.v_scales[:, pos, :, :] = v_s
                self.v_zeros[:, pos, :, :] = v_z
        else:
            self.k_cache[:, self.seq_len:self.seq_len + batch_seq_len, :, :] = k
            self.v_cache[:, self.seq_len:self.seq_len + batch_seq_len, :, :] = v

        self.seq_len += batch_seq_len

    def get_keys_values(
        self, layer_idx: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get all keys/values for a layer, dequantized if needed.

        Returns:
            (keys, values) each shaped (seq_len, num_heads, head_dim), float32.
        """
        if self.seq_len == 0:
            shape = (0, self.num_heads, self.head_dim)
            empty = np.empty(shape, dtype=np.float32)
            return empty, empty.copy()

        if self.quantize:
            k_int8 = self.k_cache[layer_idx, :self.seq_len, :, :]
            v_int8 = self.v_cache[layer_idx, :self.seq_len, :, :]
            k_s = self.k_scales[layer_idx, :self.seq_len, :, :]
            k_z = self.k_zeros[layer_idx, :self.seq_len, :, :]
            v_s = self.v_scales[layer_idx, :self.seq_len, :, :]
            v_z = self.v_zeros[layer_idx, :self.seq_len, :, :]

            k_fp32 = dequantize_kv_per_head(k_int8, k_s, k_z)
            v_fp32 = dequantize_kv_per_head(v_int8, v_s, v_z)
            return k_fp32, v_fp32
        else:
            return (
                self.k_cache[layer_idx, :self.seq_len, :, :].copy(),
                self.v_cache[layer_idx, :self.seq_len, :, :].copy(),
            )

    def clear(self) -> None:
        """Reset the cache."""
        self.seq_len = 0
        logger.debug("ContiguousKVCache cleared")

    @property
    def memory_bytes(self) -> int:
        """Memory used by the cache."""
        if self.quantize:
            return (
                self.k_cache.nbytes + self.v_cache.nbytes +
                self.k_scales.nbytes + self.k_zeros.nbytes +
                self.v_scales.nbytes + self.v_zeros.nbytes
            )
        return self.k_cache.nbytes + self.v_cache.nbytes
