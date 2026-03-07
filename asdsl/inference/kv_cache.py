"""Block-wise sparse attention for KV cache management.

Implements structured sparsity for attention computation to prevent
KV cache from ballooning on long contexts. Uses query-centric
similarity to prune less important KV pages dynamically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class KVCacheConfig:
    """Configuration for the block-sparse KV cache."""

    num_layers: int = 1
    num_kv_heads: int = 8
    head_dim: int = 96
    max_blocks: int = 32
    block_size: int = 64

    def memory_budget_bytes(self) -> int:
        """Upper bound on memory: all layers * heads * blocks * block_size * K+V * float32."""
        return (
            self.num_layers
            * self.num_kv_heads
            * self.head_dim
            * self.max_blocks
            * self.block_size
            * 2   # keys + values
            * 4   # float32 bytes
        )


@dataclass
class KVBlock:
    """A block of key-value cache entries.

    Attributes:
        block_id: Unique block identifier.
        keys: Key tensor, shape (block_size, head_dim).
        values: Value tensor, shape (block_size, head_dim).
        importance_score: Query-centric importance for eviction decisions.
        token_positions: Original token positions in this block.
        is_pinned: If True, this block cannot be evicted (pivot tokens).
    """

    block_id: int
    keys: np.ndarray
    values: np.ndarray
    importance_score: float = 0.0
    token_positions: list[int] = field(default_factory=list)
    is_pinned: bool = False


class BlockSparseKVCache:
    """Block-wise sparse KV cache with dynamic eviction.

    Stores per-token KV states for all layers. When the cache exceeds
    max_blocks * block_size tokens, the least-important token is evicted.

    ``append(k, v)`` takes a single-token snapshot where k/v have shape
    ``(num_layers, num_kv_heads, head_dim)``.

    ``get_attention_keys_values(layer_idx)`` returns all stored tokens
    for that layer as ``(seq_len, num_kv_heads, head_dim)`` arrays.
    """

    def __init__(self, config: KVCacheConfig):
        self.config = config
        self._max_tokens: int = config.max_blocks * config.block_size
        # Per-token storage: each slot holds (num_layers, num_kv_heads, head_dim)
        self._keys: list[np.ndarray] = []
        self._values: list[np.ndarray] = []
        self._importance: list[float] = []

    @property
    def length(self) -> int:
        """Number of tokens currently stored."""
        return len(self._keys)

    @property
    def memory_bytes(self) -> int:
        """Approximate memory used by stored KV tensors."""
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
            (keys, values) each shaped (seq_len, num_kv_heads, head_dim).
        """
        if not self._keys:
            shape = (0, self.config.num_kv_heads, self.config.head_dim)
            empty = np.empty(shape, dtype=np.float32)
            return empty, empty.copy()

        # Stack token snapshots and extract the requested layer
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
        if not self._keys:
            return

        q = query_state.reshape(-1).astype(np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0:
            return

        for i, k in enumerate(self._keys):
            if self._importance[i] == float("inf"):
                continue  # Pinned
            k_flat = k[layer_idx].reshape(-1).astype(np.float32)
            k_norm = float(np.linalg.norm(k_flat))
            if k_norm == 0:
                continue
            score = float(np.dot(q, k_flat) / (q_norm * k_norm))
            self._importance[i] = max(self._importance[i], score)

    def clear(self) -> None:
        """Clear all cached KV states."""
        self._keys.clear()
        self._values.clear()
        self._importance.clear()
        logger.debug("KV cache cleared")

