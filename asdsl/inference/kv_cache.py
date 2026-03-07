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

    max_context_length: int = 4096
    block_size: int = 64
    max_kv_blocks: int = 32
    num_kv_heads: int = 8
    head_dim: int = 96  # hidden_dim / num_attention_heads
    streaming_heads: int = 2  # Heads that always attend to all tokens


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

    Instead of maintaining KV entries for all tokens (which scales
    linearly with context length), this cache maintains a fixed number
    of KV blocks and dynamically evicts the least important blocks
    based on query-centric similarity scores.

    Key features:
    - Fixed memory footprint regardless of context length
    - Pivot token blocks are pinned and never evicted
    - Streaming heads always attend to recent tokens
    - Query-centric importance scoring for smart eviction
    """

    def __init__(self, config: KVCacheConfig):
        self.config = config
        self.blocks: dict[int, KVBlock] = {}
        self._next_block_id = 0
        self._current_position = 0

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    @property
    def memory_bytes(self) -> int:
        """Current memory usage of the KV cache."""
        if not self.blocks:
            return 0
        bytes_per_block = (
            self.config.block_size
            * self.config.head_dim
            * 2  # keys + values
            * 2  # float16
        )
        return self.num_blocks * bytes_per_block

    def append(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        token_positions: list[int],
        is_pivot: bool = False,
    ) -> int:
        """Add a new KV block to the cache.

        If the cache is full, evicts the least important non-pinned block.

        Args:
            keys: Key tensor for this block.
            values: Value tensor for this block.
            token_positions: Token positions represented by this block.
            is_pivot: If True, pin this block (pivot tokens).

        Returns:
            Block ID of the newly added block.
        """
        # Evict if at capacity
        if self.num_blocks >= self.config.max_kv_blocks:
            self._evict_least_important()

        block = KVBlock(
            block_id=self._next_block_id,
            keys=keys.astype(np.float16),
            values=values.astype(np.float16),
            token_positions=token_positions,
            is_pinned=is_pivot,
            importance_score=1.0 if is_pivot else 0.5,
        )
        self.blocks[block.block_id] = block
        self._next_block_id += 1
        self._current_position += len(token_positions)

        return block.block_id

    def update_importance(self, query: np.ndarray) -> None:
        """Update block importance scores based on the current query.

        Uses cosine similarity between query and block keys as a
        proxy for attention importance. Higher similarity = more
        likely to receive high attention = more important to keep.

        Args:
            query: Current query vector, shape (head_dim,).
        """
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return

        for block in self.blocks.values():
            if block.is_pinned:
                continue  # Pinned blocks always have max importance

            # Average key similarity
            key_norms = np.linalg.norm(block.keys.astype(np.float32), axis=-1)
            valid = key_norms > 0
            if not np.any(valid):
                block.importance_score = 0.0
                continue

            similarities = (
                block.keys[valid].astype(np.float32) @ query.astype(np.float32)
            ) / (key_norms[valid] * query_norm)

            block.importance_score = float(np.mean(similarities))

    def get_attention_keys_values(
        self,
        head_idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve aggregated keys and values for attention computation.

        For streaming heads, returns all blocks.
        For other heads, returns only the top-importance blocks.

        Args:
            head_idx: Attention head index.

        Returns:
            Tuple of (keys, values) concatenated from selected blocks.
        """
        is_streaming = head_idx < self.config.streaming_heads

        if is_streaming:
            selected = sorted(self.blocks.values(), key=lambda b: b.block_id)
        else:
            # Select top blocks by importance
            sorted_blocks = sorted(
                self.blocks.values(),
                key=lambda b: (b.is_pinned, b.importance_score),
                reverse=True,
            )
            max_blocks = self.config.max_kv_blocks // 2  # Non-streaming uses fewer
            selected = sorted_blocks[:max_blocks]

        if not selected:
            head_dim = self.config.head_dim
            return np.empty((0, head_dim), dtype=np.float16), np.empty(
                (0, head_dim), dtype=np.float16
            )

        all_keys = np.concatenate([b.keys for b in selected], axis=0)
        all_values = np.concatenate([b.values for b in selected], axis=0)

        return all_keys, all_values

    def _evict_least_important(self) -> None:
        """Evict the least important non-pinned block."""
        candidates = [
            b for b in self.blocks.values() if not b.is_pinned
        ]
        if not candidates:
            logger.warning("All KV blocks are pinned, cannot evict")
            return

        victim = min(candidates, key=lambda b: b.importance_score)
        del self.blocks[victim.block_id]
        logger.debug("Evicted KV block %d (importance=%.3f)", victim.block_id, victim.importance_score)

    def clear(self) -> None:
        """Clear all cached KV blocks."""
        self.blocks.clear()
        self._current_position = 0
