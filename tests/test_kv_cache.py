"""Tests for block-sparse KV cache."""

import numpy as np
import pytest

from asdsl.inference.kv_cache import BlockSparseKVCache, KVCacheConfig


class TestKVCacheBasics:
    """Basic KV cache operations."""

    def test_append_and_length(self):
        """Appending tokens should increase cache length."""
        cfg = KVCacheConfig(
            num_layers=2, num_kv_heads=2, head_dim=16,
            max_blocks=4, block_size=4,
        )
        cache = BlockSparseKVCache(cfg)
        k = np.random.randn(2, 2, 16).astype(np.float32)  # [layers, heads, dim]
        v = np.random.randn(2, 2, 16).astype(np.float32)
        cache.append(k, v)
        assert cache.length == 1

    def test_append_multiple(self):
        """Multiple appends should grow length correctly."""
        cfg = KVCacheConfig(
            num_layers=1, num_kv_heads=1, head_dim=8,
            max_blocks=8, block_size=4,
        )
        cache = BlockSparseKVCache(cfg)
        for _ in range(6):
            k = np.random.randn(1, 1, 8).astype(np.float32)
            v = np.random.randn(1, 1, 8).astype(np.float32)
            cache.append(k, v)
        assert cache.length == 6

    def test_get_attention_keys_values_shape(self):
        """Retrieved keys/values should have correct shape."""
        cfg = KVCacheConfig(
            num_layers=1, num_kv_heads=2, head_dim=8,
            max_blocks=4, block_size=4,
        )
        cache = BlockSparseKVCache(cfg)
        for _ in range(3):
            k = np.random.randn(1, 2, 8).astype(np.float32)
            v = np.random.randn(1, 2, 8).astype(np.float32)
            cache.append(k, v)

        keys, values = cache.get_attention_keys_values(layer_idx=0)
        assert keys.shape[0] == 3  # seq_len
        assert keys.shape[-1] == 8  # head_dim


class TestKVCacheEviction:
    """Eviction policy tests."""

    def test_eviction_maintains_max_length(self):
        """Cache should not exceed max_blocks * block_size entries."""
        cfg = KVCacheConfig(
            num_layers=1, num_kv_heads=1, head_dim=4,
            max_blocks=2, block_size=2,  # max 4 entries
        )
        cache = BlockSparseKVCache(cfg)
        for _ in range(8):
            k = np.random.randn(1, 1, 4).astype(np.float32)
            v = np.random.randn(1, 1, 4).astype(np.float32)
            cache.append(k, v)

        assert cache.length <= 4

    def test_importance_update(self):
        """Importance scores should be updatable without error."""
        cfg = KVCacheConfig(
            num_layers=1, num_kv_heads=1, head_dim=4,
            max_blocks=4, block_size=4,
        )
        cache = BlockSparseKVCache(cfg)
        for _ in range(3):
            k = np.random.randn(1, 1, 4).astype(np.float32)
            v = np.random.randn(1, 1, 4).astype(np.float32)
            cache.append(k, v)

        query = np.random.randn(1, 4).astype(np.float32)
        cache.update_importance(layer_idx=0, query_state=query)


class TestKVCacheConfig:
    """Configuration validation."""

    def test_memory_budget_bytes(self):
        """Memory budget calculation should be reasonable."""
        cfg = KVCacheConfig(
            num_layers=32, num_kv_heads=8, head_dim=96,
            max_blocks=64, block_size=16,
        )
        # 32 layers * 8 heads * 96 dim * 64 blocks * 16 block_size * 2 (K+V) * 4 bytes
        expected_max = 32 * 8 * 96 * 64 * 16 * 2 * 4
        assert cfg.memory_budget_bytes() <= expected_max
