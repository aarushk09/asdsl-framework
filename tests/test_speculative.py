"""Tests for SWIFT self-speculative decoding."""

import numpy as np
import pytest

from asdsl.speculative.swift import (
    SWIFTDecoder,
    SkipSchedule,
    _softmax,
    create_skip_schedule_for_phi3,
)


class MockLayerExecutor:
    """Mock layer executor for testing the SWIFT decoder."""

    def __init__(self, num_layers: int = 32, hidden_dim: int = 64, vocab_size: int = 100):
        self._num_layers = num_layers
        self._hidden_dim = hidden_dim
        self._vocab_size = vocab_size
        self.layers_executed: list[int] = []

    @property
    def num_layers(self) -> int:
        return self._num_layers

    def execute_layer(self, layer_idx: int, hidden_state: np.ndarray) -> np.ndarray:
        self.layers_executed.append(layer_idx)
        return hidden_state * 0.99  # Slight decay to simulate processing

    def execute_lm_head(self, hidden_state: np.ndarray) -> np.ndarray:
        np.random.seed(42)
        if hidden_state.ndim == 1:
            return np.random.randn(self._vocab_size).astype(np.float32)
        return np.random.randn(hidden_state.shape[0], self._vocab_size).astype(np.float32)


class TestSkipSchedule:
    """Tests for layer skip schedule configuration."""

    def test_phi3_default_schedule(self):
        """Default Phi-3 schedule should skip layers 4-27."""
        schedule = create_skip_schedule_for_phi3()
        assert schedule.total_layers == 32
        assert 0 not in schedule.skip_indices
        assert 31 not in schedule.skip_indices
        assert 15 in schedule.skip_indices  # Middle layer should be skipped
        assert len(schedule.draft_layers) == 8  # 4 first + 4 last

    def test_skip_ratio(self):
        """Skip ratio should reflect actual skip count."""
        schedule = SkipSchedule(total_layers=32, keep_first=4, keep_last=4)
        assert abs(schedule.skip_ratio - 0.75) < 0.01  # 24/32 = 75%

    def test_speedup_estimate(self):
        """Speedup estimate should be > 1 when skipping layers."""
        schedule = SkipSchedule(total_layers=32, keep_first=4, keep_last=4)
        assert schedule.speedup_estimate > 3.0  # 32/8 = 4x theoretical

    def test_no_skip(self):
        """Schedule with no skipping should have all layers as draft layers."""
        schedule = SkipSchedule(total_layers=8, keep_first=4, keep_last=4)
        assert len(schedule.skip_indices) == 0
        assert len(schedule.draft_layers) == 8


class TestSWIFTDecoder:
    """Tests for the SWIFT self-speculative decoder."""

    def test_draft_forward_skips_layers(self):
        """Draft forward should only execute non-skipped layers."""
        executor = MockLayerExecutor(num_layers=32)
        decoder = SWIFTDecoder(
            executor=executor,
            num_draft_tokens=4,
            keep_first=4,
            keep_last=4,
        )

        hidden = np.random.randn(64).astype(np.float32)
        executor.layers_executed.clear()
        decoder.draft_forward(hidden)

        # Should only execute 8 layers (first 4 + last 4)
        assert len(executor.layers_executed) == 8
        assert all(i < 4 or i >= 28 for i in executor.layers_executed)

    def test_full_forward_executes_all_layers(self):
        """Full forward should execute all 32 layers."""
        executor = MockLayerExecutor(num_layers=32)
        decoder = SWIFTDecoder(executor=executor, num_draft_tokens=4)

        hidden = np.random.randn(64).astype(np.float32)
        executor.layers_executed.clear()
        decoder.full_forward(hidden)

        assert len(executor.layers_executed) == 32

    def test_speculative_step_produces_tokens(self):
        """Speculative step should produce at least one token."""
        executor = MockLayerExecutor(num_layers=16, vocab_size=100)
        decoder = SWIFTDecoder(
            executor=executor,
            num_draft_tokens=4,
            keep_first=2,
            keep_last=2,
            adaptive_schedule=False,
        )

        hidden = np.random.randn(64).astype(np.float32)
        result = decoder.speculative_step(hidden, past_tokens=[1, 2, 3])

        assert len(result.accepted_tokens) >= 1
        assert result.num_draft_tokens == 4

    def test_zero_temperature_is_greedy(self):
        """Temperature 0 should always pick argmax."""
        executor = MockLayerExecutor(num_layers=8, vocab_size=10)
        decoder = SWIFTDecoder(
            executor=executor,
            num_draft_tokens=1,
            keep_first=2,
            keep_last=2,
            temperature=0.0,
        )
        logits = np.array([0.1, 0.2, 5.0, 0.3, 0.1])
        token = decoder._sample_token(logits)
        assert token == 2


class TestSoftmax:
    """Tests for the softmax utility."""

    def test_softmax_sums_to_one(self):
        logits = np.array([1.0, 2.0, 3.0])
        probs = _softmax(logits)
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-6)

    def test_softmax_preserves_order(self):
        logits = np.array([1.0, 3.0, 2.0])
        probs = _softmax(logits)
        assert probs[1] > probs[2] > probs[0]

    def test_softmax_handles_large_values(self):
        """Should not overflow with large logit values."""
        logits = np.array([1000.0, 1001.0, 999.0])
        probs = _softmax(logits)
        assert np.all(np.isfinite(probs))
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-6)
