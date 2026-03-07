"""Tests for the async prefetch orchestrator."""

import threading
import time

import numpy as np
import pytest

from asdsl.prefetch.orchestrator import PrefetchOrchestrator


class TestPrefetchOrchestrator:
    """Tests for the dual-thread prefetch orchestrator."""

    def test_register_and_lookup_buffers(self):
        """Registered weight buffers should be retrievable."""
        orch = PrefetchOrchestrator(num_layers=4, lookahead=1)
        buf = np.zeros(1024, dtype=np.float32)
        orch.register_weight_buffer(layer_idx=0, name="q_proj", buffer=buf)
        assert "0:q_proj" in orch._weight_buffers

    def test_start_and_stop(self):
        """Orchestrator should start and stop cleanly."""
        orch = PrefetchOrchestrator(num_layers=4, lookahead=1)
        orch.start()
        assert orch._running
        orch.stop()
        assert not orch._running

    def test_notify_layer_triggers_prefetch(self):
        """Notifying current layer should enqueue next layer for prefetch."""
        orch = PrefetchOrchestrator(num_layers=4, lookahead=1)
        buf0 = np.zeros(64, dtype=np.float32)
        buf1 = np.ones(64, dtype=np.float32)
        orch.register_weight_buffer(0, "w", buf0)
        orch.register_weight_buffer(1, "w", buf1)

        orch.start()
        orch.notify_layer_start(0)
        # Give the worker thread a moment
        time.sleep(0.05)
        orch.stop()

        assert orch._stats["prefetch_requests"] >= 1

    def test_stats_are_populated(self):
        """Stats dict should reflect orchestrator activity."""
        orch = PrefetchOrchestrator(num_layers=2, lookahead=1)
        orch.register_weight_buffer(0, "w", np.zeros(32, dtype=np.float32))
        orch.register_weight_buffer(1, "w", np.zeros(32, dtype=np.float32))
        orch.start()
        orch.notify_layer_start(0)
        time.sleep(0.05)
        stats = orch.get_stats()
        orch.stop()
        assert "prefetch_requests" in stats

    def test_speculative_prefetch(self):
        """Draft-start notification should queue verification layers."""
        orch = PrefetchOrchestrator(num_layers=8, lookahead=1)
        for i in range(8):
            orch.register_weight_buffer(i, "w", np.zeros(32, dtype=np.float32))
        orch.start()
        orch.notify_speculative_draft_start(skip_indices={2, 3, 4, 5})
        time.sleep(0.05)
        orch.stop()
        assert orch._stats["prefetch_requests"] >= 1
