"""Tests for asdsl.io.WeightStreamer and is_iouring_available.

These tests require no model weights — they use temporary files.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from asdsl.io.weight_streamer import WeightStreamer
from asdsl.io.iouring_detect import is_iouring_available


# ---------------------------------------------------------------------------
# IoRing detection
# ---------------------------------------------------------------------------

def test_iouring_detection_does_not_crash():
    """is_iouring_available() must return (bool, str) without raising."""
    available, reason = is_iouring_available()
    assert isinstance(available, bool), f"expected bool, got {type(available)}"
    assert isinstance(reason, str),  f"expected str,  got {type(reason)}"
    print(f"\n  IoRing: available={available}  reason={reason!r}")


# ---------------------------------------------------------------------------
# Thread-based fallback correctness
# ---------------------------------------------------------------------------

def test_sync_fallback_reads_file():
    """Create a 1 MB temp file, prefetch it, verify bytes match."""
    import numpy as np
    data = bytes(range(256)) * (1024 * 4)   # 1 MB deterministic pattern

    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        f.write(data)
        path = f.name

    try:
        with WeightStreamer(path, use_iouring=False) as s:
            s.submit_prefetch(layer_idx=0, byte_offset=0, byte_length=len(data))
            result = s.wait_and_get(layer_idx=0, timeout_s=5.0)

        assert result is not None, "wait_and_get returned None"
        assert result == data,     f"data mismatch (got {len(result)} bytes)"
    finally:
        os.unlink(path)


def test_prefetch_latency_hidden_by_compute():
    """Submit prefetch, sleep to 'compute', verify wait is nearly instant."""
    data = os.urandom(4 * 1024 * 1024)   # 4 MB random

    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        f.write(data)
        path = f.name

    try:
        with WeightStreamer(path, use_iouring=False) as s:
            t0 = time.perf_counter()
            s.submit_prefetch(layer_idx=7, byte_offset=0, byte_length=len(data))
            time.sleep(0.08)                     # simulate compute (80 ms)
            result = s.wait_and_get(layer_idx=7, timeout_s=5.0)
            t1 = time.perf_counter()

        assert result == data, "data mismatch after async prefetch"
        elapsed = t1 - t0
        # I/O should be hidden: total wall time should be ~0.08 s, not 0.08 + I/O
        assert elapsed < 0.30, (
            f"wait_and_get took {elapsed:.3f}s total — I/O may not be hidden"
        )
        print(f"\n  Total elapsed: {elapsed*1000:.0f} ms  "
              f"(I/O overlapped with 80 ms compute)")
    finally:
        os.unlink(path)


def test_multiple_layers_sequential():
    """Submit two layers, retrieve in order, both must match."""
    chunk_a = os.urandom(512 * 1024)   # 512 KB
    chunk_b = os.urandom(512 * 1024)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        f.write(chunk_a + chunk_b)
        path = f.name

    try:
        with WeightStreamer(path, use_iouring=False) as s:
            s.submit_prefetch(0, 0,               len(chunk_a))
            s.submit_prefetch(1, len(chunk_a),    len(chunk_b))

            got_a = s.wait_and_get(0, timeout_s=5.0)
            got_b = s.wait_and_get(1, timeout_s=5.0)

        assert got_a == chunk_a, "chunk a mismatch"
        assert got_b == chunk_b, "chunk b mismatch"
    finally:
        os.unlink(path)
