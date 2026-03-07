"""Tests for OS-level memory management."""

import platform
import sys

import numpy as np
import pytest

from asdsl.memory.manager import MemoryManager


class TestMemoryManager:
    """Tests for the cross-platform memory manager."""

    def test_allocate_aligned(self):
        """Allocated buffers should have correct size."""
        mm = MemoryManager(enable_huge_pages=False, enable_pinning=False)
        region = mm.allocate_for_weights("test", shape=(256, 256), dtype=np.float32)
        assert region.buffer.shape == (256, 256)
        assert region.buffer.dtype == np.float32

    def test_allocate_for_weights_metadata(self):
        """Memory region metadata should be correct."""
        mm = MemoryManager(enable_huge_pages=False, enable_pinning=False)
        region = mm.allocate_for_weights("layer0_q", shape=(64,), dtype=np.float16)
        assert region.name == "layer0_q"
        assert region.size_bytes == 64 * 2  # float16 = 2 bytes

    def test_stats(self):
        """Stats should track allocated memory."""
        mm = MemoryManager(enable_huge_pages=False, enable_pinning=False)
        mm.allocate_for_weights("a", shape=(100,), dtype=np.float32)
        stats = mm.get_stats()
        assert stats["total_allocated_bytes"] >= 400  # 100 * 4 bytes

    def test_release(self):
        """Releasing should remove the region."""
        mm = MemoryManager(enable_huge_pages=False, enable_pinning=False)
        mm.allocate_for_weights("temp", shape=(32,), dtype=np.float32)
        assert "temp" in mm._regions
        mm.release("temp")
        assert "temp" not in mm._regions

    def test_release_all(self):
        """Release all should clear every region."""
        mm = MemoryManager(enable_huge_pages=False, enable_pinning=False)
        mm.allocate_for_weights("a", shape=(32,), dtype=np.float32)
        mm.allocate_for_weights("b", shape=(32,), dtype=np.float32)
        mm.release_all()
        assert len(mm._regions) == 0

    def test_numa_detection_does_not_crash(self):
        """NUMA detection should run without error on any platform."""
        mm = MemoryManager(enable_huge_pages=False, enable_pinning=False)
        info = mm.get_numa_info()
        assert isinstance(info, dict)


class TestPinning:
    """Platform-specific pinning tests (may skip on some platforms)."""

    @pytest.mark.skipif(
        sys.platform not in ("linux", "darwin", "win32"),
        reason="Only test on known platforms",
    )
    def test_pinning_does_not_raise(self):
        """Pinning with enable_pinning=True should not raise on supported platforms."""
        mm = MemoryManager(enable_huge_pages=False, enable_pinning=True)
        # Small allocation – may fail to pin on CI but should not raise
        try:
            region = mm.allocate_for_weights("pin_test", shape=(64,), dtype=np.float32)
        except OSError:
            pytest.skip("Pinning not allowed in this environment")
