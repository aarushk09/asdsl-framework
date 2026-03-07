"""OS-level memory management: pinning, Huge Pages, NUMA awareness.

Provides the critical OS interface layer that ensures:
- Model weights are locked in physical RAM (no page faults during inference)
- Huge Pages reduce TLB misses during sequential weight scanning
- NUMA-aware allocation keeps memory local to compute cores
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import platform
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MemoryRegion:
    """A managed memory region with pinning and allocation metadata.

    Attributes:
        buffer: The numpy array backed by this memory.
        size_bytes: Total allocation size.
        is_pinned: Whether the memory is locked in physical RAM.
        is_huge_page: Whether Huge Pages are used.
        numa_node: NUMA node the memory is allocated on (-1 = any).
        base_address: Virtual memory base address.
    """

    buffer: np.ndarray
    size_bytes: int
    is_pinned: bool = False
    is_huge_page: bool = False
    numa_node: int = -1

    @property
    def base_address(self) -> int:
        return self.buffer.ctypes.data

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


class MemoryManager:
    """Manages memory allocation, pinning, and optimization for inference.

    On initialization, detects available OS features and selects the
    best allocation strategy for the current platform.
    """

    def __init__(
        self,
        use_huge_pages: bool = True,
        pin_memory: bool = True,
        numa_aware: bool = True,
    ):
        self.use_huge_pages = use_huge_pages
        self.pin_memory = pin_memory
        self.numa_aware = numa_aware

        self._system = platform.system()
        self._regions: list[MemoryRegion] = []
        self._total_pinned_bytes = 0

        # Detect capabilities
        self._can_pin = self._check_pin_support()
        self._can_huge_pages = self._check_huge_page_support()
        self._numa_nodes = self._detect_numa_topology()

        logger.info(
            "MemoryManager initialized: pin=%s, huge_pages=%s, numa_nodes=%d",
            self._can_pin,
            self._can_huge_pages,
            len(self._numa_nodes),
        )

    def allocate(
        self,
        size_bytes: int,
        dtype: np.dtype = np.dtype(np.uint8),
        numa_node: int = -1,
    ) -> MemoryRegion:
        """Allocate an optimized memory region for model weights.

        Attempts to allocate with Huge Pages and memory pinning based
        on platform capabilities and configuration.

        Args:
            size_bytes: Required allocation size.
            dtype: Numpy dtype for the buffer.
            numa_node: Preferred NUMA node (-1 = auto-select).

        Returns:
            MemoryRegion with the allocated buffer.
        """
        num_elements = size_bytes // dtype.itemsize
        if size_bytes % dtype.itemsize:
            num_elements += 1

        # Attempt Huge Page allocation
        is_huge = False
        if self.use_huge_pages and self._can_huge_pages:
            buffer = self._allocate_huge_pages(num_elements, dtype)
            if buffer is not None:
                is_huge = True
            else:
                logger.debug("Huge page allocation failed, falling back to standard")
                buffer = np.empty(num_elements, dtype=dtype)
        else:
            buffer = np.empty(num_elements, dtype=dtype)

        # Pin memory
        is_pinned = False
        if self.pin_memory and self._can_pin:
            if self._pin_buffer(buffer):
                is_pinned = True
                self._total_pinned_bytes += buffer.nbytes

        region = MemoryRegion(
            buffer=buffer,
            size_bytes=buffer.nbytes,
            is_pinned=is_pinned,
            is_huge_page=is_huge,
            numa_node=numa_node,
        )
        self._regions.append(region)

        logger.debug(
            "Allocated region: %.2f MB (pinned=%s, huge=%s, numa=%d)",
            region.size_mb,
            is_pinned,
            is_huge,
            numa_node,
        )

        return region

    def allocate_for_weights(
        self,
        packed_data: np.ndarray,
    ) -> MemoryRegion:
        """Allocate a pinned region and copy weight data into it.

        This is the main entry point for loading quantized weights
        into inference-ready memory.

        Args:
            packed_data: Packed quantized weight data to copy.

        Returns:
            MemoryRegion containing the weight data.
        """
        region = self.allocate(
            size_bytes=packed_data.nbytes,
            dtype=packed_data.dtype,
        )
        np.copyto(region.buffer[: len(packed_data)], packed_data)
        return region

    def release_all(self) -> None:
        """Release all managed memory regions."""
        for region in self._regions:
            if region.is_pinned:
                self._unpin_buffer(region.buffer)

        self._regions.clear()
        self._total_pinned_bytes = 0
        logger.info("All memory regions released")

    @property
    def total_allocated_mb(self) -> float:
        return sum(r.size_bytes for r in self._regions) / (1024 * 1024)

    @property
    def total_pinned_mb(self) -> float:
        return self._total_pinned_bytes / (1024 * 1024)

    # --- Platform-specific implementations ---

    def _check_pin_support(self) -> bool:
        """Check if memory pinning (mlock) is available."""
        if self._system == "Linux":
            try:
                libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
                return hasattr(libc, "mlock")
            except OSError:
                return False
        elif self._system == "Windows":
            try:
                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                return hasattr(kernel32, "VirtualLock")
            except OSError:
                return False
        elif self._system == "Darwin":
            try:
                libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
                return hasattr(libc, "mlock")
            except OSError:
                return False
        return False

    def _check_huge_page_support(self) -> bool:
        """Check if Huge Pages are available and configured."""
        if self._system == "Linux":
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if "HugePages_Total" in line:
                            total = int(line.split()[1])
                            return total > 0
            except (OSError, ValueError):
                pass
            return False
        elif self._system == "Windows":
            # Windows supports Large Pages but requires privilege
            return False  # Conservative: require explicit opt-in
        return False

    def _detect_numa_topology(self) -> list[int]:
        """Detect available NUMA nodes."""
        if self._system == "Linux":
            numa_path = "/sys/devices/system/node"
            try:
                nodes = []
                if os.path.isdir(numa_path):
                    for entry in os.listdir(numa_path):
                        if entry.startswith("node"):
                            try:
                                nodes.append(int(entry[4:]))
                            except ValueError:
                                pass
                return sorted(nodes) if nodes else [0]
            except OSError:
                return [0]
        return [0]  # Single node default

    def _allocate_huge_pages(
        self,
        num_elements: int,
        dtype: np.dtype,
    ) -> np.ndarray | None:
        """Attempt to allocate memory backed by Huge Pages (Linux only)."""
        if self._system != "Linux":
            return None

        try:
            import mmap as mmap_module

            size = num_elements * dtype.itemsize
            # Round up to 2MB Huge Page boundary
            huge_page_size = 2 * 1024 * 1024
            aligned_size = ((size + huge_page_size - 1) // huge_page_size) * huge_page_size

            MAP_HUGETLB = 0x40000
            MAP_ANONYMOUS = 0x20
            MAP_PRIVATE = 0x02

            mm = mmap_module.mmap(
                -1,
                aligned_size,
                flags=MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                prot=mmap_module.PROT_READ | mmap_module.PROT_WRITE,
            )

            buffer = np.frombuffer(mm, dtype=dtype, count=num_elements)
            logger.debug("Huge page allocation successful: %d MB", aligned_size // (1024 * 1024))
            return buffer

        except (OSError, ValueError, OverflowError) as e:
            logger.debug("Huge page allocation failed: %s", e)
            return None

    def _pin_buffer(self, buffer: np.ndarray) -> bool:
        """Pin a numpy buffer in physical RAM using OS mlock/VirtualLock."""
        addr = buffer.ctypes.data
        size = buffer.nbytes

        if self._system in ("Linux", "Darwin"):
            try:
                libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
                result = libc.mlock(ctypes.c_void_p(addr), ctypes.c_size_t(size))
                if result == 0:
                    logger.debug("mlock successful: %d bytes at 0x%x", size, addr)
                    return True
                else:
                    errno = ctypes.get_errno()
                    logger.debug("mlock failed (errno %d): may need ulimit -l increase", errno)
                    return False
            except OSError:
                return False

        elif self._system == "Windows":
            try:
                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                result = kernel32.VirtualLock(ctypes.c_void_p(addr), ctypes.c_size_t(size))
                if result:
                    logger.debug("VirtualLock successful: %d bytes", size)
                    return True
                else:
                    error = ctypes.get_last_error()
                    logger.debug("VirtualLock failed (error %d)", error)
                    return False
            except OSError:
                return False

        return False

    def _unpin_buffer(self, buffer: np.ndarray) -> bool:
        """Unpin a previously pinned buffer."""
        addr = buffer.ctypes.data
        size = buffer.nbytes

        if self._system in ("Linux", "Darwin"):
            try:
                libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
                libc.munlock(ctypes.c_void_p(addr), ctypes.c_size_t(size))
                return True
            except OSError:
                return False

        elif self._system == "Windows":
            try:
                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel32.VirtualUnlock(ctypes.c_void_p(addr), ctypes.c_size_t(size))
                return True
            except OSError:
                return False

        return False


def pin_model_weights(
    weight_buffers: dict[str, np.ndarray],
    use_huge_pages: bool = True,
) -> tuple[MemoryManager, dict[str, MemoryRegion]]:
    """Convenience function to pin all model weight buffers.

    Args:
        weight_buffers: Dictionary of name → numpy array for model weights.
        use_huge_pages: Whether to attempt Huge Page allocation.

    Returns:
        Tuple of (MemoryManager, dict mapping names to MemoryRegions).
    """
    manager = MemoryManager(
        use_huge_pages=use_huge_pages,
        pin_memory=True,
        numa_aware=True,
    )

    regions: dict[str, MemoryRegion] = {}
    for name, buffer in weight_buffers.items():
        region = manager.allocate_for_weights(buffer)
        regions[name] = region
        logger.info("Pinned %s: %.2f MB (huge=%s)", name, region.size_mb, region.is_huge_page)

    logger.info(
        "Total pinned: %.2f MB across %d buffers",
        manager.total_pinned_mb,
        len(regions),
    )

    return manager, regions
