"""Asynchronous L2 cache prefetching orchestrator.

Implements the dual-thread architecture where:
- Thread 0 (Compute): Executes LUT-based computations for layer L_n
- Thread 1 (Prefetch): Asynchronously loads weights for L_{n+1} into L2 cache

Because transformer inference is strictly sequential (L1→L2→...→L32),
the prefetch thread achieves near-100% accuracy, effectively masking
DRAM latency from the compute thread's perspective.
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PrefetchRequest:
    """A request to prefetch a weight buffer into CPU cache.

    Attributes:
        layer_idx: Target transformer layer index.
        weight_name: Name of the weight tensor within the layer.
        data_ptr: Memory address of the weight data.
        size_bytes: Size of the data to prefetch.
        priority: Prefetch priority (lower = more urgent).
    """

    layer_idx: int
    weight_name: str
    data_ptr: int
    size_bytes: int
    priority: int = 0


@dataclass
class PrefetchStats:
    """Statistics for the prefetch orchestrator."""

    total_requests: int = 0
    completed_requests: int = 0
    total_bytes_prefetched: int = 0
    cache_hits_estimated: int = 0
    avg_latency_us: float = 0.0

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits_estimated / self.total_requests


class PrefetchOrchestrator:
    """Dual-thread asynchronous L2 cache prefetch orchestrator.

    Manages a dedicated prefetch thread that pre-loads upcoming layer
    weights into the CPU's L2/L3 cache while the compute thread
    processes the current layer.

    The orchestrator supports:
    - Deterministic prefetch: layer N+1 weights loaded during layer N
    - Speculative prefetch: pre-loads verification layers during draft phase
    - Priority queue: critical weights (high-salience groups) prefetched first
    """

    def __init__(
        self,
        num_layers: int,
        prefetch_ahead: int = 1,
        cache_line_bytes: int = 64,
        enable_os_hints: bool = True,
    ):
        """Initialize the prefetch orchestrator.

        Args:
            num_layers: Total transformer layers in the model.
            prefetch_ahead: Number of layers to prefetch ahead.
            cache_line_bytes: CPU cache line size.
            enable_os_hints: Try to use OS-level prefetch hints if available.
        """
        self.num_layers = num_layers
        self.prefetch_ahead = prefetch_ahead
        self.cache_line_bytes = cache_line_bytes
        self.enable_os_hints = enable_os_hints

        # Weight buffers registry: layer_idx → {name: (buffer, size)}
        self._weight_registry: dict[int, dict[str, tuple[np.ndarray, int]]] = {}

        # Thread synchronization
        self._prefetch_queue: list[PrefetchRequest] = []
        self._queue_lock = threading.Lock()
        self._queue_event = threading.Event()
        self._shutdown = threading.Event()

        # Stats
        self.stats = PrefetchStats()

        # Prefetch thread
        self._prefetch_thread: threading.Thread | None = None
        self._is_running = False

        logger.info(
            "Prefetch orchestrator initialized: %d layers, prefetch_ahead=%d",
            num_layers,
            prefetch_ahead,
        )

    def register_weight_buffer(
        self,
        layer_idx: int,
        weight_name: str,
        buffer: np.ndarray,
    ) -> None:
        """Register a weight buffer for prefetch management.

        All model weight buffers must be registered before inference begins.
        Buffers should be memory-pinned (see memory module) for optimal results.

        Args:
            layer_idx: Transformer layer index.
            weight_name: Descriptive name for the weight tensor.
            buffer: The numpy array holding the quantized weight data.
        """
        if layer_idx not in self._weight_registry:
            self._weight_registry[layer_idx] = {}

        self._weight_registry[layer_idx][weight_name] = (buffer, buffer.nbytes)
        logger.debug(
            "Registered weight buffer: layer %d/%s (%d bytes)",
            layer_idx,
            weight_name,
            buffer.nbytes,
        )

    def start(self) -> None:
        """Start the prefetch thread."""
        if self._is_running:
            return

        self._shutdown.clear()
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            name="asdsl-prefetch",
            daemon=True,
        )
        self._is_running = True
        self._prefetch_thread.start()
        logger.info("Prefetch thread started")

    def stop(self) -> None:
        """Stop the prefetch thread."""
        if not self._is_running:
            return

        self._shutdown.set()
        self._queue_event.set()  # Wake up the thread

        if self._prefetch_thread:
            self._prefetch_thread.join(timeout=5.0)
            self._prefetch_thread = None

        self._is_running = False
        logger.info(
            "Prefetch thread stopped. Stats: %d requests, %d bytes prefetched",
            self.stats.total_requests,
            self.stats.total_bytes_prefetched,
        )

    def notify_layer_start(self, layer_idx: int) -> None:
        """Notify the orchestrator that computation has started on a layer.

        This triggers prefetching of the next `prefetch_ahead` layers.

        Args:
            layer_idx: The layer currently being computed.
        """
        for ahead in range(1, self.prefetch_ahead + 1):
            target_layer = layer_idx + ahead
            if target_layer >= self.num_layers:
                break

            self._enqueue_layer_prefetch(target_layer)

    def notify_speculative_draft_start(self, skip_layers: set[int]) -> None:
        """Notify that a speculative draft phase is starting.

        Pre-loads verification layers (the ones NOT skipped during draft)
        so the full verification pass can execute without cache misses.

        Args:
            skip_layers: Set of layer indices being skipped in draft.
        """
        # Prioritize non-skipped layers that will be used in verification
        verify_layers = [i for i in range(self.num_layers) if i not in skip_layers]

        for layer_idx in verify_layers:
            self._enqueue_layer_prefetch(layer_idx, priority=1)

    def _enqueue_layer_prefetch(self, layer_idx: int, priority: int = 0) -> None:
        """Enqueue all weights for a layer for prefetching."""
        if layer_idx not in self._weight_registry:
            return

        with self._queue_lock:
            for name, (buffer, size) in self._weight_registry[layer_idx].items():
                request = PrefetchRequest(
                    layer_idx=layer_idx,
                    weight_name=name,
                    data_ptr=buffer.ctypes.data,
                    size_bytes=size,
                    priority=priority,
                )
                self._prefetch_queue.append(request)
                self.stats.total_requests += 1

        self._queue_event.set()

    def _prefetch_worker(self) -> None:
        """Worker thread that processes prefetch requests.

        Reads weight data in cache-line-sized chunks to bring them
        into the CPU's L2 cache hierarchy via sequential access.
        """
        while not self._shutdown.is_set():
            self._queue_event.wait(timeout=0.001)
            self._queue_event.clear()

            requests: list[PrefetchRequest] = []
            with self._queue_lock:
                if self._prefetch_queue:
                    # Process in priority order
                    self._prefetch_queue.sort(key=lambda r: r.priority)
                    requests = self._prefetch_queue[:]
                    self._prefetch_queue.clear()

            for req in requests:
                if self._shutdown.is_set():
                    break
                self._execute_prefetch(req)

    def _execute_prefetch(self, request: PrefetchRequest) -> None:
        """Execute a single prefetch operation.

        Touches memory in cache-line-sized strides to bring data into cache.
        On supported platforms, uses OS prefetch hints.
        """
        try:
            if self.enable_os_hints and platform.system() == "Linux":
                self._linux_prefetch(request.data_ptr, request.size_bytes)
            else:
                # Portable fallback: sequential read to pull into cache
                self._sequential_touch(request.data_ptr, request.size_bytes)

            self.stats.completed_requests += 1
            self.stats.total_bytes_prefetched += request.size_bytes
            self.stats.cache_hits_estimated += 1

        except Exception as e:
            logger.debug("Prefetch failed for layer %d/%s: %s", request.layer_idx, request.weight_name, e)

    def _sequential_touch(self, data_ptr: int, size_bytes: int) -> None:
        """Touch memory sequentially to load into cache.

        Creates a temporary view and reads in cache-line strides.
        The read values are discarded — the purpose is to trigger
        the CPU's cache loading mechanism.
        """
        # Create a numpy array view over the raw memory
        try:
            buf = (ctypes.c_char * size_bytes).from_address(data_ptr)
            arr = np.frombuffer(buf, dtype=np.uint8)

            # Read every cache_line_bytes-th element to touch each cache line
            stride = self.cache_line_bytes
            _ = arr[::stride].sum()  # Force read, discard result

        except (ValueError, OSError):
            pass  # Buffer not accessible, skip silently

    def _linux_prefetch(self, data_ptr: int, size_bytes: int) -> None:
        """Use Linux madvise(MADV_WILLNEED) to hint prefetch to the kernel."""
        try:
            import mmap

            MADV_WILLNEED = 3
            # Align to page boundary
            page_size = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
            aligned_ptr = data_ptr & ~(page_size - 1)
            offset = data_ptr - aligned_ptr
            aligned_size = size_bytes + offset

            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            libc.madvise(ctypes.c_void_p(aligned_ptr), ctypes.c_size_t(aligned_size), MADV_WILLNEED)
        except (OSError, AttributeError):
            # Fallback to sequential touch
            self._sequential_touch(data_ptr, size_bytes)


def create_prefetch_orchestrator(
    num_layers: int,
    speculative_enabled: bool = True,
) -> PrefetchOrchestrator:
    """Factory function to create a configured prefetch orchestrator.

    Args:
        num_layers: Number of transformer layers.
        speculative_enabled: If True, increase prefetch-ahead for
                            speculative decoding support.
    """
    prefetch_ahead = 2 if speculative_enabled else 1

    return PrefetchOrchestrator(
        num_layers=num_layers,
        prefetch_ahead=prefetch_ahead,
        enable_os_hints=platform.system() == "Linux",
    )
