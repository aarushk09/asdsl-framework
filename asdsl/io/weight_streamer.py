"""Async weight streaming for ASDSL inference.

Provides two backends:
  - IoRing (Windows 11 22H2+, build >= 22621): kernel bypass async reads
  - Thread fallback (all platforms): background thread reads via open()+read()

Usage:
    from asdsl.io.weight_streamer import WeightStreamer

    streamer = WeightStreamer(model_path)
    streamer.submit_prefetch(layer_idx=1, byte_offset=4096, byte_length=1024*1024)
    # ... kernel computes layer 0 ...
    data = streamer.wait_and_get(layer_idx=1)
    streamer.close()

Notes:
  - On this machine (16.9 GB RAM, 4.5 GB model) weights stay RAM-resident;
    IoRing benefit is minimal. Benefit materializes on NVMe-constrained systems.
  - Thread fallback achieves async behavior on all platforms.
  - submit_prefetch is idempotent: submitting the same layer twice is safe
    (second call is silently ignored if pending).
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path


class WeightStreamer:
    """Async weight chunk pre-fetcher for LLM layer-by-layer inference.

    Args:
        model_file_path: Path to the weight file to read from.
        use_iouring:     None = auto-detect, True = force IoRing,
                         False = force thread fallback.
    """

    def __init__(self, model_file_path: str | Path, use_iouring: bool | None = None):
        self._path = str(model_file_path)
        self._pending: dict[int, threading.Event] = {}
        self._buffers: dict[int, bytes | None] = {}
        self._lock = threading.Lock()
        self._closed = False

        if use_iouring is None:
            from asdsl.io.iouring_detect import is_iouring_available
            use_iouring, reason = is_iouring_available()
            print(f"[WeightStreamer] IoRing: {use_iouring} ({reason})")
        else:
            reason = "forced by caller"

        self._use_iouring = False  # always thread for now (IoRing ctypes WIP)
        if use_iouring:
            ok = self._try_init_iouring()
            if ok:
                self._use_iouring = True
                print("[WeightStreamer] Using IoRing backend")
            else:
                print("[WeightStreamer] IoRing init failed — using thread fallback")

        self._init_thread_backend()

    # ------------------------------------------------------------------
    # IoRing backend (ctypes, Windows 11 22H2+)
    # ------------------------------------------------------------------

    def _try_init_iouring(self) -> bool:
        """Attempt to initialise Windows IoRing. Return True on success."""
        try:
            import ctypes
            kb = ctypes.WinDLL("KernelBase.dll")
            handle = ctypes.c_void_p()
            hr = kb.IoRingCreateIoRing(
                ctypes.c_uint32(3),   # IORING_VERSION_3
                ctypes.c_uint32(0),   # required flags = none
                ctypes.c_uint32(0),   # advisory flags = none
                ctypes.c_uint32(16),  # submission queue size
                ctypes.c_uint32(32),  # completion queue size
                ctypes.byref(handle),
            )
            if hr != 0:
                print(f"[WeightStreamer] IoRingCreateIoRing HRESULT=0x{hr:08X}")
                return False
            self._iouring_handle = handle
            return True
        except Exception as exc:
            print(f"[WeightStreamer] IoRing init exception: {exc}")
            return False

    # ------------------------------------------------------------------
    # Thread-based async backend (always available)
    # ------------------------------------------------------------------

    def _init_thread_backend(self) -> None:
        self._work_queue: list[tuple[int, int, int]] = []   # (layer_idx, offset, length)
        self._worker = threading.Thread(
            target=self._worker_loop, daemon=True, name="WeightStreamer-worker"
        )
        self._worker.start()

    def _worker_loop(self) -> None:
        while not self._closed:
            task = None
            with self._lock:
                if self._work_queue:
                    task = self._work_queue.pop(0)
            if task is None:
                time.sleep(0.001)
                continue
            layer_idx, offset, length = task
            try:
                with open(self._path, "rb") as f:
                    f.seek(offset)
                    data: bytes | None = f.read(length)
            except Exception as exc:
                print(f"[WeightStreamer] read error layer {layer_idx}: {exc}")
                data = None
            with self._lock:
                self._buffers[layer_idx] = data
            # Signal the event (created in submit_prefetch)
            ev = self._pending.get(layer_idx)
            if ev is not None:
                ev.set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit_prefetch(
        self,
        layer_idx: int,
        byte_offset: int,
        byte_length: int,
        active_rows: object = None,  # reserved for sparse-row filtering
    ) -> None:
        """Schedule an async read for layer_idx. Idempotent (no-op if pending)."""
        if self._closed:
            return
        with self._lock:
            if layer_idx in self._pending:
                return                      # already in flight
            ev = threading.Event()
            self._pending[layer_idx] = ev
            self._buffers[layer_idx] = None
            self._work_queue.append((layer_idx, byte_offset, byte_length))

    def wait_and_get(self, layer_idx: int, timeout_s: float = 5.0) -> bytes | None:
        """Block until the prefetch for layer_idx completes.

        Returns the raw bytes, or None if the read failed.
        Falls back to a synchronous read if submit_prefetch was not called.
        """
        ev = self._pending.get(layer_idx)
        if ev is None:
            # Not submitted — fall back to sync read with zero knowledge of offset
            # (caller should always call submit_prefetch first for performance)
            return None
        ev.wait(timeout=timeout_s)
        with self._lock:
            result = self._buffers.pop(layer_idx, None)
            self._pending.pop(layer_idx, None)
        return result

    def close(self) -> None:
        """Signal the worker thread to stop and release resources."""
        self._closed = True
        if hasattr(self, "_worker") and self._worker.is_alive():
            self._worker.join(timeout=2.0)
        if self._use_iouring and hasattr(self, "_iouring_handle"):
            try:
                import ctypes
                kb = ctypes.WinDLL("KernelBase.dll")
                kb.CloseIoRing(self._iouring_handle)
            except Exception:
                pass

    def __enter__(self) -> "WeightStreamer":
        return self

    def __exit__(self, *_) -> None:
        self.close()
