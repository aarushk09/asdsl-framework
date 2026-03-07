"""End-to-end inference benchmarks: token throughput, latency, memory usage."""

import time
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asdsl.config import InferenceConfig, PHI3_MINI_CONFIG
from asdsl.speculative.swift import SWIFTDecoder, SkipSchedule, create_skip_schedule_for_phi3
from asdsl.prefetch.orchestrator import PrefetchOrchestrator
from asdsl.memory.manager import MemoryManager
from asdsl.inference.kv_cache import BlockSparseKVCache, KVCacheConfig
from asdsl.kernels.simd import select_backend, fma_vnni_int8, lut_shuffle_avx2


# ---------------------------------------------------------------------------
# Simulated layer executor for benchmarking (no real model weights)
# ---------------------------------------------------------------------------

class SimulatedLayerExecutor:
    """Simulates transformer layer execution with realistic compute patterns."""

    def __init__(self, num_layers: int, hidden_dim: int, vocab_size: int):
        self._num_layers = num_layers
        self._hidden_dim = hidden_dim
        self._vocab_size = vocab_size
        # Pre-allocate random "weight" matrices for realistic memory patterns
        self._proj = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.02

    @property
    def num_layers(self) -> int:
        return self._num_layers

    def execute_layer(self, layer_idx: int, hidden_state: np.ndarray) -> np.ndarray:
        # Simulate QKV projection + feed-forward with actual computation
        h = hidden_state @ self._proj[:hidden_state.shape[-1], :hidden_state.shape[-1]]
        return h

    def execute_lm_head(self, hidden_state: np.ndarray) -> np.ndarray:
        if hidden_state.ndim == 1:
            logits = np.random.randn(self._vocab_size).astype(np.float32)
        else:
            logits = np.random.randn(hidden_state.shape[0], self._vocab_size).astype(np.float32)
        return logits


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def bench_kv_cache_throughput(
    seq_lengths: list[int] | None = None,
    repeats: int = 5,
) -> None:
    """Benchmark KV cache append + eviction throughput."""
    if seq_lengths is None:
        seq_lengths = [128, 512, 1024, 2048]

    print("=" * 72)
    print("KV Cache Throughput Benchmark")
    print("=" * 72)
    print(f"{'Seq Len':>8s}  {'Append (µs/tok)':>16s}  {'Evict (µs/tok)':>16s}")
    print("-" * 48)

    for seq_len in seq_lengths:
        cfg = KVCacheConfig(
            num_layers=32, num_kv_heads=8, head_dim=96,
            max_blocks=seq_len // 16 + 1, block_size=16,
        )
        cache = BlockSparseKVCache(cfg)

        # Append benchmark
        times = []
        for _ in range(repeats):
            cache_copy = BlockSparseKVCache(cfg)
            t0 = time.perf_counter()
            for _ in range(seq_len):
                k = np.random.randn(32, 8, 96).astype(np.float32)
                v = np.random.randn(32, 8, 96).astype(np.float32)
                cache_copy.append(k, v)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        median_s = sorted(times)[len(times) // 2]
        us_per_tok = (median_s / seq_len) * 1e6

        # Eviction benchmark (fill past capacity)
        evict_cfg = KVCacheConfig(
            num_layers=32, num_kv_heads=8, head_dim=96,
            max_blocks=seq_len // 32 + 1, block_size=16,  # Smaller capacity
        )
        evict_cache = BlockSparseKVCache(evict_cfg)
        times_e = []
        for _ in range(repeats):
            ec = BlockSparseKVCache(evict_cfg)
            t0 = time.perf_counter()
            for _ in range(seq_len):
                k = np.random.randn(32, 8, 96).astype(np.float32)
                v = np.random.randn(32, 8, 96).astype(np.float32)
                ec.append(k, v)
            t1 = time.perf_counter()
            times_e.append(t1 - t0)

        median_e = sorted(times_e)[len(times_e) // 2]
        us_per_tok_e = (median_e / seq_len) * 1e6

        print(f"{seq_len:>8d}  {us_per_tok:>16.1f}  {us_per_tok_e:>16.1f}")

    print()


def bench_speculative_decoding(
    num_layers: int = 32,
    hidden_dim: int = 256,
    vocab_size: int = 32000,
    num_steps: int = 50,
) -> None:
    """Benchmark SWIFT speculative decoding speedup."""
    print("=" * 72)
    print("SWIFT Speculative Decoding Benchmark")
    print("=" * 72)

    executor = SimulatedLayerExecutor(num_layers, hidden_dim, vocab_size)

    # Full autoregressive baseline
    hidden = np.random.randn(hidden_dim).astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(num_steps):
        for layer in range(num_layers):
            hidden = executor.execute_layer(layer, hidden)
        logits = executor.execute_lm_head(hidden)
        _ = int(np.argmax(logits))
    t_full = time.perf_counter() - t0

    # Speculative decoding
    decoder = SWIFTDecoder(
        executor=executor,
        num_draft_tokens=4,
        keep_first=4,
        keep_last=4,
        adaptive_schedule=True,
    )

    hidden = np.random.randn(hidden_dim).astype(np.float32)
    total_tokens = 0
    t0 = time.perf_counter()
    for _ in range(num_steps):
        result = decoder.speculative_step(hidden, past_tokens=[0])
        total_tokens += len(result.accepted_tokens)
    t_spec = time.perf_counter() - t0

    print(f"Full autoregressive: {num_steps} steps in {t_full*1000:.1f} ms "
          f"({num_steps/t_full:.1f} tok/s)")
    print(f"Speculative (SWIFT): {total_tokens} tokens in {t_spec*1000:.1f} ms "
          f"({total_tokens/t_spec:.1f} tok/s)")
    print(f"Speedup: {(total_tokens/t_spec) / (num_steps/t_full):.2f}x")
    schedule = decoder.schedule
    print(f"Skip ratio: {schedule.skip_ratio:.1%}, "
          f"Draft layers: {len(schedule.draft_layers)}/{schedule.total_layers}")
    print()


def bench_kernel_throughput(
    sizes: list[int] | None = None,
    repeats: int = 20,
) -> None:
    """Benchmark SIMD kernel emulation throughput."""
    if sizes is None:
        sizes = [256, 512, 1024, 2048]

    print("=" * 72)
    print("SIMD Kernel Throughput Benchmark")
    print("=" * 72)
    backend = select_backend()
    print(f"Selected backend: {backend.name}")
    print(f"{'Size':>6s}  {'LUT shuffle (µs)':>18s}  {'INT8 FMA (µs)':>16s}")
    print("-" * 48)

    for sz in sizes:
        # LUT shuffle
        lut = np.random.randn(16).astype(np.float32)
        indices = np.random.randint(0, 16, size=sz, dtype=np.uint8)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            lut_shuffle_avx2(lut, indices)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        lut_us = sorted(times)[len(times) // 2] * 1e6

        # INT8 FMA
        a = np.random.randint(-128, 127, size=(sz, sz), dtype=np.int8)
        b = np.random.randint(-128, 127, size=(sz, 16), dtype=np.int8)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fma_vnni_int8(a, b)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        fma_us = sorted(times)[len(times) // 2] * 1e6

        print(f"{sz:>6d}  {lut_us:>18.1f}  {fma_us:>16.1f}")

    print()


def bench_memory_manager() -> None:
    """Benchmark memory allocation and pinning overhead."""
    print("=" * 72)
    print("Memory Manager Benchmark")
    print("=" * 72)

    mm = MemoryManager(enable_huge_pages=False, enable_pinning=False)

    sizes_mb = [1, 10, 50, 100]
    print(f"{'Size (MB)':>10s}  {'Alloc (ms)':>12s}")
    print("-" * 28)

    for size_mb in sizes_mb:
        n_elements = (size_mb * 1024 * 1024) // 4  # float32
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            region = mm.allocate_for_weights(
                f"bench_{size_mb}", shape=(n_elements,), dtype=np.float32
            )
            t1 = time.perf_counter()
            times.append(t1 - t0)
            mm.release(f"bench_{size_mb}")

        median_ms = sorted(times)[len(times) // 2] * 1000
        print(f"{size_mb:>10d}  {median_ms:>12.2f}")

    mm.release_all()
    print()


def bench_prefetch_orchestrator() -> None:
    """Benchmark prefetch orchestrator overhead per layer."""
    print("=" * 72)
    print("Prefetch Orchestrator Benchmark")
    print("=" * 72)

    num_layers = 32
    orch = PrefetchOrchestrator(num_layers=num_layers, lookahead=2)

    # Register buffers
    for i in range(num_layers):
        orch.register_weight_buffer(i, "qkv", np.zeros(3072 * 3072, dtype=np.float32))
        orch.register_weight_buffer(i, "ffn", np.zeros(3072 * 8192, dtype=np.float32))

    orch.start()

    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        for layer in range(num_layers):
            orch.notify_layer_start(layer)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    orch.stop()

    median_ms = sorted(times)[len(times) // 2] * 1000
    per_layer_us = (median_ms / num_layers) * 1000
    stats = orch.get_stats()

    print(f"Notification overhead per layer: {per_layer_us:.1f} µs")
    print(f"Total for {num_layers} layers: {median_ms:.2f} ms")
    print(f"Prefetch requests issued: {stats['prefetch_requests']}")
    print()


def main() -> None:
    print("ASDSL Framework — End-to-End Benchmarks")
    print("=" * 72)
    print()

    bench_kv_cache_throughput(seq_lengths=[128, 512])
    bench_speculative_decoding(num_layers=16, hidden_dim=128, vocab_size=1000, num_steps=20)
    bench_kernel_throughput(sizes=[256, 512])
    bench_memory_manager()
    bench_prefetch_orchestrator()

    print("All inference benchmarks complete.")


if __name__ == "__main__":
    main()
