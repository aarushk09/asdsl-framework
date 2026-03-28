# ASDSL Framework Implementation & Optimization Plan

This roadmap adapts our hardware-level optimization strategy to the current `asdsl` framework codebase. It breaks down the system engineering challenge into phases targeting theoretical maximum throughput on our target architecture (e.g., Alder Lake i7). 

**Core Issues to Resolve:**
*   **TTFT (Time To First Token) is ~16.7s:** Indicates sequential GEMV calls instead of batched GEMM for prefill.
*   **Decode is ~1.0 tok/s:** Indicates GEMV kernel memory bandwidth underutilization.

---

## Phase 1: Fix the Quantization Format (Q4_K_M)
Our current Q4 format (likely Q4_0 naive symmetric) bottlenecks the memory pipeline. We need a `Q4_K_M` super-block structure with 6-bit scales to optimize L1 cache prefetching.

**Modifications & Target Files:**
*   **`asdsl/quantization/` & `quantize_phi4_mock.py` / `quantize_tinyllama.py`**: Update the quantization logic to pack 8 x 6-bit scales into 12 bytes before weight data. 
*   Implement `BlockQ4K` structure with `d` (super-block scale), `dmin`, `scales[]`, and `qs[]`.
*   Establish offline repacking in our quantization scripts so that inference doesn't pay formatting penalties.

**🛑 TESTING CHECKPOINT 1:**
*   Run `python test_primitives.py` to ensure block encoding/decoding math is numerically stable.
*   Run `pytest tests/test_layer.py` (if available) to ensure single-layer forward passes still match expected output distributions.

---

## Phase 2: The Decode GEMV Kernel (AVX2 Intrinsics)
This is the hot path. We must dequantize Q4 nibbles directly in AVX2 registers and accumulate without writing floats to memory so we don't thrash the L1 cache.

**Modifications & Target Files:**
*   **`asdsl/kernels/` (e.g., C++ kernel integrations) & `test_avx2_math.py`**: Implement row-parallel OpenMP GEMV decoding.
*   Use `_mm256_cvtepu8_epi32`, `_mm256_fmadd_ps` to keep `fq_lo` and `fq_hi` exclusively in YMM registers.
*   Ensure the C++ extensions are compiled and linked properly in `setup.py` / `pyproject.toml`.

**🛑 TESTING CHECKPOINT 2:**
*   Run `python test_avx2_math.py` to validate AVX2 register math against pure Python/numpy reference implementations.
*   Run `python test_engine.py` to ensure the core inference loop doesn't crash with the new kernel.

---

## Phase 3: Obliterate the 16.7s TTFT — Batched Prefill GEMM
Process the input prompt prompt [T × d_model] against the weights [d_model × d_model] as a batched tiled GEMM, loading the 7GB weight matrix exactly once.

**Modifications & Target Files:**
*   **`asdsl/kernels/` & `asdsl/inference/`**: Add a tiled GEMM implementation tailored for L1/L2 target sizes (e.g., `TILE_M=4`, `TILE_N=32`, `TILE_K=256`).
*   Update the forward pass logic in `chat.py`, `run_inference.py`, and `asdsl/inference/` to route prompt tokens through the GEMM path rather than loop-based GEMV.

**🛑 TESTING CHECKPOINT 3:**
*   Run `benchmarks/bench_inference.py` and `run_phi4_benchmark.py`.
*   **Success Metric:** Pre-fill/TTFT should drop from ~16.7s down to < 2.0s.

---

## Phase 4: Thread Architecture (P-Cores vs E-Cores)
Alder Lake's hybrid architecture breaks naive OpenMP. E-cores lack AVX2 throughput and stall the reduction on shared locks. 

**Modifications & Target Files:**
*   **`asdsl/config.py` & `asdsl/kernels/` / C++ backend**: Add CPUID detection to pin threads specifically to P-cores for GEMV tasks.
*   Remove inner `omp barrier` calls; allow pipeline synchronization at layer boundaries instead.

**🛑 TESTING CHECKPOINT 4:**
*   Run `benchmarks/comprehensive_bench.py` monitoring per-core CPU usage (e.g., using `examples/system_info.py` or system tools). 
*   **Success Metric:** Decode should reach 3.5–4.0 tok/s.

---

## Phase 5: CPU Flash Attention + KV Cache Quantization
Tiled flash attention for the CPU avoids materializing the massive attention matrix. Quantizing the KV cache to int8 (Q8) halves memory bandwidth as the context length grows.

**Modifications & Target Files:**
*   **`asdsl/kernels/` (Attention logic) & `asdsl/memory/` / `kv_part1.cpp`, `kv_part2.cpp`**: 
*   Implement tiled attention (`BLOCK_Q=32`, `BLOCK_K=64`).
*   Integrate online softmax update.
*   Update `asdsl/memory/` and KV C++ files to store `int8_t` KV data alongside dynamic fp32 scaling factors.

**🛑 TESTING CHECKPOINT 5:**
*   Run `python test_attention.py` to ensure Flash Attention outputs match standard multi-head attention up to an epsilon.
*   Run context length tests in `evals/lm_eval_harness.py` to ensure no perplexity degradation at longer contexts.

---

## Phase 6: Weight Memory Layout — 4-Row Interleave
To maximize prefetcher efficiency, repack the weight format from standard row-major to a 4-row interleaved block layout offline. 

**Modifications & Target Files:**
*   **`export_mmap_model.py` / `asdsl/weight_store.py`**: Add an offline re-packing script (`repack_q4k_interleaved`).
*   Adjust the GEMV kernels to read from the interleaved format (loading 4 rows simultaneously).
*   Switch to explicit `PrefetchVirtualMemory()` via a dedicated prefetch thread instead of just relying on passive `MapViewOfFile()` page faults (integrate with `asdsl/prefetch/`).

**🛑 TESTING CHECKPOINT 6:**
*   Run `python test_mmap.py` to test memory mapping boundaries and ensure no segfaults.
*   Run `python benchmarks/run_all.py`. Check `perf stat --events=cache-misses,cache-references` locally if possible.

---

## Phase 7: Speculative Decoding
Draft multiple tokens using a smaller model (e.g., Phi-mini) to bypass the main model's memory bandwidth wall, verifying them in parallel.

**Modifications & Target Files:**
*   **`asdsl/speculative/` & `examples/speculative_decoding.py`**:
*   Implement the acceptance logic (speculative sampling).
*   Structure dual-model memory loading so the drafting model (fast ~800MB Q4 load) stays resident.
*   Verify draft strings via single batched target model passes.

**🛑 TESTING CHECKPOINT 7:**
*   Run `examples/speculative_decoding.py` explicitly.
*   Verify acceptance rate (target ~65–80%).
*   **Success Metric:** Effective decode speed should hit 7+ tok/s.

---
## Summary of Milestones

1.  **Format & Kernel (Phases 1-2):** 2.0-2.5 tok/s. 
2.  **Batched Prefill & Threads (Phases 3-4):** < 2s TTFT, 3.5-4.0 tok/s.
3.  **Attention & Interleave Layout (Phases 5-6):** 4.5-5.5 tok/s.
4.  **Speculative Decoding (Phase 7):** 7.0+ effective tok/s.