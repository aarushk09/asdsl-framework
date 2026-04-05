# ASDSL Framework V2: Experimental CPU Inference

**Asynchronous Salience-Driven Speculative Lookup Framework**

ASDSL Framework V2 is a research-oriented, experimental CPU inference stack for running large decoder-only models (like Microsoft Phi-4). Built to explore the real-world bottlenecks of CPU-based LLM inference, it combines **4-bit packed GEMV kernels**, **OpenMP thread tuning**, and **Quantization-Cascade Speculative Decoding (QCSD)**.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![Hardware](https://img.shields.io/badge/hardware-AVX2%20%7C%20OpenMP-red)](#)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

---

## 1. Project Philosophy & Reality

CPU LLM inference is fundamentally **memory-bound, not compute-bound**. The core truth of this project is exploring how to optimize memory movement and SIMD execution rather than just raw FLOPs. 

While ASDSL explores various advanced concepts like activation sparsity and variable-bitweight quantization, **we are currently trailing highly optimized production engines like `llama.cpp` by ~2.5x in throughput.** 

This framework is built for research, profiling, and understanding *why* certain theoretical optimizations (like skipping sparse computation on CPUs) often fail in practice due to hardware realities like branch misprediction and SIMD lane breaking.

### What Works (The Solid Engineering)
- **Packed Low-Bit GEMV:** Keeping weights in 4-bit and performing on-the-fly dequantization in registers. Utilizing AVX2 + FMA with packed weights is standard, legitimate engineering for CPU inference.
- **Speculative Decoding Architecture (QCSD):** Implementing a draft-then-verify architecture (similar to Leviathan / Google Research).
- **Strict Thread Control:** Preventing OpenMP/PyTorch thread oversubscription. Pinning workloads to physical P-cores to avoid L2 cache thrashing and context switching.
- **Isolated Benchmarking:** Clean separation between PyTorch baseline, Native Kernel, and Speculative Decoding modes for accurate profiling.

### The Experimental Realities (What We're Still Solving)
- **Activation Sparsity (FatReLU):** While neural networks exhibit natural sparsity, exploiting this on standard CPUs is incredibly difficult. Our own attempts to dynamically skip `0.0` vectors during matrix multiplication (ALU bypass) actually *destroyed* throughput (dropping to 1.08 tok/s). Zeroing values does not skip computation unless memory loads are avoided entirely.
- **Transposed Sparse GEMV (Future Work):** The planned path forward is applying offline weight transposition to avoid DRAM fetches entirely for sparse activations. This requires structured sparsity and careful cache tiling.
- **Variable-Bit Quantization (ASB):** Mixing 2/3/4/8-bit blocks complicates decoding logic and hurts SIMD regularity. While interesting for research, uniform Q4/Q5 often wins in practical CPU implementations.
- **Speculative Decoding Acceptance Rates:** Our experimental QCSD implementations currently struggle with low acceptance rates (~7%), making the overhead of drafting often outweigh the benefits.

---

## 2. Current Performance Benchmarks (April 2026)

**Target Model:** microsoft/phi-4 (14B Parameters)  
**Quantization:** Q4 (4-bit Group Size 32)  
**Hardware:** Intel Core i7/i9 equivalent (12 P-Cores, AVX2/VNNI, 24.0 GB/s Peak Bandwidth)

| Framework State               | Decoding Throughput | Perplexity (WikiText-103) | Peak Memory | Accelerator |
|-------------------------------|---------------------|---------------------------|-------------|-------------|
| **Native C++ Unified Engine (Baseline)** | **2.56 tok/s** | **19.16** | **3.2 GB** | AVX2/VNNI |
| ASDSL Python Prototype (Favorable Peak) | 2.85 tok/s       | 19.16 | 3.2 GB | AVX2 |
| *llama.cpp Q4_K_M (Local Reference)*| *~2.72 tok/s*       | *~18-20* | *~3.0 GB* | *AVX2/VNNI* |

*Note: The native ASDSL engine currently trails llama.cpp by ~6% in stable runs. While legacy measurements of the Python prototype showed peaks up to ~2.85 tok/s under favorable conditions, this is recognized as an anomaly to investigate (hidden caching effects, uncounted overhead, lucky scheduling) rather than a repeatable victory. The definitive throughput of the C++ pipeline is 2.56 tok/s.*

---

## 3. Technical Architecture 

| Layer | Role |
|--------|------|
| **`experiments/phi4_cpu_run.py`** | End-to-end Python/PyTorch inference loop used for baseline validation and prototyping new speculative or quantization concepts. |
| **`asdsl/kernels/native/*.cpp`** | Native C++ modules compiled via PyBind11 (`_native_gemv`, `_native_forward`, etc.) enabling AVX2/AVX2_VNNI FMA instructions on packed int4 data. |
| **`scripts/run_full_benchmark.py`** | Strict telemetry and comparison harness for testing Python baselines against native extensions. |
| **`asdsl/speculative/`** | Dual-model / simulated speculative decoding logic (QCSD) including acceptance rate telemetry. |

---

## 4. Getting Started

### Prerequisites
- Python 3.10+
- A C++ compiler supporting `std=c++17` & OpenMP (`cmake`, MSVC, or GCC).
- An AVX2-capable CPU (Intel Core 4th-gen+ or AMD Ryzen).

### Building the Native Extensions
To bypass the slow Python baseline, compile the PyBind11 C++ routines:
```bash
python setup.py build_ext --inplace
```

### Running Benchmarks
To run the full comparative benchmark suite (testing the Native C++ Engine against the Python prototype, baseline, and the exact `llama.cpp` local reference numbers on the same hardware context), use the unified script:
```bash
python run_full_benchmark.py
```
*(Results are logged directly to standard output, simulating the TTFT delays and generation constraints verified during the Phase 22 profiling of the 14B model)*

To run a **strict, 1:1 comparable bulletproof benchmark** against the live `C++` pipeline locally under tightly bound parameters (fixed prompt, max core affinity, N=5 iterations, greedy decoding):
```bash
python strict_benchmark_harness.py
```
*Note: `strict_benchmark_harness.py` also outputs the exact 1:1 `llama-cli` argument string required to replicate the baseline test in `llama.cpp` (including disabling hyperthreading, batch sizes, and `-n` tokens).*

---

## 5. Roadmap: The Path to Competitiveness

Our primary roadmap goal is closing the 2.5x gap with established engines like `llama.cpp` through rigorous profiling and architecture simplification.

1. **Deep Profiling:** Moving away from ad-hoc timings and utilizing hardware profilers (VTune, `perf`) to track cache thrashing, NUMA faults, and pipeline stalls.
2. **Simplification:** Pausing complex variables (like mixed-precision ASB blocks) to focus exclusively on maxing out memory bandwidth for standard uniform Q4 quantization.
3. **Addressing Speculative Overhead:** Redesigning the QCSD verification logic. A 7% acceptance rate means draft modeling is currently dead weight. We need to either improve the draft model alignment or abandon speculation in favor of pure dense throughput.
4. **Offline Structured Sparsity:** Continuing the FatReLU experiment correctly. Instead of pointer-chasing at runtime, we will apply offline weight reorganizations (`down_proj_T`) to ensure DRAM fetches are entirely skipped when an activation is zero.

---
*Disclaimer: ASDSL is an experimental codebase. It recombines known inference techniques (AVX2 GEMV, Quantization, Speculative Decoding) to explore their limitations and realities on standard CPU hardware.*
