# ASDSL Framework

**Asynchronous Salience-Driven Speculative Lookup Framework**

A CPU-focused Phi-4 inference and quantization project centered on
weight compression, native AVX2 GEMV kernels, and benchmarking on
Intel i7 Evo hardware. The current measured state is a strong Tier 1
result set for 8-bit, 4-bit, and 3-bit inference; Tier 2 (`QCSD`) and
Tier 3 (activation-sparse GEMV) exist as experimental paths and are
not yet reflected in the stable benchmark table below.

[![Tests](https://img.shields.io/badge/tests-95%2F95%20passing-brightgreen)](#testing)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

---

## Table of Contents

- [What ASDSL Does](#what-asdsl-does)
- [Measured Results](#measured-results)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Running Phi-4 Inference](#running-phi-4-inference)
- [Quantization API](#quantization-api)
- [Perplexity Evaluation](#perplexity-evaluation)
- [Benchmarks & Visualizations](#benchmarks--visualizations)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [References](#references)

---

## What ASDSL Does

ASDSL is a collection of quantization, inference, and benchmarking components
for CPU-side Phi-4 experiments. The repository currently contains a mix of
stable measured paths and experimental work-in-progress paths:

| Subsystem | What it does |
|-----------|-------------|
| **MSE-Optimal Quantization** | Grid-searches per-group clipping ratios to minimize reconstruction error at float16 storage precision. No calibration data required. |
| **SpQR Outlier Separation** | Experimental low-bit outlier handling for 3-bit and 2-bit quantization. Helpful for 3-bit in current measurements; 2-bit remains unstable. |
| **AVX2 SIMD GEMV Kernels** | Native C++/AVX2+FMA fused dequant-matmul kernels for 2-bit, 3-bit, 4-bit, and 8-bit. These are real compiled extensions, not just Python stubs. |
| **SWIFT Self-Speculative Decoding** | The existing `asdsl/speculative/swift.py` module implements layer-skipping self-speculative decoding as a separate subsystem. |
| **QCSD Speculative Decoding** | Experimental path in `experiments/phi4_cpu_run.py` using low-bit and higher-bit weight banks as draft/verifier. Present, but not yet benchmark-stable. |
| **Activation-Sparse GEMV** | Experimental path for skipping low-magnitude MLP activations. Present, but not yet benchmark-stable. |
| **Salience-Driven Mixed-Precision** | Assigns higher bit-widths to weight groups that have the largest impact on output quality, guided by Hessian/Fisher-diagonal salience scores. |
| **LUT Matmul Engine** | Replaces dequantize-then-multiply with precomputed partial-sum table lookups (PSHUF/TBL). 2-bit LUT fits in L2 cache (~96 KB/layer). |
| **Block-Sparse KV Cache** | Demand-allocated 16-token blocks with importance-based eviction. Avoids pre-allocating for max_seq_len. |

---

## Measured Results

> Hardware: Intel Core i7 Evo · CPU-only · 16 GB DDR4-3200 RAM · Windows  
> Model: **Phi-4-multimodal-instruct** (14B params, 200k vocab, 32 layers)  
> Source of truth: `benchmarks/results/benchmark_results.json`

### Stable Benchmark Snapshot

The table below reflects the latest consistent measured results currently saved in
`benchmarks/results/benchmark_results.json`. These are the numbers that should be
used when discussing current repository performance.

| Configuration | Perplexity ↓ | Speed | RAM Peak | SNR (dB) | Compression | Cores | Backend |
|---------------|:------------:|:-----:|:--------:|:--------:|:-----------:|:-----:|:-------:|
| **FP16 baseline** | **11.06** | 1.85 tok/s | 7.50 GB | inf | 1.0x | 6 | chunked fp16 matvec |
| **ASDSL 8-bit** | **11.06** | **2.52 tok/s** | 6.27 GB | 43.8 | 3.9x | 6 | native AVX2 GEMV |
| **ASDSL 4-bit** | **11.89** | **2.74 tok/s** | 6.72 GB | 22.5 | 6.4x | 6 | native AVX2 GEMV |
| **ASDSL 3-bit** | **14.14** | **2.21 tok/s** | 7.14 GB | 17.7 | 6.2x | 7 | native AVX2 GEMV + SpQR |

Current interpretation:

- `8-bit` is effectively lossless relative to fp16 on this benchmark slice.
- `4-bit` is the best current quality/speed tradeoff in the stable measured path.
- `3-bit` is now usable and much faster than the original PyTorch fallback path.
- `2-bit` remains experimental and is intentionally excluded from the stable table because quality is still inconsistent.

### What Is Measured vs Experimental

Measured and stable today:

- native AVX2 GEMV for 8-bit, 4-bit, and 3-bit
- SpQR-style outlier handling helping the 3-bit path
- end-to-end Phi-4 CPU inference and WikiText-2 perplexity benchmarking

Implemented but still experimental:

- `QCSD` path in `experiments/phi4_cpu_run.py`
- activation-sparse GEMV path in `experiments/phi4_cpu_run.py`
- 2-bit draft / verification experiments

These experimental paths should not be treated as production benchmark results until
their acceptance rates, quality, and end-to-end throughput are validated in the same
benchmark workflow as the stable table above.

### About The 10 tok/s Goal

The often-cited `~10.6 tok/s` figure is a **heuristic bandwidth upper bound**, not a
physical limit or a full-system performance model. It comes from dividing approximate
memory bandwidth by approximate model footprint, and it ignores several real costs:

- KV cache traffic
- attention and softmax work
- memory latency and cache-miss behavior
- LM head cost
- Python orchestration overhead

It is still useful as a rough directional target, but it should be read as
\"order-of-magnitude headroom\" rather than a hard ceiling.

### Memory Footprint

| Configuration | Peak RAM |
|---------------|:--------:|
| FP16 | 7.50 GB |
| 8-bit | 6.27 GB |
| 4-bit | 6.72 GB |
| 3-bit | 7.14 GB |

### Historical Note

Older benchmark numbers that previously appeared in this README came from earlier runs,
different evaluator settings, and older inference paths. They have been removed here to
avoid mixing incomparable results in the same document.

### Output Quality Verification

```
$ python experiments/phi4_cpu_run.py --bits 4 --prompt "The capital of France is"
→ "The capital of France is Paris."  ✅

$ python experiments/phi4_cpu_run.py --bits 3 --prompt "The capital of France is"
→ "The capital of France is Paris. Paris is the capital city of France
   and is known for its rich history, culture, and architecture."  ✅
```

### Inference Engine Optimizations

| Optimization | Impact |
|-------------|--------|
| **AVX2 GEMV Q2/Q3/Q4/Q8** | Native C++/AVX2+FMA fused dequant-matmul for all bit-widths. Bypasses PyTorch entirely. |
| **SpQR outlier separation** | Low-bit outlier handling for experimental 3-bit and 2-bit paths. Beneficial for 3-bit in current measurements; 2-bit still needs more work. |
| **QCSD speculative decoding** | Experimental 2-bit draft / 4-bit verify path in `phi4_cpu_run.py`. Not yet included in the stable benchmark table. |
| **Activation-sparse GEMV** | Experimental bitmask-based down-projection skipping path. Not yet included in the stable benchmark table. |
| **Dual-path matvec** | bits=16: chunked f16→f32 reads. bits<=8: chunked uint8 dequant+mv with in-place scale/bias. |
| **Pre-unpacked uint8 weights** | Unpacks packed data to 1-byte-per-value at load time for fast streaming reads. |
| **Pre-computed bias** | Stores `bias = -zero * scale` per group; fuses `val * scale + bias` instead of `(val - zero) * scale`. |
| **Float16 embedding & LM head** | Stores embedding/LM-head as f16 instead of f32, saving 1.2 GB RAM. |
| **Pre-allocated KV cache** | Contiguous torch tensors per layer with snapshot/restore for QCSD. |
| **SDPA attention** | `torch.nn.functional.scaled_dot_product_attention` replaces manual Q·K·V loops. |
| **Intel i7 Evo thread tuning** | Defaults to 8 threads (P-cores only). MKL preferred over OpenBLAS. |
| **`torch.inference_mode()`** | Disables autograd bookkeeping during decode and prefill. |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ASDSL Framework                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Weights (FP32/FP16)                                      │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────────────────────────────────────────┐       │
│  │              Quantization Pipeline                    │      │
│  │  1. Salience scoring (Hessian diagonal)               │      │
│  │  2. Bit allocation per group                          │      │
│  │  3. MSE-optimal clipping (grid search, 9 ratios)      │      │
│  │  4. SpQR outlier separation (≤3-bit: 3.5σ threshold)  │      │
│  │  5. Asymmetric quant ≤4-bit / symmetric >4-bit        │      │
│  │  6. Fine-grained groups (gs=16/32/128)                │      │
│  └─────────────────────┬────────────────────────────────┘       │
│                        │                                        │
│       ┌────────────────┼───────────────────┐                    │
│       ▼                ▼                   ▼                    │
│  ┌──────────┐    ┌──────────┐       ┌───────────┐               │
│  │  Weight  │    │ AVX2     │       │  Sparse   │               │
│  │  Store   │───▶│ GEMV     │       │  GEMV     │              │
│  │ (N-bit)  │    │ Kernels  │       │  (Tier 3) │               │
│  │ Dual-bank│    │ Q2/Q3/   │       │  bitmask  │               │
│  │ (QCSD)   │    │ Q4/Q8    │       │  skip     │               │
│  └──────────┘    └────┬─────┘       └─────┬─────┘               │
│                       │                   │                     │
│                       ▼                   ▼                     │
│                ┌────────────────────────────────┐               │
│                │      Inference Engine          │               │
│                │  RMSNorm → Attn → MLP loop     │               │
│                │  BlockSparseKVCache tracking    │              │
│                │  Activation-sparse down_proj    │              │
│                └────────────────┬───────────────┘               │
│                                 │                               │
│                    ┌────────────┴────────────┐                  │
│                    ▼                         ▼                  │
│             ┌─────────────┐          ┌─────────────────┐        │
│             │    QCSD     │          │  OS Memory Mgr  │        │
│             │  Specul.    │          │  mlock + Huge   │        │
│             │  Decoder    │          │  Pages + NUMA   │        │
│             │  (Tier 2)   │          └─────────────────┘        │
│             │ 2-bit draft │                                     │
│             │ 4-bit verify│                                     │
│             └─────────────┘                                     │
└─────────────────────────────────────────────────────────────────┘
```

### SWIFT vs QCSD

There are **two different speculative-decoding ideas** in this repository:

- **SWIFT** lives in `asdsl/speculative/swift.py` and implements layer-skipping self-speculative decoding.
- **QCSD** lives in `experiments/phi4_cpu_run.py` and uses different quantization banks as draft/verifier.

They are not the same technique and should not be conflated.

### QCSD — Experimental Quantization-Cascade Speculative Decoding

The QCSD path is an experimental decode loop that tries to use a low-bit bank as the
draft path and a higher-bit bank as the verifier. The implementation exists, but the
repository does **not** currently claim stable measured QCSD speedups in the main
benchmark table.

### Activation-Sparse GEMV — Experimental

The repository also contains an experimental activation-sparse GEMV path for the MLP
down-projection. The intuition is sound, but the current implementation is still being
validated for end-to-end speedup and has not been promoted into the stable benchmark path.

---

## Quick Start

### Prerequisites

- Python 3.10 or higher
- 10+ GB free disk space (for Phi-4 model weights)
- 12+ GB RAM recommended for Phi-4; 8 GB minimum

### Installation

```bash
git clone https://github.com/aarushk09/asdsl-framework.git
cd asdsl-framework
pip install -e ".[dev]"
```

**Core dependencies** (installed automatically):

| Package | Purpose |
|---------|---------|
| `numpy >= 1.24` | Array operations, quantization math |
| `torch >= 2.1` | Tensor ops, thread control |
| `safetensors >= 0.4` | Loading Phi-4 model shards |
| `transformers >= 4.36` | Tokenizer |
| `psutil >= 5.9` | RAM and CPU monitoring |

**Optional:**

```bash
pip install datasets          # WikiText-2 perplexity evaluation
pip install lm-eval           # lm-evaluation-harness benchmarks
pip install matplotlib        # Benchmark visualization charts
```

---

## Running Phi-4 Inference

### Download the Model

ASDSL uses Phi-4's safetensors weights loaded directly from disk. Place model files at:

```
models/phi4-multimodal-instruct/
  model.safetensors.index.json
  model-00001-of-00006.safetensors
  model-00002-of-00006.safetensors
  ...
  tokenizer.json
  tokenizer_config.json
```

### Basic Inference

```bash
# FP16 baseline (no quantization)
python experiments/phi4_cpu_run.py --bits 16

# 4-bit (recommended — near-baseline quality, 6.4x compression)
python experiments/phi4_cpu_run.py --bits 4

# 4-bit with experimental paths enabled
python experiments/phi4_cpu_run.py --bits 4 --qcsd --sparse

# 3-bit with SpQR outlier separation
python experiments/phi4_cpu_run.py --bits 3

# 2-bit with SpQR outlier separation
python experiments/phi4_cpu_run.py --bits 2

# 8-bit (lossless quality)
python experiments/phi4_cpu_run.py --bits 8
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--bits N` | `16` | Quantization bit-width (2, 3, 4, 8, or 16 for FP16) |
| `--prompt TEXT` | `"What is 2+2?"` | Input prompt |
| `--max-new-tokens N` | `80` | Number of tokens to generate |
| `--threads N` | `8` | CPU thread count (8 = Intel i7 Evo P-cores) |
| `--group-size N` | `0` (auto) | Quantization group size; 0 = smart default |
| `--qcsd` | off | Enable experimental QCSD speculative decoding path |
| `--draft-bits N` | `2` | Bit-width for QCSD draft model |
| `--draft-k N` | `7` | Draft tokens per QCSD cycle |
| `--sparse` | off | Enable experimental activation-sparse GEMV path |
| `--sparse-threshold F` | `0.01` | Activation sparsity threshold |

### Smart Group Size Defaults

When `--group-size 0` (the default), ASDSL automatically picks the best group size for the
given bit-width:

| Bits | Auto group size | Rationale |
|-----:|:---------------:|-----------|
| ≤ 3  | 16 | Fine-grained groups needed to prevent catastrophic error at 8 quant levels |
| 4    | 32 | Balances quality vs. scale overhead |
| > 4  | 128 | Standard; 256 quant levels tolerate larger groups |

### Thread Control

```python
# In your own script — control CPU usage before importing numpy
from experiments.phi4_cpu_run import set_thread_count

set_thread_count(4)   # Limits OMP, MKL, OpenBLAS, VECLIB, NUMEXPR, and torch threads
```

Or use the CLI flag: `python experiments/phi4_cpu_run.py --threads 2`

---

## Quantization API

### Core Primitives (`asdsl.quantization.core`)

```python
from asdsl.quantization.core import (
    quantize_weights,
    dequantize_weights,
    compute_scale_zero,
    compute_quantization_error,
    QuantizedTensor,
)
import numpy as np

weights = np.random.randn(4096, 4096).astype(np.float32)

# Quantize to 4-bit with MSE-optimal clipping (recommended)
qtensor = quantize_weights(
    weights,
    bits=4,
    group_size=32,      # use 32 for 4-bit (or 0 for auto)
    symmetric=False,    # asymmetric — better for ≤4-bit
    optimize_clips=True # grid search for best clipping ratio
)

# Reconstruct
reconstructed = dequantize_weights(qtensor)

# Check quality
snr_db, mse = compute_quantization_error(weights, reconstructed)
print(f"SNR: {snr_db:.2f} dB  |  MSE: {mse:.6f}")
# → SNR: 22.48 dB  |  MSE: 0.000002
```

### `QuantizedTensor` Fields

| Field | Type | Description |
|-------|------|-------------|
| `data` | `np.ndarray` (uint8) | Packed integer data (sub-byte for 3/4-bit) |
| `scales` | `np.ndarray` (float16) | Per-group scale factors |
| `zeros` | `np.ndarray` (float16) or `None` | Per-group zero-points (None for symmetric) |
| `bits` | `int` | Bit-width used |
| `group_size` | `int` | Elements per quantization group |
| `shape` | `tuple` | Original tensor shape |
| `is_symmetric` | `bool` | Whether symmetric quantization was used |
| `memory_bytes` | `int` (property) | Approximate memory footprint |

### WeightStore (High-Level)

`WeightStore` handles loading, quantizing, and caching all Phi-4 projection weights:

```python
from experiments.phi4_cpu_run import WeightStore

# 4-bit with smart defaults (asymmetric + MSE-clip + group_size=32)
store = WeightStore(bits=4)

# 8-bit lossless
store = WeightStore(bits=8)

# Custom group size
store = WeightStore(bits=4, group_size=64)
```

Internally, `WeightStore` sets these optimization flags automatically:

| `bits` | `symmetric` | `optimize_clips` | `group_size` |
|-------:|:-----------:|:----------------:|:------------:|
| ≤ 3 | `False` | `True` | 16 |
| 4 | `False` | `True` | 32 |
| > 4 | `True` | `False` | 128 |

### Salience Analysis

```python
from asdsl.quantization.salience import compute_hessian_salience, allocate_bits_by_salience
import numpy as np

W = np.random.randn(3072, 3072).astype(np.float32)

# Compute Fisher-diagonal Hessian salience (requires calibration activations)
activations = [np.random.randn(1, 16, 3072).astype(np.float32) for _ in range(16)]
salience = compute_hessian_salience(W, activations)  # ~40 ms per layer

# Allocate bit-widths per group based on salience
bit_map = allocate_bits_by_salience(salience, target_avg_bits=3.5, group_size=128)
```

---

## Perplexity Evaluation

Perplexity (PPL) measures how well the model predicts held-out text — lower is better.
ASDSL includes a WikiText-2 evaluator built on top of the Phi-4 inference engine.

### Run Evaluation

```bash
# FP16 baseline
python evals/perplexity.py --bits 16

# ASDSL 4-bit
python evals/perplexity.py --bits 4

# Faster (fewer tokens — useful for quick iteration)
python evals/perplexity.py --bits 4 --max-tokens 512
```

### Results Summary

| Config | PPL | Quality |
|--------|:---:|:-------:|
| FP16 baseline | 11.06 | Baseline |
| 8-bit | 11.06 | ✅ Lossless on current benchmark slice |
| 4-bit (asymmetric + MSE-clip + gs=32) | 11.89 | ✅ Best stable tradeoff |
| 3-bit (asymmetric + MSE-clip + gs=16) | 14.14 | ✅ Usable |

### lm-evaluation-harness Integration

Run standard 0-shot benchmarks (HellaSwag, PIQA, ARC) with ASDSL as the backend:

```bash
# Install harness first
pip install lm-eval

# Run HellaSwag 0-shot with 4-bit ASDSL
python evals/lm_eval_harness.py --tasks hellaswag --bits 4

# Run multiple tasks
python evals/lm_eval_harness.py --tasks hellaswag,piqa --bits 8
```

---

## Benchmarks & Visualizations

### Run Full Benchmark Suite

```bash
# Full benchmark (perplexity + throughput + RAM + CPU for all bit-widths)
python benchmarks/comprehensive_bench.py

# Faster (fewer perplexity tokens)
python benchmarks/comprehensive_bench.py --max-tokens 128

# Skip perplexity — only measure throughput and resource usage
python benchmarks/comprehensive_bench.py --skip-perplexity
```

Results are saved to `benchmarks/results/benchmark_results.json`.

### Generate Visualizations

```bash
python benchmarks/generate_visuals.py
```

Generates 7 PNG charts in `benchmarks/results/`:

| Chart | Description |
|-------|-------------|
| `asdsl_benchmark_dashboard.png` | Main dashboard: PPL, speed, RAM, cores side-by-side |
| `asdsl_before_after.png` | Before/after improvement for 3-bit and 4-bit PPL |
| `asdsl_radar_chart.png` | Spider chart comparing all configs across 5 metrics |
| `asdsl_ppl_preservation.png` | Perplexity preservation waterfall |
| `asdsl_quant_quality.png` | SNR (dB) and compression ratio per bit-width |
| `asdsl_ram_usage.png` | Peak RAM usage comparison |
| `asdsl_summary_infographic.png` | Summary infographic with metric cards |

### Individual Benchmarks

```bash
# Quantization quality only (SNR, MSE across bit-widths)
python benchmarks/bench_quantization.py

# LUT engine build time and throughput
python benchmarks/bench_lut.py

# Inference throughput (tokens/second)
python benchmarks/bench_inference.py

# 3-bit packing/unpacking throughput (10-in-32 scheme)
python benchmarks/bench_3bit_throughput.py

# Quick SNR comparison across group sizes and modes
python benchmarks/quick_snr_test.py
```

### Run All Benchmarks

```bash
python benchmarks/run_all.py
```

---

## Project Structure

```
asdsl-framework/
│
├── asdsl/                          # Core library
│   ├── __init__.py                 # Version + package exports
│   ├── cli.py                      # `asdsl` CLI entry point
│   ├── config.py                   # Global config dataclass
│   ├── quantization/
│   │   ├── core.py                 # QuantizedTensor, quantize_weights(),
│   │   │                           # _find_optimal_scales(), compute_scale_zero()
│   │   └── salience.py             # Hessian salience scoring, bit allocation
│   ├── lut/                        # Lookup-table matmul engine
│   ├── speculative/                # SWIFT self-speculative decoding module
│   ├── prefetch/                   # Async L2 cache prefetch thread
│   ├── memory/                     # OS-level memory management (mlock, HugePages)
│   ├── inference/                  # Main inference engine
│   └── kernels/                    # Native kernel wrappers + compiled AVX2 extensions
│
├── experiments/
│   ├── phi4_cpu_run.py             # ★ End-to-end Phi-4 inference script
│   │                               #   WeightStore, forward_layer, generate(),
│   │                               #   experimental QCSD + sparse GEMV paths
│   └── phi4_integration.py         # Integration helpers
│
├── evals/
│   ├── perplexity.py               # WikiText-2 PPL evaluation
│   └── lm_eval_harness.py          # lm-eval-harness wrapper / custom model
│
├── benchmarks/
│   ├── comprehensive_bench.py      # ★ Full suite: PPL + throughput + RAM + CPU
│   ├── generate_visuals.py         # ★ Generate 7 PNG charts from JSON results
│   ├── bench_quantization.py       # Quantization quality benchmarks
│   ├── bench_lut.py                # LUT engine benchmarks
│   ├── bench_inference.py          # Inference throughput benchmarks
│   ├── bench_3bit_throughput.py    # 3-bit packing throughput
│   ├── quick_snr_test.py           # Quick SNR comparison tool
│   ├── run_all.py                  # Run all benchmarks in sequence
│   ├── RESULTS.md                  # Detailed benchmark results with all data
│   └── results/                    # Generated JSON + PNG outputs
│
├── tests/                          # 95 pytest tests
│   ├── test_quantization.py
│   ├── test_gemv_q4.py
│   ├── test_kernels.py
│   ├── test_kv_cache.py
│   ├── test_lut.py
│   ├── test_memory.py
│   ├── test_prefetch.py
│   ├── test_speculative.py
│   └── test_streaming.py
│
├── docs/                           # Extended documentation
├── examples/                       # Usage examples
├── models/                         # Model weights (not in git; download separately)
│   └── phi4-multimodal-instruct/
├── CONTRIBUTING.md
├── pyproject.toml
└── LICENSE
```

---

## How It Works

### 1 · MSE-Optimal Quantization (No Calibration Required)

Standard min-max quantization assigns a scale based on the absolute maximum value in each
group. A single large outlier forces the scale large, wasting precision on all other values.

ASDSL's `_find_optimal_scales()` instead:

1. Evaluates 9 clipping ratios: `[0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 1.0]`
2. For each ratio: clips the weight range, computes scale, rounds to **float16** (matching
   storage precision), quantizes and dequantizes, then measures MSE
3. Picks the ratio with minimum MSE **per group** — fully vectorized across groups

Evaluating at float16 precision is critical: the scale that minimizes float32 MSE is often
*not* optimal after rounding to float16. This is why fine-tuned ratios outperform naive grids.

### 2 · Asymmetric Quantization for ≤4-bit

For symmetric quantization, the range `[-abs_max, +abs_max]` is split into equal halves.
For asymmetric, the full `[min, max]` range maps to `[0, 2^bits - 1]` with a zero-point:

```
quantized = round(weight / scale) + zero_point
dequantized = (quantized - zero_point) * scale
```

Asymmetric is more accurate for low bit-widths (≤4-bit) because LLM weight distributions
are slightly skewed — they rarely have a true zero mean per group.

### 3 · Fine-Grained Group Sizes

Larger groups mean one scale serves more weights. At 3-bit (8 quant levels), one outlier
in a 128-element group can corrupt all 127 other weights. Smaller groups fix this:

| Bits | Group size | Quant levels | Why |
|-----:|:----------:|:------------:|-----|
| 3 | 16 | 8 | Critical — prevents catastrophic error |
| 4 | 32 | 16 | Balances quality vs. 6.25% scale overhead |
| 8 | 128 | 256 | Standard — 256 levels tolerate large groups |

### 4 · 3-Bit "10-in-32" Packing

Standard bit-packing for 3-bit stores 10 values in 32 bits (3×10=30 bits + 2 padding).
ASDSL's vectorized implementation achieves **26–36× speedup** over a naive Python loop
through NumPy vectorization of the pack/unpack operations.

### 5 · SWIFT Speculative Decoding

`asdsl/speculative/swift.py` implements the SWIFT-style layer-skipping speculative path.
This is distinct from the experimental QCSD path in `phi4_cpu_run.py`.

### 6 · Experimental QCSD

`experiments/phi4_cpu_run.py` also contains a separate experimental quantization-cascade
speculative decoder. It should currently be treated as a research path rather than a
stable benchmarked feature.

### 7 · Block-Sparse KV Cache

The KV cache allocates memory in 16-token blocks on demand, rather than pre-allocating
for `max_seq_len`. An importance score is maintained per block; when capacity is exceeded,
the least-important blocks are evicted. For a 200-token Phi-4 run, only 15 of 256 possible
blocks were ever allocated.

### 8 · LUT Matmul Engine

For 2-bit weights with `group_width=2`, each group of 2 weights has only 4 unique values
(2-bit × 2 positions = 16 combinations × 2 = the dot-product table fits in 256 entries).
The 2-bit LUT is 96 KB per layer — fits in L2 cache (~512 KB typical). This means the
matmul is **cache-resident**: no DRAM reads during compute.

---

## Testing

```bash
# Run all tests
pytest

# Run a specific module
pytest tests/test_quantization.py -v

# Run with coverage
pytest --cov=asdsl
```

The current suite has **95 passing tests** covering quantization, native kernels,
KV cache handling, LUT paths, speculative components, memory helpers, prefetching,
and streaming output.

---

## Roadmap

### Completed ✅

- [x] Core quantization (2–8 bit, symmetric and asymmetric)
- [x] MSE-optimal clipping (grid search, float16-aware)
- [x] Fine-grained groups (auto-selected by bit-width)
- [x] 3-bit "10-in-32" packing (26–36× speedup)
- [x] Salience scoring (Hessian diagonal, Fisher approximation)
- [x] LUT matmul engine (Python reference)
- [x] SWIFT speculative decoding module
- [x] Block-sparse KV cache with importance eviction
- [x] Async L2 cache prefetch thread
- [x] Thread control (`set_thread_count`, `--threads` CLI)
- [x] End-to-end Phi-4 (14B) inference with stable measured benchmarks
- [x] WikiText-2 perplexity evaluator
- [x] lm-evaluation-harness integration
- [x] Comprehensive benchmark suite with visualization
- [x] Native C++/AVX2 GEMV kernels for 8-bit, 4-bit, 3-bit, and 2-bit
- [x] LUT inference path without weight decompression (639× build speedup, 657× matvec speedup)
- [x] Streaming output (yield tokens as they are generated)

### In Progress / Planned 🔄

- [ ] stabilize 2-bit quality enough for use as a QCSD draft path
- [ ] validate QCSD in the main benchmark suite
- [ ] validate activation-sparse GEMV in the main benchmark suite
- [ ] GPTQ-style calibrated quantization (512 calibration samples)
- [ ] Vectorized greedy bit allocator (replace O(n²) Python loop)
- [ ] INT4/INT8 VNNI kernel path (Intel Sapphire Rapids / Meteor Lake)
- [ ] ARM NEON/SVE backend

---

## References

This framework synthesizes research from:

| Paper | Contribution |
|-------|-------------|
| **T-MAC** | LUT-based mixed-precision matrix multiplication |
| **SliM-LLM** | Salience-driven mixed-precision quantization |
| **TaCQ** | Task-circuit quantization |
| **SWIFT** | Layer-skipping self-speculative decoding |
| **BitNet** | Ternary weight quantization |
| **IntactKV** | Lossless pivot token KV cache generation |
| **BlockDialect / LServe** | Block-wise sparse attention |
| **GPTQ** | Optimal quantization via Hessian-guided rounding |
| **AWQ** | Activation-aware weight quantization |

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
