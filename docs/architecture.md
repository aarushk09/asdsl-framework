# ASDSL Architecture Guide

## Overview

The Asynchronous Salience-Driven Speculative Lookup (ASDSL) framework implements a
five-stage pipeline for efficient small language model (SLM) inference on commodity CPUs.
The architecture is designed around the Phi-3-mini (3.8B parameters) model family and
targets single-socket x86-64 and ARM platforms with 16–32 GB of DRAM.

## Pipeline Stages

```
┌─────────────────────────────────────────────────────────────────┐
│                     ASDSL Inference Pipeline                     │
├─────────┬──────────┬─────────────┬──────────────┬───────────────┤
│  Stage  │  Module  │  Key Idea   │  Speedup     │  Memory       │
├─────────┼──────────┼─────────────┼──────────────┼───────────────┤
│ 1. Quant│ quant/   │ Salience-   │ 4-8x less    │ 1.6 GB for    │
│         │          │ driven mixed│ memory       │ 3.8B @ ~3-bit │
│         │          │ precision   │              │               │
├─────────┼──────────┼─────────────┼──────────────┼───────────────┤
│ 2. LUT  │ lut/     │ Replace FMA │ ~2x faster   │ LUT tables    │
│         │          │ with table  │ than INT8    │ fit in L2     │
│         │          │ lookups     │ FMA          │               │
├─────────┼──────────┼─────────────┼──────────────┼───────────────┤
│ 3. Spec │ specul./ │ SWIFT self- │ 2.5-3.5x    │ No extra      │
│         │          │ speculative │ with 4       │ model needed  │
│         │          │ decoding    │ draft tokens │               │
├─────────┼──────────┼─────────────┼──────────────┼───────────────┤
│ 4. Pre- │ prefetch/│ Async L2    │ Hides DRAM   │ Dedicated     │
│   fetch │          │ cache       │ latency      │ prefetch      │
│         │          │ warming     │              │ thread        │
├─────────┼──────────┼─────────────┼──────────────┼───────────────┤
│ 5. Mem  │ memory/  │ mlock +     │ Eliminates   │ Huge pages    │
│         │          │ huge pages  │ page faults  │ + NUMA-aware  │
└─────────┴──────────┴─────────────┴──────────────┴───────────────┘
```

## Stage 1: Salience-Driven Mixed-Precision Quantization

### Motivation

Not all weight groups contribute equally to model quality. Attention projection
weights and pivotal token embeddings have outsized impact on output perplexity.
Assigning uniform bit widths wastes bits on unimportant groups while under-representing
critical ones.

### Algorithm

1. **Salience Scoring**: For each weight group `g`, compute salience as:
   $$S_g = \|W_g \odot \nabla_W \mathcal{L}\|_F$$
   This captures both weight magnitude and gradient importance.

2. **Hessian Refinement** (optional): Use diagonal Hessian approximation for
   more accurate sensitivity estimates.

3. **Greedy Bit Allocation**: Starting from a 2-bit floor, iteratively promote
   groups through the tier set `{2, 3, 4, 8}` in decreasing salience order
   until the global bit budget is met.

4. **Layer-Role Adjustment**: Apply per-role bias:
   - `down_proj`: +1 bit (information bottleneck)
   - `q_proj`, `k_proj`, `o_proj`: +0.5 bits (attention fidelity)

### Key Files

- `asdsl/quantization/core.py` — Low-level quantize/dequantize, bit packing
- `asdsl/quantization/salience.py` — Salience computation, bit allocation
- `asdsl/quantization/pipeline.py` — End-to-end quantization pipeline


## Stage 2: LUT-Based Matrix Multiplication

### Motivation

At sub-4-bit precision, the dominant cost of matrix-vector multiplication is not
arithmetic (which shrinks with fewer bits) but memory access. By precomputing all
possible partial sums for each weight group and storing them in lookup tables that
fit in L2 cache, we replace multiply-accumulate operations with table lookups.

### How It Works

For a weight group of `G` elements quantized to `B` bits:

1. **Build Phase**: Precompute $2^{B \cdot K}$ partial sums for each group,
   where `K` is the number of input elements processed per lookup.

2. **Lookup Phase**: At inference, use the input activation values as indices
   into the precomputed tables. A single table lookup replaces `G` multiply-add
   operations.

3. **SIMD Acceleration**: On x86, the `VPSHUFB` instruction performs 32 parallel
   4-bit lookups per cycle. On ARM, `TBL` provides similar functionality.

### Memory Footprint

For 4-bit quantization with `G=4`:
- Each table: $2^4 = 16$ float32 entries = 64 bytes
- Per-layer (3072×3072, group_size=128): ~1.5 MB → fits in L2 cache

### Key Files

- `asdsl/lut/engine.py` — Table construction, LUT-based matvec
- `asdsl/lut/permutation.py` — Weight permutation for SIMD alignment


## Stage 3: SWIFT Self-Speculative Decoding

### Motivation

Standard autoregressive decoding generates one token per forward pass. Speculative
decoding uses a fast "draft" model to generate multiple candidates, then verifies
them in parallel with the full model. SWIFT avoids the need for a separate draft
model by using the full model with layers skipped.

### Algorithm

1. **Draft Phase**: Run input through only the "draft layers" (first N and last N
   layers, skipping the middle). Generate `K` draft tokens autoregressively.

2. **Verify Phase**: Run all `K` draft tokens through the full model in a single
   forward pass (parallel verification).

3. **Accept/Reject**: Compare draft and full-model distributions. Accept tokens
   where $p_{\text{full}}(t) \geq p_{\text{draft}}(t)$, with stochastic
   acceptance for the remainder.

4. **Adaptive Schedule**: Monitor rolling acceptance rate and adjust skip ratio:
   - High acceptance (>80%) → skip more layers
   - Low acceptance (<50%) → skip fewer layers

### Expected Performance

- 4 draft tokens with 75% skip ratio → ~3x effective throughput
- No quality loss (mathematically equivalent output distribution)

### Key Files

- `asdsl/speculative/swift.py` — SWIFT decoder, skip schedule, adaptive tuning


## Stage 4: Asynchronous L2 Cache Prefetching

### Motivation

DRAM access latency (~80ns) dwarfs L2 cache latency (~4ns). If we can predict
which weight buffers will be needed next and prefetch them into L2 before the
compute thread needs them, we effectively hide the memory latency.

### Architecture

```
┌──────────────┐     notify_layer_start(N)     ┌──────────────┐
│              │ ──────────────────────────────►│              │
│  Compute     │                                │  Prefetch    │
│  Thread      │    prefetch layer N+1, N+2     │  Worker      │
│              │◄───────────────────────────────│  (daemon)    │
│              │  (weights now in L2 cache)      │              │
└──────────────┘                                └──────────────┘
```

- **Deterministic prefetch**: When compute starts layer N, prefetch layer N+1
- **Speculative prefetch**: During draft phase, pre-load verification layers

### Platform Support

- **Linux**: `madvise(MADV_WILLNEED)` for kernel-level prefetch
- **All platforms**: Sequential byte touching (`buffer[::stride]`) to force cache fill

### Key Files

- `asdsl/prefetch/orchestrator.py` — Dual-thread orchestrator, priority queue


## Stage 5: OS-Level Memory Management

### Motivation

Default memory allocation can suffer from page faults (first access to each 4KB page
triggers a kernel trap), TLB misses (too many 4KB pages for large models), and
swap-out (OS may evict model weights to disk under memory pressure).

### Techniques

| Technique | Platform | Effect |
|-----------|----------|--------|
| `mlock()` | Linux/macOS | Pins pages in RAM, prevents swap |
| `VirtualLock()` | Windows | Same as mlock |
| `MAP_HUGETLB` | Linux | 2MB huge pages, reduces TLB misses |
| NUMA binding | Linux | Keep data on same NUMA node as CPU |

### Key Files

- `asdsl/memory/manager.py` — Cross-platform memory manager


## Module Interaction Diagram

```
                    ┌─────────────────────┐
                    │   ASDSLEngine        │
                    │   (inference/engine)  │
                    └────┬──────────┬──────┘
                         │          │
              ┌──────────▼──┐  ┌───▼──────────────┐
              │ SWIFT       │  │ BlockSparseKVCache│
              │ Decoder     │  │ (inference/       │
              │ (specul.)   │  │  kv_cache)        │
              └──────┬──────┘  └──────────────────┘
                     │
           ┌─────────▼───────────┐
           │ TransformerLayer     │
           │ Executor             │
           └──┬──────┬──────┬────┘
              │      │      │
    ┌─────────▼┐ ┌──▼────┐ ┌▼───────────┐
    │ LUT      │ │ SIMD  │ │ Prefetch   │
    │ Engine   │ │Kernels│ │Orchestrator│
    │ (lut/)   │ │(kern.)│ │(prefetch/) │
    └──────────┘ └───────┘ └────────────┘
              │                  │
    ┌─────────▼──────────────────▼──┐
    │ QuantizedModel                 │
    │ (quantization/pipeline)        │
    └───────────┬────────────────────┘
                │
    ┌───────────▼────────────────────┐
    │ MemoryManager                   │
    │ (memory/manager)                │
    └─────────────────────────────────┘
```

## Configuration

The framework is configured through three dataclasses in `asdsl/config.py`:

- **`ModelConfig`**: Model architecture (num_layers, hidden_dim, num_heads, vocab_size)
- **`QuantizationConfig`**: Bit allocation (target_avg_bits, group_size, bit_tiers)
- **`InferenceConfig`**: Runtime settings (num_draft_tokens, kv_cache_blocks, prefetch_lookahead)

A preset for Phi-3-mini is available as `PHI3_MINI_CONFIG`.
