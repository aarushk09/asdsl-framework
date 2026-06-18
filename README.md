# ASDSL Framework

**Asynchronous Salience-Driven Speculative Lookup** — a CPU-first inference and quantization stack for large language models.

ASDSL treats batch-1 autoregressive decode as a **memory-bandwidth problem**: throughput is bounded by bytes moved per token divided by sustained DRAM bandwidth, not by peak FLOPs. The production path couples asymmetric 4-bit weight quantization with hand-written **AVX2 / FMA / F16C / AVX-VNNI** GEMV kernels, a 32-layer C++ `UnifiedEngine`, and a frozen benchmark protocol against [llama.cpp](https://github.com/ggerganov/llama.cpp).

| | |
|---|---|
| **Version** | 1.0.0 |
| **Primary model** | [Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) (~3.2B projection weights at listed dims) |
| **Reference baseline** | llama.cpp Q4_K_M GGUF, 12 threads, same hardware |
| **License** | Apache-2.0 |
| **Lab report** | [`benchmarks/PHASE_WALKTHROUGH.md`](benchmarks/PHASE_WALKTHROUGH.md) — phase-by-phase engineering log |
| **Phase 8 report** | [`benchmarks/PHASE8_THREADING_WALKTHROUGH.md`](benchmarks/PHASE8_THREADING_WALKTHROUGH.md) — threading + C++ decode completion audit |

---

## Table of contents

1. [What ASDSL is](#1-what-asdsl-is)
2. [Results at a glance](#2-results-at-a-glance)
3. [Quick start](#3-quick-start)
4. [Running inference](#4-running-inference)
5. [How it works](#5-how-it-works)
6. [Quantization formats](#6-quantization-formats)
7. [Native kernels and threading](#7-native-kernels-and-threading)
8. [UnifiedEngine](#8-unifiedengine)
9. [Speculative decoding](#9-speculative-decoding)
10. [Benchmarks and validation](#10-benchmarks-and-validation)
11. [Repository map](#11-repository-map)
12. [Environment variables](#12-environment-variables)
13. [Build and dependencies](#13-build-and-dependencies)
14. [Testing and CI](#14-testing-and-ci)
15. [Known limitations and roadmap](#15-known-limitations-and-roadmap)
16. [Further reading](#16-further-reading)

---

## 1. What ASDSL is

ASDSL is **not** a thin wrapper around PyTorch inference. It is an integrated framework with:

- **Quantization pipeline** — group-wise asymmetric Q2/Q3/Q4/Q8 with optional salience-driven bit allocation, importance matrices, and mixed-precision profiles.
- **Native GEMV kernels** — `preq` and `preq2` fused layouts, Q4_K ports, Q4_128, sparse paths; OpenMP scheduling with hybrid-core affinity.
- **UnifiedEngine** — full Phi-4 forward in C++: RMSNorm, QKV+GQA attention, RoPE, SiLU MLP, lm_head, KV cache.
- **Roofline profiler** — STREAM triad bandwidth, bytes-per-token accounting, theoretical decode ceiling.
- **Honest parity harness** — separate ASDSL / llama.cpp sessions, frozen manifest, thermal pre-checks, kernel preflight.

Two execution tracks coexist in the repository:

| Track | Entry | Use case |
|-------|-------|----------|
| **A — mmap / SLM** | `asdsl/inference/engine.py`, `kernels/forward_loop.cpp` | Phi-3-mini scale, LUT GEMV when weights fit L2, roofline leaderboard |
| **B — Phi-4 production** | `experiments/phi4_cpu_run.py`, `unified_bridge.py`, `_native_unified` | **All parity numbers and interactive inference** |

This README focuses on **Track B**, which is what you run for Phi-4 CPU generation today.

---

## 2. Results at a glance

**Hardware (lab machine):** Intel Core i7-1360P (4P+8E), 16 GB DDR4-3200, Windows 10, 12 threads (`physical` affinity unless noted).

### Baseline (pre–Phase 8)

| Config | decode tok/s | Notes |
|--------|--------------|-------|
| **C0** (canonical) | **9.88** | HF preq gs=32; parity + side-by-side reference |
| **C0-fast** | **~10.42** | `smt`, 16 threads, `GEMV_CHUNK_DIV=6` |
| **C0.1** (byte diet) | **~10.6–11.1** | g128 gate_up/down/lm_head; PPL ratio **1.37×** |
| **llama.cpp L0** | **~14.7–15.1** | Q4_K_M GGUF; side-by-side reference |

### Phase 8 — official doc run (2026-06-15)

Persistent `ThreadPool` + C++ `generate()` loop + optional g128 byte diet. Flags: `ASDSL_PERSISTENT_POOL=1`, `ASDSL_CPP_GENERATE=1`, `ASDSL_GEMV_CHUNK_DIV=4`. Full audit: [`PHASE8_THREADING_WALKTHROUGH.md`](benchmarks/PHASE8_THREADING_WALKTHROUGH.md).

**5-run cold parity** (prompt `"The"`, 128 tokens, `parity_benchmark.py`):

| Config | Mean tok/s | Range | vs 9.88 baseline |
|--------|------------|-------|------------------|
| **C0 + Phase 8** | **11.01** | 10.92–11.11 | **+11%** |
| **C0.3 + Phase 8** | **15.69** | 15.29–15.98 | **+59%** |

Artifacts: [`doc_run_parity_C0_20260615.json`](benchmarks/results/doc_run_parity_C0_20260615.json), [`doc_run_parity_C03_20260615.json`](benchmarks/results/doc_run_parity_C03_20260615.json).

**Side-by-side vs llama.cpp** (3 instructional prompts × 5 runs, 100 tokens, `compare_llama_cpp.py`):

| Config | ASDSL grand mean | llama grand mean | ASDSL % of llama |
|--------|------------------|------------------|------------------|
| **C0 + Phase 8** | **9.71 tok/s** | **14.79 tok/s** | **65.7%** |
| **C0.3 + Phase 8** | **13.64 tok/s** | **14.70 tok/s** | **92.8%** |

Per-prompt (C0.3): gravity **13.58**, quantization_typos **13.36**, gravity_one_sentence **13.99** tok/s.

Artifacts: [`side_by_side_C0_phase8_20260615.json`](benchmarks/results/side_by_side_C0_phase8_20260615.json), [`side_by_side_C03_phase8_20260615.json`](benchmarks/results/side_by_side_C03_phase8_20260615.json). Summary: [`doc_run_summary_20260615.json`](benchmarks/results/doc_run_summary_20260615.json).

**Quality (sliding-window PPL, 92 tokens):**

| Config | PPL | Ratio vs C0 |
|--------|-----|-------------|
| C0 | **2.73** | 1.00× |
| C0.3 | **3.56** | **1.30×** (gate ≤1.50× **pass**) |

WikiText-2 slice (1536 tokens, C0.3 doc run): ASDSL **80.55**; llama step failed (missing local file).

### Phase 8 — engineering findings (summary)

| Finding | Result |
|---------|--------|
| **Bottleneck** | Decode is **~97% GEMV** (not attention/KV at short ctx); see [`E2E_BOTTLENECK_RESEARCH.md`](benchmarks/E2E_BOTTLENECK_RESEARCH.md) |
| **Pool + cpp on C0** | **+11%** parity (11.01 vs 9.88); isolated pool **+1.25 tok/s** vs flags-off |
| **C++ decode** | Removes ~30 ms/token Python/`other` overhead (engine profile); does not shrink GEMV ms |
| **chunk_div under pool** | E2E winner **div=4** (microbench favored div=3) |
| **C0.3 byte diet** | **+59%** parity vs baseline; **~93%** of llama on instructional prompts |
| **13.9 tok/s victory bar** | **Not met** on canonical **C0**; **exceeded** on exploratory **C0.3** parity (15.69) |
| **Flags in C0 manifest** | **Not promoted** — C0 5-run mean &lt; 13.9 (by design) |
| **Barrier probe** (OMP_WAIT_POLICY) | No improvement ≥0.3 tok/s |

### Historical reference (2026-06-14, pre–Phase 8 completion)

Side-by-side C0 @ 12t (no Phase 8 flags): ASDSL **9.99** vs llama **15.15** tok/s (**65.9%**). Artifacts: [`side_by_side_comparison.json`](benchmarks/results/side_by_side_comparison.json).

### Kernel microbench (gate_up, 12t physical)

| Layout | GB/s | Notes |
|--------|------|-------|
| preq2, `CHUNK_DIV=4` | **~20–40** | E2E winner under pool |
| preq2, `CHUNK_DIV=6` | **~44** | microbench autotune best (`phase6_chunk_cache.json`) |
| Legacy preq fused | **~0.9** | pre-preq2 baseline |

### What we claim

- Reproducible **~11 tok/s** on **C0 + Phase 8** and **~16 tok/s** on **C0.3 + Phase 8** (parity, 12t physical).
- **~93%** of llama.cpp throughput on instructional prompts with **C0.3** (exploratory g128 stack; not byte-identical weights).
- Lossless **PLD** correctness; rigorous GEMV + trajectory + prefill regression tests.
- Phase 8 threading + C++ decode loop **implemented and gate-tested** (see Phase 8 walkthrough).

### What we do **not** claim

- **Canonical C0** at llama **13.9 tok/s** parity (C0 + Phase 8: **11.01** parity / **9.71** side-by-side).
- Bit-identical weights vs llama Q4_K_M (C0/C0.3 use HF preq; use **C1** + `--weight-parity` for same GGUF tensors).
- Long-form prose quality at **C0.3** on greedy 1000+ token runs (quantization + greedy repetition risk).
- RSS ≤ 3 GB (measured **~5.4 GB** peak).
- `--chat` interactive mode with unified C0.3 (use `--stream` or non-stream `generate`; `--chat` uses legacy NumPy path).

Full gate tables: [`benchmarks/PHASE_WALKTHROUGH.md`](benchmarks/PHASE_WALKTHROUGH.md), [`benchmarks/PHASE8_THREADING_WALKTHROUGH.md`](benchmarks/PHASE8_THREADING_WALKTHROUGH.md).

---

## 3. Quick start

### Prerequisites

- Python **≥ 3.10**
- C++17 compiler with **AVX2 + FMA** (MSVC on Windows, GCC/Clang on Linux)
- **OpenMP** enabled (`/openmp` or `-fopenmp`)
- ~16 GB RAM, ~11 GB disk for Phi-4 safetensors

### Install

```bash
git clone https://github.com/aarushk09/asdsl-framework.git
cd asdsl-framework
pip install -e ".[dev]"
python setup.py build_ext --inplace
```

Without native extensions, throughput falls to ~0.1–0.5 tok/s (Python fallback).

### Model weights

Place HuggingFace shards under `models/phi4-multimodal-instruct/` or run:

```bash
python experiments/phi4_integration.py
```

### Build caches (strongly recommended)

First quantize + preq build takes **15–40 minutes**. Cached loads take **~3 s**:

```bash
python benchmarks/build_caches.py
python benchmarks/diagnose_load.py   # should finish in <20s
```

Artifacts: `models/phi4_weight_cache/phi4_cpu_*.safetensors`, `phi4_preq_*_q4_32.safetensors`.

---

## 4. Running inference

### Recommended command (interactive, readable output)

PowerShell one-liner — **C0-fast** profile (pre–Phase 8 best interactive):

```powershell
cd c:\Users\aarus\projects\asdsl-framework ; $env:PYTHONPATH = "c:\Users\aarus\projects\asdsl-framework" ; $env:ASDSL_USE_UNIFIED = "1" ; $env:ASDSL_PREQ2 = "1" ; $env:ASDSL_FUSED_GEMV = "1" ; $env:ASDSL_CHUNKED_GEMV = "1" ; $env:ASDSL_AFFINITY = "smt" ; $env:ASDSL_GEMV_CHUNK_DIV = "6" ; $env:OMP_NUM_THREADS = "16" ; python experiments/phi4_cpu_run.py --bits 4 --threads 16 --max-new-tokens 128 --system-prompt "You are a helpful assistant." --prompt "Explain gravity in simple terms."
```

**First process start:** ~80 s `preq2 repack` (128 tensors; body preq2 not yet disk-cached). **Decode:** ~10 tok/s, ~100 ms/token median.

### Phase 8 — C0.3 interactive (best throughput, ~16 tok/s parity)

Use **`--stream`** or non-stream mode (not `--chat`; `--chat` uses the legacy NumPy path and fails after packed weights are freed).

```powershell
cd c:\Users\aarus\projects\asdsl-framework
$env:PYTHONPATH = "c:\Users\aarus\projects\asdsl-framework"
$env:ASDSL_USE_UNIFIED = "1"
$env:ASDSL_PREQ2 = "1"
$env:ASDSL_CHUNKED_GEMV = "1"
$env:ASDSL_FUSED_GEMV = "1"
$env:ASDSL_AFFINITY = "physical"
$env:ASDSL_GEMV_CHUNK_DIV = "4"
$env:ASDSL_C01 = "1"
$env:ASDSL_C03 = "1"
$env:ASDSL_GATEUP_GS = "128"
$env:ASDSL_DOWN_GS = "128"
$env:ASDSL_LMHEAD_GS = "128"
$env:ASDSL_PERSISTENT_POOL = "1"
$env:ASDSL_CPP_GENERATE = "1"
$env:OMP_NUM_THREADS = "12"
python experiments/phi4_cpu_run.py --bits 4 --threads 12 --stream --max-new-tokens 128 --system-prompt "You are a helpful assistant." --prompt "What is the capital of France? Answer in one word."
```

**First C0.3 run:** ~7 min one-time g128 repack. **Subsequent runs:** cached load ~3 s. Keep `--max-new-tokens` modest (≤256) for coherent greedy output.

### Canonical parity config (C0)

```powershell
$env:ASDSL_AFFINITY = "physical"
$env:OMP_NUM_THREADS = "12"
python experiments/phi4_cpu_run.py --bits 4 --threads 12 --max-new-tokens 128 --system-prompt "" --prompt "The"
```

### Phase 8 parity (C0 or C0.3)

```powershell
# C0 + Phase 8
$env:ASDSL_PERSISTENT_POOL = "1"
$env:ASDSL_CPP_GENERATE = "1"
$env:ASDSL_GEMV_CHUNK_DIV = "4"
python benchmarks/parity_benchmark.py --config C0 --runs 5 --cooldown 30 --asdsl-only

# C0.3 + Phase 8 (add byte-diet flags above + C01/C03/GS=128)
python benchmarks/parity_benchmark.py --config C0.3 --runs 5 --cooldown 30 --asdsl-only
```

### Streaming tokens

Add `--stream` to print tokens as they are generated. Per-token decode may show BPE fragments; use non-stream mode for readable paragraphs.

### Side-by-side vs llama.cpp (3 prompts × 5 runs)

**Fair comparison** (default): C0 @ 12 threads on both stacks, greedy decode, same prompts.

```powershell
cd c:\Users\aarus\projects\asdsl-framework
$env:PYTHONPATH = "c:\Users\aarus\projects\asdsl-framework"
python benchmarks/compare_llama_cpp.py --runs 5 --max-new-tokens 100 --cooldown 30 --with-ppl
```

For **Phase 8** configs, set `ASDSL_PERSISTENT_POOL=1`, `ASDSL_CPP_GENERATE=1`, and (for C0.3) C01/C03/g128 flags before running. Latest doc-run results: [§2](#phase-8--official-doc-run-2026-06-15).

**Weight parity** (same GGUF tensors as llama.cpp):

```powershell
python benchmarks/compare_llama_cpp.py --weight-parity --runs 5 --with-ppl
```

Exploratory configs (e.g. C0-fast @ 16t) require `--allow-exploratory` and still lock llama to the same thread count.

Prompts: gravity, quantization (typos), gravity one-sentence. Output: `benchmarks/results/side_by_side_comparison.json` (or dated Phase 8 artifacts in `benchmarks/results/`).

### Parity benchmark

```powershell
python benchmarks/parity_benchmark.py --config C0 --runs 5 --cooldown 30 --asdsl-only
```

Manifest: [`benchmarks/results/parity_manifest.json`](benchmarks/results/parity_manifest.json).

---

## 5. How it works

### Roofline model

Decode throughput is bounded by:

$$
\text{tok/s} \leq \frac{B_\text{sustained}}{\text{bytes/token}}
$$

where \(B_\text{sustained}\) is measured DRAM bandwidth (STREAM triad, native OpenMP when available) and **bytes/token** counts all weight, activation, and KV traffic per generated token.

For Phi-4 C0 (preq2, gs=32), measured **bytes/token ≈ 2.41 GB** (body ~2.01 GB + lm_head ~0.38 GB + KV negligible at ctx 128). Implementation: [`asdsl/profiler.py`](asdsl/profiler.py).

At **~25–35 GB/s** effective GEMV bandwidth, the kernel slice alone suggests ~7–14 tok/s. Engine profiling (Phase 8, pool on) shows decode is **~97% GEMV** at short context; the dominant win from `ASDSL_CPP_GENERATE=1` is removing **~30 ms/token** Python/`other` overhead, not shrinking per-kernel GEMV time. Residual gap vs llama on canonical C0 is primarily **GEMV efficiency and bytes/token** (C0.3 closes most of that gap). See [`benchmarks/E2E_BOTTLENECK_RESEARCH.md`](benchmarks/E2E_BOTTLENECK_RESEARCH.md).

### Per-token forward (Track B)

```
Token ID → fp16 embed → [×32 layers] → final RMSNorm → lm_head GEMV → logits → argmax
                │              │
                │              ├── RMSNorm → QKV GEMV → GQA attention → O-proj
                │              └── RMSNorm → gate_up GEMV → SiLU×gate → down GEMV
                └── KV cache updated each layer
```

Python (`phi4_cpu_run.py`) handles tokenizer, chat template, and the decode loop. When `ASDSL_USE_UNIFIED=1`, each token calls into C++ via [`asdsl/inference/unified_bridge.py`](asdsl/inference/unified_bridge.py).

### Prefill vs decode

- **Prefill:** all prompt tokens are forwarded through the engine; intermediate tokens use `forward_token_prefill` (full layer stack, **no lm_head**).
- **Decode:** each new token runs full `forward_token` including lm_head.

A prefill bug (fixed 2026-06-14) previously skipped KV updates for all but the last prompt token in `generate_stream`, causing nonsense on multi-token prompts. Regression: `benchmarks/test_generate_stream_prefill.py`.

---

## 6. Quantization formats

### Asymmetric Q4 (group size 32 default)

Per group of 32 weights along the input dimension:

- `scale` (fp16), `zero` / bias (fp16)
- 16 packed bytes (32 × 4-bit nibbles)

Dequant per weight \(w_i\):

$$
w_i = (q_i - z) \cdot s
$$

where \(q_i\) is the 4-bit code, \(z\) is zero-point, \(s\) is scale.

### Preq blocks (20 bytes / 32 weights)

Legacy fused layout used by `gemv_q4_32_preq_fused_avx2`:

- 4-byte scale+zero header
- 16-byte nibble payload
- Activation quantized to Q8 per group inside the kernel (fused path)

**Bytes per row per group:** 20. Strided access patterns limit SIMD efficiency — motivation for preq2.

### Preq2 (64-byte aligned interleaved + AVX-VNNI)

[`asdsl/quantization/repack_preq2.py`](asdsl/quantization/repack_preq2.py) converts preq blocks to:

- **meta:** `n_groups × 4` bytes per row (fp16 scale + fp16 zero)
- **quant:** 4-row bands, 64 bytes per group (cache-line aligned nibble interleave)

Kernel: [`asdsl/kernels/native/gemv_preq2_avx2.cpp`](asdsl/kernels/native/gemv_preq2_avx2.cpp) — `VPDPBUSD` dot-product-on-bytes, vectorized zero-fold, no per-group scalar dependency chains.

Enabled with `ASDSL_PREQ2=1`. Repack at load: **~80 s** for 128 tensors (cached preq blocks; preq2 body not yet persisted to disk separately).

### C0.1 / C0.3 byte diet (exploratory)

`ASDSL_C01=1` requantizes gate_up, down_proj, lm_head to **group_size=128** with imatrix-lite weighting:

- **−13.8%** bytes/token (`phase4_bytes_audit.json`)
- PPL ratio **1.37×** vs C0 (gate ≤1.45×)
- Uses legacy `gemv_q4_128_preq_avx2` for g128 tensors (preq2 g128 kernel not yet built)

`ASDSL_C03=1` extends g128 to **qkv_proj** and **o_proj** as well (full projection byte diet). With Phase 8 flags: **15.69 tok/s** parity, **92.8%** of llama side-by-side, sliding-window PPL **3.56** (1.30× vs C0).

### Weight cache pipeline

```
HF safetensors → asymmetric Q4 quantize → preq blocks → [optional] preq2 repack
                      ↓                        ↓
              phi4_cpu_*.safetensors    phi4_preq_*_q4_32.safetensors
```

Scripts: [`benchmarks/build_caches.py`](benchmarks/build_caches.py), [`asdsl/quantization/lmhead_preq2_cache.py`](asdsl/quantization/lmhead_preq2_cache.py).

---

## 7. Native kernels and threading

### Extension modules

| Module | Contents |
|--------|----------|
| `_native_gemv` | GEMV kernels: preq, preq2, Q4_128, Q4_K, Q8, Q2, lm_head |
| `_native_unified` | `UnifiedEngine`: full model forward, PLD verify hooks |
| `_native_forward` | mmap track: `generate_token`, STREAM triad |

Build: `python setup.py build_ext --inplace` → `.pyd` / `.so` copied into `asdsl/kernels/`.

### Hybrid-core threading (i7-1360P)

| Mode | `ASDSL_AFFINITY` | Behavior |
|------|------------------|----------|
| Legacy | `legacy` | 12 threads on P-core SMT only (historical parity mistake) |
| Physical | `physical` | One thread per physical core (4P+8E) — **canonical C0** |
| SMT | `smt` | 16 threads across P+E logical CPUs — **best E2E** |
| Spread | `spread` | Explicit logical CPU list |

Chunked GEMV (`ASDSL_CHUNKED_GEMV=1`): atomic row-chunk work stealing via [`gemv_chunked.hpp`](asdsl/kernels/native/gemv_chunked.hpp). Divisor `ASDSL_GEMV_CHUNK_DIV` (default 4; microbench autotune best **6**, but E2E under persistent pool favors **4**).

### Phase 8 — persistent pool + C++ decode

| Flag | Effect |
|------|--------|
| `ASDSL_PERSISTENT_POOL=1` | Reuse `ThreadPool` across tokens; preq2 GEMV + attention dispatch to pool (not fresh `#pragma omp` per call) |
| `ASDSL_CPP_GENERATE=1` | C++ greedy decode loop (`generate_with_stops`); releases GIL; removes Python per-token overhead |
| `ASDSL_C03=1` | g128 on qkv/o projections (with `ASDSL_C01=1` + `ASDSL_*_GS=128`) |

Pool workers are pinned at construction; `tl_active_pool` guard prevents pool dispatch when kernels run outside `UnifiedEngine` scope (correctness fix for tests with pool env set).

Pinning: [`omp_pcore_pinning.hpp`](asdsl/kernels/native/omp_pcore_pinning.hpp).

---

## 8. UnifiedEngine

**Source:** [`asdsl/kernels/native/unified_engine.cpp`](asdsl/kernels/native/unified_engine.cpp)

- Loads per-layer preq2 blobs, RMSNorm weights, RoPE tables, fp16 embedding table.
- `forward_token(id, pos, out_logits)` — full layer loop; `out_logits=nullptr` skips lm_head (prefill).
- `forward_token_prefill` exposed to Python for multi-token prefill.
- lm_head: preq2 fused GEMV when `lm_head_preq2` cache hit (~293 MB quant blob).
- Optional: `ASDSL_LARGE_PAGES=1` for lm_head buffers (E2E regression — keep off).

Bridge API ([`unified_bridge.py`](asdsl/inference/unified_bridge.py)):

| Function | Purpose |
|----------|---------|
| `get_or_build_unified_engine(store)` | Lazy engine init from WeightStore |
| `unified_forward_token(store, id, pos, need_logits=…)` | Single decode step |
| `greedy_generate(store, prompt_ids, n)` | Reference greedy (tests) |
| `cpp_generate(store, prompt_ids, n, …)` | C++ greedy loop when `ASDSL_CPP_GENERATE=1` |
| `pld_generate(…)` | Prompt lookup decoding |

---

## 9. Speculative decoding

| Method | Flag | Status |
|--------|------|--------|
| **PLD** (prompt lookup) | `ASDSL_USE_PLD=1` | Lossless ✓; throughput ✗ (serial preq2 verify) |
| **AHSD** (layer skip) | `ASDSL_USE_AHSD=1` | Experimental |
| **QCSD** | `--qcsd` | Dual-model draft (requires draft bank) |
| **SWIFT** | via engine API | Self-speculative layer skip (mmap track) |

PLD: [`asdsl/speculative/pld.py`](asdsl/speculative/pld.py). Default **off** (`ASDSL_USE_PLD=0` in parity manifest).

---

## 10. Benchmarks and validation

### Official doc run (2026-06-15, Phase 8 complete)

Unified session: 5-run parity (C0 + C0.3), side-by-side vs llama (both configs), sliding-window PPL gate.

| Step | C0 + Phase 8 | C0.3 + Phase 8 |
|------|--------------|----------------|
| 5-run parity (`"The"`, 128 tok) | **11.01 tok/s** | **15.69 tok/s** |
| Side-by-side grand mean | **9.71** / llama **14.79** (**65.7%**) | **13.64** / llama **14.70** (**92.8%**) |
| Sliding-window PPL | **2.73** | **3.56** (1.30×) |

Summary: [`doc_run_summary_20260615.json`](benchmarks/results/doc_run_summary_20260615.json). Full audit: [`PHASE8_THREADING_WALKTHROUGH.md`](benchmarks/PHASE8_THREADING_WALKTHROUGH.md) (Final section).

**Caveats:** Parity and side-by-side use different prompts/lengths. Run C0 PPL only in a fresh session (post–side-by-side C0 PPL was thermally contaminated). WikiText PPL: C0.3 **80.55**; llama step failed (missing local file).

### Side-by-side vs llama.cpp (3 prompts × 5 runs)

[`benchmarks/compare_llama_cpp.py`](benchmarks/compare_llama_cpp.py) — fair protocol (12 threads both stacks by default), separate ASDSL / llama sessions, thermal guard, optional PPL.

**Command:**

```powershell
$env:PYTHONPATH = "c:\Users\aarus\projects\asdsl-framework"
python benchmarks/compare_llama_cpp.py --runs 5 --max-new-tokens 100 --cooldown 30 --with-ppl
```

**Historical run (2026-06-14, pre–Phase 8):** ASDSL **9.99** vs llama **15.15** tok/s (**65.9%**). Artifacts: [`side_by_side_comparison.json`](benchmarks/results/side_by_side_comparison.json), [`side_by_side_comparison_run_log.txt`](benchmarks/results/side_by_side_comparison_run_log.txt).

- **Fairness:** C0 config, 12 threads on ASDSL and llama. Greedy decode, ignore-eos.
- **Weights:** not byte-identical (C0 HF preq vs llama Q4_K_M GGUF). Use `--weight-parity` for C1 + same GGUF.
- **Metrics:** ASDSL `decode_2_N` (tokens 2–N) vs llama CLI footer (all tokens).

### Frozen parity protocol

[`benchmarks/parity_benchmark.py`](benchmarks/parity_benchmark.py) + [`benchmarks/results/parity_manifest.json`](benchmarks/results/parity_manifest.json):

- 5 runs, 30 s cooldown, separate ASDSL / llama sessions
- Metric: **decode tok/s (tokens 2–N)** — excludes first generated token
- Kernel preflight: [`benchmarks/kernel_preflight.py`](benchmarks/kernel_preflight.py) (fail if slower than reference)
- Thermal guard: [`benchmarks/thermal_utils.py`](benchmarks/thermal_utils.py)

### Manifest configs

| ID | Description |
|----|-------------|
| **C0** | Canonical HF preq gs=32 |
| **C0.1** | Byte diet g128 (gate_up/down/lm_head) |
| **C0.3** | Full projection g128 + Phase 8 pool/cpp flags |
| **C0-fast** | smt + chunk_div=6 + 16t |
| **C1** | GGUF Q4_K native hybrid |
| **C5** | PLD on |
| **C6 / C6-smt** | Large pages / stretch |
| **L0** | llama.cpp reference |

### Test suite (correctness)

| Test | What it proves |
|------|----------------|
| `test_preq2_correctness.py` | Synthetic shapes bit-match legacy preq |
| `test_preq2_real_weights.py` | Production gate_up drift bounded (maxdiff <12) |
| `test_lmhead_preq2_correctness.py` | lm_head preq2 layout |
| `test_pld_lossless.py` | PLD greedy ≡ serial forward |
| `test_greedy_trajectory.py` | 5 prompts × 128 tokens deterministic corpus |
| `test_generate_stream_prefill.py` | `generate_stream` ≡ `greedy_generate` on long prompts |
| `test_preq_correctness.py` | Legacy preq kernels |

### Key result artifacts

```
benchmarks/results/
├── parity_manifest.json
├── doc_run_summary_20260615.json       # Phase 8 official session summary
├── doc_run_parity_C0_20260615.json
├── doc_run_parity_C03_20260615.json
├── side_by_side_C0_phase8_20260615.json
├── side_by_side_C03_phase8_20260615.json
├── phase8_final_artifacts.json
├── phaseG_kernel_bench.json
├── phase6_chunk_cache.json
├── phase4_bytes_audit.json
├── phase5_pld.json
├── phase3_rss_probe.json
├── greedy_trajectory_golden.json
├── side_by_side_comparison.json        # pre–Phase 8 (2026-06-14)
└── PHASE_WALKTHROUGH.md                # full lab report (in benchmarks/)
```

### Engineering phases (summary)

| Phase | Goal | Outcome |
|-------|------|---------|
| 0 | Baseline truth | ~9 tok/s C0, topology audit |
| 1 | Threading + chunked GEMV | Infrastructure; E2E gate missed |
| 2 | preq2 + VNNI | Kernel fast; E2E gate missed |
| 3 | lm_head preq2 | **9.88 tok/s** parity |
| 4 | C0.1 byte diet | Speed + PPL gates **met** |
| 5 | PLD | Lossless ✓; throughput ✗ |
| 6 | Stretch (smt, chunk autotune) | **~10.4 tok/s** best pre–Phase 8 |
| 7 | v1 cleanup + inference fix | Demo-ready generation |
| 8 | Persistent pool + C++ generate | **11.01** C0 / **15.69** C0.3 parity; **92.8%** llama (C0.3 side-by-side); gates + doc run complete |

---

## 11. Repository map

```
asdsl-framework/
├── experiments/
│   └── phi4_cpu_run.py          # PRIMARY CLI: load, generate, benchmark
├── asdsl/
│   ├── inference/
│   │   └── unified_bridge.py    # Python ↔ UnifiedEngine
│   ├── quantization/            # repack, imatrix, salience
│   ├── speculative/pld.py
│   ├── profiler.py              # roofline / STREAM
│   └── kernels/native/          # C++ sources (unified_engine, gemv_preq2, …)
├── benchmarks/
│   ├── parity_benchmark.py      # canonical vs llama.cpp
│   ├── compare_llama_cpp.py     # side-by-side fair run
│   ├── build_caches.py
│   ├── PHASE_WALKTHROUGH.md     # phase lab report
│   ├── PHASE8_THREADING_WALKTHROUGH.md
│   └── test_*.py                # correctness + regression
├── models/
│   ├── phi4-multimodal-instruct/   # HF weights (gitignored)
│   └── phi4_weight_cache/          # persistent quant + preq caches
├── tools/llama.cpp/             # parity reference binaries
├── .github/workflows/ci.yml     # smoke tests
├── ASDSL_ARCHITECTURE.md        # extended architecture notes
└── pyproject.toml               # v1.0.0, pytest config
```

---

## 12. Environment variables

### Production (C0-fast interactive)

| Variable | Value | Effect |
|----------|-------|--------|
| `ASDSL_USE_UNIFIED` | `1` | C++ UnifiedEngine |
| `ASDSL_PREQ2` | `1` | preq2 + VNNI kernels |
| `ASDSL_FUSED_GEMV` | `1` | Fused preq path |
| `ASDSL_CHUNKED_GEMV` | `1` | Atomic chunked scheduling |
| `ASDSL_AFFINITY` | `smt` | 16-thread hybrid placement |
| `ASDSL_GEMV_CHUNK_DIV` | `6` | Chunk divisor (autotuned) |
| `OMP_NUM_THREADS` | `16` | OpenMP threads |

### Canonical parity (C0)

| Variable | Value |
|----------|-------|
| `ASDSL_AFFINITY` | `physical` |
| `OMP_NUM_THREADS` | `12` |
| `ASDSL_USE_PLD` | `0` |
| `ASDSL_LARGE_PAGES` | `0` |

### Phase 8 (exploratory — not in canonical C0 `env_required`)

| Variable | Value | Effect |
|----------|-------|--------|
| `ASDSL_PERSISTENT_POOL` | `1` | Persistent `ThreadPool` for GEMV + attention |
| `ASDSL_CPP_GENERATE` | `1` | C++ greedy decode loop |
| `ASDSL_GEMV_CHUNK_DIV` | `4` | E2E winner under pool (vs microbench div=6) |
| `ASDSL_C03` | `1` | g128 qkv/o (with C0.1 flags) |

### Exploratory

| Variable | Effect |
|----------|--------|
| `ASDSL_C01=1` | C0.1 byte diet (g128 gate_up/down/lm_head) |
| `ASDSL_C03=1` | C0.3 — extend g128 to qkv/o projections |
| `ASDSL_USE_PLD=1` | Prompt lookup decoding |
| `ASDSL_LARGE_PAGES=1` | 2 MB pages for lm_head (E2E regression observed) |
| `ASDSL_IGNORE_EOS=1` | Parity: do not stop at EOS |

Full list: parity manifest `env_required` + per-config overrides in [`parity_manifest.json`](benchmarks/results/parity_manifest.json).

---

## 13. Build and dependencies

### Python (`pyproject.toml`)

- `numpy`, `torch`, `safetensors`, `transformers`, `psutil`
- Dev: `pytest`, `pybind11`, `ruff`, `mypy`

### Native compile flags (MSVC)

`/arch:AVX2 /fp:fast /openmp /O2` — see [`setup.py`](setup.py).

### Verify build

```bash
python -c "import asdsl.kernels._native_unified, asdsl.kernels._native_gemv; print('ok')"
python benchmarks/kernel_preflight.py
```

---

## 14. Testing and CI

### Local smoke (matches CI)

```bash
python setup.py build_ext --inplace
pytest benchmarks/test_preq2_correctness.py benchmarks/test_preq_correctness.py -q
python benchmarks/kernel_preflight.py
```

### Slow tests (require weight cache)

```bash
pytest benchmarks/test_pld_lossless.py benchmarks/test_preq2_real_weights.py -q -m slow
pytest benchmarks/test_greedy_trajectory.py -q -m slow
pytest benchmarks/test_generate_stream_prefill.py -q -m slow
```

### CI workflows

| Workflow | Purpose |
|----------|---------|
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | Build, imports, preq2, preflight, PLD unit |
| [`.github/workflows/benchmark_regression.yml`](.github/workflows/benchmark_regression.yml) | Extended regression |

---

## 15. Known limitations and roadmap

| Issue | Status | Notes |
|-------|--------|-------|
| Canonical C0 vs llama (13.9 tok/s) | Open | C0 + Phase 8: **11.01** parity; residual gap is GEMV efficiency / bytes |
| C0.3 long greedy runs | Quality risk | Repetition at 1000+ tokens; use modest `--max-new-tokens` |
| `--chat` with unified C0.3 | Broken | Uses legacy NumPy path; use `--stream` or non-stream generate |
| `--stream` tok/s footer | Bug | May show **0.00 tok/s**; non-stream footer is reliable |
| preq2 body disk cache | Open | ~80 s repack each process start |
| preq2 g128 kernel | Planned | C0.1/C0.3 use legacy g128 GEMV |
| PLD throughput | Blocked | Needs preq2 batched verify |
| RSS ≤ 3 GB gate | Not met | Measured ~5.4 GB |
| Phase 8 flags in C0 manifest | By design | Not promoted until C0 ≥ 13.9 |
| preq2 real-weight drift | Bounded | maxdiff <12 on gate_up slice; not bit-identical |

**v2 direction:** preq2 g128 layout, preq2 body cache, batched verify for speculation, further GEMV tuning on canonical C0.

---

## 16. Further reading

- [`benchmarks/PHASE_WALKTHROUGH.md`](benchmarks/PHASE_WALKTHROUGH.md) — complete phase-by-phase lab notebook with gate tables, artifacts, and bug tracker
- [`benchmarks/PHASE8_THREADING_WALKTHROUGH.md`](benchmarks/PHASE8_THREADING_WALKTHROUGH.md) — Phase 8 threading audit, isolated A/B attribution, doc-run artifacts
- [`benchmarks/E2E_BOTTLENECK_RESEARCH.md`](benchmarks/E2E_BOTTLENECK_RESEARCH.md) — forensic decode profile (GEMV-bound at short ctx)
- [`ASDSL_ARCHITECTURE.md`](ASDSL_ARCHITECTURE.md) — roofline math, five-stage pipeline, mmap track
- Parity manifest: [`benchmarks/results/parity_manifest.json`](benchmarks/results/parity_manifest.json)
- Doc run summary: [`benchmarks/results/doc_run_summary_20260615.json`](benchmarks/results/doc_run_summary_20260615.json)

---

*ASDSL Framework v1.0.0 — CPU quantization and inference research codebase. Benchmark honestly; ship what tests prove.*
