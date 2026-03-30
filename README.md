# ASDSL Framework V2

**Asynchronous Salience-Driven Speculative Lookup Framework**

ASDSL Framework V2 is a research-oriented CPU inference stack for running large decoder-only models (notably **Microsoft Phi-4 multimodal instruct**) with optional **4-bit ASDSL quantization**, **native AVX2/OpenMP GEMV kernels** (PyBind11), and **quantization-cascade speculative decoding (QCSD)**. The primary *reference* inference path lives in Python and PyTorch (`experiments/phi4_cpu_run.py`); optional C++ extensions accelerate fused GEMV and related kernels when built with `setup.py`.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![Hardware](https://img.shields.io/badge/hardware-AVX2%20%7C%20OpenMP-red)](#)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

---

## Table of contents

1. [What this repository contains](#what-this-repository-contains)
2. [The CPU inference bottleneck (context)](#the-cpu-inference-bottleneck-context)
3. [Architecture overview](#architecture-overview)
4. [Native C++ extensions and build flags](#native-c-extensions-and-build-flags)
5. [Phi-4 CPU reference path (`experiments/phi4_cpu_run.py`)](#phi-4-cpu-reference-path-experimentsphi4_cpu_runpy)
6. [Full A/B/C benchmark (`scripts/run_full_benchmark.py`)](#full-abc-benchmark-scriptsrun_full_benchmarkpy)
7. [QCSD: speculative decoding and verification](#qcsd-speculative-decoding-and-verification)
8. [Leviathan guardrail and `.qcsd_history.json`](#leviathan-guardrail-and-qcsd_historyjson)
9. [Threading and OpenMP tuning](#threading-and-openmp-tuning)
10. [Project structure (accurate layout)](#project-structure-accurate-layout)
11. [Getting started](#getting-started)
12. [Tests](#tests)
13. [Roadmap and limitations](#roadmap-and-limitations)

---

## What this repository contains

This is **not** a single monolithic “drop a `.pyd` and forget Python” engine. It combines:

| Layer | Role |
|--------|------|
| **`experiments/phi4_cpu_run.py`** | End-to-end Phi-4 text inference: load local `safetensors`, ASDSL quantize projections, `KVHistory`, RoPE, GQA, MLP, greedy `generate`, optional **QCSD** (`generate_qcsd`). |
| **`asdsl/kernels/`** | Python APIs (`gemv_q4_packed`, etc.) plus optional **native** modules (`_native_gemv`, `_native_forward`, …) compiled from `asdsl/kernels/native/*.cpp` and `forward_loop.cpp`. |
| **`scripts/run_full_benchmark.py`** | **Simulator** mode (no weights) or **`--phi4`** mode: seven-row throughput table (Profiles **A, C, D, E, F, G, B**), Leviathan QCSD gate, optional history-backed α, RSS and footprint telemetry. |
| **`asdsl/speculative/`** | Dual-model / simulated speculative decoding helpers used by benchmarks and tests. |

If the native extensions are not built, ASDSL falls back to **NumPy** or **PyTorch** paths where implemented; behavior remains correct but slower.

**Scope:** The rest of this README is a **deep technical reference** for the **Phi-4 CPU runner**, **A/B/C benchmark**, **QCSD + Leviathan + history file**, **native packed-Q4 GEMV (including batched verify)**, **compiler flags**, and **thread/OpenMP tuning**, with an accurate repo layout. It does **not** exhaust every module under `asdsl/` (e.g. all quantization research utilities, eval scripts, or LUT experiments)—use this file as the spine, then follow imports, `setup.py`, and `tests/` for subsystems not expanded here.

---

## The CPU inference bottleneck (context)

Decoder steps are dominated by **memory bandwidth** (reading large weight matrices per token) and by **framework overhead** (allocations, Python dispatch). This codebase mitigates that by:

- Fused **packed 4-bit GEMV** in C++ (weights stay nibble-packed; dequant + dot in registers) when `_native_gemv` is available.
- Optional **OpenMP** over output rows / layers in those kernels.
- **QCSD** to amortize target work when draft acceptance is high enough (guarded analytically).

Exact tok/s depend on hardware, thread counts, and whether native GEMV is linked.

---

## Architecture overview

### Data flow (Phi-4 reference script)

1. **Weights**: Read from `models/phi4-multimodal-instruct/` via `safetensors` + index JSON (`model.safetensors.index.json`).
2. **`WeightStore`**: Holds per-layer projections as packed Q4 (or uint8 paths), scales/biases, optional **draft** bank for QCSD, and an **`lm_head`** tensor.
3. **Forward**:
   - **AR (`generate`)**: one token at a time; each linear uses `matvec` → `gemv_q4_packed` (native) or dequant + PyTorch depending on flags.
   - **Batched verify (`forward_layer_batch` + `matmul_batch`)**: for QCSD verification, a **batch of hidden rows** is multiplied by weights; when `bits == 4` and `_use_native_gemv`, **`_matmul_q4_packed_batch`** calls **`gemv_q4_packed` with a 2D `x`** so the **entire batch is dispatched in one native call** (see below).

### Two different C++ surfaces

- **`asdsl/kernels/forward_loop.cpp`** → module **`asdsl.kernels._native_forward`**: mmap-oriented / GGUF-style **Q4K** helpers and related utilities (not the same code path as `gemv_q4_packed` used by Phi-4’s packed ASDSL layout).
- **`asdsl/kernels/native/gemv_q4_avx2.cpp` + `gemv_q4_kernel.cpp`** → **`asdsl.kernels._native_gemv`**: **packed Q4 fused GEMV** (`gemv_q4_packed`, `gemv_q4_unpacked`, `gemv_q4_avx2_gs64`), CPU feature probes, optional `set_num_threads` / `get_num_threads` when built with OpenMP.

Phi-4 Profile **C** throughput is dominated by **`gemv_q4_packed` → `gemv_q4_packed_impl_v2`** (AVX2 unpack + FMA), not by `forward_loop.cpp`’s Q4K GEMV.

---

## Native C++ extensions and build flags

**File:** `setup.py` (there is **no** `CMakeLists.txt` in this repo; all native code is built through setuptools + PyBind11).

### Compiler flags (all extensions share the same `extra_compile_args` / `extra_link_args`)

**Windows (MSVC)**

- `/O2` — maximize speed  
- `/Ob2` — aggressive inlining  
- `/Oi` — intrinsics  
- `/arch:AVX2`  
- `/fp:fast`  
- `/openmp` — compile-time OpenMP (linker pulls OpenMP runtime; do **not** pass `/openmp` to `link.exe` as a separate link flag)  
- `/EHsc` — C++ exception handling  

**Linux and macOS (GCC/Clang)**

- `-O3`, `-mavx2`, `-mfma`, `-mf16c`, `-ffast-math`, `-std=c++17`  
- `-fopenmp` on **compile and link**  

**macOS note:** Apple’s toolchain often needs **Homebrew `libomp`** and appropriate **`CPPFLAGS` / `LDFLAGS`** if `-fopenmp` fails at link time. See comments at the top of `setup.py`.

### Extension modules (from `setup.py`)

| Python module | Main sources |
|---------------|----------------|
| `asdsl.kernels._native_forward` | `asdsl/kernels/forward_loop.cpp` |
| `asdsl.kernels._native_gemv` | `native/gemv_q4_avx2.cpp`, `native/gemv_q4_kernel.cpp` |
| `asdsl.kernels._native_gemv_q8` | `native/gemv_q8_avx2.cpp` |
| `asdsl.kernels._native_gemv_q3` | `native/gemv_q3_avx2.cpp` |
| `asdsl.kernels._native_gemv_q2` | `native/gemv_q2_avx2.cpp` |
| `asdsl.kernels._native_sparse_gemv` | `native/gemv_sparse_avx2.cpp` |
| `asdsl.kernels._native_lut` | `native/lut_avx2.cpp` |
| `asdsl.kernels._native_inference` | `native/inference_engine.cpp` |

**Build:**

```bash
pip install pybind11
python setup.py build_ext --inplace
```

---

## Phi-4 CPU reference path (`experiments/phi4_cpu_run.py`)

### Responsibilities

- Loads **Phi-4 multimodal instruct** weights from disk (see `MODEL_DIR`, `INDEX_FILE`).
- Builds **RMSNorm**, **RoPE** (partial rotary dim), **GQA**, **SiLU MLP**, **LM head**.
- Implements **`KVHistory`** (FP32 K/V per layer) and optional **ASDSL KV tracker** for diagnostics.
- **`generate`**: standard greedy autoregressive decoding; records optional `bench_metrics_out` dicts (`tokens_per_second`, etc.).
- **`generate_qcsd`**: draft model (e.g. 2-bit bank) + **batched target verify**; see [QCSD](#qcsd-speculative-decoding-and-verification).

### Native GEMV selection

- **`WeightStore._use_native_gemv`**: when `True` and bits/layout match, **4-bit primary** uses **`asdsl.kernels.gemv_q4.gemv_q4_packed`** (C++ if available).
- **`_matmul_q4_packed_batch`**: if `_use_native_gemv`, builds NumPy views of packed weights + `(batch, cols)` activations and calls **`gemv_q4_packed`** once; applies the **same outlier correction** loop as single-vector `matvec` when outlier tables exist. Otherwise falls back to **unpack + `torch.mm`**.

### Threading — `set_thread_count(n)`

**Location:** top of `phi4_cpu_run.py`.

- Sets **`OMP_NUM_THREADS`**, **`MKL_NUM_THREADS`**, **`OPENBLAS_NUM_THREADS`**, **`VECLIB_MAXIMUM_THREADS`**, **`NUMEXPR_NUM_THREADS`** to `str(n)`.
- Calls **`torch.set_num_threads(n)`** and CPU float flush config.
- If **`asdsl.kernels._native_gemv`** is importable and exposes **`has_openmp`** and **`set_num_threads`**, calls **`_native_gemv.set_num_threads(n)`** so the OpenMP runtime matches the environment **before** native GEMV runs.

**Auto mode (`n <= 0`):** `n = max(1, (os.cpu_count() or 4) // 2)` — half of **logical** CPUs (minimum 1), aimed at reducing hyperthread contention on bandwidth-bound kernels while still allowing **`--threads N`** to override.

**CLI:** `python experiments/phi4_cpu_run.py --threads 0` uses auto; positive `N` fixes thread count. After auto, `main` overwrites the displayed `args.threads` with `int(os.environ["OMP_NUM_THREADS"])` for consistent logging.

---

## Full A/B/C benchmark (`scripts/run_full_benchmark.py`)

### Modes

1. **Default (no `--phi4`)** — **simulator** using `asdsl.speculative.dual_model` and synthetic token lists; analytical footprints and optional `--verify-leviathan-apples`.
2. **`--phi4`** — loads **`experiments/phi4_cpu_run`** only after **`_require_phi4_index_or_exit`**: verifies `models/phi4-multimodal-instruct/model.safetensors.index.json` exists to avoid expensive imports when weights are missing (**exit code 2**, message on stderr).

### Phi-4 run sequence (`run_phi4_benchmark`)

1. **`phi4.set_thread_count(threads if threads > 0 else 0)`** — so **`--threads 0`** triggers auto half-logical-CPU policy.
2. Load **`WeightStore`**, `load()`, `warm_cache()`, record RSS after load.
3. **Leviathan gate α:**  
   - If **`.qcsd_history.json`** contains valid `acceptance_rates`, **mean of all entries** is **`alpha_for_leviathan`**.  
   - Else **`alpha_for_leviathan = phi4_acceptance_estimate`** (CLI default **0.40**).  
   **`_qcsd_break_even_ok(alpha_for_leviathan, …)`** may set **`store._enable_qcsd = False`** and fall back to AR for Profile B.
4. **Profile A:** `store._use_native_gemv = False`, **`phi4.generate`**, capture `bench_metrics_out`, peak RSS.  
5. **Profile C:** `store._use_native_gemv = True`, **`phi4.generate`** again (same AR path as A, **native Q4 GEMV on**). Footprint model matches A (primary + FP32 KV estimate).  
6. **Profile B:** `store._use_native_gemv = True`; **`phi4.generate_qcsd`** if QCSD still enabled, else **`phi4.generate`**.  
7. If QCSD ran and metrics include **`acceptance_rate`**, **`_append_qcsd_acceptance_rate(root, rate)`** appends to **`.qcsd_history.json`** (cap **256** entries).

### Printed table

- **A:** AR + PyTorch matvec path + FP32 KV estimate.  
- **C:** AR + native Q4 GEMV + FP32 KV estimate (isolates kernel vs Python/QCSD).  
- **B:** Native GEMV + QCSD + Q4 KV estimate, or native AR if QCSD disabled.  

Also prints **effective thread count** from `OMP_NUM_THREADS`, **QCSD greedy acceptance**, **verify telemetry** (when QCSD), **Leviathan S** with **`alpha_gate`**, and **append notice** when history was updated.

### Notable CLI flags

| Flag | Meaning |
|------|---------|
| `--phi4` | Real Phi-4 benchmark (requires local index/weights). |
| `--threads` | Default **0** (auto). Pass **N** to override. |
| `--phi4-acceptance-estimate` | Prior α when history empty (default **0.40**). |
| `--gamma` / draft-k | Draft width for QCSD and Leviathan **g**. |
| `--verify-leviathan-apples` | Simulator: compare timed QCSD vs AR to analytical S. |

---

## QCSD: speculative decoding and verification

**Entry:** `generate_qcsd` in `experiments/phi4_cpu_run.py`.

### Phases (per decode iteration)

1. **Draft:** `run_forward(..., use_draft=True)` for up to **`draft_k`** steps on the **draft** weights (sequential small model steps). KV is snapshotted and restored so draft does not corrupt target cache.
2. **Verify (target, batched):** builds **`hidden_batch`** from **`verify_tokens`** (current greedy token + draft prefix aligned for greedy check), then **one stack** over **`NUM_LAYERS`**: **`forward_layer_batch`** → **`matmul_batch`** on the **primary** store. This is **not** `k` full separate target forwards for the stacked verify; it is one batched pass per layer.
3. **LM head:** **`lm_head_matmul_batch`** on the batch.
4. **Accept/reject:** greedy comparison of draft vs target argmax; **KV trim** + optional **`run_forward`** for **correction** or **continuation** when all draft tokens match.

### Telemetry (debugging serial-verify misconceptions)

Counters (per full decode):

- **`_verify_calls`**: incremented **once per speculative cycle** when the **batched** verify stack starts (expect **≈ number of speculative cycles**, not **`draft_k`** per cycle).  
- **`_verify_extra_run_forward`**: target **`run_forward`** after verify (correction / all-accepted tail).  

Printed at end of QCSD and echoed into **`bench_metrics_out`** as:

- `_verify_calls`, `qcsd_verify_batched_passes`, `qcsd_speculative_cycles`, `qcsd_verify_extra_run_forward`, `acceptance_rate`, `tokens_per_second`, etc.

---

## Leviathan guardrail and `.qcsd_history.json`

**Theory (Leviathan et al., 2023):** speculative speedup **S** as a function of acceptance **α**, draft length **g**, and cost ratio **c** (here modeled as **`draft_mb / target_mb`**).

**Gate:** QCSD is enabled only if **`S ≥ 1.01`** (configurable inside `_qcsd_break_even_ok`). Failure prints an analytical message including a binary-search hint for **min α** at **1.05×**.

**Adaptive α (implemented):**

- **File:** repo-root **`.qcsd_history.json`**, shape `{"acceptance_rates": [ ... ]}`.
- **Read:** before the gate, if any valid rates exist, **`alpha_for_leviathan = mean(rates)`**.
- **Write:** after a Phi-4 run where QCSD actually executed, the **measured greedy `acceptance_rate`** is **appended** (bounded list).
- **Cold start:** if history is empty, CLI prior default is **0.40** (`--phi4-acceptance-estimate`). If the gate disables QCSD, **no** acceptance is appended until a run actually completes QCSD.

This prevents a fixed optimistic prior (e.g. 0.70) from keeping QCSD on when measured acceptance is ~0.14.

---

## Threading and OpenMP tuning

| Mechanism | Where |
|-----------|--------|
| Environment variables | `set_thread_count` in `phi4_cpu_run.py` |
| PyTorch intra-op threads | `torch.set_num_threads` |
| Native OpenMP team size | `_native_gemv.set_num_threads` when built with OpenMP |
| Benchmark default | `scripts/run_full_benchmark.py` **`--threads` default 0** → auto **half logical CPUs** |

Tuning guidance: for bandwidth-bound GEMV, **too many OpenMP threads** can hurt; compare Profile **C** vs **A** while sweeping **`--threads`**.

---

## Project structure (accurate layout)

```text
asdsl-framework/
├── asdsl/
│   ├── kernels/
│   │   ├── forward_loop.cpp          # _native_forward (mmap/Q4K-style paths)
│   │   ├── native/
│   │   │   ├── gemv_q4_avx2.cpp      # pybind + gemv_q4_packed_impl_v2, batched API
│   │   │   ├── gemv_q4_kernel.cpp   # gemv_q4_avx2 (gs64) + OpenMP on rows
│   │   │   ├── gemv_q4_kernel.h
│   │   │   ├── gemv_q8_avx2.cpp, gemv_q3_avx2.cpp, gemv_q2_avx2.cpp, …
│   │   │   └── …
│   │   ├── gemv_q4.py               # Python entry: 1D or 2D x, NumPy fallback
│   │   └── __init__.py
│   ├── speculative/                 # dual_model / simulators for benchmarks
│   ├── quantization/
│   └── …
├── experiments/
│   ├── phi4_cpu_run.py              # Main Phi-4 CPU reference + QCSD
│   └── phi4_integration.py          # Setup / download guidance (see script)
├── scripts/
│   └── run_full_benchmark.py        # A/B/C benchmark, Leviathan, history
├── tests/
│   ├── test_run_full_benchmark_preflight.py
│   ├── test_leviathan_qcsd.py
│   ├── test_gemv_q4_batched.py      # Batched packed + gs64 vs loop
│   └── …
├── models/                          # Local Phi-4 weights (user-provided)
│   └── phi4-multimodal-instruct/
│       └── model.safetensors.index.json   # required for --phi4 fast-fail
├── setup.py                         # Native extension build flags
├── pyproject.toml
└── .qcsd_history.json               # created at repo root after QCSD benchmark runs (optional)
```

**Note:** Older README references to `run_phi4_benchmark.py` / `quantize_phi4_mock.py` at repo root are **obsolete**; use **`experiments/phi4_cpu_run.py`** and **`scripts/run_full_benchmark.py`** instead.

---

## Getting started

### Prerequisites

- Python **3.10+**
- PyTorch, transformers, safetensors (see `requirements.txt` / `pyproject.toml`)
- For **native** speedups: **MSVC Build Tools** (Windows) or **GCC/Clang** with OpenMP (Linux; macOS may need `libomp`)

### Install and build extensions

```bash
cd asdsl-framework
pip install -r requirements.txt   # or pip install -e ".[dev]"
pip install pybind11
python setup.py build_ext --inplace
```

### Phi-4 weights

Place checkpoints under **`models/phi4-multimodal-instruct/`** so that **`model.safetensors.index.json`** exists. Use **`python experiments/phi4_integration.py`** (or your own download flow) per that script’s instructions.

### Run inference

```bash
python experiments/phi4_cpu_run.py --bits 4 --prompt "Hello" --max-new-tokens 32 --threads 0
python experiments/phi4_cpu_run.py --qcsd --bits 4 --draft-bits 2 --draft-k 7
```

### Run full benchmark

```bash
python scripts/run_full_benchmark.py --phi4 --max-new-tokens 64 --threads 0
```

---

## Tests

| Test file | Focus |
|-----------|--------|
| `tests/test_run_full_benchmark_preflight.py` | Phi-4 index fast-fail; module load |
| `tests/test_leviathan_qcsd.py` | Leviathan S and break-even helpers |
| `tests/test_gemv_q4_batched.py` | Batched `gemv_q4_packed` / `gemv_q4_avx2_gs64` vs sequential |
| `tests/test_fused_gemv.py` | Q8 fused path (when built) |
| `tests/test_speculative_decoding.py` | Speculative decoding Python contracts |
| Others | Q3, cache tiling, STREAM OMP hygiene, Q4 KV, etc. |

```bash
pytest tests/ -q
```

---

## Benchmark configuration (locked at Phase 11)

All benchmark results below use the pinned settings in **`benchmark_config.json`** so cross-phase comparisons stay apples-to-apples (prompt length drives KV size and throughput on this memory-bound stack).

- **Prompt**: `The fundamental theorem of calculus states that`
- **Max new tokens**: 64
- **Threads**: 8
- **draft_k**: 1 for Profile G EAGLE-3 (chosen by `scripts/choose_draft_k.py` from MTP **test_top1**; QCSD Profile B uses the same `draft_k` from the config for Leviathan)
- **Inter-profile sleep**: 3 s (overridable with `ASDSL_PROFILE_SLEEP`)

CLI flags that override these values print **`[CONFIG] WARNING:`** unless you pass **`--override-config`**.

---

## Benchmark results

Hardware: Intel Core (Raptor Lake, Family 6 Model 186), 12 physical cores, 16.9 GB RAM, AVX2, Windows 11, no GPU. **Phase 12** (2026-03-30): same canonical command and **`benchmark_config.json`** as Phase 11. **EAGLE-3 change:** when every draft in a cycle matches the target verify, the bonus logits and `_last_final_hidden` now come from a **single-row `forward_layer_batch`** over `draft[-1]` instead of **`run_forward`**, so the all-accept path no longer uses the serial forward wrapper; reject cycles still call **`run_forward(correction)`** once. Load+quantize parent process ~**1093 s** on the recorded run.

## Final results

| Profile | Configuration                            | tok/s | vs baseline | vs llama.cpp |
|---------|------------------------------------------|-------|-------------|--------------|
| A       | PyTorch baseline                         | 2.18  | 1.00×       | 0.31×        |
| C       | Native Q4 GEMV (AVX2 FMA)               | 2.40  | 1.10×       | 0.34×        |
| D       | LUT vpshufb (slower on Raptor Lake†)     | 1.74  | 0.80×       | 0.25×        |
| E       | SliM 2.2-bit + LUT (4/32 layers)        | 1.67  | 0.77×       | 0.24×        |
| F       | FATReLU 85% FFN sparsity                | 2.98  | 1.37×       | 0.43×        |
| G       | FATReLU + EAGLE-3 MTP (draft_k=1)       | 1.22  | 0.56×       | 0.17×        |
| B       | Legacy QCSD (2-bit draft bank)          | 0.92  | 0.42×       | 0.13×        |

†Profile D is slower than Profile C on this hardware because `_mm_i32gather_ps` latency (~20 cycles) on Raptor Lake often outweighs the vpshufb shuffle path. The LUT approach tends to pay off more on AMD Zen 4 or ARM Neoverse class cores.

**llama.cpp Q4_K_M reference (same hardware class): ~7.0 tok/s**

EAGLE-3 acceptance rate: **~11.1%** (Profile G subprocess, Phase 12 run). Mean tokens accepted per cycle: **~0.44**. Decode summary prints **`reject_run_forward`** vs **`bonus_1row_batch`** per run. Leviathan gate for **G** at **draft_k=1**: **FAIL** (break-even α ~**22.1%**). **Profile G remains below Profile F** on this run (1.22 vs 2.98 tok/s).

### Performance notes (Phase 12)

- **Phase 11 → 12:** Throughput varies with cold load and RSS; compare **F** and **G** from the **same** benchmark session (table above).
- **EAGLE-3:** An earlier **(L+1)-wide verify every cycle** experiment regressed **G** by doing useless verify rows on reject-heavy runs; production fix keeps **L-row** verify and moves the all-accept continuation to a **dedicated 1-row batch** (same matmul work as the old `run_forward` bonus, without widening verify).
- **QCSD Profile B** still reports **`extra target run_forward after verify`** from **`generate_qcsd`**; that path is unchanged in Phase 12.

## Known limitations

- **EAGLE-3 throughput**: At **~11%** acceptance, **G < F**; need higher α (more/broader MTP training, larger draft_k only if α supports it) or cheaper verify to beat **F** and approach llama.cpp.
- **SliM calibration**: Quick-mode metadata calibrates only 4/32 layers; full calibration should shrink footprint and may change Profile E quality and speed.
- **Full SliM + FATReLU combined**: Profiles E and F are separate; stacking both is future work.

---

## Roadmap and limitations

- **QCSD** is guarded by **Leviathan** + optional **history file**; low acceptance keeps QCSD off to avoid slowdowns.
- **Profile C** is the right knob to compare **native GEMV** vs **PyTorch matvec** without speculative decoding.
- **L2 cache tiling** inside packed GEMV is a possible future optimization if profiling still shows DRAM-bound behavior after flags + thread tuning.
- **`forward_loop.cpp`** Q4K paths are separate from **Phi-4 packed Q4** in **`gemv_q4_*`**; do not assume one optimization applies to the other.

---

**License:** Apache-2.0. See [LICENSE](LICENSE).
