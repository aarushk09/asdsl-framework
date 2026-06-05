# ASDSL Framework

CPU inference stack for Phi-4 with quantized GEMV, LUT-native paths, and Phase 3 dispatch.

## Quick Start

```bash
# 1. Quantize + load (4-bit)
python experiments/phi4_cpu_run.py --bits 4 --lut --max-new-tokens 32

# 2. Calibrate kernel tags (writes projection_profiles.json)
python experiments/phi4_cpu_run.py --bits 4 --lut --calibrate --max-new-tokens 32

# 3. Run with dynamic dispatch (LUT / AVX2 / SPARSE)
python experiments/phi4_cpu_run.py --bits 4 --dispatch --max-new-tokens 32

# Build LUT AVX2 extension (required for LUT path)
python asdsl/kernels/setup_lut.py build_ext --inplace
```

## Performance Summary

Model: **Phi-4-multimodal-instruct** (14B) | Hardware: Intel i7 Raptor Lake, 15.75 GB DDR4-3200, 8 threads, Windows 10

### Throughput

Measured with `comprehensive_bench.py` (decode tokens 2–128):

| Mode | tok/s | RAM | Notes |
|------|-------|-----|-------|
| Baseline (no optimizations) | 2.36 | ~4.4 GB | Phase 1 starting point |
| + numpy forward stack | 2.94 | ~4.0 GB | Phase 7: torch→numpy boundary |
| + kernel optimizations | 3.57 | ~4.0 GB | Phase 8: RoPE + lm_head |
| + dispatch (packed path) | **3.69** | ~4.0 GB | Phase 9–10 final |
| + UnifiedEngine + preq cache | **7.14** | ~2.0 GB | Phase 12: mmap weight+preq cache, KV fix |
| + fused activation Q8 in GEMV | **~7.2** | ~2.0 GB | Phase 14: `ASDSL_FUSED_GEMV=1` (default), activation profile 0.8 ms |
| + multi-row preq refactor (4/8-row) | **~7.4** | ~2.0 GB | Phase 15: explicit `gemv_q4_32_preq_*row_*`; production already had 4-row weight unroll over shared `x_q8` |
| + 12 OMP threads (vs 8) | **~9.2–10.3** | ~2.0 GB | Phase 16: `OMP_NUM_THREADS=12` on Raptor Lake P-cores |
| + Q4_K kernel (llama dot + block_q8_K) | **21+ GB/s** microbench | — | Phase 17: `gemv_q4km_q8_avx2` parity with preq on o_proj; mixed GGUF native for q4_k layers |
| + persistent weight/preq cache | **11.7 tok/s** @ 12t | ~2.4 GB mmap | Phase 18–19: load **~3 s**; see honest table below |
| llama.cpp Q4_K_M (same machine)* | **12.8–13.9** @ 8–12t | ~2.4 GB GGUF | Phase 19 standalone; **parity** via `parity_benchmark.py` |

\*Canonical comparison (Phase 29): **separate sessions**, **12 threads both**, prompt `<|user|>The<|end|><|assistant|>`, 128 tokens. Until a cold parity run, use **11.68 vs 13.9 tok/s (84%)** from `final_honest_comparison.json` — not interleaved `phase24_variance` (70%) or stale 101% JSON. **gs=128** is exploratory only (`phase28_results.json`).

```bash
python benchmarks/parity_benchmark.py --config C0,L0   # cold day: canonical headline
python benchmarks/compare_llama_cpp.py --threads 12  # single-shot; errors if -t mismatch

# Close-the-gap paths (after python setup.py build_ext --inplace):
set ASDSL_USE_Q8_GEMV=1          # Q4×Q8 integer GEMV per projection (no UnifiedEngine)
set ASDSL_USE_UNIFIED=1          # C++ UnifiedEngine forward (preq Q4_32, best throughput)
set ASDSL_FUSED_GEMV=1           # default: FP32 norms + quant inside preq GEMV (Phase 14)
set ASDSL_GEMV_UNROLL=8          # optional: 8-row fused GEMV (4-row default; test on gate_up)
set OMP_NUM_THREADS=12           # Phase 18: 11.75 tok/s full decode (106% of llama.cpp on this box)
set ASDSL_USE_Q4KM_GGUF=1        # optional: Q4_K GEMV for q4_k GGUF layers; q5_k/q6_k stay on preq
python benchmarks/build_caches.py  # once if models/phi4_weight_cache/ is empty (~37 min)
python benchmarks/diagnose_load.py  # should finish in <20s after caches exist
python benchmarks/test_q4km_correctness.py
# Warm start: unset PHI4_NO_WEIGHT_CACHE; preq blocks restore from models/phi4_weight_cache/phi4_preq_*_q4_32.safetensors
python benchmarks/test_preq_correctness.py   # ASDSL_KEEP_PACKED=1 for 4-row vs preq check
set ASDSL_4BIT_OUTLIERS=1        # SpQR outliers at 4-bit (quality)
python benchmarks/kernel_bench_compare.py
python benchmarks/test_preq_synthetic.py
```

### Quality

Relative quantization quality (WikiText-2, first 64 tokens, same tokenizer and model):

| Eval | Score | Notes |
|------|-------|-------|
| Prose sentence (~20 tokens, 4-bit + dispatch) | 17.0 | Healthy; confirms forward correctness |
| WikiText-64, 4-bit + dispatch | 138.6 | High on raw wiki; instruct model |
| WikiText-64, FP16 | 96.5 | Reference unquantized path |
| **4-bit / FP16 ratio (WikiText-64)** | **1.44** | ~44% PPL degradation vs FP16 |
| WikiText-64, 8-bit | 96.5 | Matches FP16 (within 0.02%) |

Absolute WikiText PPL (e.g. 622 on 512 tokens) is **not** a valid quality gate for this instruct multimodal checkpoint. The Phase 3 figure **PPL = 11.90** is **not reproducible** on the current stack (same model and tokenizer now yield ~138 on WikiText-64). Use **4-bit within 10% of FP16** on a fixed corpus as the quality target (`ppl_4bit / ppl_fp16 ≤ 1.10`); current WikiText-64 ratio is ~1.44.

See `benchmarks/results/final_summary.json`, `benchmarks/results/phase10_measurements.json`, and `benchmarks/results/phase10_baseline.json`.

### Performance vs llama.cpp (Intel i7-1360P, Windows 10, CPU-only)

**Two baselines — do not mix them:**

| Label | tok/s @ 12t | vs llama | When measured | Trust for today |
|-------|-------------|----------|---------------|-----------------|
| Phase 19 peak (warm) | 11.68 | 84% | single standalone run | historical best |
| **Cold parity C0 (current)** | **~9.0** | **~65%** | 5-run `parity_benchmark.py` | **use this** |
| llama.cpp L0 | 13.9 | 100% | Phase 19 | reference |

Phase 27 cold E2E already showed **8.84–8.92 tok/s** on committed code; your post-revert run (**9.05 ± 0.65**) matches that, not the older 11.68 peak.

Run C0 (no `ASDSL_USE_Q4KM_GGUF`, no `--gguf-path`):

```powershell
Remove-Item Env:ASDSL_USE_Q4KM_GGUF -ErrorAction SilentlyContinue
$env:ASDSL_USE_UNIFIED = "1"
$env:ASDSL_FUSED_GEMV = "1"
$env:ASDSL_PREQ_G4FUSED = "0"
$env:ASDSL_AFFINITY = "legacy"
$env:ASDSL_PREQ_PREFETCH_GROUPS = "0"
$env:OMP_NUM_THREADS = "12"
python benchmarks/parity_benchmark.py --config C0 --runs 5 --cooldown 30 --asdsl-only
```

Phase 30 full native GGUF (C2) reverted after **0.18 tok/s**; do not enable until re-validated.

Full write-up: [`docs/final_results.md`](docs/final_results.md). Canonical numbers: `benchmarks/results/final_honest_comparison.json`.

### Key implementation notes

- **LUT kernel** (nCPU-inspired): correct and tested; not active on Phi-4 (projection tiles exceed L2 budget). Effective when hidden_dim ≤ ~512.
- **Sparse GEMV**: correct; profiles may tag SPARSE but inference defaults to AVX2 unless `ASDSL_SPARSE_INFER=1`.
- **Learned correction MLPs**: implemented (32 layers, ~320 KB); production training needs a real calibration corpus.
- **RoPE**: native tables use partial rotary dim 96 (48-wide cos/sin rows); required for numpy fast path (Phase 9).
- **Preq GEMV (Phase 15/23)**: `gemv_q4_preq.cpp` — OpenMP 4-row unroll, one `x_q8` quant per GEMV. Optional **4-group fusion** via `gemv_q4_32_preq_g4fused_4row_avx2` / `ASDSL_PREQ_G4FUSED=1` (default off after Phase 23 microbench). See `phase15_results.json`, `phase23_results.json`.
- **Q4_K GEMV (Phase 16–17)**: Phase 16 fixed layout (`get_scale_min_k4`, 64-weight lanes). Phase 17 ports llama.cpp `ggml_vec_dot_q4_K_q8_K` AVX2 (`maddubs` + `block_q8_K` activations) and fixes `get_scale_shuffle_k4` indexing — **~21 GB/s** on o_proj (was ~6.7 GB/s). Mixed GGUF path: `q4_k` → `gemv_q4km`, `q5_k`/`q6_k` → preq. See `benchmarks/results/phase17_results.json`.
- **Load / E2E (Phase 18–22)**: HF cache via `benchmarks/build_caches.py`. **~11.5 tok/s** preq @ 12t (warm). llama.cpp **13.72 ± 0.07** @ 12t. Q4KM `bsums` already match llama; E2E uses preq fused GEMV — parity needs preq inner-loop IPC, not format swap. See `docs/final_results.md`.

LUT microbench 256×512 targets **< 0.5 ms** with `tile_groups=64` (synthetic; Phi-4 projections exceed L2 LUT tile budget).

## Model weights

Phi-4 CPU inference expects safetensors at:

`models/phi4-multimodal-instruct/model.safetensors.index.json`

Override with `ASDSL_MODEL_DIR` if the checkout lives elsewhere.

First-time download:

```bash
python benchmarks/run_phase3_e2e.py --download-model
```

## Benchmarks

```bash
# Phase 3 success criteria (microbench + E2E PPL/tok/s + kernel assignments)
python benchmarks/run_phase3_e2e.py --max-tokens 512 --threads 8

# Or via comprehensive bench
python benchmarks/comprehensive_bench.py --bits 4 --dispatch --max-tokens 512
```

Results: `benchmarks/results/phase3_results.json`

## Tests

```bash
pytest tests/test_lut_avx2.py tests/test_lut_kernel.py tests/test_dispatch_policy.py \
  tests/test_lut_gemv_correctness.py tests/test_parallel_quantization.py -q
```

## Documentation

- [Phase 1 LUT kernel](docs/phase1_lut_kernel.md)
- [Phase 2 AVX2 LUT](docs/phase2_avx2_lut.md)
- [Phase 3 dispatch](docs/phase3_dispatch.md)
