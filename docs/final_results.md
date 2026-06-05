# ASDSL × Phi-4 CPU Inference — Final Results

## Hardware

Intel Core i7-1360P (4P+8E), 16 logical CPUs, DDR4, Windows 10, CPU-only inference.

## Model

Microsoft Phi-4-multimodal-instruct (~14B parameters), 4-bit asymmetric quantization, UnifiedEngine C++ forward.

## Canonical comparison (Phase 29 parity protocol)

Do **not** use interleaved `variance_study.py` or mixed-thread `llama_cpp_comparison.json` as the headline.

| Config | Weight path | Purpose |
|--------|-------------|---------|
| **C0** | HF safetensors + preq gs=32 | Quality default; historical 11.68 tok/s @ 12t |
| **C1** | Same GGUF + native Q4_K on q4_k layers | **Fair speed comparison** vs llama |
| **L0** | llama-cli on `phi4-mm-Q4_K_M.gguf` | Reference (~13.9 tok/s @ 12t separate-run) |

**Honest gap @ 12t, same prompt, cold CPU:** roughly **15–20%** (~1.5–2.5 tok/s), not the **~30%** from phase24 interleaved runs (10.47 vs 15.0).

| Source | ASDSL | llama | Ratio | Valid for parity? |
|--------|-------|-------|-------|-------------------|
| `final_honest_comparison.json` | 11.68 | 13.9 | 84% | Yes (separate runs) |
| `phase24_variance.json` | 10.47 | 15.0 | 70% | No (interleaved + format mix) |
| `phase27_variance.json` | 8.92 | 13.76 | 65% | No (warm CPU) |
| Phase 28 gs=128 | — | — | — | Exploratory microbench only |

**Run cold parity:** `benchmarks/parity_benchmark.py` + `benchmarks/results/parity_manifest.json`. Reconciliation: `parity_reconciliation.json`.

## Journey (selected phases)

| Phase | Key change | tok/s @ 12t | RAM |
|-------|------------|-------------|-----|
| Start | Baseline | 2.36 | 7.5 GB |
| 18 | Persistent cache | 11.68 | 2.4 GB |
| 21 | GGUF Q4K mixed | 11.06 | 2.4 GB |
| 24 | Restored preq kernel | 10.47 (interleaved study) | 2.4 GB |
| 29 | Parity protocol + Q4KM cache + kernel IPC scaffolding | *cold run pending* | 2.4 GB |

Load time with HF cache: **~3 s**. Q4KM mmap cache: `phi4_q4km_{gguf_sha16}.safetensors`.

## Quality

| Metric | Value |
|--------|-------|
| ASDSL PPL WikiText-64 | **84.34** (`ASDSL_USE_UNIFIED=1`) |
| Ratio vs FP16 on same corpus | ~1.44× |

GGUF mixed path decode works; WikiText PPL not valid on HF embed + mixed weights.

## Bandwidth vs IPC

| Path | GB/token | Effective GB/s @ ~11 tok/s |
|------|----------|---------------------------|
| ASDSL preq | 2.40 | ~26 |
| ASDSL GGUF mixed | 2.28 | ~25 |
| llama.cpp GGUF | 2.587 | ~38.8 @ 15 tok/s |

ASDSL reads fewer bytes but lower effective bandwidth → **kernel IPC gap**.

## Kernel roadmap (Phase 29)

1. **Q4_K:** `gemv_q4km_q8_avx2` — superblock prefetch (`ASDSL_Q4KM_PREFETCH_BLOCKS=2`), 8-row unroll on gate_up-sized GEMV.
2. **Q5_K / Q6_K:** stubs in `gemv_q5_q6_stubs.cpp`; engine flags `qkv_q5km` / `down_q6km`; preq fallback until native dots land.
3. **Roofline:** `benchmarks/roofline_diagnosis.py` → `parity_roofline.json`.

## Artifacts

- `benchmarks/parity_benchmark.py`, `benchmarks/results/parity_manifest.json`
- `benchmarks/results/parity_reconciliation.json`, `benchmarks/results/phase29_parity_results.json`
- `benchmarks/results/phase27_results.json` (thermal / prefetch notes)
- `benchmarks/variance_study_separate.py` (legacy separate-session helper)
- `benchmarks/test_q4km_correctness.py`
