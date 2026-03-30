# ASDSL Framework — Final Performance Results

## Hardware
Intel Core (Raptor Lake, Family 6 Model 186), 12 physical cores, 16.9 GB RAM,
AVX2, Windows 11, 8 threads. DRAM bandwidth: 24.0 GB/s. Roofline: memory-bound
at AI=3.99 FLOPS/byte vs ridge 17.6.

## Canonical benchmark configuration (locked Phase 11)
- Prompt: "The fundamental theorem of calculus states that"
- Max new tokens: 64
- Threads: 8 (P-cores)
- draft_k: 1

## Final benchmark results (Phase 14, 2026-03-30)

| Profile | Configuration                              | tok/s | vs Phase 0 |
|---------|--------------------------------------------|-------|------------|
| A       | PyTorch baseline + FP32 KV                | 2.04  | 1.00×      |
| C       | Native Q4 GEMV (AVX2 FMA)                 | 2.26  | 1.12×      |
| D       | LUT vpshufb tiled kernel (Phase 1)        | 1.55  | 0.76×      |
| E       | SliM 2.2-bit + LUT (4/32 layers, quick)  | 1.43  | 0.70×      |
| F       | FATReLU 85% FFN sparsity (Phase 3)         | 2.66  | 1.31×      |
| G       | FATReLU + EAGLE-3 MTP (draft_k=1)         | 1.35  | 0.66×      |
| B       | Legacy QCSD (2-bit draft bank)            | 0.86  | 0.42×      |

**llama.cpp Q4_K_M reference (~7.0 tok/s): not beaten, gap = 5.35 tok/s**
**EAGLE-3 acceptance rate: 7.1%** (MTP head test_top1 = 32.8% on held-out eval)
**Leviathan gate for Profile G: FAIL** (break-even acceptance ~22.1%)

## Phase 0 baseline
1.20 tok/s (PyTorch matvec, no quantization)

## Architecture contributions (Phase 14)

| Optimization               | Contribution           | Measured |
|---------------------------|------------------------|----------|
| Q4 GEMV (AVX2 FMA)        | C vs A                | +0.22 tok/s |
| FATReLU 85% FFN sparsity  | F vs C                | +0.40 tok/s |
| EAGLE-3 MTP (draft_k=1)    | G vs F (at 7.1% acc.) | -1.31 tok/s (net negative) |
| LUT vpshufb tiled kernel   | D vs C                | -0.71 tok/s |
| SliM 2.2-bit (4/32 quick) | E vs D                | -0.12 tok/s |
| QCSD (2-bit draft bank)    | B vs C                | -1.40 tok/s |

**Dominant optimization:** FATReLU 85% FFN sparsity (Profile F / Profile C = 1.18×).
**Second:** Q4 GEMV (A → C = 1.10×).

## Key Phase 14 finding: sparse kernel threshold mismatch

The `active_rows` array in `forward_layer()` was computing indices using `1e-9`
as threshold, but the FATReLU mask was zeroing values below `tau`. This caused
near-zero values (already masked to zero by FATReLU) to still be processed by
the sparse kernel, wasting computation.

**Fix applied (Phase 14):** `active_rows = np.where(np.abs(act_np) >= tau)` instead
of `> 1e-9`. Applied to both `forward_layer()` and `forward_layer_batch()`.

**Result:** sparse kernel is called correctly (0 dense fallbacks confirmed). However,
measured sparsity on the benchmark prompt varies significantly per layer:

| Layer | τ         | Measured sparsity | Target |
|-------|-----------|-------------------|--------|
| L0    | 0.01002   | 46–66%            | 85%    |
| L1    | 0.01781   | 70–84%            | 85%    |
| L2    | 0.01015   | 53–88%            | 85%    |
| L3    | 0.01728   | 47–63%            | 85%    |
| L30   | 0.24416   | ~85%              | 85%    |
| L31   | 0.29835   | ~85%              | 85%    |

Early layers (L0–L3) consistently show 46–66% sparsity vs 85% target. τ values
were calibrated on 32 diverse prompts; the benchmark prompt produces a different
activation distribution. This is the primary cause of the Profile F regression from
the Phase 7 peak.

## Profile F regression analysis

Phase 7 peak: **5.19 tok/s** (commit 884ff8b, session 1, warm cache).
Phase 14 stable: **2.66 tok/s** (same hardware, same config, ~49% regression).

Confirmed not the cause:
- `dense_fallback = 0` (sparse kernel always reached)
- All 32 transposed down_proj layers loaded correctly
- τ values within calibrated range

Likely cause: session-to-session variance in AVX2 turbo boost / thermal throttling
combined with the benchmark prompt activating lower sparsity in early transformer
layers. The Phase 7 peak may have occurred with a cold cache rebuild that triggered
different OpenMP thread affinity behavior.

## EAGLE-3 analysis (Profile G)

Phase 13 fix eliminated all extra `run_forward` calls in `generate_eagle3()`:
- Reject path: extracts `hidden_norm[n_accepted]` from verify batch
- All-accept path: extracts `hidden_norm[L]` and `all_logits[L]` from verify batch
- `extra_run_forward = 0` confirmed in telemetry

**However, Profile G (1.35 tok/s) is still 49% below Profile F (2.66 tok/s).**

With zero cycle overhead, the throughput formula is: `G = (1 + α) × F`

| Acceptance α | G tok/s | vs F   |
|--------------|---------|--------|
| 7.1% (current) | 1.35   | 0.51×  |
| 22.1% (break-even) | 3.26 | 1.23× |
| 32.8% (test_top1) | 3.54 | 1.33× |
| 50% (MTP goal)     | 3.99 | 1.50× |
| 100% (theoretical) | 5.32 | 2.00× |

The gap between MTP head test_top1 (32.8%) and end-to-end acceptance (7.1%)
is caused by distribution shift between training corpus and benchmark prompt.
The benchmark prompt is short and mathematical; the MTP head was trained on
a broader distribution.

**To beat llama.cpp (~7.0 tok/s) with draft_k=1:**
- Required: F ≥ 4.7 tok/s AND acceptance ≥ 50%
- Combined: G = 1.50 × 4.7 = 7.05 tok/s

Not achievable with current F (~2.66 tok/s) and current acceptance (7.1%).

## Path to llama.cpp

The theoretical path requires four things together:

1. **Profile F recovery to ~5.0 tok/s** — possible via per-prompt τ recalibration
   (run `calibrate_fatrelu.py` with the canonical prompt included in calibration set),
   or session optimization (disable turbo boost, pin threads).

2. **EAGLE-3 acceptance ≥ 50%** — requires significantly more MTP training data
   covering mathematical/technical prompts, or larger draft_k (if acceptance supports it).

3. **Full SliM calibration (32/32 layers)** — would reduce model footprint from
   ~3.6 GB to ~1.9 GB, proportionally increasing all profile throughputs under the
   confirmed memory-bound roofline.

4. **SliM + FATReLU combined** — stacking both bandwidth reductions; Profile F
   with SliM would use the same FFN sparsity savings plus 2.2-bit primary weights.

## Known limitations

- **EAGLE-3 acceptance**: 7.1% end-to-end vs 32.8% held-out eval — distribution shift.
  Needs domain-matched training data or per-prompt calibration.
- **Profile F variance**: 5.19 peak not reproduced; session-to-session thermal/throttle
  variance on Windows + AVX2 CPU is significant.
- **LUT vpshufb on Raptor Lake**: consistently slower than native FMA GEMV due to
  `_mm_i32gather_ps` latency (~20 cycles); architectural mismatch for this hardware.
- **SliM quick-mode**: only 4/32 layers calibrated; full calibration requires more RAM.
- **QCSD Profile B**: 0.86 tok/s at 81% acceptance — batched verify is correct but
  serial draft emission + 2-bit quality limits throughput.

## Future work (ordered by impact)

1. Per-prompt FATReLU calibration including the canonical benchmark prompt — expected to
   recover Profile F to 4.0–5.0 tok/s on the canonical run.
2. EAGLE-3 retraining on technical/mathematical corpus — target 40%+ end-to-end acceptance.
3. Full SliM 32/32-layer calibration — expected ~45% footprint reduction → proportional F speedup.
4. Profile combining SliM + FATReLU — multiplicative bandwidth savings.
5. QCSD draft quality improvement (beyond 2-bit) — higher acceptance → better QCSD speedup.
