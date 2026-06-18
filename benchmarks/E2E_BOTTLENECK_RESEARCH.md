# ASDSL E2E Bottleneck — Forensic Research Report

**Date:** 2026-06-14
**Scope:** Why ASDSL Phi-4 CPU decode runs at ~66% of llama.cpp after the GEMV/threading work of Phases 0–6, and what closes the gap.
**Method:** Full read of `unified_engine.cpp`, `gemv_preq2_avx2.cpp`, `gemv_chunked.hpp`, `thread_pool.h`, `omp_pcore_pinning.hpp`, `unified_bridge.py`, `phi4_cpu_run.py`; **three live runs of the built-in `ASDSL_ENGINE_PROFILE` instrumentation** (decode + prefill tokens, two thread configs); cross-reference against llama.cpp/ggml source PRs and 2024–2026 CPU-inference literature.

---

## ⚠️ Headline finding — the task premise is wrong (and the instrumentation proves it)

The brief states the per-token cost is *"GEMV ~25–35 ms, lm_head ~9–12 ms, NON-GEMV ~65–75 ms"* and that *"the answer is in the 65–75 ms of non-GEMV time."*

**A live capture of the engine's own profiler at a true decode token says the opposite:**

```
[ASDSL_ENGINE_PROFILE] decode token 28 (C0, 12 threads physical, div=4, ignore-eos):
  gate_up_gemv 39.29 ms
  down_gemv    18.69 ms
  qkv_gemv     11.68 ms
  o_gemv        7.45 ms          → body GEMV  = 77.11 ms (83.3%)
  lm_head      13.23 ms          → lm_head    = 13.23 ms (14.3%)
  prep_fused    0.44 ms          (fused RMSNorm/SwiGLU, FP32)
  other         1.82 ms          (embed + RoPE + KV-write + attention + residual)
  total        92.61 ms          → GEMV TOTAL = 90.34 ms (97.6%)
```

**Non-GEMV ("other" + prep) = 2.26 ms, i.e. 2.4% of the token, not 65–75 ms.** ASDSL is almost perfectly GEMV-memory-bandwidth bound. The "65–75 ms non-GEMV" figure in the plan is an artifact of the **pre-Phase-3 profiler**, which (per `PHASE_WALKTHROUGH.md` line 89) started its timer at the pre-attention RMSNorm and stopped it only after QKV+RoPE+attention, so it folded the QKV GEMV and the whole attention block into a bucket mislabeled `activation_q8`. The corrected profiler (Phase 3) and today's capture both show the body matmuls dominate.

**Consequence:** every optimization aimed at attention/RoPE/Python/KV is chasing a 2.4% bucket. The remaining gap to llama.cpp lives in **sustained GEMV streaming bandwidth**.

---

## SECTION A — Non-GEMV Time Decomposition (measured + analytical)

Phi-4-mini backbone constants (confirmed in code and the Phi-4-mini technical report): `L=32`, `hidden=3072`, `n_q=24`, `n_kv=8`, `head_dim=128`, `inter=8192`, `vocab=200064`, `rotary_dim=96` (75% rotated), tied fp16 embedding.

Two live captures anchor the numbers:

| Bucket | Prefill token (tok 3, no lm_head) | Decode token (tok 28, w/ lm_head) |
|---|---|---|
| gate_up GEMV | 31.01 ms | 39.29 ms |
| down GEMV | 17.57 ms | 18.69 ms |
| qkv GEMV | 12.70 ms | 11.68 ms |
| o GEMV | 10.63 ms | 7.45 ms |
| **body GEMV total** | **71.91 ms** | **77.11 ms** |
| lm_head GEMV | 0 (skipped) | 13.23 ms |
| prep_fused (RMSNorm+SwiGLU) | 0.31 ms | 0.44 ms |
| other (embed+RoPE+KV+attn+resid) | 3.75 ms | 1.82 ms |
| **total (C++ forward_token)** | **75.98 ms** | **92.61 ms** |

The per-operation analytical budget reconciles with the measured `other` = ~1.8–3.8 ms:

| Op | Count/token | Bytes touched | Compute | Bound | Est. ms | llama difference |
|---|---|---|---|---|---|---|
| **fp16 embed lookup** | 1 | 3072×2 B = 6 KB read, F16→F32 | 3072 cvt | mem (trivial) | ~0.01 | `get_rows`, identical, ~0 |
| **RMSNorm** | 64 (2/layer + final) | 3072×4 ×2 ≈ 24 KB/call | 2 passes, FMA | mem | ~0.3 total | fused into norm op; same |
| **RoPE** (partial 96/128) | 32 | (24+8)×96 rotated ≈ 3 K elem/layer | FMA pairs | compute (L1-resident) | ~0.3 total | `ggml_rope` vectorized; same order |
| **KV write (Q8 quant)** | 32 | 8 kv-heads×128×2(K,V) = 2 K elem/layer → 2 KB int8 | absmax+quant | mem (L1/L2) | ~0.6 total | fp16 store (no quant); ASDSL slightly more compute, fewer bytes |
| **attention (flash Q8)** | 32 | KV read = pos×8×128×2×1 B; at pos≈28 ≈ 57 KB/layer → 1.8 MB | online softmax + dot | mem, OMP over 24 heads | ~0.5–1.0 | flash/online softmax; equivalent at short ctx |
| **residual add + memcpy** | 64 | 3072×4 ×2/call | add | mem | ~0.3 total | fused in graph; same |
| **Σ non-GEMV** | | | | | **≈ 1.8–2.3 ms** ✓ | matches measured |

**KV-cache scaling (answers Part B-1 and B-6c).** KV read per token = `2·L·n_kv·head_dim·t·b_kv`. With int8 KV (`b_kv=1`): at `t=128` → 8.4 MB → **0.32 ms** @ 26 GB/s; at `t=2048` → 134 MB → **5.2 ms**; at `t=8192` → 537 MB → **21 ms**. KV traffic only equals the 2.01 GB body load at `t ≈ 15,300` tokens (int8) or `t ≈ 7,700` (fp16). **At ctx ≤ 2 K, attention/KV is a rounding error** — which is exactly why the plan correctly shelved Flash-Attention and Q4-KV "at short context." Those become relevant only past ~2–4 K context; StreamingLLM/H2O eviction would matter only past ~8 K.

**Python/bridge overhead.** The pybind `forward_token` (in `unified_engine.cpp`) reuses a single `logits_out_` buffer (no per-token alloc) and wraps the compute in `py::gil_scoped_release`. `unified_forward_token` returns `np.asarray(...)` (a no-copy view of the same dtype). The Python side then does `argmax`/sampling over 200 K logits. Wall-clock per token ranged **89 ms (warm, 11.18 tok/s)** to **118–125 ms (thermally throttled)** across runs vs ~92.6 ms of C++; the implied Python/bridge cost is **~0–15 ms, dominated by thermal variance, not interpreter overhead.** This is small but real and **fully eliminable** (Section E #2).

---

## SECTION B — llama.cpp vs ASDSL non-GEMV implementation comparison

Because non-GEMV is 2.4% of the token, the implementation differences that matter are in the **GEMV streaming machinery**, not in attention/RoPE math. Code-level diffs:

| Dimension | ASDSL (`unified_engine.cpp`) | llama.cpp / ggml | Impact |
|---|---|---|---|
| **Thread model** | A persistent, P-core-pinned, spin-waiting `ThreadPool` exists in `thread_pool.h` **but `UnifiedEngine` instantiates it as `pool_{0}` (zero workers)** and instead opens a fresh `#pragma omp parallel` (MSVC `vcomp`) inside every GEMV and every attention call. | `ggml_threadpool` (PR #7526) — one persistent pool reused across all ops; per-phase pools for decode vs batch; futex *yield barrier* (PR #13079); `--poll` spin control. | **Primary.** ASDSL pays ~161 fork/joins per token + the OS scheduler re-learns placement each op. llama created its threadpool *specifically* to kill this. |
| **Affinity** | `bind_omp_thread_physical_if_enabled()` issues `SetThreadAffinityMask(GetCurrentThread(), mask)` **on every chunk pickup** inside every GEMV's parallel region (≈ `n_threads·div` chunks × 129 GEMVs ≈ thousands of syscalls/token, re-setting the *same* mask). | Threads pinned once at pool creation; affinity mask set once. | ASDSL burns ~3–6 ms/token of redundant kernel transitions; llama: ~0. |
| **Work distribution** | Atomic `current_chunk` work-stealing (`gemv_chunked.hpp`), `chunk_div` configurable (4 default, 6 best in microbench). | Atomic `current_chunk` chunked mul_mat — same idea. | Equivalent algorithm; ASDSL's is sound (and is why E-cores don't stall — see Section C). |
| **Activation quant** | `quantize_activation_avx2` runs **single-threaded on the master** before each GEMV's parallel region (others idle). | Q8_K activation quant also serial-ish but inside the threaded graph op. | Minor bubble (~tens of µs × 129). |
| **Decode loop** | Python `generate_stream` calls into C++ **once per token**; a complete C++ `generate()` exists in `unified_engine.cpp` (with C++ argmax + GIL release) **but is not wired to the canonical path.** | Pure C++ decode loop; no interpreter on the hot path. | Small (~3–15 ms/token) but lossless to remove. |
| **RMSNorm/SwiGLU fusion** | **Already fused** — `residual_add_rmsnorm_quantize_f32`, `swiglu_quantize_inplace`, `residual_add_rmsnorm_f32` (FP32 path for preq2). Measured `prep_fused` = 0.44 ms. | Fused norm/activation in graph. | **Parity already achieved.** No headroom. |
| **RoPE** | AVX2 FMA, partial-rotary, single thread (L1-resident). | `ggml_rope` vectorized. | Equivalent; ~0.3 ms either way. |
| **Weight bpw** | preq2 = 20 B / 32 weights = **5.0 bpw**; lm_head preq2 5.0 bpw. | Q4_K_M = **5.18 bpw** (mixed Q4_K/Q6_K) for this exact GGUF. | Bytes/token are **nearly equal** (~2.4 GB) — the gap is *not* bytes. |

---

## SECTION C — Research findings (Part B questions)

**1. CPU attention.** llama.cpp uses online/flash-style softmax for decode; at `seq_len ≤ 128` it reads <2 MB of KV/token — sub-millisecond. ASDSL's `compute_attention_flash_q8` (online `m,l` state, BLOCK_K=64, int8 KV, OMP over heads) is the same class of algorithm. **Measured attention+KV+RoPE = 1.82 ms.** The "where does attention time go?" question has a concrete answer: **it doesn't — it's 2% of the token.** No 2024–25 attention paper helps at this context length.

**2. OMP launch overhead.** Literature (CLOMP, EPCC, the LLVM libomp issue #195239) puts an OpenMP barrier at **~27k–38k cycles ≈ 9–13 µs** on Intel runtimes; MSVC `vcomp` is in the same range or worse and blocks workers between regions. ASDSL fires **~161 regions/token** → ~1.5–2 ms of pure fork/join, **plus** the redundant `SetThreadAffinityMask` syscalls (~3–6 ms). The persistent `ThreadPool` in the repo (`thread_pool.h`, graduated-backoff `_mm_pause` spin) was built to remove exactly this but is **wired to zero workers**. llama solved it with `ggml_threadpool` (PR #7526) + futex yield barrier (PR #13079). **The solution already exists in-tree; it just isn't connected.**

**3. Python/bridge.** GIL is released during compute; logits buffer is reused; the numpy wrap is a view. Net Python cost ~0–15 ms/token (mostly thermal noise). A wired C++ `generate()` removes it entirely and is lossless.

**4. Fused RMSNorm + Q8 quant.** **Already implemented and measured at 0.44 ms.** No action.

**5. RoPE on CPU.** Already AVX2-vectorized, partial-rotary, L1-resident, ~0.3 ms total. No measurable headroom.

**6. Novel techniques.**
- **(a) B=2 continuous batching.** For the single-stream canonical benchmark there is no second request, so it cannot improve *latency*; it only raises aggregate throughput when ≥2 concurrent users exist (server mode). At B=2 a GEMV becomes a thin GEMM: weights read once, 2 activation columns → ~2× tokens for ~1× weight traffic, since decode is bandwidth-bound. **Verdict: secondary/server-only; does not move the C0 headline.**
- **(b) Double-buffered async weight streaming.** The GEMV is already DRAM-bandwidth-bound at ~26 GB/s vs a ~40 GB/s STREAM ceiling. Prefetch (tried) failed because the limiter is *sustained issue rate*, not latency. Explicit double-buffering can't exceed the memory controller. **Low confidence; characterize only.**
- **(c) KV compression for long ctx.** Breakeven ~8 K (fp16) / ~15 K (int8). Irrelevant to the 128-token benchmark; worth it only for long-context products.
- **(d) C++ `generate()`.** Exists, unwired. Removes Python per-token. **Pursue (cheap, lossless).**
- **(e) Intel AMX.** Absent on i7-1360P. AMX is a *compute* accelerator (int8/bf16 tiles) and decode is *bandwidth*-bound, so AMX helps **prefill/batch**, not batch-1 decode. Worth a code path **only** for a future Meteor/Arrow/Lunar Lake machine and only for prompt processing. **Secondary.**
- **(f) Hybrid P+E core LLM (2025).** Strong external consensus (llama.cpp disc. #572, multiple guides) that **E-cores hurt** memory-bound decode on Intel hybrids when a static barrier waits on the slow core. **I tested this on ASDSL and it is *false here* (Section D).** ASDSL's atomic work-stealing scheduler lets P-cores grab the chunks E-cores didn't, so E-cores add net bandwidth.

---

## SECTION D — Root cause statement

**ASDSL decode is GEMV-memory-bandwidth bound and sustains only ~26 GB/s on the body matmuls (≈65% of this machine's ~40 GB/s STREAM ceiling), whereas llama.cpp sustains ~33 GB/s (~83%) on the same ~2.4 GB/token of Q4 weights.** The per-token budget is body GEMV 77 ms + lm_head 13 ms = 90 ms of matmul (97.6%), with attention/RoPE/KV/embed at 1.8 ms and Python at ~0–15 ms. The bandwidth shortfall is not the kernel inner loop (the preq2+VNNI microbench hits 40+ GB/s on a single warm matrix) but **sustained streaming across 129 sequential GEMVs**: between every op ASDSL forks/joins an OpenMP team (~161×/token), re-issues thousands of redundant `SetThreadAffinityMask` syscalls, and runs the activation-quant prologue single-threaded — leaving the DRAM controller idle in the gaps. Bytes/token (~5.0 bpw preq2) are essentially equal to llama's Q4_K_M (~5.18 bpw), so the gap is **sustained-bandwidth efficiency, owned by the threading/streaming machinery — and the fix (a persistent pinned threadpool) already exists in `thread_pool.h` but is disabled (`pool_{0}`).**

---

## SECTION E — Ranked opportunity table

Ranked by `(expected ms saved) × confidence ÷ impl-days`. Only techniques **not already tried** and **not already done**. Baseline decode token = **92.6 ms C++ (≈ 90 ms warm wall after thermal)** → 9.88–11 tok/s.

| # | Technique | Est. ms saved/token | Conf. | Days | Lossless? | Key risk | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | **Wire persistent pinned `ThreadPool` to GEMV+attention** (replace 161 `#pragma omp parallel` + per-chunk `SetThreadAffinityMask`) | 10–22 (affinity 3–6 + fork/join 1.5–2 + sustained-BW recovery 5–14) | 0.55 | 5–7 | ✅ | routing chunked kernels through `pool_.parallel_for`; work-steal correctness vs `y[]` race | **PURSUE (the one thing)** |
| 2 | **Wire C++ `generate()` decode loop** (exists; C++ argmax, GIL-free) | 3–15 | 0.8 | 2–3 | ✅ | sampling/stop-token + streaming UX parity | **PURSUE** |
| 3 | **Byte diet to 4.5 bpw across body+lm_head** (extend C0.1 g128 / native Q4_K to all projections) | 7–11 (≈10% of 90 ms) | 0.6 | 3–5 | ⚠️ PPL | g128 PPL 1.37× (gate ≤1.50×) — within budget but verify per-tensor | **PURSUE / CHARACTERIZE** |
| 4 | **`-openmp:llvm` + `OMP_WAIT_POLICY=ACTIVE` / `KMP_BLOCKTIME=infinite`** (cheap interim if #1 slips) | 1–4 | 0.4 | 0.5 | ✅ | MSVC libomp build flags; may conflict with manual affinity | **PURSUE (trivial probe)** |
| 5 | **chunk_div + thread-count re-tune *under the persistent pool*** (div=6 already best at 44 GB/s microbench) | 3–8 (couples with #1) | 0.4 | 2 | ✅ | re-tune needed after #1 changes the regime | **CHARACTERIZE** |
| 6 | **Fused RMSNorm+Q8 quant** | ~0 (already fused, 0.44 ms) | 0.95 | — | ✅ | — | **SKIP (done)** |
| 7 | **Fused RoPE+attention** | <1 (non-GEMV is 1.8 ms) | 0.9 | 4 | ✅ | no headroom at ctx≤2K | **SKIP** |
| 8 | **Double-buffered async weight streaming** | 0–5 (BW-bound, not latency-bound) | 0.2 | 6 | ✅ | can't beat the memory controller | **SKIP / characterize** |
| 9 | **B=2 continuous batching** | 0 for single-stream latency | 0.3 | 7 | ✅ | only helps multi-user throughput | **SECONDARY (server)** |

---

## SECTION F — The one thing

**Wire the existing persistent, P-core-pinned `ThreadPool` (`thread_pool.h`) into the GEMV and attention hot path, replacing the per-op `#pragma omp parallel` regions, and pin affinity once at pool creation instead of per chunk.**

Math. Today a decode token is 92.6 ms with body+lm_head GEMV = 90.3 ms at an effective **26 GB/s** (2.41 GB ÷ 0.0903 s). The same hardware's STREAM ceiling is ~40 GB/s and llama.cpp sustains ~33 GB/s on the same bytes using exactly this technique (`ggml_threadpool`). Removing the per-token overhead recovers, conservatively:

- redundant `SetThreadAffinityMask` syscalls: **−3 to −6 ms**
- 161 `vcomp` fork/joins → 0: **−1.5 to −2 ms**
- closing DRAM idle-gap so sustained BW rises 26 → ~31 GB/s: 2.41 GB ÷ 31 = 77.7 ms GEMV → **−12.6 ms**

New token ≈ 92.6 − (4.5 + 1.7 + 12.6) ≈ **73.8 ms → 13.5 tok/s** (and ~70 ms → **14.3 tok/s** if BW reaches llama's 33 GB/s). **Confidence 0.55** that it lands in 12.5–14 tok/s; the upside (≥13.9) is real but not guaranteed by this change alone. It is fully lossless (no output change) and the pool already exists, tested, in-tree — the work is *routing* (`gemv_preq2_from_q8_impl` and `compute_attention_flash_q8` call `pool_.parallel_for` instead of `#pragma omp parallel`), not new algorithms.

---

## SECTION G — Full path to >13.9 tok/s

Cumulative, ordered by confidence-adjusted ROI. Baseline **9.88 tok/s** (cold C0) / ~11 (warm).

| Step | Change | Mechanism | Token ms | tok/s | Cum. conf. |
|---|---|---|---|---|---|
| 0 | C0 baseline (measured) | — | ~92.6 (C++) / ~90 warm | 9.9–11 | — |
| 1 | Persistent pinned threadpool (#1) | −affinity syscalls, −fork/join, +sustained BW 26→31 GB/s | ~74 | **13.5** | 0.55 |
| 2 | + C++ generate() loop (#2) | −Python/bridge per token | ~70 | **14.3** | 0.50 |
| 3 | + 4.5 bpw body+lm_head (#3) | −10% body bytes (PPL ≤1.50×) | ~63 | **15.9** | 0.40 |
| 4 | + chunk_div/thread re-tune under pool (#5) | push sustained BW toward 35 GB/s | ~58 | **17.2** | 0.30 |

Honest read: **Step 1 alone has ~50% odds of reaching 13.9.** Steps 1+2 stacked are the realistic plan to *clear* 13.9 with margin. Steps 3–4 are upside that also carries the only quality risk (keep the PPL gate ≤1.50× hard).

---

## SECTION H — What llama.cpp does that we don't (code-level)

1. **One persistent threadpool for the whole decode**, reused across all 32 layers and the lm_head (`ggml_threadpool`, PR #7526). ASDSL has the equivalent class (`ThreadPool`) but constructs it with **zero workers** and falls back to `#pragma omp parallel` per op.
2. **Affinity set once at pool creation.** ASDSL re-issues `SetThreadAffinityMask` on every chunk pickup inside every GEMV (`bind_omp_thread_physical_if_enabled()` in the chunk body).
3. **Futex yield barrier + `--poll` spin policy** (PR #13079) tuned so workers stay hot between ops without descheduling — ASDSL's `vcomp` barrier blocks/wakes workers each region.
4. **No interpreter on the hot path.** llama's decode loop is pure C++; ASDSL returns to Python every token (its own C++ `generate()` is unwired).
5. **Per-phase pools** (separate batch vs single-token pools) — ASDSL uses the same path for prefill and decode.

Everything else (attention math, RoPE, RMSNorm fusion, Q4 bpw) is at parity or already better in ASDSL. **The entire deficit is in the thread/stream scheduler — and llama's fix is a known, ported-able pattern.**

---

## SECTION I — Novel / long-range ideas (clearly speculative)

- **(Tested, refuted — reported for honesty) "Drop the E-cores."** External consensus says E-cores hurt Intel-hybrid decode. I measured 8 P-core threads (4P×SMT, div=4): body GEMV **89.6 ms / 8.43 tok/s vs 77.1 ms at 12 threads** — **E-cores help here** because ASDSL's atomic work-stealing scheduler lets P-cores absorb the chunks E-cores leave. *Do not* restrict to P-cores on this hardware. (This is also why a *static* OMP schedule would regress — keep the chunked path.)
- **Persistent-pool + per-core weight residency (speculative, med confidence).** With a pinned pool, assign each physical core a *fixed* contiguous row-band of every weight matrix for the whole session, so each core re-streams the same address ranges every token → maximal hardware-prefetcher training and zero cross-core cache ping-pong. This converts work-stealing's adaptivity into prefetch-friendly determinism; could push sustained BW past llama's 33 GB/s. Validation: measure per-core L2/LLC miss rate before committing.
- **fp16-direct lm_head skip via "shortlist" (speculative, quality-gated).** The lm_head is 13 ms/token (0.384 GB). For greedy/top-k decode, a cheap Q2 lm_head "shortlist" pass could pre-select ~1–2 K candidate rows, then do exact Q4 only on those → lm_head ~1–2 ms. Risk: argmax disagreement; must gate exact-vs-shortlist agreement ≥99.9% (the plan's deferred "lm_head exact-argmax screening" idea — now cheap to revisit since lm_head is a clean 14% slice).
- **AMX prefill path for the next machine (secondary).** On Meteor/Arrow/Lunar Lake, route *prompt processing* (B≫1, compute-bound) through AMX int8 tiles while keeping the bandwidth-bound decode on the AVX2/VNNI GEMV. Design the `weight_format` enum hook now; it does not touch the C0 decode benchmark.
- **What I do not know / would measure next.** (1) The exact sustained-BW gain from the threadpool — needs a build with the pool wired and a re-profile. (2) Whether ggml's mul_mat also benefits from cache-blocking the activation across the row-band that ASDSL's per-row kernel doesn't do. (3) Whether `-openmp:llvm` + `KMP_BLOCKTIME=infinite` recovers part of #1 for near-zero effort (worth a 30-minute probe before the multi-day pool wiring).

---

### Appendix — raw captures
- `benchmarks/results/engine_profile_run.txt` — prefill token 3 (12t, div=4)
- `benchmarks/results/engine_profile_decode2.txt` — decode token 28 (12t, div=4, ignore-eos)
- `benchmarks/results/engine_profile_pcore8.txt` — decode token 28 (8t P-core-only, div=4) — E-core ablation
- Effective BW: body 2.01 GB ÷ 77.1 ms = 26.1 GB/s; lm_head 0.384 GB ÷ 13.2 ms = 29.0 GB/s; STREAM ceiling ≈ 40 GB/s (`phase1_thread_sweep.json`); llama ≈ 33 GB/s (13.9 tok/s × 2.4 GB).
