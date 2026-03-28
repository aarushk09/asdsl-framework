# ASDSL Master Architecture Document

## 1. Framework Abstract & Philosophy

ASDSL is a CPU-first LLM inference framework whose core objective is to push decode throughput toward the hardware roofline on consumer-class systems. The guiding thesis is that batch-1 autoregressive decode is primarily a memory-streaming problem, not a peak-FLOP problem.

Design philosophy:

- Prioritize sustained memory bandwidth utilization over abstract arithmetic intensity.
- Minimize data movement and duplicate passes through DRAM.
- Keep hot kernels native and vectorized (AVX2/FMA on x86).
- Treat Python as orchestration and diagnostics; place critical loops in C++/OpenMP.
- Use quantization and KV compression to reduce bytes per generated token.
- Evaluate all optimizations against a formal roofline ceiling, not synthetic speedups.

Repository reality (post-Phase 12.3):

- There are two major execution tracks:
  - Quantized mmap-native track via [asdsl/kernels/forward_loop.cpp](asdsl/kernels/forward_loop.cpp) and [asdsl/inference/engine.py](asdsl/inference/engine.py).
  - Phi-4 multimodal-instruct reference/optimized Python+native kernel track in [experiments/phi4_cpu_run.py](experiments/phi4_cpu_run.py).
- The framework now includes explicit memory pressure controls for evaluation (model cache clear + GC + skip-perplexity mode) to avoid OOM transitions between HF perplexity and native zero-shot phases.

---

## 2. The Scientific Roofline Profiler (The Mathematical Core)

Primary implementation: [asdsl/profiler.py](asdsl/profiler.py)

### 2.1 STREAM Triad Measurement Path

Entry point: measure_stream_triad_bandwidth(array_mb, runs, warmup_runs)

Execution priority:

1. Native OpenMP triad from [asdsl/kernels/forward_loop.cpp](asdsl/kernels/forward_loop.cpp) via _native_forward.stream_triad_int8.
2. Multi-threaded numexpr float32 triad: a = b + s * c.
3. NumPy float32 fallback.

Profiler metadata records implementation label in StreamTriadResult.implementation:

- native_openmp
- numexpr_f32
- numpy_f32

Measured throughput formula:

$$
\text{Bandwidth}_{GB/s} = \frac{3 \times \text{array\_bytes} \times \text{runs}}{\text{elapsed\_seconds} \times 10^9}
$$

where factor 3 captures two reads (b,c) and one write (a).

### 2.2 Exact Bytes Per Token at Pinned Context Length

Byte accounting in bytes_per_token_breakdown:

$$
\text{Bytes/token}(t) = W + K(t)
$$

with

$$
W = \text{model file size on disk (bytes)}
$$

$$
K(t) = 2 \times L \times n_{kv} \times d_{head} \times t \times b_{kv}
$$

Definitions:

- \(L\): number of transformer layers.
- \(n_{kv}\): number of KV heads (critical for GQA/MQA correctness).
- \(d_{head}\): head dimension.
- \(t\): pinned context length used for ceiling reporting.
- \(b_{kv}\): bytes per KV element (1 for Q8 KV cache, 2 for fp16 KV cache).

### 2.3 Roofline Ceiling and Efficiency

Ceiling in tokens per second:

$$
\text{Ceiling}(t) = \frac{\text{Bandwidth}_{GB/s}}{\text{Bytes/token}(t)_{GB}}
$$

Equivalent byte-space form used by roofline_ceiling_tps:

$$
\text{Ceiling}(t) = \frac{\text{Bandwidth}_{GB/s} \times 10^9}{\text{Bytes/token}(t)_{bytes}}
$$

Observed hardware efficiency:

$$
\text{Efficiency}(\%) = \frac{\text{Observed tok/s}}{\text{Ceiling}(t)} \times 100
$$

### 2.4 Ceiling Curve

roofline_curve computes \(\text{Ceiling}(t)\) over a t-range to expose long-context degradation:

$$
t \in \{t_{min}, t_{min}+\Delta t, ..., t_{max}\}
$$

and emits per-point:

- t
- bytes_per_token
- bytes_per_token_gb
- ceiling_tps

### 2.5 Architectural Parameter Resolution

Architecture source order in [evals/leaderboard_eval.py](evals/leaderboard_eval.py):

1. Local config.json adjacent to model metadata.
2. HuggingFace config (AutoConfig) using hf-model-id.
3. Shape inference from quantized metadata keys and l0_qkv/l0_o tensors.

This prevents hardcoded KV-head assumptions and preserves GQA/MQA correctness.

---

## 3. Tensor Flow & The C++/Python Boundary

### 3.1 Native Memory Mapping

Core mmap container: class MmapWeights in [asdsl/kernels/forward_loop.cpp](asdsl/kernels/forward_loop.cpp)

Windows file mapping sequence:

- CreateFileA
- CreateFileMappingA
- MapViewOfFile

Each tensor key maps to TensorInfo:

- ptr: uint8 pointer into mapped region
- rows, cols
- dtype
- interleaved4 flag

A background prefetch thread consumes queued tensor spans and issues PrefetchVirtualMemory hints.

### 3.2 Native Interface Surface (PyBind)

Module: _native_forward, defined in [asdsl/kernels/forward_loop.cpp](asdsl/kernels/forward_loop.cpp)

Critical bindings:

- MmapWeights
- KVCache
- prefill_prompt_tokens
- compute_attention
- generate_token
- stream_triad_int8
- pin_openmp_threads_to_pcores
- detected_pcore_count

### 3.3 KV Cache Representation and Q8 Quantization

KVCache stores int8 KV plus per-block scales:

- k_cache_q8, v_cache_q8
- k_scales, v_scales
- KV_QBLOCK = 64

For each layer, position, kv-head, and block:

- Compute absmax.
- Scale to int8 range 127.
- Store int8 payload + block scale.

This directly implements Q8 KV compression and reduces KV bandwidth term in bytes/token.

### 3.4 Attention Kernel Shape Contract

compute_attention_flash_q8 receives:

- q shape: (num_heads, head_dim)
- k shape: (num_kv_heads, head_dim)
- v shape: (num_kv_heads, head_dim)

Then uses grouped query attention mapping:

$$
\text{groups} = \max(1, \lfloor n_q / n_{kv} \rfloor)
$$

with tiled flash-style accumulation over BLOCK_K = 64 and online softmax state \((m,l)\).

### 3.5 Python Reference Tensor Contracts in Phi-4 Path

Primary constants in [experiments/phi4_cpu_run.py](experiments/phi4_cpu_run.py):

- HIDDEN = 3072
- NUM_LAYERS = 32
- NUM_HEADS = 24
- NUM_KV_HEADS = 8
- HEAD_DIM = 128
- Q_DIM = 3072
- KV_DIM = 1024
- QKV_DIM = 5120
- INTER = 8192
- VOCAB = 200064
- ROTARY_DIM = 96 (partial rotary factor 0.75)

### 3.6 lm_head_matvec Boundary and Batched Projection

In [experiments/phi4_cpu_run.py](experiments/phi4_cpu_run.py), lm_head_matvec accepts:

- (hidden_dim,)
- (1, hidden_dim)
- (K, hidden_dim)

Normalization behavior:

- 1D hidden is promoted to (1, hidden_dim).
- Output is squeezed back to (vocab,) only when K = 1.

Batched projection path:

- lm_head_matmul_batch builds X with shape (K, hidden_dim).
- lm_head has shape (vocab, hidden_dim).
- Chunked weight slices are copied into a contiguous pool buffer.
- Uses torch.mm(chunk, X.T) for each chunk.
- Assembled result has shape (K, vocab).

This avoids serial torch.mv loops and improves CPU vector unit utilization by amortizing memory traversal over multiple sequence positions.

### 3.7 Important Current Boundary Limitation

In [asdsl/inference/engine.py](asdsl/inference/engine.py), perplexity uses HuggingFace forward because the mmap generate_token binding is still a stub for logits-quality evaluation. The native model artifact path is retained for roofline byte accounting metadata only.

---

## 4. Hardware Mitigations & Thread Hygiene

### 4.1 Denormal Mitigation

Applied in multiple entrypoints:

- [evals/leaderboard_eval.py](evals/leaderboard_eval.py)
- [evals/perplexity.py](evals/perplexity.py)
- [evals/lm_eval_harness.py](evals/lm_eval_harness.py)
- [experiments/phi4_cpu_run.py](experiments/phi4_cpu_run.py)

Mechanism:

- torch.set_flush_denormal(True)

Rationale:

Subnormal IEEE-754 values can route operations through slow microcode paths on many CPUs. Flushing denormals to zero avoids pathological ALU stalls in long matmul/GEMV loops.

### 4.2 Thread-Bounding Hierarchy

Runtime controls converge to a single effective thread budget:

- ASDSL_TORCH_NUM_THREADS (highest explicit app-level override)
- OMP_NUM_THREADS
- torch.set_num_threads(n)
- NUMEXPR_MAX_THREADS / NUMEXPR_NUM_THREADS
- MKL_NUM_THREADS / OPENBLAS_NUM_THREADS in phi4_cpu_run set_thread_count

Purpose:

- Prevent oversubscription across PyTorch BLAS + numexpr + OpenMP kernels.
- Reduce context-switch pressure and cache trashing.
- Keep worker count near physical P-core count for memory-bound kernels.

### 4.3 Build-Time OpenMP and ISA Controls

From [setup.py](setup.py):

- Windows: /arch:AVX2, /O2, /fp:fast, /openmp
- Linux: -mavx2, -mfma, -mf16c, -O3, -ffast-math, -fopenmp
- macOS: AVX2/FMA flags without OpenMP by default path

### 4.4 P-Core Affinity Hooks

From [asdsl/kernels/forward_loop.cpp](asdsl/kernels/forward_loop.cpp):

- set_thread_affinity(core_id)
- pin_openmp_threads_to_pcores(enabled)
- detected_pcore_count()

These APIs are the foundation for preventing hybrid-core mis-scheduling (P/E-core drift) in sustained decode loops.

---

## 5. Evaluation Harness (lm-eval) Pipeline

### 5.1 Perplexity Pipeline

Driver scripts:

- [evals/perplexity.py](evals/perplexity.py)
- [evals/leaderboard_eval.py](evals/leaderboard_eval.py)

Core computation in [asdsl/inference/engine.py](asdsl/inference/engine.py): evaluate_perplexity_phase8_native

Algorithm:

1. Resolve model id via resolve_hf_ppl_model_id.
2. Load AutoModelForCausalLM on CPU (cached per model id).
3. Chunk tokens into non-overlapping windows of length stride+1.
4. Compute logits for each chunk.
5. Apply shifted alignment:

$$
\text{shift\_logits} = \text{logits}[0: -1]
$$

$$
\text{shift\_labels} = \text{input\_ids}[1: ]
$$

6. Sum NLL over all valid targets with cross-entropy reduction=sum.
7. Aggregate:

$$
\text{avg\_nll} = \frac{\sum \text{NLL}}{N_{tokens}}
$$

$$
\text{PPL} = e^{\text{avg\_nll}}
$$

8. Throughput:

$$
\text{tok/s} = \frac{N_{processed}}{t_{elapsed}}
$$

Memory safety sweep (post-12.3):

- In engine: del model, clear_hf_causal_lm(), gc.collect().
- In leaderboard: del tokenizer, gc.collect() before zero-shot stage.
- New flag: --skip-perplexity to avoid loading HF model at all.

### 5.2 Zero-Shot PIQA Integration and Shared-Context Reuse

Leaderboard zero-shot entry uses [evals/leaderboard_eval.py](evals/leaderboard_eval.py) and calls [evals/lm_eval_harness.py](evals/lm_eval_harness.py).

In ASDSLHarnessModel.loglikelihood:

- Requests are grouped by identical context token tuple.
- For each unique context:
  - Prefill once into KVHistory.
  - Capture snapshot of KV lengths.
  - For each continuation choice, restore snapshot and score continuation.

This prevents redundant context prefills across multiple answer options in PIQA-style multiple-choice scoring.

Teacher-forced continuation scoring path:

- First continuation token scored from logits_after_ctx.
- Remaining continuation positions scored via forward_layer_batch and batched lm_head projection.
- This yields better CPU throughput than fully sequential per-choice decode.

### 5.3 Failure Modes and Mitigations

Historical issues addressed:

- OOM at stage transition from HF perplexity to native zero-shot due to cached model residency.
- Deadlock-like appearance from long zero-shot stage now mitigated by tighter memory and thread hygiene; optional stage skipping exists (--skip-zero-shot, --skip-perplexity).

---

## 6. Codebase Topography & Entrypoints

### 6.1 Critical File Map

- [asdsl/inference/engine.py](asdsl/inference/engine.py)
  - Perplexity computation, HF model lifecycle, native artifact resolution.
- [asdsl/profiler.py](asdsl/profiler.py)
  - STREAM triad measurement, bytes/token accounting, roofline math.
- [asdsl/kernels/forward_loop.cpp](asdsl/kernels/forward_loop.cpp)
  - mmap native bridge, KVCache Q8, flash-style attention, STREAM OpenMP probe.
- [experiments/phi4_cpu_run.py](experiments/phi4_cpu_run.py)
  - WeightStore, quantized GEMV dispatch, lm_head batched projection, chat/generate CLI.
- [evals/leaderboard_eval.py](evals/leaderboard_eval.py)
  - Unified benchmark orchestrator (PPL + roofline + zero-shot).
- [evals/perplexity.py](evals/perplexity.py)
  - Focused PPL entrypoint.
- [evals/lm_eval_harness.py](evals/lm_eval_harness.py)
  - lm-eval wrapper and shared-context zero-shot scoring.
- [setup.py](setup.py)
  - Native extension build flags and module list.

### 6.2 Repository Skeleton (Operational)

- asdsl/
  - inference/
  - kernels/
  - quantization/
  - memory/
  - prefetch/
  - speculative/
- evals/
- experiments/
- benchmarks/
- models/

### 6.3 Entrypoint Commands and Flags

Leaderboard evaluation:

- python evals/leaderboard_eval.py --max-tokens 1024
- Key flags:
  - --bits {2,3,4,8,16}
  - --stride
  - --stream-array-mb
  - --stream-runs
  - --kv-bytes-per-element {1,2}
  - --hf-model-id
  - --skip-perplexity
  - --skip-zero-shot
  - --zero-shot-task
  - --zero-shot-limit
  - --curve-min-t --curve-max-t --curve-step

Perplexity-only:

- python evals/perplexity.py --max-tokens 1024 --stride 1024
- Key flags:
  - --bits
  - --hf-model-id

Phi-4 inference / QCSD benchmark path:

- python experiments/phi4_cpu_run.py --prompt "..." --bits 4
- python experiments/phi4_cpu_run.py --qcsd --draft-bits 2 --draft-k 7
- Key flags:
  - --chat
  - --stream
  - --threads
  - --group-size
  - --sparse --sparse-threshold

---

## 7. Advanced Optimization Roadmap (HPC Expansion)

### 7.1 L2/L3 Cache Tiling and Blocking in Native Loops

Objective:

Raise sustained effective bandwidth by reducing conflict misses and improving prefetch predictability in GEMV/GEMM-like decode/prefill kernels.

Required work:

- Introduce explicit multi-level tiles matching L2-private and shared L3 capacities.
- Tune tile geometry by architecture class (mobile LPDDR vs desktop DDR).
- Fuse dequant and dot-product in-register across larger contiguous strips.
- Use software prefetch distance tuned by empirical LLC miss curves.

Expected effect:

- Lower bytes wasted per useful MAC.
- Higher realized fraction of STREAM bandwidth in decode path.

### 7.2 Custom Sub-4-Bit Quantization Mappings

Objective:

Reduce weight-stream bytes below q4 while preserving tail behavior in sensitive channels.

Required work:

- Implement packed q3/q2 data layouts with groupwise scale/zero and outlier side-channel.
- Calibrate per-layer clipping strategy under perplexity and zero-shot constraints.
- Co-design kernel unpack path to avoid scalar unpack bottlenecks.

Expected effect:

- Lower weight term in bytes/token.
- Higher roofline ceiling at fixed bandwidth, with quality-retention determined by outlier handling.

### 7.3 Speculative Decoding Integration (Draft/Verify)

Objective:

Increase effective tokens generated per expensive target-model pass.

Required work:

- Keep draft model resident in cache-friendly footprint.
- Batch verify multiple drafted positions through shared prefill/state transitions.
- Integrate acceptance-aware scheduling with KV snapshot/rollback semantics.
- Expose acceptance, verifier-call, and effective-byte metrics in profiler output.

Expected effect:

- Effective throughput multiplier without violating physical bandwidth ceiling for target passes.
- Better amortization of weight streaming over accepted token count.

---

## Known Architectural Risks (Current)

- Configuration divergence between some native mmap artifacts (e.g., 40-layer metadata paths) and Phi-4 multimodal python constants (32 layers) can create evaluation inconsistency if tokenizer/model/artifact sets are mixed.
- Any benchmark that does not pin t context length is not reproducible for roofline comparison.
- Oversubscription between BLAS, numexpr, and OpenMP remains a top regression vector when new kernels are introduced.

---

## Agent-Facing Implementation Checklist

Before changing performance-critical code, verify all of the following:

1. Roofline uses measured bandwidth (not constants).
2. Bytes/token uses on-disk model size and correct n_kv_heads.
3. Perplexity computes shifted full-token NLL sum.
4. HF model memory is explicitly released before native zero-shot stage.
5. Thread and denormal controls are set before model load and compute loops.
6. Any new kernel change reports impact on:
   - observed tok/s
   - ceiling tok/s at pinned t
   - efficiency percent
   - quality metrics (PPL and zero-shot accuracy)
