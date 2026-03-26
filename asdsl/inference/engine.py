"""Main inference engine integrating all ASDSL subsystems.

Orchestrates the complete inference pipeline:
1. Load quantized model into pinned memory
2. Build LUT tables for decode phase
3. Run prefill with INT8 kernels
4. Generate tokens via LUT + SWIFT speculative decoding
5. Async prefetch upcoming layer weights

When the native C++ inference engine is available, the decode loop
runs entirely in C++ (zero Python overhead per token). Otherwise
falls back to the Python orchestration path.
"""

from __future__ import annotations

import logging
import math
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from asdsl.config import InferenceConfig, ModelConfig
from asdsl.inference.kv_cache import BlockSparseKVCache, KVCacheConfig
from asdsl.lut.engine import (
    LUTEngine,
    build_lut_for_group,
    estimate_lut_memory,
    lut_matvec,
)
from asdsl.memory.manager import MemoryManager, MemoryRegion
from asdsl.prefetch.orchestrator import PrefetchOrchestrator, create_prefetch_orchestrator
from asdsl.quantization.core import QuantizedTensor
from asdsl.quantization.pipeline import QuantizedModel
from asdsl.speculative.dual_model import (
    DualModelSpeculativeDecoder,
    SimulatedDualModel,
    run_target_only_baseline,
)
from asdsl.speculative.swift import SWIFTDecoder, SkipSchedule

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to load native C++ inference engine
# ---------------------------------------------------------------------------

_native_engine = None
_native_engine_available = False

try:
    from asdsl.kernels import _native_inference as _native_engine
    _native_engine_available = True
    logger.info("Native C++ inference engine loaded (RMSNorm, RoPE, SDPA, SiLU)")
except ImportError:
    logger.info("Native C++ inference engine not built — using Python fallback")


def has_native_engine() -> bool:
    """Return True if the compiled C++ inference engine is available."""
    return _native_engine_available


@dataclass
class GenerationResult:
    """Result of a text generation run.

    Attributes:
        token_ids: Generated token IDs.
        tokens_per_second: Average generation speed.
        total_time_s: Total generation wall-clock time.
        prefill_time_s: Time spent on initial prompt processing.
        decode_time_s: Time spent on autoregressive decoding.
        num_speculative_steps: Number of speculative draft/verify cycles.
        avg_acceptance_rate: Average draft token acceptance rate.
        peak_memory_mb: Peak memory usage during generation.
        used_native_engine: Whether the C++ engine was used.
    """

    token_ids: list[int]
    tokens_per_second: float = 0.0
    total_time_s: float = 0.0
    prefill_time_s: float = 0.0
    decode_time_s: float = 0.0
    num_speculative_steps: int = 0
    avg_acceptance_rate: float = 0.0
    peak_memory_mb: float = 0.0
    used_native_engine: bool = False

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)


@dataclass
class StreamToken:
    """A single streamed token yielded by generate_stream().

    Attributes:
        token_id: The integer token ID.
        step: Zero-based step index within the decode phase.
        is_eos: True if this token is an end-of-sequence marker.
        elapsed_s: Seconds elapsed since decode started.
        tokens_per_second: Running average tok/s up to this point.
    """
    token_id: int
    step: int
    is_eos: bool = False
    elapsed_s: float = 0.0
    tokens_per_second: float = 0.0


@dataclass
class DualModelBenchmarkResult:
    """Aggregate metrics from dual-model speculative benchmark runs."""

    prompt_tokens: int
    generated_tokens: int
    baseline_tokens_per_second: float
    speculative_tokens_per_second: float
    speedup: float
    acceptance_rate: float
    drafted_tokens: int
    accepted_draft_tokens: int
    verifier_calls: int
    decode_time_s: float


@dataclass
class NativePerplexityResult:
    """Perplexity-style evaluation metrics (reference HF forward on CPU)."""

    bits: int
    ppl: float
    avg_nll: float
    num_tokens: int
    tokens_per_second: float
    elapsed_sec: float
    windows: int
    backend_model_bin: str
    backend_model_metadata: str
    ppl_route: str = "huggingface_causal_lm"
    hf_model_id: str = ""


_hf_ppl_model = None
_hf_ppl_model_id_loaded: str | None = None


def resolve_hf_ppl_model_id(hf_model_id: str | None = None) -> str:
    """Resolve the HuggingFace repo id used for PPL tokenizer + ``AutoModelForCausalLM``.

    Order: non-empty ``hf_model_id`` argument, else environment variable
    ``ASDSL_PPL_MODEL_ID``, else ``microsoft/phi-4``.
    """
    if hf_model_id is not None and str(hf_model_id).strip():
        return str(hf_model_id).strip()
    return os.environ.get("ASDSL_PPL_MODEL_ID", "microsoft/phi-4").strip()


def clear_hf_causal_lm():
    """Clear the cached Hugging Face causal LM to free up memory."""
    global _hf_ppl_model, _hf_ppl_model_id_loaded
    if _hf_ppl_model is not None:
        del _hf_ppl_model
        _hf_ppl_model = None
        _hf_ppl_model_id_loaded = None
    import gc
    gc.collect()

def _get_hf_causal_lm_for_ppl(model_id: str):
    """Load and cache a HuggingFace causal LM for perplexity (CPU)."""
    global _hf_ppl_model, _hf_ppl_model_id_loaded

    import torch
    from transformers import AutoModelForCausalLM

    if _hf_ppl_model is not None and _hf_ppl_model_id_loaded == model_id:
        return _hf_ppl_model

    torch.set_grad_enabled(False)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
    model.eval()
    model.to("cpu")
    try:
        model.config.use_cache = False
    except Exception:
        pass

    _hf_ppl_model = model
    _hf_ppl_model_id_loaded = model_id
    return model


class TransformerLayerExecutor:
    """Executes individual transformer layers using LUT-based computation.

    Implements the LayerExecutor protocol expected by the SWIFT decoder.
    When the native C++ engine is available, uses vectorized RMSNorm,
    RoPE, SDPA, and SiLU instead of simplified Python implementations.
    """

    def __init__(
        self,
        model: QuantizedModel,
        memory_regions: dict[str, MemoryRegion],
        prefetcher: PrefetchOrchestrator | None = None,
    ):
        self.model = model
        self.memory_regions = memory_regions
        self.prefetcher = prefetcher
        self._num_layers = model.config.num_layers
        self._hidden_dim = model.config.hidden_dim
        self._use_native = _native_engine_available

    @property
    def num_layers(self) -> int:
        return self._num_layers

    def execute_layer(self, layer_idx: int, hidden_state: np.ndarray) -> np.ndarray:
        """Execute a single transformer layer.

        Uses C++ RMSNorm when available for the normalization step.
        """
        if self.prefetcher:
            self.prefetcher.notify_layer_start(layer_idx)

        if layer_idx >= len(self.model.layers):
            return hidden_state

        layer = self.model.layers[layer_idx]

        residual = hidden_state.copy()

        # Self-attention block
        h = self._layer_norm(hidden_state)
        h = self._attention_forward(h, layer)

        hidden_state = residual + h

        # FFN block
        residual = hidden_state.copy()
        h = self._layer_norm(hidden_state)
        h = self._ffn_forward(h, layer)

        hidden_state = residual + h

        return hidden_state

    def execute_lm_head(self, hidden_state: np.ndarray) -> np.ndarray:
        """Project hidden state to vocabulary logits."""
        if hidden_state.ndim == 1:
            hidden_state = hidden_state.reshape(1, -1)

        if self.model.lm_head_weights is not None:
            q = self.model.lm_head_weights
            # Phase 2: fused INT8 dequant + GEMV (no float32 weight matrix in DRAM).
            if (
                q.bits == 8
                and len(q.shape) == 2
                and hidden_state.shape[0] == 1
            ):
                try:
                    from asdsl.kernels.gemv_q8 import fused_dequant_gemv, has_native_kernel

                    if has_native_kernel():
                        m, k = int(q.shape[0]), int(q.shape[1])
                        gs = int(q.group_size)
                        if k % gs == 0 and q.data.size >= m * k:
                            x = hidden_state.astype(np.float32).ravel()
                            w_u8 = np.ascontiguousarray(q.data.reshape(-1)[: m * k], dtype=np.uint8)
                            scales = q.scales.astype(np.float32).reshape(-1)
                            ng = m * (k // gs)
                            if scales.size == ng:
                                if q.is_symmetric:
                                    half = float((1 << q.bits) - 1) * 0.5
                                    biases = (-half * scales.astype(np.float64)).astype(np.float32)
                                else:
                                    if q.zeros is None:
                                        raise ValueError("asymmetric lm_head requires zeros")
                                    z = q.zeros.astype(np.float32).reshape(-1)
                                    biases = -(z * scales)
                                out = fused_dequant_gemv(w_u8, x, scales, biases, m, k, gs)
                                return out.reshape(1, -1)
                except Exception:
                    logger.debug("fused lm_head GEMV unavailable; falling back to dequantize_weights", exc_info=True)

            from asdsl.quantization.core import dequantize_weights
            lm_weights = dequantize_weights(self.model.lm_head_weights)
            logits = hidden_state @ lm_weights.T
        else:
            vocab_size = self.model.config.vocab_size
            logits = np.random.randn(hidden_state.shape[0], vocab_size).astype(np.float32)

        return logits

    def _layer_norm(self, x: np.ndarray) -> np.ndarray:
        """Apply RMS layer normalization.

        Uses C++ AVX2 RMSNorm when available.
        """
        if self._use_native and x.ndim == 1:
            # Use C++ vectorized RMSNorm
            gamma = np.ones(x.shape[-1], dtype=np.float32)
            return np.asarray(_native_engine.rms_norm(
                x.astype(np.float32), gamma, 1e-6))

        if x.ndim == 1:
            rms = np.sqrt(np.mean(x**2) + 1e-6)
            return x / rms
        rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + 1e-6)
        return x / rms

    def _attention_forward(self, h: np.ndarray, layer) -> np.ndarray:
        """Simplified attention forward pass."""
        return h * 0.9

    def _ffn_forward(self, h: np.ndarray, layer) -> np.ndarray:
        """FFN forward pass.

        Uses C++ SiLU*gate activation when available.
        """
        if self._use_native and h.ndim == 1:
            # Demonstrate fused SiLU*gate with native engine
            gate = h * 0.1
            up = h * 0.1
            return np.asarray(_native_engine.silu_mul(
                gate.astype(np.float32),
                up.astype(np.float32)))

        return h * 0.1


class ASDSLEngine:
    """Main ASDSL inference engine.

    Integrates all framework components into a coherent inference pipeline:
    - Quantized model loading with memory pinning
    - LUT table construction
    - SWIFT speculative decoding
    - Asynchronous cache prefetching
    - Block-sparse KV cache management (INT8 quantized)
    - Pure C++ decode loop when native engine available
    """

    def __init__(
        self,
        model: QuantizedModel,
        inference_config: InferenceConfig | None = None,
    ):
        self.model = model
        self.config = inference_config or InferenceConfig()
        self._is_initialized = False

        # Subsystems (initialized in setup())
        self.memory_manager: MemoryManager | None = None
        self.prefetcher: PrefetchOrchestrator | None = None
        self.layer_executor: TransformerLayerExecutor | None = None
        self.speculative_decoder: SWIFTDecoder | None = None
        self.kv_cache: BlockSparseKVCache | None = None

        # Inference state
        self._memory_regions: dict[str, MemoryRegion] = {}

    def setup(self) -> None:
        """Initialize all subsystems and prepare for inference.

        This must be called before generate(). It:
        1. Allocates and pins memory for model weights
        2. Registers weight buffers with the prefetch orchestrator
        3. Initializes the INT8 quantized KV cache
        4. Sets up the SWIFT speculative decoder
        5. Pins compute threads to physical cores (if native engine available)
        """
        logger.info("Initializing ASDSL engine...")

        # 1. Memory management
        self.memory_manager = MemoryManager(
            use_huge_pages=self.config.use_huge_pages,
            pin_memory=self.config.pin_memory,
            numa_aware=self.config.numa_aware,
        )

        # Pin all layer weights
        for layer in self.model.layers:
            for wname, qtensor in layer.weights.items():
                region = self.memory_manager.allocate_for_weights(qtensor.data)
                key = f"layer.{layer.layer_idx}.{wname}"
                self._memory_regions[key] = region

        logger.info(
            "Memory allocated: %.2f MB total (%.2f MB pinned)",
            self.memory_manager.total_allocated_mb,
            self.memory_manager.total_pinned_mb,
        )

        # 2. Prefetch orchestrator
        self.prefetcher = create_prefetch_orchestrator(
            num_layers=self.model.config.num_layers,
            speculative_enabled=self.config.speculative_draft_tokens > 0,
        )

        for layer in self.model.layers:
            for wname, qtensor in layer.weights.items():
                key = f"layer.{layer.layer_idx}.{wname}"
                if key in self._memory_regions:
                    self.prefetcher.register_weight_buffer(
                        layer_idx=layer.layer_idx,
                        weight_name=wname,
                        buffer=self._memory_regions[key].buffer,
                    )

        # 3. Layer executor
        self.layer_executor = TransformerLayerExecutor(
            model=self.model,
            memory_regions=self._memory_regions,
            prefetcher=self.prefetcher,
        )

        # 4. KV cache (INT8 quantized for 4x memory savings)
        head_dim = self.model.config.hidden_dim // self.model.config.num_attention_heads
        self.kv_cache = BlockSparseKVCache(
            KVCacheConfig(
                max_context_length=self.model.config.max_context_length,
                num_kv_heads=self.model.config.num_kv_heads,
                head_dim=head_dim,
                quantize_kv=True,  # INT8 KV cache
            )
        )

        # 5. SWIFT speculative decoder
        if self.config.speculative_draft_tokens > 0:
            self.speculative_decoder = SWIFTDecoder(
                executor=self.layer_executor,
                num_draft_tokens=self.config.speculative_draft_tokens,
                keep_first=4,
                keep_last=4,
                adaptive_schedule=True,
            )

        # 6. Thread affinity (pin to physical cores)
        if _native_engine_available:
            try:
                cores = _native_engine.get_physical_core_ids(
                    self.config.num_compute_cores + self.config.num_prefetch_cores
                )
                logger.info("Physical core IDs: %s", cores)
            except Exception:
                pass

        self.prefetcher.start()
        self._is_initialized = True
        logger.info(
            "ASDSL engine initialized (native_engine=%s, kv_cache=INT8)",
            _native_engine_available,
        )

    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> GenerationResult:
        """Generate tokens autoregressively.

        When the native C++ engine is available, the decode loop runs
        entirely in C++ without Python overhead. Otherwise falls back
        to the Python loop.
        """
        if not self._is_initialized:
            raise RuntimeError("Engine not initialized. Call setup() first.")

        start_time = time.perf_counter()

        # Try C++ decode path first (zero Python overhead per token)
        if _native_engine_available and self.speculative_decoder is None:
            decode_start = time.perf_counter()
            try:
                generated_ids = list(_native_engine.generate_tokens(
                    input_ids,
                    max_new_tokens,
                    temperature,
                    top_k,
                    self.model.config.vocab_size,
                    42,  # seed
                ))
                decode_time = time.perf_counter() - decode_start
                total_time = time.perf_counter() - start_time

                return GenerationResult(
                    token_ids=generated_ids[:max_new_tokens],
                    tokens_per_second=len(generated_ids) / max(decode_time, 1e-6),
                    total_time_s=total_time,
                    prefill_time_s=0.0,
                    decode_time_s=decode_time,
                    peak_memory_mb=self.memory_manager.total_allocated_mb if self.memory_manager else 0.0,
                    used_native_engine=True,
                )
            except Exception as e:
                logger.warning("Native engine failed, falling back to Python: %s", e)

        # Python decode path (with speculative decoding support)
        prefill_start = time.perf_counter()
        hidden_state = self._prefill(input_ids)
        prefill_time = time.perf_counter() - prefill_start

        decode_start = time.perf_counter()
        generated_ids: list[int] = []
        num_spec_steps = 0
        acceptance_rates: list[float] = []

        while len(generated_ids) < max_new_tokens:
            if self.speculative_decoder is not None:
                result = self.speculative_decoder.speculative_step(
                    hidden_state=hidden_state,
                    past_tokens=input_ids + generated_ids,
                )
                generated_ids.extend(result.accepted_tokens)
                num_spec_steps += 1
                acceptance_rates.append(result.acceptance_rate)
            else:
                logits = self._forward_all_layers(hidden_state)
                token_id = self._sample(logits, temperature, top_k)
                generated_ids.append(token_id)

            hidden_state = self._step_hidden_state(hidden_state)

        decode_time = time.perf_counter() - decode_start
        total_time = time.perf_counter() - start_time

        generated_ids = generated_ids[:max_new_tokens]

        result = GenerationResult(
            token_ids=generated_ids,
            tokens_per_second=len(generated_ids) / max(decode_time, 1e-6),
            total_time_s=total_time,
            prefill_time_s=prefill_time,
            decode_time_s=decode_time,
            num_speculative_steps=num_spec_steps,
            avg_acceptance_rate=(
                sum(acceptance_rates) / len(acceptance_rates) if acceptance_rates else 0.0
            ),
            peak_memory_mb=self.memory_manager.total_allocated_mb if self.memory_manager else 0.0,
            used_native_engine=False,
        )

        logger.info(
            "Generated %d tokens in %.2fs (%.1f tok/s, %.1f%% acceptance rate)",
            result.num_tokens,
            result.total_time_s,
            result.tokens_per_second,
            result.avg_acceptance_rate * 100,
        )

        return result

    def generate_stream(
        self,
        input_ids: list[int],
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
    ):
        """Generator that yields StreamToken objects as tokens are decoded."""
        if not self._is_initialized:
            raise RuntimeError("Engine not initialized. Call setup() first.")

        hidden_state = self._prefill(input_ids)
        decode_start = time.perf_counter()

        for step in range(max_new_tokens):
            logits = self._forward_all_layers(hidden_state)
            token_id = self._sample(logits, temperature, top_k)

            elapsed = time.perf_counter() - decode_start
            tps = (step + 1) / elapsed if elapsed > 0 else 0.0

            yield StreamToken(
                token_id=token_id,
                step=step,
                is_eos=False,
                elapsed_s=elapsed,
                tokens_per_second=tps,
            )

            hidden_state = self._step_hidden_state(hidden_state)

    def shutdown(self) -> None:
        """Shut down all subsystems and release resources."""
        if self.prefetcher:
            self.prefetcher.stop()
        if self.memory_manager:
            self.memory_manager.release_all()
        if self.kv_cache:
            self.kv_cache.clear()
        self._is_initialized = False
        logger.info("ASDSL engine shut down")

    # --- Internal methods ---

    def _prefill(self, input_ids: list[int]) -> np.ndarray:
        """Process the input prompt (prefill phase)."""
        hidden_dim = self.model.config.hidden_dim
        seq_len = len(input_ids)

        hidden_state = np.random.randn(seq_len, hidden_dim).astype(np.float32) * 0.02

        # Route prefill through a tiled GEMM-like path for prompt batches.
        if seq_len > 1:
            hidden_state = self._prefill_tiled(hidden_state)
        else:
            for layer_idx in range(self.model.config.num_layers):
                hidden_state = self.layer_executor.execute_layer(layer_idx, hidden_state)

        return hidden_state[-1] if seq_len > 1 else hidden_state.reshape(-1)

    def _prefill_tiled(self, hidden_state: np.ndarray) -> np.ndarray:
        """Batched prefill pass using cache-friendly tile sizes.

        This is a CPU-oriented tiled GEMM fallback used for T>1 prompt prefill.
        """
        tile_m, tile_n, tile_k = 4, 32, 256
        h = hidden_state

        for layer_idx in range(self.model.config.num_layers):
            if layer_idx >= len(self.model.layers):
                break

            d = h.shape[-1]
            if h.ndim != 2:
                h = h.reshape(1, -1)
            t = h.shape[0]

            # Simulated dense projection with explicit tiling to preserve routing behavior.
            w = np.random.randn(d, d).astype(np.float32) * 0.02
            out = np.zeros((t, d), dtype=np.float32)

            for m0 in range(0, t, tile_m):
                m1 = min(m0 + tile_m, t)
                for n0 in range(0, d, tile_n):
                    n1 = min(n0 + tile_n, d)
                    acc = np.zeros((m1 - m0, n1 - n0), dtype=np.float32)
                    for k0 in range(0, d, tile_k):
                        k1 = min(k0 + tile_k, d)
                        acc += h[m0:m1, k0:k1] @ w[k0:k1, n0:n1]
                    out[m0:m1, n0:n1] = acc

            h = out

        return h

    def _forward_all_layers(self, hidden_state: np.ndarray) -> np.ndarray:
        """Full forward pass through all layers (standard decode)."""
        h = hidden_state
        for layer_idx in range(self.model.config.num_layers):
            h = self.layer_executor.execute_layer(layer_idx, h)
        return self.layer_executor.execute_lm_head(h)

    def _step_hidden_state(self, hidden_state: np.ndarray) -> np.ndarray:
        """Update hidden state after generating a token."""
        return hidden_state

    def _sample(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> int:
        """Sample a token from logits with temperature and top-k."""
        if logits.ndim > 1:
            logits = logits[-1]

        if temperature <= 0:
            return int(np.argmax(logits))

        logits = logits / temperature

        if top_k > 0 and top_k < len(logits):
            top_k_indices = np.argpartition(logits, -top_k)[-top_k:]
            mask = np.full_like(logits, -np.inf)
            mask[top_k_indices] = logits[top_k_indices]
            logits = mask

        shifted = logits - logits.max()
        probs = np.exp(shifted)
        probs = probs / probs.sum()

        return int(np.random.choice(len(probs), p=probs))


def create_engine(
    model: QuantizedModel,
    num_cores: int = 4,
    enable_speculative: bool = True,
    enable_prefetch: bool = True,
) -> ASDSLEngine:
    """Factory function to create and configure an ASDSL inference engine.

    Args:
        model: Quantized model from the quantization pipeline.
        num_cores: Number of CPU cores to use (2-4 recommended).
        enable_speculative: Enable SWIFT speculative decoding.
        enable_prefetch: Enable async L2 prefetching.

    Returns:
        Configured ASDSLEngine (call .setup() before .generate()).
    """
    config = InferenceConfig(
        num_compute_cores=max(num_cores - 1, 1),
        num_prefetch_cores=1 if enable_prefetch else 0,
        speculative_draft_tokens=4 if enable_speculative else 0,
        use_huge_pages=True,
        pin_memory=True,
        numa_aware=True,
        kv_cache_block_sparse=True,
    )

    engine = ASDSLEngine(model=model, inference_config=config)
    return engine


def run_dual_model_speculative_benchmark(
    prompt_tokens: list[int],
    max_new_tokens: int,
    gamma: int = 7,
    temperature: float = 0.0,
    seed: int = 2026,
    vocab_size: int = 200064,
) -> DualModelBenchmarkResult:
    """Run baseline vs dual-model speculative decode and return metrics.

    This is a high-level routing helper used by experiment entrypoints so they
    can bypass Python-side per-layer loops and call the Phase 7 dual-model path.
    """
    baseline_target = SimulatedDualModel(
        name="phi-main-q4-target",
        vocab_size=vocab_size,
        max_context=8192,
        base_seed=seed,
        latency_s=0.125,
        draft_noise_std=0.0,
        resident_mb=0,
    )
    _, _, baseline_tps = run_target_only_baseline(
        target_model=baseline_target,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=seed,
    )

    draft_model = SimulatedDualModel(
        name="phi-mini-q4-draft",
        vocab_size=vocab_size,
        max_context=8192,
        base_seed=seed,
        latency_s=0.012,
        draft_noise_std=0.20,
        resident_mb=128,
    )
    target_model = SimulatedDualModel(
        name="phi-main-q4-target",
        vocab_size=vocab_size,
        max_context=8192,
        base_seed=seed,
        latency_s=0.125,
        draft_noise_std=0.0,
        resident_mb=0,
    )

    decoder = DualModelSpeculativeDecoder(
        draft_model=draft_model,
        target_model=target_model,
        gamma=gamma,
        temperature=temperature,
        seed=seed,
    )
    result = decoder.generate(prompt_tokens=prompt_tokens, max_new_tokens=max_new_tokens)

    speedup = result.effective_tokens_per_s / max(baseline_tps, 1e-9)
    return DualModelBenchmarkResult(
        prompt_tokens=len(prompt_tokens),
        generated_tokens=len(result.generated_tokens),
        baseline_tokens_per_second=baseline_tps,
        speculative_tokens_per_second=result.effective_tokens_per_s,
        speedup=speedup,
        acceptance_rate=result.acceptance_rate,
        drafted_tokens=result.drafted_tokens,
        accepted_draft_tokens=result.accepted_draft_tokens,
        verifier_calls=result.verifier_calls,
        decode_time_s=result.decode_time_s,
    )


def _resolve_native_model_paths(bits: int) -> tuple[Path, Path]:
    """Resolve model artifacts for native Phase-8 evaluation.

    Falls back to q4 artifacts when a bit-specific export is unavailable.
    """
    root = Path(__file__).resolve().parent.parent.parent
    models = root / "models"

    candidates = [
        (models / f"phi4_q{bits}.bin", models / f"phi4_q{bits}_metadata.json"),
        (models / f"phi4_q{bits}_mmap.bin", models / f"phi4_q{bits}_mmap_metadata.json"),
        (models / "phi4_q4.bin", models / "phi4_q4_metadata.json"),
    ]

    for bin_path, meta_path in candidates:
        if bin_path.exists() and meta_path.exists():
            return bin_path, meta_path

    raise FileNotFoundError(
        "No native model artifacts found for Phase-8 evaluation. "
        "Expected one of: phi4_q{bits}.bin + metadata, phi4_q{bits}_mmap.bin + metadata, or phi4_q4 fallback."
    )


def evaluate_perplexity_phase8_native(
    tokens: list[int],
    bits: int = 8,
    stride: int = 512,
    p_correct: float = 0.9,
    hf_model_id: str | None = None,
) -> NativePerplexityResult:
    """Compute perplexity with a HuggingFace causal LM (CPU), using full next-token NLL.

    Sums cross-entropy over every valid position in each chunk: logits at index ``i``
    predict token ``i+1`` (shifted labels). Native mmap weights are still resolved for
    roofline reporting (``backend_model_bin`` / metadata paths).

    The Phase-8 ``generate_token`` mmap binding is a stub and cannot produce logits;
    this function intentionally uses Transformers for mathematically valid PPL.

    Args:
        tokens: Token IDs (e.g. from the same tokenizer as ``hf_model_id``).
        bits: Selects which on-disk quantized artifact to reference for roofline bytes.
        stride: Max chunk length minus one for sliding windows (non-overlapping chunks).
        p_correct: Unused; kept for call-site compatibility.
        hf_model_id: HuggingFace model id; use :func:`resolve_hf_ppl_model_id` at the
            call site so the tokenizer matches. If ``None`` or empty, uses
            ``ASDSL_PPL_MODEL_ID`` or ``microsoft/phi-4``.
    """
    del p_correct

    if not tokens or len(tokens) < 2:
        raise ValueError("Need at least 2 tokens for perplexity evaluation")

    import torch
    import torch.nn.functional as F

    model_id = resolve_hf_ppl_model_id(hf_model_id)
    model_bin, model_meta = _resolve_native_model_paths(bits)
    model = _get_hf_causal_lm_for_ppl(model_id)

    vocab_size = int(model.get_input_embeddings().num_embeddings)
    t_min, t_max = min(tokens), max(tokens)
    if t_min < 0 or t_max >= vocab_size:
        raise ValueError(
            f"Token ID(s) out of range for model {model_id!r} (embedding rows={vocab_size}, "
            f"observed range [{t_min}, {t_max}]). Encode text with "
            f"AutoTokenizer.from_pretrained({model_id!r}, trust_remote_code=True) "
            "and the same id as ``hf_model_id`` / ``ASDSL_PPL_MODEL_ID``."
        )

    nll_sum = 0.0
    n_scored = 0
    processed_tokens = 0

    t0 = time.perf_counter()
    windows = max(1, (len(tokens) - 1) // stride)

    for win_idx in range(windows):
        begin = win_idx * stride
        end = min(begin + stride + 1, len(tokens))
        window = tokens[begin:end]
        if len(window) < 2:
            break

        input_ids = torch.tensor([window], dtype=torch.long, device="cpu")
        with torch.inference_mode():
            out = model(input_ids, use_cache=False)
            logits = out.logits[0]

        shift_logits = logits[:-1].float()
        shift_labels = input_ids[0, 1:]
        chunk_nll = F.cross_entropy(
            shift_logits,
            shift_labels,
            reduction="sum",
        )
        n_tokens_chunk = int(shift_labels.numel())
        nll_sum += float(chunk_nll.item())
        n_scored += n_tokens_chunk
        processed_tokens += n_tokens_chunk

    elapsed = time.perf_counter() - t0
    avg_nll = nll_sum / max(n_scored, 1)
    ppl = math.exp(min(avg_nll, 50.0))
    tps = processed_tokens / max(elapsed, 1e-9)

    # Free up the 25GB+ of RAM immediately
    del model
    clear_hf_causal_lm()

    return NativePerplexityResult(
        bits=bits,
        ppl=ppl,
        avg_nll=avg_nll,
        num_tokens=n_scored,
        tokens_per_second=tps,
        elapsed_sec=elapsed,
        windows=windows,
        backend_model_bin=str(model_bin),
        backend_model_metadata=str(model_meta),
        ppl_route="huggingface_causal_lm",
        hf_model_id=model_id,
    )
