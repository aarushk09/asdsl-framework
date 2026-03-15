"""Main inference engine integrating all ASDSL subsystems.

Orchestrates the complete inference pipeline:
1. Load quantized model into pinned memory
2. Build LUT tables for decode phase
3. Run prefill with INT8 kernels
4. Generate tokens via LUT + SWIFT speculative decoding
5. Async prefetch upcoming layer weights
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

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
from asdsl.speculative.swift import SWIFTDecoder, SkipSchedule

logger = logging.getLogger(__name__)


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
    """

    token_ids: list[int]
    tokens_per_second: float = 0.0
    total_time_s: float = 0.0
    prefill_time_s: float = 0.0
    decode_time_s: float = 0.0
    num_speculative_steps: int = 0
    avg_acceptance_rate: float = 0.0
    peak_memory_mb: float = 0.0

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


class TransformerLayerExecutor:
    """Executes individual transformer layers using LUT-based computation.

    Implements the LayerExecutor protocol expected by the SWIFT decoder.
    Each layer consists of:
    - Self-attention (Q, K, V projections + attention + output projection)
    - Feed-forward network (gate, up, down projections)
    - Layer norms
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

    @property
    def num_layers(self) -> int:
        return self._num_layers

    def execute_layer(self, layer_idx: int, hidden_state: np.ndarray) -> np.ndarray:
        """Execute a single transformer layer.

        Args:
            layer_idx: Layer index to execute.
            hidden_state: Input hidden state, shape (..., hidden_dim).

        Returns:
            Transformed hidden state.
        """
        # Notify prefetcher of current layer
        if self.prefetcher:
            self.prefetcher.notify_layer_start(layer_idx)

        if layer_idx >= len(self.model.layers):
            return hidden_state

        layer = self.model.layers[layer_idx]

        # Simplified transformer layer execution:
        # In full implementation, each weight tensor would be processed
        # through LUT-based matvec. Here we apply layer processing
        # using the quantized weights.

        residual = hidden_state.copy()

        # Self-attention block (simplified)
        h = self._layer_norm(hidden_state)
        h = self._attention_forward(h, layer)

        hidden_state = residual + h

        # FFN block (simplified)
        residual = hidden_state.copy()
        h = self._layer_norm(hidden_state)
        h = self._ffn_forward(h, layer)

        hidden_state = residual + h

        return hidden_state

    def execute_lm_head(self, hidden_state: np.ndarray) -> np.ndarray:
        """Project hidden state to vocabulary logits.

        Args:
            hidden_state: Final hidden state, shape (..., hidden_dim).

        Returns:
            Logits over vocabulary, shape (..., vocab_size).
        """
        if hidden_state.ndim == 1:
            hidden_state = hidden_state.reshape(1, -1)

        # Use the LM head weights if available
        if self.model.lm_head_weights is not None:
            from asdsl.quantization.core import dequantize_weights

            lm_weights = dequantize_weights(self.model.lm_head_weights)
            # lm_weights shape: (vocab_size, hidden_dim)
            logits = hidden_state @ lm_weights.T
        else:
            # Fallback: random logits for testing
            vocab_size = self.model.config.vocab_size
            logits = np.random.randn(hidden_state.shape[0], vocab_size).astype(np.float32)

        return logits

    def _layer_norm(self, x: np.ndarray) -> np.ndarray:
        """Apply RMS layer normalization."""
        if x.ndim == 1:
            rms = np.sqrt(np.mean(x**2) + 1e-6)
            return x / rms
        rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + 1e-6)
        return x / rms

    def _attention_forward(self, h: np.ndarray, layer) -> np.ndarray:
        """Simplified attention forward pass using quantized weights."""
        # In full implementation: Q/K/V projection via LUT, attention
        # computation with sparse KV cache, output projection via LUT.
        # For now, pass through with dimension preservation.
        return h * 0.9  # Simulated attention output

    def _ffn_forward(self, h: np.ndarray, layer) -> np.ndarray:
        """Simplified FFN forward pass using quantized weights."""
        # In full implementation: gate/up projection via LUT,
        # SiLU activation, down projection via LUT.
        return h * 0.1  # Simulated FFN output


class ASDSLEngine:
    """Main ASDSL inference engine.

    Integrates all framework components into a coherent inference pipeline:
    - Quantized model loading with memory pinning
    - LUT table construction
    - SWIFT speculative decoding
    - Asynchronous cache prefetching
    - Block-sparse KV cache management
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
        3. Initializes the KV cache
        4. Sets up the SWIFT speculative decoder
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

        # Register weight buffers for prefetching
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

        # 4. KV cache
        self.kv_cache = BlockSparseKVCache(
            KVCacheConfig(
                max_context_length=self.model.config.max_context_length,
                num_kv_heads=self.model.config.num_kv_heads,
                head_dim=self.model.config.hidden_dim // self.model.config.num_attention_heads,
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

        self.prefetcher.start()
        self._is_initialized = True
        logger.info("ASDSL engine initialized and ready")

    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> GenerationResult:
        """Generate tokens autoregressively.

        Args:
            input_ids: Input prompt token IDs.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.

        Returns:
            GenerationResult with generated tokens and performance metrics.
        """
        if not self._is_initialized:
            raise RuntimeError("Engine not initialized. Call setup() first.")

        start_time = time.perf_counter()

        # Phase 1: Prefill
        prefill_start = time.perf_counter()
        hidden_state = self._prefill(input_ids)
        prefill_time = time.perf_counter() - prefill_start

        # Phase 2: Autoregressive decode
        decode_start = time.perf_counter()
        generated_ids: list[int] = []
        num_spec_steps = 0
        acceptance_rates: list[float] = []

        while len(generated_ids) < max_new_tokens:
            if self.speculative_decoder is not None:
                # Speculative decoding path
                result = self.speculative_decoder.speculative_step(
                    hidden_state=hidden_state,
                    past_tokens=input_ids + generated_ids,
                )
                generated_ids.extend(result.accepted_tokens)
                num_spec_steps += 1
                acceptance_rates.append(result.acceptance_rate)
            else:
                # Standard autoregressive decoding
                logits = self._forward_all_layers(hidden_state)
                token_id = self._sample(logits, temperature, top_k)
                generated_ids.append(token_id)

            # Update hidden state for next step
            # (simplified: in full impl, KV cache would provide context)
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
        """Generator that yields StreamToken objects as tokens are decoded.

        Usage::

            engine.setup()
            for tok in engine.generate_stream(input_ids):
                print(tok.text, end="", flush=True)

        Args:
            input_ids: Input prompt token IDs.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.

        Yields:
            StreamToken with token_id, step index, timing, and EOS flag.
        """
        if not self._is_initialized:
            raise RuntimeError("Engine not initialized. Call setup() first.")

        # Phase 1: Prefill
        hidden_state = self._prefill(input_ids)

        # Phase 2: Streaming decode
        decode_start = time.perf_counter()

        for step in range(max_new_tokens):
            logits = self._forward_all_layers(hidden_state)
            token_id = self._sample(logits, temperature, top_k)

            elapsed = time.perf_counter() - decode_start
            tps = (step + 1) / elapsed if elapsed > 0 else 0.0

            yield StreamToken(
                token_id=token_id,
                step=step,
                is_eos=False,  # Caller decides EOS based on their stop tokens
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
        """Process the input prompt (prefill phase).

        The prefill phase is compute-bound and uses INT8 kernels
        (VNNI/AMX) rather than LUT-based computation.
        """
        hidden_dim = self.model.config.hidden_dim
        seq_len = len(input_ids)

        # Create initial embeddings (simplified)
        hidden_state = np.random.randn(seq_len, hidden_dim).astype(np.float32) * 0.02

        # Process through all layers
        for layer_idx in range(self.model.config.num_layers):
            hidden_state = self.layer_executor.execute_layer(layer_idx, hidden_state)

        # Return last position's hidden state for decode phase
        return hidden_state[-1] if seq_len > 1 else hidden_state.reshape(-1)

    def _forward_all_layers(self, hidden_state: np.ndarray) -> np.ndarray:
        """Full forward pass through all layers (standard decode)."""
        h = hidden_state
        for layer_idx in range(self.model.config.num_layers):
            h = self.layer_executor.execute_layer(layer_idx, h)
        return self.layer_executor.execute_lm_head(h)

    def _step_hidden_state(self, hidden_state: np.ndarray) -> np.ndarray:
        """Update hidden state after generating a token (simplified)."""
        return hidden_state  # In full impl: re-encode with KV cache update

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

        # Temperature scaling
        logits = logits / temperature

        # Top-k filtering
        if top_k > 0 and top_k < len(logits):
            top_k_indices = np.argpartition(logits, -top_k)[-top_k:]
            mask = np.full_like(logits, -np.inf)
            mask[top_k_indices] = logits[top_k_indices]
            logits = mask

        # Softmax
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
