# Inference Guide

This guide covers running end-to-end inference with the ASDSL framework,
including speculative decoding, KV cache management, and performance tuning.

## Quick Start

```python
from asdsl.inference.engine import create_engine
from asdsl.config import InferenceConfig

config = InferenceConfig(
    num_draft_tokens=4,
    max_kv_cache_blocks=64,
    prefetch_lookahead=2,
    enable_huge_pages=False,
)

engine = create_engine(
    model_path="path/to/quantized_model.npz",
    config=config,
)

result = engine.generate(
    prompt_tokens=[1, 2, 3, 4, 5],
    max_new_tokens=100,
)

print(f"Generated {len(result.tokens)} tokens")
print(f"Throughput: {result.tokens_per_second:.1f} tok/s")
print(f"Acceptance rate: {result.acceptance_rate:.1%}")
```

## SWIFT Speculative Decoding

### How It Works

SWIFT (Self-speculative decoding With Intermediate Fast Tokens) generates
multiple draft tokens by running the model with most layers skipped, then
verifies all drafts in a single full forward pass.

```python
from asdsl.speculative.swift import SWIFTDecoder, create_skip_schedule_for_phi3

schedule = create_skip_schedule_for_phi3()
print(f"Draft layers: {schedule.draft_layers}")
print(f"Skip ratio: {schedule.skip_ratio:.0%}")
print(f"Theoretical speedup: {schedule.speedup_estimate:.1f}x")
```

### Tuning Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `num_draft_tokens` | 4 | More drafts = higher throughput if acceptance is high |
| `keep_first` | 4 | First N layers always executed (embedding extraction) |
| `keep_last` | 4 | Last N layers always executed (output refinement) |
| `temperature` | 0.0 | 0 = greedy, higher = more diverse outputs |
| `adaptive_schedule` | True | Auto-tune skip ratio based on acceptance rate |

### Adaptive Schedule

When enabled, the decoder monitors the rolling acceptance rate:
- **>80% acceptance**: Increase skip ratio (skip more layers, faster drafts)
- **<50% acceptance**: Decrease skip ratio (more accurate drafts, fewer rejections)

This self-tunes to the optimal operating point for each prompt/domain.

## KV Cache

The block-sparse KV cache provides fixed-memory attention with intelligent eviction:

```python
from asdsl.inference.kv_cache import BlockSparseKVCache, KVCacheConfig

kv_config = KVCacheConfig(
    num_layers=32,
    num_kv_heads=8,
    head_dim=96,
    max_blocks=64,
    block_size=16,
)
cache = BlockSparseKVCache(kv_config)
```

### Eviction Policy

When the cache is full, blocks are evicted based on importance scores:

1. **Query-centric importance**: Each block's importance is the maximum cosine
   similarity between its keys and the current query state.
2. **Pivot pinning**: Blocks identified as "pivot tokens" (high gradient norms)
   receive boosted importance scores and are less likely to be evicted.
3. **Streaming heads**: The first block in the sequence is always retained
   (attention sink pattern).

## Memory Management

### Pinning

Memory pinning prevents the OS from swapping model weights to disk:

```python
from asdsl.memory.manager import MemoryManager

mm = MemoryManager(enable_pinning=True, enable_huge_pages=False)
region = mm.allocate_for_weights("layer_0", shape=(3072, 3072), dtype=np.float32)
# Weights are now locked in physical RAM
```

### Huge Pages

On Linux, 2MB huge pages reduce TLB misses for large allocations:

```python
mm = MemoryManager(enable_pinning=True, enable_huge_pages=True)
```

**Prerequisite**: Reserve huge pages first:
```bash
echo 1024 | sudo tee /proc/sys/vm/nr_hugepages
```

## Prefetch Orchestrator

The prefetch orchestrator runs a background thread that warms the L2 cache
before each layer's weights are needed:

```python
from asdsl.prefetch.orchestrator import PrefetchOrchestrator

orch = PrefetchOrchestrator(num_layers=32, lookahead=2)
# Register weight buffers for each layer
for i in range(32):
    orch.register_weight_buffer(i, "qkv", qkv_weights[i])
    orch.register_weight_buffer(i, "ffn", ffn_weights[i])

orch.start()
# During inference, call orch.notify_layer_start(i) before each layer
```

## Performance Tuning Checklist

1. **Enable memory pinning** if you have enough RAM (model size + ~30% overhead)
2. **Try huge pages** on Linux for models > 2GB
3. **Set `num_draft_tokens=4`** — this is optimal for most scenarios
4. **Use `adaptive_schedule=True`** to auto-tune skip ratio
5. **Set `prefetch_lookahead=2`** — gives the prefetch thread enough lead time
6. **Monitor acceptance rate** — if < 50%, reduce skip ratio or increase `keep_first`/`keep_last`
7. **Use 3.5-bit average** for best quality/size tradeoff on Phi-3-mini

## Expected Performance (Phi-3-mini, single-core)

| Configuration | Tokens/s | Memory | Quality (Perplexity) |
|---------------|----------|--------|---------------------|
| FP16 baseline | ~5       | 7.6 GB | Reference           |
| ASDSL 4-bit   | ~25      | 2.0 GB | +0.1 ppl            |
| ASDSL 3.5-bit | ~30      | 1.7 GB | +0.3 ppl            |
| ASDSL 3-bit   | ~35      | 1.5 GB | +0.8 ppl            |

*Performance varies by CPU generation. AVX-512 and VNNI provide additional gains.*
