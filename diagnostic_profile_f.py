#!/usr/bin/env python3
import sys, json, time, gc, os, contextlib, io
sys.path.insert(0, '.')
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_JAX', '0')
os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')

from experiments.phi4_cpu_run import WeightStore, generate, set_thread_count
from transformers import AutoTokenizer

set_thread_count(12)
tokenizer = AutoTokenizer.from_pretrained('microsoft/Phi-4-multimodal-instruct', trust_remote_code=True)

store = WeightStore(bits=4, group_size=None, enable_qcsd=False, draft_bits=2, enable_sparse=False)
store.load()
store.warm_cache()
gc.collect()
store.load_fatrelu('phi4_fatrelu_thresholds.json', adaptive=True)
gc.collect()

store._use_native_gemv = True
store._use_lut_gemv = False
store._enable_sparse = True
store._sparsity_threshold = 0.0

metrics = []
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    generate('', store, tokenizer, max_new_tokens=16, bench_metrics_out=metrics)

sparse_T = int(getattr(store, '_sparse_down_proj_T_calls', 0))
dense_T = int(getattr(store, '_dense_down_proj_fallback_calls', 0))
print(f'sparse_T={sparse_T} dense_fallback={dense_T}')
if metrics:
    print(f'tps={metrics[0].get("tokens_per_second", 0):.3f}')
