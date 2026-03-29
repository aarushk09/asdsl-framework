import time
import torch
import numpy as np
import sys
from experiments.phi4_cpu_run import WeightStore, generate
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/Phi-4-multimodal-instruct", trust_remote_code=True)

store = WeightStore(bits=4)
store.load()
store.load_slim('phi4_slim_meta.json')
store._use_lut_gemv = True
store._use_native_gemv = True

store.warm_cache()

metrics = []
generate("What is 2+2?", store, tokenizer, max_new_tokens=4, bench_metrics_out=metrics)
print("Result:", metrics)

