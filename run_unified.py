import time
import argparse
import numpy as np
import torch
import gc
from experiments.phi4_cpu_run import WeightStore
from asdsl.kernels._native_unified import EngineConfig, UnifiedEngine

def main():
    print("Loading weights...")
    store = WeightStore(bits=4, group_size=64, enable_qcsd=False, enable_sparse=False)
    store.load()
    store.warm_cache()
    print("Loaded. Preparing UnifiedEngine...")

    config = EngineConfig()
    config.num_layers = 32
    config.hidden_size = 3072
    config.num_heads = 32
    config.num_kv_heads = 8
    config.head_dim = 96
    config.intermediate_size = 8192
    config.vocab_size = 100352
    config.rms_norm_eps = 1e-5
    config.group_size = 64
    config.max_seq_len = 2048

    token_embd = store.embed_f16.numpy().astype(np.float32)
    output_norm = store.final_norm.numpy().astype(np.float32)
    output_proj = token_embd  # Tied embedding!
    
    # RoPE tables
    cos_table = store._cos_table.numpy().astype(np.float32)
    sin_table = store._sin_table.numpy().astype(np.float32)

    layers_dict = {}
    for l in range(config.num_layers):
        l_dict = {
            'rms_att': store.layer_norms[l]['input_layernorm'].numpy().astype(np.float32),
            'rms_ffn': store.layer_norms[l]['post_attention_layernorm'].numpy().astype(np.float32),
            'qkv_proj': store._quant_packed_np[f"{l}.qkv_proj"],
            'qkv_scales': store._quant_sc_np[f"{l}.qkv_proj"].astype(np.float32),
            'o_proj': store._quant_packed_np[f"{l}.o_proj"],
            'o_scales': store._quant_sc_np[f"{l}.o_proj"].astype(np.float32),
            'gate_up_proj': store._quant_packed_np[f"{l}.gate_up_proj"],
            'gate_up_scales': store._quant_sc_np[f"{l}.gate_up_proj"].astype(np.float32),
            'down_proj': store._quant_packed_np[f"{l}.down_proj"],
            'down_scales': store._quant_sc_np[f"{l}.down_proj"].astype(np.float32),
        }
        layers_dict[l] = l_dict

    print("Init UnifiedEngine")
    engine = UnifiedEngine(
        config,
        token_embd,
        output_norm,
        output_proj,
        cos_table,
        sin_table,
        layers_dict
    )
    print("Done. Generating...")
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-multimodal-instruct", trust_remote_code=True)
    prompt = "Explain gravity in one sentence."
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    
    t0 = time.perf_counter()
    out = engine.generate(prompt_tokens, 20)
    t1 = time.perf_counter()
    
    print(f"Generated {len(out)} total tokens in {t1-t0:.2f}s ({20/(t1-t0):.2f} tok/s)")
    print("Output:", tokenizer.decode(out))

if __name__ == '__main__':
    main()
