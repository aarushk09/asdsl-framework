import time
import math
import os
os.environ["USE_TF"] = "0"

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import asdsl.kernels._native_forward as native_forward
import warnings
warnings.filterwarnings('ignore')

def main():
    print("==================================================")
    print(" ASDSL C++ Inference Engine - Hardware Accelerated")
    print("==================================================")
    
    # TinyLlama fits easily in memory, uses GQA matching our C++ pipeline, and RoPE Theta 10000
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"[*] Loading HuggingFace model '{model_id}'...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Load directly to CPU
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    
    cfg = model.config
    dim = cfg.hidden_size
    hidden_dim = cfg.intermediate_size
    num_heads = cfg.num_attention_heads
    num_kv_heads = cfg.num_key_value_heads
    num_layers = cfg.num_hidden_layers
    head_dim = dim // num_heads
    vocab_size = cfg.vocab_size

    print(f"[*] Architecture: {num_layers} Layers | Dim: {dim} | Heads: {num_heads} | KV-Heads: {num_kv_heads}")
    print("[*] Extracting raw float32 weights from PyTorch to NumPy for C++ Engine...")

    # Extract weights to contiguous numpy arrays
    layers_w_cpp = []
    for l in range(num_layers):
        layer = model.model.layers[l]
        rms1_w = layer.input_layernorm.weight.detach().numpy().astype(np.float32)
        
        q_w = layer.self_attn.q_proj.weight.detach()
        k_w = layer.self_attn.k_proj.weight.detach()
        v_w = layer.self_attn.v_proj.weight.detach()
        qkv_w = torch.cat([q_w, k_w, v_w], dim=0).numpy().flatten().astype(np.float32)
        
        o_w = layer.self_attn.o_proj.weight.detach().numpy().flatten().astype(np.float32)
        
        rms2_w = layer.post_attention_layernorm.weight.detach().numpy().astype(np.float32)
        
        gate_w = layer.mlp.gate_proj.weight.detach().numpy().flatten().astype(np.float32)
        up_w = layer.mlp.up_proj.weight.detach().numpy().flatten().astype(np.float32)
        down_w = layer.mlp.down_proj.weight.detach().numpy().flatten().astype(np.float32)
        
        layers_w_cpp.append((rms1_w, qkv_w, o_w, rms2_w, gate_w, up_w, down_w))
        
    token_emb = model.model.embed_tokens.weight.detach().numpy().flatten().astype(np.float32)
    final_rms_w = model.model.norm.weight.detach().numpy().astype(np.float32)
    lm_head_w = model.lm_head.weight.detach().numpy().flatten().astype(np.float32)
    
    # Clean up massive torch model object if possible to free RAM
    del model
    import gc
    gc.collect()

    print("[*] Initialization complete. C++ Engine Ready.")
    print("==================================================\n")

    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    
    max_seq_len = 1024
    cache = native_forward.KVCache(num_layers, max_seq_len, num_kv_heads, head_dim)
    
    print(f"Prompt: '{prompt}'")
    print("\n--- PREFILL PHASE ---")
    
    seq_pos = 0
    next_token = input_ids[0]
    
    # Feed prompt tokens sequentially to build KV cache history
    for i in range(len(input_ids) - 1):
        native_forward.generate_token(
            input_ids[i], seq_pos, token_emb, layers_w_cpp,
            final_rms_w, lm_head_w,
            num_layers, dim, hidden_dim, num_heads, num_kv_heads, head_dim, vocab_size,
            cache
        )
        seq_pos += 1

    # The last token in the prompt generates our first new token!
    last_prompt_token = input_ids[-1]
    next_token = native_forward.generate_token(
        last_prompt_token, seq_pos, token_emb, layers_w_cpp,
        final_rms_w, lm_head_w,
        num_layers, dim, hidden_dim, num_heads, num_kv_heads, head_dim, vocab_size,
        cache
    )
    seq_pos += 1
    
    print("\n--- DECODE PHASE ---")
    # To print beautifully to console
    print(prompt, end="", flush=True)
    decoded_first = tokenizer.decode([next_token], skip_special_tokens=True)
    print(decoded_first, end="", flush=True)
    
    start_time = time.perf_counter()
    tokens_generated = 1  # We already generated one token
    max_new_tokens = 50
    
    while tokens_generated < max_new_tokens:
        if next_token == tokenizer.eos_token_id:
            break
            
        next_token = native_forward.generate_token(
            next_token, seq_pos, token_emb, layers_w_cpp,
            final_rms_w, lm_head_w,
            num_layers, dim, hidden_dim, num_heads, num_kv_heads, head_dim, vocab_size,
            cache
        )
        seq_pos += 1
        tokens_generated += 1
        
        word = tokenizer.decode([next_token], skip_special_tokens=True)
        print(word, end="", flush=True)

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    tps = tokens_generated / elapsed
    
    print("\n\n==================================================")
    print(f"Total Time:    {elapsed:.3f} seconds")
    print(f"Generated:     {tokens_generated} tokens")
    print(f"Throughput:    {tps:.2f} tok/s")
    print("==================================================")

if __name__ == '__main__':
    main()
