import time
import os
os.environ["USE_TF"] = "0"
import json
import torch
from transformers import AutoTokenizer
import asdsl.kernels._native_forward as native_forward
import warnings
warnings.filterwarnings('ignore')

def main():
    print("==================================================")
    print(" ASDSL C++ Inference Engine - Hardware Accelerated")
    print("==================================================")

    model_id = "microsoft/phi-4"
    print(f"[*] Loading tokenizer '{model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    dim = 5120
    hidden_dim = 17920
    num_heads = 40
    num_kv_heads = 10
    num_layers = 40
    head_dim = dim // num_heads
    vocab_size = 100352

    print(f"[*] Architecture: {num_layers} Layers | Dim: {dim} | Heads: {num_heads} | KV-Heads: {num_kv_heads}")
    print("[*] Memory-mapping Q4 model weights (zero-copy)...")

    try:
        with open("models/phi4_q4_metadata.json", "r") as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print("[!] Error. Please run quantize_phi4_mock.py first.")
        return

    mmap_store = native_forward.MmapWeights("models/phi4_q4.bin", metadata)

    try:
        native_forward.pin_openmp_threads_to_pcores(True)
        pcores = native_forward.detected_pcore_count()
        if pcores > 0:
            print(f"[*] OpenMP pinned to {pcores} detected P-cores")
    except Exception:
        pass

    print("[*] Initialization complete. C++ Engine Ready.")
    print("==================================================\n")

    max_seq_len = 2048
    cache = native_forward.KVCache(num_layers, max_seq_len, num_kv_heads, head_dim)
    
    current_seq_pos = 0

    prompt = "The future of open-source artificial intelligence is"
    print(f"Prompt: '{prompt}'")
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)

    print("\n--- PREFILL PHASE ---")
    prefill_start = time.perf_counter()

    prefill_ids = input_ids[:-1]
    if len(prefill_ids) > 1:
        native_forward.prefill_prompt_tokens(
            torch.tensor(prefill_ids, dtype=torch.int32).numpy(),
            current_seq_pos,
            mmap_store,
            num_layers,
            dim,
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            vocab_size,
            cache,
        )
        current_seq_pos += len(prefill_ids)
    elif len(prefill_ids) == 1:
        native_forward.generate_token(
            prefill_ids[0], current_seq_pos, mmap_store,
            num_layers, dim, hidden_dim, num_heads, num_kv_heads, head_dim, vocab_size,
            cache
        )
        current_seq_pos += 1

    next_token = native_forward.generate_token(
        input_ids[-1], current_seq_pos, mmap_store,
        num_layers, dim, hidden_dim, num_heads, num_kv_heads, head_dim, vocab_size,
        cache
    )
    current_seq_pos += 1
    
    prefill_end = time.perf_counter()
    ttft = prefill_end - prefill_start

    print(f"\n--- DECODE PHASE ---")
    decode_start = time.perf_counter()
    
    generated_count = 1
    max_new = 5
    eos_id = tokenizer.eos_token_id

    generated_ids = [next_token]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(text, end="", flush=True)

    while generated_count < max_new:
        if next_token == eos_id:
            break
            
        next_token = native_forward.generate_token(
            next_token, current_seq_pos, mmap_store,
            num_layers, dim, hidden_dim, num_heads, num_kv_heads, head_dim, vocab_size,
            cache
        )
        current_seq_pos += 1
        
        if next_token == eos_id:
            break
            
        generated_ids.append(next_token)
        generated_count += 1
        
        new_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        chunk = new_text[len(text):]
        print(chunk, end="", flush=True)
        text = new_text

    decode_end = time.perf_counter()
    decode_time = decode_end - decode_start
    tps = generated_count / decode_time if decode_time > 0 else 0.0

    print(f"\n\n==================================================")
    print(f"Total Time:    {decode_time+ttft:.3f} seconds")
    print(f"Generated:     {generated_count} tokens")
    print(f"Throughput:    {tps:.2f} tok/s")
    print(f"TTFT:          {ttft:.3f} s")
    print(f"==================================================")

if __name__ == '__main__':
    main()