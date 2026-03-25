import time
import os
os.environ["USE_TF"] = "0"
import json
import torch
import numpy as np
from transformers import AutoTokenizer
import asdsl.kernels._native_forward as native_forward
import warnings
warnings.filterwarnings('ignore')

def main():
    print("==================================================")
    print(" ASDSL C++ Inference Engine - Hardware Accelerated")
    print("                Interactive Chat                  ")
    print("==================================================")

    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"[*] Loading tokenizer '{model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Hardcoded known config for TinyLlama string layout params to skip full load
    dim = 2048
    hidden_dim = 5632
    num_heads = 32
    num_kv_heads = 4
    num_layers = 22
    head_dim = dim // num_heads
    vocab_size = 32000

    print(f"[*] Architecture: {num_layers} Layers | Dim: {dim} | Heads: {num_heads} | KV-Heads: {num_kv_heads}")
    print("[*] Memory-mapping Q4 model weights (zero-copy)...")

    try:
        with open("models/tinyllama_q4_metadata.json", "r") as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print("[!] Error: Could not find models/tinyllama_q4_metadata.json. Please run quantize_tinyllama.py first.")
        return

    mmap_store = native_forward.MmapWeights("models/tinyllama_q4.bin", metadata)

    print("[*] Initialization complete. C++ Engine Ready.")
    print("Type '/reset' to clear context. Type 'exit' to quit.")
    print("==================================================\n")

    max_seq_len = 2048
    cache = native_forward.KVCache(num_layers, max_seq_len, num_kv_heads, head_dim)
    
    current_seq_pos = 0

    # We start without a system prompt in exactly this form or we can init the system prompt immediately.
    # We will just manage string concatenations directly. For TinyLlama, it's:
    # <|system|>\nYou are a helpful assistant.</s>\n<|user|>\n...</s>\n<|assistant|>\n
    
    system_prompt = "<|system|>\nYou are a helpful AI assistant.</s>\n"
    
    system_ids = tokenizer.encode(system_prompt, add_special_tokens=True) # Usually includes BOS token initially

    while True:
        try:
            prompt = input("\nUser: ")
        except (KeyboardInterrupt, EOFError):
            break

        if prompt.strip().lower() == "exit":
            break
            
        if prompt.strip().lower() == "/reset":
            cache = native_forward.KVCache(num_layers, max_seq_len, num_kv_heads, head_dim)
            current_seq_pos = 0
            print("[*] KV Cache Reset.")
            continue

        # Format input
        if current_seq_pos == 0:
            # First turn: System block + User block
            formatted_text = system_prompt + f"<|user|>\n{prompt}</s>\n<|assistant|>\n"
            input_ids = tokenizer.encode(formatted_text, add_special_tokens=True)
        else:
            # Multi-turn: Just append User block
            formatted_text = f"<|user|>\n{prompt}</s>\n<|assistant|>\n"
            # Important: we don't want another BOS token
            input_ids = tokenizer.encode(formatted_text, add_special_tokens=False)

        if current_seq_pos + len(input_ids) + 50 >= max_seq_len:
            print("[!] Exceeded context limit. Resetting cache.")
            cache = native_forward.KVCache(num_layers, max_seq_len, num_kv_heads, head_dim)
            current_seq_pos = 0
            formatted_text = system_prompt + f"<|user|>\n{prompt}</s>\n<|assistant|>\n"
            input_ids = tokenizer.encode(formatted_text, add_special_tokens=True)

        print("Assistant: ", end="", flush=True)

        prefill_start = time.perf_counter()

        # Prefill route: batch GEMM for T>1, scalar GEMV decode path for T=1.
        prefill_ids = input_ids[:-1]
        if len(prefill_ids) > 1:
            native_forward.prefill_prompt_tokens(
                np.asarray(prefill_ids, dtype=np.int32),
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

        # The last token in the input triggers the next output token
        next_token = native_forward.generate_token(
            input_ids[-1], current_seq_pos, mmap_store,
            num_layers, dim, hidden_dim, num_heads, num_kv_heads, head_dim, vocab_size,
            cache
        )
        current_seq_pos += 1
        
        prefill_end = time.perf_counter()
        ttft = prefill_end - prefill_start

        # Decode Phase
        decode_start = time.perf_counter()
        
        generated_count = 1
        max_new = 500
        eos_id = tokenizer.eos_token_id
        
        generated_ids = [next_token]

        # Use skip_special_tokens=True so we NEVER see <s> or </s>.
        # This makes string slicing fundamentally robust.
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Stop early if the very first token somehow hallucinates the stop sequence
        if "<|user|>" in text:
            text = text.split("<|user|>")[0]
            print(text, end="", flush=True)
            generated_count = 1  # forces skip of while loop
        else:
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
                
                if "<|user|>" in new_text:
                    # Strip the <|user|> tag and everything after it
                    new_text = new_text.split("<|user|>")[0]
                    chunk = new_text[len(text):]
                    print(chunk, end="", flush=True)
                    break

                chunk = new_text[len(text):]
                print(chunk, end="", flush=True)
                text = new_text

        decode_end = time.perf_counter()
        decode_time = decode_end - decode_start
        # tps avoids div by zero if generated_count == 1 and decode_time is ~0
        tps = generated_count / decode_time if decode_time > 0 else 0.0

        print(f"\n[TTFT: {ttft:.3f}s | Decode: {tps:.2f} tok/s | Tokens: {generated_count}]")

if __name__ == '__main__':
    main()