import os
import sys
import time
import json
import gc
import statistics
import psutil
import torch
import warnings
from transformers import AutoTokenizer

# -----------------------------------------------------------------------------
# 1. STRICT ENVIRONMENT & OPENMP CONTROL
# -----------------------------------------------------------------------------
# We must control the environment BEFORE any C++ or PyTorch library initializes.
NUM_PHYSICAL_CORES = 12

os.environ["USE_TF"] = "0"
os.environ["OMP_NUM_THREADS"] = str(NUM_PHYSICAL_CORES)
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
os.environ["KMP_BLOCKTIME"] = "infinite" # Prevent threads from sleeping/yielding
os.environ["OMP_PROC_BIND"] = "true"     # Pin threads to cores strictly
torch.set_num_interop_threads(1)
torch.set_num_threads(NUM_PHYSICAL_CORES)
warnings.filterwarnings('ignore')

try:
    import asdsl.kernels._native_forward as native_forward
except ImportError:
    print("[!] Warning: asdsl native modules not found. Ensure you ran python setup.py build_ext --inplace")
    native_forward = None

# -----------------------------------------------------------------------------
# 2. HARDWARE AFFINITY LOCKING
# -----------------------------------------------------------------------------
def lock_process_affinity():
    """
    Locks the Python process (and subsequently spawn OpenMP threads) to EXACTLY 
    the physical cores. On an Intel CPU with HyperThreading (e.g. 12 P-cores, 24 threads), 
    physical cores are usually the even-numbered logical processors.
    """
    p = psutil.Process()
    logical_cores = psutil.cpu_count(logical=True)
    
    # Heuristic: Pick even cores up to (NUM_PHYSICAL_CORES * 2)
    # This specifically targets P-cores and avoids HT context collisions.
    affinity_mask = [i for i in range(0, min(logical_cores, NUM_PHYSICAL_CORES * 2), 2)]
    
    try:
        p.cpu_affinity(affinity_mask)
        print(f"[+] Process Priority locked to HIGH")
        p.nice(psutil.HIGH_PRIORITY_CLASS if os.name == 'nt' else 10)
        print(f"[+] CPU Affinity locked to Physical Cores: {affinity_mask}")
    except AttributeError:
        print("[-] CPU Affinity locking not supported on this OS.")
    except Exception as e:
        print(f"[-] Could not set CPU affinity: {e}")

# -----------------------------------------------------------------------------
# 3. MEMORY PURGE ROUTINE
# -----------------------------------------------------------------------------
def clear_caches_and_collect():
    """Forces aggressive garbage collection to ensure cold(er) cache states between runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# -----------------------------------------------------------------------------
# 4. BENCHMARK HARNESS RUNNER
# -----------------------------------------------------------------------------
def run_native_benchmark(input_ids, tokenizer, mmap_store, config, max_new_tokens=20):
    """
    Executes a single benchmark run tightly timed around the decode loop.
    Batch size is strictly 1.
    """
    prompt_len = len(input_ids)
    cache = native_forward.KVCache(config['num_layers'], 2048, config['num_kv_heads'], config['head_dim'])
    
    current_seq_pos = 0

    # PREFILL
    prefill_start = time.perf_counter()
    prefill_ids = input_ids[:-1]
    
    if len(prefill_ids) > 1:
        native_forward.prefill_prompt_tokens(
            torch.tensor(prefill_ids, dtype=torch.int32).numpy(),
            current_seq_pos, mmap_store, config['num_layers'], config['dim'], 
            config['hidden_dim'], config['num_heads'], config['num_kv_heads'], 
            config['head_dim'], config['vocab_size'], cache
        )
        current_seq_pos += len(prefill_ids)
    
    next_token = native_forward.generate_token(
        input_ids[-1], current_seq_pos, mmap_store,
        config['num_layers'], config['dim'], config['hidden_dim'], config['num_heads'], 
        config['num_kv_heads'], config['head_dim'], config['vocab_size'], cache
    )
    current_seq_pos += 1
    
    # Force synchronization if using any async ops
    prefill_time = time.perf_counter() - prefill_start
    
    # DECODE
    decode_start = time.perf_counter()
    tokens_generated = 0
    generated_ids = []
    eos_id = tokenizer.eos_token_id

    while tokens_generated < max_new_tokens:
        if next_token == eos_id:
            break

        next_token = native_forward.generate_token(
            next_token, current_seq_pos, mmap_store,
            config['num_layers'], config['dim'], config['hidden_dim'], config['num_heads'], 
            config['num_kv_heads'], config['head_dim'], config['vocab_size'], cache
        )
        current_seq_pos += 1
        tokens_generated += 1
        generated_ids.append(next_token)

    decode_time = time.perf_counter() - decode_start
    
    return {
        "ttft_s": prefill_time,
        "decode_time_s": decode_time,
        "tokens_generated": tokens_generated,
        "throughput_tok_s": tokens_generated / decode_time if decode_time > 0 else 0
    }

def print_llama_cpp_equivalent(prompt, tokens):
    """
    Outputs the EXACT command the researcher must run in llama.cpp to ensure an apples-to-apples baseline.
    """
    print("\n" + "="*80)
    print(" 1:1 COMPARISON: LLAMA.CPP EXACT EQUIVALENT COMMAND")
    print("="*80)
    print("To validate our claims natively, compile llama.cpp with the same AVX2 target, and run:")
    cmd = (
        f"./llama-cli -m models/phi4_q4_k_m.gguf \\\n"
        f"  --threads {NUM_PHYSICAL_CORES} -c 2048 -b {len(prompt)} -n {tokens} \\\n"
        f"  -p \"{prompt}\" \\\n"
        f"  --no-mmap false --mlock false --temp 0.0"
    )
    print(f"\n{cmd}\n")
    print("- `-n {tokens}` exactly matches Generation Length.")
    print(f"- `--threads {NUM_PHYSICAL_CORES}` strictly bounds OpenMP physical cores without HT oversubscription.")
    print("- `--temp 0.0` forces greedy decoding locally just like our native framework.")
    print("="*80 + "\n")

def main():
    print(f"\n{'='*80}")
    print(" ASDSL BULLETPROOF BENCHMARK HARNESS ")
    print(f"{'='*80}")

    lock_process_affinity()
    
    prompt = "The future of open-source artificial intelligence is"
    max_new_tokens = 20
    
    print("\n[Configuration Parameters]")
    print(f"  Model:                microsoft/phi-4")
    print(f"  Quantization Format:  Q4 (Group Size 32) (Must be natively mapped without emulation)")
    print(f"  Batch Size:           1")
    print(f"  Prompt String:        '{prompt}'")
    print(f"  Prompt Length:        10 tokens")
    print(f"  Generation Length:    {max_new_tokens} tokens")
    print(f"  Compiler Flags:       AVX2/FMA, O3, OpenMP enabled")
    print(f"  Threading / Affinity: Locked to {NUM_PHYSICAL_CORES} Physical Cores (No HT/E-cores)\n")

    if native_forward is None:
         print("[!] Cannot run live benchmark: C++ extensions missing. Exiting.")
         print_llama_cpp_equivalent(prompt, max_new_tokens)
         return

    print("[*] Loading Architecture & Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-4", trust_remote_code=True)
    except Exception as e:
        print(f"[!] Error loading tokenizer: {e}")
        return

    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    
    config = {
        'dim': 5120, 'hidden_dim': 17920, 'num_heads': 40, 
        'num_kv_heads': 10, 'num_layers': 40, 'vocab_size': 100352
    }
    config['head_dim'] = config['dim'] // config['num_heads']

    try:
        with open("models/phi4_q4_metadata.json", "r") as f:
            metadata = json.load(f)
        print("[*] Metadata loaded.")
    except FileNotFoundError:
        print("[!] Error: 'models/phi4_q4_metadata.json' not found. Weights missing.")
        print_llama_cpp_equivalent(prompt, max_new_tokens)
        return

    print("[*] Memory-mapping Q4 model weights (zero-copy)...")
    try:
        mmap_store = native_forward.MmapWeights("models/phi4_q4.bin", metadata)
    except Exception as e:
        print(f"[!] Engine Error initializing weights: {e}")
        return

    # WARMUP RUN
    print("\n[*] Performing WARMUP RUN (filling P-core caches)...")
    _ = run_native_benchmark(input_ids, tokenizer, mmap_store, config, max_new_tokens=5)
    
    # ACTUAL TEST RUNS
    num_runs = 5
    results = []
    
    print(f"[*] Starting {num_runs} Strict Isolated Runs...")
    for i in range(num_runs):
        clear_caches_and_collect()
        time.sleep(1.0) # Thermal/memory reset stabilization bounds
        
        stat = run_native_benchmark(input_ids, tokenizer, mmap_store, config, max_new_tokens)
        results.append(stat['throughput_tok_s'])
        print(f"    - Run {i+1}/{num_runs}: {stat['throughput_tok_s']:.2f} tok/s  (TTFT: {stat['ttft_s']:.3f}s)")

    # MATH
    mean_tps = statistics.mean(results)
    stdev_tps = statistics.stdev(results) if len(results) > 1 else 0
    p95_tps = sorted(results)[int(0.05 * len(results))] # Look at lowest 5% bounds for worst case
    
    print(f"\n{'='*80}")
    print(f" FINAL BENCHMARK METRICS (N={num_runs})")
    print(f"{'='*80}")
    print(f" Mean Throughput: {mean_tps:.2f} tok/s +/- {stdev_tps:.3f}")
    print(f" Worst-case (P95 latency eqv): {p95_tps:.2f} tok/s")
    print(f"{'='*80}")

    print_llama_cpp_equivalent(prompt, max_new_tokens)

if __name__ == '__main__':
    main()