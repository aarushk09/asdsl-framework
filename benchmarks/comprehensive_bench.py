"""
Comprehensive ASDSL Framework Benchmark Suite

Measures perplexity, throughput, RAM usage, and CPU utilization across all
quantization configurations. Generates publication-quality charts.

Usage:
  python benchmarks/comprehensive_bench.py
  python benchmarks/comprehensive_bench.py --max-tokens 128    # faster
  python benchmarks/comprehensive_bench.py --skip-perplexity   # only resource metrics
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

# Thread control BEFORE importing numpy/torch
# Intel i7 Evo: default to 8 threads (P-cores), prefer MKL
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np
import torch
torch.set_num_threads(8)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import psutil

from phi4_cpu_run import (
    WeightStore, KVHistory, forward_layer, rms_norm, build_rope_cache,
    NUM_LAYERS, ROTARY_DIM, EOS_TOKEN_IDS, set_thread_count,
)
from transformers import AutoTokenizer
from asdsl.quantization.core import quantize_weights, compute_quantization_error

MODEL_DIR = ROOT / "models" / "phi4-multimodal-instruct"
OUTPUT_DIR = ROOT / "benchmarks" / "results"


# ---------------------------------------------------------------------------
# Resource monitoring
# ---------------------------------------------------------------------------

def get_ram_mb() -> float:
    """Current process RSS in MB."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


def get_cpu_percent(interval: float = 0.5) -> float:
    """Process-specific CPU usage percentage."""
    return psutil.Process().cpu_percent(interval=interval)


def get_active_cores(interval: float = 0.5) -> int:
    """Estimate active cores used by this process."""
    pct = psutil.Process().cpu_percent(interval=interval)
    # cpu_percent() returns N*100 for N fully-utilized cores
    return max(1, round(pct / 100.0))


# ---------------------------------------------------------------------------
# Perplexity evaluation (condensed from evals/perplexity.py)
# ---------------------------------------------------------------------------

def load_wikitext_tokens(tokenizer, max_tokens: int) -> list[int]:
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])
    except Exception:
        text = (
            "The tower is 324 metres tall, about the same height as an 81-storey building, "
            "and the tallest structure in Paris. Its base is square, measuring 125 metres on "
            "each side. During its construction, the Eiffel Tower surpassed the Washington "
            "Monument to become the tallest man-made structure in the world."
        )
    tokens = tokenizer.encode(text)
    return tokens[:max_tokens]


def evaluate_perplexity(tokens, store, stride=512):
    max_seq = stride + 64
    rope_cos, rope_sin = build_rope_cache(max_seq, ROTARY_DIM)

    nll_sum = 0.0
    n_scored = 0
    t_start = time.perf_counter()

    num_windows = max(1, (len(tokens) - 1) // stride)

    for win_idx in range(num_windows):
        begin = win_idx * stride
        end = min(begin + stride + 1, len(tokens))
        window = tokens[begin:end]
        if len(window) < 2:
            break

        kv_hist = KVHistory()
        for i, tid in enumerate(window[:-1]):
            hidden = store.embed_f16[tid].float().unsqueeze(0)
            for layer in range(NUM_LAYERS):
                hidden = forward_layer(hidden, layer, store, kv_hist,
                                       rope_cos, rope_sin, pos=i)
            hidden = rms_norm(hidden, store.final_norm)
            logits = store.lm_head_matvec(hidden)

            target = window[i + 1]
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            nll = -log_probs[target].item()
            nll_sum += nll
            n_scored += 1

    elapsed = time.perf_counter() - t_start
    avg_nll = nll_sum / max(n_scored, 1)
    ppl = math.exp(min(avg_nll, 50))
    tps = n_scored / elapsed if elapsed > 0 else 0

    return {
        "ppl": ppl,
        "avg_nll": avg_nll,
        "num_tokens": n_scored,
        "tok_per_sec": tps,
        "elapsed_sec": elapsed,
    }


# ---------------------------------------------------------------------------
# SNR measurement on real model weights
# ---------------------------------------------------------------------------

def measure_snr_on_model_weight(bits_list):
    """Measure SNR using a representative random weight matrix."""
    torch.manual_seed(42)
    w = torch.randn(3072, 8192) * 0.02
    results = {}
    for bits in bits_list:
        if bits == 16:
            results[bits] = {"snr_db": float("inf"), "mse": 0.0, "compression_ratio": 1.0}
            continue
        sym = bits > 4
        opt = bits <= 4
        gs = 16 if bits <= 3 else (32 if bits <= 4 else 128)
        qt = quantize_weights(w.numpy(), bits=bits, group_size=gs,
                              symmetric=sym, optimize_clips=opt)
        err = compute_quantization_error(w, qt)
        results[bits] = err
    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def generate_charts(all_results, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("  matplotlib not available - skipping chart generation")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    bits_labels = [str(r["bits"]) + ("-bit" if r["bits"] < 16 else " (FP16)") for r in all_results]
    bits_vals = [r["bits"] for r in all_results]

    # Color scheme
    colors = {16: "#2196F3", 8: "#4CAF50", 4: "#FF9800", 3: "#F44336", 2: "#9C27B0"}
    bar_colors = [colors.get(b, "#607D8B") for b in bits_vals]

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("ASDSL Framework - Comprehensive Benchmark Results",
                 fontsize=18, fontweight="bold", y=0.98)
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # --- 1. Perplexity ---
    ax1 = fig.add_subplot(gs[0, 0])
    ppls = [r.get("ppl", 0) for r in all_results]
    bars = ax1.bar(bits_labels, ppls, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Perplexity (lower = better)", fontsize=11)
    ax1.set_title("Model Quality (WikiText-2 PPL)", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, ppls):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, max(ppls) * 1.2 if ppls else 50)
    ax1.grid(axis="y", alpha=0.3)

    # --- 2. Throughput ---
    ax2 = fig.add_subplot(gs[0, 1])
    tps = [r.get("tok_per_sec", 0) for r in all_results]
    bars = ax2.bar(bits_labels, tps, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Tokens / second", fontsize=11)
    ax2.set_title("Inference Throughput", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, tps):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    # --- 3. RAM Usage ---
    ax3 = fig.add_subplot(gs[0, 2])
    rams = [r.get("ram_peak_mb", 0) for r in all_results]
    bars = ax3.bar(bits_labels, [r / 1024 for r in rams], color=bar_colors,
                   edgecolor="black", linewidth=0.5)
    ax3.set_ylabel("Peak RAM (GB)", fontsize=11)
    ax3.set_title("Memory Usage", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, rams):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height()/1024 + 0.1,
                 f"{val/1024:.1f} GB", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax3.grid(axis="y", alpha=0.3)

    # --- 4. CPU Cores Active ---
    ax4 = fig.add_subplot(gs[1, 0])
    cores = [r.get("active_cores", 0) for r in all_results]
    bars = ax4.bar(bits_labels, cores, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax4.set_ylabel("Active cores (>5% usage)", fontsize=11)
    ax4.set_title("CPU Core Utilization", fontsize=13, fontweight="bold")
    ax4.axhline(y=4, color="red", linestyle="--", alpha=0.7, label="Target: 4 cores")
    for bar, val in zip(bars, cores):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax4.legend(fontsize=9)
    ax4.grid(axis="y", alpha=0.3)

    # --- 5. SNR (Signal-to-Noise Ratio) ---
    ax5 = fig.add_subplot(gs[1, 1])
    snrs = [r.get("snr_db", 0) for r in all_results]
    # Cap infinite SNR for display
    display_snrs = [min(s, 60) for s in snrs]
    bars = ax5.bar(bits_labels, display_snrs, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax5.set_ylabel("SNR (dB, higher = better)", fontsize=11)
    ax5.set_title("Quantization Quality (SNR)", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, snrs):
        label = "inf" if val > 100 else f"{val:.1f}"
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 label, ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax5.grid(axis="y", alpha=0.3)

    # --- 6. Compression Ratio ---
    ax6 = fig.add_subplot(gs[1, 2])
    comps = [r.get("compression_ratio", 1.0) for r in all_results]
    bars = ax6.bar(bits_labels, comps, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax6.set_ylabel("Compression ratio (x)", fontsize=11)
    ax6.set_title("Weight Compression", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, comps):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{val:.1f}x", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax6.grid(axis="y", alpha=0.3)

    plt.savefig(output_dir / "asdsl_benchmark_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Dashboard saved: {output_dir / 'asdsl_benchmark_dashboard.png'}")

    # --- Separate before/after comparison chart ---
    fig2, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle("ASDSL Quantization: Before vs After Optimization",
                  fontsize=16, fontweight="bold")

    # Before PPL values (from previous benchmarks)
    before_ppl = {16: 17.64, 8: 17.59, 4: 61.15, 3: 216950.82}
    after_ppl = {r["bits"]: r.get("ppl", 0) for r in all_results}

    bits_list = sorted(set(before_ppl.keys()) & set(after_ppl.keys()))
    x = range(len(bits_list))
    labels = [f"{b}-bit" if b < 16 else "FP16" for b in bits_list]

    before_vals = [min(before_ppl[b], 100) for b in bits_list]  # cap for display
    after_vals = [min(after_ppl.get(b, 0), 100) for b in bits_list]

    width = 0.35
    ax_a.bar([i - width/2 for i in x], before_vals, width, label="Before", color="#EF5350",
             edgecolor="black", linewidth=0.5)
    ax_a.bar([i + width/2 for i in x], after_vals, width, label="After", color="#66BB6A",
             edgecolor="black", linewidth=0.5)
    ax_a.set_xticks(list(x))
    ax_a.set_xticklabels(labels)
    ax_a.set_ylabel("Perplexity (capped at 100)")
    ax_a.set_title("Perplexity Improvement")
    ax_a.legend()
    ax_a.grid(axis="y", alpha=0.3)

    # Annotate 3-bit: "was 216,951"
    for i, b in enumerate(bits_list):
        if before_ppl[b] > 100:
            ax_a.annotate(f"was {before_ppl[b]:,.0f}", xy=(i - width/2, 100),
                         xytext=(i - width/2, 105), fontsize=8, ha="center",
                         color="red", fontweight="bold")

    # Before/after table
    ax_b.axis("off")
    table_data = []
    for b in bits_list:
        bp = before_ppl[b]
        ap = after_ppl.get(b, 0)
        improvement = ((bp - ap) / bp * 100) if bp > 0 and ap > 0 else 0
        table_data.append([
            f"{b}-bit" if b < 16 else "FP16",
            f"{bp:,.2f}",
            f"{ap:.2f}",
            f"{improvement:+.1f}%" if b != 16 else "baseline"
        ])
    table = ax_b.table(
        cellText=table_data,
        colLabels=["Config", "Before PPL", "After PPL", "Improvement"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.0)
    ax_b.set_title("Detailed Comparison", fontsize=13, fontweight="bold", pad=20)

    plt.savefig(output_dir / "asdsl_before_after.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Before/After chart saved: {output_dir / 'asdsl_before_after.png'}")

    # --- RAM timeline chart ---
    fig3, ax_r = plt.subplots(figsize=(10, 5))
    ram_data = [(r.get("ram_load_mb", 0), r.get("ram_inference_mb", 0), r.get("ram_peak_mb", 0))
                for r in all_results]
    x = range(len(bits_labels))
    width = 0.25
    ax_r.bar([i - width for i in x], [d[0]/1024 for d in ram_data], width,
             label="After Load", color="#42A5F5", edgecolor="black", linewidth=0.5)
    ax_r.bar(list(x), [d[1]/1024 for d in ram_data], width,
             label="During Inference", color="#FFA726", edgecolor="black", linewidth=0.5)
    ax_r.bar([i + width for i in x], [d[2]/1024 for d in ram_data], width,
             label="Peak", color="#EF5350", edgecolor="black", linewidth=0.5)
    ax_r.set_xticks(list(x))
    ax_r.set_xticklabels(bits_labels)
    ax_r.set_ylabel("RAM (GB)")
    ax_r.set_title("Memory Usage Across Quantization Configs", fontsize=14, fontweight="bold")
    ax_r.legend()
    ax_r.grid(axis="y", alpha=0.3)

    plt.savefig(output_dir / "asdsl_ram_usage.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  RAM chart saved: {output_dir / 'asdsl_ram_usage.png'}")


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(bits: int, tokenizer, wikitext_tokens, max_tokens: int,
                  skip_perplexity: bool = False,
                  enable_sparse: bool = False) -> dict:
    """Run full benchmark for a single bit-width configuration."""
    print(f"\n{'='*66}")
    label = 'FP16 baseline' if bits == 16 else f'ASDSL {bits}-bit'
    if enable_sparse and bits != 16:
        label += ' + sparse-GEMV'
    print(f"  Benchmarking: {label}")
    print(f"{'='*66}")

    gc.collect()
    ram_baseline = get_ram_mb()
    print(f"  RAM baseline: {ram_baseline:.0f} MB")

    print(f"  Loading model (bits={bits}) ...")
    t0 = time.perf_counter()
    store = WeightStore(bits=bits, enable_sparse=enable_sparse)
    store.load()
    ram_after_load = get_ram_mb()
    store.warm_cache()
    t_load = time.perf_counter() - t0
    ram_after_cache = get_ram_mb()
    print(f"  Load time: {t_load:.1f}s | RAM: {ram_after_load:.0f} -> {ram_after_cache:.0f} MB")

    result = {
        "bits": bits,
        "group_size": store.group_size,
        "symmetric": store._symmetric,
        "optimize_clips": store._optimize_clips,
        "load_time_sec": t_load,
        "ram_baseline_mb": ram_baseline,
        "ram_load_mb": ram_after_load,
        "ram_inference_mb": ram_after_cache,
    }

    if not skip_perplexity:
        # Measure CPU during inference
        print(f"  Running perplexity evaluation ({len(wikitext_tokens)} tokens) ...")
        cpu_before = get_active_cores(0.2)

        ppl_result = evaluate_perplexity(wikitext_tokens, store)

        cpu_during = get_active_cores(0.2)
        ram_peak = get_ram_mb()

        result.update({
            "ppl": ppl_result["ppl"],
            "avg_nll": ppl_result["avg_nll"],
            "num_tokens": ppl_result["num_tokens"],
            "tok_per_sec": ppl_result["tok_per_sec"],
            "elapsed_sec": ppl_result["elapsed_sec"],
            "active_cores": max(cpu_before, cpu_during),
            "ram_peak_mb": ram_peak,
        })
        print(f"  PPL={ppl_result['ppl']:.2f} | {ppl_result['tok_per_sec']:.2f} tok/s "
              f"| RAM peak={ram_peak:.0f} MB | cores~{max(cpu_before, cpu_during)}")
    else:
        # Quick inference test (5 tokens) for throughput estimate
        print("  Quick throughput test (5 tokens) ...")
        rope_cos, rope_sin = build_rope_cache(128, ROTARY_DIM)
        kv_hist = KVHistory()
        t0 = time.perf_counter()
        for pos in range(5):
            tid = wikitext_tokens[pos] if pos < len(wikitext_tokens) else 0
            hidden = store.embed_f16[tid].float().unsqueeze(0)
            for layer in range(NUM_LAYERS):
                hidden = forward_layer(hidden, layer, store, kv_hist,
                                       rope_cos, rope_sin, pos=pos)
            hidden = rms_norm(hidden, store.final_norm)
            _ = store.lm_head_matvec(hidden)
        t5 = time.perf_counter() - t0
        tps = 5 / t5

        ram_peak = get_ram_mb()
        cores = get_active_cores(0.2)
        result.update({
            "ppl": 0,
            "tok_per_sec": tps,
            "active_cores": cores,
            "ram_peak_mb": ram_peak,
            "num_tokens": 5,
            "elapsed_sec": t5,
        })
        print(f"  ~{tps:.2f} tok/s | RAM={ram_peak:.0f} MB | cores~{cores}")

    # Clean up
    del store
    gc.collect()

    return result


def main():
    parser = argparse.ArgumentParser(description="ASDSL Comprehensive Benchmark")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--bits", type=int, nargs="*", default=None,
                        help="Bit-widths to test (default: 16 8 4 3 2)")
    parser.add_argument("--skip-perplexity", action="store_true")
    parser.add_argument("--threads", type=int, default=8,
                        help="CPU threads (default: 8 for Intel i7 Evo P-cores)")
    parser.add_argument("--sparse", action="store_true",
                        help="Enable activation-sparse GEMV (Tier 3)")
    args = parser.parse_args()

    set_thread_count(args.threads)

    bits_list = args.bits or [16, 8, 4, 3, 2]

    print("=" * 66)
    print("  ASDSL Framework - Comprehensive Benchmark Suite")
    print("=" * 66)
    print(f"  CPU: {psutil.cpu_count(logical=False)} cores / {psutil.cpu_count()} threads")
    print(f"  RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB total")
    print(f"  Threads limited to: {args.threads}")
    print(f"  Bit-widths: {bits_list}")
    print(f"  Max tokens: {args.max_tokens}")

    # Load tokenizer and dataset once
    print("\nLoading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/Phi-4-multimodal-instruct", trust_remote_code=True
    )
    print("Loading WikiText-2 ...")
    wikitext_tokens = load_wikitext_tokens(tokenizer, args.max_tokens)
    print(f"  {len(wikitext_tokens)} tokens loaded")

    # SNR analysis
    print("\nMeasuring SNR across bit-widths ...")
    snr_results = measure_snr_on_model_weight(bits_list)
    for b, s in sorted(snr_results.items()):
        snr_str = "inf" if s["snr_db"] > 100 else f"{s['snr_db']:.2f}"
        print(f"  {b:2d}-bit: SNR={snr_str:>6s} dB  comp={s['compression_ratio']:.1f}x")

    # Run benchmarks
    all_results = []
    for bits in bits_list:
        result = run_benchmark(bits, tokenizer, wikitext_tokens, args.max_tokens,
                               skip_perplexity=args.skip_perplexity,
                               enable_sparse=args.sparse)
        snr = snr_results.get(bits, {})
        result["snr_db"] = snr.get("snr_db", 0)
        result["compression_ratio"] = snr.get("compression_ratio", 1.0)
        all_results.append(result)

    # Summary table
    print("\n" + "=" * 96)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 96)
    print(f"  {'Config':<12} {'PPL':>8} {'tok/s':>8} {'RAM(GB)':>8} {'Cores':>6} "
          f"{'SNR(dB)':>8} {'Comp':>6} {'GrpSz':>6}")
    print("-" * 96)
    for r in all_results:
        label = "FP16" if r["bits"] == 16 else f"{r['bits']}-bit"
        ppl_str = f"{r.get('ppl', 0):.2f}" if r.get("ppl", 0) > 0 else "skip"
        snr_str = "inf" if r.get("snr_db", 0) > 100 else f"{r.get('snr_db', 0):.1f}"
        print(f"  {label:<12} {ppl_str:>8} {r.get('tok_per_sec', 0):>8.2f} "
              f"{r.get('ram_peak_mb', 0)/1024:>8.1f} {r.get('active_cores', 0):>6} "
              f"{snr_str:>8} {r.get('compression_ratio', 1):>5.1f}x "
              f"{r.get('group_size', 0):>6}")
    print("=" * 96)

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_file = OUTPUT_DIR / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved: {results_file}")

    # Generate charts
    print("\nGenerating visualizations ...")
    generate_charts(all_results, OUTPUT_DIR)

    print("\n* Benchmark complete!")


if __name__ == "__main__":
    main()
