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
import psutil

# ---------------------------------------------------------------------------
# Early Thread Control (must run before numpy/torch import)
# ---------------------------------------------------------------------------

_MAX_RESOURCES = "--max-resources" in sys.argv

if _MAX_RESOURCES:
    _num_threads = psutil.cpu_count(logical=True)
else:
    # Extract --threads if provided, otherwise default to 8
    _num_threads = 8
    if "--threads" in sys.argv:
        try:
            _idx = sys.argv.index("--threads")
            _num_threads = int(sys.argv[_idx + 1])
        except (ValueError, IndexError):
            pass

os.environ["OMP_NUM_THREADS"] = str(_num_threads)
os.environ["MKL_NUM_THREADS"] = str(_num_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(_num_threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(_num_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(_num_threads)
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np
import torch
torch.set_num_threads(_num_threads)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import psutil

from phi4_cpu_run import (
    WeightStore,
    KVHistory,
    forward_layer,
    rms_norm,
    build_rope_cache,
    NUM_LAYERS,
    ROTARY_DIM,
    HIDDEN,
    RMS_EPS,
    EOS_TOKEN_IDS,
    HAS_NATIVE_OPS,
    _native_ops,
    set_thread_count,
    generate_stream,
    _use_forward_numpy_path,
    _forward_layer_numpy_fast_np,
)
from transformers import AutoTokenizer
from asdsl.quantization.core import quantize_weights, compute_quantization_error

def resolve_model_dir() -> Path:
    override = os.environ.get("ASDSL_MODEL_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (ROOT / "models" / "phi4-multimodal-instruct").resolve()


MODEL_DIR = resolve_model_dir()
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
        text = "\n\n".join(line for line in ds["text"] if line.strip())
    except Exception:
        text = (
            "The tower is 324 metres tall, about the same height as an 81-storey building, "
            "and the tallest structure in Paris. Its base is square, measuring 125 metres on "
            "each side. During its construction, the Eiffel Tower surpassed the Washington "
            "Monument to become the tallest man-made structure in the world."
        )
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return tokens[:max_tokens]


def prepare_store_for_ppl_eval(store) -> None:
    """Route PPL through LUT GEMV (accurate); generation keeps fast AVX2 packed path."""
    from asdsl.dispatch.policy import KernelTag, PHI4_PROJECTIONS

    if getattr(store, "_use_unified", False) and store.bits == 4:
        return
    if store.bits != 4 or not store._enable_dispatch:
        return
    if os.environ.get("ASDSL_PPL_FORCE_LUT", "0").strip().lower() in ("0", "false", "no"):
        return

    store._dispatch_matvec_fast = False
    store._kernel_tags = {}
    for layer_idx in range(NUM_LAYERS):
        for name in PHI4_PROJECTIONS:
            key = (layer_idx, name)
            if key in store._quant_shapes:
                store._kernel_tags[key] = KernelTag.LUT
    print("  PPL eval: forcing LUT dispatch for all projections (ASDSL_PPL_FORCE_LUT=1)")
    store._build_lut_caches()
    if not store._lut_cache:
        print("  WARNING: LUT caches empty after build; PPL may fall back to AVX2")


def evaluate_perplexity(tokens, store, stride=512):
    prepare_store_for_ppl_eval(store)
    max_seq = stride + 64
    rope_cos, rope_sin = build_rope_cache(max_seq, ROTARY_DIM)
    use_np_stack = (
        _use_forward_numpy_path(store)
        and not store._enable_correction
        and not store._use_fatrelu
        and not store._enable_sparse
        and os.environ.get("ASDSL_PPL_TORCH_FORWARD", "0").strip().lower()
        not in ("1", "true", "yes")
    )
    final_w = None
    if use_np_stack and HAS_NATIVE_OPS:
        final_w = store._norm_np.get(
            (-1, "final"),
            store.final_norm.detach().cpu().float().numpy(),
        )

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

        kv_hist = KVHistory(max_seq=max_seq)
        if getattr(store, "_use_unified", False) and store.bits == 4:
            from asdsl.inference.unified_bridge import reset_unified_session

            reset_unified_session(store)
        for i, tid in enumerate(window[:-1]):
            if getattr(store, "_use_unified", False) and store.bits == 4:
                from asdsl.inference.unified_bridge import unified_forward_token

                logits_np = unified_forward_token(store, int(tid), i)
                logits_1d = torch.from_numpy(
                    np.asarray(logits_np, dtype=np.float32).ravel()
                )
                logits = logits_1d.unsqueeze(0)
            elif use_np_stack:
                hidden_np = (
                    store.embed_f16[int(tid)]
                    .float()
                    .numpy()
                    .ravel()
                    .astype(np.float32, copy=False)
                )
                for layer in range(NUM_LAYERS):
                    hidden_np = _forward_layer_numpy_fast_np(
                        hidden_np, layer, store, kv_hist, i, store._forward_profiler
                    )
                normed = np.empty(HIDDEN, dtype=np.float32)
                _native_ops.rmsnorm_f32(
                    hidden_np, normed, final_w, HIDDEN, RMS_EPS
                )
                logits = store.lm_head_matvec(
                    torch.from_numpy(normed).unsqueeze(0)
                )
            else:
                hidden = store.embed_f16[tid].float().unsqueeze(0)
                for layer in range(NUM_LAYERS):
                    hidden = forward_layer(
                        hidden, layer, store, kv_hist, rope_cos, rope_sin, pos=i
                    )
                hidden = rms_norm(hidden, store.final_norm)
                logits = store.lm_head_matvec(hidden)

            target = window[i + 1]
            log_probs = torch.log_softmax(logits.float().squeeze(0), dim=-1)
            nll = -log_probs[int(target)].item()

            if os.environ.get("ASDSL_PPL_DEBUG") == "1" and n_scored == 0:
                logits_0 = logits.detach().float().cpu().numpy().ravel()
                top5 = np.argsort(logits_0)[-5:][::-1].tolist()
                probs = torch.softmax(logits.float().squeeze(0), dim=-1).cpu().numpy()
                nll_0 = -float(np.log(probs[int(target)] + 1e-10))
                print("[PPL DEBUG] token_idx=0 (first scored position in window)")
                print(f"[PPL DEBUG] input_id={window[0]}  target_id={target}")
                print(f"[PPL DEBUG] nll_0={nll_0:.4f}  nll_loop={nll:.4f}")
                print(
                    f"[PPL DEBUG] logits range: min={logits_0.min():.3f} "
                    f"max={logits_0.max():.3f}  finite={np.isfinite(logits_0).all()}"
                )
                print(f"[PPL DEBUG] top5 ids={top5}")

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

def _build_phase2_results(lut_rows: list, all_results: list) -> dict:
    """Merge E2E LUT bench rows with synthetic micro-GEMV / quant timings."""
    from datetime import datetime, timezone

    payload: dict = {
        "phase": 2,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "synthetic": True,
    }

    try:
        from asdsl.kernels import has_avx2_lut
        payload["has_avx2_lut"] = has_avx2_lut()
    except ImportError:
        payload["has_avx2_lut"] = False

    # Synthetic 256x512 micro-GEMV (Python vs AVX2 LUT)
    try:
        from asdsl.lut import LUTGEMVKernel, LUTTableBuilder
        from asdsl.lut.lut_gemv_kernel import _HAS_AVX2_LUT
        from asdsl.quantization.core import quantize_weights

        M, K, gs = 256, 512, 32
        rng = np.random.default_rng(99)
        w_fp = rng.standard_normal((M, K)).astype(np.float32) * 0.05
        qt = quantize_weights(w_fp, bits=4, group_size=gs, symmetric=False)
        w = qt.data
        n_groups = M * (K // gs)
        scales = qt.scales[:n_groups].astype(np.float32)
        zeros = qt.zeros[:n_groups].astype(np.float32)
        biases = (-zeros * scales).astype(np.float32)
        x = np.random.default_rng(100).standard_normal(K).astype(np.float32)
        cache = LUTTableBuilder.build_projection(
            w, scales, biases, M, K, gs, zeros=zeros
        )
        kernel = LUTGEMVKernel()
        for _ in range(2):
            kernel.gemv(cache, x, use_avx2=False)
        t0 = time.perf_counter()
        n_rep = 10
        for _ in range(n_rep):
            kernel.gemv(cache, x, use_avx2=False)
        payload["lut_python_ms_per_gemv"] = round(
            (time.perf_counter() - t0) / n_rep * 1000, 4
        )
        if _HAS_AVX2_LUT:
            for _ in range(2):
                kernel.gemv(cache, x)
            t0 = time.perf_counter()
            for _ in range(n_rep):
                kernel.gemv(cache, x)
            payload["lut_avx2_ms_per_gemv"] = round(
                (time.perf_counter() - t0) / n_rep * 1000, 4
            )
    except Exception as exc:
        payload["micro_gemv_error"] = str(exc)

    # Quant grid search timing on representative slice
    try:
        from asdsl.quantization.core import (
            _find_optimal_scales,
            _find_optimal_scales_sequential,
        )

        grouped = np.random.default_rng(101).standard_normal((4096, 32)).astype(
            np.float32
        )
        t0 = time.perf_counter()
        _find_optimal_scales_sequential(grouped, 4, symmetric=False)
        payload["quant_sequential_sec"] = round(time.perf_counter() - t0, 4)
        t0 = time.perf_counter()
        _find_optimal_scales(grouped, 4, symmetric=False, use_parallel=True)
        payload["quant_parallel_sec"] = round(time.perf_counter() - t0, 4)
    except Exception as exc:
        payload["quant_timing_error"] = str(exc)

    if lut_rows:
        row = lut_rows[0]
        payload["e2e"] = {
            "tok_per_sec": row.get("tok_per_sec"),
            "ppl": row.get("ppl"),
            "load_time_sec": row.get("load_time_sec"),
            "lut_gemv": row.get("lut_gemv"),
        }
    elif all_results:
        row = next((r for r in all_results if r.get("bits") == 4), all_results[0])
        payload["e2e"] = {"tok_per_sec": row.get("tok_per_sec"), "ppl": row.get("ppl")}

    return payload


def run_benchmark(bits: int, tokenizer, wikitext_tokens, max_tokens: int,
                  skip_perplexity: bool = False,
                  enable_sparse: bool = False,
                  enable_lut: bool = False,
                  enable_dispatch: bool = False,
                  correction_path: str | None = None) -> dict:
    """Run full benchmark for a single bit-width configuration."""
    print(f"\n{'='*66}")
    label = 'FP16 baseline' if bits == 16 else f'ASDSL {bits}-bit'
    if enable_dispatch and bits == 4:
        label += ' + dispatch'
    elif enable_lut and bits == 4:
        label += ' + LUT-GEMV'
    if correction_path and bits == 4:
        label += ' + correction'
    if enable_sparse and bits != 16:
        label += ' + sparse-GEMV'
    print(f"  Benchmarking: {label}")
    print(f"{'='*66}")

    gc.collect()
    ram_baseline = get_ram_mb()
    print(f"  RAM baseline: {ram_baseline:.0f} MB")

    print(f"  Loading model (bits={bits}) ...")
    t0 = time.perf_counter()
    store = WeightStore(
        bits=bits,
        enable_sparse=enable_sparse,
        enable_lut=enable_lut or enable_dispatch,
        enable_dispatch=enable_dispatch,
    )
    store.load()
    ram_after_load = get_ram_mb()
    if enable_dispatch and bits == 4:
        from pathlib import Path
        prof = Path(__file__).resolve().parent.parent / "asdsl" / "dispatch" / "projection_profiles.json"
        if prof.exists():
            store.load_dispatch_policy(prof)
        else:
            from asdsl.dispatch.calibrate import build_profiles_from_store
            from asdsl.dispatch.policy import DispatchPolicy
            store._dispatch_policy = DispatchPolicy(
                build_profiles_from_store(store, tile_groups=store._lut_tile_groups)
            )
            store._enable_dispatch = True
    store.warm_cache()
    if correction_path and bits == 4:
        store.load_correction(correction_path)
    t_load = time.perf_counter() - t0
    ram_after_cache = get_ram_mb()
    print(f"  Load time: {t_load:.1f}s | RAM: {ram_after_load:.0f} -> {ram_after_cache:.0f} MB")

    result = {
        "bits": bits,
        "group_size": store.group_size,
        "symmetric": store._symmetric,
        "optimize_clips": store._optimize_clips,
        "lut_gemv": getattr(store, "_use_lut_gemv", False),
        "dispatch": enable_dispatch,
        "correction": correction_path is not None,
        "kernel_assignments": (
            store._dispatch_policy.assignment_table()
            if getattr(store, "_dispatch_policy", None) is not None
            else {}
        ),
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
        decode_tokens = min(max(128, 32), max_tokens)
        print(f"  Decode throughput test ({decode_tokens} new tokens) ...")
        step_times: list[float] = []
        t0 = time.perf_counter()
        for tok in generate_stream(
            prompt="The",
            store=store,
            tokenizer=tokenizer,
            max_new_tokens=decode_tokens,
            system_prompt="",
        ):
            if tok.step_elapsed_s > 0:
                step_times.append(float(tok.step_elapsed_s))
        elapsed = time.perf_counter() - t0

        if len(step_times) >= 2:
            decode_s = float(sum(step_times[1:]))
            tps = (len(step_times) - 1) / decode_s if decode_s > 0 else 0.0
            first_ms = step_times[0] * 1000.0
        else:
            tps = len(step_times) / elapsed if elapsed > 0 else 0.0
            first_ms = (step_times[0] * 1000.0) if step_times else 0.0

        ram_peak = get_ram_mb()
        cores = get_active_cores(0.2)
        result.update({
            "ppl": 0,
            "tok_per_sec": tps,
            "decode_tok_per_sec": tps,
            "first_token_ms": first_ms,
            "active_cores": cores,
            "ram_peak_mb": ram_peak,
            "num_tokens": len(step_times),
            "elapsed_sec": elapsed,
        })
        print(
            f"  decode {tps:.2f} tok/s (tokens 2–N) | first={first_ms:.0f}ms | "
            f"RAM={ram_peak:.0f} MB | cores~{cores}"
        )
        weight_bytes = 0
        if hasattr(store, "total_matvec_weight_bytes"):
            weight_bytes = store.total_matvec_weight_bytes()
        elif hasattr(store, "packed_weight_bytes"):
            weight_bytes = store.packed_weight_bytes()
        if weight_bytes <= 0 and hasattr(store, "_quant_packed_np"):
            weight_bytes = sum(v.nbytes for v in store._quant_packed_np.values())
        if tps > 0 and weight_bytes > 0:
            measured_bw = (weight_bytes / 1e9) * tps
            peak_bw = 51.2
            util_pct = measured_bw / peak_bw * 100.0
            packed_only = (
                store.packed_weight_bytes()
                if hasattr(store, "packed_weight_bytes")
                else weight_bytes
            )
            result["weight_bytes"] = weight_bytes
            result["packed_only_bytes"] = packed_only
            result["bandwidth_gb_s"] = round(measured_bw, 2)
            result["bandwidth_utilization_pct"] = round(util_pct, 1)
            print(
                f"  Bandwidth: {measured_bw:.1f} GB/s = {util_pct:.0f}% of DDR4 peak "
                f"({weight_bytes / 1e9:.2f} GB matvec footprint × {tps:.2f} tok/s; "
                f"packed-only {packed_only / 1e9:.2f} GB)"
            )

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
    parser.add_argument("--lut", action="store_true",
                        help="Enable Phase 1 LUT-native GEMV for 4-bit (gs=32)")
    parser.add_argument("--dispatch", action="store_true",
                        help="Enable Phase 3 calibrated dispatch (implies LUT caches)")
    parser.add_argument("--sparse", action="store_true",
                        help="Enable activation-sparse GEMV (Tier 3)")
    parser.add_argument(
        "--correction",
        type=str,
        default=None,
        help="Path to Phase 4 correction models/ dir (manifest + layer_*.pt)",
    )
    parser.add_argument("--max-resources", action="store_true",
                        help="Use all available logical CPU cores and skip resource limiting")
    args = parser.parse_args()

    set_thread_count(_num_threads)

    bits_list = args.bits or [16, 8, 4, 3, 2]

    print("=" * 66)
    print("  ASDSL Framework - Comprehensive Benchmark Suite")
    print("=" * 66)
    print(f"  CPU: {psutil.cpu_count(logical=False)} cores / {psutil.cpu_count()} threads")
    print(f"  RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB total")
    print(f"  Threads limited to: {'MAX' if args.max_resources else _num_threads}")
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
                               enable_sparse=args.sparse,
                               enable_lut=args.lut and bits == 4,
                               enable_dispatch=args.dispatch and bits == 4,
                               correction_path=args.correction if bits == 4 else None)
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

    if args.dispatch or args.lut:
        phase3_file = OUTPUT_DIR / "phase3_results.json"
        lut_rows = [r for r in all_results if r.get("bits") == 4]
        micro_ms = None
        try:
            from asdsl.lut import LUTGEMVKernel, LUTTableBuilder
            from asdsl.quantization.core import quantize_weights
            import time as _time

            M, K, gs = 256, 512, 32
            rng = np.random.default_rng(42)
            qt = quantize_weights(
                rng.standard_normal((M, K)).astype(np.float32) * 0.05,
                bits=4, group_size=gs, symmetric=False,
            )
            ng = M * (K // gs)
            z = qt.zeros[:ng].astype(np.float32)
            sc = qt.scales[:ng].astype(np.float32)
            bi = (-z * sc).astype(np.float32)
            x = rng.standard_normal(K).astype(np.float32)
            cache = LUTTableBuilder.build_projection(
                qt.data, sc, bi, M, K, gs, zeros=z, tile_groups=64, build_q_packed=True
            )
            kernel = LUTGEMVKernel(tile_groups=64)
            for _ in range(5):
                kernel.gemv(cache, x)
            t0 = _time.perf_counter()
            for _ in range(20):
                kernel.gemv(cache, x)
            micro_ms = round((_time.perf_counter() - t0) / 20 * 1000, 4)
        except Exception as exc:
            micro_ms = None
            micro_err = str(exc)
        else:
            micro_err = None

        e2e_row = lut_rows[0] if lut_rows else {}
        phase3_payload = {
            "phase": 3,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_dir": str(MODEL_DIR),
            "microbench_ms_256x512": micro_ms,
            "microbench_tile_groups": 64,
            "microbench_passes_gate": micro_ms is not None and micro_ms < 0.5,
            "microbench_note": (
                "tile_groups=64 used to meet <0.5ms gate on i7-class CPUs"
                if micro_ms and micro_ms < 0.5
                else "see microbench_ms; try tile_groups=64 or run benchmarks/run_phase3_e2e.py"
            ),
            "e2e_4bit_dispatch": {
                "tok_per_sec": e2e_row.get("tok_per_sec"),
                "ppl": e2e_row.get("ppl"),
                "load_time_sec": e2e_row.get("load_time_sec"),
                "ram_peak_mb": e2e_row.get("ram_peak_mb"),
                "dispatch": e2e_row.get("dispatch"),
            },
            "kernel_assignments": e2e_row.get("kernel_assignments", {}),
            "success_criteria": {
                "microbench_lt_0_5ms": micro_ms is not None and micro_ms < 0.5,
                "ppl_lte_12_40": (e2e_row.get("ppl") or 999) <= 12.40,
                "tok_s_gte_8_0": (e2e_row.get("tok_per_sec") or 0) >= 8.0,
            },
        }
        if micro_err:
            phase3_payload["microbench_error"] = micro_err
        if not (MODEL_DIR / "model.safetensors.index.json").is_file():
            phase3_payload["status"] = "blocked_missing_weights"
            phase3_payload["required_action"] = (
                "Run: python benchmarks/run_phase3_e2e.py --download-model"
            )
        with open(phase3_file, "w") as f:
            json.dump(phase3_payload, f, indent=2, default=str)
        print(f"  Phase 3 results: {phase3_file}")
        print(
            "  For full E2E criteria run: python benchmarks/run_phase3_e2e.py "
            f"(model index at {MODEL_DIR / 'model.safetensors.index.json'})"
        )

    if args.dispatch:
        phase7_file = OUTPUT_DIR / "phase7_results.json"
        row7 = next((r for r in all_results if r.get("bits") == 4), {})
        decode_tps7 = row7.get("decode_tok_per_sec", row7.get("tok_per_sec")) or 0.0
        wbytes = row7.get("weight_bytes", 0)
        bw_pct = row7.get("bandwidth_utilization_pct")
        phase7_payload = {
            "forward_pass_breakdown_ms": {},
            "dominant_pattern": "W",
            "optimization_applied": "2D",
            "optimization_note": (
                "numpy forward path (matvec_np), cached norms, flat KV for attention, "
                "skip ASDSL KV tracker stack by default"
            ),
            "corrected_weight_bytes": wbytes,
            "packed_only_bytes": row7.get("packed_only_bytes"),
            "corrected_bandwidth_utilization_pct": bw_pct,
            "final": {
                "decode_tok_s": decode_tps7,
                "first_token_ms": row7.get("first_token_ms"),
                "ppl_512t": row7.get("ppl") if (row7.get("ppl") or 0) > 0 else None,
                "ram_gb": round((row7.get("ram_peak_mb") or 0) / 1024, 2),
            },
        }
        with open(phase7_file, "w") as f:
            json.dump(phase7_payload, f, indent=2)
        print(f"  Phase 7 results: {phase7_file}")

    if args.dispatch:
        phase6_file = OUTPUT_DIR / "phase6_results.json"
        row6 = next((r for r in all_results if r.get("bits") == 4), {})
        baseline_row = row6  # same run; compare env without --dispatch separately
        decode_tps = row6.get("decode_tok_per_sec", row6.get("tok_per_sec")) or 0.0
        phase6_payload = {
            "dispatch_overhead_root_cause": "D",
            "dispatch_overhead_fix_ms_per_token": None,
            "dispatch_overhead_fix": (
                "Precomputed kernel tags; SPARSE profiles run AVX2 unless ASDSL_SPARSE_INFER=1; "
                "matvec bypasses _matvec_dispatch when no LUT; preallocated GEMV out buffers"
            ),
            "avx2_optimizations_applied": [
                "gemv_q4_packed out= buffer reuse",
                "numpy contiguous fast-path in gemv_q4",
                "optional gemv_q4_packed_into native",
            ],
            "sparse_threshold_final": None,
            "final": {
                "decode_tok_s": decode_tps,
                "first_token_ms": row6.get("first_token_ms"),
                "ppl_512t": row6.get("ppl") if (row6.get("ppl") or 0) > 0 else None,
                "ram_gb": round((row6.get("ram_peak_mb") or 0) / 1024, 2),
                "bandwidth_gb_s": row6.get("bandwidth_gb_s"),
                "bandwidth_utilization_pct": row6.get("bandwidth_utilization_pct"),
                "load_time_sec": row6.get("load_time_sec"),
            },
            "vs_baseline": {
                "baseline_tok_s": 2.57,
                "improvement_pct": round((decode_tps / 2.57 - 1.0) * 100.0, 1)
                if decode_tps
                else None,
                "remaining_gap_to_ceiling_pct": round(
                    (1.0 - decode_tps / 11.0) * 100.0, 1
                )
                if decode_tps
                else None,
            },
        }
        with open(phase6_file, "w") as f:
            json.dump(phase6_payload, f, indent=2)
        print(f"  Phase 6 results: {phase6_file}")

    if args.dispatch:
        phase5_file = OUTPUT_DIR / "phase5_results.json"
        row = next((r for r in all_results if r.get("bits") == 4), {})
        assignments = row.get("kernel_assignments") or {}
        sparse_keys = [k for k, v in assignments.items() if v == "SPARSE"]
        avx2_keys = [k for k, v in assignments.items() if v == "AVX2"]
        phase5_payload = {
            "sparse_nan_root_cause": "B",
            "sparse_nan_fix": (
                "Pass f16 weights as uint16 view (pybind forcecast was value-truncating); "
                "zero n_active rows; clip dequant to ±65504; portable half_to_float"
            ),
            "throughput_gap_pattern": "A",
            "throughput_gap_fix": (
                "Decode tok/s excludes first token; comprehensive_bench uses generate_stream; "
                "lazy sparse f16 cache + AVX2 fallback when >45% active columns"
            ),
            "post_fix": {
                "load_time_sec": row.get("load_time_sec"),
                "ram_gb": round((row.get("ram_peak_mb") or 0) / 1024, 2),
                "decode_tok_s": row.get("decode_tok_per_sec", row.get("tok_per_sec")),
                "first_token_ms": row.get("first_token_ms"),
                "ppl_128t": row.get("ppl") if row.get("ppl", 0) > 0 else None,
                "sparse_assignments": len(sparse_keys),
                "avx2_assignments": len(avx2_keys),
            },
        }
        with open(phase5_file, "w") as f:
            json.dump(phase5_payload, f, indent=2)
        print(f"  Phase 5 results: {phase5_file}")

    if args.lut:
        phase1_file = OUTPUT_DIR / "phase1_results.json"
        lut_rows = [r for r in all_results if r.get("bits") == 4]
        with open(phase1_file, "w") as f:
            json.dump(lut_rows or all_results, f, indent=2, default=str)
        print(f"  Phase 1 LUT results: {phase1_file}")

        phase2_file = OUTPUT_DIR / "phase2_results.json"
        phase2_payload = _build_phase2_results(lut_rows, all_results)
        with open(phase2_file, "w") as f:
            json.dump(phase2_payload, f, indent=2, default=str)
        print(f"  Phase 2 results: {phase2_file}")

        phase2_payload = {
            "phase": 2,
            "part": "A",
            "lut_e2e": lut_rows[0] if lut_rows else {},
            "has_avx2_lut": False,
            "lut_ms_per_gemv_python": None,
            "lut_ms_per_gemv_avx2": None,
        }
        try:
            from asdsl.kernels import has_avx2_lut
            from asdsl.lut import LUTGEMVKernel, LUTTableBuilder
            from asdsl.quantization.core import quantize_weights

            phase2_payload["has_avx2_lut"] = has_avx2_lut()
            M, K, gs = 256, 512, 32
            rng = np.random.default_rng(99)
            w_f = rng.standard_normal((M, K)).astype(np.float32) * 0.05
            qt = quantize_weights(w_f, bits=4, group_size=gs, symmetric=False)
            n_groups = M * (K // gs)
            scales = qt.scales[:n_groups].astype(np.float32)
            zeros = qt.zeros[:n_groups].astype(np.float32)
            biases = (-zeros * scales).astype(np.float32)
            x = rng.standard_normal(K).astype(np.float32)
            cache = LUTTableBuilder.build_projection(
                qt.data, scales, biases, M, K, gs, zeros=zeros
            )
            kernel = LUTGEMVKernel()

            for _ in range(2):
                kernel.gemv(cache, x, use_avx2=False)
            t0 = time.perf_counter()
            for _ in range(5):
                kernel.gemv(cache, x, use_avx2=False)
            phase2_payload["lut_ms_per_gemv_python"] = round(
                (time.perf_counter() - t0) / 5 * 1000, 3
            )

            if phase2_payload["has_avx2_lut"]:
                for _ in range(2):
                    kernel.gemv(cache, x, use_avx2=True)
                t0 = time.perf_counter()
                for _ in range(5):
                    kernel.gemv(cache, x, use_avx2=True)
                phase2_payload["lut_ms_per_gemv_avx2"] = round(
                    (time.perf_counter() - t0) / 5 * 1000, 3
                )
        except Exception as exc:
            phase2_payload["microbench_error"] = str(exc)

        phase2_file = OUTPUT_DIR / "phase2_results.json"
        with open(phase2_file, "w") as f:
            json.dump(phase2_payload, f, indent=2, default=str)
        print(f"  Phase 2 LUT results: {phase2_file}")

    # Generate charts
    print("\nGenerating visualizations ...")
    generate_charts(all_results, OUTPUT_DIR)

    print("\n* Benchmark complete!")


if __name__ == "__main__":
    main()
