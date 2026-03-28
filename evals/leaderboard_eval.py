"""Leaderboard evaluation with scientific roofline profiling.

Runs Phase-8 native perplexity-style evaluation at long context, computes
roofline ceilings from measured STREAM Triad bandwidth and exact bytes/token,
and optionally executes lightweight 0-shot task evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import torch

    if hasattr(torch, "set_flush_denormal"):
        try:
            torch.set_flush_denormal(True)
        except Exception:
            pass
    _t = os.environ.get("ASDSL_TORCH_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS")
    _n = max(1, int(_t)) if _t is not None else min(8, max(1, os.cpu_count() or 4))
    torch.set_num_threads(_n)
    os.environ.setdefault("NUMEXPR_MAX_THREADS", str(_n))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(_n))
except ImportError:
    pass

from asdsl.inference.engine import evaluate_perplexity_phase8_native, resolve_hf_ppl_model_id
from asdsl.profiler import (
    ModelArchitecture,
    bytes_per_token_breakdown,
    estimate_memory_bandwidth_gbps,
    infer_architecture_from_q_metadata,
    load_architecture_from_config,
    roofline_ceiling_tps,
    roofline_curve,
)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evals"))

from perplexity import load_wikitext2
from transformers import AutoConfig, AutoTokenizer


def _avg_context_length_from_windows(total_tokens: int, stride: int) -> float:
    if total_tokens < 2:
        return 1.0

    contexts: list[int] = []
    windows = max(1, (total_tokens - 1) // stride)
    for win_idx in range(windows):
        begin = win_idx * stride
        end = min(begin + stride + 1, total_tokens)
        window_len = end - begin
        if window_len < 2:
            continue
        contexts.append(window_len - 1)

    if not contexts:
        return 1.0
    return float(sum(contexts) / len(contexts))


def _resolve_architecture(model_meta_path: Path, hf_model_id: str) -> tuple[ModelArchitecture, str]:
    local_config = model_meta_path.with_name("config.json")
    if local_config.exists():
        return load_architecture_from_config(local_config), f"local_config:{local_config}"

    # Prefer explicit model config from transformers for correct GQA/MQA fields.
    try:
        cfg = AutoConfig.from_pretrained(hf_model_id, trust_remote_code=True)
        return (
            ModelArchitecture(
                num_layers=int(cfg.num_hidden_layers),
                num_attention_heads=int(cfg.num_attention_heads),
                num_kv_heads=int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)),
                hidden_size=int(cfg.hidden_size),
                head_dim=int(cfg.hidden_size // max(cfg.num_attention_heads, 1)),
            ),
            f"huggingface_config:{hf_model_id}",
        )
    except Exception:
        pass

    return infer_architecture_from_q_metadata(model_meta_path), f"metadata_inference:{model_meta_path}"


def _run_zero_shot(task: str, bits: int, limit: int | None) -> tuple[float | None, str]:
    try:
        import lm_eval
        from evals.lm_eval_harness import ASDSLHarnessModel
    except Exception as exc:
        return None, f"lm_eval_unavailable:{exc}"

    try:
        model = ASDSLHarnessModel(bits=bits)
        results = lm_eval.simple_evaluate(
            model=model,
            tasks=[task],
            num_fewshot=0,
            limit=limit,
        )
        task_res = results.get("results", {}).get(task, {})
        for key in ("acc,none", "acc_norm,none", "acc"):
            if key in task_res:
                return float(task_res[key]), key
        for key, value in task_res.items():
            if key.startswith("acc") and isinstance(value, (int, float)):
                return float(value), key
        return None, "no_accuracy_metric"
    except Exception as exc:
        return None, f"zero_shot_error:{exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description="ASDSL leaderboard evaluation with roofline diagnostics")
    parser.add_argument("--bits", type=int, default=8, choices=[2, 3, 4, 8, 16])
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--stream-array-mb", type=int, default=256)
    parser.add_argument("--stream-runs", type=int, default=12)
    parser.add_argument("--kv-bytes-per-element", type=int, default=1, choices=[1, 2])
    parser.add_argument("--zero-shot-task", type=str, default="piqa")
    parser.add_argument("--zero-shot-limit", type=int, default=64)
    parser.add_argument(
        "--hf-model-id",
        type=str,
        default=None,
        help=(
            "HuggingFace model id for perplexity and WikiText tokenization "
            "(default: $ASDSL_PPL_MODEL_ID or microsoft/phi-4)."
        ),
    )
    parser.add_argument("--skip-zero-shot", action="store_true")
    parser.add_argument("--skip-perplexity", action="store_true")
    parser.add_argument("--curve-min-t", type=int, default=64)
    parser.add_argument("--curve-max-t", type=int, default=2048)
    parser.add_argument("--curve-step", type=int, default=64)
    args = parser.parse_args()

    print("=" * 78)
    print("ASDSL Leaderboard Evaluation - Phase 8 + Scientific Roofline")
    print("=" * 78)

    ppl_model_id = resolve_hf_ppl_model_id(args.hf_model_id)
    print("\n[1/4] Loading tokenizer + WikiText-2 ...")
    print(f"  HF model / tokenizer: {ppl_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(ppl_model_id, trust_remote_code=True)
    tokens = load_wikitext2(tokenizer, max_tokens=args.max_tokens)

    if args.skip_perplexity:
        print("\n[2/4] Skipping perplexity (--skip-perplexity) ...")
        from asdsl.inference.engine import _resolve_native_model_paths, NativePerplexityResult
        model_bin, model_meta = _resolve_native_model_paths(args.bits)
        ppl_result = NativePerplexityResult(
            bits=args.bits,
            ppl=0.0,
            avg_nll=0.0,
            num_tokens=len(tokens),
            tokens_per_second=0.0,
            elapsed_sec=0.0,
            windows=0,
            backend_model_bin=str(model_bin),
            backend_model_metadata=str(model_meta),
            ppl_route="skipped",
            hf_model_id=ppl_model_id,
        )
    else:
        print("\n[2/4] Running perplexity (HuggingFace causal LM, summed NLL over chunk) ...")
        ppl_result = evaluate_perplexity_phase8_native(
            tokens=tokens,
            bits=args.bits,
            stride=args.stride,
            hf_model_id=ppl_model_id,
        )
    
    # Explicitly clear HF resources to free RAM before loading native C++ weights
    del tokenizer
    import gc
    gc.collect()

    model_bin_path = Path(ppl_result.backend_model_bin)
    model_meta_path = Path(ppl_result.backend_model_metadata)

    print("\n[3/4] Measuring STREAM Triad bandwidth (int8) ...")
    hw = estimate_memory_bandwidth_gbps(
        sample_mb=args.stream_array_mb,
        repeats=args.stream_runs,
    )

    arch, arch_source = _resolve_architecture(model_meta_path, ppl_model_id)
    t_measured = int(round(_avg_context_length_from_windows(len(tokens), args.stride)))
    bpt = bytes_per_token_breakdown(
        model_file_size_bytes=model_bin_path.stat().st_size,
        num_layers=arch.num_layers,
        num_kv_heads=arch.num_kv_heads,
        head_dim=arch.head_dim,
        t_context=t_measured,
        kv_bytes_per_element=args.kv_bytes_per_element,
    )
    ceiling_tps = roofline_ceiling_tps(hw.memory_bandwidth_gbps, bpt.total_bytes)
    efficiency_pct = (ppl_result.tokens_per_second / max(ceiling_tps, 1e-12)) * 100.0

    curve_t = list(range(args.curve_min_t, args.curve_max_t + 1, args.curve_step))
    curve_points = roofline_curve(
        memory_bandwidth_gb_s=hw.memory_bandwidth_gbps,
        model_file_size_bytes=model_bin_path.stat().st_size,
        num_layers=arch.num_layers,
        num_kv_heads=arch.num_kv_heads,
        head_dim=arch.head_dim,
        kv_bytes_per_element=args.kv_bytes_per_element,
        t_values=curve_t,
    )

    zero_shot_acc = None
    zero_shot_metric = "skipped"
    if not args.skip_zero_shot:
        print("\n[4/4] Running lightweight 0-shot regression task ...")
        print(
            "Zero-shot settings: "
            f"task={args.zero_shot_task}, limit={args.zero_shot_limit}"
        )
        zero_shot_acc, zero_shot_metric = _run_zero_shot(
            task=args.zero_shot_task,
            bits=args.bits,
            limit=args.zero_shot_limit,
        )

    print("\n" + "=" * 78)
    print("Scientific profiler output")
    print("=" * 78)
    tri_dtype = hw.triad.dtype if hw.triad is not None else "?"
    print(f"STREAM Triad bandwidth ({tri_dtype}): {hw.memory_bandwidth_gbps:.2f} GB/s")
    if hw.triad is not None:
        print(
            "Triad details: "
            f"impl={hw.triad.implementation}, "
            f"runs={hw.triad.runs}, warmup={hw.triad.warmup_runs}, "
            f"array={hw.triad.array_bytes / (1024**2):.1f} MB, "
            f"elapsed={hw.triad.elapsed_sec:.4f}s"
        )
    print(f"Architecture source: {arch_source}")
    print(
        "Architecture: "
        f"layers={arch.num_layers}, kv_heads={arch.num_kv_heads}, head_dim={arch.head_dim}, "
        f"attn_heads={arch.num_attention_heads}"
    )
    print(f"Pinned context length t: {bpt.t_context}")
    print(f"Weight term: {bpt.weight_bytes / 1e9:.6f} GB")
    print(f"KV term: {bpt.kv_bytes / 1e9:.6f} GB")
    print(f"Bytes/token(t): {bpt.total_gb:.6f} GB")
    print(f"Roofline ceiling(t): {ceiling_tps:.4f} tok/s")
    print(f"Observed throughput: {ppl_result.tokens_per_second:.4f} tok/s")
    print(f"Efficiency: {efficiency_pct:.2f}%")

    print("\nPerplexity run")
    print(f"PPL ({ppl_result.ppl_route}): {ppl_result.ppl:.3f}")
    print(f"HF model: {ppl_result.hf_model_id}")
    print(f"Avg NLL: {ppl_result.avg_nll:.5f}")
    print(f"Windows: {ppl_result.windows}, scored tokens: {ppl_result.num_tokens}")
    print(f"Backend weights: {ppl_result.backend_model_bin}")
    print(f"Backend metadata: {ppl_result.backend_model_metadata}")

    if args.skip_zero_shot:
        print("\nZero-shot: skipped")
    elif zero_shot_acc is None:
        print(f"\nZero-shot ({args.zero_shot_task}): unavailable ({zero_shot_metric})")
    else:
        print(f"\nZero-shot ({args.zero_shot_task}) {zero_shot_metric}: {zero_shot_acc:.4f}")

    out_dir = ROOT / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "leaderboard_eval_report.json"
    out = {
        "stream_triad": {
            "bandwidth_gb_s": hw.memory_bandwidth_gbps,
            "method": hw.method,
            "dtype": hw.triad.dtype if hw.triad else None,
            "implementation": hw.triad.implementation if hw.triad else None,
            "array_mb": args.stream_array_mb,
            "runs": args.stream_runs,
        },
        "roofline": {
            "t_measured": bpt.t_context,
            "weights_gb": bpt.weight_bytes / 1e9,
            "kv_gb": bpt.kv_bytes / 1e9,
            "bytes_per_token_gb": bpt.total_gb,
            "ceiling_tps": ceiling_tps,
            "observed_tps": ppl_result.tokens_per_second,
            "efficiency_pct": efficiency_pct,
            "arch_source": arch_source,
            "num_layers": arch.num_layers,
            "num_attention_heads": arch.num_attention_heads,
            "num_kv_heads": arch.num_kv_heads,
            "head_dim": arch.head_dim,
            "kv_bytes_per_element": args.kv_bytes_per_element,
        },
        "ppl": {
            "bits": args.bits,
            "ppl": ppl_result.ppl,
            "avg_nll": ppl_result.avg_nll,
            "num_tokens": ppl_result.num_tokens,
            "tokens_per_second": ppl_result.tokens_per_second,
            "elapsed_sec": ppl_result.elapsed_sec,
            "windows": ppl_result.windows,
            "ppl_route": ppl_result.ppl_route,
            "hf_model_id": ppl_result.hf_model_id,
            "backend_model_bin": ppl_result.backend_model_bin,
            "backend_model_metadata": ppl_result.backend_model_metadata,
        },
        "zero_shot": {
            "task": args.zero_shot_task,
            "metric": zero_shot_metric,
            "accuracy": zero_shot_acc,
            "limit": None if args.skip_zero_shot else args.zero_shot_limit,
        },
        "curve": curve_points,
    }
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved report: {out_path}")


if __name__ == "__main__":
    main()
