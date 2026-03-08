"""
ASDSL Full LM-Eval Harness Runner
----------------------------------
Runs standard 0-shot benchmarks across all quantization configs:
  FP16 (baseline), 8-bit, 4-bit, 3-bit

Tasks: piqa, arc_easy, winogrande, hellaswag
Results are saved incrementally so partial runs are preserved.

Usage:
  python evals/run_full_eval.py
  python evals/run_full_eval.py --bits 16 8 4 --tasks piqa arc_easy
  python evals/run_full_eval.py --limit 30     # override per-task limit
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import lm_eval
from lm_eval_harness import ASDSLHarnessModel

RESULTS_DIR = ROOT / "benchmarks" / "results"

# Per-task limits tuned for ~2-3 hour total runtime on CPU
# piqa: 2 choices, ~25 tok/example -> fast
# arc_easy: 4 choices, ~30 tok/example -> medium
# winogrande: 2 choices, ~30 tok/example -> fast
# hellaswag: 4 choices, ~90 tok/example -> slow
DEFAULT_LIMITS: dict[str, dict[int, int]] = {
    "piqa":       {16: 30, 8: 20, 4: 20, 3: 15},
    "arc_easy":   {16: 20, 8: 15, 4: 12, 3: 10},
    "winogrande": {16: 25, 8: 18, 4: 15, 3: 12},
    "hellaswag":  {16: 12, 8: 8,  4: 6,  3: 5},
}


def run_task(model: ASDSLHarnessModel, task: str, limit: int, bits: int) -> dict:
    print(f"\n  --- Task: {task} (limit={limit}) ---", flush=True)
    t0 = time.perf_counter()
    results = lm_eval.simple_evaluate(
        model=model,
        tasks=[task],
        num_fewshot=0,
        limit=limit,
    )
    elapsed = time.perf_counter() - t0
    task_results = results.get("results", {}).get(task, {})
    # Extract primary accuracy metric
    acc = task_results.get("acc,none", task_results.get("acc_norm,none", None))
    print(f"  {task}: acc={acc:.4f}  ({elapsed/60:.1f} min)", flush=True)
    return {
        "task": task,
        "bits": bits,
        "limit": limit,
        "acc": acc,
        "acc_norm": task_results.get("acc_norm,none"),
        "acc_stderr": task_results.get("acc_stderr,none"),
        "elapsed_sec": elapsed,
        "full": task_results,
    }


def main():
    parser = argparse.ArgumentParser(description="ASDSL Full LM-Eval Suite")
    parser.add_argument("--bits", type=int, nargs="*", default=[16, 8, 4, 3],
                        choices=[2, 3, 4, 8, 16])
    parser.add_argument("--tasks", type=str, nargs="*",
                        default=["piqa", "arc_easy", "winogrande", "hellaswag"])
    parser.add_argument("--limit", type=int, default=None,
                        help="Override per-task limit for all tasks and configs")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / "lm_eval_results.json"

    # Load existing partial results
    all_results: list[dict] = []
    if out_file.exists():
        try:
            all_results = json.loads(out_file.read_text())
            done_keys = {(r["bits"], r["task"]) for r in all_results}
            print(f"Loaded {len(all_results)} existing results, skipping completed configs.")
        except Exception:
            done_keys = set()
    else:
        done_keys = set()

    total_configs = len(args.bits) * len(args.tasks)
    done_count = sum(1 for b in args.bits for t in args.tasks if (b, t) in done_keys)
    print(f"\nASDSL LM-Eval Harness")
    print(f"Configs to run: {len(args.bits)} bits x {len(args.tasks)} tasks = {total_configs} total")
    print(f"Already done: {done_count}")
    print(f"Remaining: {total_configs - done_count}")

    for bits in args.bits:
        label = "FP16" if bits == 16 else f"ASDSL {bits}-bit"
        pending_tasks = [t for t in args.tasks if (bits, t) not in done_keys]
        if not pending_tasks:
            print(f"\n[{label}] All tasks already done, skipping.")
            continue

        print(f"\n{'='*66}")
        print(f"  Loading model: {label}")
        print(f"{'='*66}")
        t_load = time.perf_counter()
        model = ASDSLHarnessModel(bits=bits)
        load_sec = time.perf_counter() - t_load
        print(f"  Model loaded in {load_sec:.0f}s")

        for task in pending_tasks:
            if (bits, task) in done_keys:
                continue
            limit = args.limit or DEFAULT_LIMITS.get(task, {}).get(bits, 10)
            try:
                result = run_task(model, task, limit, bits)
                all_results.append(result)
                done_keys.add((bits, task))
                # Save incrementally after each task
                out_file.write_text(json.dumps(all_results, indent=2, default=str))
                print(f"  Saved: {out_file}", flush=True)
            except Exception as e:
                print(f"  ERROR on {task} bits={bits}: {e}", flush=True)

        del model  # free RAM before loading next config

    # Print final summary table
    print(f"\n{'='*80}")
    print("  ASDSL LM-EVAL HARNESS - FINAL RESULTS (0-shot accuracy)")
    print(f"{'='*80}")
    tasks = [t for t in args.tasks]
    bits_list = [b for b in args.bits]

    # Header
    header = f"  {'Config':<12}"
    for t in tasks:
        short = t[:10]
        header += f"  {short:>10}"
    print(header)
    print("  " + "-" * (12 + 12 * len(tasks)))

    # Rows
    for bits in bits_list:
        label = "FP16" if bits == 16 else f"{bits}-bit"
        row = f"  {label:<12}"
        for task in tasks:
            r = next((x for x in all_results if x["bits"] == bits and x["task"] == task), None)
            if r and r.get("acc") is not None:
                row += f"  {r['acc']:>10.4f}"
            else:
                row += f"  {'n/a':>10}"
        print(row)
    print(f"{'='*80}")

    # Published Phi-4 baselines for comparison (from Microsoft blog/paper)
    print("\n  Published Phi-4 FP16 baselines (from Microsoft):")
    print("  piqa: 0.837 | arc_easy: 0.887 | winogrande: 0.793 | hellaswag: 0.621")
    print(f"{'='*80}")
    print(f"\n  Full results: {out_file}")


if __name__ == "__main__":
    main()
