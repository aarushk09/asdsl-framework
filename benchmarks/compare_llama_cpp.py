"""Side-by-side ASDSL vs llama.cpp benchmark: 3 prompts × N runs each + optional PPL.

Fair comparison defaults (--fair, on by default):
  - Same thread count on ASDSL (OMP/MKL/OpenBLAS) and llama.cpp (-t / -p)
  - Canonical config C0 @ 12 threads (physical affinity), or C1 with --weight-parity
  - Same prompts, greedy decode (temp=0), ignore-eos on both sides
  - Aborts if llama-cli reports a different thread count than requested

Session order (parity protocol):
  1. ASDSL only — all prompts, all runs (no llama-cli between runs)
  2. Idle cooldown between frameworks
  3. llama.cpp only — same prompts, same run count

Usage:
  python benchmarks/compare_llama_cpp.py --fair --with-ppl
  python benchmarks/compare_llama_cpp.py --weight-parity --runs 5
  python benchmarks/compare_llama_cpp.py --allow-exploratory --asdsl-config C0-fast

Output: benchmarks/results/side_by_side_comparison.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.parity_benchmark import (  # noqa: E402
    env_for_config,
    load_manifest,
    resolve_config_key,
    verify_native_extensions,
)

GGUF = ROOT / "models" / "llama_cpp" / "phi4-mm-Q4_K_M.gguf"
LLAMA_CLI = ROOT / "tools" / "llama.cpp" / "llama-cli.exe"
LLAMA_BENCH = ROOT / "tools" / "llama.cpp" / "llama-bench.exe"
LLAMA_PPL = ROOT / "tools" / "llama.cpp" / "llama-perplexity.exe"
OUT = ROOT / "benchmarks" / "results" / "side_by_side_comparison.json"

# Same instructional prompts used in interactive validation (2026-06-14).
BENCHMARK_PROMPTS: list[dict[str, str]] = [
    {
        "id": "gravity",
        "user": "Explain gravity in simple terms.",
        "system": "You are a helpful assistant.",
    },
    {
        "id": "quantization_typos",
        "user": "Explain quantizatssion in llmss in simple terms.",
        "system": "You are a helpful assistant.",
    },
    {
        "id": "gravity_one_sentence",
        "user": "Explain gravity in one sentence.",
        "system": "You are a helpful assistant.",
    },
]

DEFAULT_SYSTEM = "You are a helpful assistant."

# Parity manifest: canonical 12-thread decode (matches llama L0).
CANONICAL_THREADS = 12
CANONICAL_ASDSL_CONFIG = "C0"
WEIGHT_PARITY_ASDSL_CONFIG = "C1"  # same GGUF tensors as llama
EXPLORATORY_CONFIGS = frozenset(
    {"C0-fast", "C0.1", "C3", "C4", "C5", "C6", "C6-smt", "C6-SMT"}
)


def sync_thread_env(env: dict[str, str], threads: int) -> dict[str, str]:
    """Lock all CPU thread pools to the same count (fair vs llama -t)."""
    out = {**env}
    t = str(threads)
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        out[key] = t
    return out


def threads_for_manifest_config(cfg_key: str, manifest: dict) -> int:
    spec = manifest.get("configs", {}).get(cfg_key, {})
    if spec.get("threads_cli"):
        return int(spec["threads_cli"])
    return int(manifest["stacks"]["asdsl"].get("threads_cli", CANONICAL_THREADS))


def resolve_fair_setup(
    args: argparse.Namespace,
    manifest: dict,
) -> tuple[str, int, int, list[str], bool]:
    """Return (config_key, asdsl_threads, llama_threads, fairness_notes, use_gguf)."""
    notes: list[str] = []
    if args.weight_parity:
        cfg_key = resolve_config_key(WEIGHT_PARITY_ASDSL_CONFIG, manifest)
        notes.append("weight_parity: ASDSL C1 loads same GGUF Q4_K tensors as llama.cpp")
    elif args.asdsl_config:
        cfg_key = resolve_config_key(args.asdsl_config, manifest)
    else:
        cfg_key = resolve_config_key(CANONICAL_ASDSL_CONFIG, manifest)

    if not cfg_key:
        raise SystemExit(f"ERROR: unknown ASDSL config {args.asdsl_config!r}")

    spec = manifest["configs"][cfg_key]
    if spec.get("disabled"):
        raise SystemExit(f"ERROR: config {cfg_key} is disabled in manifest")

    cfg_threads = threads_for_manifest_config(cfg_key, manifest)

    if args.fair:
        if cfg_key in EXPLORATORY_CONFIGS and not args.allow_exploratory:
            raise SystemExit(
                f"ERROR: {cfg_key} is exploratory (non-canonical threads/affinity). "
                f"For fair vs llama use default C0 @ {CANONICAL_THREADS}t, or "
                f"--weight-parity for C1, or pass --allow-exploratory knowingly."
            )
        if not args.weight_parity and cfg_key == CANONICAL_ASDSL_CONFIG:
            threads = CANONICAL_THREADS
            if cfg_threads != CANONICAL_THREADS:
                notes.append(
                    f"forced threads {CANONICAL_THREADS} for canonical C0 (manifest had {cfg_threads})"
                )
        else:
            threads = cfg_threads
        if args.llama_threads and args.llama_threads != threads:
            raise SystemExit(
                f"ERROR: --llama-threads {args.llama_threads} != ASDSL {threads} "
                f"(fair mode requires identical thread counts). Use --no-fair to override."
            )
        llama_threads = threads
        notes.extend(
            [
                f"threads: ASDSL and llama both use {threads}",
                "decode: greedy, temperature=0, ignore-eos on both stacks",
                "metric: ASDSL decode_2_N (tokens 2-N) vs llama cli footer (all tokens)",
            ]
        )
        if cfg_key == CANONICAL_ASDSL_CONFIG and not args.weight_parity:
            notes.append(
                "weight_caveat: C0 uses HF asymmetric preq; llama uses Q4_K_M GGUF — "
                "speed comparison is canonical stacks, not identical bytes. Use --weight-parity for same tensors."
            )
    else:
        threads = cfg_threads
        llama_threads = args.llama_threads or threads
        notes.append("fair_mode: OFF — results may not be comparable; document env manually")

    return cfg_key, threads, llama_threads, notes, spec.get("gguf_path", False)


def print_fairness_banner(
    cfg_key: str,
    threads: int,
    llama_threads: int,
    notes: list[str],
    use_gguf: bool,
) -> None:
    print("\n" + "=" * 72, flush=True)
    print("FAIRNESS PROTOCOL", flush=True)
    print("=" * 72, flush=True)
    print(f"  ASDSL config     : {cfg_key}", flush=True)
    print(f"  ASDSL threads    : {threads} (OMP/MKL/OpenBLAS synced)", flush=True)
    print(f"  llama.cpp threads: {llama_threads} (-t and llama-bench -p)", flush=True)
    print(f"  Same GGUF weights: {use_gguf}", flush=True)
    print(f"  GGUF path        : {GGUF}", flush=True)
    for n in notes:
        print(f"  • {n}", flush=True)
    print("=" * 72 + "\n", flush=True)


def stats(vals: list[float]) -> dict:
    if not vals:
        return {}
    a = np.array(vals, dtype=np.float64)
    trimmed = np.sort(a)
    if len(trimmed) >= 5:
        trimmed = trimmed[1:-1]
    return {
        "runs": [round(float(x), 3) for x in vals],
        "mean": round(float(a.mean()), 3),
        "std": round(float(a.std()), 3),
        "min": round(float(a.min()), 3),
        "max": round(float(a.max()), 3),
        "trimmed_mean": round(float(trimmed.mean()), 3) if len(trimmed) else round(float(a.mean()), 3),
    }


def _chat_prompt_string(tokenizer, system: str, user: str) -> str:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def run_asdsl_once(
    env: dict[str, str],
    threads: int,
    user_prompt: str,
    system_prompt: str,
    max_new_tokens: int,
    use_gguf: bool = False,
) -> dict:
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "phi4_cpu_run.py"),
        "--bits",
        "4",
        "--max-new-tokens",
        str(max_new_tokens),
        "--prompt",
        user_prompt,
        "--system-prompt",
        system_prompt,
        "--threads",
        str(threads),
    ]
    if use_gguf:
        cmd.extend(["--gguf-path", str(GGUF)])
    timeout = 7200 if use_gguf else 1200
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        return {"ok": False, "error": out[-2500:]}
    m2n = re.search(r"decode\s+([\d.]+)\s+tok/s\s+\(tokens\s+2-(\d+)\)", out)
    decode_2_n = float(m2n.group(1)) if m2n else None
    m_full = re.search(
        r"Generated\s*:\s*(\d+)\s+tokens\s*\|\s*([\d.]+)\s+tok/s",
        out,
    )
    full_1_n = float(m_full.group(2)) if m_full else None
    m_inf = re.search(r"Inference:\s*(.+)", out)
    preview = ""
    if "Assistant:" in out:
        tail = out.split("Assistant:", 1)[-1]
        tail = re.sub(r"=+\s*", " ", tail)
        tail = re.sub(r"\s+", " ", tail).strip()
        preview = tail[:280]
    return {
        "ok": True,
        "decode_2_N": decode_2_n,
        "full_1_N": full_1_n,
        "inference_mode": m_inf.group(1).strip() if m_inf else None,
        "response_preview": preview,
    }


def run_llama_once(
    chat_prompt: str,
    threads: int,
    max_new_tokens: int,
) -> dict:
    if not LLAMA_CLI.is_file():
        return {"ok": False, "error": f"missing {LLAMA_CLI}"}
    cmd = [
        str(LLAMA_CLI),
        "-m",
        str(GGUF),
        "-p",
        chat_prompt,
        "-n",
        str(max_new_tokens),
        "-t",
        str(threads),
        "--temp",
        "0",
        "--ignore-eos",
        "-ngl",
        "0",
        "-no-cnv",
        "--no-display-prompt",
        "-st",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT / "tools" / "llama.cpp"),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        return {"ok": False, "error": out[-2500:]}
    m = re.search(r"Generation:\s*([\d.]+)\s*t/s", out)
    footer = float(m.group(1)) if m else None
    reported = threads
    tm = re.search(r"threads\s*:\s*(\d+)", out, re.I)
    if tm:
        reported = int(tm.group(1))
    preview = re.sub(r"\s+", " ", out.strip())[-280:]
    return {
        "ok": True,
        "cli_footer_tok_s": footer,
        "threads_reported": reported,
        "thread_mismatch": reported != threads,
        "response_preview": preview,
    }


def run_llama_bench_tg128(threads: int) -> dict:
    if not LLAMA_BENCH.is_file():
        return {"ok": False, "error": "llama-bench missing"}
    proc = subprocess.run(
        [
            str(LLAMA_BENCH),
            "-m",
            str(GGUF),
            "-t",
            str(threads),
            "-p",
            str(threads),
            "-n",
            "1",
            "-ngl",
            "0",
        ],
        cwd=str(ROOT / "tools" / "llama.cpp"),
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    tg = None
    for line in out.splitlines():
        m = re.search(r"tg128\s+[\w.]+\s+([\d.]+)", line, re.I)
        if m:
            tg = float(m.group(1))
    return {"ok": proc.returncode == 0, "tg128_tok_s": tg, "raw_tail": out[-400:]}


def run_asdsl_ppl(env: dict[str, str], threads: int, max_tokens: int = 2048) -> dict:
    """WikiText-2 slice perplexity (ASDSL unified path)."""
    os.environ.update(env)
    try:
        from benchmarks.comprehensive_bench import evaluate_perplexity, load_wikitext_tokens
        from experiments.phi4_cpu_run import WeightStore, load_tokenizer, set_thread_count

        set_thread_count(threads)
        tok = load_tokenizer()
        tokens = load_wikitext_tokens(tok, max_tokens)
        store = WeightStore(bits=4, group_size=32, enable_qcsd=False)
        store.load()
        if os.environ.get("ASDSL_USE_Q4KM_GGUF", "0").strip().lower() in ("1", "true", "yes"):
            store.load_from_gguf(str(GGUF))
        store.warm_cache()
        store._use_unified = True
        result = evaluate_perplexity(tokens, store, stride=512)
        return {"ok": True, **result}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def run_llama_ppl(threads: int) -> dict:
    if not LLAMA_PPL.is_file() or not GGUF.is_file():
        return {"ok": False, "error": "llama-perplexity or GGUF missing"}
    proc = subprocess.run(
        [
            str(LLAMA_PPL),
            "-m",
            str(GGUF),
            "-f",
            "wikitext",
            "--file-type",
            "wikitext-2-raw-v1",
            "-t",
            str(threads),
            "-ngl",
            "0",
        ],
        cwd=str(ROOT / "tools" / "llama.cpp"),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=3600,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    ppl = None
    for line in out.splitlines():
        m = re.search(r"^\|\s*[\d.]+\s*\|\s*([\d.]+)\s*\|", line)
        if m:
            ppl = float(m.group(1))
        m2 = re.search(r"Perplexity:\s*([\d.]+)", line, re.I)
        if m2:
            ppl = float(m2.group(1))
    if ppl is None:
        for line in reversed(out.splitlines()):
            m3 = re.search(r"([\d.]+)\s*±", line)
            if m3:
                ppl = float(m3.group(1))
                break
    return {
        "ok": proc.returncode == 0 and ppl is not None,
        "ppl": ppl,
        "raw_tail": out[-1200:],
    }


def session_asdsl(
    prompts: list[dict],
    env: dict[str, str],
    threads: int,
    n_runs: int,
    max_new_tokens: int,
    cooldown_s: int,
    use_gguf: bool = False,
) -> dict:
    from benchmarks.thermal_utils import wait_until_cool

    results: dict[str, dict] = {}
    thermal = wait_until_cool(max_temp_c=65, initial_sleep_s=60)
    for spec in prompts:
        pid = spec["id"]
        print(f"\n=== ASDSL prompt: {pid} ===", flush=True)
        runs_decode: list[float] = []
        runs_full: list[float] = []
        run_details: list[dict] = []
        for i in range(n_runs):
            wait_until_cool(max_temp_c=65, initial_sleep_s=0, max_wait_s=300)
            print(f"  run {i + 1}/{n_runs}", flush=True)
            r = run_asdsl_once(
                env,
                threads,
                spec["user"],
                spec.get("system", DEFAULT_SYSTEM),
                max_new_tokens,
                use_gguf=use_gguf,
            )
            if not r.get("ok"):
                return {"error": f"ASDSL failed {pid}", "detail": r, "thermal": thermal}
            if r.get("decode_2_N") is not None:
                runs_decode.append(r["decode_2_N"])
            if r.get("full_1_N") is not None:
                runs_full.append(r["full_1_N"])
            print(
                f"    decode_2_N={r.get('decode_2_N')} full_1_N={r.get('full_1_N')}",
                flush=True,
            )
            run_details.append(r)
            if i + 1 < n_runs:
                time.sleep(cooldown_s)
        results[pid] = {
            "prompt": spec,
            "decode_2_N": stats(runs_decode),
            "full_1_N": stats(runs_full),
            "runs_detail": run_details,
        }
    return {"thermal": thermal, "prompts": results}


def session_llama(
    prompts: list[dict],
    tokenizer,
    threads: int,
    n_runs: int,
    max_new_tokens: int,
    cooldown_s: int,
    session_idle_s: int,
) -> dict:
    from benchmarks.thermal_utils import wait_until_cool

    subprocess.run(["taskkill", "/F", "/IM", "llama-cli.exe", "/T"], capture_output=True)
    time.sleep(2)
    print(f"\n=== Idle {session_idle_s}s before llama session ===", flush=True)
    time.sleep(session_idle_s)
    thermal = wait_until_cool(max_temp_c=65, initial_sleep_s=30)
    results: dict[str, dict] = {}
    mismatches = 0
    for spec in prompts:
        pid = spec["id"]
        chat = _chat_prompt_string(
            tokenizer, spec.get("system", DEFAULT_SYSTEM), spec["user"]
        )
        print(f"\n=== llama.cpp prompt: {pid} ===", flush=True)
        runs_footer: list[float] = []
        run_details: list[dict] = []
        for i in range(n_runs):
            wait_until_cool(max_temp_c=65, initial_sleep_s=0, max_wait_s=300)
            print(f"  run {i + 1}/{n_runs}", flush=True)
            r = run_llama_once(chat, threads, max_new_tokens)
            if not r.get("ok"):
                return {"error": f"llama failed {pid}", "detail": r, "thermal": thermal}
            if r.get("thread_mismatch"):
                mismatches += 1
            v = r.get("cli_footer_tok_s")
            if v is not None:
                runs_footer.append(v)
            print(f"    footer={v} threads={r.get('threads_reported')}", flush=True)
            run_details.append(r)
            if i + 1 < n_runs:
                time.sleep(cooldown_s)
        results[pid] = {
            "prompt": spec,
            "cli_footer": stats(runs_footer),
            "runs_detail": run_details,
        }
    bench = run_llama_bench_tg128(threads)
    return {
        "thermal": thermal,
        "prompts": results,
        "llama_bench_tg128": bench,
        "thread_mismatch_runs": mismatches,
    }


def aggregate_comparison(asdsl: dict, llama: dict) -> dict:
    rows: list[dict] = []
    for pid, a_block in asdsl.get("prompts", {}).items():
        l_block = llama.get("prompts", {}).get(pid, {})
        a_mean = (a_block.get("decode_2_N") or {}).get("trimmed_mean")
        l_mean = (l_block.get("cli_footer") or {}).get("trimmed_mean")
        ratio = None
        if a_mean and l_mean and l_mean > 0:
            ratio = round(100.0 * a_mean / l_mean, 1)
        rows.append(
            {
                "prompt_id": pid,
                "asdsl_decode_2_N_trimmed_mean": a_mean,
                "llama_footer_trimmed_mean": l_mean,
                "asdsl_pct_of_llama": ratio,
            }
        )
    a_all = [
        (v.get("decode_2_N") or {}).get("trimmed_mean")
        for v in asdsl.get("prompts", {}).values()
    ]
    l_all = [
        (v.get("cli_footer") or {}).get("trimmed_mean")
        for v in llama.get("prompts", {}).values()
    ]
    a_all = [x for x in a_all if x is not None]
    l_all = [x for x in l_all if x is not None]
    grand_ratio = None
    if a_all and l_all and np.mean(l_all) > 0:
        grand_ratio = round(100.0 * float(np.mean(a_all)) / float(np.mean(l_all)), 1)
    return {
        "per_prompt": rows,
        "asdsl_grand_mean_decode_2_N": round(float(np.mean(a_all)), 3) if a_all else None,
        "llama_grand_mean_footer": round(float(np.mean(l_all)), 3) if l_all else None,
        "asdsl_pct_of_llama_grand_mean": grand_ratio,
    }


def print_summary(doc: dict) -> None:
    print("\n" + "=" * 72, flush=True)
    print("SIDE-BY-SIDE SUMMARY (trimmed mean = drop min/max of 5 runs)", flush=True)
    print("=" * 72, flush=True)
    cmp = doc.get("comparison", {})
    for row in cmp.get("per_prompt", []):
        print(
            f"  {row['prompt_id']:22s}  ASDSL {row.get('asdsl_decode_2_N_trimmed_mean')} tok/s  "
            f"llama {row.get('llama_footer_trimmed_mean')} tok/s  "
            f"({row.get('asdsl_pct_of_llama')}% of llama)",
            flush=True,
        )
    print(
        f"\n  GRAND MEAN  ASDSL {cmp.get('asdsl_grand_mean_decode_2_N')}  "
        f"llama {cmp.get('llama_grand_mean_footer')}  "
        f"({cmp.get('asdsl_pct_of_llama_grand_mean')}% of llama)",
        flush=True,
    )
    if doc.get("ppl"):
        ppl = doc["ppl"]
        print(
            f"\n  PPL  ASDSL {ppl.get('asdsl', {}).get('ppl')}  "
            f"llama {ppl.get('llama', {}).get('ppl')}",
            flush=True,
        )
    if doc.get("llama_bench_tg128"):
        print(f"  llama-bench tg128: {doc['llama_bench_tg128'].get('tg128_tok_s')} tok/s", flush=True)
    print("=" * 72 + "\n", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="ASDSL vs llama.cpp side-by-side (3 prompts × N runs)")
    ap.add_argument(
        "--asdsl-config",
        default=None,
        help=f"parity manifest config (fair default: {CANONICAL_ASDSL_CONFIG})",
    )
    ap.add_argument(
        "--llama-threads",
        type=int,
        default=0,
        help="llama threads (fair mode: must match ASDSL; ignored when 0)",
    )
    ap.add_argument("--fair", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--weight-parity",
        action="store_true",
        help=f"use {WEIGHT_PARITY_ASDSL_CONFIG} (same GGUF tensors as llama)",
    )
    ap.add_argument(
        "--allow-exploratory",
        action="store_true",
        help="allow non-canonical configs (C0-fast, C6-smt, etc.)",
    )
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--cooldown", type=int, default=30, help="seconds between runs")
    ap.add_argument("--session-idle", type=int, default=120, help="seconds between ASDSL and llama sessions")
    ap.add_argument("--with-ppl", action="store_true", help="run WikiText PPL for both stacks (slow)")
    ap.add_argument("--ppl-tokens", type=int, default=2048, help="WikiText tokens for ASDSL PPL")
    ap.add_argument("--skip-kernel-preflight", action="store_true")
    ap.add_argument("-o", "--output", type=Path, default=OUT)
    args = ap.parse_args()

    manifest = load_manifest()
    try:
        cfg_key, threads, llama_threads, fairness_notes, use_gguf = resolve_fair_setup(
            args, manifest
        )
    except SystemExit as exc:
        print(str(exc), flush=True)
        return 1

    native = verify_native_extensions()
    if not native.get("ok"):
        print("ERROR: native extensions not built; run: python setup.py build_ext --inplace", flush=True)
        return 1

    env = env_for_config(cfg_key, manifest)
    env = sync_thread_env(env, threads)

    if not GGUF.is_file():
        print(f"ERROR: missing GGUF at {GGUF}", flush=True)
        return 1

    print_fairness_banner(cfg_key, threads, llama_threads, fairness_notes, use_gguf)

    doc: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_note": manifest.get("hardware"),
        "fairness_protocol": {
            "fair_mode": args.fair,
            "weight_parity": args.weight_parity,
            "asdsl_config": cfg_key,
            "asdsl_threads": threads,
            "llama_threads": llama_threads,
            "same_gguf_weights": use_gguf,
            "decode": "greedy, temperature=0, ignore-eos",
            "notes": fairness_notes,
        },
        "asdsl_config": cfg_key,
        "asdsl_effective_env": {
            k: env[k]
            for k in sorted(env)
            if k.startswith("ASDSL_")
            or k
            in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "llama_threads": llama_threads,
        "n_runs_per_prompt": args.runs,
        "max_new_tokens": args.max_new_tokens,
        "prompts": BENCHMARK_PROMPTS,
        "native_extensions": native,
    }

    if not args.skip_kernel_preflight:
        from benchmarks.kernel_preflight import run_kernel_preflight

        pf = run_kernel_preflight(threads=threads)
        doc["kernel_preflight"] = pf
        if not pf.get("ok"):
            print(f"ERROR: kernel preflight failed: {pf.get('error')}", flush=True)
            return 1

    subprocess.run(["taskkill", "/F", "/IM", "llama-cli.exe", "/T"], capture_output=True)
    time.sleep(1)

    print(f"=== ASDSL session ({cfg_key}, {threads} threads) ===", flush=True)
    asdsl_sess = session_asdsl(
        BENCHMARK_PROMPTS,
        env,
        threads,
        args.runs,
        args.max_new_tokens,
        args.cooldown,
        use_gguf=use_gguf,
    )
    doc["session_asdsl"] = asdsl_sess
    if "error" in asdsl_sess:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        print(f"ERROR: {asdsl_sess['error']}", flush=True)
        return 1

    from experiments.phi4_cpu_run import load_tokenizer

    tokenizer = load_tokenizer()
    print(f"\n=== llama.cpp session ({llama_threads} threads) ===", flush=True)
    llama_sess = session_llama(
        BENCHMARK_PROMPTS,
        tokenizer,
        llama_threads,
        args.runs,
        args.max_new_tokens,
        args.cooldown,
        args.session_idle,
    )
    doc["session_llama"] = llama_sess
    if "error" in llama_sess:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        print(f"ERROR: {llama_sess['error']}", flush=True)
        return 1

    mismatches = int(llama_sess.get("thread_mismatch_runs", 0))
    if args.fair and mismatches > 0:
        msg = (
            f"ERROR: llama-cli reported wrong thread count on {mismatches} run(s) "
            f"(requested {llama_threads}). Results are not fair — aborting."
        )
        doc["fairness_error"] = msg
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        print(msg, flush=True)
        return 1

    doc["llama_bench_tg128"] = llama_sess.get("llama_bench_tg128")
    doc["comparison"] = aggregate_comparison(asdsl_sess, llama_sess)

    if args.with_ppl:
        print("\n=== PPL: ASDSL (WikiText-2 slice) ===", flush=True)
        doc["ppl"] = {"asdsl": run_asdsl_ppl(env, threads, args.ppl_tokens)}
        print(json.dumps(doc["ppl"]["asdsl"], indent=2), flush=True)
        print("\n=== PPL: llama.cpp ===", flush=True)
        doc["ppl"]["llama"] = run_llama_ppl(llama_threads)
        print(json.dumps(doc["ppl"]["llama"], indent=2), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print_summary(doc)
    print(f"Wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
