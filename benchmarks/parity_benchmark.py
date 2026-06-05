"""Canonical ASDSL vs llama.cpp parity benchmark (separate sessions, 12 threads).

Usage:
  python benchmarks/parity_benchmark.py --config C0,L0
  python benchmarks/parity_benchmark.py --config C1 --asdsl-only
  python benchmarks/parity_benchmark.py --config L0 --llama-only

See benchmarks/results/parity_manifest.json for locked protocol.
"""

from __future__ import annotations

import argparse
import hashlib
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

MANIFEST_PATH = ROOT / "benchmarks" / "results" / "parity_manifest.json"
GGUF = ROOT / "models" / "llama_cpp" / "phi4-mm-Q4_K_M.gguf"
LLAMA_CLI = ROOT / "tools" / "llama.cpp" / "llama-cli.exe"
LLAMA_BENCH = ROOT / "tools" / "llama.cpp" / "llama-bench.exe"
CHAT_PROMPT = "<|user|>The<|end|><|assistant|>"
N_TOK = 128
THREADS = 12
N_RUNS = 5
COOL_S = 30


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stats(vals: list[float]) -> dict:
    a = np.array(vals, dtype=np.float64)
    return {
        "runs": [round(float(x), 3) for x in vals],
        "mean": round(float(a.mean()), 3),
        "std": round(float(a.std()), 3),
        "min": round(float(a.min()), 3),
        "max": round(float(a.max()), 3),
    }


def high_variance_warning(s_asdsl: dict, s_llama: dict | None) -> list[str]:
    warnings: list[str] = []
    if s_asdsl.get("std", 0) > 0.2:
        warnings.append(f"ASDSL std {s_asdsl['std']} > 0.2")
    if s_llama and s_llama.get("std", 0) > 0.3:
        warnings.append(f"llama std {s_llama['std']} > 0.3")
    return warnings


def env_for_config(cfg_id: str, manifest: dict) -> dict[str, str]:
    e = os.environ.copy()
    e.update(manifest["stacks"]["asdsl"]["env_required"])
    e["OMP_NUM_THREADS"] = str(THREADS)
    e["MKL_NUM_THREADS"] = str(THREADS)
    e["PYTHONIOENCODING"] = "utf-8"
    if cfg_id in manifest["configs"]:
        spec = manifest["configs"][cfg_id]
        for k, v in (spec.get("env") or {}).items():
            e[k] = str(v)
    return e


def run_asdsl_once(cfg_id: str, manifest: dict) -> dict:
    spec = manifest["configs"][cfg_id]
    env = env_for_config(cfg_id, manifest)
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "phi4_cpu_run.py"),
        "--bits",
        "4",
        "--max-new-tokens",
        str(N_TOK),
        "--prompt",
        manifest["prompt"]["asdsl_user_prompt"],
        "--system-prompt",
        manifest["prompt"]["asdsl_system_prompt"],
        "--threads",
        str(THREADS),
    ]
    if spec.get("gguf_path"):
        cmd.extend(["--gguf-path", str(GGUF)])
    timeout = 7200 if spec.get("gguf_path") else 1200
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
        return {"ok": False, "error": out[-2000:]}
    m2n = re.search(r"decode\s+([\d.]+)\s+tok/s\s+\(tokens\s+2-(\d+)\)", out)
    decode_2_n = float(m2n.group(1)) if m2n else None
    m_full = re.search(
        r"Generated\s*:\s*(\d+)\s+tokens\s*\|\s*([\d.]+)\s+tok/s",
        out,
    )
    full_1_n = float(m_full.group(2)) if m_full else None
    if decode_2_n is None and full_1_n is not None:
        decode_2_n = full_1_n
    return {
        "ok": True,
        "decode_2_N": decode_2_n,
        "full_1_N": full_1_n,
        "threads_requested": THREADS,
    }


def run_llama_cli_once() -> dict:
    if not LLAMA_CLI.is_file():
        return {"ok": False, "error": f"missing {LLAMA_CLI}"}
    cmd = [
        str(LLAMA_CLI),
        "-m",
        str(GGUF),
        "-p",
        CHAT_PROMPT,
        "-n",
        str(N_TOK),
        "-t",
        str(THREADS),
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
        timeout=600,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        return {"ok": False, "error": out[-2000:]}
    m = re.search(r"Generation:\s*([\d.]+)\s*t/s", out)
    footer = float(m.group(1)) if m else None
    threads_used = THREADS
    tm = re.search(r"threads\s*:\s*(\d+)", out, re.I)
    if tm:
        threads_used = int(tm.group(1))
    return {
        "ok": True,
        "cli_footer_tok_s": footer,
        "threads_reported": threads_used,
        "threads_requested": THREADS,
        "thread_mismatch": threads_used != THREADS,
    }


def run_llama_bench_tg128() -> dict:
    if not LLAMA_BENCH.is_file():
        return {"ok": False, "error": f"missing {LLAMA_BENCH}"}
    cmd = [
        str(LLAMA_BENCH),
        "-m",
        str(GGUF),
        "-t",
        str(THREADS),
        "-p",
        str(THREADS),
        "-n",
        "1",
        "-ngl",
        "0",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT / "tools" / "llama.cpp"),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    tg = None
    for line in out.splitlines():
        if "tg128" in line.lower() or "tg 128" in line.lower():
            parts = line.split()
            for p in parts:
                try:
                    tg = float(p.replace(",", ""))
                    break
                except ValueError:
                    continue
        m = re.search(r"tg128\s+[\w.]+\s+([\d.]+)", line, re.I)
        if m:
            tg = float(m.group(1))
            break
    return {"ok": proc.returncode == 0, "tg128_tok_s": tg, "raw_tail": out[-800:]}


def session_asdsl(configs: list[str], manifest: dict) -> dict:
    from benchmarks.thermal_utils import wait_until_cool

    subprocess.run(["taskkill", "/F", "/IM", "llama-cli.exe", "/T"], capture_output=True)
    time.sleep(2)
    thermal = wait_until_cool(max_temp_c=65, initial_sleep_s=60)
    results: dict[str, dict] = {}
    for cfg in configs:
        if cfg == "L0" or cfg not in manifest["configs"]:
            continue
        if manifest["configs"][cfg].get("stack") == "llama":
            continue
        runs_2n: list[float] = []
        runs_full: list[float] = []
        print(f"\n=== ASDSL session: {cfg} ===", flush=True)
        for i in range(N_RUNS):
            wait_until_cool(max_temp_c=65, initial_sleep_s=0, max_wait_s=300)
            print(f"  run {i + 1}/{N_RUNS}", flush=True)
            r = run_asdsl_once(cfg, manifest)
            if not r.get("ok"):
                print(r.get("error", "")[-500:])
                return {"error": f"{cfg} run failed", "detail": r}
            if r.get("decode_2_N") is not None:
                runs_2n.append(r["decode_2_N"])
            if r.get("full_1_N") is not None:
                runs_full.append(r["full_1_N"])
            print(f"    decode_2_N={r.get('decode_2_N')} full_1_N={r.get('full_1_N')}", flush=True)
            if i + 1 < N_RUNS:
                time.sleep(COOL_S)
        s2 = stats(runs_2n) if runs_2n else {}
        sf = stats(runs_full) if runs_full else {}
        results[cfg] = {
            "decode_2_N": s2,
            "full_1_N": sf,
            "high_variance_warning": high_variance_warning(s2, None),
        }
    return {"thermal": thermal, "configs": results}


def session_llama(manifest: dict) -> dict:
    from benchmarks.thermal_utils import wait_until_cool

    subprocess.run(["taskkill", "/F", "/IM", "llama-cli.exe", "/T"], capture_output=True)
    time.sleep(2)
    print("\n=== Idle between ASDSL and llama sessions (recommend >= 5 min) ===", flush=True)
    thermal = wait_until_cool(max_temp_c=65, initial_sleep_s=120)
    footer_runs: list[float] = []
    mismatches = 0
    for i in range(N_RUNS):
        wait_until_cool(max_temp_c=65, initial_sleep_s=0, max_wait_s=300)
        print(f"  llama run {i + 1}/{N_RUNS}", flush=True)
        r = run_llama_cli_once()
        if not r.get("ok"):
            return {"error": "llama failed", "detail": r}
        if r.get("thread_mismatch"):
            mismatches += 1
            print(
                f"    WARNING: llama threads {r.get('threads_reported')} != {THREADS}",
                flush=True,
            )
        v = r.get("cli_footer_tok_s")
        if v is not None:
            footer_runs.append(v)
        print(f"    footer={v}", flush=True)
        if i + 1 < N_RUNS:
            time.sleep(COOL_S)
    bench = run_llama_bench_tg128()
    s = stats(footer_runs)
    return {
        "thermal": thermal,
        "L0": {
            "cli_footer": s,
            "llama_bench_tg128": bench,
            "thread_mismatch_runs": mismatches,
            "high_variance_warning": high_variance_warning({}, s),
        },
    }


def main() -> int:
    global N_RUNS, COOL_S
    ap = argparse.ArgumentParser(description="Parity benchmark (manifest-driven)")
    ap.add_argument(
        "--config",
        default="C0,L0",
        help="Comma-separated config IDs (C0,C1,C2,C3,L0)",
    )
    ap.add_argument("--runs", type=int, default=N_RUNS, help="Runs per config (default 5)")
    ap.add_argument(
        "--cooldown",
        type=int,
        default=COOL_S,
        help="Seconds between runs (default 30)",
    )
    ap.add_argument("--asdsl-only", action="store_true")
    ap.add_argument("--llama-only", action="store_true")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=ROOT / "benchmarks" / "results" / "parity_run_latest.json",
    )
    args = ap.parse_args()
    N_RUNS = max(1, int(args.runs))
    COOL_S = max(0, int(args.cooldown))
    manifest = load_manifest()
    cfg_ids = [c.strip().upper() for c in args.config.split(",") if c.strip()]
    asdsl_cfgs = [c for c in cfg_ids if c != "L0"]
    want_llama = "L0" in cfg_ids and not args.asdsl_only
    want_asdsl = bool(asdsl_cfgs) and not args.llama_only

    out_doc: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(MANIFEST_PATH.relative_to(ROOT)),
        "threads": THREADS,
        "n_runs": N_RUNS,
        "gguf_sha256": sha256_file(GGUF),
    }

    if want_asdsl:
        out_doc["session_asdsl"] = session_asdsl(asdsl_cfgs, manifest)
        if "error" in out_doc["session_asdsl"]:
            args.output.write_text(json.dumps(out_doc, indent=2), encoding="utf-8")
            return 1

    if want_llama:
        out_doc["session_llama"] = session_llama(manifest)
        if "error" in out_doc.get("session_llama", {}):
            args.output.write_text(json.dumps(out_doc, indent=2), encoding="utf-8")
            return 1

    if want_asdsl and want_llama:
        a_mean = None
        l_mean = None
        for c in asdsl_cfgs:
            s = out_doc.get("session_asdsl", {}).get("configs", {}).get(c, {})
            a_mean = (s.get("decode_2_N") or {}).get("mean")
        l_mean = (out_doc.get("session_llama", {}).get("L0", {}).get("cli_footer") or {}).get(
            "mean"
        )
        if a_mean and l_mean:
            out_doc["ratio_pct"] = round(100.0 * a_mean / l_mean, 1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out_doc, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
