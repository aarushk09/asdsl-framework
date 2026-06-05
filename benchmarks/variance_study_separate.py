"""Decoupled variance: ASDSL-only or llama-only session (no thermal cross-heating)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GGUF = ROOT / "models" / "llama_cpp" / "phi4-mm-Q4_K_M.gguf"
LLAMA = ROOT / "tools" / "llama.cpp" / "llama-cli.exe"
N_RUNS = 5
COOL_S = 30
N_TOK = 128


def run_asdsl() -> float | None:
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "12",
            "MKL_NUM_THREADS": "12",
            "ASDSL_USE_UNIFIED": "1",
            "ASDSL_IGNORE_EOS": "1",
            "ASDSL_FUSED_GEMV": "1",
            "ASDSL_AFFINITY": "spread",
            "ASDSL_PREQ_G4FUSED": "0",
            "PYTHONIOENCODING": "utf-8",
        }
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments" / "phi4_cpu_run.py"),
            "--bits",
            "4",
            "--max-new-tokens",
            str(N_TOK),
            "--prompt",
            "The",
            "--threads",
            "12",
        ],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        print(out[-1500:], flush=True)
        return None
    m = re.search(r"decode\s+([\d.]+)\s+tok/s", out)
    if not m:
        m = re.search(r"Generated\s*:\s*\d+\s+tokens\s*\|\s*([\d.]+)\s+tok/s", out)
    return float(m.group(1)) if m else None


def run_llama() -> float | None:
    proc = subprocess.run(
        [
            str(LLAMA),
            "-m",
            str(GGUF),
            "-p",
            "<|user|>The<|end|><|assistant|>",
            "-n",
            str(N_TOK),
            "-t",
            "12",
            "--temp",
            "0",
            "--ignore-eos",
            "-ngl",
            "0",
            "-no-cnv",
            "--no-display-prompt",
            "-st",
        ],
        cwd=str(ROOT / "tools" / "llama.cpp"),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        print(out[-1500:], flush=True)
        return None
    m = re.search(r"Generation:\s*([\d.]+)\s*t/s", out)
    return float(m.group(1)) if m else None


def stats(vals: list[float]) -> dict:
    a = np.array(vals, dtype=np.float64)
    return {
        "runs": vals,
        "mean": round(float(a.mean()), 2),
        "std": round(float(a.std()), 2),
        "min": round(float(a.min()), 2),
        "max": round(float(a.max()), 2),
    }


def main() -> int:
    from benchmarks.thermal_utils import wait_until_cool

    mode = (sys.argv[1] if len(sys.argv) > 1 else "asdsl").lower()
    if mode not in ("asdsl", "llama"):
        print("Usage: variance_study_separate.py [asdsl|llama]")
        return 1

    subprocess.run(["taskkill", "/F", "/IM", "llama-cli.exe", "/T"], capture_output=True)
    time.sleep(2)
    startup = wait_until_cool(max_temp_c=65, initial_sleep_s=60)

    runs: list[float] = []
    for i in range(N_RUNS):
        print(f"\n--- {mode} run {i + 1}/{N_RUNS} ---", flush=True)
        wait_until_cool(max_temp_c=65, initial_sleep_s=0, max_wait_s=300)
        t = run_asdsl() if mode == "asdsl" else run_llama()
        if t is None:
            print("  FAILED", flush=True)
            return 1
        print(f"  {t:.2f} tok/s", flush=True)
        runs.append(t)
        if i + 1 < N_RUNS:
            time.sleep(COOL_S)

    s = stats(runs)
    result = {
        "mode": mode,
        "group_size": int(os.environ.get("ASDSL_GROUP_SIZE", "32")),
        "thermal_guard": startup,
        "stats": s,
    }
    out = ROOT / "benchmarks" / "results" / f"phase28_variance_{mode}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n{mode}: {s['mean']} ± {s['std']}  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
