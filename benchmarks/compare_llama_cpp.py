"""Compare ASDSL decode vs llama.cpp on the same GGUF checkpoint (12 threads).

Downloads GGUF if missing. Records harmonized metrics and fails on thread mismatch.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GGUF_DIR = ROOT / "models" / "llama_cpp"
GGUF = GGUF_DIR / "phi4-mm-Q4_K_M.gguf"
GGUF_URL = (
    "https://huggingface.co/ShayanCyan/phi4-multimodal-quantisized-gguf/"
    "resolve/main/phi4-mm-Q4_K_M.gguf"
)
LLAMA_CLI = ROOT / "tools" / "llama.cpp" / "llama-cli.exe"
LLAMA_BENCH = ROOT / "tools" / "llama.cpp" / "llama-bench.exe"
OUT = ROOT / "benchmarks" / "results" / "llama_cpp_comparison.json"
CHAT_PROMPT = "<|user|>The<|end|><|assistant|>"
N_TOK = 128
DEFAULT_THREADS = 12


def ensure_gguf() -> None:
    if GGUF.is_file():
        return
    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {GGUF.name} (~2.5 GB) ...", flush=True)
    urllib.request.urlretrieve(GGUF_URL, GGUF)


def run_asdsl(threads: int) -> dict:
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
            "ASDSL_USE_UNIFIED": "1",
            "ASDSL_IGNORE_EOS": "1",
            "ASDSL_FUSED_GEMV": "1",
            "ASDSL_AFFINITY": "spread",
            "ASDSL_PREQ_G4FUSED": "0",
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
            "--system-prompt",
            "",
            "--threads",
            str(threads),
        ],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1200,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(out[-3000:])
    m = re.search(r"decode\s+([\d.]+)\s+tok/s", out)
    decode = float(m.group(1)) if m else None
    return {"decode_2_N_tok_s": decode, "threads": threads}


def run_llama_cli(threads: int) -> dict:
    cmd = [
        str(LLAMA_CLI),
        "-m",
        str(GGUF),
        "-p",
        CHAT_PROMPT,
        "-n",
        str(N_TOK),
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
        timeout=600,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(out[-3000:])
    m = re.search(r"Generation:\s*([\d.]+)\s*t/s", out)
    footer = float(m.group(1)) if m else None
    reported = threads
    tm = re.search(r"threads\s*:\s*(\d+)", out, re.I)
    if tm:
        reported = int(tm.group(1))
    return {
        "cli_footer_tok_s": footer,
        "threads_requested": threads,
        "threads_reported": reported,
    }


def run_llama_bench(threads: int) -> dict:
    if not LLAMA_BENCH.is_file():
        return {"ok": False, "note": "llama-bench not built"}
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
    return {"ok": proc.returncode == 0, "tg128_tok_s": tg}


def check_thread_parity(asdsl_threads: int, llama: dict) -> None:
    req = llama.get("threads_requested", asdsl_threads)
    rep = llama.get("threads_reported")
    if rep is None:
        return
    if rep != asdsl_threads:
        msg = (
            f"THREAD MISMATCH: llama-cli reported {rep} threads but ASDSL used "
            f"{asdsl_threads} (requested -t {req}). "
            "Results are not comparable — fix llama build or flags."
        )
        print(f"ERROR: {msg}", flush=True)
        raise SystemExit(2)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    args = ap.parse_args()
    threads = args.threads

    subprocess.run(["taskkill", "/F", "/IM", "llama-cli.exe", "/T"], capture_output=True)
    time.sleep(1)
    ensure_gguf()

    if not LLAMA_CLI.is_file():
        print(f"ERROR: build llama-cli at {LLAMA_CLI}", flush=True)
        return 1

    print(f"=== compare_llama_cpp @ {threads} threads ===", flush=True)
    asdsl = run_asdsl(threads)
    llama = run_llama_cli(threads)
    check_thread_parity(threads, llama)
    bench = run_llama_bench(threads)

    ratio = None
    if asdsl.get("decode_2_N_tok_s") and llama.get("cli_footer_tok_s"):
        ratio = round(
            100.0 * asdsl["decode_2_N_tok_s"] / llama["cli_footer_tok_s"],
            1,
        )

    doc = {
        "threads": threads,
        "n_tokens": N_TOK,
        "prompt": CHAT_PROMPT,
        "asdsl_hf_preq": asdsl,
        "llama_gguf": llama,
        "llama_bench": bench,
        "asdsl_pct_of_llama_cli": ratio,
        "note": (
            "HF preq ASDSL vs llama Q4_K_M GGUF; for fair parity use "
            "parity_benchmark.py C1 vs L0 with ASDSL_USE_Q4KM_GGUF=1"
        ),
        "thread_check": "pass" if llama.get("threads_reported") == threads else "fail",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(json.dumps(doc, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
