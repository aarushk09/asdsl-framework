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

# Parent-shell values win over manifest for exploratory A/B (Phase 6+).
EXPLORATORY_ENV_KEYS = (
    "ASDSL_AFFINITY",
    "ASDSL_GEMV_CHUNK_DIV",
    "ASDSL_LARGE_PAGES",
    "ASDSL_C01",
    "ASDSL_C03",
    "ASDSL_USE_PLD",
    "ASDSL_PERSISTENT_POOL",
    "ASDSL_CPP_GENERATE",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
)


def threads_for_config(cfg_id: str, manifest: dict) -> int:
    if os.environ.get("OMP_NUM_THREADS"):
        return int(os.environ["OMP_NUM_THREADS"])
    key = resolve_config_key(cfg_id, manifest) or cfg_id
    spec = manifest.get("configs", {}).get(key, {})
    return int(
        spec.get("threads_cli", manifest["stacks"]["asdsl"].get("threads_cli", THREADS))
    )


def normalize_config_id(raw: str) -> str:
    """Preserve hyphenated exploratory IDs (e.g. C6-smt, C0.1)."""
    raw = raw.strip()
    if not raw:
        return raw
    if "-" in raw:
        head, tail = raw.split("-", 1)
        return f"{head.upper()}-{tail.lower()}"
    return raw.upper()


def resolve_config_key(cfg_id: str, manifest: dict) -> str | None:
    """Return manifest config key if present (case-insensitive for hyphenated ids)."""
    configs = manifest.get("configs", {})
    if cfg_id in configs:
        return cfg_id
    lower = cfg_id.lower()
    for key in configs:
        if key.lower() == lower:
            return key
    return None


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


def verify_native_extensions() -> dict[str, str | bool]:
    """Preflight: native .pyd modules must load from this repo (parity subprocess path)."""
    status: dict[str, str | bool] = {"ok": True}
    for mod in ("asdsl.kernels._native_unified", "asdsl.kernels._native_gemv"):
        try:
            __import__(mod)
            status[mod] = True
        except ImportError as exc:
            status["ok"] = False
            status[mod] = str(exc)
    return status


def env_for_config(cfg_id: str, manifest: dict) -> dict[str, str]:
    key = resolve_config_key(cfg_id, manifest) or cfg_id
    # Manifest defaults, then per-config env, then parent-shell exploratory overrides.
    e = {**os.environ, **manifest["stacks"]["asdsl"]["env_required"]}
    threads = threads_for_config(key, manifest)
    e["OMP_NUM_THREADS"] = str(threads)
    e["MKL_NUM_THREADS"] = str(threads)
    e["PYTHONIOENCODING"] = "utf-8"
    root_s = str(ROOT)
    existing = e.get("PYTHONPATH", "")
    if root_s not in existing.split(os.pathsep):
        e["PYTHONPATH"] = root_s + (os.pathsep + existing if existing else "")
    if key in manifest["configs"]:
        spec = manifest["configs"][key]
        for k, v in (spec.get("env") or {}).items():
            e[k] = str(v)
    for k in EXPLORATORY_ENV_KEYS:
        if k in os.environ:
            e[k] = os.environ[k]
    if "OMP_NUM_THREADS" in os.environ:
        e["OMP_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]
        e["MKL_NUM_THREADS"] = os.environ.get("MKL_NUM_THREADS", os.environ["OMP_NUM_THREADS"])
    return e


def effective_env_snapshot(cfg_id: str, manifest: dict) -> dict[str, str]:
    """Resolved env for a config (for JSON output; exploratory shell keys included)."""
    env = env_for_config(cfg_id, manifest)
    keys = sorted(
        set(manifest["stacks"]["asdsl"]["env_required"])
        | set(EXPLORATORY_ENV_KEYS)
        | {"ASDSL_GROUP_SIZE"}
    )
    return {k: env[k] for k in keys if k in env}


def run_asdsl_once(cfg_id: str, manifest: dict) -> dict:
    key = resolve_config_key(cfg_id, manifest)
    if not key:
        return {"ok": False, "error": f"unknown config {cfg_id!r}"}
    spec = manifest["configs"][key]
    env = env_for_config(key, manifest)
    threads = int(env["OMP_NUM_THREADS"])
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
        str(threads),
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
    inference_mode = None
    m_inf = re.search(r"Inference:\s*(.+)", out)
    if m_inf:
        inference_mode = m_inf.group(1).strip()
    unified_active = bool(
        inference_mode and "UnifiedEngine" in inference_mode
    )
    if decode_2_n is None and full_1_n is not None:
        decode_2_n = full_1_n
    acceptance_rate = None
    draft_tokens = None
    verify_ms = None
    draft_ms = None
    speculative_cycles = None
    m_spec = re.search(
        r"acceptance_rate=([\d.]+)\s+draft_tokens=(\d+)\s+verify_ms=([\d.]+)\s+"
        r"draft_ms=([\d.]+)\s+speculative_cycles=(\d+)",
        out,
    )
    if m_spec:
        acceptance_rate = float(m_spec.group(1))
        draft_tokens = int(m_spec.group(2))
        verify_ms = float(m_spec.group(3))
        draft_ms = float(m_spec.group(4))
        speculative_cycles = int(m_spec.group(5))
    return {
        "ok": True,
        "decode_2_N": decode_2_n,
        "full_1_N": full_1_n,
        "inference_mode": inference_mode,
        "unified_active": unified_active,
        "acceptance_rate": acceptance_rate,
        "draft_tokens": draft_tokens,
        "verify_ms": verify_ms,
        "draft_ms": draft_ms,
        "speculative_cycles": speculative_cycles,
        "threads_requested": threads,
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
        key = resolve_config_key(cfg, manifest)
        if cfg == "L0" or not key:
            if cfg != "L0" and cfg not in ("",):
                print(f"WARNING: unknown config {cfg!r} — skipped", flush=True)
            continue
        if manifest["configs"][key].get("stack") == "llama":
            continue
        runs_2n: list[float] = []
        runs_full: list[float] = []
        print(f"\n=== ASDSL session: {key} ===", flush=True)
        for i in range(N_RUNS):
            wait_until_cool(max_temp_c=65, initial_sleep_s=0, max_wait_s=300)
            print(f"  run {i + 1}/{N_RUNS}", flush=True)
            r = run_asdsl_once(key, manifest)
            if not r.get("ok"):
                print(r.get("error", "")[-500:])
                return {"error": f"{cfg} run failed", "detail": r}
            if r.get("decode_2_N") is not None:
                runs_2n.append(r["decode_2_N"])
            if r.get("full_1_N") is not None:
                runs_full.append(r["full_1_N"])
            print(
                f"    decode_2_N={r.get('decode_2_N')} full_1_N={r.get('full_1_N')} "
                f"unified={r.get('unified_active')} mode={r.get('inference_mode')!r}",
                flush=True,
            )
            if i + 1 < N_RUNS:
                time.sleep(COOL_S)
        s2 = stats(runs_2n) if runs_2n else {}
        sf = stats(runs_full) if runs_full else {}
        results[key] = {
            "label": manifest["configs"][key].get("label"),
            "effective_env": effective_env_snapshot(key, manifest),
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
        help="Comma-separated config IDs (C0,C1,C2,C3,C4,L0)",
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
    ap.add_argument(
        "--skip-kernel-preflight",
        action="store_true",
        help="Skip gate_up preq2 ±15%% reference check",
    )
    args = ap.parse_args()
    N_RUNS = max(1, int(args.runs))
    COOL_S = max(0, int(args.cooldown))
    manifest = load_manifest()
    cfg_ids = [normalize_config_id(c) for c in args.config.split(",") if c.strip()]
    asdsl_cfgs = [c for c in cfg_ids if c != "L0"]
    want_llama = "L0" in cfg_ids and not args.asdsl_only
    want_asdsl = bool(asdsl_cfgs) and not args.llama_only

    out_doc: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(MANIFEST_PATH.relative_to(ROOT)),
        "config_ids_requested": cfg_ids,
        "threads": THREADS,
        "n_runs": N_RUNS,
        "gguf_sha256": sha256_file(GGUF),
    }

    if want_asdsl:
        from benchmarks.kernel_preflight import run_kernel_preflight

        preflight = run_kernel_preflight(skip=args.skip_kernel_preflight)
        out_doc["kernel_preflight"] = preflight
        if not preflight.get("ok"):
            print(f"ERROR: kernel preflight failed: {preflight.get('error')}", flush=True)
            args.output.write_text(json.dumps(out_doc, indent=2), encoding="utf-8")
            return 1
        if not preflight.get("skipped"):
            print(
                f"Kernel preflight: gate_up preq2 {preflight.get('measured_ms')} ms "
                f"({preflight.get('measured_gb_s')} GB/s)",
                flush=True,
            )
        native = verify_native_extensions()
        out_doc["native_extensions"] = native
        if not native.get("ok"):
            print(
                "ERROR: native extensions not importable; run: python setup.py build_ext --inplace",
                flush=True,
            )
            args.output.write_text(json.dumps(out_doc, indent=2), encoding="utf-8")
            return 1
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

    if "C4" in asdsl_cfgs and want_asdsl:
        spec_path = ROOT / "benchmarks" / "results" / "speculative_parity.json"
        c4_runs: list[dict] = []
        session = out_doc.get("session_asdsl", {})
        c4_stats = (session.get("configs") or {}).get("C4", {})
        for key in ("decode_2_N", "full_1_N"):
            block = c4_stats.get(key) or {}
            if block.get("runs"):
                c4_runs.extend(block["runs"])
        spec_doc = {
            "timestamp_utc": out_doc.get("timestamp_utc"),
            "config": "C4",
            "decode_2_N": c4_stats.get("decode_2_N"),
            "full_1_N": c4_stats.get("full_1_N"),
            "note": "Per-run acceptance_rate parsed from phi4_cpu_run speculative profile line",
        }
        spec_path.write_text(json.dumps(spec_doc, indent=2), encoding="utf-8")
        print(f"Wrote {spec_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
