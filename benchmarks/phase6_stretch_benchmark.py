"""Phase 6: A/B each stretch flag vs C0 baseline (large pages, chunk div, SMT)."""

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

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "benchmarks" / "results" / "parity_manifest.json"
OUT_PATH = ROOT / "benchmarks" / "results" / "phase6_stretch.json"
CHUNK_CACHE = ROOT / "benchmarks" / "results" / "phase6_chunk_cache.json"
THREADS_DEFAULT = 12


def _load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _base_env(manifest: dict, threads: int) -> dict[str, str]:
    env = {**os.environ, **manifest["stacks"]["asdsl"]["env_required"]}
    env.update(manifest["configs"]["C0"]["env"])
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env["PYTHONIOENCODING"] = "utf-8"
    root_s = str(ROOT)
    existing = env.get("PYTHONPATH", "")
    if root_s not in existing.split(os.pathsep):
        env["PYTHONPATH"] = root_s + (os.pathsep + existing if existing else "")
    return env


def _parse_decode(out: str) -> float | None:
    m2n = re.search(r"decode\s+([\d.]+)\s+tok/s\s+\(tokens\s+2-(\d+)\)", out)
    if m2n:
        return float(m2n.group(1))
    m_full = re.search(r"Generated\s*:\s*(\d+)\s+tokens\s*\|\s*([\d.]+)\s+tok/s", out)
    if m_full:
        return float(m_full.group(2))
    return None


def _run_profile(
    label: str,
    extra_env: dict[str, str],
    *,
    manifest: dict,
    max_new_tokens: int,
    threads: int,
    timeout: int,
) -> dict:
    env = _base_env(manifest, threads)
    env.update(extra_env)
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "phi4_cpu_run.py"),
        "--bits",
        "4",
        "--max-new-tokens",
        str(max_new_tokens),
        "--prompt",
        manifest["prompt"]["asdsl_user_prompt"],
        "--system-prompt",
        manifest["prompt"]["asdsl_system_prompt"],
        "--threads",
        str(threads),
    ]
    t0 = time.perf_counter()
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
    wall_s = time.perf_counter() - t0
    out = (proc.stdout or "") + (proc.stderr or "")
    tps = _parse_decode(out)
    mode = None
    m_inf = re.search(r"Inference:\s*(.+)", out)
    if m_inf:
        mode = m_inf.group(1).strip()
    return {
        "label": label,
        "ok": proc.returncode == 0 and tps is not None,
        "decode_tok_s": tps,
        "wall_s": round(wall_s, 2),
        "threads": threads,
        "env": extra_env,
        "inference_mode": mode,
        "tail": out[-1200:] if proc.returncode != 0 or tps is None else "",
    }


def _pct_delta(base: float | None, val: float | None) -> float | None:
    if base is None or val is None or base <= 0:
        return None
    return round(100.0 * (val - base) / base, 2)


def run_phase6(
    *,
    max_new_tokens: int = 128,
    skip_autotune: bool = False,
    quick: bool = False,
) -> dict:
    manifest = _load_manifest()
    timeout = 3600 if not quick else 1800

    chunk_div = 4
    if not skip_autotune:
        subprocess.run(
            [sys.executable, str(ROOT / "benchmarks" / "phase6_chunk_autotune.py")],
            cwd=str(ROOT),
            check=False,
        )
    if CHUNK_CACHE.is_file():
        chunk_div = int(json.loads(CHUNK_CACHE.read_text(encoding="utf-8")).get("best_div", 4))

    profiles: list[dict] = []

    baseline_env: dict[str, str] = {}
    profiles.append(
        _run_profile(
            "C0_baseline",
            baseline_env,
            manifest=manifest,
            max_new_tokens=max_new_tokens,
            threads=THREADS_DEFAULT,
            timeout=timeout,
        )
    )
    base_tps = profiles[0].get("decode_tok_s")

    candidates = [
        ("large_pages", {"ASDSL_LARGE_PAGES": "1"}),
        ("chunk_div", {"ASDSL_GEMV_CHUNK_DIV": str(chunk_div)}),
        (
            "large_pages+chunk_div",
            {"ASDSL_LARGE_PAGES": "1", "ASDSL_GEMV_CHUNK_DIV": str(chunk_div)},
        ),
        ("smt_16t", {"ASDSL_AFFINITY": "smt"}),
    ]

    for label, extra in candidates:
        threads = 16 if label == "smt_16t" else THREADS_DEFAULT
        print(f"Profile {label} ...", flush=True)
        profiles.append(
            _run_profile(
                label,
                extra,
                manifest=manifest,
                max_new_tokens=max_new_tokens,
                threads=threads,
                timeout=timeout,
            )
        )

    winners: list[str] = []
    for p in profiles[1:]:
        delta = _pct_delta(base_tps, p.get("decode_tok_s"))
        p["delta_pct_vs_baseline"] = delta
        if delta is not None and delta >= 1.0:
            winners.append(p["label"])

    doc = {
        "phase": 6,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "max_new_tokens": max_new_tokens,
        "chunk_div_autotuned": chunk_div,
        "baseline_tok_s": base_tps,
        "profiles": profiles,
        "winners_ge_1pct": winners,
        "gates": {
            "target_c0_tok_s": 16.0,
            "baseline_meets_target": bool(base_tps and base_tps >= 16.0),
            "any_flag_ge_1pct": bool(winners),
        },
        "deferred": {
            "clang_cl_libomp": "manual A/B — rebuild with clang-cl + libomp, compare C0",
            "lm_head_screening": "time-boxed spike not pursued — needs Q2 companion + bound audit",
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(json.dumps(doc["gates"], indent=2))
    print(f"Winners (≥1%): {winners or 'none'}")
    print(f"Wrote {OUT_PATH}")
    return doc


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--skip-autotune", action="store_true")
    p.add_argument("--quick", action="store_true", help="shorter timeout only")
    args = p.parse_args()
    run_phase6(
        max_new_tokens=args.max_new_tokens,
        skip_autotune=args.skip_autotune,
        quick=args.quick,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
