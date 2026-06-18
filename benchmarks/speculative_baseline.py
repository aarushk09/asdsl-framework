"""Unified speculative decoding baseline reporter (C0 / QCSD / SWIFT / AHSD / SDQS)."""

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
sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "benchmarks" / "results" / "parity_manifest.json"
DEFAULT_PROMPT = "The"
DEFAULT_N_TOK = 64
THREADS = 12


def _load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _base_env(manifest: dict) -> dict[str, str]:
    env = os.environ.copy()
    env.update(manifest["stacks"]["asdsl"]["env_required"])
    env["OMP_NUM_THREADS"] = str(THREADS)
    env["MKL_NUM_THREADS"] = str(THREADS)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _parse_run_output(out: str) -> dict:
    metrics: dict[str, float | int | None] = {
        "decode_2_N": None,
        "full_1_N": None,
        "acceptance_rate": None,
        "draft_tokens": None,
        "verify_ms": None,
        "draft_ms": None,
        "speculative_cycles": None,
    }
    m2n = re.search(r"decode\s+([\d.]+)\s+tok/s\s+\(tokens\s+2-(\d+)\)", out)
    if m2n:
        metrics["decode_2_N"] = float(m2n.group(1))
    m_full = re.search(r"Generated\s*:\s*(\d+)\s+tokens\s*\|\s*([\d.]+)\s+tok/s", out)
    if m_full:
        metrics["full_1_N"] = float(m_full.group(2))
    m_spec = re.search(
        r"acceptance_rate=([\d.]+)\s+draft_tokens=(\d+)\s+verify_ms=([\d.]+)\s+"
        r"draft_ms=([\d.]+)\s+speculative_cycles=(\d+)",
        out,
    )
    if m_spec:
        metrics["acceptance_rate"] = float(m_spec.group(1))
        metrics["draft_tokens"] = int(m_spec.group(2))
        metrics["verify_ms"] = float(m_spec.group(3))
        metrics["draft_ms"] = float(m_spec.group(4))
        metrics["speculative_cycles"] = int(m_spec.group(5))
    m_acc = re.search(r"acceptance rate\s+([\d.]+)%", out, re.I)
    if m_acc and metrics["acceptance_rate"] is None:
        metrics["acceptance_rate"] = float(m_acc.group(1)) / 100.0
    return metrics


def _run_phi4(
    *,
    extra_args: list[str],
    extra_env: dict[str, str] | None = None,
    timeout: int = 1800,
) -> dict:
    manifest = _load_manifest()
    env = _base_env(manifest)
    if extra_env:
        env.update(extra_env)
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "phi4_cpu_run.py"),
        "--bits",
        "4",
        "--max-new-tokens",
        str(DEFAULT_N_TOK),
        "--prompt",
        DEFAULT_PROMPT,
        "--threads",
        str(THREADS),
        *extra_args,
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
    parsed = _parse_run_output(out)
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "wall_s": round(wall_s, 2),
        "metrics": parsed,
        "tail": out[-1500:] if not proc.returncode == 0 else "",
    }


def _run_swift_cpp(max_new_tokens: int, draft_k: int) -> dict:
    """Direct C++ SWIFT smoke via unified_bridge when weights are cached."""
    try:
        from asdsl.inference.unified_bridge import ahsd_generate, get_or_build_unified_engine
        from experiments.phi4_cpu_run import (
            NUM_LAYERS,
            WeightStore,
            set_thread_count,
            weight_cache_path_for_store,
        )
        from transformers import AutoTokenizer
    except ImportError as exc:
        return {"ok": False, "error": f"import failed: {exc}"}

    cache_dir = ROOT / "models" / "phi4_weight_cache"
    if not cache_dir.is_dir() or not any(cache_dir.glob("*.safetensors")):
        return {"ok": False, "error": "phi4 weight cache missing"}

    set_thread_count(THREADS)
    os.environ.setdefault("ASDSL_USE_UNIFIED", "1")
    store = WeightStore(bits=4, group_size=32, enable_qcsd=False)
    store.load()
    store.warm_cache()
    eng = get_or_build_unified_engine(store)
    eng.reset_session()
    tok = AutoTokenizer.from_pretrained(
        "microsoft/Phi-4-multimodal-instruct", trust_remote_code=True
    )
    prompt_ids = tok.encode(DEFAULT_PROMPT, add_special_tokens=True)
    t0 = time.perf_counter()
    tokens = eng.generate_swift(prompt_ids, int(max_new_tokens), int(draft_k))
    wall_s = time.perf_counter() - t0
    n_gen = max(0, len(tokens) - len(prompt_ids))
    tps = n_gen / wall_s if wall_s > 0 else 0.0
    return {
        "ok": True,
        "metrics": {
            "full_1_N": float(tps),
            "decode_2_N": float(tps),
            "draft_k": draft_k,
            "generated_tokens": n_gen,
        },
        "wall_s": round(wall_s, 2),
    }


def collect_baselines(
    *,
    include_qcsd: bool = True,
    include_ahsd: bool = True,
    include_sdqs: bool = False,
    include_swift: bool = True,
    draft_k: int = 3,
) -> dict:
    manifest = _load_manifest()
    doc: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_note": manifest.get("hardware", {}),
        "prompt": DEFAULT_PROMPT,
        "max_new_tokens": DEFAULT_N_TOK,
        "threads": THREADS,
        "profiles": {},
    }

    doc["profiles"]["C0_AR"] = _run_phi4(extra_args=[], extra_env={})

    if include_qcsd:
        doc["profiles"]["QCSD"] = _run_phi4(
            extra_args=["--qcsd", "--draft-k", str(draft_k)],
            extra_env={"ASDSL_SPECULATIVE_PROFILE": "1"},
        )

    if include_ahsd:
        doc["profiles"]["AHSD"] = _run_phi4(
            extra_args=[],
            extra_env={
                "ASDSL_USE_AHSD": "1",
                "ASDSL_AHSD_DRAFT_K": str(draft_k),
                "ASDSL_SPECULATIVE_PROFILE": "1",
            },
        )

    if include_sdqs:
        doc["profiles"]["SDQS"] = _run_phi4(
            extra_args=["--qcsd"],
            extra_env={
                "ASDSL_USE_SDQS": "1",
                "ASDSL_AHSD_DRAFT_K": str(draft_k),
                "ASDSL_SPECULATIVE_PROFILE": "1",
            },
        )

    if include_swift:
        doc["profiles"]["SWIFT_CPP"] = _run_swift_cpp(DEFAULT_N_TOK, draft_k)

    c0 = (doc["profiles"].get("C0_AR") or {}).get("metrics", {}).get("decode_2_N")
    for name, prof in doc["profiles"].items():
        if name == "C0_AR" or not prof.get("ok"):
            continue
        m = prof.get("metrics", {})
        spd = m.get("decode_2_N") or m.get("full_1_N")
        if c0 and spd:
            m["speedup_vs_C0"] = round(float(spd) / float(c0), 3)

    return doc


def main() -> int:
    ap = argparse.ArgumentParser(description="Speculative decoding baseline reporter")
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("--draft-k", type=int, default=3)
    ap.add_argument("--no-qcsd", action="store_true")
    ap.add_argument("--no-ahsd", action="store_true")
    ap.add_argument("--sdqs", action="store_true")
    ap.add_argument("--no-swift", action="store_true")
    ap.add_argument(
        "--phase",
        choices=("0", "A", "all"),
        default="all",
        help="Phase 0 = C0+QCSD+SWIFT; Phase A adds gate probes",
    )
    args = ap.parse_args()

    doc = collect_baselines(
        include_qcsd=not args.no_qcsd and args.phase in ("0", "all"),
        include_ahsd=not args.no_ahsd and args.phase in ("all",),
        include_sdqs=args.sdqs,
        include_swift=not args.no_swift and args.phase in ("0", "all"),
        draft_k=args.draft_k,
    )

    if args.phase == "A":
        from asdsl.profiler import measure_dual_stream_bandwidth

        try:
            dual = measure_dual_stream_bandwidth(size_a_mb=512, size_b_mb=256, runs=4)
            doc["gates"] = {
                "dual_stream": {
                    "combined_bandwidth_gb_s": dual.combined_bandwidth_gb_s,
                    "retention_a_pct": dual.retention_a_pct,
                    "retention_b_pct": dual.retention_b_pct,
                    "gate_a2_pass": dual.combined_bandwidth_gb_s >= 35.0
                    and dual.retention_a_pct >= 80.0
                    and dual.retention_b_pct >= 80.0,
                }
            }
        except RuntimeError as exc:
            doc["gates"] = {"dual_stream": {"error": str(exc), "gate_a2_pass": False}}

    out_path = args.output
    if out_path is None:
        if args.phase == "0":
            out_path = ROOT / "benchmarks" / "results" / "speculative_phase0_baseline.json"
        elif args.phase == "A":
            out_path = ROOT / "benchmarks" / "results" / "speculative_phaseA_gates.json"
        else:
            out_path = ROOT / "benchmarks" / "results" / "speculative_baseline_latest.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    failed = [k for k, v in doc["profiles"].items() if not v.get("ok")]
    if failed:
        print(f"WARNING: failed profiles: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
