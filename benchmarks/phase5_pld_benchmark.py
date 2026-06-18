"""Phase 5: PLD throughput, verify overhead, and copy-heavy secondary workloads."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "benchmarks" / "results" / "parity_manifest.json"
OUT_DIR = ROOT / "benchmarks" / "results"
THREADS = 12

COPY_HEAVY = {
    "repeat_paragraph": (
        "Summarize: "
        + "The quick brown fox jumps over the lazy dog. " * 24
    ),
    "code_continuation": (
        "Complete:\n"
        "def fib(n):\n"
        "    if n <= 1: return n\n"
        "    return fib(n-1) + fib(n-2)\n\n"
        "def fib(n):\n"
        "    if n <= 1: return n\n"
        "    return fib(n-1) + fib(n-2)\n\n"
        "def fib(n):\n"
    ),
}


def _load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _apply_c0_env(manifest: dict) -> None:
    for k, v in manifest["stacks"]["asdsl"]["env_required"].items():
        os.environ[k] = str(v)
    for k, v in manifest["configs"]["C0"]["env"].items():
        os.environ[k] = str(v)
    os.environ["OMP_NUM_THREADS"] = str(THREADS)
    os.environ["MKL_NUM_THREADS"] = str(THREADS)


def _encode_prompt(tokenizer, text: str, system: str = "") -> list[int]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": text})
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    return [int(x) for x in ids]


def _bench_decode(
    store,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    use_pld: bool,
    warmup: int = 1,
) -> dict:
    from asdsl.inference.unified_bridge import (
        greedy_generate,
        pld_generate_timed,
    )

    fn = pld_generate_timed if use_pld else None
    for _ in range(warmup):
        if use_pld:
            pld_generate_timed(store, prompt_ids, max(8, max_new_tokens // 4))
        else:
            greedy_generate(store, prompt_ids, max(8, max_new_tokens // 4))

    if use_pld:
        res = pld_generate_timed(store, prompt_ids, max_new_tokens)
        return {
            "tokens_per_second": round(res.tokens_per_second, 3),
            "decode_tokens": res.decode_tokens,
            "decode_s": round(res.decode_s, 3),
            "mode": "pld",
        }

    t0 = time.perf_counter()
    out = greedy_generate(store, prompt_ids, max_new_tokens)
    decode_s = time.perf_counter() - t0
    n = max(0, len(out) - len(prompt_ids))
    tps = n / decode_s if decode_s > 0 else 0.0
    return {
        "tokens_per_second": round(tps, 3),
        "decode_tokens": n,
        "decode_s": round(decode_s, 3),
        "mode": "greedy",
    }


def run_phase5(
    *,
    max_new_tokens: int = 128,
    canonical_prompt: str = "The",
    skip_verify_probe: bool = False,
) -> dict:
    manifest = _load_manifest()
    _apply_c0_env(manifest)

    pytest_unified = __import__("pytest", fromlist=["importorskip"])
    pytest_unified.importorskip("asdsl.kernels._native_unified")

    from asdsl.inference.unified_bridge import measure_verify_overhead
    from experiments.phi4_cpu_run import WeightStore, set_thread_count
    from transformers import AutoTokenizer

    cache = ROOT / "models" / "phi4_weight_cache"
    if not cache.is_dir():
        raise SystemExit(f"missing weight cache: {cache}")

    set_thread_count(THREADS)
    store = WeightStore(bits=4, group_size=32, enable_qcsd=False)
    print("Loading weights ...", flush=True)
    store.load()
    store.warm_cache()

    tok = AutoTokenizer.from_pretrained(
        manifest["model"]["hf_checkpoint"], trust_remote_code=True
    )
    system = manifest["prompt"].get("asdsl_system_prompt", "")

    doc: dict = {
        "phase": 5,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": "C0",
        "threads": THREADS,
        "max_new_tokens": max_new_tokens,
        "workloads": {},
    }

    if not skip_verify_probe:
        print("Measuring verify overhead ...", flush=True)
        doc["verify_overhead"] = measure_verify_overhead(store)

    canon_ids = _encode_prompt(tok, canonical_prompt, system)
    print(f"Canonical prompt ({canonical_prompt!r}) ...", flush=True)
    ar = _bench_decode(store, canon_ids, max_new_tokens=max_new_tokens, use_pld=False)
    pld = _bench_decode(store, canon_ids, max_new_tokens=max_new_tokens, use_pld=True)
    doc["workloads"]["canonical_the"] = {
        "prompt": canonical_prompt,
        "greedy": ar,
        "pld": pld,
        "speedup": round(pld["tokens_per_second"] / ar["tokens_per_second"], 3)
        if ar["tokens_per_second"] > 0
        else None,
    }

    copy_speedups: list[float] = []
    for name, text in COPY_HEAVY.items():
        print(f"Copy-heavy: {name} ...", flush=True)
        ids = _encode_prompt(tok, text, system)
        ar_w = _bench_decode(store, ids, max_new_tokens=max_new_tokens, use_pld=False)
        pld_w = _bench_decode(store, ids, max_new_tokens=max_new_tokens, use_pld=True)
        sp = (
            pld_w["tokens_per_second"] / ar_w["tokens_per_second"]
            if ar_w["tokens_per_second"] > 0
            else 0.0
        )
        copy_speedups.append(sp)
        doc["workloads"][name] = {
            "prompt_chars": len(text),
            "greedy": ar_w,
            "pld": pld_w,
            "speedup": round(sp, 3),
        }

    doc["gates"] = {
        "canonical_regression_pct": round(
            100.0
            * (pld["tokens_per_second"] - ar["tokens_per_second"])
            / ar["tokens_per_second"],
            2,
        )
        if ar["tokens_per_second"] > 0
        else None,
        "copy_heavy_min_speedup": round(min(copy_speedups), 3) if copy_speedups else None,
        "copy_heavy_mean_speedup": round(float(sum(copy_speedups) / len(copy_speedups)), 3)
        if copy_speedups
        else None,
        "verify_ratio_k4": doc.get("verify_overhead", {}).get("ratio_k4"),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "phase5_pld.json"
    out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(json.dumps(doc["gates"], indent=2))
    print(f"Wrote {out_path}")
    return doc


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--prompt", default="The")
    p.add_argument("--skip-verify-probe", action="store_true")
    args = p.parse_args()
    run_phase5(
        max_new_tokens=args.max_new_tokens,
        canonical_prompt=args.prompt,
        skip_verify_probe=args.skip_verify_probe,
    )


if __name__ == "__main__":
    main()
