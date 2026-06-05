"""Time each WeightStore / UnifiedEngine load step (find slow or stuck phases)."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

os.environ.setdefault("ASDSL_USE_UNIFIED", "1")
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "12"))


def _step(label: str) -> float:
    print(f"{label}...", flush=True)
    return time.perf_counter()


def main() -> int:
    t_all = time.perf_counter()
    print("=== diagnose_load ===", flush=True)
    print(f"  PHI4_NO_WEIGHT_CACHE={os.environ.get('PHI4_NO_WEIGHT_CACHE', '')!r}", flush=True)
    print(f"  PHI4_NO_PREQ_CACHE={os.environ.get('PHI4_NO_PREQ_CACHE', '')!r}", flush=True)
    cache_dir = ROOT / "models" / "phi4_weight_cache"
    if cache_dir.is_dir():
        files = list(cache_dir.glob("*"))
        print(f"  cache_dir: {len(files)} file(s)", flush=True)
        for f in sorted(files)[:8]:
            print(f"    {f.name} ({f.stat().st_size / 1e6:.1f} MB)", flush=True)
    else:
        print("  cache_dir: MISSING (first load will quantize ~15-30 min)", flush=True)

    t0 = _step("Step 1: import phi4_cpu_run")
    import phi4_cpu_run as runner  # noqa: E402

    print(f"  done: {time.perf_counter() - t0:.1f}s", flush=True)

    t0 = _step("Step 2: tokenizer")
    from transformers import AutoTokenizer

    model_dir = os.environ.get("ASDSL_MODEL_DIR", "").strip() or str(
        ROOT / "models" / "phi4-multimodal-instruct"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    print(f"  done: {time.perf_counter() - t0:.1f}s", flush=True)

    t0 = _step("Step 3: WeightStore()")
    store = runner.WeightStore(bits=4)
    store._use_unified = True
    print(f"  done: {time.perf_counter() - t0:.1f}s", flush=True)

    wpath = runner.weight_cache_path_for_store(store)
    ppath = runner.preq_cache_path_for_store(store)
    print(f"  weight_cache path exists: {wpath.is_file()} ({wpath.name})", flush=True)
    print(f"  preq_cache path exists: {ppath.is_file()} ({ppath.name})", flush=True)

    t0 = _step("Step 4: store.load()")
    store.load()
    print(
        f"  done: {time.perf_counter() - t0:.1f}s "
        f"(from_cache={getattr(store, '_loaded_from_cache', False)})",
        flush=True,
    )

    t0 = _step("Step 5: store.warm_cache()")
    store.warm_cache()
    print(f"  done: {time.perf_counter() - t0:.1f}s (preq_built={store._preq_built})", flush=True)

    t0 = _step("Step 6: build_unified_engine")
    from asdsl.inference.unified_bridge import build_unified_engine

    eng = build_unified_engine(store)
    print(f"  done: {time.perf_counter() - t0:.1f}s", flush=True)

    t0 = _step("Step 7: one forward_token")
    logits = eng.forward_token(100, 0)
    print(f"  done: {time.perf_counter() - t0:.1f}s vocab={len(logits)}", flush=True)

    total = time.perf_counter() - t_all
    print(f"ALL DONE in {total:.1f}s", flush=True)
    if total > 60 and not getattr(store, "_loaded_from_cache", False):
        print(
            "NOTE: Slow because weight cache was missing. "
            "Run: python benchmarks/build_caches.py once.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
