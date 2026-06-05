"""One-shot: HF load + warm_cache + persist weight & preq caches for fast benchmarks."""
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


def main() -> int:
    for var in ("PHI4_NO_WEIGHT_CACHE", "PHI4_NO_PREQ_CACHE"):
        if os.environ.get(var, "").strip().lower() in ("1", "true", "yes"):
            print(f"ERROR: unset {var} before building caches", flush=True)
            return 1

    import phi4_cpu_run as runner

    print("=== build_caches (first run may take 15-40 min) ===", flush=True)
    t0 = time.perf_counter()
    store = runner.WeightStore(bits=4)
    store._use_unified = True
    store.load()
    store.warm_cache()

    if runner._weight_cache_enabled() and not getattr(store, "_loaded_from_cache", False):
        wpath = runner.weight_cache_path_for_store(store)
        print(f"Saving weight cache to {wpath.name} ...", flush=True)
        runner.save_weight_store_cache(store, wpath)
        print(f"  saved ({wpath.stat().st_size / 1e9:.2f} GB)", flush=True)

    ppath = runner.preq_cache_path_for_store(store)
    if store._preq_built and ppath.is_file():
        print(f"Preq cache: {ppath.name} ({ppath.stat().st_size / 1e9:.2f} GB)", flush=True)

    from asdsl.inference.unified_bridge import build_unified_engine

    build_unified_engine(store)
    print(f"Done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
