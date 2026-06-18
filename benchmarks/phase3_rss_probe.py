"""Phase 3: peak RSS during decode and engine-init timing."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return -1.0


def main() -> int:
    os.environ.setdefault("ASDSL_AFFINITY", "physical")
    os.environ.setdefault("ASDSL_CHUNKED_GEMV", "1")
    os.environ.setdefault("ASDSL_PREQ2", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "12")

    from experiments.phi4_cpu_run import WeightStore

    out: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": 3,
        "env": {
            "ASDSL_AFFINITY": os.environ.get("ASDSL_AFFINITY"),
            "ASDSL_CHUNKED_GEMV": os.environ.get("ASDSL_CHUNKED_GEMV"),
            "ASDSL_PREQ2": os.environ.get("ASDSL_PREQ2"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        },
    }

    rss0 = _rss_mb()
    t0 = time.perf_counter()
    store = WeightStore(bits=4)
    store.load()
    store._use_unified = True
    from asdsl.inference.unified_bridge import get_or_build_unified_engine

    eng = get_or_build_unified_engine(store)
    init_s = time.perf_counter() - t0
    rss_after_init = _rss_mb()

    eng.reset_session()
    prompt = [464]  # "The" token id in Phi-4 chat template context
    for i, tok in enumerate(prompt):
        eng.forward_token(int(tok), i)
    for i in range(32):
        eng.forward_token_argmax(464, len(prompt) + i)
    rss_peak_decode = _rss_mb()

    out["rss_mb"] = {
        "before_load": rss0,
        "after_engine_init": rss_after_init,
        "after_32_decode_tokens": rss_peak_decode,
    }
    out["engine_init_s"] = init_s
    out["gate_rss_le_3gb"] = rss_peak_decode > 0 and rss_peak_decode <= 3072.0
    out["gate_init_le_5s"] = init_s <= 5.0

    path = ROOT / "benchmarks" / "results" / "phase3_rss_probe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
