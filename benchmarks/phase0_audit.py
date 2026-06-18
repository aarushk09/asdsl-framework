"""Phase 0: topology, AVX-VNNI, bandwidth ceiling, byte accounting."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    from asdsl.profiler import bytes_per_token_breakdown, measure_stream_triad_bandwidth

    out: dict = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "phase": 0}

    try:
        from asdsl.kernels import _native_gemv as ng

        out["cpu_topology"] = dict(ng.get_cpu_topology())
        out["avx2"] = bool(ng.check_avx2())
        out["fma"] = bool(ng.check_fma())
        out["avx_vnni"] = bool(ng.check_avx_vnni()) if hasattr(ng, "check_avx_vnni") else None
        out["vnni_any"] = bool(ng.check_vnni())
    except Exception as exc:
        out["native_gemv_error"] = str(exc)

    triad = measure_stream_triad_bandwidth(array_mb=512, runs=8)
    out["stream_triad_gb_s"] = triad.bandwidth_gb_s
    out["stream_triad_impl"] = triad.implementation

    model_bytes = int(2.41e9)
    bpt = bytes_per_token_breakdown(
        model_file_size_bytes=model_bytes,
        num_layers=32,
        num_kv_heads=8,
        head_dim=128,
        t_context=128,
        kv_bytes_per_element=1,
    )
    out["bytes_per_token"] = {
        "weight_bytes": bpt.weight_bytes,
        "kv_bytes": bpt.kv_bytes,
        "total_bytes": bpt.total_bytes,
        "total_gb": bpt.total_gb,
    }

    preq_gb = 2.01
    lm_head_gb = 200064 * 3072 * 20 / 32 / 1e9
    out["bytes_per_token_estimated_gb"] = {
        "body_preq": preq_gb,
        "lm_head_q4_32": round(lm_head_gb, 4),
        "total": round(preq_gb + lm_head_gb, 4),
    }

    membw_path = ROOT / "benchmarks" / "results" / "membw_probe.json"
    if membw_path.is_file():
        out["membw_probe"] = json.loads(membw_path.read_text(encoding="utf-8"))
        out["bw_ceiling_gb_s"] = out["membw_probe"].get("bw_ceiling_gb_s")

    path = ROOT / "benchmarks" / "results" / "phase0_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
