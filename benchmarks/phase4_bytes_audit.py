"""Phase 4: bytes/token model for C0 vs C0.1 (g128 gate_up/down/lm_head)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

NUM_LAYERS = 32
HIDDEN = 3072
INTER = 8192
VOCAB = 200064


def _bytes_per_token_c0() -> dict:
    gs = 32
    block = 20
    qkv = NUM_LAYERS * (5120 * (HIDDEN // gs) * block)
    o = NUM_LAYERS * (HIDDEN * (HIDDEN // gs) * block)
    gu = NUM_LAYERS * (2 * INTER * (HIDDEN // gs) * block)
    dn = NUM_LAYERS * (HIDDEN * (INTER // gs) * block)
    lm = VOCAB * (HIDDEN // gs) * block
    body = qkv + o + gu + dn
    total = body + lm
    return {
        "config": "C0",
        "group_size_body": gs,
        "lm_head_gs": gs,
        "body_bytes": body,
        "lm_head_bytes": lm,
        "total_bytes": total,
        "total_gb": total / 1e9,
    }


def _bytes_per_token_c01() -> dict:
    gs_body = 32
    block32 = 20
    gs128 = 128
    block128 = 66
    qkv = NUM_LAYERS * (5120 * (HIDDEN // gs_body) * block32)
    o = NUM_LAYERS * (HIDDEN * (HIDDEN // gs_body) * block32)
    gu = NUM_LAYERS * (2 * INTER * (HIDDEN // gs128) * block128)
    dn = NUM_LAYERS * (HIDDEN * (INTER // gs128) * block128)
    lm = VOCAB * (HIDDEN // gs128) * block128
    body = qkv + o + gu + dn
    total = body + lm
    c0 = _bytes_per_token_c0()
    return {
        "config": "C0.1",
        "group_size_body": gs_body,
        "lm_head_gs": gs128,
        "body_bytes": body,
        "lm_head_bytes": lm,
        "total_bytes": total,
        "total_gb": total / 1e9,
        "reduction_vs_c0_pct": round(100.0 * (1.0 - total / c0["total_bytes"]), 2),
    }


def main() -> int:
    c0 = _bytes_per_token_c0()
    c01 = _bytes_per_token_c01()
    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": 4,
        "C0": c0,
        "C0.1": c01,
        "target_reduction_pct": 12.0,
        "gate_reduction_met": c01["reduction_vs_c0_pct"] >= 10.0,
    }
    path = ROOT / "benchmarks" / "results" / "phase4_bytes_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
