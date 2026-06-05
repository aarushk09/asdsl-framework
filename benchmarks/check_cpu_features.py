#!/usr/bin/env python3
"""Report CPU SIMD features relevant to ASDSL kernels."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "benchmarks" / "results" / "cpu_features.json"


def main() -> None:
    info: dict = {}
    try:
        from asdsl.kernels import _native_gemv as ng

        info["avx2"] = bool(ng.check_avx2())
        info["fma"] = bool(ng.check_fma())
        info["avx512"] = bool(ng.check_avx512()) if hasattr(ng, "check_avx512") else None
        info["vnni"] = bool(ng.check_vnni()) if hasattr(ng, "check_vnni") else None
        info["openmp"] = bool(getattr(ng, "has_openmp", False))
    except ImportError as exc:
        info["native_gemv_error"] = str(exc)

    try:
        import subprocess

        out = subprocess.check_output(
            ["wmic", "cpu", "get", "Name"], stderr=subprocess.DEVNULL, text=True
        )
        lines = [ln.strip() for ln in out.splitlines() if ln.strip() and ln.strip() != "Name"]
        if lines:
            info["cpu_name"] = lines[0]
    except Exception:
        pass

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
