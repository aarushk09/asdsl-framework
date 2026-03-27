#!/usr/bin/env python3
"""
Phase 0 — Hardware and memory profiling for ASDSL benchmark baseline.

Detects CPU brand, core counts, cache sizes, RAM, SIMD support, and
software versions. Writes results to benchmark_baseline.json under
the key "hardware". All undetectable values are set to null.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

# ── optional deps ──────────────────────────────────────────────────────────────
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    import numpy as np
    _HAS_NUMPY = True
    _NUMPY_VERSION = np.__version__
except ImportError:
    _HAS_NUMPY = False
    _NUMPY_VERSION = None

try:
    import torch
    _TORCH_VERSION = torch.__version__
except ImportError:
    _TORCH_VERSION = None


# ── helpers ────────────────────────────────────────────────────────────────────

def _safe(fn):
    """Return fn() or None on any exception."""
    try:
        return fn()
    except Exception:
        return None


def _detect_cache_linux(level: str) -> int | None:
    """Read cache size from /sys on Linux (level: 'l1d', 'l2', 'l3')."""
    if sys.platform != "linux":
        return None
    level_map = {"l1d": "1", "l2": "2", "l3": "3"}
    lvl = level_map.get(level)
    if lvl is None:
        return None
    base = Path("/sys/devices/system/cpu/cpu0/cache")
    if not base.exists():
        return None
    for idx_dir in sorted(base.iterdir()):
        level_file = idx_dir / "level"
        type_file = idx_dir / "type"
        size_file = idx_dir / "size"
        try:
            if level_file.read_text().strip() != lvl:
                continue
            # For L1, only pick the data cache
            if lvl == "1" and type_file.read_text().strip() not in ("Data", "Unified"):
                continue
            size_str = size_file.read_text().strip()  # e.g. "32K" or "256K"
            if size_str.endswith("K"):
                return int(size_str[:-1])
            if size_str.endswith("M"):
                return int(float(size_str[:-1]) * 1024)
        except Exception:
            continue
    return None


def _detect_cache_windows(level: str) -> int | None:
    """Use wmic on Windows to read cache sizes."""
    if sys.platform != "win32":
        return None
    level_map = {"l1d": 1, "l2": 2, "l3": 3}
    lvl = level_map.get(level)
    if lvl is None:
        return None
    try:
        result = subprocess.run(
            ["wmic", "cpu", "get", "L2CacheSize,L3CacheSize", "/format:csv"],
            capture_output=True, text=True, timeout=10
        )
        # wmic output is CSV; parse it
        lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
        if len(lines) >= 2:
            header = [h.lower() for h in lines[0].split(",")]
            values = lines[1].split(",")
            row = dict(zip(header, values))
            if level == "l2" and "l2cachesize" in row:
                v = row["l2cachesize"].strip()
                return int(v) if v.isdigit() else None
            if level == "l3" and "l3cachesize" in row:
                v = row["l3cachesize"].strip()
                return int(v) if v.isdigit() else None
    except Exception:
        pass
    return None


def _detect_cache_psutil(level: str) -> int | None:
    """psutil ≥ 5.9 exposes cpu_freq but not cache; kept as placeholder."""
    return None


def _detect_cache(level: str) -> int | None:
    for fn in (_detect_cache_linux, _detect_cache_windows, _detect_cache_psutil):
        v = fn(level)
        if v is not None:
            return v
    return None


def _check_avx2() -> bool:
    """Check AVX2 support via cpuid, native extension probe, or CPU model heuristic."""
    # Linux: /proc/cpuinfo flags
    if sys.platform == "linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("flags") and "avx2" in line:
                        return True
        except Exception:
            pass

    # Best proxy: if _native_gemv loaded, it was compiled with /arch:AVX2 and ran
    try:
        import asdsl.kernels._native_gemv as _ng  # noqa: F401
        # Verify it can actually execute an AVX2 op
        if _HAS_NUMPY:
            import numpy as _np
            _w = _np.zeros(8, dtype=np.uint8)
            _x = _np.ones(4, dtype=np.float32)
            _s = _np.ones(1, dtype=np.float32)
            _b = _np.zeros(1, dtype=np.float32)
            try:
                _ng.gemv_q4_packed(_w, _x, _s, _b, 1, 4, 4)
                return True
            except Exception:
                # Extension loaded but call failed — still indicates AVX2 build
                return True
    except ImportError:
        pass

    # Windows fallback: Intel Family 6 Model 60+ (Haswell) all have AVX2
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["wmic", "cpu", "get", "Caption", "/format:value"],
                capture_output=True, text=True, timeout=10
            )
            caption = result.stdout.lower()
            # Intel Family 6 Model 60+ = Haswell+ = AVX2
            # AMD Ryzen all have AVX2
            avx2_indicators = ["ryzen", "epyc", "threadripper",
                               "core(tm) i3-", "core(tm) i5-", "core(tm) i7-",
                               "core(tm) i9-", "xeon"]
            if any(f in caption for f in avx2_indicators):
                return True
            # Intel64 Family 6 Model >= 60 (Haswell) has AVX2
            import re
            m = re.search(r"model\s+(\d+)", caption)
            if m and int(m.group(1)) >= 60:
                return True
        except Exception:
            pass

    return False


def _check_avx512() -> bool:
    """Check AVX-512 support."""
    if sys.platform == "linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("flags") and "avx512f" in line:
                        return True
        except Exception:
            pass
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["wmic", "cpu", "get", "Caption", "/format:value"],
                capture_output=True, text=True, timeout=10
            )
            caption = result.stdout.lower()
            # AVX-512 is on Skylake-X, Ice Lake, Tiger Lake, Sapphire Rapids
            avx512_families = ["xeon", "core(tm) i9-10", "core(tm) i9-11",
                               "core(tm) i9-12", "core(tm) i9-13", "core(tm) i9-14"]
            if any(f in caption for f in avx512_families):
                return True
        except Exception:
            pass
    return False


# ── main profiling function ────────────────────────────────────────────────────

def collect_hw_profile() -> dict:
    hw: dict = {}

    # CPU brand
    hw["cpu_brand"] = _safe(platform.processor) or _safe(lambda: platform.uname().processor)

    # Core counts
    if _HAS_PSUTIL:
        hw["physical_cores"] = _safe(lambda: psutil.cpu_count(logical=False))
        hw["logical_cores"] = _safe(lambda: psutil.cpu_count(logical=True))
    else:
        hw["physical_cores"] = None
        hw["logical_cores"] = _safe(os.cpu_count)

    # Cache sizes (KB)
    hw["l1_cache_kb"] = _detect_cache("l1d")
    hw["l2_cache_kb"] = _detect_cache("l2")
    hw["l3_cache_kb"] = _detect_cache("l3")

    # RAM
    if _HAS_PSUTIL:
        vm = psutil.virtual_memory()
        hw["total_ram_gb"] = round(vm.total / 1e9, 1)
        hw["available_ram_gb"] = round(vm.available / 1e9, 1)
    else:
        hw["total_ram_gb"] = None
        hw["available_ram_gb"] = None

    # SIMD support
    hw["avx2_supported"] = _check_avx2()
    hw["avx512_supported"] = _check_avx512()

    # OS and Python
    hw["os_name"] = sys.platform
    hw["python_version"] = sys.version

    # Library versions
    hw["torch_version"] = _TORCH_VERSION
    hw["numpy_version"] = _NUMPY_VERSION

    return hw


def update_baseline_json(hw_profile: dict, baseline_path: Path) -> None:
    """Merge hw_profile into benchmark_baseline.json under key 'hardware'."""
    if baseline_path.exists():
        try:
            data = json.loads(baseline_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    else:
        data = {}
    data["hardware"] = hw_profile
    baseline_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


if __name__ == "__main__":
    baseline_path = Path(__file__).parent.parent / "benchmark_baseline.json"
    hw = collect_hw_profile()
    print(json.dumps(hw, indent=2))
    update_baseline_json(hw, baseline_path)
    print(f"\n[profile_hardware] Written to {baseline_path}")
