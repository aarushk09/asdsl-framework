"""CPU thermal / throttle helpers for reproducible benchmarks on Windows."""

from __future__ import annotations

import subprocess
import time

import numpy as np

_BASELINE_DOT_MS: float | None = None


def get_cpu_temp_celsius() -> float | None:
    """Package temperature in °C via WMI (may require admin on some systems)."""
    try:
        ps = (
            "Get-CimInstance -Namespace root/wmi -ClassName MSAcpi_ThermalZoneTemperature "
            "| Select-Object -First 1 -ExpandProperty CurrentTemperature"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        raw = (result.stdout or "").strip().split()[-1]
        tenths_k = float(raw)
        return tenths_k / 10.0 - 273.15
    except Exception:
        return None


def _dot_ms() -> float:
    a = np.random.randn(1024, 1024).astype(np.float32)
    b = np.random.randn(1024, 1024).astype(np.float32)
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        np.dot(a, b)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1000.0


def establish_throttle_baseline() -> float:
    """Record matmul speed after cooling; call once before benchmarks."""
    global _BASELINE_DOT_MS
    for _ in range(3):
        _dot_ms()
    _BASELINE_DOT_MS = _dot_ms()
    return _BASELINE_DOT_MS


def is_cpu_throttled(throttle_ratio: float = 1.12) -> bool:
    """True if matmul is slower than the post-cool baseline."""
    if _BASELINE_DOT_MS is None:
        return False
    return _dot_ms() > _BASELINE_DOT_MS * throttle_ratio


def wait_until_cool(
    max_temp_c: float = 65.0,
    check_interval_s: int = 30,
    max_wait_s: int = 600,
    initial_sleep_s: int = 60,
    establish_baseline: bool = True,
) -> dict:
    """Wait for low CPU temperature and non-throttled compute before benchmarking."""
    info: dict = {"initial_sleep_s": initial_sleep_s, "max_temp_c": max_temp_c}
    if initial_sleep_s > 0:
        print(f"Thermal guard: initial sleep {initial_sleep_s}s ...", flush=True)
        time.sleep(initial_sleep_s)

    waited = 0
    while waited < max_wait_s:
        temp = get_cpu_temp_celsius()
        throttled = is_cpu_throttled()
        info["temp_c"] = temp
        info["throttled"] = throttled
        info["waited_s"] = waited

        temp_ok = temp is None or temp < max_temp_c
        if temp_ok and not throttled:
            if establish_baseline and _BASELINE_DOT_MS is None:
                bl = establish_throttle_baseline()
                info["baseline_dot_ms"] = bl
            print(
                f"Thermal guard: temp={temp}°C throttled={throttled} — proceeding",
                flush=True,
            )
            info["proceeded"] = True
            return info

        if temp is not None and temp >= max_temp_c:
            print(
                f"Thermal guard: temp={temp:.1f}°C > {max_temp_c}°C — "
                f"waiting {check_interval_s}s ...",
                flush=True,
            )
        elif throttled:
            print(
                f"Thermal guard: compute throttle detected — waiting {check_interval_s}s ...",
                flush=True,
            )
        else:
            print(f"Thermal guard: waiting {check_interval_s}s ...", flush=True)

        time.sleep(check_interval_s)
        waited += check_interval_s

    if establish_baseline and _BASELINE_DOT_MS is None:
        info["baseline_dot_ms"] = establish_throttle_baseline()
    print("Thermal guard: max wait exceeded — proceeding anyway.", flush=True)
    info["proceeded"] = True
    info["max_wait_exceeded"] = True
    return info
