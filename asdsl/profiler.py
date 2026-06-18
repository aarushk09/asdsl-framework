"""Scientific profiling utilities for memory-bound inference roofline analysis."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil


@dataclass(frozen=True)
class StreamTriadResult:
    bandwidth_gb_s: float
    elapsed_sec: float
    runs: int
    warmup_runs: int
    array_bytes: int
    dtype: str
    implementation: str = "numpy_f32"  # native_openmp | numexpr_f32 | numpy_f32
    # Populated when implementation == "native_openmp" (OpenMP runtime / affinity diagnostics).
    omp_max_threads: int | None = None
    detected_pcores: int | None = None
    pin_openmp_pcores: bool | None = None


@dataclass(frozen=True)
class DualStreamTriadResult:
    """Concurrent dual-array STREAM triad (Q4 + Q2 draft footprint probe)."""

    bandwidth_a_gb_s: float
    bandwidth_b_gb_s: float
    combined_bandwidth_gb_s: float
    retention_a_pct: float
    retention_b_pct: float
    elapsed_sec: float
    runs: int
    warmup_runs: int
    array_a_bytes: int
    array_b_bytes: int
    single_stream_a_gb_s: float
    single_stream_b_gb_s: float
    implementation: str = "native_openmp"


@dataclass
class HardwareProfile:
    memory_bandwidth_gbps: float
    ram_gb: float
    cpu_physical_cores: int
    cpu_logical_cores: int
    method: str
    triad: StreamTriadResult | None = None


@dataclass(frozen=True)
class ModelArchitecture:
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    hidden_size: int
    head_dim: int


@dataclass(frozen=True)
class BytesPerTokenBreakdown:
    t_context: int
    weight_bytes: int
    kv_bytes: int
    total_bytes: int
    kv_bytes_per_element: int
    num_layers: int
    num_kv_heads: int
    head_dim: int

    @property
    def total_gb(self) -> float:
        return self.total_bytes / 1e9


def measure_stream_triad_bandwidth(
    array_mb: int = 256,
    runs: int = 12,
    warmup_runs: int = 2,
    *,
    require_native_openmp: bool = False,
) -> StreamTriadResult:
    """Measure DRAM bandwidth with a STREAM Triad (``a = b + scalar * c``).

    Uses three contiguous arrays of footprint ``array_mb`` MiB each. Effective
    bytes moved per timed pass: ``3 × array_bytes × runs``.

    Prefers (1) OpenMP float32 triad in ``_native_forward`` (``stream_triad_f32``)
    for vectorized DRAM saturation; else (2) OpenMP int8 triad (``stream_triad_int8``);
    else (3) multi-threaded ``numexpr`` float32; else (4) single-threaded NumPy float32.

    If ``require_native_openmp`` is True, missing extension, missing symbol, or
    runtime errors raise ``RuntimeError`` (no silent fallback).
    """
    if array_mb < 256:
        raise ValueError("array_mb must be >= 256 to exceed CPU caches")
    if runs < 1:
        raise ValueError("runs must be >= 1")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be >= 0")

    def _native_stream_triad() -> StreamTriadResult:
        try:
            from asdsl.kernels import _native_forward as native_forward
        except ImportError as e:
            raise RuntimeError(
                "asdsl.kernels._native_forward is not importable; install pybind11 and "
                "rebuild:  pip install pybind11  &&  python setup.py build_ext --inplace"
            ) from e
        if hasattr(native_forward, "stream_triad_f32"):
            raw = native_forward.stream_triad_f32(int(array_mb), int(runs), int(warmup_runs))
        elif hasattr(native_forward, "stream_triad_int8"):
            raw = native_forward.stream_triad_int8(int(array_mb), int(runs), int(warmup_runs))
        else:
            raise RuntimeError(
                "_native_forward was built without stream_triad_f32/stream_triad_int8; "
                "rebuild the extension (pip install pybind11 && python setup.py build_ext --inplace)."
            )
        return StreamTriadResult(
            bandwidth_gb_s=float(raw["bandwidth_gb_s"]),
            elapsed_sec=float(raw["elapsed_sec"]),
            runs=int(raw["runs"]),
            warmup_runs=int(raw["warmup_runs"]),
            array_bytes=int(raw["array_bytes"]),
            dtype=str(raw["dtype"]),
            implementation="native_openmp",
            omp_max_threads=int(raw["omp_max_threads"]),
            detected_pcores=int(raw["detected_pcores"]),
            pin_openmp_pcores=bool(raw["pin_openmp_pcores"]),
        )

    if require_native_openmp:
        return _native_stream_triad()

    try:
        from asdsl.kernels import _native_forward as native_forward

        if hasattr(native_forward, "stream_triad_f32"):
            raw = native_forward.stream_triad_f32(int(array_mb), int(runs), int(warmup_runs))
            return StreamTriadResult(
                bandwidth_gb_s=float(raw["bandwidth_gb_s"]),
                elapsed_sec=float(raw["elapsed_sec"]),
                runs=int(raw["runs"]),
                warmup_runs=int(raw["warmup_runs"]),
                array_bytes=int(raw["array_bytes"]),
                dtype=str(raw["dtype"]),
                implementation="native_openmp",
                omp_max_threads=int(raw["omp_max_threads"]),
                detected_pcores=int(raw["detected_pcores"]),
                pin_openmp_pcores=bool(raw["pin_openmp_pcores"]),
            )
        if hasattr(native_forward, "stream_triad_int8"):
            raw = native_forward.stream_triad_int8(int(array_mb), int(runs), int(warmup_runs))
            return StreamTriadResult(
                bandwidth_gb_s=float(raw["bandwidth_gb_s"]),
                elapsed_sec=float(raw["elapsed_sec"]),
                runs=int(raw["runs"]),
                warmup_runs=int(raw["warmup_runs"]),
                array_bytes=int(raw["array_bytes"]),
                dtype=str(raw["dtype"]),
                implementation="native_openmp",
                omp_max_threads=int(raw["omp_max_threads"]),
                detected_pcores=int(raw["detected_pcores"]),
                pin_openmp_pcores=bool(raw["pin_openmp_pcores"]),
            )
    except Exception:
        pass

    # Float32 triad: better reflects achievable DRAM throughput on modern CPUs than int8 Python
    # loops (narrow types under-fill SIMD / execution ports). Traffic = 3 × array_bytes × runs.
    num_f32 = (array_mb * 1024 * 1024) // 4
    if num_f32 < 1:
        raise ValueError("array_mb too small for float32 triad buffers")

    a = np.empty(num_f32, dtype=np.float32)
    b = np.random.standard_normal(num_f32).astype(np.float32, copy=False)
    c = np.random.standard_normal(num_f32).astype(np.float32, copy=False)
    scalar = np.float32(3.0)

    impl = "numpy_f32"
    try:
        import numexpr as ne

        nt = int(os.environ.get("NUMEXPR_MAX_THREADS", os.cpu_count() or 8))
        ne.set_num_threads(max(nt, 1))
        impl = "numexpr_f32"
        for _ in range(warmup_runs):
            ne.evaluate("b + s * c", local_dict={"b": b, "c": c, "s": scalar}, out=a)
        t0 = time.perf_counter()
        for _ in range(runs):
            ne.evaluate("b + s * c", local_dict={"b": b, "c": c, "s": scalar}, out=a)
        elapsed = time.perf_counter() - t0
    except ImportError:
        for _ in range(warmup_runs):
            a[:] = b + scalar * c
        t0 = time.perf_counter()
        for _ in range(runs):
            a[:] = b + scalar * c
        elapsed = time.perf_counter() - t0

    if elapsed <= 0:
        raise RuntimeError("STREAM Triad timing produced non-positive elapsed time")

    array_bytes = int(a.nbytes)
    bandwidth_gb_s = (3.0 * array_bytes * runs) / elapsed / 1e9
    return StreamTriadResult(
        bandwidth_gb_s=float(bandwidth_gb_s),
        elapsed_sec=float(elapsed),
        runs=int(runs),
        warmup_runs=int(warmup_runs),
        array_bytes=array_bytes,
        dtype="float32",
        implementation=impl,
    )


def measure_dual_stream_bandwidth(
    size_a_mb: int = 2400,
    size_b_mb: int = 750,
    runs: int = 8,
    warmup_runs: int = 2,
) -> DualStreamTriadResult:
    """Measure concurrent DRAM bandwidth on two independent float32 arrays.

    Calls ``stream_triad_dual_f32`` in ``_native_forward`` when built; raises
    ``RuntimeError`` if the symbol is missing.
    """
    if size_a_mb < 64 or size_b_mb < 64:
        raise ValueError("size_a_mb and size_b_mb must be >= 64")
    if runs < 1:
        raise ValueError("runs must be >= 1")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be >= 0")

    try:
        from asdsl.kernels import _native_forward as native_forward
    except ImportError as exc:
        raise RuntimeError(
            "asdsl.kernels._native_forward is not importable; rebuild native extensions"
        ) from exc

    if not hasattr(native_forward, "stream_triad_dual_f32"):
        raise RuntimeError(
            "_native_forward was built without stream_triad_dual_f32; "
            "rebuild: python setup.py build_ext --inplace"
        )

    raw = native_forward.stream_triad_dual_f32(
        int(size_a_mb), int(size_b_mb), int(runs), int(warmup_runs)
    )
    combined = float(raw["bandwidth_gb_s"])
    elapsed = float(raw["elapsed_sec"])
    runs_i = int(raw["runs"])
    array_a_bytes = int(raw["array_a_bytes"])
    array_b_bytes = int(raw["array_b_bytes"])

    single_a = measure_stream_triad_bandwidth(
        array_mb=min(size_a_mb, 512), runs=runs_i, warmup_runs=warmup_runs
    )
    single_b = measure_stream_triad_bandwidth(
        array_mb=min(size_b_mb, 512), runs=runs_i, warmup_runs=warmup_runs
    )
    # Per-stream effective bandwidth from combined elapsed (parallel sections).
    bytes_a = 3.0 * array_a_bytes * runs_i
    bytes_b = 3.0 * array_b_bytes * runs_i
    bw_a = bytes_a / elapsed / 1e9 if elapsed > 0 else 0.0
    bw_b = bytes_b / elapsed / 1e9 if elapsed > 0 else 0.0
    ret_a = 100.0 * bw_a / max(single_a.bandwidth_gb_s, 1e-6)
    ret_b = 100.0 * bw_b / max(single_b.bandwidth_gb_s, 1e-6)

    return DualStreamTriadResult(
        bandwidth_a_gb_s=float(bw_a),
        bandwidth_b_gb_s=float(bw_b),
        combined_bandwidth_gb_s=float(combined),
        retention_a_pct=float(ret_a),
        retention_b_pct=float(ret_b),
        elapsed_sec=elapsed,
        runs=runs_i,
        warmup_runs=int(raw["warmup_runs"]),
        array_a_bytes=array_a_bytes,
        array_b_bytes=array_b_bytes,
        single_stream_a_gb_s=float(single_a.bandwidth_gb_s),
        single_stream_b_gb_s=float(single_b.bandwidth_gb_s),
        implementation="native_openmp",
    )


def estimate_memory_bandwidth_gbps(sample_mb: int = 256, repeats: int = 12) -> HardwareProfile:
    """Backward-compatible wrapper returning measured STREAM Triad bandwidth."""
    ram_gb = psutil.virtual_memory().total / (1024**3)
    physical = psutil.cpu_count(logical=False) or 1
    logical = psutil.cpu_count(logical=True) or physical

    triad = measure_stream_triad_bandwidth(array_mb=sample_mb, runs=max(repeats, 1))
    return HardwareProfile(
        memory_bandwidth_gbps=triad.bandwidth_gb_s,
        ram_gb=ram_gb,
        cpu_physical_cores=physical,
        cpu_logical_cores=logical,
        method=f"stream_triad_{triad.dtype}_{sample_mb}MB_{triad.implementation}",
        triad=triad,
    )


def load_architecture_from_config(config_path: Path) -> ModelArchitecture:
    """Load model architecture fields from a config JSON file."""
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    num_layers = int(cfg["num_hidden_layers"])
    num_heads = int(cfg["num_attention_heads"])
    num_kv_heads = int(cfg.get("num_key_value_heads", num_heads))
    hidden_size = int(cfg["hidden_size"])
    head_dim = hidden_size // max(num_heads, 1)
    return ModelArchitecture(
        num_layers=num_layers,
        num_attention_heads=num_heads,
        num_kv_heads=num_kv_heads,
        hidden_size=hidden_size,
        head_dim=head_dim,
    )


def infer_architecture_from_q_metadata(metadata_path: Path) -> ModelArchitecture:
    """Infer architecture from quantized metadata when config JSON is unavailable."""
    with metadata_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    layer_ids: set[int] = set()
    for key in meta:
        m = re.match(r"l(\d+)_", key)
        if m:
            layer_ids.add(int(m.group(1)))
    if not layer_ids:
        raise ValueError(f"Could not infer layers from metadata: {metadata_path}")

    # Infer hidden size from l0_o shape [hidden, hidden].
    o_shape = meta.get("l0_o", {}).get("shape")
    qkv_shape = meta.get("l0_qkv", {}).get("shape")
    if not o_shape or not qkv_shape:
        raise ValueError(f"Metadata missing l0_o/l0_qkv shape entries: {metadata_path}")

    hidden_size = int(o_shape[0])
    qkv_out = int(qkv_shape[0])
    kv_total = max((qkv_out - hidden_size) // 2, 1)

    # Try common head dimensions and pick the first consistent decomposition.
    candidate_head_dims = [256, 192, 160, 128, 96, 80, 64, 48, 40, 32]
    chosen_head_dim = None
    chosen_num_heads = None
    chosen_num_kv = None
    for hd in candidate_head_dims:
        if hidden_size % hd != 0:
            continue
        if kv_total % hd != 0:
            continue
        nh = hidden_size // hd
        nkv = kv_total // hd
        if nkv <= nh:
            chosen_head_dim = hd
            chosen_num_heads = nh
            chosen_num_kv = nkv
            break

    if chosen_head_dim is None:
        raise ValueError(
            f"Could not infer attention heads from metadata qkv shape in {metadata_path}"
        )

    return ModelArchitecture(
        num_layers=max(layer_ids) + 1,
        num_attention_heads=int(chosen_num_heads),
        num_kv_heads=int(chosen_num_kv),
        hidden_size=hidden_size,
        head_dim=int(chosen_head_dim),
    )


def bytes_per_token_breakdown(
    *,
    model_file_size_bytes: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    t_context: int,
    kv_bytes_per_element: int = 1,
) -> BytesPerTokenBreakdown:
    """Compute exact bytes/token at a pinned context length t.

    Bytes/token(t) = model_weight_bytes + 2 * L * n_kv_heads * d_head * t * kv_bytes_per_element
    """
    if model_file_size_bytes <= 0:
        raise ValueError("model_file_size_bytes must be > 0")
    if min(num_layers, num_kv_heads, head_dim, t_context, kv_bytes_per_element) <= 0:
        raise ValueError("All architecture and t parameters must be > 0")

    weight_bytes = int(model_file_size_bytes)
    kv_bytes = int(2 * num_layers * num_kv_heads * head_dim * t_context * kv_bytes_per_element)
    total_bytes = int(weight_bytes + kv_bytes)
    return BytesPerTokenBreakdown(
        t_context=int(t_context),
        weight_bytes=weight_bytes,
        kv_bytes=kv_bytes,
        total_bytes=total_bytes,
        kv_bytes_per_element=int(kv_bytes_per_element),
        num_layers=int(num_layers),
        num_kv_heads=int(num_kv_heads),
        head_dim=int(head_dim),
    )


def roofline_ceiling_tps(memory_bandwidth_gb_s: float, bytes_per_token: int) -> float:
    """Compute memory roofline ceiling in tokens/sec."""
    if memory_bandwidth_gb_s <= 0:
        raise ValueError("memory_bandwidth_gb_s must be > 0")
    if bytes_per_token <= 0:
        raise ValueError("bytes_per_token must be > 0")
    return (memory_bandwidth_gb_s * 1e9) / float(bytes_per_token)


def theoretical_autoregressive_limit_tps(memory_bandwidth_gbps: float, model_size_gb: float) -> float:
    """Backward-compatible simplified ceiling helper (weights-only approximation)."""
    if model_size_gb <= 0:
        raise ValueError("model_size_gb must be > 0")
    return memory_bandwidth_gbps / model_size_gb


def roofline_curve(
    *,
    memory_bandwidth_gb_s: float,
    model_file_size_bytes: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    kv_bytes_per_element: int,
    t_values: list[int],
) -> list[dict[str, float | int]]:
    """Return ceiling points over a sequence-length range."""
    points: list[dict[str, float | int]] = []
    for t in t_values:
        bpt = bytes_per_token_breakdown(
            model_file_size_bytes=model_file_size_bytes,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            t_context=t,
            kv_bytes_per_element=kv_bytes_per_element,
        )
        points.append(
            {
                "t": int(t),
                "bytes_per_token": int(bpt.total_bytes),
                "bytes_per_token_gb": float(bpt.total_gb),
                "ceiling_tps": float(roofline_ceiling_tps(memory_bandwidth_gb_s, bpt.total_bytes)),
            }
        )
    return points
