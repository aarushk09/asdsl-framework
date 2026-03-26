"""Tests for fused uint8 dequantization + GEMV (native AVX2/OpenMP).

Build native extension first::

    pip install pybind11 && python setup.py build_ext --inplace
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from numpy.testing import assert_allclose


def _reference_dequant_buffer_gemv(
    w_u8: np.ndarray,
    x: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    m: int,
    k: int,
    group_size: int,
) -> np.ndarray:
    """Explicit dequant to float32 matrix in memory, then GEMV (high traffic)."""
    groups_per_row = k // group_size
    w = w_u8.reshape(m, groups_per_row, group_size).astype(np.float32)
    sc = scales.reshape(m, groups_per_row, 1).astype(np.float32)
    bi = biases.reshape(m, groups_per_row, 1).astype(np.float32)
    wf = (w * sc + bi).reshape(m, k)
    return wf @ x


def _fused_bytes_per_call(m: int, k: int, group_size: int) -> float:
    """DRAM traffic model for fused path (read weights/scales/biases/x, write y)."""
    groups_per_row = k // group_size
    return float(
        m * k  # uint8 W
        + m * groups_per_row * 4 * 2  # scales + biases f32
        + k * 4  # x f32
        + m * 4  # y f32
    )


def _buffer_path_bytes_per_call(m: int, k: int, group_size: int) -> float:
    """Traffic model: read q, s, b; write full W f32; read W and x; write y."""
    groups_per_row = k // group_size
    return float(
        m * k
        + m * groups_per_row * 4 * 2
        + m * k * 4  # materialized W f32 write (then read in matmul)
        + m * k * 4  # matmul read W
        + k * 4
        + m * 4
    )


@pytest.fixture(scope="module")
def native_q8():
    pytest.importorskip("asdsl.kernels._native_gemv_q8")
    from asdsl.kernels import _native_gemv_q8 as mod

    return mod


def test_fused_matches_dequant_buffer(native_q8) -> None:
    rng = np.random.default_rng(42)
    m, k, group_size = 64, 256, 32
    assert k % group_size == 0
    groups_per_row = k // group_size

    w_u8 = rng.integers(0, 256, size=m * k, dtype=np.uint8)
    x = rng.standard_normal(k).astype(np.float32)
    scales = rng.uniform(0.01, 0.1, size=m * groups_per_row).astype(np.float32)
    biases = rng.standard_normal(m * groups_per_row).astype(np.float32)

    y_ref = _reference_dequant_buffer_gemv(w_u8, x, scales, biases, m, k, group_size)
    from asdsl.kernels.gemv_q8 import fused_dequant_gemv

    y_fused = fused_dequant_gemv(w_u8, x, scales, biases, m, k, group_size)

    assert_allclose(y_fused, y_ref, rtol=5e-4, atol=5e-4)


def test_fused_faster_and_bandwidth(native_q8) -> None:
    rng = np.random.default_rng(7)
    # Large footprint so materializing float32 W each iter hits DRAM; small problems
    # let highly threaded BLAS beat the OpenMP row parallel fused path.
    m, k, group_size = 1024, 8192, 128
    groups_per_row = k // group_size

    w_u8 = rng.integers(0, 256, size=m * k, dtype=np.uint8)
    x = rng.standard_normal(k).astype(np.float32)
    scales = rng.uniform(0.01, 0.1, size=m * groups_per_row).astype(np.float32)
    biases = rng.standard_normal(m * groups_per_row).astype(np.float32)

    from asdsl.kernels.gemv_q8 import fused_dequant_gemv

    n_warmup, n_iter = 3, 100

    for _ in range(n_warmup):
        _ = _reference_dequant_buffer_gemv(w_u8, x, scales, biases, m, k, group_size)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = _reference_dequant_buffer_gemv(w_u8, x, scales, biases, m, k, group_size)
    t_buf = time.perf_counter() - t0

    for _ in range(n_warmup):
        _ = fused_dequant_gemv(w_u8, x, scales, biases, m, k, group_size)
    t1 = time.perf_counter()
    for _ in range(n_iter):
        _ = fused_dequant_gemv(w_u8, x, scales, biases, m, k, group_size)
    t_fused = time.perf_counter() - t1

    bytes_fused = _fused_bytes_per_call(m, k, group_size) * n_iter
    bytes_buf = _buffer_path_bytes_per_call(m, k, group_size) * n_iter
    gbs_fused = bytes_fused / t_fused / 1e9
    gbs_buf = bytes_buf / t_buf / 1e9

    print(
        f"[test_fused_gemv] buffer path: {t_buf:.4f}s total, ~{gbs_buf:.2f} GB/s (model); "
        f"fused: {t_fused:.4f}s total, ~{gbs_fused:.2f} GB/s (model)"
    )

    assert _fused_bytes_per_call(m, k, group_size) < 0.55 * _buffer_path_bytes_per_call(
        m, k, group_size
    ), "fused path should move far fewer bytes than dequant-buffer + GEMV (no W_f32 in RAM)"

    assert t_fused < t_buf, (
        f"fused path should beat full dequant materialization each iter; "
        f"fused={t_fused:.4f}s buffer={t_buf:.4f}s over {n_iter} iters"
    )
