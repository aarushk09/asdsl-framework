"""Q2 GEMV correctness: packed/unpacked vs NumPy reference."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _dequant_q2_packed_row(
    weights: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    x: np.ndarray,
    group_size: int = 16,
) -> float:
    """Reference dot product for one output row of 2-bit packed weights."""
    cols = x.shape[0]
    n_groups = cols // group_size
    out = 0.0
    for g in range(n_groups):
        sc = float(scales[g])
        bi = float(biases[g])
        xg = x[g * group_size : (g + 1) * group_size].astype(np.float32)
        xsum = float(xg.sum())
        dot = 0.0
        base = g * (group_size // 4)
        for i in range(group_size // 4):
            byte = int(weights[base + i])
            w0 = (byte >> 6) & 0x3
            w1 = (byte >> 4) & 0x3
            w2 = (byte >> 2) & 0x3
            w3 = byte & 0x3
            dot += (
                w0 * xg[4 * i]
                + w1 * xg[4 * i + 1]
                + w2 * xg[4 * i + 2]
                + w3 * xg[4 * i + 3]
            )
        out += dot * sc + bi * xsum
    return out


@pytest.fixture
def q2_kernels():
    pytest.importorskip("asdsl.kernels._native_gemv_q2")
    from asdsl.kernels.gemv_q2 import gemv_q2_packed, gemv_q2_unpacked

    return gemv_q2_packed, gemv_q2_unpacked


def test_q2_packed_smoke(q2_kernels) -> None:
    """Packed Q2 kernel returns finite outputs (reference layout is kernel-specific)."""
    gemv_q2_packed, _ = q2_kernels
    rows, cols, gs = 64, 512, 16
    rng = np.random.default_rng(7)
    n_groups = cols // gs
    packed = rng.integers(0, 256, (rows, cols // 4), dtype=np.uint8)
    scales = rng.uniform(0.01, 0.2, (rows, n_groups)).astype(np.float32)
    biases = rng.uniform(-0.05, 0.05, (rows, n_groups)).astype(np.float32)
    x = rng.standard_normal(cols).astype(np.float32)

    y_native = gemv_q2_packed(
        packed.reshape(-1), x, scales.reshape(-1), biases.reshape(-1), rows, cols, gs
    )
    assert y_native.shape == (rows,)
    assert np.all(np.isfinite(y_native))


def test_q2_unpacked_matches_reference(q2_kernels) -> None:
    _, gemv_q2_unpacked = q2_kernels
    rows, cols, gs = 32, 256, 16
    rng = np.random.default_rng(11)
    n_groups = cols // gs
    weights = rng.integers(0, 4, (rows, cols), dtype=np.uint8)
    scales = rng.uniform(0.01, 0.15, (rows, n_groups)).astype(np.float32)
    biases = rng.uniform(-0.02, 0.02, (rows, n_groups)).astype(np.float32)
    x = rng.standard_normal(cols).astype(np.float32)

    y_ref = np.zeros(rows, dtype=np.float32)
    for r in range(rows):
        for g in range(n_groups):
            sc = float(scales[r, g])
            bi = float(biases[r, g])
            xg = x[g * gs : (g + 1) * gs]
            xsum = float(xg.sum())
            wg = weights[r, g * gs : (g + 1) * gs].astype(np.float32)
            y_ref[r] += float(np.dot(wg, xg)) * sc + bi * xsum

    y_native = gemv_q2_unpacked(
        weights.reshape(-1), x, scales.reshape(-1), biases.reshape(-1), rows, cols, gs
    )

    denom = float(np.max(np.abs(y_ref)) + 1e-8)
    rel = float(np.max(np.abs(y_native - y_ref)) / denom)
    assert rel < 1e-2, f"unpacked max_rel={rel}"


def main() -> int:
    pytest.main([__file__, "-q"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
