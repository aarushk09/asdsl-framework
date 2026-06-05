"""Preq GEMV correctness: g4fused vs 4row baseline on synthetic blocks."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def dequant_preq_row(blocks_row: np.ndarray, x_q8: np.ndarray, x_sc: np.ndarray, gs: int = 32) -> np.ndarray:
    """Reference matvec for one row of preq blocks."""
    n_groups = x_q8.shape[0] // gs
    out = 0.0
    for g in range(n_groups):
        blk = blocks_row[g]
        ws = np.frombuffer(blk[:2].tobytes(), dtype=np.float16)[0].astype(np.float32)
        wz = np.frombuffer(blk[2:4].tobytes(), dtype=np.float16)[0].astype(np.float32)
        nibbles = blk[4:]
        xg = x_q8[g * gs : (g + 1) * gs].astype(np.float32)
        xs = float(x_sc[g])
        xsum = float(xg.sum())
        dot = 0.0
        for i in range(16):
            w_lo = int(nibbles[i] & 0x0F)
            w_hi = int(nibbles[i] >> 4)
            dot += w_lo * xg[2 * i] + w_hi * xg[2 * i + 1]
        out += dot * ws * xs - wz * xsum * xs
    return out


def main() -> int:
    from asdsl.kernels import _native_gemv as ng

    if not hasattr(ng, "gemv_q4_32_preq_g4fused_4row_avx2"):
        print("SKIP: g4fused not built")
        return 0

    rows, cols, gs = 512, 3072, 32
    n_groups = cols // gs
    rng = np.random.default_rng(42)
    blocks = rng.integers(0, 256, (rows, n_groups, 20), dtype=np.uint8)
    # Valid fp16 scales in first 4 bytes
    for r in range(rows):
        for g in range(n_groups):
            blocks[r, g, 0:2] = np.array([0x00, 0x3C], dtype=np.uint8)  # scale ~1.0
            blocks[r, g, 2:4] = np.array([0x00, 0x00], dtype=np.uint8)

    x = rng.standard_normal(cols).astype(np.float32)
    y_base = np.zeros(rows, np.float32)
    y_g4 = np.zeros(rows, np.float32)

    os.environ["ASDSL_PREQ_G4FUSED"] = "0"
    ng.gemv_q4_32_preq_4row_avx2(blocks.reshape(-1), x, y_base, rows, cols, gs)
    ng.gemv_q4_32_preq_g4fused_4row_avx2(blocks.reshape(-1), x, y_g4, rows, cols, gs)

    denom = float(np.max(np.abs(y_base)) + 1e-8)
    rel = float(np.max(np.abs(y_g4 - y_base)) / denom)
    print(f"g4fused vs 4row max_rel={rel:.6f} {'PASS' if rel < 0.001 else 'FAIL'}")
    return 0 if rel < 0.001 else 1


if __name__ == "__main__":
    raise SystemExit(main())
