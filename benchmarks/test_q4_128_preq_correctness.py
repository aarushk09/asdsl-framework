"""Q4_128 preq GEMV vs dequantized-block fp32 reference."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def dequant_blocks(blocks_3d: np.ndarray, group_size: int = 128) -> np.ndarray:
    rows, n_groups, block_size = blocks_3d.shape
    cols = n_groups * group_size
    out = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        col = 0
        for g in range(n_groups):
            blk = blocks_3d[r, g]
            ws = float(np.frombuffer(blk[:2].tobytes(), dtype=np.float16)[0])
            nib = blk[2:block_size]
            for i in range(len(nib)):
                lo = int(nib[i] & 0x0F)
                hi = int(nib[i] >> 4)
                out[r, col] = (lo - 8) * ws
                out[r, col + 1] = (hi - 8) * ws
                col += 2
    return out


def main() -> int:
    from asdsl.kernels import _native_gemv as ng
    from asdsl.quantization.repack_q4_128 import (
        BLOCK_SIZE,
        repack_fp32_to_q4_128_blocks,
        blocks_to_flat,
    )

    if not hasattr(ng, "gemv_q4_128_preq_avx2"):
        print("FAIL: gemv_q4_128_preq_avx2 not built")
        return 1

    rows, cols, gs = 256, 3072, 128
    rng = np.random.default_rng(7)
    w = rng.standard_normal((rows, cols)).astype(np.float32) * 0.05
    blocks_3d = repack_fp32_to_q4_128_blocks(w, rows, cols, gs)
    flat = blocks_to_flat(blocks_3d)
    w_dq = dequant_blocks(blocks_3d, gs)
    x = rng.standard_normal(cols).astype(np.float32)
    y_ref = w_dq @ x

    x_q8 = np.zeros(cols, dtype=np.int8)
    x_sc = np.zeros(cols // gs, dtype=np.float32)
    ng.quantize_activation_avx2(x, x_q8, x_sc, cols, gs)
    y = np.zeros(rows, dtype=np.float32)
    ng.gemv_q4_128_preq_avx2(flat, x_q8, x_sc, y, rows, cols, gs)

    denom = float(np.max(np.abs(y_ref)) + 1e-8)
    rel = float(np.max(np.abs(y - y_ref)) / denom)
    print(f"max_rel_error={rel:.6f} ({'PASS' if rel < 0.01 else 'FAIL'})")
    return 0 if rel < 0.01 else 1


if __name__ == "__main__":
    raise SystemExit(main())
