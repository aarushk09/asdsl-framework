"""Q4_K_M GEMV correctness: native vs Q8-quant reference (same path as kernel)."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GGUF = ROOT / "models" / "llama_cpp" / "phi4-mm-Q4_K_M.gguf"
TENSOR = "blk.0.ffn_up.weight"
MAX_REL = 0.01


def fp16_to_f32(h: int) -> float:
    return struct.unpack("<e", struct.pack("<H", h & 0xFFFF))[0]


def get_6bit(packed: bytes, idx: int) -> int:
    off = (idx * 6) // 8
    b0 = packed[off]
    b1 = packed[off + 1] if off + 1 < len(packed) else 0
    shift = (idx * 6) % 8
    return ((b0 >> shift) | (b1 << (8 - shift))) & 0x3F


def dot_q4k_block_ref(
    blk: np.ndarray, x_q8_256: np.ndarray, x_scales_8: np.ndarray
) -> float:
    d = fp16_to_f32(int(blk[0]) | (int(blk[1]) << 8))
    dmin = fp16_to_f32(int(blk[2]) | (int(blk[3]) << 8))
    scales = blk[4:16]
    qs_base = blk[16:144]
    acc = 0.0
    for sb in range(8):
        sub_scale = d * float(get_6bit(scales.tobytes(), sb))
        sub_min = dmin * float(get_6bit(scales.tobytes(), sb + 8))
        qs = qs_base[sb * 16 : (sb + 1) * 16]
        xq = x_q8_256[sb * 32 : (sb + 1) * 32].astype(np.int32)
        x_scale = float(x_scales_8[sb])
        dot_int = 0
        xsum = 0
        for i in range(16):
            lo = int(qs[i]) & 0x0F
            hi = int(qs[i]) >> 4
            dot_int += lo * int(xq[2 * i]) + hi * int(xq[2 * i + 1])
        for i in range(32):
            xsum += int(xq[i])
        acc += (sub_scale * dot_int - sub_min * xsum) * x_scale
    return acc


def quantize_x(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gs = 32
    n_groups = x.shape[0] // gs
    x_q8 = np.zeros(x.shape[0], dtype=np.int8)
    x_sc = np.zeros(n_groups, dtype=np.float32)
    for g in range(n_groups):
        seg = x[g * gs : (g + 1) * gs]
        amax = float(np.max(np.abs(seg)))
        if amax < 1e-12:
            continue
        inv = 127.0 / amax
        x_sc[g] = amax / 127.0
        x_q8[g * gs : (g + 1) * gs] = np.clip(np.round(seg * inv), -127, 127).astype(np.int8)
    return x_q8, x_sc


def gemv_q4km_ref(weights: np.ndarray, x: np.ndarray, rows: int, cols: int) -> np.ndarray:
    n_blocks = cols // 256
    y = np.zeros(rows, dtype=np.float32)
    x_q8, x_sc = quantize_x(x)
    for row in range(rows):
        base = row * n_blocks * 144
        acc = 0.0
        for b in range(n_blocks):
            blk = weights[base + b * 144 : base + (b + 1) * 144]
            acc += dot_q4k_block_ref(
                blk,
                x_q8[b * 256 : (b + 1) * 256],
                x_sc[b * 8 : (b + 1) * 8],
            )
        y[row] = acc
    return y


def test_synthetic(ng) -> bool:
    rows, cols = 64, 3072
    rng = np.random.default_rng(42)
    n_blocks = cols // 256
    blocks = rng.integers(0, 256, (rows, n_blocks, 144), dtype=np.uint8)
    for b in range(n_blocks):
        blocks[:, b, 0:2] = np.array([0x00, 0x3C], dtype=np.uint8)  # d ~ 1.0 fp16
        blocks[:, b, 2:4] = np.array([0x00, 0x00], dtype=np.uint8)  # dmin ~ 0
        blocks[:, b, 4:16] = 0
    flat = np.ascontiguousarray(blocks.reshape(-1))
    x = rng.standard_normal(cols).astype(np.float32)
    y_ref = gemv_q4km_ref(flat, x, rows, cols)
    y = np.zeros(rows, np.float32)
    ng.gemv_q4km_q8_avx2(flat, x, y, rows, cols)
    rel = float(np.max(np.abs(y - y_ref)) / (np.max(np.abs(y_ref)) + 1e-8))
    ok = rel < MAX_REL
    print(f"synthetic q4km max_rel={rel:.6f} {'PASS' if ok else 'FAIL'}")
    return ok


def test_gguf(ng) -> bool | None:
    if not GGUF.is_file():
        print(f"SKIP GGUF: missing {GGUF}")
        return None
    try:
        from asdsl.io.gguf_loader import read_gguf_tensors, q4k_blocks_rowmajor
    except ImportError:
        print("SKIP GGUF: asdsl.io.gguf_loader missing")
        return None

    tensors = read_gguf_tensors(str(GGUF), name_filter="blk.0.ffn_up")
    info = tensors.get(TENSOR)
    if info is None or str(info.get("type", "")).lower() != "q4_k":
        print("SKIP GGUF: blk.0.ffn_up not Q4_K")
        return None

    w_q4 = q4k_blocks_rowmajor(info)
    rows, cols = info["logical_shape"]
    x = np.random.default_rng(0).standard_normal(cols).astype(np.float32)
    y_ref = gemv_q4km_ref(w_q4.reshape(-1), x, rows, cols)
    y = np.zeros(rows, np.float32)
    ng.gemv_q4km_q8_avx2(w_q4, x, y, rows, cols)
    rel = float(np.max(np.abs(y - y_ref)) / (np.max(np.abs(y_ref)) + 1e-8))
    ok = rel < MAX_REL
    print(f"GGUF ffn_up q4km max_rel={rel:.6f} {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    from asdsl.kernels import _native_gemv as ng

    if not hasattr(ng, "gemv_q4km_q8_avx2"):
        print("FAIL: gemv_q4km_q8_avx2 not built")
        return 1

    ok_syn = test_synthetic(ng)
    ok_gguf = test_gguf(ng)
    if ok_gguf is None:
        return 0 if ok_syn else 1
    return 0 if (ok_syn and ok_gguf) else 1


if __name__ == "__main__":
    raise SystemExit(main())
