import numpy as np
import pytest

from asdsl.io.gguf_loader import q4k_dequantize_to_fp32


def _pack_6bit(values):
    out = np.zeros(12, dtype=np.uint8)
    for idx, v in enumerate(values):
        bit_pos = idx * 6
        byte_idx = bit_pos // 8
        bit_off = bit_pos % 8
        vv = int(v) & 0x3F
        if bit_off <= 2:
            out[byte_idx] |= np.uint8(vv << bit_off)
        else:
            out[byte_idx] |= np.uint8((vv << bit_off) & 0xFF)
            out[byte_idx + 1] |= np.uint8(vv >> (8 - bit_off))
    return out


def test_q4km_dequantize_correctness():
    block = np.zeros(144, dtype=np.uint8)
    block[0:2] = np.frombuffer(np.float16(0.5).tobytes(), dtype=np.uint8)
    block[2:4] = np.frombuffer(np.float16(0.25).tobytes(), dtype=np.uint8)

    scales = [2] * 8 + [1] * 8
    block[4:16] = _pack_6bit(scales)

    qvals = np.arange(16, dtype=np.uint8)
    packed = (qvals << 4) | qvals
    block[16:144] = np.tile(packed, 8)

    out = q4k_dequantize_to_fp32(block, 256)
    expected_first = np.repeat(np.arange(16, dtype=np.float32), 2) - 0.25
    np.testing.assert_allclose(out[:32], expected_first, rtol=1e-6, atol=1e-6)


def test_q4km_gemv_matches_dequant():
    ng = pytest.importorskip("asdsl.kernels._native_gemv")
    if not hasattr(ng, "gemv_q4km_q8_avx2"):
        pytest.skip("Q4_K_M GEMV not built")

    rng = np.random.default_rng(123)
    rows, cols = 8, 256
    row_bytes = 144
    weights = rng.integers(0, 256, size=rows * row_bytes, dtype=np.uint8)

    # Keep d/dmin finite and non-zero to avoid degenerate scaling.
    for r in range(rows):
        off = r * row_bytes
        weights[off:off + 2] = np.frombuffer(np.float16(0.2).tobytes(), dtype=np.uint8)
        weights[off + 2:off + 4] = np.frombuffer(np.float16(0.05).tobytes(), dtype=np.uint8)

    x = rng.standard_normal(cols).astype(np.float32)
    y = np.zeros(rows, dtype=np.float32)
    ng.gemv_q4km_q8_avx2(weights, x, y, rows, cols)

    w_f32 = np.vstack(
        [q4k_dequantize_to_fp32(weights[r * row_bytes:(r + 1) * row_bytes], cols) for r in range(rows)]
    )
    y_ref = w_f32 @ x

    np.testing.assert_allclose(y, y_ref, rtol=0.12, atol=6.0)
