import numpy as np
import asdsl.kernels._native_forward as native_forward


def pack_scales_8x6(scales):
    s = np.asarray(scales, dtype=np.uint8)
    out = np.zeros(6, dtype=np.uint8)
    out[0] = (s[0] & 0x3F) | ((s[1] & 0x03) << 6)
    out[1] = ((s[1] >> 2) & 0x0F) | ((s[2] & 0x0F) << 4)
    out[2] = ((s[2] >> 4) & 0x03) | ((s[3] & 0x3F) << 2)
    out[3] = (s[4] & 0x3F) | ((s[5] & 0x03) << 6)
    out[4] = ((s[5] >> 2) & 0x0F) | ((s[6] & 0x0F) << 4)
    out[5] = ((s[6] >> 4) & 0x03) | ((s[7] & 0x3F) << 2)
    return out


def unpack_scales_8x6(buf):
    b = np.asarray(buf, dtype=np.uint8)
    out = np.zeros(8, dtype=np.uint8)
    out[0] = b[0] & 0x3F
    out[1] = ((b[0] >> 6) & 0x03) | ((b[1] & 0x0F) << 2)
    out[2] = ((b[1] >> 4) & 0x0F) | ((b[2] & 0x03) << 4)
    out[3] = (b[2] >> 2) & 0x3F
    out[4] = b[3] & 0x3F
    out[5] = ((b[3] >> 6) & 0x03) | ((b[4] & 0x0F) << 2)
    out[6] = ((b[4] >> 4) & 0x0F) | ((b[5] & 0x03) << 4)
    out[7] = (b[5] >> 2) & 0x3F
    return out


def make_q4k_row_bytes(out_dim, in_dim, seed=7):
    assert in_dim % 256 == 0
    rng = np.random.default_rng(seed)
    blocks_per_row = in_dim // 256
    bytes_per_block = 140
    row_bytes = np.zeros(out_dim * blocks_per_row * bytes_per_block, dtype=np.uint8)

    for r in range(out_dim):
        for b in range(blocks_per_row):
            base = (r * blocks_per_row + b) * bytes_per_block
            d = np.float16(rng.uniform(0.02, 0.2))
            dmin = np.float16(rng.uniform(-0.2, 0.0))
            scales = rng.integers(0, 64, size=8, dtype=np.uint8)
            q = rng.integers(0, 16, size=256, dtype=np.uint8)

            row_bytes[base : base + 2] = np.frombuffer(d.tobytes(), dtype=np.uint8)
            row_bytes[base + 2 : base + 4] = np.frombuffer(dmin.tobytes(), dtype=np.uint8)
            row_bytes[base + 4 : base + 10] = pack_scales_8x6(scales)
            row_bytes[base + 10 : base + 12] = 0

            q = q.reshape(8, 32)
            for sb in range(8):
                lo = q[sb, 0::2]
                hi = q[sb, 1::2]
                packed = (hi << 4) | lo
                row_bytes[base + 12 + sb * 16 : base + 12 + (sb + 1) * 16] = packed

    return row_bytes


def numpy_ref_q4k_gemv(row_bytes, x, out_dim, in_dim):
    blocks_per_row = in_dim // 256
    bytes_per_block = 140
    out = np.zeros(out_dim, dtype=np.float32)

    for r in range(out_dim):
        acc = 0.0
        for b in range(blocks_per_row):
            base = (r * blocks_per_row + b) * bytes_per_block
            d = np.frombuffer(row_bytes[base : base + 2].tobytes(), dtype=np.float16)[0].astype(np.float32)
            dmin = np.frombuffer(row_bytes[base + 2 : base + 4].tobytes(), dtype=np.float16)[0].astype(np.float32)
            scales = unpack_scales_8x6(row_bytes[base + 4 : base + 10]).astype(np.float32)

            for sb in range(8):
                packed = row_bytes[base + 12 + sb * 16 : base + 12 + (sb + 1) * 16]
                lo = packed & 0x0F
                hi = (packed >> 4) & 0x0F
                q = np.empty(32, dtype=np.float32)
                q[0::2] = lo
                q[1::2] = hi
                fq = q * (scales[sb] * d / 15.0) + dmin
                xb = x[b * 256 + sb * 32 : b * 256 + (sb + 1) * 32]
                acc += float(np.dot(fq, xb))
        out[r] = acc

    return out


def test_avx2_math():
    print("--- Testing AVX2 Q4_K_M Register Math ---")
    out_dim, in_dim = 4, 512
    rng = np.random.default_rng(123)
    x = rng.standard_normal(in_dim, dtype=np.float32)
    row_bytes = make_q4k_row_bytes(out_dim, in_dim, seed=13)

    c_out = native_forward.gemv_q4k_row(row_bytes, x, out_dim, in_dim)
    np_out = numpy_ref_q4k_gemv(row_bytes, x, out_dim, in_dim)

    delta = np.max(np.abs(c_out - np_out))
    print(f"Max Delta: {delta:.8f}")

    if np.allclose(c_out, np_out, atol=1e-4, rtol=1e-4):
        print("SUCCESS: AVX2 register math matches NumPy reference.")
    else:
        print("FAILURE: AVX2 math mismatch detected.")

if __name__ == '__main__':
    test_avx2_math()
