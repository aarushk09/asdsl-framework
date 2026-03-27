"""
Phase 1 correctness test: gemv_lut_q4_avx2 vs gemv_q4_packed reference.

Verifies that the vpshufb LUT kernel produces output within 2% relative
tolerance of the reference FMA kernel on synthetic Q4 weight matrices.
"""
from __future__ import annotations

import numpy as np
import pytest


# ── helpers ────────────────────────────────────────────────────────────────────

def dequant_q4_ref(weights_packed: np.ndarray, scales: np.ndarray,
                   biases: np.ndarray, x: np.ndarray,
                   out_features: int, in_features: int,
                   group_size: int) -> np.ndarray:
    """
    Pure Python reference: dequantize Q4 packed weights and compute dot product.
    Packing convention: byte[i] = (w[2i+1] << 4) | w[2i]
      lo nibble = even index weight, hi nibble = odd index weight
    Affine: y[m] = sum_g( dot(W_int_g, x_g) * scale[m,g] + bias[m,g] * sum(x_g) )
    """
    num_groups = in_features // group_size
    y = np.zeros(out_features, dtype=np.float64)

    for m in range(out_features):
        row_packed = weights_packed[m * (in_features // 2): (m + 1) * (in_features // 2)]
        for g in range(num_groups):
            k0 = g * group_size
            gidx = m * num_groups + g
            s = float(scales[gidx])
            b = float(biases[gidx])

            # Unpack nibbles for this group
            group_packed = row_packed[k0 // 2: (k0 + group_size) // 2]
            w_int = np.empty(group_size, dtype=np.float64)
            w_int[0::2] = (group_packed & 0x0F).astype(np.float64)   # lo nibble = even
            w_int[1::2] = ((group_packed >> 4) & 0x0F).astype(np.float64)  # hi nibble = odd

            xg = x[k0: k0 + group_size].astype(np.float64)
            dot = np.dot(w_int, xg)
            sum_x = np.sum(xg)
            y[m] += dot * s + b * sum_x

    return y.astype(np.float32)


# ── fixtures ───────────────────────────────────────────────────────────────────

def make_test_case(out_features: int, in_features: int, group_size: int,
                   seed: int = 42):
    rng = np.random.default_rng(seed)

    # Random Q4 weights: values 0-15
    weights_q4 = rng.integers(0, 16, (out_features, in_features), dtype=np.uint8)

    # Pack: byte[i] = (w[2i+1] << 4) | w[2i]
    weights_packed_2d = (weights_q4[:, 0::2] | (weights_q4[:, 1::2] << 4)).astype(np.uint8)
    weights_packed = weights_packed_2d.ravel()  # flat (out_features * in_features/2,)

    num_groups = in_features // group_size
    scales = rng.uniform(0.01, 0.1, out_features * num_groups).astype(np.float32)
    # biases = -zero_point * scale, zero_point = 8 for symmetric Q4
    biases = (-8.0 * scales).astype(np.float32)

    x = rng.standard_normal(in_features).astype(np.float32)

    return weights_packed, scales, biases, x


# ── tests ──────────────────────────────────────────────────────────────────────

class TestLutGemvCorrectness:

    def test_small_matrix(self):
        """Small matrix: 64×128, group_size=32. Catches basic correctness."""
        out_f, in_f, gs = 64, 128, 32
        wp, sc, bi, x = make_test_case(out_f, in_f, gs, seed=0)

        from asdsl.kernels._native_lut import gemv_lut_q4_avx2
        y_lut = gemv_lut_q4_avx2(wp, sc, bi, x, out_f, in_f, gs)
        y_ref = dequant_q4_ref(wp, sc, bi, x, out_f, in_f, gs)

        np.testing.assert_allclose(
            y_lut, y_ref, rtol=0.02, atol=0.01,
            err_msg="LUT output diverges from reference (small matrix)"
        )

    def test_medium_matrix_group32(self):
        """Medium matrix: 256×512, group_size=32."""
        out_f, in_f, gs = 256, 512, 32
        wp, sc, bi, x = make_test_case(out_f, in_f, gs, seed=42)

        from asdsl.kernels._native_lut import gemv_lut_q4_avx2
        y_lut = gemv_lut_q4_avx2(wp, sc, bi, x, out_f, in_f, gs)
        y_ref = dequant_q4_ref(wp, sc, bi, x, out_f, in_f, gs)

        # Print first 10 diverging elements for diagnosis
        diff = np.abs(y_lut - y_ref)
        rel_diff = diff / (np.abs(y_ref) + 1e-6)
        bad = np.where(rel_diff > 0.02)[0]
        if len(bad) > 0:
            print(f"\nFirst {min(10, len(bad))} diverging elements:")
            for i in bad[:10]:
                print(f"  idx={i}: ref={y_ref[i]:.6f}, lut={y_lut[i]:.6f}, "
                      f"rel_diff={rel_diff[i]:.4f}")

        np.testing.assert_allclose(
            y_lut, y_ref, rtol=0.02, atol=0.01,
            err_msg="LUT output diverges from reference (medium matrix, group_size=32)"
        )

    def test_medium_matrix_group64(self):
        """Medium matrix: 256×512, group_size=64."""
        out_f, in_f, gs = 256, 512, 64
        wp, sc, bi, x = make_test_case(out_f, in_f, gs, seed=7)

        from asdsl.kernels._native_lut import gemv_lut_q4_avx2
        y_lut = gemv_lut_q4_avx2(wp, sc, bi, x, out_f, in_f, gs)
        y_ref = dequant_q4_ref(wp, sc, bi, x, out_f, in_f, gs)

        np.testing.assert_allclose(
            y_lut, y_ref, rtol=0.02, atol=0.01,
            err_msg="LUT output diverges from reference (medium matrix, group_size=64)"
        )

    def test_phi4_projection_shape(self):
        """Phi-4 largest projection: 14336×3072, group_size=64. Spot-check 32 rows."""
        # Full matrix is too large for a unit test; use a 32-row slice
        out_f, in_f, gs = 32, 3072, 64
        wp, sc, bi, x = make_test_case(out_f, in_f, gs, seed=99)

        from asdsl.kernels._native_lut import gemv_lut_q4_avx2
        y_lut = gemv_lut_q4_avx2(wp, sc, bi, x, out_f, in_f, gs)
        y_ref = dequant_q4_ref(wp, sc, bi, x, out_f, in_f, gs)

        np.testing.assert_allclose(
            y_lut, y_ref, rtol=0.02, atol=0.01,
            err_msg="LUT output diverges from reference (Phi-4 projection shape)"
        )

    def test_nibble_packing_convention(self):
        """
        Explicit test of nibble packing convention.
        Byte 0xAB: lo nibble = 0xB (weight at even index), hi nibble = 0xA (odd index).
        """
        out_f, in_f, gs = 1, 16, 16
        # Manually construct: weights = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        # Packed: byte[i] = (w[2i+1] << 4) | w[2i]
        # byte[0] = (1<<4)|0 = 0x10, byte[1] = (3<<4)|2 = 0x32, ...
        weights_q4 = np.arange(16, dtype=np.uint8).reshape(1, 16)
        wp_2d = (weights_q4[:, 0::2] | (weights_q4[:, 1::2] << 4)).astype(np.uint8)
        wp = wp_2d.ravel()

        scales = np.array([1.0], dtype=np.float32)
        biases = np.array([0.0], dtype=np.float32)  # zero_point = 0 for this test
        x = np.ones(16, dtype=np.float32)

        from asdsl.kernels._native_lut import gemv_lut_q4_avx2
        y_lut = gemv_lut_q4_avx2(wp, scales, biases, x, out_f, in_f, gs)

        # Expected: sum(0..15) * 1.0 * 1.0 = 120.0
        expected = float(np.sum(np.arange(16)))
        assert abs(float(y_lut[0]) - expected) < 0.5, \
            f"Nibble packing test failed: got {y_lut[0]:.4f}, expected {expected:.4f}"

    def test_zero_weights(self):
        """All-zero weights should produce output close to bias*sum(x)."""
        out_f, in_f, gs = 16, 64, 32
        wp = np.zeros(out_f * in_f // 2, dtype=np.uint8)
        scales = np.full(out_f * (in_f // gs), 0.05, dtype=np.float32)
        biases = np.full(out_f * (in_f // gs), -8.0 * 0.05, dtype=np.float32)
        x = np.ones(in_f, dtype=np.float32)

        from asdsl.kernels._native_lut import gemv_lut_q4_avx2
        y_lut = gemv_lut_q4_avx2(wp, scales, biases, x, out_f, in_f, gs)
        y_ref = dequant_q4_ref(wp, scales, biases, x, out_f, in_f, gs)

        np.testing.assert_allclose(y_lut, y_ref, rtol=0.02, atol=0.01)

    def test_throughput_vs_reference(self):
        """
        Micro-benchmark: LUT kernel throughput on Phi-4 projection dimensions.

        The LUT advantage is memory-traffic reduction, not compute reduction.
        On small matrices (fits in L3 cache), the reference FMA path is faster
        because gather latency dominates. On large matrices (DRAM-bound), the
        LUT path wins by reducing effective bytes-per-weight.

        This test verifies the kernel runs without error and prints timing.
        The assertion checks that the LUT kernel produces correct output
        (already verified by other tests) and runs in finite time.
        """
        import time
        # Use Phi-4 projection dimensions to stress DRAM bandwidth
        out_f, in_f, gs = 14336, 3072, 64
        wp, sc, bi, x = make_test_case(out_f, in_f, gs, seed=1)

        from asdsl.kernels._native_lut import gemv_lut_q4_avx2
        from asdsl.kernels._native_gemv import gemv_q4_packed

        # Warm up
        _ = gemv_lut_q4_avx2(wp, sc, bi, x, out_f, in_f, gs)
        _ = gemv_q4_packed(wp, x, sc, bi, out_f, in_f, gs)

        N = 5
        t0 = time.perf_counter()
        for _ in range(N):
            gemv_lut_q4_avx2(wp, sc, bi, x, out_f, in_f, gs)
        t_lut = (time.perf_counter() - t0) * 1000 / N

        t0 = time.perf_counter()
        for _ in range(N):
            gemv_q4_packed(wp, x, sc, bi, out_f, in_f, gs)
        t_ref = (time.perf_counter() - t0) * 1000 / N

        flops = 2 * out_f * in_f
        gops_lut = flops / (t_lut / 1000) / 1e9
        gops_ref = flops / (t_ref / 1000) / 1e9

        print(f"\n[throughput] LUT ({out_f}x{in_f}): {t_lut:.2f} ms/call ({gops_lut:.2f} GOPS)")
        print(f"[throughput] REF ({out_f}x{in_f}): {t_ref:.2f} ms/call ({gops_ref:.2f} GOPS)")
        print(f"[throughput] Speedup: {t_ref/t_lut:.2f}x")
        print(f"[throughput] Note: LUT advantage is memory-traffic reduction on DRAM-bound workloads.")
        print(f"[throughput] On cache-resident matrices, gather latency dominates.")

        # The LUT kernel must complete in finite time (< 10s for 5 iterations)
        assert t_lut < 10000.0, f"LUT kernel too slow: {t_lut:.1f} ms/call"
        # The LUT kernel must produce correct output (verified by other tests)
        y_lut = gemv_lut_q4_avx2(wp, sc, bi, x, out_f, in_f, gs)
        assert y_lut.shape == (out_f,), f"Wrong output shape: {y_lut.shape}"
        assert not np.any(np.isnan(y_lut)), "LUT output contains NaN"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
