/**
 * 8-bit fused dequantization + GEMV (AVX2 + FMA + OpenMP).
 *
 * Computes y = dequant(W_q8) @ x where:
 *   dequant(w_int) = w_int * scale + bias      (per quantization group)
 *
 * For each output element y[m]:
 *   y[m] = sum_g [ scale_g * dot(W_int_g, x_g) + bias_g * sum(x_g) ]
 *
 * There are zero intermediate writes of float32 weights to RAM: uint8 weights are
 * widened to float in SIMD registers, FMA-accumulated with x, then scale/bias
 * apply per group.
 *
 * OpenMP: same P-core pinning and thread cap as Phase 1 (omp_pcore_pinning.hpp).
 *
 * Weight format: uint8, linear memory layout (one byte per weight).
 *
 * SIMD strategy (per 32-value chunk):
 *   1. Load 32 uint8 weights (1 YMM register) using _mm256_loadu_si256
 *   2. Convert 32 uint8 values -> 32 float32 values:
 *      - _mm256_cvtepi8_epi32 and _mm256_cvtepu8_epi32
 *      - Actually better: unpack to 16-bit, then 32-bit, then CVTDQ2PS
 *      - Or use _mm256_cvtepu8_epi32 on 128-bit chunks
 *   3. VFMADD231PS to accumulate w_float * x_float
 *   4. Apply scale/bias at group boundaries
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <immintrin.h>
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <stdexcept>

#include "omp_pcore_pinning.hpp"

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif

namespace py = pybind11;

namespace {
// ~16KB float slice of x per K-tile (plus weights); multiple tiles when K is large.
constexpr int CACHE_K_BLOCK_FLOATS = 4096;
}  // namespace

/* ===================================================================
 * AVX2 Utility Functions
 * =================================================================== */

static inline float hsum256_ps(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    lo = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, lo);
    lo = _mm_add_ss(lo, shuf);
    return _mm_cvtss_f32(lo);
}

static inline void q8_fused_accum_group(
    int g,
    int group_size,
    const uint8_t* __restrict w_row,
    const float* __restrict x,
    const float* __restrict sc_row,
    const float* __restrict bi_row,
    float& row_sum) {
    const int k_start = g * group_size;
    __m256 v_sum = _mm256_setzero_ps();
    float g_sum_x = 0.0f;

    int kk = 0;
    for (; kk <= group_size - 8; kk += 8) {
        __m256 v_x = _mm256_loadu_ps(x + k_start + kk);
        g_sum_x += x[k_start + kk + 0] + x[k_start + kk + 1] +
                   x[k_start + kk + 2] + x[k_start + kk + 3] +
                   x[k_start + kk + 4] + x[k_start + kk + 5] +
                   x[k_start + kk + 6] + x[k_start + kk + 7];

        __m128i raw_bytes =
            _mm_loadl_epi64((const __m128i*)(w_row + k_start + kk));
        __m256i v_w_i32 = _mm256_cvtepu8_epi32(raw_bytes);
        __m256 v_w = _mm256_cvtepi32_ps(v_w_i32);
        v_sum = _mm256_fmadd_ps(v_w, v_x, v_sum);
    }

    float dot_val = hsum256_ps(v_sum);
    for (; kk < group_size; ++kk) {
        const float x_k = x[k_start + kk];
        const float w_k = static_cast<float>(w_row[k_start + kk]);
        dot_val += w_k * x_k;
        g_sum_x += x_k;
    }

    row_sum += sc_row[g] * dot_val + bi_row[g] * g_sum_x;
}

/* ===================================================================
 * Core Kernel: 8-bit GEMV (uint8 unpacked)
 * =================================================================== */

static void fused_dequant_gemv_q8_impl(
    const uint8_t* __restrict w,
    const float*   __restrict x,
    const float*   __restrict scales,
    const float*   __restrict biases,
    float*         __restrict y,
    int M,
    int K,
    int group_size,
    bool cache_tiling) {
    const int groups_per_row = K / group_size;
    const int tile_groups =
        cache_tiling ? std::max(1, CACHE_K_BLOCK_FLOATS / std::max(group_size, 1)) : groups_per_row;

    asdsl_omp_pinning::configure_openmp_for_pcores();
#pragma omp parallel
    {
        asdsl_omp_pinning::bind_omp_thread_to_pcore_if_enabled();
#pragma omp for schedule(static)
        for (int m = 0; m < M; ++m) {
            float row_sum = 0.0f;
            const uint8_t* w_row = w + static_cast<size_t>(m) * K;
            const float* sc_row = scales + static_cast<size_t>(m) * groups_per_row;
            const float* bi_row = biases + static_cast<size_t>(m) * groups_per_row;

            if (!cache_tiling || tile_groups >= groups_per_row) {
                for (int g = 0; g < groups_per_row; ++g) {
                    q8_fused_accum_group(g, group_size, w_row, x, sc_row, bi_row, row_sum);
                }
            } else {
                for (int g_base = 0; g_base < groups_per_row; g_base += tile_groups) {
                    const int g_end = std::min(g_base + tile_groups, groups_per_row);
                    for (int g = g_base; g < g_end; ++g) {
                        q8_fused_accum_group(g, group_size, w_row, x, sc_row, bi_row, row_sum);
                    }
                }
            }

            y[m] = row_sum;
        }
    }
}

/* ===================================================================
 * Pybind11 Export
 * =================================================================== */

static py::array_t<float> py_gemv_q8_unpacked(
    py::array_t<uint8_t> w_in,
    py::array_t<float> x_in,
    py::array_t<float> scales_in,
    py::array_t<float> biases_in,
    int M,
    int K,
    int group_size,
    bool cache_tiling = true) {
    if (K % group_size != 0) {
        throw std::invalid_argument("K must be a multiple of group_size");
    }

    constexpr auto c_contig = py::array::c_style | py::array::forcecast;

    auto w = py::array_t<uint8_t, c_contig>::ensure(w_in);
    auto x = py::array_t<float, c_contig>::ensure(x_in);
    auto s = py::array_t<float, c_contig>::ensure(scales_in);
    auto b = py::array_t<float, c_contig>::ensure(biases_in);

    if (!w || !x || !s || !b) {
        throw std::runtime_error("Input matrices must be contiguous C-style arrays.");
    }

    auto y = py::array_t<float>(M);

    fused_dequant_gemv_q8_impl(
        w.data(), x.data(), s.data(), b.data(),
        y.mutable_data(),
        M, K, group_size,
        cache_tiling);

    return y;
}

static py::array_t<float> py_fused_dequant_gemv(
    py::array_t<uint8_t> w_in,
    py::array_t<float> x_in,
    py::array_t<float> scales_in,
    py::array_t<float> biases_in,
    int M,
    int K,
    int group_size,
    bool cache_tiling = true) {
    return py_gemv_q8_unpacked(w_in, x_in, scales_in, biases_in, M, K, group_size, cache_tiling);
}

static void py_set_pin_openmp_pcores(bool enabled) {
    asdsl_omp_pinning::pin_openmp_pcores_enabled() = enabled;
    if (enabled) {
        asdsl_omp_pinning::configure_openmp_for_pcores();
    }
}

static int py_detected_pcore_count() {
    return asdsl_omp_pinning::detected_pcore_count();
}

static bool has_avx2() {
#if defined(_MSC_VER)
    int info[4];
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5)) != 0;
#elif defined(__GNUC__) || defined(__clang__)
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return false;
    return (ebx & (1 << 5)) != 0;
#else
    return false;
#endif
}

PYBIND11_MODULE(_native_gemv_q8, m) {
    m.doc() = "AVX2-accelerated fused 8-bit dequant + GEMV";

    m.def(
        "fused_dequant_gemv",
        &py_fused_dequant_gemv,
        py::arg("w_u8"),
        py::arg("x"),
        py::arg("scales"),
        py::arg("biases"),
        py::arg("m"),
        py::arg("k"),
        py::arg("group_size") = 128,
        py::arg("cache_tiling") = true,
        "Fused uint8 dequantization + GEMV (no materialized f32 weight matrix).");
    m.def(
        "gemv_q8_unpacked",
        &py_gemv_q8_unpacked,
        py::arg("w_in"),
        py::arg("x_in"),
        py::arg("scales_in"),
        py::arg("biases_in"),
        py::arg("M"),
        py::arg("K"),
        py::arg("group_size"),
        py::arg("cache_tiling") = true);
    m.def(
        "set_pin_openmp_pcores",
        &py_set_pin_openmp_pcores,
        py::arg("enabled") = true,
        "Enable/disable P-core pinning for this module's OpenMP regions.");
    m.def("detected_pcore_count", &py_detected_pcore_count, "Number of detected P-core affinity masks (Windows).");

    m.attr("has_avx2") = has_avx2();
#ifdef _OPENMP
    m.attr("has_openmp") = true;
#else
    m.attr("has_openmp") = false;
#endif
}
