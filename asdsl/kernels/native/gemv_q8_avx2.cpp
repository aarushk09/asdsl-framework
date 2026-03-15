/**
 * 8-bit GEMV Kernel with AVX2 + FMA for ASDSL Framework
 *
 * Computes y = dequant(W_q8) @ x where:
 *   dequant(w_int) = w_int * scale + bias      (per quantization group)
 *
 * Mathematically, for each output element y[m]:
 *   y[m] = sum_g [ scale_g * dot(W_int_g, x_g) + bias_g * sum(x_g) ]
 *
 * This avoids ever materializing the full dequantized float32 weight matrix.
 * Instead we compute the float dot product in registers, then apply
 * the per-group scale/bias.
 *
 * Weight format: uint8, linear memory layout.
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
#include <cstdint>
#include <cstddef>
#include <stdexcept>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

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

/* ===================================================================
 * Core Kernel: 8-bit GEMV (uint8 unpacked)
 * =================================================================== */

static void gemv_q8_unpacked_impl(
    const uint8_t* __restrict w,
    const float*   __restrict x,
    const float*   __restrict scales,
    const float*   __restrict biases,
    float*         __restrict y,
    int M, int K, int group_size
) {
    const int groups_per_row = K / group_size;

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; ++m) {
        float row_sum = 0.0f;
        const uint8_t* w_row = w + m * K;
        const float* sc_row = scales + m * groups_per_row;
        const float* bi_row = biases + m * groups_per_row;

        for (int g = 0; g < groups_per_row; ++g) {
            const int k_start = g * group_size;
            __m256 v_sum = _mm256_setzero_ps();
            float g_sum_x = 0.0f;

            // Process 8 elements per iteration
            int k = 0;
            for (; k <= group_size - 8; k += 8) {
                // Load 8 float32 activations
                __m256 v_x = _mm256_loadu_ps(x + k_start + k);
                
                // Keep track of sum(x) for the bias correction: sum += sum(x) * bias
                g_sum_x += x[k_start + k + 0] + x[k_start + k + 1] +
                           x[k_start + k + 2] + x[k_start + k + 3] +
                           x[k_start + k + 4] + x[k_start + k + 5] +
                           x[k_start + k + 6] + x[k_start + k + 7];

                // Load 8 uint8 weights: extend 8 bytes -> 8 int32 -> 8 float32
                __m128i raw_bytes = _mm_loadl_epi64((const __m128i*)(w_row + k_start + k));
                __m256i v_w_i32 = _mm256_cvtepu8_epi32(raw_bytes);
                __m256 v_w = _mm256_cvtepi32_ps(v_w_i32);

                // dot += w * x
                v_sum = _mm256_fmadd_ps(v_w, v_x, v_sum);
            }

            // Horizontal sum for the 8-wide FMA
            float dot_val = hsum256_ps(v_sum);

            // Tail cleanup if group_size isn't a multiple of 8
            for (; k < group_size; ++k) {
                float x_k = x[k_start + k];
                float w_k = (float)w_row[k_start + k];
                dot_val += w_k * x_k;
                g_sum_x += x_k;
            }

            // Apply per-group scale and bias
            row_sum += sc_row[g] * dot_val + bi_row[g] * g_sum_x;
        }

        y[m] = row_sum;
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
    int M, int K, int group_size
) {
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

    gemv_q8_unpacked_impl(
        w.data(), x.data(), s.data(), b.data(),
        y.mutable_data(),
        M, K, group_size
    );

    return y;
}

static bool has_avx2() {
#if defined(_MSC_VER)
    int info[4];
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5)) != 0;
#elif defined(__GNUC__) || defined(__clang__)
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return false;
    return (ebx & bit_AVX2) != 0;
#else
    return false;
#endif
}

PYBIND11_MODULE(_native_gemv_q8, m) {
    m.doc() = "AVX2-accelerated 8-bit GEMV kernel";

    m.def("gemv_q8_unpacked", &py_gemv_q8_unpacked, "8-bit AVX2 GEMV (unpacked uint8)");

    m.attr("has_avx2") = has_avx2();
#ifdef _OPENMP
    m.attr("has_openmp") = true;
#else
    m.attr("has_openmp") = false;
#endif
}
