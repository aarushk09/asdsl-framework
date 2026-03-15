/**
 * Fused 3-bit GEMV Kernel with AVX2 + FMA for ASDSL Framework
 *
 * Computes y = dequant(W_q3) @ x where W is packed in "10-in-32" format:
 *   10 × 3-bit values stored in one uint32 (30 of 32 bits used).
 *
 * Dequantization: dequant(w_int) = w_int * scale + bias  (per group)
 *   bias = -zero_point * scale  (precomputed by caller)
 *
 * The kernel operates on pre-unpacked uint8 weights (1 byte per value)
 * for direct compatibility with WeightStore._quant_u8.
 *
 * SIMD strategy: same as Q4/Q8 unpacked kernels — load 8 uint8 values,
 * zero-extend to int32, convert to float32, FMA with activations.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <immintrin.h>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <vector>
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
 * Core Kernel: 3-bit GEMV (unpacked uint8, 1 byte per value)
 * =================================================================== */

static void gemv_q3_unpacked_impl(
    const uint8_t* __restrict w,
    const float*   __restrict x,
    const float*   __restrict scales,
    const float*   __restrict biases,
    float*         __restrict y,
    int M, int K, int group_size
) {
    const int groups_per_row = K / group_size;

    // Precompute sum(x) per group (shared across all rows)
    std::vector<float> group_sum_x(groups_per_row);
    for (int g = 0; g < groups_per_row; ++g) {
        __m256 acc = _mm256_setzero_ps();
        const float* xg = x + g * group_size;
        int j = 0;
        for (; j <= group_size - 8; j += 8) {
            acc = _mm256_add_ps(acc, _mm256_loadu_ps(xg + j));
        }
        float sum = hsum256_ps(acc);
        for (; j < group_size; ++j) {
            sum += xg[j];
        }
        group_sum_x[g] = sum;
    }

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; ++m) {
        const uint8_t* row = w + static_cast<size_t>(m) * K;
        float row_sum = 0.0f;

        for (int g = 0; g < groups_per_row; ++g) {
            const int k0   = g * group_size;
            const int gidx = m * groups_per_row + g;

            __m256 dot = _mm256_setzero_ps();

            int j = 0;
            for (; j <= group_size - 8; j += 8) {
                __m128i raw_bytes = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i*>(row + k0 + j));
                __m256i v_w_i32 = _mm256_cvtepu8_epi32(raw_bytes);
                __m256 v_w = _mm256_cvtepi32_ps(v_w_i32);

                dot = _mm256_fmadd_ps(
                    v_w,
                    _mm256_loadu_ps(x + k0 + j),
                    dot);
            }

            float dot_val = hsum256_ps(dot);

            // Scalar tail for non-multiple-of-8 group sizes
            for (; j < group_size; ++j) {
                dot_val += static_cast<float>(row[k0 + j]) * x[k0 + j];
            }

            row_sum += scales[gidx] * dot_val + biases[gidx] * group_sum_x[g];

            _mm_prefetch(
                reinterpret_cast<const char*>(row + k0 + group_size + 64),
                _MM_HINT_T0);
        }

        y[m] = row_sum;
    }
}

/* ===================================================================
 * Core Kernel: 3-bit GEMV on packed 10-in-32 weights
 *
 * Reads the raw packed format directly: every 4 bytes contain
 * 10 × 3-bit values. Unpacks inline using shifts and masks.
 * =================================================================== */

static void gemv_q3_packed_impl(
    const uint8_t* __restrict w_packed_bytes,
    const float*   __restrict x,
    const float*   __restrict scales,
    const float*   __restrict biases,
    float*         __restrict y,
    int M, int K, int group_size
) {
    const int groups_per_row = K / group_size;
    const uint32_t* w_packed = reinterpret_cast<const uint32_t*>(w_packed_bytes);
    const int words_per_row = (K + 9) / 10;

    std::vector<float> group_sum_x(groups_per_row);
    for (int g = 0; g < groups_per_row; ++g) {
        __m256 acc = _mm256_setzero_ps();
        const float* xg = x + g * group_size;
        int j = 0;
        for (; j <= group_size - 8; j += 8) {
            acc = _mm256_add_ps(acc, _mm256_loadu_ps(xg + j));
        }
        float sum = hsum256_ps(acc);
        for (; j < group_size; ++j) sum += xg[j];
        group_sum_x[g] = sum;
    }

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; ++m) {
        const uint32_t* row = w_packed + static_cast<size_t>(m) * words_per_row;
        float row_sum = 0.0f;

        int k_global = 0;
        int word_idx = 0;

        for (int g = 0; g < groups_per_row; ++g) {
            const int gidx = m * groups_per_row + g;
            float dot_val = 0.0f;

            for (int gj = 0; gj < group_size; /* incremented inside */) {
                uint32_t word = row[word_idx];
                int vals_in_word = std::min(10, group_size - gj);
                vals_in_word = std::min(vals_in_word, K - k_global);

                // Extract 3-bit values from the word
                for (int vi = 0; vi < vals_in_word && vi < 10; ++vi) {
                    uint8_t val = (word >> (vi * 3)) & 0x07;
                    dot_val += static_cast<float>(val) * x[k_global];
                    k_global++;
                    gj++;
                }

                // If we used all 10 values from this word, advance
                word_idx++;

                // If we still need values for this group but the word
                // boundary split mid-group, continue with next word
            }

            row_sum += scales[gidx] * dot_val + biases[gidx] * group_sum_x[g];
        }

        y[m] = row_sum;
    }
}

/* ===================================================================
 * CPU Feature Detection
 * =================================================================== */

static bool check_avx2_support() {
#if defined(_MSC_VER)
    int info[4];
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5)) != 0;
#elif defined(__GNUC__) || defined(__clang__)
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return (ebx & (1 << 5)) != 0;
    }
    return false;
#else
    return false;
#endif
}

static bool check_fma_support() {
#if defined(_MSC_VER)
    int info[4];
    __cpuid(info, 1);
    return (info[2] & (1 << 12)) != 0;
#elif defined(__GNUC__) || defined(__clang__)
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        return (ecx & (1 << 12)) != 0;
    }
    return false;
#else
    return false;
#endif
}

/* ===================================================================
 * pybind11 Bindings
 * =================================================================== */

static py::array_t<float> py_gemv_q3_unpacked(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> w,
    py::array_t<float,   py::array::c_style | py::array::forcecast> x,
    py::array_t<float,   py::array::c_style | py::array::forcecast> scales,
    py::array_t<float,   py::array::c_style | py::array::forcecast> biases,
    int M, int K, int group_size
) {
    auto wb = w.request();
    auto xb = x.request();
    auto sb = scales.request();
    auto bb = biases.request();

    if (M <= 0 || K <= 0 || group_size <= 0)
        throw std::invalid_argument("M, K, group_size must be positive");
    if (K % group_size != 0)
        throw std::invalid_argument("K must be divisible by group_size");
    if (xb.size != K)
        throw std::invalid_argument("x length must equal K");
    if (wb.size != static_cast<int64_t>(M) * K)
        throw std::invalid_argument("weight buffer size mismatch");

    int64_t expected_groups = static_cast<int64_t>(M) * (K / group_size);
    if (sb.size != expected_groups || bb.size != expected_groups)
        throw std::invalid_argument("scales/biases size mismatch");

    auto result = py::array_t<float>(M);
    auto rb = result.request();

    {
        py::gil_scoped_release release;
        gemv_q3_unpacked_impl(
            static_cast<const uint8_t*>(wb.ptr),
            static_cast<const float*>(xb.ptr),
            static_cast<const float*>(sb.ptr),
            static_cast<const float*>(bb.ptr),
            static_cast<float*>(rb.ptr),
            M, K, group_size);
    }

    return result;
}

static py::array_t<float> py_gemv_q3_packed(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> w_packed,
    py::array_t<float,   py::array::c_style | py::array::forcecast> x,
    py::array_t<float,   py::array::c_style | py::array::forcecast> scales,
    py::array_t<float,   py::array::c_style | py::array::forcecast> biases,
    int M, int K, int group_size
) {
    auto wb = w_packed.request();
    auto xb = x.request();
    auto sb = scales.request();
    auto bb = biases.request();

    if (M <= 0 || K <= 0 || group_size <= 0)
        throw std::invalid_argument("M, K, group_size must be positive");
    if (K % group_size != 0)
        throw std::invalid_argument("K must be divisible by group_size");

    auto result = py::array_t<float>(M);
    auto rb = result.request();

    {
        py::gil_scoped_release release;
        gemv_q3_packed_impl(
            static_cast<const uint8_t*>(wb.ptr),
            static_cast<const float*>(xb.ptr),
            static_cast<const float*>(sb.ptr),
            static_cast<const float*>(bb.ptr),
            static_cast<float*>(rb.ptr),
            M, K, group_size);
    }

    return result;
}

PYBIND11_MODULE(_native_gemv_q3, m) {
    m.doc() = "Fused 3-bit GEMV kernels with AVX2 + FMA for ASDSL";

    m.def("gemv_q3_unpacked", &py_gemv_q3_unpacked,
        "Fused 3-bit GEMV on pre-unpacked uint8 weights (1 value/byte).",
        py::arg("w"), py::arg("x"),
        py::arg("scales"), py::arg("biases"),
        py::arg("M"), py::arg("K"), py::arg("group_size"));

    m.def("gemv_q3_packed", &py_gemv_q3_packed,
        "Fused 3-bit GEMV on packed 10-in-32 weights.",
        py::arg("w_packed"), py::arg("x"),
        py::arg("scales"), py::arg("biases"),
        py::arg("M"), py::arg("K"), py::arg("group_size"));

    m.def("check_avx2", &check_avx2_support,
        "Runtime check: does this CPU support AVX2?");
    m.def("check_fma", &check_fma_support,
        "Runtime check: does this CPU support FMA3?");

#ifdef _OPENMP
    m.def("get_num_threads", []() { return omp_get_max_threads(); });
    m.def("set_num_threads", [](int n) { omp_set_num_threads(n); },
        py::arg("n"));
    m.attr("has_openmp") = true;
#else
    m.def("get_num_threads", []() { return 1; });
    m.def("set_num_threads", [](int) {}, py::arg("n"));
    m.attr("has_openmp") = false;
#endif
}
