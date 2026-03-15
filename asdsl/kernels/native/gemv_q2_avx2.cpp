/**
 * Fused 2-bit GEMV Kernel with AVX2 + FMA for ASDSL Framework
 *
 * Computes y = dequant(W_q2) @ x where W uses 2-bit quantization
 * (4 values per byte).
 *
 * Dequantization: dequant(w_int) = w_int * scale + bias  (per group)
 *
 * Operates on pre-unpacked uint8 weights (1 byte per value) for
 * direct compatibility with WeightStore._quant_u8.
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
 * Core Kernel: 2-bit GEMV (unpacked uint8, 1 byte per value)
 * =================================================================== */

static void gemv_q2_unpacked_impl(
    const uint8_t* __restrict w,
    const float*   __restrict x,
    const float*   __restrict scales,
    const float*   __restrict biases,
    float*         __restrict y,
    int M, int K, int group_size
) {
    const int groups_per_row = K / group_size;

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
 * Sparse GEMV: skip columns where activation bitmask is zero
 *
 * Iterates over 32-column blocks. If the bitmask word is zero,
 * the entire block is skipped (no weight loads from DRAM).
 * =================================================================== */

static void gemv_q2_sparse_impl(
    const uint8_t*  __restrict w,
    const float*    __restrict x,
    const float*    __restrict scales,
    const float*    __restrict biases,
    const uint32_t* __restrict bitmask,
    float*          __restrict y,
    int M, int K, int group_size
) {
    const int groups_per_row = K / group_size;
    const int n_words = (K + 31) / 32;

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; ++m) {
        const uint8_t* row = w + static_cast<size_t>(m) * K;
        float row_sum = 0.0f;

        for (int g = 0; g < groups_per_row; ++g) {
            const int k0   = g * group_size;
            const int gidx = m * groups_per_row + g;
            float dot_val = 0.0f;
            float sum_x   = 0.0f;

            for (int j = 0; j < group_size; ++j) {
                int k = k0 + j;
                int word_idx = k / 32;
                int bit_pos  = k % 32;
                if (bitmask[word_idx] & (1u << bit_pos)) {
                    float xv = x[k];
                    dot_val += static_cast<float>(row[k]) * xv;
                    sum_x   += xv;
                }
            }

            row_sum += scales[gidx] * dot_val + biases[gidx] * sum_x;
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

static py::array_t<float> py_gemv_q2_unpacked(
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
        gemv_q2_unpacked_impl(
            static_cast<const uint8_t*>(wb.ptr),
            static_cast<const float*>(xb.ptr),
            static_cast<const float*>(sb.ptr),
            static_cast<const float*>(bb.ptr),
            static_cast<float*>(rb.ptr),
            M, K, group_size);
    }

    return result;
}

static py::array_t<float> py_gemv_q2_sparse(
    py::array_t<uint8_t,  py::array::c_style | py::array::forcecast> w,
    py::array_t<float,    py::array::c_style | py::array::forcecast> x,
    py::array_t<float,    py::array::c_style | py::array::forcecast> scales,
    py::array_t<float,    py::array::c_style | py::array::forcecast> biases,
    py::array_t<uint32_t, py::array::c_style | py::array::forcecast> bitmask,
    int M, int K, int group_size
) {
    auto wb = w.request();
    auto xb = x.request();
    auto sb = scales.request();
    auto bb = biases.request();
    auto mb = bitmask.request();

    if (M <= 0 || K <= 0 || group_size <= 0)
        throw std::invalid_argument("M, K, group_size must be positive");

    auto result = py::array_t<float>(M);
    auto rb = result.request();

    {
        py::gil_scoped_release release;
        gemv_q2_sparse_impl(
            static_cast<const uint8_t*>(wb.ptr),
            static_cast<const float*>(xb.ptr),
            static_cast<const float*>(sb.ptr),
            static_cast<const float*>(bb.ptr),
            static_cast<const uint32_t*>(mb.ptr),
            static_cast<float*>(rb.ptr),
            M, K, group_size);
    }

    return result;
}

PYBIND11_MODULE(_native_gemv_q2, m) {
    m.doc() = "Fused 2-bit GEMV kernels with AVX2 + FMA for ASDSL";

    m.def("gemv_q2_unpacked", &py_gemv_q2_unpacked,
        "Fused 2-bit GEMV on pre-unpacked uint8 weights.",
        py::arg("w"), py::arg("x"),
        py::arg("scales"), py::arg("biases"),
        py::arg("M"), py::arg("K"), py::arg("group_size"));

    m.def("gemv_q2_sparse", &py_gemv_q2_sparse,
        "Sparse 2-bit GEMV with activation bitmask.",
        py::arg("w"), py::arg("x"),
        py::arg("scales"), py::arg("biases"),
        py::arg("bitmask"),
        py::arg("M"), py::arg("K"), py::arg("group_size"));

    m.def("check_avx2", &check_avx2_support);
    m.def("check_fma", &check_fma_support);

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
