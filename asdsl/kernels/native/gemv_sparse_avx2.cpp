/**
 * Activation-Sparse GEMV Kernel with AVX2 for ASDSL Framework (Tier 3)
 *
 * Computes y = dequant(W) @ x, skipping columns where the activation
 * bitmask indicates near-zero values.
 *
 * After the SiLU gating in Phi-4's MLP, 80-95% of activations are
 * near-zero. By skipping those columns we avoid loading the
 * corresponding weight rows from DRAM entirely.
 *
 * Supports 4-bit, 3-bit, and 8-bit unpacked uint8 weights.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <immintrin.h>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <algorithm>

#if defined(_MSC_VER)
#include <intrin.h>
static inline int ctz32(uint32_t v) {
    unsigned long idx;
    _BitScanForward(&idx, v);
    return (int)idx;
}
#else
static inline int ctz32(uint32_t v) { return __builtin_ctz(v); }
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
 * Core: sparse GEMV on unpacked uint8 weights (any bit-width)
 *
 * Processes in 32-column blocks. If a bitmask word is 0, the entire
 * block is skipped. Within a non-zero block, only columns whose
 * bit is set are processed.
 *
 * This version uses a row-major approach: for each row, accumulate
 * only the active columns. This is cache-friendly for row-major
 * weight layout and exploits sparsity at the column level.
 * =================================================================== */

static void gemv_sparse_unpacked_impl(
    const uint8_t*  __restrict w,
    const float*    __restrict x,
    const float*    __restrict scales,
    const float*    __restrict biases,
    const uint32_t* __restrict bitmask,
    float*          __restrict y,
    int M, int K, int group_size,
    int n_active_cols,
    const int32_t*  __restrict active_col_indices
) {
    const int groups_per_row = K / group_size;

    if (active_col_indices != nullptr && n_active_cols > 0) {
        // Fast path: caller provides a pre-computed list of active columns
        #pragma omp parallel for schedule(static)
        for (int m = 0; m < M; ++m) {
            const uint8_t* row = w + static_cast<size_t>(m) * K;

            // Per-group accumulators
            std::vector<float> g_dot(groups_per_row, 0.0f);
            std::vector<float> g_sumx(groups_per_row, 0.0f);

            for (int i = 0; i < n_active_cols; ++i) {
                int k = active_col_indices[i];
                int g = k / group_size;
                float xv = x[k];
                g_dot[g] += static_cast<float>(row[k]) * xv;
                g_sumx[g] += xv;
            }

            float row_sum = 0.0f;
            for (int g = 0; g < groups_per_row; ++g) {
                int gidx = m * groups_per_row + g;
                row_sum += scales[gidx] * g_dot[g] + biases[gidx] * g_sumx[g];
            }
            y[m] = row_sum;
        }
    } else {
        // Bitmask-based path: scan bitmask words
        const int n_words = (K + 31) / 32;

        #pragma omp parallel for schedule(static)
        for (int m = 0; m < M; ++m) {
            const uint8_t* row = w + static_cast<size_t>(m) * K;

            std::vector<float> g_dot(groups_per_row, 0.0f);
            std::vector<float> g_sumx(groups_per_row, 0.0f);

            for (int wi = 0; wi < n_words; ++wi) {
                uint32_t mask = bitmask[wi];
                if (mask == 0) continue;

                while (mask) {
                    int bit = ctz32(mask);
                    mask &= mask - 1;

                    int k = wi * 32 + bit;
                    if (k >= K) break;

                    int g = k / group_size;
                    float xv = x[k];
                    g_dot[g] += static_cast<float>(row[k]) * xv;
                    g_sumx[g] += xv;
                }
            }

            float row_sum = 0.0f;
            for (int g = 0; g < groups_per_row; ++g) {
                int gidx = m * groups_per_row + g;
                row_sum += scales[gidx] * g_dot[g] + biases[gidx] * g_sumx[g];
            }
            y[m] = row_sum;
        }
    }
}

/* ===================================================================
 * Bitmask generation helper (vectorized)
 * =================================================================== */

static void compute_bitmask_impl(
    const float* __restrict x,
    uint32_t*    __restrict bitmask,
    int K,
    float threshold
) {
    const int n_words = (K + 31) / 32;
    std::memset(bitmask, 0, n_words * sizeof(uint32_t));

    __m256 v_thresh = _mm256_set1_ps(threshold);
    __m256 v_neg_thresh = _mm256_set1_ps(-threshold);

    for (int wi = 0; wi < n_words; ++wi) {
        uint32_t word = 0;
        int base = wi * 32;

        int j = 0;
        for (; j <= 24 && base + j + 7 < K; j += 8) {
            __m256 v_x = _mm256_loadu_ps(x + base + j);
            // |x| >= threshold  ⟺  x >= threshold OR x <= -threshold
            __m256 cmp_pos = _mm256_cmp_ps(v_x, v_thresh, _CMP_GE_OQ);
            __m256 cmp_neg = _mm256_cmp_ps(v_x, v_neg_thresh, _CMP_LE_OQ);
            __m256 cmp = _mm256_or_ps(cmp_pos, cmp_neg);
            int mask8 = _mm256_movemask_ps(cmp);
            word |= (static_cast<uint32_t>(mask8) << j);
        }

        // Scalar tail
        for (; j < 32 && base + j < K; ++j) {
            if (std::fabs(x[base + j]) >= threshold) {
                word |= (1u << j);
            }
        }

        bitmask[wi] = word;
    }
}

/* ===================================================================
 * pybind11 Bindings
 * =================================================================== */

static py::array_t<float> py_gemv_sparse_unpacked(
    py::array_t<uint8_t,  py::array::c_style | py::array::forcecast> w,
    py::array_t<float,    py::array::c_style | py::array::forcecast> x,
    py::array_t<float,    py::array::c_style | py::array::forcecast> scales,
    py::array_t<float,    py::array::c_style | py::array::forcecast> biases,
    py::array_t<uint32_t, py::array::c_style | py::array::forcecast> bitmask,
    int M, int K, int group_size
) {
    if (M <= 0 || K <= 0 || group_size <= 0)
        throw std::invalid_argument("M, K, group_size must be positive");
    if (K % group_size != 0)
        throw std::invalid_argument("K must be divisible by group_size");

    auto result = py::array_t<float>(M);

    {
        py::gil_scoped_release release;
        gemv_sparse_unpacked_impl(
            w.data(), x.data(), scales.data(), biases.data(),
            bitmask.data(),
            result.mutable_data(),
            M, K, group_size,
            0, nullptr);
    }

    return result;
}

static py::array_t<float> py_gemv_sparse_with_indices(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> w,
    py::array_t<float,   py::array::c_style | py::array::forcecast> x,
    py::array_t<float,   py::array::c_style | py::array::forcecast> scales,
    py::array_t<float,   py::array::c_style | py::array::forcecast> biases,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> active_indices,
    int M, int K, int group_size
) {
    if (M <= 0 || K <= 0 || group_size <= 0)
        throw std::invalid_argument("M, K, group_size must be positive");

    auto result = py::array_t<float>(M);
    int n_active = static_cast<int>(active_indices.size());

    {
        py::gil_scoped_release release;
        gemv_sparse_unpacked_impl(
            w.data(), x.data(), scales.data(), biases.data(),
            nullptr,
            result.mutable_data(),
            M, K, group_size,
            n_active, active_indices.data());
    }

    return result;
}

static py::array_t<uint32_t> py_compute_bitmask(
    py::array_t<float, py::array::c_style | py::array::forcecast> x,
    float threshold
) {
    int K = static_cast<int>(x.size());
    int n_words = (K + 31) / 32;
    auto result = py::array_t<uint32_t>(n_words);

    compute_bitmask_impl(x.data(), result.mutable_data(), K, threshold);

    return result;
}

PYBIND11_MODULE(_native_sparse_gemv, m) {
    m.doc() = "Activation-sparse GEMV with AVX2 for ASDSL (Tier 3)";

    m.def("gemv_sparse_unpacked", &py_gemv_sparse_unpacked,
        "Sparse GEMV with bitmask-based column skipping.",
        py::arg("w"), py::arg("x"),
        py::arg("scales"), py::arg("biases"),
        py::arg("bitmask"),
        py::arg("M"), py::arg("K"), py::arg("group_size"));

    m.def("gemv_sparse_with_indices", &py_gemv_sparse_with_indices,
        "Sparse GEMV with pre-computed active column indices.",
        py::arg("w"), py::arg("x"),
        py::arg("scales"), py::arg("biases"),
        py::arg("active_indices"),
        py::arg("M"), py::arg("K"), py::arg("group_size"));

    m.def("compute_bitmask", &py_compute_bitmask,
        "Vectorized bitmask generation: |x[i]| >= threshold.",
        py::arg("x"), py::arg("threshold"));

#ifdef _OPENMP
    m.attr("has_openmp") = true;
#else
    m.attr("has_openmp") = false;
#endif
}
