/**
 * OpenMP-parallel lm_head GEMV: logits = W_f16 @ x  (W is vocab x hidden, row-major f16).
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <immintrin.h>
#include <cstdint>
#include <cstddef>
#include <algorithm>

#include "engine_flags.hpp"
#include "thread_pool.h"

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

void lm_head_gemv_f16_impl(
    const uint16_t* __restrict w_f16,
    const float* __restrict x,
    float* __restrict y,
    int M,
    int K
) {
    auto process_row = [&](int m) {
        const uint16_t* row = w_f16 + static_cast<size_t>(m) * K;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();

        int k = 0;
        for (; k + 15 < K; k += 16) {
            const __m128i w8 = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(row + k));
            __m256 wv0 = _mm256_cvtph_ps(w8);
            acc0 = _mm256_fmadd_ps(wv0, _mm256_loadu_ps(x + k), acc0);

            const __m128i w8b = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(row + k + 8));
            __m256 wv1 = _mm256_cvtph_ps(w8b);
            acc1 = _mm256_fmadd_ps(wv1, _mm256_loadu_ps(x + k + 8), acc1);
        }

        float sum = hsum256_ps(acc0) + hsum256_ps(acc1);
        for (; k < K; ++k) {
            const __m128i ph = _mm_cvtsi32_si128(static_cast<int>(row[k]));
            const float wf = _mm_cvtss_f32(_mm_cvtph_ps(ph));
            sum += wf * x[k];
        }
        y[m] = sum;
    };

#ifdef _OPENMP
    if (asdsl::persistent_pool_enabled() && asdsl::tl_active_pool != nullptr) {
        asdsl::ThreadPool& pool = asdsl::ThreadPool::get_instance();
        const int n_threads = std::max(1, pool.thread_count() + 1);
        const int grain = std::max(1, (M + n_threads - 1) / n_threads);
        pool.parallel_for(0, M, grain, [&](int m) { process_row(m); });
    } else {
        #pragma omp parallel for schedule(static)
        for (int m = 0; m < M; ++m) {
            process_row(m);
        }
    }
#else
    for (int m = 0; m < M; ++m) {
        process_row(m);
    }
#endif
}

py::array_t<float> py_lm_head_gemv_f16(
    py::array_t<uint16_t, py::array::c_style | py::array::forcecast> w_f16,
    py::array_t<float, py::array::c_style | py::array::forcecast> x,
    int M,
    int K
) {
    auto wb = w_f16.request();
    auto xb = x.request();
    if (static_cast<int>(xb.shape[0]) != K) {
        throw std::invalid_argument("x length must equal K");
    }
    if (static_cast<int>(wb.shape[0]) != M || static_cast<int>(wb.shape[1]) != K) {
        throw std::invalid_argument("w_f16 must be shape [M, K]");
    }

    py::array_t<float> y(M);
    auto yb = y.request();

    {
        py::gil_scoped_release release;
        lm_head_gemv_f16_impl(
            static_cast<const uint16_t*>(wb.ptr),
            static_cast<const float*>(xb.ptr),
            static_cast<float*>(yb.ptr),
            M,
            K);
    }
    return y;
}

// py_lm_head_gemv_f16 is registered from gemv_q4_avx2.cpp (_native_gemv module).
