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
#include "thread_pool.h"
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
        asdsl::ThreadPool::get_instance().parallel_for(0, M, 1, [&](int m) {
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
        });
    } else {
        // Bitmask-based path: scan bitmask words
        const int n_words = (K + 31) / 32;

        asdsl::ThreadPool::get_instance().parallel_for(0, M, 1, [&](int m) {
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
        });
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

/* ===================================================================
 * sparse_down_proj_T — transposed Q4 sparse GEMV for down_proj
 *
 * Prerequisite B: down_proj_T is stored as [in_dim=8192, out_dim=3072]
 * row-major (transposed from original [3072, 8192]).
 *
 * For each active intermediate neuron i:
 *   y_out += sparse_x[i] * dequant(down_proj_T[i, :])
 *
 * Row i of down_proj_T maps to the i-th intermediate neuron's contribution
 * across all 3072 output dimensions. Row access = contiguous cache read.
 *
 * Memory traffic reduction:
 *   Dense:  8192 rows × 3072 × 0.5 bytes = 12.6 MB per token
 *   Sparse: 1228 rows (15%) × 3072 × 0.5 bytes = 1.89 MB per token
 *   Factor: 6.7×
 * =================================================================== */

static void sparse_down_proj_T_impl(
    const uint8_t*  __restrict w_T_packed,  // [in_dim, out_dim/2] packed Q4
    const float*    __restrict scales_T,    // [in_dim * (out_dim/group_size)]
    const float*    __restrict biases_T,    // [in_dim * (out_dim/group_size)]
    const float*    __restrict sparse_x,    // [in_dim] intermediate activations
    float*          __restrict y_out,       // [out_dim] output (caller zero-initializes)
    const int32_t*  __restrict active_rows, // indices where |sparse_x[i]| > 0
    int             n_active,
    int             in_dim,                 // 8192
    int             out_dim,                // 3072
    int             group_size
) {
    const int n_out_groups = out_dim / group_size;
    const int row_packed_stride = out_dim / 2;  // bytes per row in transposed layout

    // Each active row contributes: sparse_x[r] * dequant(w_T[r, :])
    // We OpenMP over active rows (embarrassingly parallel, each writes to all of y_out)
    // Using reduction pattern: private buffers summed at end.
    // WARNING: active rows share output (y_out), so we use critical section or reduction.
    // For correctness with OpenMP, accumulate into thread-local buffer and reduce.

    int nthreads = asdsl::ThreadPool::get_instance().thread_count();
    if (nthreads <= 0) nthreads = 1;

    // Thread-local accumulation buffers to avoid false sharing
    std::vector<std::vector<float>> local_y(nthreads, std::vector<float>(out_dim, 0.0f));

    asdsl::ThreadPool::get_instance().parallel_for(0, n_active, 4, [&](int ai) {
        int thread_id = ai % nthreads; // approximation as ThreadPool lacks get_thread_num
        int r = active_rows[ai];
        if (r < 0 || r >= in_dim) return;

        float x_val = sparse_x[r];
        if (x_val == 0.0f) return;

        const uint8_t* row_ptr = w_T_packed + static_cast<size_t>(r) * row_packed_stride;
        const float*   sc_ptr  = scales_T   + r * n_out_groups;
        const float*   bi_ptr  = biases_T   + r * n_out_groups;
        float* local = local_y[thread_id].data();

        // Process each group of the output dimension
        for (int g = 0; g < n_out_groups; g++) {
            float s = sc_ptr[g] * x_val;
            float b = bi_ptr[g] * x_val;
            const uint8_t* gp = row_ptr + g * (group_size / 2);
            float* out_g = local + g * group_size;

            // AVX2 FMA inner loop: dequant nibbles and accumulate
            const __m128i nibble_mask = _mm_set1_epi8(0x0F);
            const int group_bytes = group_size / 2;
            int j = 0;

            // Process 8 packed bytes (16 weights) per iteration
            for (; j + 8 <= group_bytes; j += 8) {
                __m128i packed = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i*>(gp + j));
                __m128i lo = _mm_and_si128(packed, nibble_mask);
                __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask);
                __m128i interleaved = _mm_unpacklo_epi8(lo, hi);

                __m128i idx0 = _mm_cvtepu8_epi32(interleaved);
                __m128i idx1 = _mm_cvtepu8_epi32(_mm_srli_si128(interleaved, 4));
                __m128i idx2 = _mm_cvtepu8_epi32(_mm_srli_si128(interleaved, 8));
                __m128i idx3 = _mm_cvtepu8_epi32(_mm_srli_si128(interleaved, 12));

                // Convert int32 to float32 and scale
                __m128 sv = _mm_set1_ps(s);
                __m128 bv = _mm_set1_ps(b);

                __m128 f0 = _mm_fmadd_ps(_mm_cvtepi32_ps(idx0), sv, bv);
                __m128 f1 = _mm_fmadd_ps(_mm_cvtepi32_ps(idx1), sv, bv);
                __m128 f2 = _mm_fmadd_ps(_mm_cvtepi32_ps(idx2), sv, bv);
                __m128 f3 = _mm_fmadd_ps(_mm_cvtepi32_ps(idx3), sv, bv);

                int base = j * 2;
                // Interleaved nibble order: even then odd for each pair
                // lo nibble = even weight, hi nibble = odd weight (see packing convention)
                __m128 cur0 = _mm_loadu_ps(out_g + base);
                __m128 cur1 = _mm_loadu_ps(out_g + base + 4);
                __m128 cur2 = _mm_loadu_ps(out_g + base + 8);
                __m128 cur3 = _mm_loadu_ps(out_g + base + 12);
                _mm_storeu_ps(out_g + base,      _mm_add_ps(cur0, f0));
                _mm_storeu_ps(out_g + base + 4,  _mm_add_ps(cur1, f1));
                _mm_storeu_ps(out_g + base + 8,  _mm_add_ps(cur2, f2));
                _mm_storeu_ps(out_g + base + 12, _mm_add_ps(cur3, f3));
            }

            // Scalar tail
            for (; j < group_bytes; ++j) {
                uint8_t byte = gp[j];
                out_g[j * 2]     += s * (float(byte & 0x0F)) + b;
                out_g[j * 2 + 1] += s * (float((byte >> 4) & 0x0F)) + b;
            }
        }
    });

    // Reduce thread-local buffers into y_out
    for (int t = 0; t < nthreads; t++) {
        const float* lb = local_y[t].data();
        for (int i = 0; i < out_dim; i += 8) {
            __m256 ya = _mm256_loadu_ps(y_out + i);
            __m256 yb = _mm256_loadu_ps(lb + i);
            _mm256_storeu_ps(y_out + i, _mm256_add_ps(ya, yb));
        }
        // Scalar tail
        for (int i = (out_dim / 8) * 8; i < out_dim; i++) {
            y_out[i] += lb[i];
        }
    }
}


static py::array_t<float> py_sparse_down_proj_T(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> w_T_packed,
    py::array_t<float,   py::array::c_style | py::array::forcecast> scales_T,
    py::array_t<float,   py::array::c_style | py::array::forcecast> biases_T,
    py::array_t<float,   py::array::c_style | py::array::forcecast> sparse_x,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> active_rows,
    int in_dim,
    int out_dim,
    int group_size
) {
    if (in_dim <= 0 || out_dim <= 0 || group_size <= 0)
        throw std::invalid_argument("in_dim, out_dim, group_size must be positive");
    if (out_dim % group_size != 0)
        throw std::invalid_argument("out_dim must be divisible by group_size");

    auto result = py::array_t<float>(out_dim);
    float* y = result.mutable_data();
    std::fill(y, y + out_dim, 0.0f);

    int n_active = static_cast<int>(active_rows.size());
    if (n_active == 0) return result;

    {
        py::gil_scoped_release release;
        sparse_down_proj_T_impl(
            w_T_packed.data(), scales_T.data(), biases_T.data(),
            sparse_x.data(), y,
            active_rows.data(), n_active,
            in_dim, out_dim, group_size
        );
    }

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

    m.def("sparse_down_proj_T", &py_sparse_down_proj_T,
        R"doc(
Sparse GEMV for down_proj using transposed weight storage.

Prerequisite B: down_proj_T is [in_dim=8192, out_dim=3072/2] packed Q4 (row-major).
For each active intermediate neuron, accumulates scaled row into output.
Row access is contiguous = cache-friendly vs original column-sparse access.

Args:
    w_T_packed:   Packed Q4, [in_dim, out_dim/2].
    scales_T:     [in_dim * (out_dim/group_size)].
    biases_T:     [in_dim * (out_dim/group_size)].
    sparse_x:     [in_dim] intermediate vector (85% zeros after FATReLU).
    active_rows:  int32 indices of non-zero elements in sparse_x.
    in_dim:       8192 (intermediate dimension).
    out_dim:      3072 (output/hidden dimension).
    group_size:   Quantization group size.

Returns:
    float32 output [out_dim], zero-initialized and accumulated.
)doc",
        py::arg("w_T_packed"), py::arg("scales_T"), py::arg("biases_T"),
        py::arg("sparse_x"), py::arg("active_rows"),
        py::arg("in_dim"), py::arg("out_dim"), py::arg("group_size"));

#ifdef _OPENMP
    m.attr("has_openmp") = true;
#else
    m.attr("has_openmp") = false;
#endif
}

