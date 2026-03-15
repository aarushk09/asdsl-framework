/**
 * Fused 4-bit GEMV Kernel with AVX2 + FMA for ASDSL Framework
 *
 * Computes y = dequant(W_q4) @ x where:
 *   dequant(w_int) = w_int * scale + bias      (per quantization group)
 *   bias = -zero_point * scale                  (precomputed by caller)
 *
 * Mathematically, for each output element y[m]:
 *   y[m] = sum_g [ scale_g * dot(W_int_g, x_g) + bias_g * sum(x_g) ]
 *
 * This avoids ever materializing the full dequantized weight matrix.
 * Instead we compute the integer dot product in registers, then apply
 * the per-group affine correction using precomputed sum(x) per group.
 *
 * Weight packing: 2 nibbles per byte, low nibble = even index.
 *   byte[i] = (w[2i+1] << 4) | w[2i]
 * This matches ASDSL's _pack_bits(data, 4) output format.
 *
 * SIMD strategy (per 16-value chunk from 8 packed bytes):
 *   1. Load 8 bytes via MOVQ
 *   2. AND 0x0F → low nibbles; PSRLW 4 + AND 0x0F → high nibbles
 *   3. PUNPCKLBW to interleave into linear w[0..15] order
 *   4. PMOVZXBD + CVTDQ2PS to widen uint8→float32 (two groups of 8)
 *   5. VFMADD231PS to accumulate w_float * x_float
 *   6. Horizontal sum + scalar scale/bias correction at group boundary
 *
 * Build flags:
 *   MSVC:      /arch:AVX2 /O2 /fp:fast
 *   GCC/Clang: -mavx2 -mfma -O3 -ffast-math
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <immintrin.h>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <vector>
#include <stdexcept>
#include <algorithm>

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

/**
 * Horizontal sum: reduce 8-wide float vector to scalar.
 * Uses movehdup + movehl to avoid the slow HADDPS instruction.
 */
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

/**
 * Convert the lowest 8 bytes of a __m128i (uint8 values) to __m256
 * of 8 float32 values.  Uses PMOVZXBD for zero-extension to int32,
 * CVTDQ2PS for int32→float, then combines the two 128-bit halves.
 */
static inline __m256 cvt_lo8_u8_ps(__m128i bytes) {
    __m128i i32_lo = _mm_cvtepu8_epi32(bytes);
    __m128i i32_hi = _mm_cvtepu8_epi32(_mm_srli_si128(bytes, 4));
    return _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_cvtepi32_ps(i32_lo)),
        _mm_cvtepi32_ps(i32_hi),
        1
    );
}

/* ===================================================================
 * Core Kernel: Packed 4-bit GEMV
 * =================================================================== */

static void gemv_q4_packed_impl(
    const uint8_t* __restrict w_packed,
    const float*   __restrict x,
    const float*   __restrict scales,
    const float*   __restrict biases,
    float*         __restrict y,
    int M, int K, int group_size
) {
    const int groups_per_row = K / group_size;
    const int packed_stride  = K / 2;
    const __m128i nibble_mask = _mm_set1_epi8(0x0F);

    // Phase 1: precompute sum(x) per quantization group.
    // This term is shared across all M rows, so we pay O(K) once
    // instead of O(M*K) if computed inline.
    std::vector<float> group_sum_x(groups_per_row);
    for (int g = 0; g < groups_per_row; ++g) {
        __m256 acc = _mm256_setzero_ps();
        const float* xg = x + g * group_size;
        for (int j = 0; j < group_size; j += 8) {
            acc = _mm256_add_ps(acc, _mm256_loadu_ps(xg + j));
        }
        group_sum_x[g] = hsum256_ps(acc);
    }

    // Phase 2: row-parallel fused dequant + dot product.
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; ++m) {
        const uint8_t* row = w_packed + static_cast<size_t>(m) * packed_stride;
        float row_sum = 0.0f;

        for (int g = 0; g < groups_per_row; ++g) {
            const int k0   = g * group_size;
            const int gidx = m * groups_per_row + g;

            __m256 dot = _mm256_setzero_ps();

            for (int j = 0; j < group_size; j += 16) {
                const uint8_t* pp = row + (k0 + j) / 2;
                const float*   xp = x + k0 + j;

                // Load 8 packed bytes → 16 four-bit values.
                // MOVQ loads only the low 64 bits; upper 64 zeroed.
                __m128i raw = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i*>(pp));

                // Nibble extraction via 16-bit lane shift.
                // After AND mask, cross-byte contamination is removed.
                __m128i lo = _mm_and_si128(raw, nibble_mask);
                __m128i hi = _mm_and_si128(
                    _mm_srli_epi16(raw, 4), nibble_mask);

                // Byte interleave → linear order: w[0],w[1],w[2],...
                __m128i vals = _mm_unpacklo_epi8(lo, hi);

                // Convert first 8 uint8 → 8 float32, FMA with x
                dot = _mm256_fmadd_ps(
                    cvt_lo8_u8_ps(vals),
                    _mm256_loadu_ps(xp),
                    dot);

                // Convert next 8 uint8 → 8 float32, FMA with x+8
                dot = _mm256_fmadd_ps(
                    cvt_lo8_u8_ps(_mm_srli_si128(vals, 8)),
                    _mm256_loadu_ps(xp + 8),
                    dot);

                // Prefetch next weight cache line (64 bytes ahead)
                _mm_prefetch(
                    reinterpret_cast<const char*>(pp + 64), _MM_HINT_T0);
            }

            // Per-group affine correction:
            //   row_sum += dot(W_int, x) * scale + bias * sum(x)
            float d = hsum256_ps(dot);
            row_sum += d * scales[gidx] + biases[gidx] * group_sum_x[g];
        }

        y[m] = row_sum;
    }
}

/* ===================================================================
 * Core Kernel: Unpacked uint8 GEMV (drop-in for WeightStore path)
 * =================================================================== */

static void gemv_q4_unpacked_impl(
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
        for (int j = 0; j < group_size; j += 8) {
            acc = _mm256_add_ps(acc, _mm256_loadu_ps(xg + j));
        }
        group_sum_x[g] = hsum256_ps(acc);
    }

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; ++m) {
        const uint8_t* row = w + static_cast<size_t>(m) * K;
        float row_sum = 0.0f;

        for (int g = 0; g < groups_per_row; ++g) {
            const int k0   = g * group_size;
            const int gidx = m * groups_per_row + g;

            __m256 dot = _mm256_setzero_ps();

            // No nibble unpacking needed: each uint8 is one value
            for (int j = 0; j < group_size; j += 8) {
                __m128i bytes = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i*>(row + k0 + j));
                dot = _mm256_fmadd_ps(
                    cvt_lo8_u8_ps(bytes),
                    _mm256_loadu_ps(x + k0 + j),
                    dot);
            }

            float d = hsum256_ps(dot);
            row_sum += d * scales[gidx] + biases[gidx] * group_sum_x[g];

            _mm_prefetch(
                reinterpret_cast<const char*>(row + k0 + group_size + 64),
                _MM_HINT_T0);
        }

        y[m] = row_sum;
    }
}

/* ===================================================================
 * Runtime CPU Feature Detection (CPUID)
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
 * Input validation helper
 * =================================================================== */

static void validate_gemv_args(
    int64_t w_size, int64_t x_size,
    int64_t s_size, int64_t b_size,
    int M, int K, int group_size,
    int64_t expected_w_size
) {
    if (M <= 0 || K <= 0 || group_size <= 0)
        throw std::invalid_argument("M, K, group_size must be positive");
    if (K % 2 != 0)
        throw std::invalid_argument("K must be even for 4-bit packing");
    if (K % group_size != 0)
        throw std::invalid_argument("K must be divisible by group_size");
    if (group_size % 8 != 0)
        throw std::invalid_argument(
            "group_size must be a multiple of 8 for AVX2 vectorization");
    if (x_size != K)
        throw std::invalid_argument("x length must equal K");
    if (w_size != expected_w_size)
        throw std::invalid_argument("weight buffer size mismatch");

    int64_t expected_groups = static_cast<int64_t>(M) * (K / group_size);
    if (s_size != expected_groups)
        throw std::invalid_argument("scales size mismatch");
    if (b_size != expected_groups)
        throw std::invalid_argument("biases size mismatch");
}

/* ===================================================================
 * pybind11 Bindings
 * =================================================================== */

static py::array_t<float> py_gemv_q4_packed(
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

    validate_gemv_args(
        wb.size, xb.size, sb.size, bb.size,
        M, K, group_size,
        static_cast<int64_t>(M) * K / 2);

    auto result = py::array_t<float>(M);
    auto rb = result.request();

    {
        py::gil_scoped_release release;
        gemv_q4_packed_impl(
            static_cast<const uint8_t*>(wb.ptr),
            static_cast<const float*>(xb.ptr),
            static_cast<const float*>(sb.ptr),
            static_cast<const float*>(bb.ptr),
            static_cast<float*>(rb.ptr),
            M, K, group_size);
    }

    return result;
}

static py::array_t<float> py_gemv_q4_unpacked(
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

    validate_gemv_args(
        wb.size, xb.size, sb.size, bb.size,
        M, K, group_size,
        static_cast<int64_t>(M) * K);

    auto result = py::array_t<float>(M);
    auto rb = result.request();

    {
        py::gil_scoped_release release;
        gemv_q4_unpacked_impl(
            static_cast<const uint8_t*>(wb.ptr),
            static_cast<const float*>(xb.ptr),
            static_cast<const float*>(sb.ptr),
            static_cast<const float*>(bb.ptr),
            static_cast<float*>(rb.ptr),
            M, K, group_size);
    }

    return result;
}

PYBIND11_MODULE(_native_gemv, m) {
    m.doc() = R"doc(
Fused 4-bit GEMV kernels with AVX2 + FMA for ASDSL.

Provides two kernel variants:
  - gemv_q4_packed:   operates on 4-bit packed weights (2 values/byte)
  - gemv_q4_unpacked: operates on pre-unpacked uint8 weights (1 value/byte)

Both compute: y = dequant(W) @ x
where dequant(w) = w * scale + bias  (per quantization group)
)doc";

    m.def("gemv_q4_packed", &py_gemv_q4_packed,
        R"doc(
Fused 4-bit GEMV on packed weights.

Args:
    w_packed: Packed uint8 array, shape (M*K/2,). Two 4-bit values per byte.
    x:        Input vector, shape (K,), float32.
    scales:   Per-group scales, shape (M * K/group_size,), float32.
    biases:   Per-group biases, shape (M * K/group_size,), float32.
              Precomputed as -zero_point * scale.
    M:        Number of output rows.
    K:        Input dimension (columns).
    group_size: Quantization group size (must be multiple of 16).

Returns:
    Output vector, shape (M,), float32.
)doc",
        py::arg("w_packed"), py::arg("x"),
        py::arg("scales"), py::arg("biases"),
        py::arg("M"), py::arg("K"), py::arg("group_size"));

    m.def("gemv_q4_unpacked", &py_gemv_q4_unpacked,
        R"doc(
Fused 4-bit GEMV on pre-unpacked uint8 weights.

Same interface as gemv_q4_packed, but w is shape (M*K,) with
one byte per quantized value. Drop-in replacement for the
existing WeightStore._matvec_quant path.
)doc",
        py::arg("w"), py::arg("x"),
        py::arg("scales"), py::arg("biases"),
        py::arg("M"), py::arg("K"), py::arg("group_size"));

    m.def("check_avx2", &check_avx2_support,
        "Runtime check: does this CPU support AVX2?");

    m.def("check_fma", &check_fma_support,
        "Runtime check: does this CPU support FMA3?");

#ifdef _OPENMP
    m.def("get_num_threads", []() { return omp_get_max_threads(); },
        "Get max OpenMP thread count.");
    m.def("set_num_threads", [](int n) { omp_set_num_threads(n); },
        "Set OpenMP thread count.", py::arg("n"));
    m.attr("has_openmp") = true;
#else
    m.def("get_num_threads", []() { return 1; });
    m.def("set_num_threads", [](int) {},
        "No-op: built without OpenMP.", py::arg("n"));
    m.attr("has_openmp") = false;
#endif
}
