/**
 * preq2 GEMV — 64B-aligned interleaved quant + sidecar meta + AVX-VNNI.
 * Bit-identical dequant vs Q4_32 preq when repacked from same source blocks.
 */

#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <cstdlib>

#include "gemv_q4_kernel.h"
#include "gemv_chunked.hpp"
#include "engine_flags.hpp"
#include "thread_pool.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(_MSC_VER) && defined(__AVX2__)
#  if _MSC_VER >= 1928
#    define HAVE_AVXVNNI 1
#  endif
#elif defined(__AVXVNNI__)
#  define HAVE_AVXVNNI 1
#endif

namespace {

static constexpr int PREQ2_GS = 32;
static constexpr int PREQ2_META_BYTES = 4;
static constexpr int PREQ2_GROUP_QUANT = 64;
static constexpr int PREQ2_ROW_BAND = 4;

static inline float cvtsh_ss(uint16_t h) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<int>(h))));
}

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

static inline __m256i decode_nibbles_256(const uint8_t* packed) {
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i pk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed));
    const __m128i lo = _mm_and_si128(pk, mask);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(pk, 4), mask);
    return _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi));
}

static inline __m256i dot_group_vnni(const uint8_t* nibbles, const int8_t* xq) {
    const __m256i w = decode_nibbles_256(nibbles);
    const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq));
#ifdef HAVE_AVXVNNI
    return _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), w, x);
#else
    const __m256i ones = _mm256_set1_epi16(1);
    return _mm256_madd_epi16(_mm256_maddubs_epi16(w, x), ones);
#endif
}

static inline void compute_x_sums_scaled(
    const int8_t* x_q8,
    const float* x_scales,
    int n_groups,
    int32_t* x_sums,
    float* xsum_scaled) {
    for (int g = 0; g < n_groups; ++g) {
        const int8_t* xg = x_q8 + g * PREQ2_GS;
        __m256i vsum = _mm256_setzero_si256();
        for (int j = 0; j < PREQ2_GS; j += 8) {
            vsum = _mm256_add_epi32(
                vsum,
                _mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(xg + j))));
        }
        __m128i lo = _mm256_castsi256_si128(vsum);
        __m128i hi = _mm256_extracti128_si256(vsum, 1);
        lo = _mm_add_epi32(lo, hi);
        lo = _mm_hadd_epi32(lo, lo);
        lo = _mm_hadd_epi32(lo, lo);
        const int32_t xs = _mm_cvtsi128_si32(lo);
        x_sums[g] = xs;
        xsum_scaled[g] = static_cast<float>(xs) * x_scales[g];
    }
}

static inline void accumulate_row_groups(
    float& out,
    const uint8_t* meta_row,
    const uint8_t* quant_row_nibbles,
    const int8_t* x_q8,
    const float* x_scales,
    const float* xsum_scaled,
    int n_groups) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    float corr0 = 0.0f;
    float corr1 = 0.0f;

    for (int g = 0; g < n_groups; g += 2) {
        for (int sub = 0; sub < 2; ++sub) {
            const int gi = g + sub;
            if (gi >= n_groups) {
                break;
            }
            const uint8_t* meta = meta_row + gi * PREQ2_META_BYTES;
            uint16_t sf = 0;
            uint16_t zf = 0;
            std::memcpy(&sf, meta, 2);
            std::memcpy(&zf, meta + 2, 2);
            const float ws = cvtsh_ss(sf);
            const float wz = cvtsh_ss(zf);
            const float xs = x_scales[gi];
            const float combined = ws * xs;
            const float corr_term = wz * ws * xsum_scaled[gi];

            const __m256i dot_i = dot_group_vnni(
                quant_row_nibbles + gi * 16,
                x_q8 + gi * PREQ2_GS);
            const __m256 term = _mm256_mul_ps(
                _mm256_cvtepi32_ps(dot_i),
                _mm256_set1_ps(combined));

            if (sub == 0) {
                acc0 = _mm256_add_ps(acc0, term);
                corr0 += corr_term;
            } else {
                acc1 = _mm256_add_ps(acc1, term);
                corr1 += corr_term;
            }
        }
    }

    out = hsum256_ps(acc0) + hsum256_ps(acc1) - corr0 - corr1;
}

static inline void accumulate_band_4rows(
    float* y,
    int row0,
    const uint8_t* meta_base,
    int meta_row_stride,
    const uint8_t* quant_band,
    int n_groups,
    const int8_t* x_q8,
    const float* x_scales,
    const float* xsum_scaled) {
    const uint8_t* meta0 = meta_base + static_cast<size_t>(row0) * meta_row_stride;
    const uint8_t* meta1 = meta_base + static_cast<size_t>(row0 + 1) * meta_row_stride;
    const uint8_t* meta2 = meta_base + static_cast<size_t>(row0 + 2) * meta_row_stride;
    const uint8_t* meta3 = meta_base + static_cast<size_t>(row0 + 3) * meta_row_stride;

    for (int r = 0; r < PREQ2_ROW_BAND; ++r) {
        y[row0 + r] = 0.0f;
    }

    for (int g = 0; g < n_groups; ++g) {
        const uint8_t* q64 = quant_band + static_cast<size_t>(g) * PREQ2_GROUP_QUANT;
        const __m256i x_all = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_q8 + g * PREQ2_GS));
        const float xs = x_scales[g];
        const float xsum_s = xsum_scaled[g];

        const uint8_t* metas[PREQ2_ROW_BAND] = {meta0, meta1, meta2, meta3};
        for (int r = 0; r < PREQ2_ROW_BAND; ++r) {
            const uint8_t* meta = metas[r] + g * PREQ2_META_BYTES;
            uint16_t sf = 0;
            uint16_t zf = 0;
            std::memcpy(&sf, meta, 2);
            std::memcpy(&zf, meta + 2, 2);
            const float ws = cvtsh_ss(sf);
            const float wz = cvtsh_ss(zf);
            const float combined = ws * xs;

            const __m256i w = decode_nibbles_256(q64 + r * 16);
#ifdef HAVE_AVXVNNI
            const __m256i dot_i = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), w, x_all);
#else
            const __m256i ones = _mm256_set1_epi16(1);
            const __m256i dot_i = _mm256_madd_epi16(_mm256_maddubs_epi16(w, x_all), ones);
#endif
            const float dot_f = hsum256_ps(_mm256_mul_ps(
                _mm256_cvtepi32_ps(dot_i),
                _mm256_set1_ps(combined)));
            y[row0 + r] += dot_f - wz * ws * xsum_s;
        }
    }
}

static void gemv_preq2_from_q8_impl(
    const uint8_t* meta,
    const uint8_t* quant,
    const int8_t* x_q8,
    const float* x_scales,
    float* y,
    int out_features,
    int in_features) {
    const int n_groups = in_features / PREQ2_GS;
    const size_t meta_row_stride = static_cast<size_t>(n_groups) * PREQ2_META_BYTES;
    const int n_bands = (out_features + PREQ2_ROW_BAND - 1) / PREQ2_ROW_BAND;
    const size_t band_stride = static_cast<size_t>(n_groups) * PREQ2_GROUP_QUANT;

    int32_t x_sums[640];
    float xsum_scaled[640];
    if (n_groups > 640) {
        throw std::invalid_argument("gemv_preq2: n_groups exceeds buffer");
    }
    compute_x_sums_scaled(x_q8, x_scales, n_groups, x_sums, xsum_scaled);

    auto process_one_row = [&](int row) {
        const uint8_t* meta_row = meta + static_cast<size_t>(row) * meta_row_stride;
        const int band = row / PREQ2_ROW_BAND;
        const int slot = row % PREQ2_ROW_BAND;
        const uint8_t* qband = quant + static_cast<size_t>(band) * band_stride;
        float acc = 0.0f;
        __m256 acc_v = _mm256_setzero_ps();
        __m256 acc_v2 = _mm256_setzero_ps();
        float corr = 0.0f;
        float corr2 = 0.0f;
        for (int g = 0; g < n_groups; ++g) {
            const uint8_t* meta_g = meta_row + g * PREQ2_META_BYTES;
            uint16_t sf = 0;
            uint16_t zf = 0;
            std::memcpy(&sf, meta_g, 2);
            std::memcpy(&zf, meta_g + 2, 2);
            const float ws = cvtsh_ss(sf);
            const float wz = cvtsh_ss(zf);
            const float xs = x_scales[g];
            const float combined = ws * xs;
            const float corr_term = wz * ws * xsum_scaled[g];
            const __m256i dot_i = dot_group_vnni(
                qband + g * PREQ2_GROUP_QUANT + slot * 16,
                x_q8 + g * PREQ2_GS);
            const __m256 term = _mm256_mul_ps(
                _mm256_cvtepi32_ps(dot_i),
                _mm256_set1_ps(combined));
            if ((g & 1) == 0) {
                acc_v = _mm256_add_ps(acc_v, term);
                corr += corr_term;
            } else {
                acc_v2 = _mm256_add_ps(acc_v2, term);
                corr2 += corr_term;
            }
        }
        acc = hsum256_ps(acc_v) + hsum256_ps(acc_v2) - corr - corr2;
        y[row] = acc;
    };

    auto process_range = [&](int row_begin, int row_end) {
        int row = row_begin;
        while (row + PREQ2_ROW_BAND <= row_end && row + PREQ2_ROW_BAND <= out_features) {
            accumulate_band_4rows(
                y, row, meta, meta_row_stride, quant, n_groups,
                x_q8, x_scales, xsum_scaled);
            row += PREQ2_ROW_BAND;
        }
        for (; row < row_end; ++row) {
            process_one_row(row);
        }
    };

#ifdef _OPENMP
    if (asdsl::persistent_pool_enabled() && asdsl::tl_active_pool != nullptr) {
        asdsl::ThreadPool& pool = asdsl::ThreadPool::get_instance();
        const int n_threads = std::max(1, pool.thread_count() + 1);
        const int chunk_size = asdsl_chunked::compute_chunk_size(
            out_features, PREQ2_ROW_BAND, n_threads);
        const int n_chunks = (out_features + chunk_size - 1) / chunk_size;
        pool.parallel_for(0, n_chunks, 1, [&](int chunk) {
            const int rb = chunk * chunk_size;
            const int re = std::min(rb + chunk_size, out_features);
            process_range(rb, re);
        });
    } else if (asdsl_chunked::chunked_gemv_enabled()) {
        // 4-row band kernel must not cross chunked row boundaries (race on y[]);
        // use per-row path under atomic chunk scheduling.
        asdsl_chunked::parallel_row_chunks(out_features, PREQ2_ROW_BAND, [&](int rb, int re) {
            for (int row = rb; row < re; ++row) {
                process_one_row(row);
            }
        });
    } else {
        #pragma omp parallel for schedule(static)
        for (int row = 0; row < out_features; ++row) {
            process_one_row(row);
        }
    }
#else
    process_range(0, out_features);
#endif
}

}  // namespace

void gemv_preq2_fused_avx2(
    const uint8_t* meta,
    const uint8_t* quant,
    const float* x_fp32,
    float* y,
    int out_features,
    int in_features,
    int group_size) {
    if (!meta || !quant || !x_fp32 || !y) {
        throw std::invalid_argument("gemv_preq2_fused_avx2: null pointer");
    }
    if (group_size != 32) {
        throw std::invalid_argument("gemv_preq2_fused_avx2: group_size must be 32");
    }
    thread_local static int8_t tl_x_q8[20480];
    thread_local static float tl_x_scales[640];
    quantize_activation_avx2(x_fp32, tl_x_q8, tl_x_scales, in_features, group_size);
    gemv_preq2_from_q8_impl(meta, quant, tl_x_q8, tl_x_scales, y, out_features, in_features);
}

void gemv_preq2_avx2(
    const uint8_t* meta,
    const uint8_t* quant,
    const int8_t* x_q8,
    const float* x_scales,
    float* y,
    int out_features,
    int in_features,
    int group_size) {
    if (!meta || !quant || !x_q8 || !x_scales || !y) {
        throw std::invalid_argument("gemv_preq2_avx2: null pointer");
    }
    if (group_size != 32) {
        throw std::invalid_argument("gemv_preq2_avx2: group_size must be 32");
    }
    gemv_preq2_from_q8_impl(meta, quant, x_q8, x_scales, y, out_features, in_features);
}
