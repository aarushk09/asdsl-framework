/* Q4_32 preq GEMV — Phase 15/22 optimized path (linked with gemv_q4_avx2.cpp). */

#include <immintrin.h>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <cstdlib>

#include "gemv_q4_kernel.h"
#include "gemv_chunked.hpp"
#include "omp_pcore_pinning.hpp"

#ifdef _OPENMP
#include <omp.h>
#include "thread_pool.h"
#endif

namespace asdsl_preq {

static std::atomic<uint64_t> g_preq_classic_accum_calls{0};
static std::atomic<uint64_t> g_preq_xloaded_accum_calls{0};

/** Groups ahead to prefetch (0 = off). Default 8; tune via ASDSL_PREQ_PREFETCH_GROUPS. */
static int preq_prefetch_groups_ahead() {
    static int ahead = -1;
    if (ahead < 0) {
        const char* v = std::getenv("ASDSL_PREQ_PREFETCH_GROUPS");
        if (!v || v[0] == '\0') {
            ahead = 8;
        } else if (v[0] == '0') {
            ahead = 0;
        } else {
            ahead = std::atoi(v);
            if (ahead < 0) {
                ahead = 0;
            }
            if (ahead > 16) {
                ahead = 16;
            }
        }
    }
    return ahead;
}

static bool preq_g4fused_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char* v = std::getenv("ASDSL_PREQ_G4FUSED");
        if (!v || v[0] == '\0') {
            enabled = 0;
        } else if (v[0] == '0') {
            enabled = 0;
        } else {
            enabled = 1;
        }
    }
    return enabled != 0;
}

/** When set, 4-group inner loop runs only on large row counts (e.g. gate_up). */
static bool preq_g4_gate_up_only() {
    static int gate_up_only = -1;
    if (gate_up_only < 0) {
        const char* v = std::getenv("ASDSL_PREQ_G4GATE_UP_ONLY");
        gate_up_only = (v && v[0] == '1') ? 1 : 0;
    }
    return gate_up_only != 0;
}

/** Phase G regression fix: g4 tiling regresses gate_up E2E; keep pre-G default off. */
static bool preq_use_g4_inner(int out_features, bool force_g4) {
    if (force_g4) {
        return true;
    }
    if (preq_g4_gate_up_only()) {
        return out_features >= 8192;
    }
    return preq_g4fused_enabled();
}

static inline void preq_prefetch_groups(
    const uint8_t* const* row_blocks,
    int nrows,
    int prefetch_g,
    int block_size,
    const int8_t* x_q8,
    int group_size) {
    if (prefetch_g < 0) {
        return;
    }
    for (int r = 0; r < nrows; ++r) {
        _mm_prefetch(
            reinterpret_cast<const char*>(row_blocks[r] + static_cast<size_t>(prefetch_g) * block_size),
            _MM_HINT_T0);
    }
    _mm_prefetch(
        reinterpret_cast<const char*>(x_q8 + prefetch_g * group_size),
        _MM_HINT_T0);
}

static inline int32_t hsum256_epi32(__m256i v) {
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i lo = _mm256_castsi256_si128(v);
    lo = _mm_add_epi32(lo, hi);
    __m128i shuf = _mm_shuffle_epi32(lo, _MM_SHUFFLE(1, 0, 3, 2));
    lo = _mm_add_epi32(lo, shuf);
    shuf = _mm_shuffle_epi32(lo, _MM_SHUFFLE(0, 1, 0, 1));
    lo = _mm_add_epi32(lo, shuf);
    return _mm_cvtsi128_si32(lo);
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


static inline float preq_cvtsh_ss(uint16_t h) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<int>(h))));
}

static inline __m256i preq_dot_group_maddubs_avx2_xloaded(
    const uint8_t* w_group_nibbles,
    __m256i x_all) {
    const __m128i mask_nibble = _mm_set1_epi8(0x0F);
    const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w_group_nibbles));
    const __m128i lo = _mm_and_si128(packed, mask_nibble);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), mask_nibble);
    const __m256i w = _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi));
    const __m256i ones_16 = _mm256_set1_epi16(1);
    const __m256i prod16 = _mm256_maddubs_epi16(w, x_all);
    return _mm256_madd_epi16(prod16, ones_16);
}

static inline __m256i preq_dot_group_maddubs_avx2(
    const uint8_t* w_group_nibbles,
    const int8_t* x_group_q8) {
    return preq_dot_group_maddubs_avx2_xloaded(
        w_group_nibbles,
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_group_q8)));
}

static inline float hsum128_ps(__m128 v) {
    __m128 shuf = _mm_movehdup_ps(v);
    v = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, v);
    v = _mm_add_ss(v, shuf);
    return _mm_cvtss_f32(v);
}

static inline void preq_row_accumulate_g4(
    float& acc_scalar,
    float& corr_scalar,
    const uint8_t* row_blocks,
    int g,
    int block_size,
    const int8_t* x_q8,
    const float* x_scales,
    const int32_t* x_sums) {
    const uint8_t* b0 = row_blocks + static_cast<size_t>(g) * block_size;
    const uint8_t* b1 = b0 + block_size;
    const uint8_t* b2 = b1 + block_size;
    const uint8_t* b3 = b2 + block_size;

    uint16_t sh0 = 0, zh0 = 0, sh1 = 0, zh1 = 0, sh2 = 0, zh2 = 0, sh3 = 0, zh3 = 0;
    std::memcpy(&sh0, b0, 2);
    std::memcpy(&zh0, b0 + 2, 2);
    std::memcpy(&sh1, b1, 2);
    std::memcpy(&zh1, b1 + 2, 2);
    std::memcpy(&sh2, b2, 2);
    std::memcpy(&zh2, b2 + 2, 2);
    std::memcpy(&sh3, b3, 2);
    std::memcpy(&zh3, b3 + 2, 2);

    const float ws0 = preq_cvtsh_ss(sh0);
    const float ws1 = preq_cvtsh_ss(sh1);
    const float ws2 = preq_cvtsh_ss(sh2);
    const float ws3 = preq_cvtsh_ss(sh3);
    const float wz0 = preq_cvtsh_ss(zh0);
    const float wz1 = preq_cvtsh_ss(zh1);
    const float wz2 = preq_cvtsh_ss(zh2);
    const float wz3 = preq_cvtsh_ss(zh3);

    const int8_t* xq0 = x_q8 + (g + 0) * 32;
    const int8_t* xq1 = x_q8 + (g + 1) * 32;
    const int8_t* xq2 = x_q8 + (g + 2) * 32;
    const int8_t* xq3 = x_q8 + (g + 3) * 32;

    const float xs0 = x_scales[g + 0];
    const float xs1 = x_scales[g + 1];
    const float xs2 = x_scales[g + 2];
    const float xs3 = x_scales[g + 3];

    const int32_t d0 = hsum256_epi32(preq_dot_group_maddubs_avx2(b0 + 4, xq0));
    const int32_t d1 = hsum256_epi32(preq_dot_group_maddubs_avx2(b1 + 4, xq1));
    const int32_t d2 = hsum256_epi32(preq_dot_group_maddubs_avx2(b2 + 4, xq2));
    const int32_t d3 = hsum256_epi32(preq_dot_group_maddubs_avx2(b3 + 4, xq3));

    const __m128 dot_vec = _mm_set_ps(
        static_cast<float>(d3), static_cast<float>(d2),
        static_cast<float>(d1), static_cast<float>(d0));
    const __m128 ws_vec = _mm_set_ps(ws3, ws2, ws1, ws0);
    const __m128 wz_vec = _mm_set_ps(wz3, wz2, wz1, wz0);
    const __m128 xs_vec = _mm_set_ps(xs3, xs2, xs1, xs0);
    const __m128 xsum_vec = _mm_set_ps(
        static_cast<float>(x_sums[g + 3]), static_cast<float>(x_sums[g + 2]),
        static_cast<float>(x_sums[g + 1]), static_cast<float>(x_sums[g + 0]));

    const __m128 term = _mm_sub_ps(
        _mm_mul_ps(_mm_mul_ps(dot_vec, ws_vec), xs_vec),
        _mm_mul_ps(_mm_mul_ps(wz_vec, xsum_vec), xs_vec));
    acc_scalar += hsum128_ps(term);
    corr_scalar += hsum128_ps(_mm_mul_ps(_mm_mul_ps(wz_vec, xsum_vec), xs_vec));
}

/** Phase G: load x_q8 for 4 groups once, accumulate NRows output rows (shared activation tile). */
template <int NRows>
static inline void preq_tile_accumulate_g4(
    float acc_g4[NRows],
    float corr[NRows],
    const uint8_t* const row_blocks[NRows],
    int g,
    int block_size,
    const int8_t* x_q8,
    const float* x_scales,
    const int32_t* x_sums) {
    const __m256i x_loaded[4] = {
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_q8 + (g + 0) * 32)),
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_q8 + (g + 1) * 32)),
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_q8 + (g + 2) * 32)),
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_q8 + (g + 3) * 32)),
    };
    const float xs0 = x_scales[g + 0];
    const float xs1 = x_scales[g + 1];
    const float xs2 = x_scales[g + 2];
    const float xs3 = x_scales[g + 3];
    const float xsum0 = static_cast<float>(x_sums[g + 0]);
    const float xsum1 = static_cast<float>(x_sums[g + 1]);
    const float xsum2 = static_cast<float>(x_sums[g + 2]);
    const float xsum3 = static_cast<float>(x_sums[g + 3]);
    const __m128 xs_vec = _mm_set_ps(xs3, xs2, xs1, xs0);
    const __m128 xsum_vec = _mm_set_ps(xsum3, xsum2, xsum1, xsum0);
    const __m128 xs_xsum = _mm_mul_ps(xs_vec, xsum_vec);

    for (int r = 0; r < NRows; ++r) {
        const uint8_t* row_base = row_blocks[r];
        const uint8_t* b0 = row_base + static_cast<size_t>(g) * block_size;
        const uint8_t* b1 = b0 + block_size;
        const uint8_t* b2 = b1 + block_size;
        const uint8_t* b3 = b2 + block_size;

        uint16_t sh0 = 0, zh0 = 0, sh1 = 0, zh1 = 0, sh2 = 0, zh2 = 0, sh3 = 0, zh3 = 0;
        std::memcpy(&sh0, b0, 2);
        std::memcpy(&zh0, b0 + 2, 2);
        std::memcpy(&sh1, b1, 2);
        std::memcpy(&zh1, b1 + 2, 2);
        std::memcpy(&sh2, b2, 2);
        std::memcpy(&zh2, b2 + 2, 2);
        std::memcpy(&sh3, b3, 2);
        std::memcpy(&zh3, b3 + 2, 2);

        const float ws0 = preq_cvtsh_ss(sh0);
        const float ws1 = preq_cvtsh_ss(sh1);
        const float ws2 = preq_cvtsh_ss(sh2);
        const float ws3 = preq_cvtsh_ss(sh3);
        const float wz0 = preq_cvtsh_ss(zh0);
        const float wz1 = preq_cvtsh_ss(zh1);
        const float wz2 = preq_cvtsh_ss(zh2);
        const float wz3 = preq_cvtsh_ss(zh3);

        const int32_t d0 = hsum256_epi32(preq_dot_group_maddubs_avx2_xloaded(b0 + 4, x_loaded[0]));
        const int32_t d1 = hsum256_epi32(preq_dot_group_maddubs_avx2_xloaded(b1 + 4, x_loaded[1]));
        const int32_t d2 = hsum256_epi32(preq_dot_group_maddubs_avx2_xloaded(b2 + 4, x_loaded[2]));
        const int32_t d3 = hsum256_epi32(preq_dot_group_maddubs_avx2_xloaded(b3 + 4, x_loaded[3]));

        const __m128 dot_vec = _mm_set_ps(
            static_cast<float>(d3), static_cast<float>(d2),
            static_cast<float>(d1), static_cast<float>(d0));
        const __m128 ws_vec = _mm_set_ps(ws3, ws2, ws1, ws0);
        const __m128 wz_vec = _mm_set_ps(wz3, wz2, wz1, wz0);

        const __m128 term = _mm_sub_ps(
            _mm_mul_ps(_mm_mul_ps(dot_vec, ws_vec), xs_vec),
            _mm_mul_ps(wz_vec, xs_xsum));
        acc_g4[r] += hsum128_ps(term);
        corr[r] += hsum128_ps(_mm_mul_ps(wz_vec, xs_xsum));
    }
}

static inline void preq_row_accumulate_one_group_xloaded(
    __m256& acc_f,
    float& corr_scalar,
    const uint8_t* block,
    __m256i x_loaded,
    float x_scale,
    int32_t x_sum_g) {
    g_preq_xloaded_accum_calls.fetch_add(1, std::memory_order_relaxed);
    uint16_t scale_fp16 = 0;
    uint16_t zero_fp16 = 0;
    std::memcpy(&scale_fp16, block, 2);
    std::memcpy(&zero_fp16, block + 2, 2);
    const float w_scale = preq_cvtsh_ss(scale_fp16);
    const float w_zero = preq_cvtsh_ss(zero_fp16);
    const __m256i dot_int = preq_dot_group_maddubs_avx2_xloaded(block + 4, x_loaded);
    const __m256 scale_v = _mm256_set1_ps(w_scale * x_scale);
    acc_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot_int), scale_v, acc_f);
    corr_scalar += w_zero * static_cast<float>(x_sum_g) * w_scale * x_scale;
}

static inline void preq_row_accumulate_one_group(
    __m256& acc_f,
    float& corr_scalar,
    const uint8_t* block,
    const int8_t* x_group_q8,
    float x_scale,
    int32_t x_sum_g) {
    g_preq_classic_accum_calls.fetch_add(1, std::memory_order_relaxed);
    uint16_t scale_fp16 = 0;
    uint16_t zero_fp16 = 0;
    std::memcpy(&scale_fp16, block, 2);
    std::memcpy(&zero_fp16, block + 2, 2);
    const float w_scale = preq_cvtsh_ss(scale_fp16);
    const float w_zero = preq_cvtsh_ss(zero_fp16);
    const __m256i dot_int = preq_dot_group_maddubs_avx2(block + 4, x_group_q8);
    const __m256 scale_v = _mm256_set1_ps(w_scale * x_scale);
    acc_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot_int), scale_v, acc_f);
    corr_scalar += w_zero * static_cast<float>(x_sum_g) * w_scale * x_scale;
}

template <int NRows, bool ForceG4Fused = false>
static void gemv_q4_32_preq_nrow_from_q8_impl(
    const uint8_t* blocks,
    const int8_t* x_q8,
    const float* x_scales,
    float* y,
    int out_features,
    int in_features,
    int group_size) {
    static_assert(NRows == 4 || NRows == 8, "NRows must be 4 or 8");
    const int n_groups = in_features / group_size;
    const int block_size = 20;
    const size_t row_stride = static_cast<size_t>(n_groups) * static_cast<size_t>(block_size);
    const int n_chunks = out_features / NRows;
    const bool use_g4 = (n_groups >= 4) && ((n_groups % 4) == 0)
        && preq_use_g4_inner(out_features, ForceG4Fused);
    const int g4_end = use_g4 ? ((n_groups / 4) * 4) : 0;

    int32_t x_sums_buf[640];
    for (int g = 0; g < n_groups; ++g) {
        const int8_t* xg = x_q8 + g * group_size;
        __m256i vsum = _mm256_setzero_si256();
        for (int j = 0; j < group_size; j += 8) {
            vsum = _mm256_add_epi32(
                vsum,
                _mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(xg + j))));
        }
        __m128i lo128 = _mm256_castsi256_si128(vsum);
        __m128i hi128 = _mm256_extracti128_si256(vsum, 1);
        lo128 = _mm_add_epi32(lo128, hi128);
        lo128 = _mm_hadd_epi32(lo128, lo128);
        lo128 = _mm_hadd_epi32(lo128, lo128);
        x_sums_buf[g] = _mm_cvtsi128_si32(lo128);
    }

    auto process_chunk = [&](int chunk_begin, int chunk_end) {
    for (int chunk = chunk_begin; chunk < chunk_end; ++chunk) {
        const int row0 = chunk * NRows;
        __m256 acc_f[NRows];
        float corr[NRows];
        float acc_g4[NRows] = {};
        for (int r = 0; r < NRows; ++r) {
            acc_f[r] = _mm256_setzero_ps();
            corr[r] = 0.0f;
        }
        const uint8_t* row_blocks[NRows];
        for (int r = 0; r < NRows; ++r) {
            row_blocks[r] = blocks + static_cast<size_t>(row0 + r) * row_stride;
        }
        const int prefetch_ahead = preq_prefetch_groups_ahead();
        if (use_g4) {
            for (int g = 0; g < g4_end; g += 4) {
                if (prefetch_ahead > 0) {
                    const int gp = g + prefetch_ahead;
                    if (gp < n_groups) {
                        preq_prefetch_groups(row_blocks, NRows, gp, block_size, x_q8, group_size);
                    }
                }
                preq_tile_accumulate_g4<NRows>(
                    acc_g4, corr, row_blocks, g, block_size,
                    x_q8, x_scales, x_sums_buf);
            }
        }
        for (int g = g4_end; g < n_groups; ++g) {
            if (prefetch_ahead > 0) {
                const int gp = g + prefetch_ahead;
                if (gp < n_groups) {
                    preq_prefetch_groups(row_blocks, NRows, gp, block_size, x_q8, group_size);
                }
            }
            const int8_t* x_group_q8 = x_q8 + g * group_size;
            const float x_scale = x_scales[g];
            const int32_t x_sum_g = x_sums_buf[g];
            for (int r = 0; r < NRows; ++r) {
                const uint8_t* block = row_blocks[r] + static_cast<size_t>(g) * block_size;
                preq_row_accumulate_one_group(
                    acc_f[r], corr[r], block, x_group_q8, x_scale, x_sum_g);
            }
        }
        for (int r = 0; r < NRows; ++r) {
            y[row0 + r] = acc_g4[r] + hsum256_ps(acc_f[r]) - corr[r];
        }
    }
    };

#ifdef _OPENMP
    if (asdsl_chunked::chunked_gemv_enabled()) {
        asdsl_chunked::parallel_row_chunks(n_chunks * NRows, NRows, [&](int rb, int re) {
            const int cb = rb / NRows;
            const int ce = (re + NRows - 1) / NRows;
            process_chunk(cb, std::min(ce, n_chunks));
        });
    } else {
        #pragma omp parallel for schedule(static)
        for (int chunk = 0; chunk < n_chunks; ++chunk) {
            process_chunk(chunk, chunk + 1);
        }
    }
#else
    process_chunk(0, n_chunks);
#endif

    for (int row = n_chunks * NRows; row < out_features; ++row) {
        __m256 acc_f = _mm256_setzero_ps();
        float acc_g4 = 0.0f;
        float corr = 0.0f;
        const uint8_t* row_blocks = blocks + static_cast<size_t>(row) * row_stride;
        const int prefetch_ahead = preq_prefetch_groups_ahead();
        const uint8_t* row_blocks_arr[8];
        row_blocks_arr[0] = row_blocks;
        if (use_g4) {
            for (int g = 0; g < g4_end; g += 4) {
                if (prefetch_ahead > 0) {
                    const int gp = g + prefetch_ahead;
                    if (gp < n_groups) {
                        preq_prefetch_groups(row_blocks_arr, 1, gp, block_size, x_q8, group_size);
                    }
                }
                preq_row_accumulate_g4(
                    acc_g4, corr, row_blocks, g, block_size, x_q8, x_scales, x_sums_buf);
            }
        }
        for (int g = g4_end; g < n_groups; ++g) {
            if (prefetch_ahead > 0) {
                const int gp = g + prefetch_ahead;
                if (gp < n_groups) {
                    preq_prefetch_groups(row_blocks_arr, 1, gp, block_size, x_q8, group_size);
                }
            }
            const uint8_t* block = row_blocks + static_cast<size_t>(g) * block_size;
            preq_row_accumulate_one_group(
                acc_f, corr, block, x_q8 + g * group_size, x_scales[g], x_sums_buf[g]);
        }
        y[row] = acc_g4 + hsum256_ps(acc_f) - corr;
    }
}

template <int NRows, bool ForceG4Fused = false>
static void gemv_q4_32_preq_nrow_avx2_impl(
    const uint8_t* blocks,
    const float* x_fp32,
    float* y,
    int out_features,
    int in_features,
    int group_size) {
    thread_local static int8_t tl_x_q8[20480];
    thread_local static float tl_x_scales[640];
    quantize_activation_avx2(x_fp32, tl_x_q8, tl_x_scales, in_features, group_size);
    gemv_q4_32_preq_nrow_from_q8_impl<NRows, ForceG4Fused>(
        blocks, tl_x_q8, tl_x_scales, y, out_features, in_features, group_size);
}

uint64_t preq_classic_accum_call_count() {
    return g_preq_classic_accum_calls.load(std::memory_order_relaxed);
}

uint64_t preq_xloaded_accum_call_count() {
    return g_preq_xloaded_accum_calls.load(std::memory_order_relaxed);
}

void preq_reset_accum_call_counts() {
    g_preq_classic_accum_calls.store(0, std::memory_order_relaxed);
    g_preq_xloaded_accum_calls.store(0, std::memory_order_relaxed);
}

} // namespace asdsl_preq

void quantize_activation_avx2(
    const float* x,
    int8_t* x_q8,
    float* x_scales,
    int in_features,
    int group_size) {
    const int n_groups = in_features / group_size;
    for (int g = 0; g < n_groups; g++) {
        const float* xg = x + g * group_size;
        int8_t* xq = x_q8 + g * group_size;

        __m256 max_abs = _mm256_setzero_ps();
        for (int j = 0; j < group_size; j += 8) {
            __m256 v = _mm256_loadu_ps(xg + j);
            __m256 a = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v);
            max_abs = _mm256_max_ps(max_abs, a);
        }
        __m128 lo = _mm256_castps256_ps128(max_abs);
        __m128 hi = _mm256_extractf128_ps(max_abs, 1);
        lo = _mm_max_ps(lo, hi);
        lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
        lo = _mm_max_ps(lo, _mm_movehdup_ps(lo));
        float amax = _mm_cvtss_f32(lo);

        if (amax < 1e-9f) {
            std::memset(xq, 0, static_cast<size_t>(group_size));
            x_scales[g] = 1.0f;
            continue;
        }

        float scale = amax / 127.0f;
        float inv_scale = 127.0f / amax;
        x_scales[g] = scale;

        __m256 vscale = _mm256_set1_ps(inv_scale);
        for (int j = 0; j < group_size; j += 8) {
            __m256 v = _mm256_mul_ps(_mm256_loadu_ps(xg + j), vscale);
            __m256i int32 = _mm256_cvtps_epi32(v);
            __m128i int16 = _mm_packs_epi32(
                _mm256_castsi256_si128(int32), _mm256_extracti128_si256(int32, 1));
            __m128i int8 = _mm_packs_epi16(int16, int16);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(xq + j), int8);
        }
    }
}

void gemv_q4_32_preq_avx2(
    const uint8_t* blocks,
    const int8_t* x_q8,
    const float* x_scales,
    float* y,
    int out_features,
    int in_features,
    int group_size) {
    if (!blocks || !x_q8 || !x_scales || !y) {
        throw std::invalid_argument("gemv_q4_32_preq_avx2: null input pointer");
    }
    if (out_features <= 0 || in_features <= 0 || group_size <= 0) {
        throw std::invalid_argument("gemv_q4_32_preq_avx2: invalid dimensions");
    }
    if ((in_features % group_size) != 0) {
        throw std::invalid_argument("gemv_q4_32_preq_avx2: in_features must be divisible by group_size");
    }
    if (group_size != 32) {
        throw std::invalid_argument("gemv_q4_32_preq_avx2: group_size must be 32");
    }
    const int n_groups = in_features / group_size;
    if (n_groups > 640) {
        throw std::invalid_argument("gemv_q4_32_preq_avx2: n_groups exceeds buffer");
    }
    asdsl_preq::gemv_q4_32_preq_nrow_from_q8_impl<4, false>(
        blocks, x_q8, x_scales, y, out_features, in_features, group_size);
}

void gemv_q4_32_preq_4row_avx2(
    const uint8_t* blocks,
    const float* x_fp32,
    float* y,
    int out_features,
    int in_features,
    int group_size) {
    if (!blocks || !x_fp32 || !y) {
        throw std::invalid_argument("gemv_q4_32_preq_4row_avx2: null input pointer");
    }
    if (group_size != 32) {
        throw std::invalid_argument("gemv_q4_32_preq_4row_avx2: group_size must be 32");
    }
    if (out_features <= 0 || in_features <= 0) {
        throw std::invalid_argument("gemv_q4_32_preq_4row_avx2: invalid dimensions");
    }
    const int n_groups = in_features / group_size;
    if (n_groups > 640) {
        throw std::invalid_argument("gemv_q4_32_preq_4row_avx2: n_groups exceeds buffer");
    }
    asdsl_preq::gemv_q4_32_preq_nrow_avx2_impl<4, false>(
        blocks, x_fp32, y, out_features, in_features, group_size);
}

void gemv_q4_32_preq_g4fused_4row_avx2(
    const uint8_t* blocks,
    const float* x_fp32,
    float* y,
    int out_features,
    int in_features,
    int group_size) {
    if (!blocks || !x_fp32 || !y) {
        throw std::invalid_argument("gemv_q4_32_preq_g4fused_4row_avx2: null input pointer");
    }
    if (group_size != 32) {
        throw std::invalid_argument("gemv_q4_32_preq_g4fused_4row_avx2: group_size must be 32");
    }
    if (out_features <= 0 || in_features <= 0) {
        throw std::invalid_argument("gemv_q4_32_preq_g4fused_4row_avx2: invalid dimensions");
    }
    const int n_groups = in_features / group_size;
    if (n_groups > 640 || (n_groups % 4) != 0) {
        throw std::invalid_argument(
            "gemv_q4_32_preq_g4fused_4row_avx2: n_groups must be <= 640 and divisible by 4");
    }
    asdsl_preq::gemv_q4_32_preq_nrow_avx2_impl<4, true>(
        blocks, x_fp32, y, out_features, in_features, group_size);
}

void gemv_q4_32_preq_8row_avx2(
    const uint8_t* blocks,
    const float* x_fp32,
    float* y,
    int out_features,
    int in_features,
    int group_size) {
    if (!blocks || !x_fp32 || !y) {
        throw std::invalid_argument("gemv_q4_32_preq_8row_avx2: null input pointer");
    }
    if (group_size != 32) {
        throw std::invalid_argument("gemv_q4_32_preq_8row_avx2: group_size must be 32");
    }
    if (out_features <= 0 || in_features <= 0) {
        throw std::invalid_argument("gemv_q4_32_preq_8row_avx2: invalid dimensions");
    }
    const int n_groups = in_features / group_size;
    if (n_groups > 640) {
        throw std::invalid_argument("gemv_q4_32_preq_8row_avx2: n_groups exceeds buffer");
    }
    asdsl_preq::gemv_q4_32_preq_nrow_avx2_impl<8, false>(
        blocks, x_fp32, y, out_features, in_features, group_size);
}

void gemv_q4_32_preq_fused_avx2(
    const uint8_t* blocks,
    const float* x_fp32,
    float* y,
    int out_features,
    int in_features,
    int group_size) {
    static int unroll = -1;
    if (unroll < 0) {
        const char* v = std::getenv("ASDSL_GEMV_UNROLL");
        unroll = (v && v[0] == '8') ? 8 : 4;
    }
    if (unroll == 8) {
        gemv_q4_32_preq_8row_avx2(blocks, x_fp32, y, out_features, in_features, group_size);
    } else {
        gemv_q4_32_preq_4row_avx2(blocks, x_fp32, y, out_features, in_features, group_size);
    }
}
