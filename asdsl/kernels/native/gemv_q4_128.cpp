/**
 * gemv_q4_128.cpp — Q4 GEMV optimized for group_size=128 (Q4_128 format)
 *
 * Kernel uses maddubs+madd_epi16 with optional AVX-VNNI dpbusd path.
 * AVX-VNNI: _mm256_dpbusd_avx_epi32 replaces 2-insn maddubs+madd with 1 insn.
 * Raptor Lake (i5-13500H) supports AVXVNNI via ISA extension since Alder Lake.
 */

#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#ifdef _MSC_VER
#  include <intrin.h>
#  define ASDSL_FORCEINLINE __forceinline
#  pragma warning(disable: 4100)
static inline float _cvtsh_ss_compat(unsigned short h) {
    __m128i v = _mm_cvtsi32_si128((int)h);
    return _mm_cvtss_f32(_mm_cvtph_ps(v));
}
#  define _cvtsh_ss(h) _cvtsh_ss_compat((unsigned short)(h))
#else
#  define ASDSL_FORCEINLINE __attribute__((always_inline)) inline
#endif

// ── AVX-VNNI detection (compile-time on MSVC 19.28+ with /arch:AVX2) ────────
// _mm256_dpbusd_avx_epi32(acc, a_uint8, b_int8) ≡ maddubs(a,b) + madd_epi16(·,1)
// but in a single uop. Saves ~4 instructions per inner 32-element chunk.
#if defined(_MSC_VER) && defined(__AVX2__)
#  if _MSC_VER >= 1928        // VS 2019 16.8+
#    define HAVE_AVXVNNI 1
#  endif
#elif defined(__AVXVNNI__)
#  define HAVE_AVXVNNI 1
#endif

static constexpr int Q4_128_BLOCK  = 66;   // 2B scale + 64B nibbles
static constexpr int Q4_128_GS     = 128;

static ASDSL_FORCEINLINE float hsum256_q4128(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    lo = _mm_add_ps(lo, hi);
    __m128 s = _mm_movehdup_ps(lo);
    lo = _mm_add_ps(lo, s);
    s  = _mm_movehl_ps(s, lo);
    lo = _mm_add_ss(lo, s);
    return _mm_cvtss_f32(lo);
}

// Fast int8 horizontal sum over 128 elements using _mm256_sad_epu8.
// We bias to uint8 (+128), sum, then subtract 128*128 to un-bias.
static ASDSL_FORCEINLINE int32_t x_group_sum128(const int8_t* xg) {
    const __m256i bias = _mm256_set1_epi8(static_cast<char>(0x80)); // +128
    __m256i acc = _mm256_setzero_si256();
    for (int j = 0; j < 128; j += 32) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(xg + j));
        // Bias: uint8 = int8 + 128 (XOR sign bit)
        __m256i ub = _mm256_xor_si256(v, bias);
        // sad_epu8 sums 8 uint8s per 64-bit lane → 8 uint16 results
        acc = _mm256_add_epi32(acc, _mm256_sad_epu8(ub, _mm256_setzero_si256()));
    }
    // Horizontal sum of 4 x 64-bit partial sums
    __m128i lo128 = _mm256_castsi256_si128(acc);
    __m128i hi128 = _mm256_extracti128_si256(acc, 1);
    lo128 = _mm_add_epi64(lo128, hi128);
    lo128 = _mm_add_epi64(lo128, _mm_srli_si128(lo128, 8));
    int32_t biased_sum = static_cast<int32_t>(_mm_cvtsi128_si64(lo128));
    return biased_sum - 128 * 128;  // un-bias
}

// Decode 16 packed nibbles (8 bytes = 16 nibbles = 16 uint4 weights) into
// a 256-bit uint8 vector (values in [0..15]).
static ASDSL_FORCEINLINE __m256i decode_nibbles_256(const uint8_t* packed,
                                                     const __m128i& mask_lo) {
    __m128i pk = _mm_loadu_si128((const __m128i*)packed);
    __m128i lo = _mm_and_si128(pk, mask_lo);
    __m128i hi = _mm_and_si128(_mm_srli_epi16(pk, 4), mask_lo);
    return _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi));
}

// ─────────────────────────────────────────────────────────────────────────────
// Main GEMV: Q4_128 × Q8 with 8-row unrolling + optional VNNI
// ─────────────────────────────────────────────────────────────────────────────
void gemv_q4_128_preq_avx2(
    const uint8_t* blocks,
    const int8_t*  x_q8,
    const float*   x_scales,
    float*         y,
    int out_features,
    int in_features,
    int group_size
) {
    if (group_size != 128)
        throw std::invalid_argument("gemv_q4_128_preq_avx2: group_size must be 128");
    if ((in_features % 128) != 0)
        throw std::invalid_argument("gemv_q4_128_preq_avx2: in_features must be divisible by 128");

    const int    n_groups    = in_features / 128;
    const size_t row_stride  = static_cast<size_t>(n_groups) * Q4_128_BLOCK;
    const __m128i mask_nibble = _mm_set1_epi8(0x0F);
#ifndef HAVE_AVXVNNI
    const __m256i ones_16    = _mm256_set1_epi16(1);
#endif

    // Precompute per-group x_sums for centering correction
    // (n_groups ≤ 17920/128 = 140, fits on stack)
    int32_t x_sums[140];
    for (int g = 0; g < n_groups; g++)
        x_sums[g] = x_group_sum128(x_q8 + g * 128);

    // ── Tail: 1-row fallback ─────────────────────────────────────────────────
    auto one_row = [&](int row) {
        const uint8_t* rb = blocks + static_cast<size_t>(row) * row_stride;
        __m256 acc_f = _mm256_setzero_ps();
        float corr = 0.0f;
        for (int g = 0; g < n_groups; g++) {
            const uint8_t* blk = rb + g * Q4_128_BLOCK;
            uint16_t sf16; memcpy(&sf16, blk, 2);
            float ws = _cvtsh_ss(sf16);
            float combined = ws * x_scales[g];
            const uint8_t* wp = blk + 2;
            const int8_t*  xp = x_q8 + g * 128;
            __m256i acc_i = _mm256_setzero_si256();
            for (int i = 0; i < 128; i += 32) {
                __m256i x32 = _mm256_loadu_si256((const __m256i*)(xp + i));
                __m256i w32 = decode_nibbles_256(wp + i/2, mask_nibble);
#ifdef HAVE_AVXVNNI
                acc_i = _mm256_dpbusd_avx_epi32(acc_i, w32, x32);
#else
                acc_i = _mm256_add_epi32(acc_i, _mm256_madd_epi16(_mm256_maddubs_epi16(w32, x32), ones_16));
#endif
            }
            acc_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc_i), _mm256_set1_ps(combined), acc_f);
            corr += 8.0f * static_cast<float>(x_sums[g]) * combined;
        }
        y[row] = hsum256_q4128(acc_f) - corr;
    };

    // ── Main 8-row parallel loop ─────────────────────────────────────────────
    const int n8 = out_features / 8;
    #pragma omp parallel for schedule(static)
    for (int r8 = 0; r8 < n8; r8++) {
        const int r0 = r8 * 8;
        const uint8_t* rb0 = blocks + static_cast<size_t>(r0    ) * row_stride;
        const uint8_t* rb1 = blocks + static_cast<size_t>(r0 + 1) * row_stride;
        const uint8_t* rb2 = blocks + static_cast<size_t>(r0 + 2) * row_stride;
        const uint8_t* rb3 = blocks + static_cast<size_t>(r0 + 3) * row_stride;
        const uint8_t* rb4 = blocks + static_cast<size_t>(r0 + 4) * row_stride;
        const uint8_t* rb5 = blocks + static_cast<size_t>(r0 + 5) * row_stride;
        const uint8_t* rb6 = blocks + static_cast<size_t>(r0 + 6) * row_stride;
        const uint8_t* rb7 = blocks + static_cast<size_t>(r0 + 7) * row_stride;

        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
        __m256 acc4 = _mm256_setzero_ps(), acc5 = _mm256_setzero_ps();
        __m256 acc6 = _mm256_setzero_ps(), acc7 = _mm256_setzero_ps();
        float  c0=0, c1=0, c2=0, c3=0, c4=0, c5=0, c6=0, c7=0;

        for (int g = 0; g < n_groups; g++) {
            const int8_t* xp = x_q8 + g * 128;
            const float   xs = x_scales[g];
            const float   c8 = 8.0f * static_cast<float>(x_sums[g]) * xs;
            const size_t  gb = static_cast<size_t>(g) * Q4_128_BLOCK;

            uint16_t sf0,sf1,sf2,sf3,sf4,sf5,sf6,sf7;
            memcpy(&sf0, rb0+gb, 2); memcpy(&sf1, rb1+gb, 2);
            memcpy(&sf2, rb2+gb, 2); memcpy(&sf3, rb3+gb, 2);
            memcpy(&sf4, rb4+gb, 2); memcpy(&sf5, rb5+gb, 2);
            memcpy(&sf6, rb6+gb, 2); memcpy(&sf7, rb7+gb, 2);
            const float ws0=_cvtsh_ss(sf0), ws1=_cvtsh_ss(sf1);
            const float ws2=_cvtsh_ss(sf2), ws3=_cvtsh_ss(sf3);
            const float ws4=_cvtsh_ss(sf4), ws5=_cvtsh_ss(sf5);
            const float ws6=_cvtsh_ss(sf6), ws7=_cvtsh_ss(sf7);

            c0+=c8*ws0; c1+=c8*ws1; c2+=c8*ws2; c3+=c8*ws3;
            c4+=c8*ws4; c5+=c8*ws5; c6+=c8*ws6; c7+=c8*ws7;

            const uint8_t* wp0=rb0+gb+2, *wp1=rb1+gb+2, *wp2=rb2+gb+2, *wp3=rb3+gb+2;
            const uint8_t* wp4=rb4+gb+2, *wp5=rb5+gb+2, *wp6=rb6+gb+2, *wp7=rb7+gb+2;

            __m256i ai0=_mm256_setzero_si256(), ai1=_mm256_setzero_si256();
            __m256i ai2=_mm256_setzero_si256(), ai3=_mm256_setzero_si256();
            __m256i ai4=_mm256_setzero_si256(), ai5=_mm256_setzero_si256();
            __m256i ai6=_mm256_setzero_si256(), ai7=_mm256_setzero_si256();

            for (int i = 0; i < 128; i += 32) {
                __m256i xv = _mm256_loadu_si256((const __m256i*)(xp + i));
#ifdef HAVE_AVXVNNI
#  define ACCUM(AI, WP) ai##AI = _mm256_dpbusd_avx_epi32(ai##AI, decode_nibbles_256((WP)+i/2, mask_nibble), xv)
#else
#  define ACCUM(AI, WP) do { \
    __m256i _w = decode_nibbles_256((WP)+i/2, mask_nibble); \
    ai##AI = _mm256_add_epi32(ai##AI, _mm256_madd_epi16(_mm256_maddubs_epi16(_w, xv), ones_16)); \
} while(0)
#endif
                ACCUM(0, wp0); ACCUM(1, wp1); ACCUM(2, wp2); ACCUM(3, wp3);
                ACCUM(4, wp4); ACCUM(5, wp5); ACCUM(6, wp6); ACCUM(7, wp7);
#undef ACCUM
            }

            __m256 vxs = _mm256_set1_ps(xs);
            acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ai0), _mm256_mul_ps(_mm256_set1_ps(ws0), vxs), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ai1), _mm256_mul_ps(_mm256_set1_ps(ws1), vxs), acc1);
            acc2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ai2), _mm256_mul_ps(_mm256_set1_ps(ws2), vxs), acc2);
            acc3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ai3), _mm256_mul_ps(_mm256_set1_ps(ws3), vxs), acc3);
            acc4 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ai4), _mm256_mul_ps(_mm256_set1_ps(ws4), vxs), acc4);
            acc5 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ai5), _mm256_mul_ps(_mm256_set1_ps(ws5), vxs), acc5);
            acc6 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ai6), _mm256_mul_ps(_mm256_set1_ps(ws6), vxs), acc6);
            acc7 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ai7), _mm256_mul_ps(_mm256_set1_ps(ws7), vxs), acc7);
        }
        y[r0  ] = hsum256_q4128(acc0) - c0;
        y[r0+1] = hsum256_q4128(acc1) - c1;
        y[r0+2] = hsum256_q4128(acc2) - c2;
        y[r0+3] = hsum256_q4128(acc3) - c3;
        y[r0+4] = hsum256_q4128(acc4) - c4;
        y[r0+5] = hsum256_q4128(acc5) - c5;
        y[r0+6] = hsum256_q4128(acc6) - c6;
        y[r0+7] = hsum256_q4128(acc7) - c7;
    }

    // Tail rows
    for (int row = n8 * 8; row < out_features; row++) one_row(row);
}

// ─────────────────────────────────────────────────────────────────────────────
// Batched GEMM for prompt processing (forward_batch)
// ─────────────────────────────────────────────────────────────────────────────
void gemm_q4_128_q8_avx2(
    const uint8_t* blocks,
    const float*   x_batch,
    float*         y_batch,
    int out_features, int in_features, int group_size, int batch_size
) {
    if (group_size != 128)
        throw std::invalid_argument("gemm_q4_128_q8_avx2: group_size must be 128");

    const int    n_groups   = in_features / 128;
    const size_t row_stride = static_cast<size_t>(n_groups) * Q4_128_BLOCK;
    const __m128i mask_nibble = _mm_set1_epi8(0x0F);

    #pragma omp parallel for schedule(static)
    for (int row = 0; row < out_features; row++) {
        const uint8_t* rb = blocks + static_cast<size_t>(row) * row_stride;
        float acc[64] = {};

        for (int g = 0; g < n_groups; g++) {
            const uint8_t* blk = rb + g * Q4_128_BLOCK;
            uint16_t sf16; memcpy(&sf16, blk, 2);
            float ws = _cvtsh_ss(sf16);
            const uint8_t* wp = blk + 2;

            // Dequantize 128 weights to float32 (scalar loop - only called for prompt, not decode)
            float wf[128];
            for (int i = 0; i < 64; i++) {
                uint8_t b = wp[i];
                wf[i*2  ] = ((b & 0x0F) - 8) * ws;
                wf[i*2+1] = ((b >>    4) - 8) * ws;
            }

            const int k0 = g * 128;
            for (int b = 0; b < batch_size && b < 64; b++) {
                const float* xb = x_batch + static_cast<size_t>(b) * in_features + k0;
                __m256 vdot = _mm256_setzero_ps();
                for (int i = 0; i < 128; i += 8)
                    vdot = _mm256_fmadd_ps(_mm256_loadu_ps(wf+i), _mm256_loadu_ps(xb+i), vdot);
                acc[b] += hsum256_q4128(vdot);
            }
        }
        for (int b = 0; b < batch_size && b < 64; b++)
            y_batch[static_cast<size_t>(b)*out_features+row] = acc[b];
    }
}
