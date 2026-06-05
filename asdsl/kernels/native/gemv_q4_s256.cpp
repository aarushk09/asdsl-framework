/**
 * gemv_q4_s256.cpp — Q4 GEMV using 256-weight superblocks with 32-weight sub-scales.
 *
 * Format Q4_S256 (144 bytes per 256 weights = 4.5 bpw — identical to Q4_K_M's bandwidth):
 *   [uint16 sub_scales[8]  → 16B]   8 × FP16 scale, one per 32-weight sub-block
 *   [uint8  nibbles[128]   → 128B]  256 weights packed as nibbles (lo=even, hi=odd)
 *   Total: 144 bytes per superblock
 *
 * WHY this beats our Q4_32 (also 4.5 bpw, same number of sub-scales):
 *   Q4_32: 18-byte blocks; scale + nibbles interleaved every 18B → cache-line splits
 *   Q4_S256: 144-byte superblocks; all 8 scales grouped in first 16B (fits one cache line
 *            quarter), then 128B of nibbles — one contiguous fetch per 256 weights.
 *   Additionally, VNNI (dpbusd) replaces maddubs+madd_epi16 (2 instructions → 1).
 *
 * Activation side: engine keeps group_size=32 (same as Q4_32) for accuracy parity.
 * x_scales has in_features/32 entries; we use 8 consecutive entries per superblock.
 */

#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#ifdef _MSC_VER
#  include <intrin.h>
#  pragma warning(disable: 4100)
static inline float _cvtsh_ss_s256(unsigned short h) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)h)));
}
#  define CVT_FP16(h) _cvtsh_ss_s256((unsigned short)(h))
#else
#  define CVT_FP16(h) _cvtsh_ss(h)
#endif

// ─── AVX-VNNI ────────────────────────────────────────────────────────────────
#if defined(_MSC_VER) && defined(__AVX2__) && (_MSC_VER >= 1928)
#  define HAVE_AVXVNNI_S256 1
#elif defined(__AVXVNNI__)
#  define HAVE_AVXVNNI_S256 1
#endif

static constexpr int S256_BLOCK   = 144;   // bytes per superblock
static constexpr int S256_WEIGHTS = 256;   // weights per superblock
static constexpr int S256_SUBS    = 8;     // sub-blocks per superblock
static constexpr int S256_SUBW    = 32;    // weights per sub-block
static constexpr int S256_SCALES  = 16;    // bytes for 8 FP16 scales
static constexpr int S256_NIBS    = 128;   // bytes of nibbles

static inline float hsum256_s256(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehdup_ps(lo));
    return _mm_cvtss_f32(_mm_add_ss(lo, _mm_movehl_ps(lo, lo)));
}

// Decode 16 packed nibble bytes into 32 uint8 weights in [0..15].
static inline __m256i decode32_nibbles(const uint8_t* p, __m128i mask4) {
    __m128i pk = _mm_loadu_si128((const __m128i*)p);
    __m128i lo = _mm_and_si128(pk, mask4);
    __m128i hi = _mm_and_si128(_mm_srli_epi16(pk, 4), mask4);
    return _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi));
}

// Fast int8 horizontal sum over 32 elements.
static inline int32_t hsum_q8_32(const int8_t* xp) {
    // bias to uint8 (+128), sum with SAD, unbias
    const __m256i bias = _mm256_set1_epi8(static_cast<char>(0x80));
    __m256i v  = _mm256_loadu_si256((const __m256i*)xp);
    __m256i ub = _mm256_xor_si256(v, bias);
    __m256i s  = _mm256_sad_epu8(ub, _mm256_setzero_si256());
    __m128i lo = _mm256_castsi256_si128(s);
    __m128i hi = _mm256_extracti128_si256(s, 1);
    lo = _mm_add_epi64(lo, hi);
    lo = _mm_add_epi64(lo, _mm_srli_si128(lo, 8));
    return static_cast<int32_t>(_mm_cvtsi128_si64(lo)) - 128 * 32;
}

/**
 * gemv_q4_s256_preq_avx2 — main decode GEMV.
 * group_size param is the ACTIVATION group size (must be 32).
 * Weight superblock is always 256 weights with 8×FP16 sub-scales.
 */
void gemv_q4_s256_preq_avx2(
    const uint8_t* blocks,    // Q4_S256 blocks [out_features × n_superblocks × 144B]
    const int8_t*  x_q8,     // pre-quantized activations [in_features], group=32
    const float*   x_scales, // per-32-elem activation scales [in_features/32]
    float*         y,
    int out_features,
    int in_features,
    int /*group_size*/        // must be 32, enforced by caller
) {
    if ((in_features % 256) != 0)
        throw std::invalid_argument("gemv_q4_s256: in_features must be divisible by 256");

    const int    n_super    = in_features / 256;           // superblocks per row
    const int    n_sub_tot  = in_features / 32;            // total 32-elem sub-groups
    const size_t row_stride = static_cast<size_t>(n_super) * S256_BLOCK;
    const __m128i mask4     = _mm_set1_epi8(0x0F);
#ifndef HAVE_AVXVNNI_S256
    const __m256i ones16    = _mm256_set1_epi16(1);
#endif

    // Precompute x_sums per 32-elem group (for centering correction)
    // Max: 17920/32 = 560 entries — heap allocate to avoid stack overflow
    int32_t x_sums_buf[560];
    for (int g = 0; g < n_sub_tot; g++)
        x_sums_buf[g] = hsum_q8_32(x_q8 + g * 32);

    // ── One-row tail helper ──────────────────────────────────────────────────
    auto one_row = [&](int row) {
        const uint8_t* rb = blocks + static_cast<size_t>(row) * row_stride;
        __m256 acc_f = _mm256_setzero_ps();
        float  corr  = 0.0f;
        for (int sb = 0; sb < n_super; sb++) {
            const uint8_t* blk      = rb + sb * S256_BLOCK;
            const uint16_t* scales  = (const uint16_t*)blk;       // 8 FP16 sub-scales
            const uint8_t*  nibbles = blk + S256_SCALES;           // 128B nibbles
            const int sub0 = sb * 8;
            for (int k = 0; k < 8; k++) {
                float ws = CVT_FP16(scales[k]);
                float xs = x_scales[sub0 + k];
                float sc = ws * xs;
                const int8_t*  xp = x_q8 + (sub0 + k) * 32;
                const uint8_t* wp = nibbles + k * 16;
                __m256i x32 = _mm256_loadu_si256((const __m256i*)xp);
                __m256i w32 = decode32_nibbles(wp, mask4);
                __m256i ai  = _mm256_setzero_si256();
#ifdef HAVE_AVXVNNI_S256
                ai = _mm256_dpbusd_avx_epi32(ai, w32, x32);
#else
                ai = _mm256_add_epi32(ai, _mm256_madd_epi16(_mm256_maddubs_epi16(w32, x32), ones16));
#endif
                acc_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ai), _mm256_set1_ps(sc), acc_f);
                corr += 8.0f * x_sums_buf[sub0 + k] * sc;
            }
        }
        y[row] = hsum256_s256(acc_f) - corr;
    };

    // ── 4-row parallel main loop ─────────────────────────────────────────────
    const int n4 = out_features / 4;
    #pragma omp parallel for schedule(static)
    for (int r4 = 0; r4 < n4; r4++) {
        const int r0 = r4 * 4;
        const size_t rs = row_stride;
        const uint8_t* rb0 = blocks + static_cast<size_t>(r0    ) * rs;
        const uint8_t* rb1 = blocks + static_cast<size_t>(r0 + 1) * rs;
        const uint8_t* rb2 = blocks + static_cast<size_t>(r0 + 2) * rs;
        const uint8_t* rb3 = blocks + static_cast<size_t>(r0 + 3) * rs;

        __m256 a0=_mm256_setzero_ps(), a1=_mm256_setzero_ps();
        __m256 a2=_mm256_setzero_ps(), a3=_mm256_setzero_ps();
        float  c0=0, c1=0, c2=0, c3=0;

        for (int sb = 0; sb < n_super; sb++) {
            const size_t boff = static_cast<size_t>(sb) * S256_BLOCK;
            // Load all 8 sub-scales for each of the 4 rows (16B each = one cache line quarter)
            const uint16_t* sc0 = (const uint16_t*)(rb0 + boff);
            const uint16_t* sc1 = (const uint16_t*)(rb1 + boff);
            const uint16_t* sc2 = (const uint16_t*)(rb2 + boff);
            const uint16_t* sc3 = (const uint16_t*)(rb3 + boff);
            const uint8_t*  nib0 = rb0 + boff + S256_SCALES;
            const uint8_t*  nib1 = rb1 + boff + S256_SCALES;
            const uint8_t*  nib2 = rb2 + boff + S256_SCALES;
            const uint8_t*  nib3 = rb3 + boff + S256_SCALES;
            const int sub0 = sb * 8;

            for (int k = 0; k < 8; k++) {
                const int8_t*  xp  = x_q8     + (sub0 + k) * 32;
                const float    xs  = x_scales  [sub0 + k];
                const float    c8x = 8.0f * x_sums_buf[sub0 + k] * xs;
                const uint8_t* wp0 = nib0 + k * 16;
                const uint8_t* wp1 = nib1 + k * 16;
                const uint8_t* wp2 = nib2 + k * 16;
                const uint8_t* wp3 = nib3 + k * 16;

                float ws0=CVT_FP16(sc0[k]), ws1=CVT_FP16(sc1[k]);
                float ws2=CVT_FP16(sc2[k]), ws3=CVT_FP16(sc3[k]);
                c0 += c8x*ws0; c1 += c8x*ws1; c2 += c8x*ws2; c3 += c8x*ws3;

                __m256i xv = _mm256_loadu_si256((const __m256i*)xp);
                __m256i ai0=_mm256_setzero_si256(), ai1=_mm256_setzero_si256();
                __m256i ai2=_mm256_setzero_si256(), ai3=_mm256_setzero_si256();

#ifdef HAVE_AVXVNNI_S256
#  define DOT4(AI,WP) AI = _mm256_dpbusd_avx_epi32(AI, decode32_nibbles(WP, mask4), xv)
#else
#  define DOT4(AI,WP) do { \
    __m256i _w = decode32_nibbles(WP, mask4); \
    AI = _mm256_add_epi32(AI, _mm256_madd_epi16(_mm256_maddubs_epi16(_w, xv), ones16)); \
} while(0)
#endif
                DOT4(ai0, wp0); DOT4(ai1, wp1); DOT4(ai2, wp2); DOT4(ai3, wp3);
#undef DOT4

                __m256 vxs = _mm256_set1_ps(xs);
                a0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ai0), _mm256_mul_ps(_mm256_set1_ps(ws0),vxs), a0);
                a1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ai1), _mm256_mul_ps(_mm256_set1_ps(ws1),vxs), a1);
                a2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ai2), _mm256_mul_ps(_mm256_set1_ps(ws2),vxs), a2);
                a3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ai3), _mm256_mul_ps(_mm256_set1_ps(ws3),vxs), a3);
            }
        }
        y[r0  ] = hsum256_s256(a0) - c0;
        y[r0+1] = hsum256_s256(a1) - c1;
        y[r0+2] = hsum256_s256(a2) - c2;
        y[r0+3] = hsum256_s256(a3) - c3;
    }
    for (int row = n4*4; row < out_features; row++) one_row(row);
}
