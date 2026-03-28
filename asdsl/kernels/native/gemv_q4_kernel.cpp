/**
 * AVX2+FMA 4-bit packed GEMV (group size 64).
 *
 * Unpacking: _mm256_and_si256 / _mm_and_si128 for low nibbles; widen with
 * unpack_epi8(zero), then _mm256_srli_epi16 / _mm_srli_epi16 + mask for highs;
 * interleave with unpacklo/hi_epi8. FMA accumulates dequantized dot products
 * (scale applied once per 64 columns).
 */

#include "gemv_q4_kernel.h"

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

namespace {

constexpr int kGroupSize = 64;

inline float hsum256_ps(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    lo = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, lo);
    lo = _mm_add_ss(lo, shuf);
    return _mm_cvtss_f32(lo);
}

inline __m256 cvt_u8x8_to_ps(__m128i bytes) {
    __m128i i32_lo = _mm_cvtepu8_epi32(bytes);
    __m128i i32_hi = _mm_cvtepu8_epi32(_mm_srli_si128(bytes, 4));
    return _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_cvtepi32_ps(i32_lo)),
        _mm_cvtepi32_ps(i32_hi),
        1);
}

inline float dot32_u8_f32(__m256i w_seq, const float* x) {
    __m128i w128 = _mm256_castsi256_si128(w_seq);
    __m256 f0 = cvt_u8x8_to_ps(w128);
    __m256 f1 = cvt_u8x8_to_ps(_mm_srli_si128(w128, 8));
    __m128i w128h = _mm256_extracti128_si256(w_seq, 1);
    __m256 f2 = cvt_u8x8_to_ps(w128h);
    __m256 f3 = cvt_u8x8_to_ps(_mm_srli_si128(w128h, 8));

    __m256 acc = _mm256_setzero_ps();
    acc = _mm256_fmadd_ps(f0, _mm256_loadu_ps(x), acc);
    acc = _mm256_fmadd_ps(f1, _mm256_loadu_ps(x + 8), acc);
    acc = _mm256_fmadd_ps(f2, _mm256_loadu_ps(x + 16), acc);
    acc = _mm256_fmadd_ps(f3, _mm256_loadu_ps(x + 24), acc);
    return hsum256_ps(acc);
}

/**
 * 16 packed bytes -> 32 linear uint8 weights (each 0..15) in __m256i.
 * Uses _mm256_and_si256 + _mm256_srli_epi16 on widened bytes (per requirement).
 */
inline __m256i unpack_16_packed_bytes(__m128i r) {
    const __m256i raw = _mm256_inserti128_si256(_mm256_setzero_si256(), r, 0);
    const __m256i z = _mm256_setzero_si256();
    const __m256i nibble_mask = _mm256_set1_epi8(0x0F);

    __m256i lo = _mm256_and_si256(raw, nibble_mask);

    __m256i wide_lo = _mm256_unpacklo_epi8(raw, z);
    __m256i wide_hi = _mm256_unpackhi_epi8(raw, z);
    __m256i hi_a = _mm256_and_si256(_mm256_srli_epi16(wide_lo, 4), nibble_mask);
    __m256i hi_b = _mm256_and_si256(_mm256_srli_epi16(wide_hi, 4), nibble_mask);

    __m128i hi0 = _mm256_castsi256_si128(hi_a);
    __m128i hi1 = _mm256_castsi256_si128(hi_b);
    __m128i hi_bytes = _mm_packus_epi16(hi0, hi1);

    __m128i lo128 = _mm256_castsi256_si128(lo);
    __m128i s0 = _mm_unpacklo_epi8(lo128, hi_bytes);
    __m128i s1 = _mm_unpackhi_epi8(lo128, hi_bytes);
    return _mm256_inserti128_si256(_mm256_castsi128_si256(s0), s1, 1);
}

inline float dot_group64(const uint8_t* packed, const float* x) {
    __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed));
    __m128i r0 = _mm256_castsi256_si128(raw);
    __m128i r1 = _mm256_extracti128_si256(raw, 1);

    __m256i w0 = unpack_16_packed_bytes(r0);
    __m256i w1 = unpack_16_packed_bytes(r1);

    return dot32_u8_f32(w0, x) + dot32_u8_f32(w1, x + 32);
}

}  // namespace

void gemv_q4_avx2(
    const uint8_t* w_packed,
    const float* x,
    const float* scales,
    float* out,
    int rows,
    int cols) {
    if (rows <= 0 || cols <= 0) {
        return;
    }
    if ((cols % kGroupSize) != 0 || (cols % 2) != 0) {
        return;
    }

    const int groups_per_row = cols / kGroupSize;
    const int packed_stride = cols / 2;

    // Parallelize output rows; per-r accumulators and pointers are loop-local.
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < rows; ++r) {
        const uint8_t* row = w_packed + static_cast<std::size_t>(r) * packed_stride;
        float sum = 0.0f;

        for (int g = 0; g < groups_per_row; ++g) {
            const uint8_t* pg = row + g * (kGroupSize / 2);
            const float* xg = x + g * kGroupSize;
            const int sidx = r * groups_per_row + g;
            sum += scales[sidx] * dot_group64(pg, xg);
        }

        out[r] = sum;
    }
}
