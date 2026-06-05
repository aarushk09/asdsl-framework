#include "unified_engine.h"
#include <immintrin.h>
#include <vector>
#include <cstring>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

void gemm_q4_q8_avx2(
    const uint8_t* weights_packed,
    const float*   scales,
    const float*   x,
    float*         y,
    int            out_features,
    int            in_features,
    int            group_size,
    int            batch_size
) {
    if (batch_size == 1) {
        gemv_q4_q8_avx2(weights_packed, scales, x, y, out_features, in_features, group_size);
        return;
    }

    const int n_groups = in_features / group_size;
    const int packed_stride = in_features / 2;

    std::vector<int8_t> x_q8_buf(static_cast<size_t>(in_features) * batch_size);
    std::vector<float>  x_scales_buf(static_cast<size_t>(n_groups) * batch_size);
    int8_t* x_q8    = x_q8_buf.data();
    float*  x_scaled = x_scales_buf.data();

    #pragma omp parallel for schedule(static)
    for (int b = 0; b < batch_size; ++b) {
        const float* xb = x + b * in_features;
        int8_t* xq_b = x_q8 + b * in_features;
        float* xs_b = x_scaled + b * n_groups;
        
        for (int g = 0; g < n_groups; g++) {
            const float* xg = xb + g * group_size;
            int8_t* xq = xq_b + g * group_size;
            
            __m256 max_abs = _mm256_setzero_ps();
            for (int j = 0; j < group_size; j += 8) {
                __m256 v = _mm256_loadu_ps(xg + j);
                __m256 m = _mm256_and_ps(v, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
                max_abs = _mm256_max_ps(max_abs, m);
            }
            __m128 lo = _mm256_castps256_ps128(max_abs);
            __m128 hi = _mm256_extractf128_ps(max_abs, 1);
            lo = _mm_max_ps(lo, hi);
            lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
            lo = _mm_max_ps(lo, _mm_movehdup_ps(lo));
            float amax = _mm_cvtss_f32(lo);
            
            if (amax < 1e-9f) {
                memset(xq, 0, group_size);
                xs_b[g] = 1.0f;
                continue;
            }
            
            float scale = amax / 127.0f;
            float inv_scale = 127.0f / amax;
            xs_b[g] = scale;
            
            __m256 vscale = _mm256_set1_ps(inv_scale);
            for (int j = 0; j < group_size; j += 8) {
                __m256 vf = _mm256_mul_ps(_mm256_loadu_ps(xg + j), vscale);
                __m256i vi = _mm256_cvtps_epi32(vf);
                __m128i lo4 = _mm_packs_epi32(_mm256_castsi256_si128(vi), _mm256_extracti128_si256(vi, 1));
                __m128i packed = _mm_packs_epi16(lo4, lo4);
                _mm_storel_epi64((__m128i*)(xq + j), packed);
            }
        }
    }

    #pragma omp parallel for schedule(static)
    for (int row = 0; row < out_features; ++row) {
        const uint8_t* w_row = weights_packed + (size_t)row * packed_stride;

        for (int b = 0; b < batch_size; ++b) {
            float acc = 0.0f;
            const int8_t* xq_b = x_q8 + b * in_features;
            const float* xs_b = x_scaled + b * n_groups;
            
            for (int g = 0; g < n_groups; g++) {
                const int8_t* x_group_q8 = xq_b + g * group_size;
                const uint8_t* w_group = w_row + g * (group_size / 2);
                float w_scale = scales[row * n_groups + g];
                float x_scale = xs_b[g];

                __m256i acc_int = _mm256_setzero_si256();
                const __m128i mask_nibble = _mm_set1_epi8(0x0F);
                const __m256i eight_256 = _mm256_set1_epi16(8);

                for (int i = 0; i < group_size; i += 32) {
                    __m128i packed = _mm_loadu_si128((const __m128i*)(w_group + i / 2));
                    
                    __m128i lo_nib = _mm_and_si128(packed, mask_nibble);
                    __m128i hi_nib = _mm_and_si128(_mm_srli_epi16(packed, 4), mask_nibble);
                    
                    __m128i w0_15_u8  = _mm_unpacklo_epi8(lo_nib, hi_nib);
                    __m128i w16_31_u8 = _mm_unpackhi_epi8(lo_nib, hi_nib);
                    
                    __m256i w_u8 = _mm256_set_m128i(w16_31_u8, w0_15_u8);
                    
                    __m256i w_s16_lo = _mm256_sub_epi16(_mm256_cvtepu8_epi16(_mm256_extracti128_si256(w_u8, 0)), eight_256);
                    __m256i w_s16_hi = _mm256_sub_epi16(_mm256_cvtepu8_epi16(_mm256_extracti128_si256(w_u8, 1)), eight_256);
                    
                    __m128i xb0 = _mm_loadu_si128((const __m128i*)(x_group_q8 + i));
                    __m128i xb1 = _mm_loadu_si128((const __m128i*)(x_group_q8 + i + 16));
                    
                    __m256i x_v_lo = _mm256_cvtepi8_epi16(xb0);
                    __m256i x_v_hi = _mm256_cvtepi8_epi16(xb1);
                    
                    __m256i t0 = _mm256_madd_epi16(w_s16_lo, x_v_lo);
                    __m256i t1 = _mm256_madd_epi16(w_s16_hi, x_v_hi);
                    
                    acc_int = _mm256_add_epi32(acc_int, t0);
                    acc_int = _mm256_add_epi32(acc_int, t1);
                }
                
                __m128i hi = _mm256_extracti128_si256(acc_int, 1);
                __m128i lo = _mm_add_epi32(_mm256_castsi256_si128(acc_int), hi);
                lo = _mm_hadd_epi32(lo, lo);
                lo = _mm_hadd_epi32(lo, lo);
                int dp = _mm_cvtsi128_si32(lo);
                
                acc += ((float)dp) * w_scale * x_scale;
            }
            y[b * out_features + row] = acc;
        }
    }
}


// Helper for FP16 -> FP32 conversion (reuse from gemv)
inline float _cvtsh_ss_batch(unsigned short x) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16(x)));
}

void gemm_q4_32_q8_avx2(
    const uint8_t* blocks,
    const float*   x,
    float*         y,
    int            out_features,
    int            in_features,
    int            group_size,
    int            batch_size
) {
    if (batch_size == 1) {
        gemv_q4_32_q8_avx2(blocks, x, y, out_features, in_features, group_size);
        return;
    }

    const int n_groups   = in_features / group_size;
    const int block_size = 18; // 2B fp16 scale + 16B packed nibbles

    std::vector<int8_t>  x_q8_buf(static_cast<size_t>(in_features) * batch_size);
    std::vector<float>   x_scales_buf(static_cast<size_t>(n_groups) * batch_size);
    // Per-(batch, group) int8 sum for unsigned-weight maddubs bias correction.
    // dot(w_uint8, x_int8) - 8*sum(x_int8) = dot(w_centered, x_int8).
    std::vector<int32_t> x_sums_buf(static_cast<size_t>(n_groups) * batch_size, 0);
    int8_t*  x_q8    = x_q8_buf.data();
    float*   x_scaled = x_scales_buf.data();
    int32_t* x_sums   = x_sums_buf.data();

    // Phase 1: quantize each batch row and compute per-group x_sums in one pass.
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < batch_size; ++b) {
        const float* xb  = x         + b * in_features;
        int8_t*      xq_b = x_q8    + b * in_features;
        float*       xs_b = x_scaled + b * n_groups;
        int32_t*     xsum_b = x_sums + b * n_groups;

        for (int g = 0; g < n_groups; g++) {
            const float* xg = xb + g * group_size;
            int8_t*      xq = xq_b + g * group_size;

            __m256 max_abs = _mm256_setzero_ps();
            for (int j = 0; j < group_size; j += 8) {
                __m256 v = _mm256_loadu_ps(xg + j);
                max_abs = _mm256_max_ps(max_abs,
                    _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v));
            }
            __m128 lo = _mm256_castps256_ps128(max_abs);
            __m128 hi = _mm256_extractf128_ps(max_abs, 1);
            lo = _mm_max_ps(lo, hi);
            lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
            lo = _mm_max_ps(lo, _mm_movehdup_ps(lo));
            float amax = _mm_cvtss_f32(lo);

            if (amax < 1e-9f) {
                memset(xq, 0, group_size);
                xs_b[g] = 1.0f;
                xsum_b[g] = 0;
                continue;
            }

            float inv_scale = 127.0f / amax;
            xs_b[g] = amax / 127.0f;

            __m256 vscale = _mm256_set1_ps(inv_scale);
            __m256i vsum  = _mm256_setzero_si256();
            for (int j = 0; j < group_size; j += 8) {
                __m256 vf = _mm256_mul_ps(_mm256_loadu_ps(xg + j), vscale);
                __m256i vi = _mm256_cvtps_epi32(vf);
                __m128i lo4 = _mm_packs_epi32(_mm256_castsi256_si128(vi),
                                               _mm256_extracti128_si256(vi, 1));
                __m128i i8  = _mm_packs_epi16(lo4, lo4);
                _mm_storel_epi64((__m128i*)(xq + j), i8);
                // Accumulate sum for bias correction using the clamped int8 values
                vsum = _mm256_add_epi32(vsum,
                    _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(xq + j))));
            }
            __m128i sl = _mm256_castsi256_si128(vsum);
            __m128i sh = _mm256_extracti128_si256(vsum, 1);
            sl = _mm_add_epi32(sl, sh);
            sl = _mm_hadd_epi32(sl, sl);
            sl = _mm_hadd_epi32(sl, sl);
            xsum_b[g] = _mm_cvtsi128_si32(sl);
        }
    }

    // Phase 2: GEMM using maddubs_epi16 + bias correction.
    // maddubs(w_uint8, x_int8) + madd(ones) in 2 instructions vs the old
    // cvtepu8_epi16 + sub_epi16(8) + cvtepi8_epi16 + madd_epi16 × 2 (5 instructions).
    const __m128i mask_nibble = _mm_set1_epi8(0x0F);
    const __m256i ones_16     = _mm256_set1_epi16(1);
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < out_features; ++row) {
        const uint8_t* row_blocks = blocks + static_cast<size_t>(row) * n_groups * block_size;

        for (int b = 0; b < batch_size; ++b) {
            float          acc    = 0.0f;
            const int8_t*  xq_b   = x_q8    + b * in_features;
            const float*   xs_b   = x_scaled + b * n_groups;
            const int32_t* xsum_b = x_sums   + b * n_groups;

            for (int g = 0; g < n_groups; ++g) {
                const uint8_t* block = row_blocks + g * block_size;

                uint16_t scale_fp16;
                memcpy(&scale_fp16, block, 2);
                float w_scale = _cvtsh_ss_batch(scale_fp16);

                const uint8_t* w_group    = block + 2;
                const int8_t*  x_group_q8 = xq_b + g * group_size;
                float          x_scale    = xs_b[g];

                __m256i acc_int = _mm256_setzero_si256();

                for (int i = 0; i < group_size; i += 32) {
                    __m128i packed    = _mm_loadu_si128((const __m128i*)(w_group + i / 2));
                    __m128i lo_nib    = _mm_and_si128(packed, mask_nibble);
                    __m128i hi_nib    = _mm_and_si128(_mm_srli_epi16(packed, 4), mask_nibble);
                    __m256i w_u8      = _mm256_set_m128i(_mm_unpackhi_epi8(lo_nib, hi_nib),
                                                         _mm_unpacklo_epi8(lo_nib, hi_nib));
                    __m256i x_all     = _mm256_set_m128i(
                        _mm_loadu_si128((const __m128i*)(x_group_q8 + i + 16)),
                        _mm_loadu_si128((const __m128i*)(x_group_q8 + i)));
                    __m256i prod16    = _mm256_maddubs_epi16(w_u8, x_all);
                    acc_int = _mm256_add_epi32(acc_int, _mm256_madd_epi16(prod16, ones_16));
                }

                __m128i hi_128 = _mm256_extracti128_si256(acc_int, 1);
                __m128i lo_128 = _mm_add_epi32(_mm256_castsi256_si128(acc_int), hi_128);
                lo_128 = _mm_hadd_epi32(lo_128, lo_128);
                lo_128 = _mm_hadd_epi32(lo_128, lo_128);
                int dp = _mm_cvtsi128_si32(lo_128) - 8 * xsum_b[g];

                acc += static_cast<float>(dp) * w_scale * x_scale;
            }
            y[static_cast<size_t>(b) * out_features + row] = acc;
        }
    }
}

