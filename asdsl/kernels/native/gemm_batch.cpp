#include "unified_engine.h"
#include <immintrin.h>
#include <vector>
#include <cstring>
#include <cmath>

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

    auto& pool = asdsl::get_global_thread_pool();
    const int n_groups = in_features / group_size;
    const int packed_stride = in_features / 2;

    // Thread-local scratch: allocated once per thread, reused each call.
    // Max in_features = 17920 (Phi-4 gate_up), max n_groups = 17920/32 = 560.
    constexpr int TL_MAX_IN = 20480;
    constexpr int TL_MAX_GROUPS = 640;
    thread_local static int8_t tl_x_q8[TL_MAX_IN];
    thread_local static float  tl_x_scales[TL_MAX_GROUPS];

    // For batch, we need B rows of scratch — heap-allocate only for batch > 1
    // (batch_size==1 is already handled by the early-return above).
    std::vector<int8_t> x_q8_buf;
    std::vector<float>  x_scales_buf;
    int8_t* x_q8;
    float*  x_scaled;
    if (batch_size == 1 || (in_features <= TL_MAX_IN && n_groups <= TL_MAX_GROUPS)) {
        // Use thread-local for the single-token path (batch==1 exits above,
        // so this covers the rare single-row case if it ever reaches here)
        x_q8_buf.resize(in_features * batch_size);
        x_scales_buf.resize(n_groups * batch_size);
        x_q8    = x_q8_buf.data();
        x_scaled = x_scales_buf.data();
    } else {
        x_q8_buf.resize(in_features * batch_size);
        x_scales_buf.resize(n_groups * batch_size);
        x_q8    = x_q8_buf.data();
        x_scaled = x_scales_buf.data();
    }
    (void)tl_x_q8; (void)tl_x_scales; // suppress unused warning

    pool.parallel_for(0, batch_size, 1, [&](int b) {
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
    });

    const int grain_q = std::max(1, out_features / pool.thread_count());
    pool.parallel_for(0, out_features, grain_q, [&](int row) {
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
    });
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

    auto& pool = asdsl::get_global_thread_pool();
    const int n_groups = in_features / group_size;
    const int block_size = 18; // 2 byte fp16 scale + 16 byte data

    // Thread-local scratch: zero allocation per call after the first.
    // Max in_features = 17920 (Phi-4 gate_up), max n_groups = 17920/32 = 560.
    constexpr int TL32_MAX_IN = 20480;
    constexpr int TL32_MAX_GROUPS = 640;
    // For batch GEMM (batch_size > 1) we need B * in_features bytes.
    // Use heap only for the batch dimension; within each thread's slice,
    // scratch is per-batch-row (no thread-local needed for batch path).
    std::vector<int8_t> x_q8_buf(in_features * batch_size);
    std::vector<float>  x_scales_buf(n_groups * batch_size);
    int8_t* x_q8    = x_q8_buf.data();
    float*  x_scaled = x_scales_buf.data();
    (void)TL32_MAX_IN; (void)TL32_MAX_GROUPS;

    pool.parallel_for(0, batch_size, 1, [&](int b) {
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
    });

    const int grain_r = std::max(1, out_features / pool.thread_count());
    pool.parallel_for(0, out_features, grain_r, [&](int row) {
        const uint8_t* row_blocks = blocks + row * n_groups * block_size;    

        for (int b = 0; b < batch_size; ++b) {
            float acc = 0.0f;
            const int8_t* xq_b = x_q8 + b * in_features;
            const float* xs_b = x_scaled + b * n_groups;

            for (int g = 0; g < n_groups; ++g) {
                const uint8_t* block = row_blocks + g * block_size;
                
                uint16_t scale_fp16;
                memcpy(&scale_fp16, block, 2);
                float w_scale = _cvtsh_ss_batch(scale_fp16);

                const uint8_t* w_group = block + 2;
                const int8_t* x_group_q8 = xq_b + g * group_size;
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
    });
}

