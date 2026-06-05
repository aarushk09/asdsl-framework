/**
 * Dense 4-bit packed GEMV: y[r] = sum_g scale[r,g] * dot(unpack(W[r,g,:]), x[g*64:(g+1)*64])
 *
 * Weight layout: 2 nibbles per byte, low 4 bits = even column, high 4 bits = odd column.
 */

#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * GEMV for 4-bit packed weights with group size 64 (one scale per 64 columns).
 *
 * @param w_packed Packed weights, row-major: each row uses (cols/2) bytes.
 * @param x        Input vector, length cols.
 * @param scales   Per (row, group) scales, length rows * (cols/64), row-major.
 * @param out      Output, length rows.
 * @param rows     Number of matrix rows.
 * @param cols     Number of columns (must be divisible by 64 and even).
 */
void gemv_q4_avx2(
    const uint8_t* w_packed,
    const float* x,
    const float* scales,
    float* out,
    int rows,
    int cols);

#ifdef __cplusplus
}
#endif

void gemv_q4_32_preq_avx2(
    const uint8_t* blocks,
    const int8_t* x_q8,
    const float* x_scales,
    float* y,
    int out_features,
    int in_features,
    int group_size);
/** FP32 activations: per-group Q8 quant inside GEMV (no separate quantize pass). */
void gemv_q4_32_preq_fused_avx2(
    const uint8_t* blocks,
    const float* x_fp32,
    float* y,
    int out_features,
    int in_features,
    int group_size);
/** Phase 23: 4-row + 4-group fused inner loop (128 weights / group iteration). */
void gemv_q4_32_preq_g4fused_4row_avx2(
    const uint8_t* blocks,
    const float* x_fp32,
    float* y,
    int out_features,
    int in_features,
    int group_size);

void gemv_q4_32_preq_4row_avx2(
    const uint8_t* blocks,
    const float* x_fp32,
    float* y,
    int out_features,
    int in_features,
    int group_size);
/** Optional 8-row unroll (benchmark via ASDSL_GEMV_UNROLL=8). */
void gemv_q4_32_preq_8row_avx2(
    const uint8_t* blocks,
    const float* x_fp32,
    float* y,
    int out_features,
    int in_features,
    int group_size);
void quantize_activation_avx2(
    const float* x,
    int8_t* x_q8,
    float* x_scales,
    int in_features,
    int group_size);

/** Q4_128 symmetric preq GEMV (66-byte blocks, group_size=128). */
void gemv_q4_128_preq_avx2(
    const uint8_t* blocks,
    const int8_t* x_q8,
    const float* x_scales,
    float* y,
    int out_features,
    int in_features,
    int group_size);
void gemv_q4_32_q8_avx2(const uint8_t* blocks, const float* x, float* y, int out_features, int in_features, int group_size);
void gemm_q4_32_q8_avx2(const uint8_t* blocks, const float* x, float* y, int out_features, int in_features, int group_size, int batch_size);
void gemv_asb_avx2(const uint8_t* asb_blocks, const float* x, float* y, int out_features, int in_features, int group_size);
void gemv_q4km_q8_avx2(const uint8_t* weights_q4km, const float* x, float* y, int out_features, int in_features);
