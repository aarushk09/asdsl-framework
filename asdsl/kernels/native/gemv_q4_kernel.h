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
