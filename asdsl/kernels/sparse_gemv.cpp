#include <immintrin.h>
#include <cstdint>

// Generates a packed bitmask indicating which activations exceed threshold.
// src: pointer to float activations, n: count (must be multiple of 8)
// out_mask: pointer to uint8_t bitmask output (n/8 bytes)
extern "C" void generate_sparse_mask_avx2(const float* src, int n, uint8_t* out_mask, float threshold) {
    __m256 thresh_vec = _mm256_set1_ps(threshold);
    for (int i = 0; i < n; i += 8) {
        __m256 vals = _mm256_loadu_ps(src + i);
        __m256 abs_vals = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), vals); // abs via sign bit mask
        __m256 cmp = _mm256_cmp_ps(abs_vals, thresh_vec, _CMP_GT_OS);
        int mask = _mm256_movemask_ps(cmp);
        out_mask[i / 8] = (uint8_t)mask;
    }
}
