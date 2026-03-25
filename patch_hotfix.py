import re
import codecs

with codecs.open('asdsl/kernels/forward_loop.cpp', 'r', 'utf-16le') as f:
    code = f.read()

# Restore the fast loop but remove heap allocation entirely!
gemv_search = r"""void gemv_q4_avx2_row\(const BlockQ4_32\* weights, const float\* activations, float\* output, int out_dim, int in_dim\) \{.*?output\[i_start \+ r\] \+= row_sums\[r\];\s*\n\s*\}\s*\n\s*\}\s*\n\s*\}"""

gemv_replace = """void gemv_q4_avx2_row(const BlockQ4_32* weights, const float* activations, float* output, int out_dim, int in_dim) {
    int blocks_per_row = in_dim / 32;

    const int ROW_TILE = 32;
    const int MAX_DIM = 24576; // Safe static buffer to cover max in_dim (17920)

    #pragma omp parallel for
    for (int i_start = 0; i_start < out_dim; i_start += ROW_TILE) {
        int i_end = (i_start + ROW_TILE > out_dim) ? out_dim : i_start + ROW_TILE;

        // Zero-Allocation Memory Tiling! Replace std::vector with aligned C-array.
        alignas(32) float local_a[MAX_DIM];
        for(int k=0; k<in_dim; ++k) {
            local_a[k] = activations[k];
        }

        for (int i = i_start; i < i_end; ++i) {
            const BlockQ4_32* row_w = weights + i * blocks_per_row;
            __m256 sum256 = _mm256_setzero_ps();
            
            for (int b = 0; b < blocks_per_row; ++b) {
                const BlockQ4_32& block = row_w[b];
                float scale = f16_to_f32(block.scale);
                __m256 v_scale = _mm256_set1_ps(scale);

                float w_f[32];
                for (int k = 0; k < 16; ++k) {
                    uint8_t val = block.weights[k];
                    w_f[k*2] = (float)(((int8_t)(val & 0x0F)) - 8);
                    w_f[k*2+1] = (float)(((int8_t)(val >> 4)) - 8);
                }

                for(int s=0; s<32; s+=8) {
                    __m256 w0 = _mm256_loadu_ps(&w_f[s]);
                    w0 = _mm256_mul_ps(w0, v_scale);
                    __m256 a0 = _mm256_loadu_ps(&local_a[b * 32 + s]);
                    sum256 = _mm256_fmadd_ps(w0, a0, sum256);
                }
            }

            float temp[8];
            _mm256_storeu_ps(temp, sum256);
            float sum = 0.0f;
            for(int j=0; j<8; ++j) sum += temp[j];
            
            output[i] += sum;
        }
    }
}"""

code = re.sub(gemv_search, gemv_replace, code, flags=re.DOTALL)

with codecs.open('asdsl/kernels/forward_loop.cpp', 'w', 'utf-16le') as f:
    f.write(code)

print("Restored original loop with MAX_DIM stack arrays.")
