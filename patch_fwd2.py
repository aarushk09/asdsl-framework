import re
with open('asdsl/kernels/forward_loop.cpp', 'r', encoding='utf-16le') as f:
    src = f.read()

new_gemv = """
inline void gemv_q4_avx2_row(const BlockQ4_32* weights, const float* activations, float* output, int out_dim, int in_dim) {
    int blocks_per_row = in_dim / 32;

    #pragma omp parallel for
    for (int i = 0; i < out_dim; ++i) {
        const BlockQ4_32* row_w = weights + i * blocks_per_row;
        const float* row_a = activations;
        
        __m256 sum_vec = _mm256_setzero_ps();
        
        for (int b = 0; b < blocks_per_row; ++b) {
            const BlockQ4_32& block = row_w[b];
            float scale = f16_to_f32(block.scale);
            __m256 v_scale = _mm256_set1_ps(scale);
            
            float w_f[32];
            for(int k=0; k<16; ++k) {
                uint8_t val = block.weights[k];
                w_f[k*2] = (float)(((int8_t)(val & 0x0F)) - 8);
                w_f[k*2+1] = (float)(((int8_t)(val >> 4)) - 8);
            }
            
            for(int s=0; s<32; s+=8) {
                __m256 w0 = _mm256_loadu_ps(&w_f[s]);
                w0 = _mm256_mul_ps(w0, v_scale);
                __m256 a0 = _mm256_loadu_ps(&row_a[b * 32 + s]);
                sum_vec = _mm256_fmadd_ps(w0, a0, sum_vec);
            }
        }
        
        float temp[8];
        _mm256_storeu_ps(temp, sum_vec);
        float sum = 0.0f;
        for(int j=0; j<8; ++j) sum += temp[j];
        
        output[i] += sum;
    }
}

inline void fused_rmsnorm_gemv_q4(const float* x, const float* rms_w, const BlockQ4_32* weights, float* output, int out_dim, int in_dim, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < in_dim; ++i) sum_sq += x[i] * x[i];
    float inv_rms = 1.0f / std::sqrt(sum_sq / in_dim + eps);

    int blocks_per_row = in_dim / 32;
    const int ROW_TILE = 32;

    #pragma omp parallel for
    for (int i_start = 0; i_start < out_dim; i_start += ROW_TILE) {
        int i_end = (i_start + ROW_TILE > out_dim) ? out_dim : i_start + ROW_TILE;

        // FUSION: Normalize X directly into the fast L1 cache buffer. 
        // We only do this locally for the tile! This saves massive memory bandwidth.
        std::vector<float> local_a(in_dim);
        for(int k=0; k<in_dim; ++k) {
            local_a[k] = x[k] * inv_rms * rms_w[k];
        }

        for (int i = i_start; i < i_end; ++i) {
            const BlockQ4_32* row_w = weights + i * blocks_per_row;
            __m256 sum256 = _mm256_setzero_ps();
            
            for (int b = 0; b < blocks_per_row; ++b) {
                const BlockQ4_32& block = row_w[b];
                float scale = f16_to_f32(block.scale);
                __m256 v_scale = _mm256_set1_ps(scale);

                // SIMD unpack and FMA
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
}
"""

match = re.search(r"inline void gemv_q4_avx2_row.*?\}[\r\n\s]*inline void fused_rmsnorm_gemv_q4.*?\}[\r\n\s]*\}[\r\n\s]*\}", src, re.DOTALL)
if match:
    src = src[:match.start()] + new_gemv + src[match.end():]
else:
    print("Match failed")

with open('asdsl/kernels/forward_loop.cpp', 'w', encoding='utf-16le') as f:
    f.write(src)
