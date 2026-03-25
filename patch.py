import re
with open('asdsl/kernels/forward_loop.cpp', 'r', encoding='utf-16le') as f:
    src = f.read()

new_gemv = """
inline void gemv_q4_avx2_row(const BlockQ4_32* weights, const float* activations, float* output, int out_dim, int in_dim) {
    int blocks_per_row = in_dim / 32;
    const int ROW_TILE = 64; 
    const int COL_TILE = 8; 

    #pragma omp parallel for
    for (int i_start = 0; i_start < out_dim; i_start += ROW_TILE) {
        int i_end = (i_start + ROW_TILE > out_dim) ? out_dim : i_start + ROW_TILE;

        for (int b_start = 0; b_start < blocks_per_row; b_start += COL_TILE) {
            int b_end = (b_start + COL_TILE > blocks_per_row) ? blocks_per_row : b_start + COL_TILE;

            // Load activations for this tile into L1 cache footprint
            float local_a[256];
            for (int b = b_start; b < b_end; ++b) {
                for(int k=0; k<32; ++k) {
                    local_a[(b - b_start)*32 + k] = activations[b*32 + k];
                }
            }

            for (int i = i_start; i < i_end; ++i) {
                const BlockQ4_32* row_w = weights + i * blocks_per_row;
                __m256 sum256 = _mm256_setzero_ps();
                
                for (int b = b_start; b < b_end; ++b) {
                    const BlockQ4_32& block = row_w[b];
                    float scale = f16_to_f32(block.scale);
                    __m256 v_scale = _mm256_set1_ps(scale);

                    for (int k = 0; k < 16; k+=8) {
                        float w_f[16];
                        for(int j=0; j<8; ++j) {
                            uint8_t val = block.weights[k+j];
                            w_f[j*2] = (float)(((int8_t)(val & 0x0F)) - 8);
                            w_f[j*2+1] = (float)(((int8_t)(val >> 4)) - 8);
                        }
                        __m256 w0 = _mm256_loadu_ps(&w_f[0]);
                        __m256 w1 = _mm256_loadu_ps(&w_f[8]);
                        
                        w0 = _mm256_mul_ps(w0, v_scale);
                        w1 = _mm256_mul_ps(w1, v_scale);

                        int local_offset = (b - b_start) * 32 + k * 2;
                        __m256 a0 = _mm256_loadu_ps(&local_a[local_offset]);
                        __m256 a1 = _mm256_loadu_ps(&local_a[local_offset + 8]);

                        sum256 = _mm256_fmadd_ps(w0, a0, sum256);
                        sum256 = _mm256_fmadd_ps(w1, a1, sum256);
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
}

inline void fused_rmsnorm_gemv_q4(const float* x, const float* rms_w, const BlockQ4_32* weights, float* output, int out_dim, int in_dim, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < in_dim; ++i) sum_sq += x[i] * x[i];
    float inv_rms = 1.0f / std::sqrt(sum_sq / in_dim + eps);

    int blocks_per_row = in_dim / 32;
    const int ROW_TILE = 64; 
    const int COL_TILE = 8; 

    #pragma omp parallel for
    for (int i_start = 0; i_start < out_dim; i_start += ROW_TILE) {
        int i_end = (i_start + ROW_TILE > out_dim) ? out_dim : i_start + ROW_TILE;

        for (int b_start = 0; b_start < blocks_per_row; b_start += COL_TILE) {
            int b_end = (b_start + COL_TILE > blocks_per_row) ? blocks_per_row : b_start + COL_TILE;

            // FUSION: Normalize X directly into the fast L1 cache tile
            float local_a[256];
            for (int b = b_start; b < b_end; ++b) {
                for(int k=0; k<32; ++k) {
                    int idx = b * 32 + k;
                    local_a[(b - b_start)*32 + k] = x[idx] * inv_rms * rms_w[idx];
                }
            }

            for (int i = i_start; i < i_end; ++i) {
                const BlockQ4_32* row_w = weights + i * blocks_per_row;
                __m256 sum256 = _mm256_setzero_ps();
                
                for (int b = b_start; b < b_end; ++b) {
                    const BlockQ4_32& block = row_w[b];
                    float scale = f16_to_f32(block.scale);
                    __m256 v_scale = _mm256_set1_ps(scale);

                    for (int k = 0; k < 16; k+=8) {
                        float w_f[16];
                        for(int j=0; j<8; ++j) {
                            uint8_t val = block.weights[k+j];
                            w_f[j*2] = (float)(((int8_t)(val & 0x0F)) - 8);
                            w_f[j*2+1] = (float)(((int8_t)(val >> 4)) - 8);
                        }
                        __m256 w0 = _mm256_loadu_ps(&w_f[0]);
                        __m256 w1 = _mm256_loadu_ps(&w_f[8]);
                        
                        w0 = _mm256_mul_ps(w0, v_scale);
                        w1 = _mm256_mul_ps(w1, v_scale);

                        int local_offset = (b - b_start) * 32 + k * 2;
                        __m256 a0 = _mm256_loadu_ps(&local_a[local_offset]);
                        __m256 a1 = _mm256_loadu_ps(&local_a[local_offset + 8]);

                        sum256 = _mm256_fmadd_ps(w0, a0, sum256);
                        sum256 = _mm256_fmadd_ps(w1, a1, sum256);
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
}
"""

old_gemv_regex = r"void gemv_q4_avx2_row\(.*?output\[i\] \+= sum;\s*\n\s*\}\s*\n\s*\}"

new_code = src
match = re.search(old_gemv_regex, src, re.DOTALL)
if match:
    new_code = src[:match.start()] + new_gemv + src[match.end():]
else:
    print("Could not find gemv")

fusion_old = r"""    std::vector<float> h\(dim\);\s*\n\s*memcpy\(h\.data\(\), x, dim \* sizeof\(float\)\);\s*\n\s*apply_rmsnorm\(h\.data\(\), rms1_w, dim, 1e-5f\);\s*\n\s*int q_dim = num_heads \* head_dim;\s*\n\s*int kv_dim = num_kv_heads \* head_dim;\s*\n\s*int qkv_dim = q_dim \+ 2 \* kv_dim;\s*\n\s*std::vector<float> qkv\(qkv_dim, 0\.0f\);\s*\n\s*gemv_q4_avx2_row\(qkv_w, h\.data\(\), qkv\.data\(\), qkv_dim, dim\);"""

fusion_new = """    int q_dim = num_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;
    int qkv_dim = q_dim + 2 * kv_dim;
    std::vector<float> qkv(qkv_dim, 0.0f);

    // FUSED RMSNorm + QKV Projection tile
    fused_rmsnorm_gemv_q4(x, rms1_w, qkv_w, qkv.data(), qkv_dim, dim, 1e-5f);
    
    std::vector<float> h(dim);
    memcpy(h.data(), x, dim * sizeof(float));
    apply_rmsnorm(h.data(), rms1_w, dim, 1e-5f);"""

new_code2 = re.sub(fusion_old, fusion_new, new_code, flags=re.DOTALL)
if new_code2 == new_code:
    print("Could not find fusion block")

with open('asdsl/kernels/forward_loop.cpp', 'w', encoding='utf-16le') as f:
    f.write(new_code2)
print("done")
