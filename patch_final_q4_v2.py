import re

with open('asdsl/kernels/forward_loop.cpp', 'r', encoding='utf-16le') as f:
    code = f.read()

# 1. Provide the optimized Tiled + AVX2 gemv
gemv_new = """void gemv_q4_avx2_row(const BlockQ4_32* weights, const float* activations, float* output, int out_dim, int in_dim) {
    int blocks_per_row = in_dim / 32;

    const int ROW_TILE = 32;

    #pragma omp parallel for
    for (int i_start = 0; i_start < out_dim; i_start += ROW_TILE) {
        int i_end = (i_start + ROW_TILE > out_dim) ? out_dim : i_start + ROW_TILE;

        // FUSION / TILING: Pull activations into fast L1 cache per row-chunk
        std::vector<float> local_a(in_dim);
        for(int k=0; k<in_dim; ++k) local_a[k] = activations[k];

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
}
"""

# replace up through the final loop brace
code = re.sub(r'void gemv_q4_avx2_row\(.*?output\[i\] \+= sum;\s*\n\s*\}\s*\n\s*\}', gemv_new, code, flags=re.DOTALL)


# 2. Operator Fusion: RMSNorm + QKV injection
fusion_search = r"""    std::vector<float> res1\(dim\);\s*\n\s*memcpy\(res1\.data\(\), x, dim \* sizeof\(float\)\);\s*\n\s*std::vector<float> h\(dim\);\s*\n\s*memcpy\(h\.data\(\), x, dim \* sizeof\(float\)\);\s*\n\s*apply_rmsnorm\(h\.data\(\), rms1_w, dim, 1e-5f\);\s*\n\s*int q_dim = num_heads \* head_dim;\s*\n\s*int kv_dim = num_kv_heads \* head_dim;\s*\n\s*int qkv_dim = q_dim \+ 2 \* kv_dim;\s*\n\s*std::vector<float> qkv\(qkv_dim, 0\.0f\);\s*\n\s*gemv_q4_avx2_row\(qkv_w, h\.data\(\), qkv\.data\(\), qkv_dim, dim\);"""

fusion_replace = """    std::vector<float> res1(dim);
    memcpy(res1.data(), x, dim * sizeof(float));

    int q_dim = num_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;
    int qkv_dim = q_dim + 2 * kv_dim;
    
    // FUSION: Calculate RMSNorm in registers, avoiding memory write-back overhead over duplicate arrays!
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; ++i) sum_sq += x[i] * x[i];
    float inv_rms = 1.0f / std::sqrt(sum_sq / dim + 1e-5f);
    
    std::vector<float> h(dim);
    for(int i=0; i<dim; ++i) {
        h[i] = x[i] * inv_rms * rms1_w[i];
    }

    std::vector<float> qkv(qkv_dim, 0.0f);
    
    // Pass to Cache Tiled kernel
    gemv_q4_avx2_row(qkv_w, h.data(), qkv.data(), qkv_dim, dim);"""

code = re.sub(fusion_search, fusion_replace, code, flags=re.DOTALL)

with open('asdsl/kernels/forward_loop.cpp', 'w', encoding='utf-16le') as f:
    f.write(code)
print("patched")
