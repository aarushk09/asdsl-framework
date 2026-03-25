import re
import codecs

with codecs.open('asdsl/kernels/forward_loop.cpp', 'r', 'utf-16le') as f:
    code = f.read()

# 1. Update gemv_q4_avx2_row to use proper Cache Tiling with Aligned Stack Arrays
gemv_search = r"""void gemv_q4_avx2_row\(const BlockQ4_32\* weights, const float\* activations, float\* output, int out_dim, int in_dim\) \{.*?output\[i\] \+= sum;\s*\n\s*\}\s*\n\s*\}\s*\n\s*\}"""

gemv_replace = """void gemv_q4_avx2_row(const BlockQ4_32* weights, const float* activations, float* output, int out_dim, int in_dim) {
    int blocks_per_row = in_dim / 32;

    const int ROW_TILE = 32;
    const int COL_TILE = 256; // Process 256 elements (8 blocks) per cache tile

    #pragma omp parallel for
    for (int i_start = 0; i_start < out_dim; i_start += ROW_TILE) {
        int i_end = (i_start + ROW_TILE > out_dim) ? out_dim : i_start + ROW_TILE;
        int num_rows = i_end - i_start;
        alignas(32) float row_sums[32] = {0.0f};

        for (int k_start = 0; k_start < in_dim; k_start += COL_TILE) {
            int k_end = (k_start + COL_TILE > in_dim) ? in_dim : k_start + COL_TILE;
            int num_k = k_end - k_start;
            
            // ZERO-ALLOCATION ALIGNED STACK ARRAY TILING
            alignas(32) float local_a[256]; 
            for(int k=0; k < num_k; ++k) {
                local_a[k] = activations[k_start + k];
            }

            for (int r = 0; r < num_rows; ++r) {
                int i = i_start + r;
                const BlockQ4_32* row_w = weights + i * blocks_per_row;
                __m256 sum256 = _mm256_setzero_ps();
                
                int b_start = k_start / 32;
                int b_end = k_end / 32;

                for (int b = b_start; b < b_end; ++b) {
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
                        __m256 a0 = _mm256_loadu_ps(&local_a[(b - b_start) * 32 + s]);
                        sum256 = _mm256_fmadd_ps(w0, a0, sum256);
                    }
                }

                float temp[8];
                _mm256_storeu_ps(temp, sum256);
                float sum = 0.0f;
                for(int j=0; j<8; ++j) sum += temp[j];
                row_sums[r] += sum;
            }
        }

        // Add accumulated sums correctly to output
        for(int r = 0; r < num_rows; ++r) {
            output[i_start + r] += row_sums[r];
        }
    }
}"""

code = re.sub(gemv_search, gemv_replace, code, flags=re.DOTALL)


# 2. Update forward_layer to completely remove std::vector calls
layer_search = r"""void forward_layer\(float\* x, const float\* rms1_w, const BlockQ4_32\* qkv_w, const BlockQ4_32\* o_w, const float\* rms2_w,.*?for\(int i=0; i<dim; \+\+i\) x\[i\] = res2\[i\] \+ down\[i\];\s*\n\s*\}"""

layer_replace = """void forward_layer(float* x, const float* rms1_w, const BlockQ4_32* qkv_w, const BlockQ4_32* o_w, const float* rms2_w,
                   const BlockQ4_32* gate_w, const BlockQ4_32* up_w, const BlockQ4_32* down_w,
                   int dim, int hidden_dim, int num_heads, int num_kv_heads, int head_dim,
                   int layer_id, int seq_pos, KVCache& cache) {

    alignas(32) float res1[18000];
    memcpy(res1, x, dim * sizeof(float));

    int q_dim = num_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;
    int qkv_dim = q_dim + 2 * kv_dim;

    float sum_sq = 0.0f;
    for (int i = 0; i < dim; ++i) sum_sq += x[i] * x[i];
    float inv_rms = 1.0f / std::sqrt(sum_sq / dim + 1e-5f);

    alignas(32) float h[18000];
    for(int i=0; i<dim; ++i) {
        h[i] = x[i] * inv_rms * rms1_w[i];
    }

    alignas(32) float qkv[10000];
    memset(qkv, 0, qkv_dim * sizeof(float));

    // Pass to Cache Tiled kernel
    gemv_q4_avx2_row(qkv_w, h, qkv, qkv_dim, dim);

    float* q = qkv;
    float* k = qkv + q_dim;
    float* v = qkv + q_dim + kv_dim;

    apply_rope(q, k, seq_pos, head_dim, num_heads, num_kv_heads, 10000.0f);    

    alignas(32) float attn_out[10000];
    memset(attn_out, 0, q_dim * sizeof(float));
    compute_attention(attn_out, q, k, v, layer_id, seq_pos, num_heads, cache);

    alignas(32) float o_out[18000];
    memset(o_out, 0, dim * sizeof(float));
    gemv_q4_avx2_row(o_w, attn_out, o_out, dim, q_dim);

    for(int i=0; i<dim; ++i) x[i] = res1[i] + o_out[i];

    alignas(32) float res2[18000];
    memcpy(res2, x, dim * sizeof(float));

    memcpy(h, x, dim * sizeof(float));
    apply_rmsnorm(h, rms2_w, dim, 1e-5f);

    alignas(32) float gate[18000];
    memset(gate, 0, hidden_dim * sizeof(float));
    alignas(32) float up[18000];
    memset(up, 0, hidden_dim * sizeof(float));

    gemv_q4_avx2_row(gate_w, h, gate, hidden_dim, dim);
    gemv_q4_avx2_row(up_w, h, up, hidden_dim, dim);

    #pragma omp parallel for
    for (int i=0; i<hidden_dim; ++i) {
        float x_gate = gate[i];
        float silu = x_gate / (1.0f + std::exp(-x_gate));
        gate[i] = silu * up[i];
    }

    alignas(32) float down[18000];
    memset(down, 0, dim * sizeof(float));
    gemv_q4_avx2_row(down_w, gate, down, dim, hidden_dim);       

    for(int i=0; i<dim; ++i) x[i] = res2[i] + down[i];
}"""

code = re.sub(layer_search, layer_replace, code, flags=re.DOTALL)

with codecs.open('asdsl/kernels/forward_loop.cpp', 'w', 'utf-16le') as f:
    f.write(code)

print("Zero-Allocation Cache Tiling applied successfully.")
