import re

with open('asdsl/kernels/forward_loop.cpp', 'r') as f:
    text = f.read()

replacement = """
void pin_thread_to_core(int core_id) {
    DWORD_PTR mask = (1ULL << core_id);
    SetThreadAffinityMask(GetCurrentThread(), mask);
}

struct KVCache {
    int num_layers;
    int max_seq_len;
    int num_kv_heads;
    int head_dim;
    std::vector<float> k_cache;
    std::vector<float> v_cache;

    KVCache(int layers, int seq_len, int kv_heads, int dim)
        : num_layers(layers), max_seq_len(seq_len), num_kv_heads(kv_heads), head_dim(dim) {
        size_t size = (size_t)layers * seq_len * kv_heads * dim;
        k_cache.resize(size, 0.0f);
        v_cache.resize(size, 0.0f);
    }

    inline size_t get_offset(int layer, int pos) const {
        return ((size_t)layer * max_seq_len + pos) * num_kv_heads * head_dim;
    }

    void set_history(int layer, int pos, const float* k, const float* v) {
        size_t off = get_offset(layer, pos);
        memcpy(&k_cache[off], k, num_kv_heads * head_dim * sizeof(float));
        memcpy(&v_cache[off], v, num_kv_heads * head_dim * sizeof(float));
    }
};

void compute_attention(float* out, const float* q, const float* k, const float* v, int layer_id, int seq_pos, int num_heads, KVCache& cache) {
    int num_kv_heads = cache.num_kv_heads;
    int head_dim = cache.head_dim;
    float scale = 1.0f / std::sqrt((float)head_dim);
    int groups = num_heads / num_kv_heads;
    
    size_t cache_off = cache.get_offset(layer_id, seq_pos);
    memcpy(&cache.k_cache[cache_off], k, num_kv_heads * head_dim * sizeof(float));
    memcpy(&cache.v_cache[cache_off], v, num_kv_heads * head_dim * sizeof(float));
    
    std::vector<float> scores(seq_pos + 1);

    for (int h = 0; h < num_heads; ++h) {
        int kv_h = h / groups;
        const float* q_h = q + h * head_dim;
        float* out_h = out + h * head_dim;
        float max_score = -1e9f;
        
        for (int p = 0; p <= seq_pos; ++p) {
            const float* k_p = &cache.k_cache[cache.get_offset(layer_id, p) + kv_h * head_dim];
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += q_h[d] * k_p[d];
            }
            score *= scale;
            scores[p] = score;
            if (score > max_score) max_score = score;
        }
        
        float exp_sum = 0.0f;
        for (int p = 0; p <= seq_pos; ++p) {
            scores[p] = std::exp(scores[p] - max_score);
            exp_sum += scores[p];
        }
        for (int p = 0; p <= seq_pos; ++p) {
            scores[p] /= exp_sum;
        }
        
        for (int d = 0; d < head_dim; ++d) {
            out_h[d] = 0.0f;
        }
        for (int p = 0; p <= seq_pos; ++p) {
            float s = scores[p];
            const float* v_p = &cache.v_cache[cache.get_offset(layer_id, p) + kv_h * head_dim];
            for (int d = 0; d < head_dim; ++d) {
                out_h[d] += s * v_p[d];
            }
        }
    }
}
"""
text = re.sub(r'void pin_thread_to_core\(int core_id\).*?\};\n', replacement, text, flags=re.DOTALL)

with open('asdsl/kernels/forward_loop.cpp', 'w') as f:
    f.write(text)

"""
