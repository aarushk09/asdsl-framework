import re
import codecs

with codecs.open('asdsl/kernels/forward_loop.cpp', 'r', 'utf-16le') as f:
    code = f.read()

# 1. Provide the InferenceArena definition before forward_layer
arena_def = """
struct InferenceArena {
    float* res1;
    float* h;
    float* qkv;
    float* attn_out;
    float* o_out;
    float* res2;
    float* gate;
    float* up;
    float* down;

    InferenceArena(int dim, int hidden_dim, int q_dim, int qkv_dim) {
        res1 = (float*)_aligned_malloc(dim * sizeof(float), 32);
        h = (float*)_aligned_malloc(dim * sizeof(float), 32);
        qkv = (float*)_aligned_malloc(qkv_dim * sizeof(float), 32);
        attn_out = (float*)_aligned_malloc(q_dim * sizeof(float), 32);
        o_out = (float*)_aligned_malloc(dim * sizeof(float), 32);
        res2 = (float*)_aligned_malloc(dim * sizeof(float), 32);
        gate = (float*)_aligned_malloc(hidden_dim * sizeof(float), 32);
        up = (float*)_aligned_malloc(hidden_dim * sizeof(float), 32);
        down = (float*)_aligned_malloc(dim * sizeof(float), 32);
    }

    ~InferenceArena() {
        _aligned_free(res1);
        _aligned_free(h);
        _aligned_free(qkv);
        _aligned_free(attn_out);
        _aligned_free(o_out);
        _aligned_free(res2);
        _aligned_free(gate);
        _aligned_free(up);
        _aligned_free(down);
    }
};

static InferenceArena* global_arena = nullptr;

"""

if "struct InferenceArena" not in code:
    code = code.replace("void forward_layer(float* x", arena_def + "void forward_layer(float* x")

# 2. Fix gemv_q4_avx2_row to remove local_a copy and load direct from activations
gemv_search = r"""        // Zero-Allocation Memory Tiling! Replace std::vector with aligned C-array.\s*\n\s*alignas\(32\) float local_a\[MAX_DIM\];\s*\n\s*for\(int k=0; k<in_dim; \+\+k\) \{\s*\n\s*local_a\[k\] = activations\[k\];\s*\n\s*\}"""
code = re.sub(gemv_search, "", code)

gemv_load_search = r"""__m256 a0 = _mm256_loadu_ps\(&local_a\[b \* 32 \+ s\]\);"""
gemv_load_replace = r"""__m256 a0 = _mm256_loadu_ps(&activations[b * 32 + s]);"""
code = re.sub(gemv_load_search, gemv_load_replace, code)

# 3. Modify forward_layer to accept InferenceArena* and use its buffers instead of stack allocations
layer_search = r"""void forward_layer\(float\* x, const float\* rms1_w, const BlockQ4_32\* qkv_w, const BlockQ4_32\* o_w, const float\* rms2_w,
                   const BlockQ4_32\* gate_w, const BlockQ4_32\* up_w, const BlockQ4_32\* down_w,
                   int dim, int hidden_dim, int num_heads, int num_kv_heads, int head_dim,
                   int layer_id, int seq_pos, KVCache& cache\) \{.*?for\(int i=0; i<dim; \+\+i\) x\[i\] = res2\[i\] \+ down\[i\];\s*\n\s*\}"""

layer_replace = """void forward_layer(float* x, const float* rms1_w, const BlockQ4_32* qkv_w, const BlockQ4_32* o_w, const float* rms2_w,
                   const BlockQ4_32* gate_w, const BlockQ4_32* up_w, const BlockQ4_32* down_w,
                   int dim, int hidden_dim, int num_heads, int num_kv_heads, int head_dim,
                   int layer_id, int seq_pos, KVCache& cache, InferenceArena* arena) {

    float* res1 = arena->res1;
    float* h = arena->h;
    float* qkv = arena->qkv;
    float* attn_out = arena->attn_out;
    float* o_out = arena->o_out;
    float* res2 = arena->res2;
    float* gate = arena->gate;
    float* up = arena->up;
    float* down = arena->down;

    memcpy(res1, x, dim * sizeof(float));

    int q_dim = num_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;
    int qkv_dim = q_dim + 2 * kv_dim;

    float sum_sq = 0.0f;
    for (int i = 0; i < dim; ++i) sum_sq += x[i] * x[i];
    float inv_rms = 1.0f / std::sqrt(sum_sq / dim + 1e-5f);

    for(int i=0; i<dim; ++i) {
        h[i] = x[i] * inv_rms * rms1_w[i];
    }

    memset(qkv, 0, qkv_dim * sizeof(float));

    // Pass to Cache Tiled kernel
    gemv_q4_avx2_row(qkv_w, h, qkv, qkv_dim, dim);

    float* q = qkv;
    float* k = qkv + q_dim;
    float* v = qkv + q_dim + kv_dim;

    apply_rope(q, k, seq_pos, head_dim, num_heads, num_kv_heads, 10000.0f);    

    memset(attn_out, 0, q_dim * sizeof(float));
    compute_attention(attn_out, q, k, v, layer_id, seq_pos, num_heads, cache);

    memset(o_out, 0, dim * sizeof(float));
    gemv_q4_avx2_row(o_w, attn_out, o_out, dim, q_dim);

    for(int i=0; i<dim; ++i) x[i] = res1[i] + o_out[i];

    memcpy(res2, x, dim * sizeof(float));

    memcpy(h, x, dim * sizeof(float));
    apply_rmsnorm(h, rms2_w, dim, 1e-5f);

    memset(gate, 0, hidden_dim * sizeof(float));
    memset(up, 0, hidden_dim * sizeof(float));

    gemv_q4_avx2_row(gate_w, h, gate, hidden_dim, dim);
    gemv_q4_avx2_row(up_w, h, up, hidden_dim, dim);

    #pragma omp parallel for
    for (int i=0; i<hidden_dim; ++i) {
        float x_gate = gate[i];
        float silu = x_gate / (1.0f + std::exp(-x_gate));
        gate[i] = silu * up[i];
    }

    memset(down, 0, dim * sizeof(float));
    gemv_q4_avx2_row(down_w, gate, down, dim, hidden_dim);       

    for(int i=0; i<dim; ++i) x[i] = res2[i] + down[i];
}"""

code = re.sub(layer_search, layer_replace, code, flags=re.DOTALL)


# 4. Modify generate_token_mmap to init and pass the global arena, and print NAN
gen_search = r"""int generate_token_mmap.*?forward_layer\(x\.data\(\), rms1_w, qkv_w, o_w, rms2_w, gate_w, up_w, down_w, dim, hidden_dim, num_heads, num_kv_heads, head_dim, l, seq_pos, cache\);.*?float max_val = logits\[0\];"""

def gen_replace(match):
    matched = match.group(0)
    
    # insert arena initialization before layer loop
    arena_init = """    int q_dim = num_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;
    int qkv_dim = q_dim + 2 * kv_dim;
    if (global_arena == nullptr) {
        global_arena = new InferenceArena(dim, hidden_dim, q_dim, qkv_dim);
    }
    for (int l = 0; l < num_layers; ++l) {"""
    
    m1 = re.sub(r'for \(int l = 0; l < num_layers; \+\+l\) \{', arena_init, matched)
    
    # update forward_layer call
    m2 = m1.replace("cache);", "cache, global_arena);")
    
    # insert NaN check after max_val = logits[0];
    m3 = m2 + """
    if (std::isnan(max_val) || std::isinf(max_val)) {
        printf("NAN OR INF DETECTED IN LOGITS! max_val: %f\\n", max_val);
    }"""
    return m3

code = re.sub(gen_search, gen_replace, code, flags=re.DOTALL)

with codecs.open('asdsl/kernels/forward_loop.cpp', 'w', 'utf-16le') as f:
    f.write(code)

print("Patch applied for global Arena and direct activations tracking.")
