import re

with open("asdsl/kernels/forward_loop.cpp", "r") as f:
    text = f.read()

# 1. Replace gemv_q4_avx2 with the full version
new_gemv = """
// helper to convert FP16 to FP32
static inline float f16_to_f32(uint16_t h) {
    __m128i vi = _mm_cvtsi32_si128(h);
    __m128 v = _mm_cvtph_ps(vi);
    return _mm_cvtss_f32(v);
}

void gemv_q4_avx2_row(const BlockQ4_32* weights, const float* activations, float* output, int out_dim, int in_dim) {
    int blocks_per_row = in_dim / 32;

    #pragma omp parallel for
    for (int i = 0; i < out_dim; ++i) {
        float sum = 0.0f;
        const BlockQ4_32* row_w = weights + i * blocks_per_row;
        const float* row_a = activations;

        for (int b = 0; b < blocks_per_row; ++b) {
            const BlockQ4_32& block = row_w[b];
            float scale = f16_to_f32(block.scale);

            for (int k = 0; k < 16; ++k) {
                uint8_t val = block.weights[k];
                // unpacking logic matched to python quantizer
                float w0 = (float)(((int8_t)(val & 0x0F)) - 8) * scale;
                float w1 = (float)(((int8_t)(val >> 4)) - 8) * scale;
                sum += w0 * row_a[b * 32 + k * 2];
                sum += w1 * row_a[b * 32 + k * 2 + 1];
            }
        }
        output[i] += sum;
    }
}
"""

text = re.sub(
    r"void gemv_q4_avx2\(const BlockQ4_32\* .*?\}\n\}",
    new_gemv.strip(),
    text,
    flags=re.DOTALL
)

# 2. Update forward_layer signature and body
old_fwd_sig = """void forward_layer(float* x, const float* rms1_w, const float* qkv_w, const float* o_w, const float* rms2_w, 
                  const float* gate_w, const float* up_w, const float* down_w,"""
new_fwd_sig = """void forward_layer(float* x, const float* rms1_w, const BlockQ4_32* qkv_w, const BlockQ4_32* o_w, const float* rms2_w, 
                  const BlockQ4_32* gate_w, const BlockQ4_32* up_w, const BlockQ4_32* down_w,"""

text = text.replace(old_fwd_sig, new_fwd_sig)

text = text.replace("gemv_f32(qkv_w, h.data(), qkv.data(), qkv_dim, dim);", "gemv_q4_avx2_row(qkv_w, h.data(), qkv.data(), qkv_dim, dim);")
text = text.replace("gemv_f32(o_w, attn_out.data(), o_out.data(), dim, q_dim);", "gemv_q4_avx2_row(o_w, attn_out.data(), o_out.data(), dim, q_dim);")
text = text.replace("gemv_f32(gate_w, h.data(), gate.data(), hidden_dim, dim);", "gemv_q4_avx2_row(gate_w, h.data(), gate.data(), hidden_dim, dim);")
text = text.replace("gemv_f32(up_w, h.data(), up.data(), hidden_dim, dim);", "gemv_q4_avx2_row(up_w, h.data(), up.data(), hidden_dim, dim);")
text = text.replace("gemv_f32(down_w, gate.data(), down.data(), dim, hidden_dim);", "gemv_q4_avx2_row(down_w, gate.data(), down.data(), dim, hidden_dim);")

# 3. py_forward_layer (remove completely if not strictly used as it takes py::list of arrays but user wants mmap integration)
# Actually, the user script only calls generate_token. I'll replace generate_token explicitly!

old_gen = r"int generate_token\(int token_id.*?int max_id = i;\n        \}\n    \}\n\n    return max_id;\n\}"
new_gen = """
int generate_token_mmap(int token_id, int seq_pos,
                        MmapWeights& store,
                        int num_layers, int dim, int hidden_dim, int num_heads, int num_kv_heads, int head_dim, int vocab_size,
                        KVCache& cache) {

    const float* token_emb = (const float*)store.tensors["embed"];
    const float* final_rms_w = (const float*)store.tensors["final_rms"];
    const float* lm_head_w = (const float*)store.tensors["lm_head"];

    std::vector<float> x(dim);
    memcpy(x.data(), token_emb + (size_t)token_id * dim, dim * sizeof(float));

    for (int l = 0; l < num_layers; ++l) {
        std::string pfx = "l" + std::to_string(l) + "_";
        const float* rms1_w = (const float*)store.tensors[pfx + "rms1"];
        const float* rms2_w = (const float*)store.tensors[pfx + "rms2"];
        const BlockQ4_32* qkv_w  = (const BlockQ4_32*)store.tensors[pfx + "qkv"];
        const BlockQ4_32* o_w    = (const BlockQ4_32*)store.tensors[pfx + "o"];
        const BlockQ4_32* gate_w = (const BlockQ4_32*)store.tensors[pfx + "gate"];
        const BlockQ4_32* up_w   = (const BlockQ4_32*)store.tensors[pfx + "up"];
        const BlockQ4_32* down_w = (const BlockQ4_32*)store.tensors[pfx + "down"];

        forward_layer(x.data(), rms1_w, qkv_w, o_w, rms2_w, gate_w, up_w, down_w,
                      dim, hidden_dim, num_heads, num_kv_heads, head_dim,
                      l, seq_pos, cache);
    }

    apply_rmsnorm(x.data(), final_rms_w, dim, 1e-5f);

    std::vector<float> logits(vocab_size, 0.0f);
    gemv_f32(lm_head_w, x.data(), logits.data(), vocab_size, dim);

    int max_id = 0;
    float max_val = logits[0];
    for (int i = 1; i < vocab_size; ++i) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_id = i;
        }
    }

    return max_id;
}
"""

text = re.sub(old_gen, new_gen.strip(), text, flags=re.DOTALL)

# Remove py_generate_token from PYBIND11_MODULE and add generate_token_mmap
text = re.sub(r'm\.def\("generate_token".*?\);', 'm.def("generate_token_mmap", &generate_token_mmap, "Generate single token directly from Q4 mmap storage");', text)

with open("asdsl/kernels/forward_loop.cpp", "w") as f:
    f.write(text)