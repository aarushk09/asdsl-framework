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
text = re.sub(r"void py_forward_layer\(.*?\}\n\}", "", text, flags=re.DOTALL)
text = re.sub(r"int py_generate_token\(.*?\}\n\}", "", text, flags=re.DOTALL)
# Wait, I'll just use simple textual cuts to avoid messy regex for those.
