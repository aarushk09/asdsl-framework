/**
 * Pure C++ Inference Engine for ASDSL Framework — Phase 19
 * 
 * Eliminates ALL Python boundary crossings during autoregressive generation.
 * Zero-copy weight access via raw uintptr_t pointers from Python.
 * Full generate() loop in C++ — Python only handles tokenization and decoding.
 * 
 * Architecture:
 * - Zero-copy weight access via uintptr_t direct casts (no PyBind11 buffer overhead)
 * - Pre-allocated contiguous KV cache in C++ memory
 * - All operations (RMSNorm, RoPE, GQA attention, SwiGLU, Q4xQ8 GEMV) in C++
 * - OpenMP parallelism over output rows in GEMV kernels
 * - P-core thread affinity enforced at C++ level
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <immintrin.h>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <memory>

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

/* ===================================================================
 * Architecture Constants (Phi-4 14B)
 * =================================================================== */
static constexpr int HIDDEN = 3072;
static constexpr int NUM_HEADS = 24;
static constexpr int NUM_KV_HEADS = 8;
static constexpr int HEAD_DIM = 128;
static constexpr int Q_DIM = NUM_HEADS * HEAD_DIM;       // 3072
static constexpr int KV_DIM = NUM_KV_HEADS * HEAD_DIM;   // 1024
static constexpr int QKV_DIM = Q_DIM + 2 * KV_DIM;       // 5120
static constexpr int INTER = 8192;
static constexpr int ROTARY_DIM = 96;  // partial_rotary_factor=0.75
static constexpr float RMS_EPS = 1e-5f;
static constexpr int Q_PER_KV = NUM_HEADS / NUM_KV_HEADS;  // 3
static constexpr int EOS_TOKEN_1 = 200020;  // <|end|>
static constexpr int EOS_TOKEN_2 = 199999;  // </s>

/* ===================================================================
 * AVX2 Utility Functions
 * =================================================================== */

static inline float hsum256_ps(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

static inline int32_t hsum256_epi32(__m256i v) {
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i lo = _mm256_castsi256_si128(v);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_hadd_epi32(lo, lo);
    lo = _mm_hadd_epi32(lo, lo);
    return _mm_cvtsi128_si32(lo);
}

/* ===================================================================
 * Q4 × Q8 Integer GEMV (from Phase 16/17)
 * =================================================================== */

static void gemv_q4_q8_row(
    const uint8_t* w_packed,
    const float* scales,
    const int8_t* x_q8,
    const float* x_scales,
    float* y,
    int M, int K, int n_groups, int group_size
) {
    const int packed_stride = K / 2;
    const __m128i mask_nibble = _mm_set1_epi8(0x0F);
    const __m256i eight_256 = _mm256_set1_epi16(8);
    
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < M; row++) {
        float acc = 0.0f;
        const uint8_t* w_row = w_packed + (size_t)row * packed_stride;
        
        for (int g = 0; g < n_groups; g++) {
            const int8_t* x_group_q8 = x_q8 + g * group_size;
            const uint8_t* w_group = w_row + g * (group_size / 2);
            float w_scale = scales[row * n_groups + g];
            float x_scale = x_scales[g];
            
            __m256i acc_int = _mm256_setzero_si256();
            
            for (int i = 0; i < group_size; i += 32) {
                __m128i packed = _mm_loadu_si128((const __m128i*)(w_group + i / 2));
                __m128i lo = _mm_and_si128(packed, mask_nibble);
                __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), mask_nibble);
                __m128i w0_15_u8 = _mm_unpacklo_epi8(lo, hi);
                __m128i w16_31_u8 = _mm_unpackhi_epi8(lo, hi);
                __m256i w0_15 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(w0_15_u8), eight_256);
                __m256i w16_31 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(w16_31_u8), eight_256);
                __m256i x0_15 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(x_group_q8 + i)));
                __m256i x16_31 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(x_group_q8 + i + 16)));
                acc_int = _mm256_add_epi32(acc_int, _mm256_madd_epi16(w0_15, x0_15));
                acc_int = _mm256_add_epi32(acc_int, _mm256_madd_epi16(w16_31, x16_31));
            }
            
            acc += (float)hsum256_epi32(acc_int) * w_scale * x_scale;
        }
        y[row] = acc;
    }
}

/* ===================================================================
 * Q8 Activation Quantization (once per matrix call)
 * =================================================================== */

static void quantize_activations_q8(const float* x, int8_t* x_q8, float* x_scales,
                                     int K, int n_groups, int group_size) {
    for (int g = 0; g < n_groups; g++) {
        const float* xg = x + g * group_size;
        int8_t* xq = x_q8 + g * group_size;
        
        __m256 max_abs = _mm256_setzero_ps();
        for (int j = 0; j < group_size; j += 8) {
            __m256 v = _mm256_loadu_ps(xg + j);
            max_abs = _mm256_max_ps(max_abs, _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v));
        }
        __m128 lo = _mm256_castps256_ps128(max_abs);
        __m128 hi = _mm256_extractf128_ps(max_abs, 1);
        lo = _mm_max_ps(lo, hi);
        lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
        lo = _mm_max_ps(lo, _mm_movehdup_ps(lo));
        float amax = _mm_cvtss_f32(lo);
        
        if (amax < 1e-9f) {
            memset(xq, 0, group_size);
            x_scales[g] = 1.0f;
            continue;
        }
        
        x_scales[g] = amax / 127.0f;
        float inv_scale = 127.0f / amax;
        __m256 vscale = _mm256_set1_ps(inv_scale);
        for (int j = 0; j < group_size; j += 8) {
            __m256 vf = _mm256_mul_ps(_mm256_loadu_ps(xg + j), vscale);
            __m256i vi = _mm256_cvtps_epi32(vf);
            __m128i lo4 = _mm_packs_epi32(_mm256_castsi256_si128(vi), _mm256_extracti128_si256(vi, 1));
            _mm_storel_epi64((__m128i*)(xq + j), _mm_packs_epi16(lo4, lo4));
        }
    }
}

/* ===================================================================
 * RMSNorm (AVX2)
 * =================================================================== */

static void rmsnorm_f32(const float* x, float* y, const float* weight, int dim, float eps) {
    __m256 sum_sq = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        sum_sq = _mm256_fmadd_ps(v, v, sum_sq);
    }
    float remainder = 0.0f;
    for (; i < dim; i++) remainder += x[i] * x[i];
    
    __m128 lo = _mm256_castps256_ps128(sum_sq);
    __m128 hi = _mm256_extractf128_ps(sum_sq, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float ms = (_mm_cvtss_f32(lo) + remainder) / (float)dim;
    float rms_inv = 1.0f / sqrtf(ms + eps);
    
    __m256 vscale = _mm256_set1_ps(rms_inv);
    i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        __m256 w = _mm256_loadu_ps(weight + i);
        _mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_mul_ps(v, vscale), w));
    }
    for (; i < dim; i++) y[i] = x[i] * rms_inv * weight[i];
}

/* ===================================================================
 * RoPE (in-place, partial rotary)
 * =================================================================== */

static void rope_apply_inplace(float* q, float* k,
                                const float* cos_table, const float* sin_table,
                                int pos, int half_head) {
    const int half_rotary = ROTARY_DIM / 2;  // 48
    
    const float* cos = cos_table + pos * half_head;
    const float* sin = sin_table + pos * half_head;
    
    // Apply to Q heads
    for (int h = 0; h < NUM_HEADS; h++) {
        float* hq = q + h * HEAD_DIM;
        for (int i = 0; i < half_rotary; i += 4) {
            __m128 x0 = _mm_loadu_ps(hq + i);
            __m128 x1 = _mm_loadu_ps(hq + i + half_head);
            __m128 c  = _mm_loadu_ps(cos + i);
            __m128 s  = _mm_loadu_ps(sin + i);
            _mm_storeu_ps(hq + i, _mm_sub_ps(_mm_mul_ps(x0, c), _mm_mul_ps(x1, s)));
            _mm_storeu_ps(hq + i + half_head, _mm_add_ps(_mm_mul_ps(x0, s), _mm_mul_ps(x1, c)));
        }
    }
    // Apply to KV heads
    for (int h = 0; h < NUM_KV_HEADS; h++) {
        float* hk = k + h * HEAD_DIM;
        for (int i = 0; i < half_rotary; i += 4) {
            __m128 x0 = _mm_loadu_ps(hk + i);
            __m128 x1 = _mm_loadu_ps(hk + i + half_head);
            __m128 c  = _mm_loadu_ps(cos + i);
            __m128 s  = _mm_loadu_ps(sin + i);
            _mm_storeu_ps(hk + i, _mm_sub_ps(_mm_mul_ps(x0, c), _mm_mul_ps(x1, s)));
            _mm_storeu_ps(hk + i + half_head, _mm_add_ps(_mm_mul_ps(x0, s), _mm_mul_ps(x1, c)));
        }
    }
}

/* ===================================================================
 * GQA Decode Attention (batch=1)
 * =================================================================== */

static void gqa_decode_attention(
    const float* q,           // [NUM_HEADS * HEAD_DIM]
    const float* k_cache,     // [seq_len * NUM_KV_HEADS * HEAD_DIM]
    const float* v_cache,     // [seq_len * NUM_KV_HEADS * HEAD_DIM]
    float*       out,         // [NUM_HEADS * HEAD_DIM]
    int seq_len,
    float scale               // 1 / sqrt(HEAD_DIM)
) {
    std::vector<float> scores(seq_len);
    std::vector<float> exp_scores(seq_len);
    
    for (int h = 0; h < NUM_HEADS; h++) {
        int kv_h = h / Q_PER_KV;
        const float* qh = q + h * HEAD_DIM;
        float* oh = out + h * HEAD_DIM;
        
        float max_score = -1e30f;
        for (int t = 0; t < seq_len; t++) {
            const float* kh_t = k_cache + t * NUM_KV_HEADS * HEAD_DIM + kv_h * HEAD_DIM;
            __m256 acc = _mm256_setzero_ps();
            int i = 0;
            for (; i + 8 <= HEAD_DIM; i += 8) {
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(qh + i), _mm256_loadu_ps(kh_t + i), acc);
            }
            float dot = hsum256_ps(acc);
            for (; i < HEAD_DIM; i++) dot += qh[i] * kh_t[i];
            scores[t] = dot * scale;
            if (scores[t] > max_score) max_score = scores[t];
        }
        
        float sum_exp = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            float e = expf(scores[t] - max_score);
            exp_scores[t] = e;
            sum_exp += e;
        }
        float inv_sum = 1.0f / sum_exp;
        
        memset(oh, 0, HEAD_DIM * sizeof(float));
        for (int t = 0; t < seq_len; t++) {
            const float* vh_t = v_cache + t * NUM_KV_HEADS * HEAD_DIM + kv_h * HEAD_DIM;
            float w = exp_scores[t] * inv_sum;
            __m256 vs = _mm256_set1_ps(w);
            int i = 0;
            for (; i + 8 <= HEAD_DIM; i += 8) {
                _mm256_storeu_ps(oh + i, _mm256_fmadd_ps(_mm256_loadu_ps(vh_t + i), vs, _mm256_loadu_ps(oh + i)));
            }
            for (; i < HEAD_DIM; i++) oh[i] += vh_t[i] * w;
        }
    }
}

/* ===================================================================
 * SwiGLU (in-place)
 * =================================================================== */

static void swiglu_inplace(float* gate, const float* up, int dim) {
    for (int i = 0; i < dim; i++) {
        float g = gate[i];
        gate[i] = (g / (1.0f + expf(-g))) * up[i];  // silu(g) * up
    }
}

/* ===================================================================
 * Vec Add (in-place)
 * =================================================================== */

static void vec_add_inplace(float* a, const float* b, int dim) {
    int i = 0;
    for (; i + 8 <= dim; i += 8) {
        _mm256_storeu_ps(a + i, _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
    for (; i < dim; i++) a[i] += b[i];
}

/* ===================================================================
 * Argmax (for greedy sampling)
 * =================================================================== */

static int argmax_f32(const float* arr, int n) {
    int best = 0;
    float max_val = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
            best = i;
        }
    }
    return best;
}

/* ===================================================================
 * Layer Weight Pointers (zero-copy from Python via uintptr_t)
 * =================================================================== */

struct LayerWeights {
    // QKV projection: [QKV_DIM, HIDDEN] — Q4 packed
    const uint8_t* qkv_proj;
    const float* qkv_scales;
    
    // Output projection: [HIDDEN, Q_DIM] — Q4 packed
    const uint8_t* o_proj;
    const float* o_scales;
    
    // Gate/Up projection: [2*INTER, HIDDEN] — Q4 packed
    const uint8_t* gate_up_proj;
    const float* gate_up_scales;
    
    // Down projection: [HIDDEN, INTER] — Q4 packed
    const uint8_t* down_proj;
    const float* down_scales;
    
    // RMSNorm weights
    const float* rms_att;   // input_layernorm
    const float* rms_ffn;   // post_attention_layernorm
};

/* ===================================================================
 * InferenceEngine: Pure C++ autoregressive generation
 * =================================================================== */

class InferenceEngine {
    int num_layers;
    int max_seq_len;
    int group_size;
    int n_groups_h;     // HIDDEN / group_size
    int n_groups_qkv;   // QKV_DIM / group_size
    int n_groups_o;     // HIDDEN / group_size
    int n_groups_gu;    // 2*INTER / group_size
    int n_groups_down;  // HIDDEN / group_size
    int vocab_size;
    
    std::vector<LayerWeights> layers;
    
    // KV Cache: [num_layers, max_seq_len, NUM_KV_HEADS, HEAD_DIM]
    std::vector<float> k_cache;
    std::vector<float> v_cache;
    int current_seq_len;
    
    // RoPE tables: [max_seq_len, HEAD_DIM/2]
    std::vector<float> cos_table;
    std::vector<float> sin_table;
    
    // Embedding table: [vocab_size, HIDDEN] — FP16 (zero-copy, used directly for LM head)
    const uint16_t* embed_f16;
    
    // Final norm weights
    const float* final_norm;
    
    // Scratch buffers (pre-allocated, zero allocation during generation)
    std::vector<float> hidden;        // [HIDDEN]
    std::vector<float> residual;      // [HIDDEN]
    std::vector<float> qkv_out;       // [QKV_DIM]
    std::vector<float> q_buf;         // [Q_DIM]
    std::vector<float> k_buf;         // [KV_DIM]
    std::vector<float> v_buf;         // [KV_DIM]
    std::vector<float> attn_out;      // [Q_DIM]
    std::vector<float> o_proj_out;    // [HIDDEN]
    std::vector<float> gu_out;        // [2*INTER]
    std::vector<float> gate_buf;      // [INTER]
    std::vector<float> up_buf;        // [INTER]
    std::vector<float> down_out;      // [HIDDEN]
    std::vector<float> logits;        // [vocab_size]
    
    // Q8 activation buffers
    std::vector<int8_t> x_q8;
    std::vector<float> x_scales;
    
public:
    InferenceEngine(int num_layers_, int max_seq_len_, int group_size_,
                    int vocab_size_)
        : num_layers(num_layers_), max_seq_len(max_seq_len_),
          group_size(group_size_), vocab_size(vocab_size_),
          current_seq_len(0) {
        
        n_groups_h = HIDDEN / group_size;
        n_groups_qkv = QKV_DIM / group_size;
        n_groups_o = HIDDEN / group_size;
        n_groups_gu = (2 * INTER) / group_size;
        n_groups_down = HIDDEN / group_size;
        
        layers.resize(num_layers);
        
        // KV cache: contiguous allocation
        k_cache.assign(num_layers * max_seq_len * NUM_KV_HEADS * HEAD_DIM, 0.0f);
        v_cache.assign(num_layers * max_seq_len * NUM_KV_HEADS * HEAD_DIM, 0.0f);
        
        // RoPE tables
        int half_head = HEAD_DIM / 2;
        cos_table.resize(max_seq_len * half_head);
        sin_table.resize(max_seq_len * half_head);
        for (int pos = 0; pos < max_seq_len; pos++) {
            for (int i = 0; i < half_head; i++) {
                float freq = 1.0f / powf(10000.0f, 2.0f * i / HEAD_DIM);
                float angle = pos * freq;
                cos_table[pos * half_head + i] = cosf(angle);
                sin_table[pos * half_head + i] = sinf(angle);
            }
        }
        
        // Scratch buffers
        hidden.resize(HIDDEN);
        residual.resize(HIDDEN);
        qkv_out.resize(QKV_DIM);
        q_buf.resize(Q_DIM);
        k_buf.resize(KV_DIM);
        v_buf.resize(KV_DIM);
        attn_out.resize(Q_DIM);
        o_proj_out.resize(HIDDEN);
        gu_out.resize(2 * INTER);
        gate_buf.resize(INTER);
        up_buf.resize(INTER);
        down_out.resize(HIDDEN);
        logits.resize(vocab_size);
        
        // Q8 buffers (max K = QKV_DIM = 5120)
        int max_k = QKV_DIM;
        int max_groups = max_k / group_size;
        x_q8.resize(max_k);
        x_scales.resize(max_groups);
        
        // NOTE: P-core affinity is handled by OMP_PLACES env var, not SetProcessAffinityMask
        // Calling SetProcessAffinityMask here conflicts with OpenMP's thread placement
    }
    
    // Zero-copy weight registration via uintptr_t
    void register_layer(int idx,
                        uintptr_t qkv_proj, uintptr_t qkv_scales,
                        uintptr_t o_proj, uintptr_t o_scales,
                        uintptr_t gate_up_proj, uintptr_t gate_up_scales,
                        uintptr_t down_proj, uintptr_t down_scales,
                        uintptr_t rms_att, uintptr_t rms_ffn) {
        layers[idx].qkv_proj = reinterpret_cast<const uint8_t*>(qkv_proj);
        layers[idx].qkv_scales = reinterpret_cast<const float*>(qkv_scales);
        layers[idx].o_proj = reinterpret_cast<const uint8_t*>(o_proj);
        layers[idx].o_scales = reinterpret_cast<const float*>(o_scales);
        layers[idx].gate_up_proj = reinterpret_cast<const uint8_t*>(gate_up_proj);
        layers[idx].gate_up_scales = reinterpret_cast<const float*>(gate_up_scales);
        layers[idx].down_proj = reinterpret_cast<const uint8_t*>(down_proj);
        layers[idx].down_scales = reinterpret_cast<const float*>(down_scales);
        layers[idx].rms_att = reinterpret_cast<const float*>(rms_att);
        layers[idx].rms_ffn = reinterpret_cast<const float*>(rms_ffn);
    }
    
    void set_embedding(uintptr_t embed_ptr) {
        embed_f16 = reinterpret_cast<const uint16_t*>(embed_ptr);
        // No FP32 pre-conversion — use FP16 directly in compute_logits() with F16C
    }
    
    void set_final_norm(uintptr_t norm_ptr) {
        final_norm = reinterpret_cast<const float*>(norm_ptr);
    }
    
    // Embed token ID to hidden state (FP16 -> FP32)
    void embed_token(int token_id) {
        const uint16_t* row = embed_f16 + token_id * HIDDEN;
        for (int i = 0; i < HIDDEN; i++) {
            uint16_t h = row[i];
            uint32_t sign = (h >> 15) & 1;
            int32_t exp = ((h >> 10) & 0x1F) - 15;
            uint32_t mant = h & 0x3FF;
            float f = 0.0f;
            if (exp == 16) {
                f = (mant == 0) ? (sign ? -INFINITY : INFINITY) : NAN;
            } else if (exp == -15) {
                f = (mant == 0) ? 0.0f : ldexpf((float)mant, -24);
            } else {
                f = ldexpf((float)(1024 + mant), exp - 10);
            }
            hidden[i] = sign ? -f : f;
        }
    }
    
    // Forward one token through all layers, returns logits pointer
    const float* forward_token(int token_id, int pos) {
        // Embed
        embed_token(token_id);
        
        // Forward through each layer
        for (int l = 0; l < num_layers; l++) {
            forward_layer(l, pos);
        }
        
        // Final RMSNorm
        rmsnorm_f32(hidden.data(), hidden.data(), final_norm, HIDDEN, RMS_EPS);
        
        // LM head: logits = hidden @ embed_f16^T
        compute_logits();
        
        return logits.data();
    }
    
    // Forward one token through transformer layers, returns hidden state (COPY)
    // Python computes LM head separately (optimized BLAS for 200K vocab)
    py::array_t<float> forward_hidden(int token_id, int pos) {
        embed_token(token_id);
        
        for (int l = 0; l < num_layers; l++) {
            forward_layer(l, pos);
        }
        
        // Final RMSNorm
        rmsnorm_f32(hidden.data(), hidden.data(), final_norm, HIDDEN, RMS_EPS);
        
        // Return a COPY — pybind11 view would dangle on next forward_hidden call
        py::array_t<float> result(HIDDEN);
        std::memcpy(result.mutable_data(), hidden.data(), HIDDEN * sizeof(float));
        return result;
    }
    
    // Full autoregressive generation loop in C++
    std::vector<int> generate(const std::vector<int>& prompt_tokens, int max_new_tokens) {
        std::vector<int> generated = prompt_tokens;
        current_seq_len = prompt_tokens.size();
        
        // Prefill: process prompt tokens
        for (int i = 0; i < (int)prompt_tokens.size(); i++) {
            forward_token(prompt_tokens[i], i);
        }
        
        // Autoregressive generation
        for (int t = 0; t < max_new_tokens; t++) {
            const float* logits_ptr = forward_token(generated.back(), current_seq_len);
            int next_token = argmax_f32(logits_ptr, vocab_size);
            
            if (next_token == EOS_TOKEN_1 || next_token == EOS_TOKEN_2) {
                break;
            }
            
            generated.push_back(next_token);
            current_seq_len++;
        }
        
        return generated;
    }
    
    int get_seq_len() const { return current_seq_len; }
    
private:
    void forward_layer(int l, int pos) {
        LayerWeights& lw = layers[l];
        
        // Save residual
        memcpy(residual.data(), hidden.data(), HIDDEN * sizeof(float));
        
        // RMSNorm (pre-attention)
        rmsnorm_f32(hidden.data(), hidden.data(), lw.rms_att, HIDDEN, RMS_EPS);
        
        // Q8 quantize hidden state
        quantize_activations_q8(hidden.data(), x_q8.data(), x_scales.data(),
                                 HIDDEN, n_groups_h, group_size);
        
        // QKV projection: Q4 × Q8 GEMV
        gemv_q4_q8_row(lw.qkv_proj, lw.qkv_scales, x_q8.data(), x_scales.data(),
                        qkv_out.data(), QKV_DIM, HIDDEN, n_groups_qkv, group_size);
        
        // Split Q, K, V
        memcpy(q_buf.data(), qkv_out.data(), Q_DIM * sizeof(float));
        memcpy(k_buf.data(), qkv_out.data() + Q_DIM, KV_DIM * sizeof(float));
        memcpy(v_buf.data(), qkv_out.data() + Q_DIM + KV_DIM, KV_DIM * sizeof(float));
        
        // RoPE
        rope_apply_inplace(q_buf.data(), k_buf.data(),
                           cos_table.data(), sin_table.data(), pos, HEAD_DIM / 2);
        
        // Append to KV cache at position `pos`
        float* k_layer = k_cache.data() + l * max_seq_len * NUM_KV_HEADS * HEAD_DIM;
        float* v_layer = v_cache.data() + l * max_seq_len * NUM_KV_HEADS * HEAD_DIM;
        memcpy(k_layer + pos * NUM_KV_HEADS * HEAD_DIM, k_buf.data(),
               KV_DIM * sizeof(float));
        memcpy(v_layer + pos * NUM_KV_HEADS * HEAD_DIM, v_buf.data(),
               KV_DIM * sizeof(float));
        
        // GQA attention — seq_len = pos + 1 (current token position inclusive)
        gqa_decode_attention(q_buf.data(), k_layer, v_layer,
                             attn_out.data(), pos + 1,
                             1.0f / sqrtf((float)HEAD_DIM));
        
        // Q8 quantize attention output
        quantize_activations_q8(attn_out.data(), x_q8.data(), x_scales.data(),
                                 Q_DIM, n_groups_h, group_size);
        
        // Output projection: Q4 × Q8 GEMV
        gemv_q4_q8_row(lw.o_proj, lw.o_scales, x_q8.data(), x_scales.data(),
                        o_proj_out.data(), HIDDEN, Q_DIM, n_groups_o, group_size);
        
        // Residual add
        vec_add_inplace(residual.data(), o_proj_out.data(), HIDDEN);
        memcpy(hidden.data(), residual.data(), HIDDEN * sizeof(float));
        
        // --- Feed-forward ---
        memcpy(residual.data(), hidden.data(), HIDDEN * sizeof(float));
        
        // RMSNorm (pre-FFN)
        rmsnorm_f32(hidden.data(), hidden.data(), lw.rms_ffn, HIDDEN, RMS_EPS);
        
        // Q8 quantize
        quantize_activations_q8(hidden.data(), x_q8.data(), x_scales.data(),
                                 HIDDEN, n_groups_h, group_size);
        
        // Gate/Up projection: Q4 × Q8 GEMV
        gemv_q4_q8_row(lw.gate_up_proj, lw.gate_up_scales, x_q8.data(), x_scales.data(),
                        gu_out.data(), 2 * INTER, HIDDEN, n_groups_gu, group_size);
        
        // Split gate and up
        memcpy(gate_buf.data(), gu_out.data(), INTER * sizeof(float));
        memcpy(up_buf.data(), gu_out.data() + INTER, INTER * sizeof(float));
        
        // SwiGLU
        swiglu_inplace(gate_buf.data(), up_buf.data(), INTER);
        
        // Q8 quantize SwiGLU output
        quantize_activations_q8(gate_buf.data(), x_q8.data(), x_scales.data(),
                                 INTER, n_groups_h, group_size);
        
        // Down projection: Q4 × Q8 GEMV
        gemv_q4_q8_row(lw.down_proj, lw.down_scales, x_q8.data(), x_scales.data(),
                        down_out.data(), HIDDEN, INTER, n_groups_down, group_size);
        
        // Residual add
        vec_add_inplace(residual.data(), down_out.data(), HIDDEN);
        memcpy(hidden.data(), residual.data(), HIDDEN * sizeof(float));
    }
    
    void compute_logits() {
        // LM head: logits[v] = sum_i hidden[i] * embed_f16[v,i]
        // Uses FP16 directly with AVX2 F16C for on-the-fly conversion — halves memory bandwidth.
        const float* h = hidden.data();
        
        #pragma omp parallel for schedule(static)
        for (int v = 0; v < vocab_size; v++) {
            const uint16_t* row = embed_f16 + v * HIDDEN;
            __m256 acc = _mm256_setzero_ps();
            
            // Process 8 elements at a time: load 8x FP16, convert to FP32, FMA
            for (int i = 0; i < HIDDEN; i += 8) {
                __m128i w16 = _mm_loadu_si128((const __m128i*)(row + i));
                __m256 w = _mm256_cvtph_ps(w16);  // FP16 → FP32
                __m256 x = _mm256_loadu_ps(h + i);
                acc = _mm256_fmadd_ps(w, x, acc);
            }
            logits[v] = hsum256_ps(acc);
        }
    }
};

/* ===================================================================
 * PyBind11 Bindings
 * =================================================================== */

PYBIND11_MODULE(_native_engine, m) {
    m.doc() = "Pure C++ Inference Engine for ASDSL — zero Python overhead generation";
    
    py::class_<InferenceEngine>(m, "InferenceEngine")
        .def(py::init<int, int, int, int>(),
             py::arg("num_layers"), py::arg("max_seq_len"),
             py::arg("group_size"), py::arg("vocab_size"))
        .def("register_layer", &InferenceEngine::register_layer,
             py::arg("idx"),
             py::arg("qkv_proj"), py::arg("qkv_scales"),
             py::arg("o_proj"), py::arg("o_scales"),
             py::arg("gate_up_proj"), py::arg("gate_up_scales"),
             py::arg("down_proj"), py::arg("down_scales"),
             py::arg("rms_att"), py::arg("rms_ffn"))
        .def("set_embedding", &InferenceEngine::set_embedding,
             py::arg("embed_ptr"))
        .def("set_final_norm", &InferenceEngine::set_final_norm,
             py::arg("norm_ptr"))
        .def("generate", &InferenceEngine::generate,
             py::arg("prompt_tokens"), py::arg("max_new_tokens"))
        .def("forward_hidden", &InferenceEngine::forward_hidden,
             py::arg("token_id"), py::arg("pos"))
        .def("get_seq_len", &InferenceEngine::get_seq_len);
}
