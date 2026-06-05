#pragma once

#include <vector>
#include <cstdint>
#include <unordered_map>
#include "thread_pool.h"

namespace asdsl {

ThreadPool& get_global_thread_pool();

struct LayerWeights {
    const float* rms_att;
    const uint8_t* qkv_proj;
    

    const uint8_t* o_proj;
    
    const float* rms_ffn;

    const uint8_t* gate_up_proj;


    const uint8_t* down_proj;

    float fatrelu_threshold = 0.0f;
    /** Per-projection: use gemv_q4km_q8_avx2 (GGUF Q4_K blocks) instead of preq. */
    bool qkv_q4km = false;
    bool o_q4km = false;
    bool gate_up_q4km = false;
    bool down_q4km = false;
    /** Reserved: native Q5_K / Q6_K (use preq fallback; Phase 30 reverted). */
    bool qkv_q5km = false;
    bool down_q6km = false;
};

struct EngineConfig {
    int num_layers;
    int hidden_size;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int intermediate_size;
    int vocab_size;
    float rms_norm_eps;
    int group_size;       // activation quantization group size (32 or 128)
    int max_seq_len;
    int rotary_dim;
    // weight_format: 0 = Q4_32 preq (20-byte blocks),
    //                1 = Q4_S256 preq,
    //                2 = raw GGUF Q4_K (144-byte blocks, gemv_q4km_q8_avx2)
    int weight_format = 0;
};

struct EngineWeights {
    const float* token_embd;
    const float* output_norm;
    const float* output_proj; // Now expects float*
    
    const float* cos_table;
    const float* sin_table;

    std::unordered_map<int, LayerWeights> layers;
};

} // namespace asdsl 

void gemv_q4_32_q8_avx2(const uint8_t* blocks, const float* x, float* y, int out_features, int in_features, int group_size);
void gemv_q4_32_preq_avx2(const uint8_t* blocks, const int8_t* x_q8, const float* x_scales, float* y, int out_features, int in_features, int group_size);
void gemv_q4_32_preq_fused_avx2(const uint8_t* blocks, const float* x_fp32, float* y, int out_features, int in_features, int group_size);
void gemv_q4_32_preq_4row_avx2(const uint8_t* blocks, const float* x_fp32, float* y, int out_features, int in_features, int group_size);
void gemv_q4_32_preq_8row_avx2(const uint8_t* blocks, const float* x_fp32, float* y, int out_features, int in_features, int group_size);
void quantize_activation_avx2(const float* x, int8_t* x_q8, float* x_scales, int in_features, int group_size);
void gemm_q4_32_q8_avx2(const uint8_t* blocks, const float* x, float* y, int out_features, int in_features, int group_size, int batch_size);

// Q4_128: group_size=128 format (66-byte blocks: 2B fp16 scale + 64B nibbles)
void gemv_q4_128_preq_avx2(const uint8_t* blocks, const int8_t* x_q8, const float* x_scales, float* y, int out_features, int in_features, int group_size);
void gemm_q4_128_q8_avx2(const uint8_t* blocks, const float* x_batch, float* y_batch, int out_features, int in_features, int group_size, int batch_size);

// Q4_S256: 256-weight superblocks with 8x FP16 sub-scales at 32-weight granularity
// Same 4.5 bpw as Q4_K_M; activation group_size must be 32.
void gemv_q4_s256_preq_avx2(const uint8_t* blocks, const int8_t* x_q8, const float* x_scales, float* y, int out_features, int in_features, int group_size);
void gemv_q4km_q8_avx2(const uint8_t* weights_q4km, const float* x, float* y, int out_features, int in_features);

namespace asdsl {
class UnifiedEngine {
public:
    UnifiedEngine(const EngineConfig& config, const EngineWeights& weights);
    ~UnifiedEngine() = default;

    std::vector<int32_t> generate(const std::vector<int32_t>& prompt, int max_tokens);
    std::vector<int32_t> generate_swift(const std::vector<int32_t>& prompt, int max_tokens, int draft_k);


public:
    // Batched forward pass for prompt processing
    void forward_batch(const int32_t* tokens, int num_tokens, int start_pos, float* out_logits, bool all_logits = false);
    
    // Original single token forward pass
    void forward_token(int token_id, int pos, float* out_logits);
    void forward_token_draft(int token_id, int pos, float* out_logits);

    // Run all transformer layers then project with a caller-supplied FP32 LM
    // head weight matrix instead of the internal Q4-quantised one.  Used by
    // check_q4_lm_precision.py to obtain a FP32 reference without a second
    // engine instance.  lm_head_fp32 must be [vocab_size, hidden_size] row-major.
    void forward_token_fp32_lmhead(int token_id, int pos,
                                   const float* lm_head_fp32, float* out_logits);

    /** Clear KV cache between generation sessions (reuse engine instance). */
    void reset_session();

    EngineConfig config_;
    EngineWeights weights_;

private:

    // Buffer state
    std::vector<float> hidden_;
    std::vector<float> residual_;
    std::vector<float> qkv_out_;
    
    std::vector<float> q_buf_;
    std::vector<float> k_buf_;
    std::vector<float> v_buf_;
    
    std::vector<float> att_out_;
    std::vector<float> gate_buf_;
    std::vector<float> up_buf_;
    std::vector<float> gu_out_;
    std::vector<float> down_out_;
    std::vector<float> logits_;

    // KV Cache natively allocated
    static constexpr int KV_QBLOCK = 64;
    std::vector<int8_t> k_cache_q8_;
    std::vector<float>  k_cache_scales_;
    std::vector<int8_t> v_cache_q8_;
    std::vector<float>  v_cache_scales_;
    int blocks_per_head_;
    
    // Pre-allocated attention scratch (avoids per-call heap allocation)
    // scores_buf_: [num_heads * max_seq_len] — head h uses [h*max_seq_len..]
    // num_buf_: [num_heads * head_dim] — head h uses [h*head_dim..]
    // tile_num_buf_: [num_heads * head_dim] — head h uses [h*head_dim..]
    std::vector<float>  scores_buf_;
    std::vector<float>  num_buf_;
    std::vector<float>  tile_num_buf_;

    std::vector<int8_t> hidden_q8_;
    std::vector<float>  hidden_scales_;
    std::vector<int8_t> gate_q8_;
    std::vector<float>  gate_scales_;

    // LM head stored as 18-byte Q4 blocks (2B fp16 scale + 16B packed nibbles per 32 elements).
    // Halves memory traffic vs Q8: 200064 * 3072 * 0.5625B ≈ 345 MB vs 663 MB.
    std::vector<uint8_t> lm_head_q4_blocks_;
    bool lm_head_q4_quantized_ = false;

    // Zero-worker pool: parallelism is handled by #pragma omp parallel for.
    // Keeping the member avoids larger refactors; zero workers means no competing
    // spin-wait threads consuming P-core execution resources alongside OpenMP workers.
    ThreadPool pool_{0};

    // forward_batch scratch (avoids per-call heap for large prompts)
    std::vector<float> b_hidden_;
    std::vector<float> b_residual_;
    std::vector<float> b_qkv_out_;
    std::vector<float> b_att_out_;
    std::vector<float> b_gu_out_;
    std::vector<float> b_gate_batch_;

    size_t kv_base(int layer, int pos, int head) const;
    size_t scale_base(int layer, int pos, int head) const;

    void rmsnorm_f32(float* out, const float* in, const float* weight, int size, float eps);
    void rmsnorm_quantize_f32(const float* in, const float* weight, int8_t* out_q8, float* out_scales, int size, int group_size, float eps);
    void residual_add_rmsnorm_quantize_f32(float* hidden, float* residual, const float* rms_weight, int8_t* out_q8, float* out_scales, int size, int group_size, float eps);
    void residual_add_rmsnorm_f32(float* hidden, float* residual, const float* rms_weight, int size, float eps);
    void swiglu_quantize_inplace(float* gate, float* up, int8_t* out_q8, float* out_scales, int size, int group_size, float threshold);
    void rope_apply_inplace(float* q, float* k, const float* cos_table, const float* sin_table, int pos, int rotary_dim);
    void swiglu_inplace(float* gate, float* up, int size, float threshold = 0.0f);
    void compute_attention_flash_q8(float* out, const float* q, int layer_id, int seq_pos);
    void set_kv_cache(int layer, int pos, const float* k, const float* v);
    void vec_add_inplace(float* dest, const float* src, int size);
};

} // namespace asdsl

void gemm_q4_q8_avx2(
    const uint8_t* weights_packed,
    const float*   scales,
    const float*   x,
    float*         y,
    int            out_features,
    int            in_features,
    int            group_size,
    int            batch_size
);

void gemv_q4_q8_avx2(
    const uint8_t* weights_packed,
    const float*   scales,
    const float*   x,
    float*         y,
    int            out_features,
    int            in_features,
    int            group_size
);
