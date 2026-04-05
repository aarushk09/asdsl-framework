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
    int group_size;
    int max_seq_len;
    int rotary_dim;
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
void gemm_q4_32_q8_avx2(const uint8_t* blocks, const float* x, float* y, int out_features, int in_features, int group_size, int batch_size);

namespace asdsl {
class UnifiedEngine {
public:
    UnifiedEngine(const EngineConfig& config, const EngineWeights& weights);
    ~UnifiedEngine() = default;

    std::vector<int32_t> generate(const std::vector<int32_t>& prompt, int max_tokens);
    std::vector<int32_t> generate_swift(const std::vector<int32_t>& prompt, int max_tokens, int draft_k);


private:
    // Batched forward pass for prompt processing
    void forward_batch(const int32_t* tokens, int num_tokens, int start_pos, float* out_logits, bool all_logits = false);
    
    // Original single token forward pass
    void forward_token(int token_id, int pos, float* out_logits);
    void forward_token_draft(int token_id, int pos, float* out_logits);

    EngineConfig config_;
    EngineWeights weights_;

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
    
    size_t kv_base(int layer, int pos, int head) const;
    size_t scale_base(int layer, int pos, int head) const;

    void rmsnorm_f32(float* out, const float* in, const float* weight, int size, float eps);
    void rope_apply_inplace(float* q, float* k, const float* cos_table, const float* sin_table, int pos, int rotary_dim);
    void swiglu_inplace(float* gate, float* up, int size);
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
