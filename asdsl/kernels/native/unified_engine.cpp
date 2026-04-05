#include "unified_engine.h"
#include "gemv_q4_kernel.h"
#include <immintrin.h>
#include <cmath>
#include <cstring>
#include <limits>
#include <algorithm>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace asdsl {

// We must avoid OpenMP and use the thread_pool.
// To do this, we need to access a global/singleton ThreadPool.
ThreadPool& get_global_thread_pool() {
    return ThreadPool::get_instance();
}

static inline float hsum256_ps(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

UnifiedEngine::UnifiedEngine(const EngineConfig& config, const EngineWeights& weights)
    : config_(config), weights_(weights) {
    
    hidden_.resize(config_.hidden_size, 0.0f);
    residual_.resize(config_.hidden_size, 0.0f);
    
    int q_dim = config_.num_heads * config_.head_dim;
    int kv_dim = config_.num_kv_heads * config_.head_dim;
    qkv_out_.resize(q_dim + 2 * kv_dim, 0.0f);
    
    q_buf_.resize(q_dim, 0.0f);
    k_buf_.resize(kv_dim, 0.0f);
    v_buf_.resize(kv_dim, 0.0f);
    
    att_out_.resize(q_dim, 0.0f);
    
    gu_out_.resize(2 * config_.intermediate_size, 0.0f);
    gate_buf_.resize(config_.intermediate_size, 0.0f);
    up_buf_.resize(config_.intermediate_size, 0.0f);
    down_out_.resize(config_.hidden_size, 0.0f);
    
    logits_.resize(config_.vocab_size, 0.0f);
    
    blocks_per_head_ = config_.head_dim / KV_QBLOCK;
    size_t kv_size = static_cast<size_t>(config_.num_layers) * config_.max_seq_len * config_.num_kv_heads * config_.head_dim;
    size_t scale_size = static_cast<size_t>(config_.num_layers) * config_.max_seq_len * config_.num_kv_heads * blocks_per_head_;
    
    k_cache_q8_.resize(kv_size, 0);
    v_cache_q8_.resize(kv_size, 0);
    k_cache_scales_.resize(scale_size, 1.0f);
    v_cache_scales_.resize(scale_size, 1.0f);
}

size_t UnifiedEngine::kv_base(int layer, int pos, int head) const {
    return static_cast<size_t>(layer) * config_.max_seq_len * config_.num_kv_heads * config_.head_dim +
           static_cast<size_t>(pos) * config_.num_kv_heads * config_.head_dim +
           static_cast<size_t>(head) * config_.head_dim;
}

size_t UnifiedEngine::scale_base(int layer, int pos, int head) const {
    return static_cast<size_t>(layer) * config_.max_seq_len * config_.num_kv_heads * blocks_per_head_ +
           static_cast<size_t>(pos) * config_.num_kv_heads * blocks_per_head_ +
           static_cast<size_t>(head) * blocks_per_head_;
}

void UnifiedEngine::vec_add_inplace(float* dest, const float* src, int size) {
    int i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 d = _mm256_loadu_ps(dest + i);
        __m256 s = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dest + i, _mm256_add_ps(d, s));
    }
    for (; i < size; ++i) {
        dest[i] += src[i];
    }
}

void UnifiedEngine::rmsnorm_f32(float* out, const float* in, const float* weight, int size, float eps) {
    __m256 sum_v = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 x = _mm256_loadu_ps(in + i);
        sum_v = _mm256_fmadd_ps(x, x, sum_v);
    }
    float sum = hsum256_ps(sum_v);
    for (; i < size; ++i) {
        sum += in[i] * in[i];
    }
    
    float inv_rms = 1.0f / std::sqrt(sum / size + eps);
    __m256 inv_rms_v = _mm256_set1_ps(inv_rms);

    i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 x = _mm256_loadu_ps(in + i);
        __m256 w = _mm256_loadu_ps(weight + i);
        __m256 res = _mm256_mul_ps(_mm256_mul_ps(x, inv_rms_v), w);
        _mm256_storeu_ps(out + i, res);
    }
    for (; i < size; ++i) {
        out[i] = in[i] * inv_rms * weight[i];
    }
}

void UnifiedEngine::rope_apply_inplace(float* q, float* k, const float* cos_table, const float* sin_table, int pos, int rotary_dim) {
    const int half_rotary = rotary_dim / 2;
    // VERY IMPORTANT: The Python reference script cos_table has width `half_head = config.head_dim / 2`
    // but run_unified_bin provides `config.rotary_dim // 2`.
    // Wait, let's check `run_unified_bin.py` !!
    // In `run_unified_bin.py`:
    // dim = np.arange(config.rotary_dim//2, dtype=np.float32)
    // t = pos[:, None] * inv_freq[None, :]
    // So cos_table has width `half_rotary` (48)! Not 64!
    const float* cos_p = cos_table + pos * half_rotary;
    const float* sin_p = sin_table + pos * half_rotary;
    // RoPE only applies to the first rotary_dim elements.
    // So the two halves of the rotated portion are at offset 0 and offset half_rotary.

    for (int h = 0; h < config_.num_heads; ++h) {
        float* qh = q + h * config_.head_dim;
        for (int i = 0; i < half_rotary; ++i) {
            float x_first = qh[i];
            float x_second = qh[half_rotary + i];
            float c = cos_p[i];
            float s = sin_p[i];
            qh[i]                 = x_first * c - x_second * s;
            qh[half_rotary + i]   = x_first * s + x_second * c;
        }
    }
    for (int h = 0; h < config_.num_kv_heads; ++h) {
        float* kh = k + h * config_.head_dim;
        for (int i = 0; i < half_rotary; ++i) {
            float x_first = kh[i];
            float x_second = kh[half_rotary + i];
            float c = cos_p[i];
            float s = sin_p[i];
            kh[i]                 = x_first * c - x_second * s;
            kh[half_rotary + i]   = x_first * s + x_second * c;
        }
    }
}

void UnifiedEngine::swiglu_inplace(float* gate, float* up, int size, float threshold) {
    int i = 0;
    for (; i < size; ++i) {
        float x = gate[i];
        float sig = 1.0f / (1.0f + std::exp(-x));
        float val = x * sig * up[i];
        
        if (threshold > 0.0f && std::abs(val) < threshold) {
            val = 0.0f;
        }
        
        gate[i] = val;
    }
}

void UnifiedEngine::set_kv_cache(int layer, int pos, const float* k, const float* v) {
    for (int h = 0; h < config_.num_kv_heads; ++h) {
        const float* kh = k + h * config_.head_dim;
        const float* vh = v + h * config_.head_dim;
        size_t kb = kv_base(layer, pos, h);
        size_t sb = scale_base(layer, pos, h);

        for (int b = 0; b < blocks_per_head_; ++b) {
            const int off = b * KV_QBLOCK;
            const int len = std::min(KV_QBLOCK, config_.head_dim - off);
            float k_absmax = 0.0f;
            float v_absmax = 0.0f;
            for (int i = 0; i < len; ++i) {
                k_absmax = std::max(k_absmax, std::fabs(kh[off + i]));
                v_absmax = std::max(v_absmax, std::fabs(vh[off + i]));
            }
            float ks = (k_absmax > 1e-12f) ? (k_absmax / 127.0f) : 1e-12f;
            float vs = (v_absmax > 1e-12f) ? (v_absmax / 127.0f) : 1e-12f;
            k_cache_scales_[sb + b] = ks;
            v_cache_scales_[sb + b] = vs;

            float kinv = 1.0f / ks;
            float vinv = 1.0f / vs;
            for (int i = 0; i < len; ++i) {
                int qk = static_cast<int>(std::nearbyint(kh[off + i] * kinv));
                int qv = static_cast<int>(std::nearbyint(vh[off + i] * vinv));
                qk = std::max(-127, std::min(127, qk));
                qv = std::max(-127, std::min(127, qv));
                k_cache_q8_[kb + off + i] = static_cast<int8_t>(qk);
                v_cache_q8_[kb + off + i] = static_cast<int8_t>(qv);
            }
        }
    }
}

void UnifiedEngine::compute_attention_flash_q8(float* out, const float* q, int layer_id, int seq_pos) {
    const int groups = std::max(1, config_.num_heads / config_.num_kv_heads);
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(config_.head_dim));
    const int BLOCK_K = 32;

    auto& pool = get_global_thread_pool();
    pool.parallel_for(0, config_.num_heads, 1, [&](int h) {
        const int kv_h = h / groups;
        const float* qh = q + h * config_.head_dim;
        std::vector<float> num(config_.head_dim, 0.0f);
        float m = -std::numeric_limits<float>::infinity();
        float l = 0.0f;

        for (int tk = 0; tk <= seq_pos; tk += BLOCK_K) {
            const int kend = std::min(seq_pos + 1, tk + BLOCK_K);
            const int span = kend - tk;
            std::vector<float> scores(span);
            float tile_max = -std::numeric_limits<float>::infinity();

            for (int p = 0; p < span; ++p) {
                const int pos = tk + p;
                const size_t kb = kv_base(layer_id, pos, kv_h);
                const size_t sb = scale_base(layer_id, pos, kv_h);
                float dot = 0.0f;
                for (int b = 0; b < blocks_per_head_; ++b) {
                    const int off = b * KV_QBLOCK;
                    const int len = std::min(KV_QBLOCK, config_.head_dim - off);
                    const float ks = k_cache_scales_[sb + b];
                    const int8_t* kq = k_cache_q8_.data() + kb + off;
                    for (int i = 0; i < len; ++i) {
                        dot += qh[off + i] * (static_cast<float>(kq[i]) * ks);
                    }
                }
                const float s = dot * inv_sqrt_d;
                scores[p] = s;
                tile_max = std::max(tile_max, s);
            }

            const float new_m = std::max(m, tile_max);
            const float old_scale = (l > 0.0f) ? std::exp(m - new_m) : 0.0f;
            float tile_l = 0.0f;
            std::vector<float> tile_num(config_.head_dim, 0.0f);

            for (int p = 0; p < span; ++p) {
                const int pos = tk + p;
                const float w = std::exp(scores[p] - new_m);
                tile_l += w;
                const size_t vb = kv_base(layer_id, pos, kv_h);
                const size_t sb = scale_base(layer_id, pos, kv_h);
                for (int b = 0; b < blocks_per_head_; ++b) {
                    const int off = b * KV_QBLOCK;
                    const int len = std::min(KV_QBLOCK, config_.head_dim - off);
                    const float vs = v_cache_scales_[sb + b];
                    const int8_t* vq = v_cache_q8_.data() + vb + off;
                    for (int i = 0; i < len; ++i) {
                        tile_num[off + i] += w * (static_cast<float>(vq[i]) * vs);
                    }
                }
            }

            m = new_m;
            l = l * old_scale + tile_l;
            for (int i = 0; i < config_.head_dim; ++i) {
                num[i] = num[i] * old_scale + tile_num[i];
            }
        }

        const float inv_l = 1.0f / l;
        float* out_h = out + h * config_.head_dim;
        for (int i = 0; i < config_.head_dim; ++i) {
            out_h[i] = num[i] * inv_l;
        }
    });
}


static void gemv_f32_f32_omp(const float* W, const float* x, float* y, int rows, int cols) {
    auto& pool = get_global_thread_pool();
    pool.parallel_for(0, rows, 64, [&](int i) {
        float sum = 0.0f;
        const float* w_row = W + i * cols;
        for (int j = 0; j < cols; ++j) {
            sum += w_row[j] * x[j];
        }
        y[i] = sum;
    });
}


void UnifiedEngine::forward_batch(const int32_t* tokens, int num_tokens, int start_pos, float* out_logits, bool all_logits) {
    std::cout << "Starting forward_batch max_seq_len: " << config_.max_seq_len << std::endl;
    auto& pool = get_global_thread_pool();

    // Allocate dynamic batch buffers
    std::vector<float> b_hidden(num_tokens * config_.hidden_size);
    std::vector<float> b_residual(num_tokens * config_.hidden_size);
    
    int q_dim = config_.num_heads * config_.head_dim;
    int kv_dim = config_.num_kv_heads * config_.head_dim;
    int qkv_total = q_dim + 2 * kv_dim;
    std::vector<float> b_qkv_out(num_tokens * qkv_total);
    std::vector<float> b_att_out(num_tokens * config_.hidden_size);
    std::vector<float> b_gu_out(num_tokens * 2 * config_.intermediate_size);
    std::vector<float> b_swiglu_out(num_tokens * config_.intermediate_size);
        


    // 1. Token Embeddings
    pool.parallel_for(0, num_tokens, 1, [&](int i) {
        const float* emb_row = weights_.token_embd + tokens[i] * config_.hidden_size;
        std::memcpy(b_hidden.data() + i * config_.hidden_size, emb_row, config_.hidden_size * sizeof(float));
    });

    for (int l = 0; l < config_.num_layers; ++l) {  
        const LayerWeights& lw = weights_.layers.at(l);

                // Pre-attention RN
        std::memcpy(b_residual.data(), b_hidden.data(), num_tokens * config_.hidden_size * sizeof(float));
         
pool.parallel_for(0, num_tokens, 1, [&](int i) {
            rmsnorm_f32(b_hidden.data() + i * config_.hidden_size, b_hidden.data() + i * config_.hidden_size, lw.rms_att, config_.hidden_size, config_.rms_norm_eps);
        });

        // QKV Projection Batch
         
        gemm_q4_32_q8_avx2(lw.qkv_proj, b_hidden.data(), b_qkv_out.data(),
                        qkv_total, config_.hidden_size, config_.group_size, num_tokens);



         
        // Attention processing per-token
        // We will process attention token-by-token (causal mask effectively applied by updating cache and attending)
        for (int i = 0; i < num_tokens; ++i) {
            int pos = start_pos + i;
            float* qkv_row = b_qkv_out.data() + i * qkv_total;
            float* q_p = qkv_row;
            float* k_p = qkv_row + q_dim;
            float* v_p = qkv_row + q_dim + kv_dim;

            // RoPE
            rope_apply_inplace(q_p, k_p, weights_.cos_table, weights_.sin_table, pos, config_.rotary_dim);

            // Cache update
            set_kv_cache(l, pos, k_p, v_p);

            // Attention
            compute_attention_flash_q8(b_att_out.data() + i * config_.hidden_size, q_p, l, pos);
        }

        // O proj Batch
        gemm_q4_32_q8_avx2(lw.o_proj, b_att_out.data(), b_hidden.data(),
                        config_.hidden_size, q_dim, config_.group_size, num_tokens);



        // Residual

        pool.parallel_for(0, num_tokens, 1, [&](int i) {
            vec_add_inplace(b_hidden.data() + i * config_.hidden_size,
                            b_residual.data() + i * config_.hidden_size, config_.hidden_size);
        });

        // Pre-FFN RN
        std::memcpy(b_residual.data(), b_hidden.data(), num_tokens * config_.hidden_size * sizeof(float));

        pool.parallel_for(0, num_tokens, 1, [&](int i) {  
            rmsnorm_f32(b_hidden.data() + i * config_.hidden_size, b_hidden.data() + i * config_.hidden_size, lw.rms_ffn, config_.hidden_size, config_.rms_norm_eps);
        });

        // Gate/Up proj Batch
        // Gate/Up proj Batch

        gemm_q4_32_q8_avx2(lw.gate_up_proj, b_hidden.data(), b_gu_out.data(),
                        2 * config_.intermediate_size, config_.hidden_size, config_.group_size, num_tokens);        

        // SwiGLU Batch

        pool.parallel_for(0, num_tokens, 1, [&](int i) {
            float* gu_row = b_gu_out.data() + i * 2 * config_.intermediate_size;
            swiglu_inplace(gu_row, gu_row + config_.intermediate_size, config_.intermediate_size, lw.fatrelu_threshold);
        });

        // Down proj Batch(we can reuse b_gu_out's first half as input)
        // b_gu_out is (num_tokens x 2*intermediate), gate is first half.
        // Wait, gemm_q4_q8_avx2 expects input vectors to be contiguous. 
        // We need to pack the SwiGLU output into a shape of (num_tokens x intermediate_size).
        std::vector<float> b_gate(num_tokens * config_.intermediate_size);
         
        pool.parallel_for(0, num_tokens, 1, [&](int i) {
            std::memcpy(b_gate.data() + i * config_.intermediate_size, 
                       b_gu_out.data() + i * 2 * config_.intermediate_size, 
                       config_.intermediate_size * sizeof(float));
        });

         
        gemm_q4_32_q8_avx2(lw.down_proj, b_gate.data(), b_hidden.data(),
                        config_.hidden_size, config_.intermediate_size, config_.group_size, num_tokens);

        // Residual
         
        pool.parallel_for(0, num_tokens, 1, [&](int i) {
            vec_add_inplace(b_hidden.data() + i * config_.hidden_size, 
                            b_residual.data() + i * config_.hidden_size, config_.hidden_size);
        });
    }

    if (out_logits) {
        if (all_logits) {
            for (int i = 0; i < num_tokens; ++i) {
                float* cur_hidden = b_hidden.data() + i * config_.hidden_size;
                float* cur_logits = out_logits + i * config_.vocab_size;
                rmsnorm_f32(cur_hidden, cur_hidden, weights_.output_norm, config_.hidden_size, config_.rms_norm_eps);
                gemv_f32_f32_omp(weights_.output_proj, cur_hidden, cur_logits, config_.vocab_size, config_.hidden_size);
            }
        } else {
            // Output from the LAST token
            float* last_hidden = b_hidden.data() + (num_tokens - 1) * config_.hidden_size;
            rmsnorm_f32(last_hidden, last_hidden, weights_.output_norm, config_.hidden_size, config_.rms_norm_eps);
            gemv_f32_f32_omp(weights_.output_proj, last_hidden, out_logits, config_.vocab_size, config_.hidden_size);
        }
    }
}

void UnifiedEngine::forward_token(int token_id, int pos, float* out_logits) {
    auto& pool = get_global_thread_pool();
    const float* emb_row = weights_.token_embd + token_id * config_.hidden_size;
    std::memcpy(hidden_.data(), emb_row, config_.hidden_size * sizeof(float));
    // std::cout << "[DB] emb[0]=" << hidden_[0] << " ";
    // std::cout << "[DB] emb ";

    int q_dim = config_.num_heads * config_.head_dim;
    int kv_dim = config_.num_kv_heads * config_.head_dim;
    int qkv_total = q_dim + 2 * kv_dim;

    for (int l = 0; l < config_.num_layers; ++l) {  
        const LayerWeights& lw = weights_.layers.at(l);

        // Pre-attention RN
        std::memcpy(residual_.data(), hidden_.data(), config_.hidden_size * sizeof(float));
        rmsnorm_f32(hidden_.data(), hidden_.data(), lw.rms_att, config_.hidden_size, config_.rms_norm_eps);

        // QKV
        gemv_asb_avx2(lw.qkv_proj, hidden_.data(), qkv_out_.data(),
                        qkv_total, config_.hidden_size, config_.group_size);

        // Split
        // RoPE & Cache & Attention Fused
        rope_apply_inplace(qkv_out_.data(), qkv_out_.data() + q_dim, weights_.cos_table, weights_.sin_table, pos, config_.rotary_dim);
        set_kv_cache(l, pos, qkv_out_.data() + q_dim, qkv_out_.data() + q_dim + kv_dim);
        compute_attention_flash_q8(att_out_.data(), qkv_out_.data(), l, pos);

        // O proj
        gemv_asb_avx2(lw.o_proj, att_out_.data(), hidden_.data(),
                        config_.hidden_size, q_dim, config_.group_size);

        // Residual
        vec_add_inplace(hidden_.data(), residual_.data(), config_.hidden_size);

        // Pre-FFN RN
        std::memcpy(residual_.data(), hidden_.data(), config_.hidden_size * sizeof(float));
        rmsnorm_f32(hidden_.data(), hidden_.data(), lw.rms_ffn, config_.hidden_size, config_.rms_norm_eps);

        // Gate/Up
        gemv_asb_avx2(lw.gate_up_proj, hidden_.data(), gu_out_.data(),
                        2 * config_.intermediate_size, config_.hidden_size, config_.group_size);

        // SwiGLU
        swiglu_inplace(gu_out_.data(), gu_out_.data() + config_.intermediate_size, config_.intermediate_size, lw.fatrelu_threshold);

        // Down proj
        gemv_asb_avx2(lw.down_proj, gu_out_.data(), hidden_.data(),
                        config_.hidden_size, config_.intermediate_size, config_.group_size);

         
         
if (l == 23) {   }
        // Residual
        vec_add_inplace(hidden_.data(), residual_.data(), config_.hidden_size);
    }

    if (out_logits) {
        rmsnorm_f32(hidden_.data(), hidden_.data(), weights_.output_norm, config_.hidden_size, config_.rms_norm_eps);
        
        gemv_f32_f32_omp(weights_.output_proj, hidden_.data(), out_logits, config_.vocab_size, config_.hidden_size);
        // std::cout << "  [DB] Logit[0]=" << out_logits[0] << " Logit[100]=" << out_logits[100] << std::endl;
    }
}

std::vector<int32_t> UnifiedEngine::generate(const std::vector<int32_t>& prompt, int max_tokens) {
    std::cout << "Starting generate with " << prompt.size() << " tokens." << std::endl;
    std::vector<int32_t> output = prompt;
    int current_pos = 0;

    if (prompt.size() > 0) {
        std::cout << "Calling forward_batch..." << std::endl;
        forward_batch(prompt.data(), prompt.size(), 0, logits_.data());
        current_pos += prompt.size();
        std::cout << "Forward batch complete." << std::endl;
    }

    for (int t = 0; t < max_tokens; ++t) {

        int best_token = 0;
        float best_val = logits_[0];
        for (int i = 1; i < config_.vocab_size; ++i) {
            if (logits_[i] > best_val) {
                best_val = logits_[i];
                best_token = i;
            }
        }

        output.push_back(best_token);
        if (best_token == 199999 || best_token == 200020) { // Naive Phi-4 EOS
            break;
        }

        // std::cout << "Token " << best_token << "... ";
        forward_token(best_token, current_pos++, logits_.data());
        // std::cout << "forward_token complete. ";
    }
    // std::cout << std::endl;

    return output;
}


void UnifiedEngine::forward_token_draft(int token_id, int pos, float* out_logits) {
    auto& pool = get_global_thread_pool();
    const float* emb_row = weights_.token_embd + token_id * config_.hidden_size;
    std::memcpy(hidden_.data(), emb_row, config_.hidden_size * sizeof(float));

    int q_dim = config_.num_heads * config_.head_dim;
    int kv_dim = config_.num_kv_heads * config_.head_dim;
    int qkv_total = q_dim + 2 * kv_dim;

    for (int l = 0; l < config_.num_layers; ++l) {
        if (l >= 4 && l <= 27) continue;

        const LayerWeights& lw = weights_.layers.at(l);

        std::memcpy(residual_.data(), hidden_.data(), config_.hidden_size * sizeof(float));
        rmsnorm_f32(hidden_.data(), hidden_.data(), lw.rms_att, config_.hidden_size, config_.rms_norm_eps);

        gemv_asb_avx2(lw.qkv_proj, hidden_.data(), qkv_out_.data(), qkv_total, config_.hidden_size, config_.group_size);

        rope_apply_inplace(qkv_out_.data(), qkv_out_.data() + q_dim, weights_.cos_table, weights_.sin_table, pos, config_.rotary_dim);
        set_kv_cache(l, pos, qkv_out_.data() + q_dim, qkv_out_.data() + q_dim + kv_dim);
        compute_attention_flash_q8(att_out_.data(), qkv_out_.data(), l, pos);

        gemv_asb_avx2(lw.o_proj, att_out_.data(), hidden_.data(), config_.hidden_size, q_dim, config_.group_size);

        vec_add_inplace(hidden_.data(), residual_.data(), config_.hidden_size);
        std::memcpy(residual_.data(), hidden_.data(), config_.hidden_size * sizeof(float));

        rmsnorm_f32(hidden_.data(), hidden_.data(), lw.rms_ffn, config_.hidden_size, config_.rms_norm_eps);

        gemv_asb_avx2(lw.gate_up_proj, hidden_.data(), gu_out_.data(), 2 * config_.intermediate_size, config_.hidden_size, config_.group_size);
        swiglu_inplace(gu_out_.data(), gu_out_.data() + config_.intermediate_size, config_.intermediate_size, lw.fatrelu_threshold);

        std::memcpy(gate_buf_.data(), gu_out_.data(), config_.intermediate_size * sizeof(float));
        gemv_asb_avx2(lw.down_proj, gate_buf_.data(), hidden_.data(), config_.hidden_size, config_.intermediate_size, config_.group_size);

        vec_add_inplace(hidden_.data(), residual_.data(), config_.hidden_size);
    }
    
    if (out_logits) {
        rmsnorm_f32(hidden_.data(), hidden_.data(), weights_.output_norm, config_.hidden_size, config_.rms_norm_eps);
        gemv_f32_f32_omp(weights_.output_proj, hidden_.data(), out_logits, config_.vocab_size, config_.hidden_size);
    }
}

std::vector<int32_t> UnifiedEngine::generate_swift(const std::vector<int32_t>& prompt, int max_tokens, int draft_k) {
    std::cout << "Starting SWIFT generate with " << prompt.size() << " prompt tokens." << std::endl;
    std::vector<int32_t> output = prompt;
    int current_pos = 0;

    if (prompt.size() > 0) {
        forward_batch(prompt.data(), prompt.size(), 0, logits_.data(), false);
        current_pos += prompt.size();
    }

    int best_token = 0;
    float best_val = logits_[0];
    for (int i = 1; i < config_.vocab_size; ++i) {
        if (logits_[i] > best_val) { best_val = logits_[i]; best_token = i; }
    }
    output.push_back(best_token);

    std::vector<float> verify_logits(draft_k * config_.vocab_size);
    int accepted_total = 0;
    int drafted_total = 0;

    while (output.size() < prompt.size() + max_tokens) {
        if (best_token == 199999 || best_token == 200020) break;

        std::vector<int32_t> draft_tokens;
        draft_tokens.push_back(best_token); // The first token to feed to the draft layer
        
        int draft_pos = current_pos;
        for (int k = 0; k < draft_k; ++k) {
            forward_token_draft(draft_tokens.back(), draft_pos, logits_.data());
            
            int draft_t = 0; float d_val = logits_[0];
            for (int i = 1; i < config_.vocab_size; ++i) {
                if (logits_[i] > d_val) { d_val = logits_[i]; draft_t = i; }
            }
            draft_tokens.push_back(draft_t);
            draft_pos++;
        }
        drafted_total += draft_k;

        forward_batch(draft_tokens.data(), draft_k, current_pos, verify_logits.data(), true);

        int accepted = 0;
        int next_token_from_target = -1;

        for (int k = 0; k < draft_k; ++k) {
            int target_t = 0; float t_val = verify_logits[k * config_.vocab_size];
            for (int i = 1; i < config_.vocab_size; ++i) {
                if (verify_logits[k * config_.vocab_size + i] > t_val) {
                    t_val = verify_logits[k * config_.vocab_size + i];
                    target_t = i;
                }
            }
            
            next_token_from_target = target_t;
            
            if (target_t == draft_tokens[k + 1]) {
                accepted++;
                output.push_back(target_t);
                if (target_t == 199999 || target_t == 200020) break;
            } else {
                output.push_back(target_t);
                break;
            }
        }
        
        accepted_total += accepted;
        current_pos += accepted + 1; // 1 step further (if accepted=0, +1 for target token)
        best_token = output.back();
        
        if (best_token == 199999 || best_token == 200020) break;
        if (output.size() >= prompt.size() + max_tokens) break;
    }
    
    std::cout << "SWIFT Acceptance rate: " << (float)accepted_total / drafted_total * 100.0f << "%" << std::endl;
    return output;
}

} // namespace asdsl

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace asdsl {

py::class_<EngineConfig> register_config(py::module_& m) {
    auto cls = py::class_<EngineConfig>(m, "EngineConfig")
        .def(py::init<>())
        .def_readwrite("num_layers", &asdsl::EngineConfig::num_layers)
        .def_readwrite("hidden_size", &asdsl::EngineConfig::hidden_size)
        .def_readwrite("num_heads", &asdsl::EngineConfig::num_heads)
        .def_readwrite("num_kv_heads", &asdsl::EngineConfig::num_kv_heads)
        .def_readwrite("head_dim", &asdsl::EngineConfig::head_dim)
        .def_readwrite("intermediate_size", &asdsl::EngineConfig::intermediate_size)
        .def_readwrite("vocab_size", &asdsl::EngineConfig::vocab_size)
        .def_readwrite("rms_norm_eps", &asdsl::EngineConfig::rms_norm_eps)
        .def_readwrite("group_size", &asdsl::EngineConfig::group_size)
        .def_readwrite("max_seq_len", &asdsl::EngineConfig::max_seq_len)
        .def_readwrite("rotary_dim", &asdsl::EngineConfig::rotary_dim);
    return cls;
}

// A wrapper to safely initialize the C++ engine from numpy arrays:
class UnifiedEnginePy {
    std::unique_ptr<UnifiedEngine> engine_;
    EngineWeights weights_;
    
    // We hold references to py::array so memory isn't GC'd
    std::vector<py::array> keep_alive_;

    template<typename T>
    const T* get_ptr(py::array_t<T>& arr) {
        if (arr.size() == 0) return nullptr;
        keep_alive_.push_back(arr);
        return arr.data();
    }

public:
    UnifiedEnginePy(
        EngineConfig config,
        py::array_t<float> token_embd,
        py::array_t<float> output_norm,
        py::array_t<float> output_proj,
        py::array_t<float> cos_table,
        py::array_t<float> sin_table,
        py::dict layers_dict
    ) {
        weights_.token_embd = get_ptr(token_embd);
        weights_.output_norm = get_ptr(output_norm);
        weights_.output_proj = get_ptr(output_proj);
        weights_.cos_table = get_ptr(cos_table);
        weights_.sin_table = get_ptr(sin_table);

        for (auto item : layers_dict) {
            int layer_idx = item.first.cast<int>();
            py::dict l_dict = item.second.cast<py::dict>();
            
            LayerWeights lw;
            
            auto rms_att = l_dict["rms_att"].cast<py::array_t<float>>();
            lw.rms_att = get_ptr(rms_att);
            
            auto qkv_proj = l_dict["qkv_proj"].cast<py::array_t<uint8_t>>();
            lw.qkv_proj = get_ptr(qkv_proj);
            
            
            
            auto o_proj = l_dict["o_proj"].cast<py::array_t<uint8_t>>();
            lw.o_proj = get_ptr(o_proj);
            
            
            
            auto rms_ffn = l_dict["rms_ffn"].cast<py::array_t<float>>();
            lw.rms_ffn = get_ptr(rms_ffn);
            
            auto gate_up_proj = l_dict["gate_up_proj"].cast<py::array_t<uint8_t>>();
            lw.gate_up_proj = get_ptr(gate_up_proj);
            
            
            
            auto down_proj = l_dict["down_proj"].cast<py::array_t<uint8_t>>();
            lw.down_proj = get_ptr(down_proj);
            
            if (l_dict.contains("fatrelu_threshold")) {
                lw.fatrelu_threshold = l_dict["fatrelu_threshold"].cast<float>();
            }

            weights_.layers[layer_idx] = lw;
        }

        engine_ = std::make_unique<UnifiedEngine>(config, weights_);
    }

    std::vector<int32_t> generate(std::vector<int32_t> prompt, int max_tokens) {
        // Release GIL during generation process
        py::gil_scoped_release release;
        return engine_->generate(prompt, max_tokens);
    }

    std::vector<int32_t> generate_swift(std::vector<int32_t> prompt, int max_tokens, int draft_k) {
        py::gil_scoped_release release;
        return engine_->generate_swift(prompt, max_tokens, draft_k);
    }
};

}

PYBIND11_MODULE(_native_unified, m) {
    asdsl::register_config(m);
    
    py::class_<asdsl::UnifiedEnginePy>(m, "UnifiedEngine")
        .def(py::init<
            asdsl::EngineConfig,
            py::array_t<float>,
            py::array_t<float>,
            py::array_t<float>,
            py::array_t<float>,
            py::array_t<float>,
            py::dict
        >(), 
        py::arg("config"),
        py::arg("token_embd"),
        py::arg("output_norm"),
        py::arg("output_proj"),
        py::arg("cos_table"),
        py::arg("sin_table"),
        py::arg("layers_dict"))
        .def("generate", &asdsl::UnifiedEnginePy::generate)
        .def("generate_swift", &asdsl::UnifiedEnginePy::generate_swift);
}
