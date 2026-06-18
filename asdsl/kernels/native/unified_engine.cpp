#include "unified_engine.h"
#include "gemv_q2_kernels.h"
#include "gemv_q4_kernel.h"
#include "large_pages.hpp"
#include <immintrin.h>
#include <cmath>
#include <cstring>
#include <limits>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef ASDSL_PROFILE
namespace {
struct ForwardTokenProfile {
    std::chrono::nanoseconds rmsnorm_quantize{};
    std::chrono::nanoseconds gemv_qkv{};
    std::chrono::nanoseconds rope{};
    std::chrono::nanoseconds kv_cache{};
    std::chrono::nanoseconds attention{};
    std::chrono::nanoseconds quantize_att{};
    std::chrono::nanoseconds gemv_o{};
    std::chrono::nanoseconds gemv_gateup{};
    std::chrono::nanoseconds swiglu{};
    std::chrono::nanoseconds quantize_gate{};
    std::chrono::nanoseconds gemv_down{};
    std::chrono::nanoseconds lm_head{};
    void reset() { *this = ForwardTokenProfile{}; }
    void print(int n_tokens) const {
        using namespace std::chrono;
        auto ms = [](nanoseconds ns) { return duration_cast<microseconds>(ns).count() / 1000.0; };
        std::cerr << "[ASDSL_PROFILE] forward_token totals over " << n_tokens << " token(s) (ms):\n";
        std::cerr << "  rmsnorm_quantize " << ms(rmsnorm_quantize) << "\n";
        std::cerr << "  gemv_qkv " << ms(gemv_qkv) << "\n";
        std::cerr << "  rope " << ms(rope) << "\n";
        std::cerr << "  kv_cache " << ms(kv_cache) << "\n";
        std::cerr << "  attention " << ms(attention) << "\n";
        std::cerr << "  quantize_att " << ms(quantize_att) << "\n";
        std::cerr << "  gemv_o " << ms(gemv_o) << "\n";
        std::cerr << "  gemv_gateup " << ms(gemv_gateup) << "\n";
        std::cerr << "  swiglu " << ms(swiglu) << "\n";
        std::cerr << "  quantize_gate " << ms(quantize_gate) << "\n";
        std::cerr << "  gemv_down " << ms(gemv_down) << "\n";
        std::cerr << "  lm_head " << ms(lm_head) << "\n";
    }
};
ForwardTokenProfile g_forward_prof;
} // namespace
#endif

namespace {
struct EngineRuntimeProfile {
    double gemv_qkv_ms = 0;
    double gemv_o_ms = 0;
    double gemv_gateup_ms = 0;
    double gemv_down_ms = 0;
    double activation_q8_ms = 0;
    double prep_fused_ms = 0;
    double lm_head_ms = 0;
    double other_ms = 0;
    void reset() { *this = EngineRuntimeProfile{}; }
};

bool engine_profile_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char* v = std::getenv("ASDSL_ENGINE_PROFILE");
        enabled = (v && v[0] != '0') ? 1 : 0;
    }
    return enabled != 0;
}

int engine_profile_target_token() {
    static int target = -2;
    if (target == -2) {
        const char* v = std::getenv("ASDSL_ENGINE_PROFILE_TOKEN");
        target = v ? std::atoi(v) : 3;
    }
    return target;
}

using Clock = std::chrono::high_resolution_clock;
inline double ms_since(const Clock::time_point& t0) {
    return std::chrono::duration<double, std::milli>(
        Clock::now() - t0).count();
}

} // namespace

namespace asdsl {

bool fused_gemv_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char* v = std::getenv("ASDSL_FUSED_GEMV");
        enabled = (v && (v[0] == '0')) ? 0 : 1;
    }
    return enabled != 0;
}

inline bool use_q4_32_preq_fused(const EngineConfig& cfg) {
    return fused_gemv_enabled() && cfg.weight_format == 0 && cfg.group_size == 32;
}

static bool preq2_gemv_enabled() {
    const char* v = std::getenv("ASDSL_PREQ2");
    if (!v || v[0] == '\0') {
        return true;  // match phi4_cpu_run / parity_manifest default
    }
    return v[0] != '0';
}

static bool c01_gemv_enabled() {
    const char* v = std::getenv("ASDSL_C01");
    return v && v[0] != '0';
}

static int lm_head_group_size(const EngineConfig& cfg) {
    return cfg.lm_head_group_size > 0 ? cfg.lm_head_group_size : cfg.group_size;
}

static void fused_preq_gemv(
    const asdsl::Preq2Weights& p2,
    const uint8_t* preq_blocks,
    const float* x,
    float* y,
    int out_features,
    int in_features,
    int group_size) {
    const bool use_p2 = preq2_gemv_enabled() && p2.meta && p2.quant;
    if (use_p2) {
        gemv_preq2_fused_avx2(p2.meta, p2.quant, x, y, out_features, in_features, group_size);
    } else {
        gemv_q4_32_preq_fused_avx2(preq_blocks, x, y, out_features, in_features, group_size);
    }
}

// Returns the active pool for the current call stack.
// Inside UnifiedEngine methods, tl_active_pool is always set to pool_
// by the ScopedActivePool guard placed at each public entry point.
ThreadPool& get_global_thread_pool() {
    return ThreadPool::get_instance();
}

// Fast hsum: movehdup+movehl+add_ss uses any execution port (1 cycle each).
// Avoids hadd_ps which has 3-cycle latency and is restricted to ports 1/5.
static inline float hsum256_ps(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    lo = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, lo);
    lo = _mm_add_ss(lo, shuf);
    return _mm_cvtss_f32(lo);
}

static inline float hmax256_ps(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    lo = _mm_max_ps(lo, hi);
    lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_max_ss(lo, _mm_movehdup_ps(lo));
    return _mm_cvtss_f32(lo);
}

// Forward declaration (defined below in swiglu_inplace area)
inline __m256 fast_exp_avx2(__m256 x);

static inline float cvtsh_ss(uint16_t h) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<int>(h))));
}

static void load_embed_row_f32(const asdsl::EngineWeights& weights, int token_id, int hidden_size, float* dst) {
    if (weights.embed_fp16 && weights.token_embd_f16) {
        const uint16_t* row = weights.token_embd_f16 + static_cast<size_t>(token_id) * hidden_size;
        int j = 0;
        for (; j + 8 <= hidden_size; j += 8) {
            const __m128i h8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + j));
            const __m256 f8 = _mm256_cvtph_ps(h8);
            _mm256_storeu_ps(dst + j, f8);
        }
        for (; j < hidden_size; ++j) {
            dst[j] = cvtsh_ss(row[j]);
        }
        return;
    }
    const float* row = weights.token_embd + static_cast<size_t>(token_id) * hidden_size;
    std::memcpy(dst, row, static_cast<size_t>(hidden_size) * sizeof(float));
}

static inline float output_proj_elem_f32(const asdsl::EngineWeights& weights, int row, int col, int cols) {
    const size_t idx = static_cast<size_t>(row) * cols + col;
    if (weights.output_proj_fp16 && weights.output_proj_f16) {
        return cvtsh_ss(weights.output_proj_f16[idx]);
    }
    return weights.output_proj[idx];
}

static int argmax_f32_avx2(const float* logits, int n) {
    if (n <= 0) return 0;
    int best_i = 0;
    float best_v = logits[0];
    int i = 1;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(logits + i);
        float chunk_max = hmax256_ps(v);
        if (chunk_max > best_v) {
            float t[8];
            _mm256_storeu_ps(t, v);
            for (int j = 0; j < 8; ++j) {
                if (t[j] > best_v) {
                    best_v = t[j];
                    best_i = i + j;
                }
            }
        }
    }
    for (; i < n; ++i) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best_i = i;
        }
    }
    return best_i;
}


UnifiedEngine::UnifiedEngine(const EngineConfig& config, const EngineWeights& weights)
    : config_(config), weights_(weights) {
    if (config_.hidden_size <= 0 || config_.intermediate_size <= 0 || config_.vocab_size <= 0) {
        throw std::invalid_argument("UnifiedEngine: invalid non-positive model dimensions");
    }
    if (config_.group_size <= 0) {
        throw std::invalid_argument("UnifiedEngine: group_size must be positive");
    }
    if ((config_.hidden_size % config_.group_size) != 0) {
        throw std::invalid_argument("UnifiedEngine: hidden_size must be divisible by group_size");
    }
    if ((config_.intermediate_size % config_.group_size) != 0) {
        throw std::invalid_argument("UnifiedEngine: intermediate_size must be divisible by group_size");
    }
    
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
    
    scores_buf_.resize(config_.num_heads * config_.max_seq_len, 0.0f);
    num_buf_.resize(config_.num_heads * config_.head_dim, 0.0f);
    tile_num_buf_.resize(config_.num_heads * config_.head_dim, 0.0f);

    hidden_q8_.resize(config_.hidden_size, 0);
    hidden_scales_.resize(config_.hidden_size / config_.group_size, 0.0f);
    hidden_q8_g128_.resize(config_.hidden_size, 0);
    hidden_scales_g128_.resize(config_.hidden_size / 128, 0.0f);
    gate_q8_.resize(config_.intermediate_size, 0);
    gate_scales_.resize(config_.intermediate_size / config_.group_size, 0.0f);
    gate_q8_g128_.resize(config_.intermediate_size, 0);
    gate_scales_g128_.resize(config_.intermediate_size / 128, 0.0f);

    const int lm_gs = config_.lm_head_group_size > 0 ? config_.lm_head_group_size : config_.group_size;

    // ── lm_head preq2 from disk cache (skip fp16 quantize) ─────────────────────
    if (weights_.lm_head_preq2_meta_in && weights_.lm_head_preq2_quant_in
        && weights_.lm_head_preq2_meta_size > 0 && weights_.lm_head_preq2_quant_size > 0
        && lm_gs == 32 && preq2_gemv_enabled()) {
        const int cols = config_.hidden_size;
        const int rows = config_.vocab_size;
        const int n_groups = cols / 32;
        constexpr int meta_b = 4;
        constexpr int quant_group = 64;
        constexpr int row_band = 4;
        const size_t expect_meta = static_cast<size_t>(rows) * n_groups * meta_b;
        const int n_bands = (rows + row_band - 1) / row_band;
        const size_t expect_quant = static_cast<size_t>(n_bands) * n_groups * quant_group;
        if (weights_.lm_head_preq2_meta_size == expect_meta
            && weights_.lm_head_preq2_quant_size == expect_quant) {
            lm_head_preq2_meta_.assign(
                weights_.lm_head_preq2_meta_in,
                weights_.lm_head_preq2_meta_in + weights_.lm_head_preq2_meta_size);
            lm_head_preq2_quant_.assign(
                weights_.lm_head_preq2_quant_in,
                weights_.lm_head_preq2_quant_in + weights_.lm_head_preq2_quant_size);
            lm_head_preq2_ready_ = true;
        }
    }

    // ── lm_head Q4_32 quantization (group_size=32) ──────────────────────────────
    if (!lm_head_preq2_ready_
        && (weights_.output_proj || weights_.output_proj_f16) && lm_gs == 32) {
        const int cols       = config_.hidden_size;
        const int rows       = config_.vocab_size;
        const int group_size = 32;
        const int n_groups   = cols / group_size;
        const int block_size = 20;
        const size_t row_bytes   = static_cast<size_t>(n_groups) * block_size;
        const size_t total_bytes = static_cast<size_t>(rows)    * row_bytes;
        lm_head_q4_blocks_.resize(total_bytes);
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; ++r) {
            std::vector<float> row_fp32;
            const float* src_row;
            if (weights_.output_proj_fp16) {
                row_fp32.resize(static_cast<size_t>(cols));
                for (int c = 0; c < cols; ++c) {
                    row_fp32[static_cast<size_t>(c)] = output_proj_elem_f32(weights_, r, c, cols);
                }
                src_row = row_fp32.data();
            } else {
                src_row = weights_.output_proj + static_cast<size_t>(r) * cols;
            }
            uint8_t*     dst_row = lm_head_q4_blocks_.data() + static_cast<size_t>(r) * row_bytes;
            for (int g = 0; g < n_groups; ++g) {
                const float* xg  = src_row + g * group_size;
                uint8_t*     blk = dst_row + g * block_size;
                __m256 vmax = _mm256_setzero_ps();
                for (int j = 0; j < group_size; j += 8)
                    vmax = _mm256_max_ps(vmax, _mm256_andnot_ps(_mm256_set1_ps(-0.0f), _mm256_loadu_ps(xg+j)));
                __m128 lo128 = _mm256_castps256_ps128(vmax), hi128 = _mm256_extractf128_ps(vmax,1);
                lo128 = _mm_max_ps(lo128,hi128); lo128=_mm_max_ps(lo128,_mm_movehl_ps(lo128,lo128));
                lo128 = _mm_max_ps(lo128,_mm_movehdup_ps(lo128));
                float amax = _mm_cvtss_f32(lo128);
                if (amax < 1e-12f) {
                    uint16_t z=0; std::memcpy(blk,&z,2);
                    uint16_t zf=0x4800; std::memcpy(blk+2,&zf,2);
                    std::memset(blk+4,0x88,16); continue;
                }
                float scale = amax/7.0f;
                uint16_t sf16 = static_cast<uint16_t>(_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(scale),_MM_FROUND_TO_NEAREST_INT),0));
                std::memcpy(blk,&sf16,2);
                uint16_t zf16 = static_cast<uint16_t>(_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(8.0f),_MM_FROUND_TO_NEAREST_INT),0));
                std::memcpy(blk+2,&zf16,2);
                float inv = 7.0f/amax; uint8_t* nb = blk+4;
                for (int j=0;j<group_size;j+=2) {
                    int q0=static_cast<int>(std::round(xg[j]*inv))+8, q1=static_cast<int>(std::round(xg[j+1]*inv))+8;
                    q0=q0<0?0:q0>15?15:q0; q1=q1<0?0:q1>15?15:q1;
                    nb[j/2]=static_cast<uint8_t>((q1<<4)|q0);
                }
            }
        }
        lm_head_q4_quantized_ = true;

        if (preq2_gemv_enabled()) {
            constexpr int meta_b = 4;
            constexpr int quant_group = 64;
            constexpr int row_band = 4;
            const size_t meta_bytes = static_cast<size_t>(rows) * n_groups * meta_b;
            const int n_bands = (rows + row_band - 1) / row_band;
            const size_t quant_bytes = static_cast<size_t>(n_bands) * n_groups * quant_group;
            lm_head_preq2_meta_.resize(meta_bytes);
            lm_head_preq2_quant_.resize(quant_bytes);
            #pragma omp parallel for schedule(static)
            for (int r = 0; r < rows; ++r) {
                const uint8_t* src_row = lm_head_q4_blocks_.data() + static_cast<size_t>(r) * row_bytes;
                uint8_t* meta_row = lm_head_preq2_meta_.data() + static_cast<size_t>(r) * n_groups * meta_b;
                for (int g = 0; g < n_groups; ++g) {
                    const uint8_t* blk = src_row + g * block_size;
                    std::memcpy(meta_row + g * meta_b, blk, meta_b);
                    const int band = r / row_band;
                    const int slot = r % row_band;
                    uint8_t* qdst = lm_head_preq2_quant_.data()
                        + (static_cast<size_t>(band) * n_groups + g) * quant_group + slot * 16;
                    std::memcpy(qdst, blk + 4, 16);
                }
            }
            lm_head_preq2_ready_ = true;
        }
    }

    if (lm_head_preq2_ready_) {
        lm_head_q4_blocks_.clear();
        lm_head_q4_blocks_.shrink_to_fit();
        lm_head_q4_quantized_ = false;
        weights_.output_proj = nullptr;
        weights_.output_proj_f16 = nullptr;
        weights_.output_proj_fp16 = false;
    }

    // ── lm_head Q4_128 quantization (group_size=128) ──────────────────────────
    // This cuts lm_head bandwidth from 2.06 GB/token (FP32) to 0.265 GB/token.
    if ((weights_.output_proj || weights_.output_proj_f16) && lm_gs == 128) {
        const int cols       = config_.hidden_size;   // 5120
        const int rows       = config_.vocab_size;    // 100352
        const int gs         = 128;
        const int n_groups   = cols / gs;             // 40
        const size_t blk_sz  = 66;                   // 2B fp16 + 64B nibbles
        const size_t row_bytes   = static_cast<size_t>(n_groups) * blk_sz;
        const size_t total_bytes = static_cast<size_t>(rows)     * row_bytes;
        lm_head_q4_blocks_.resize(total_bytes);
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; ++r) {
            std::vector<float> row_fp32;
            const float* src_row;
            if (weights_.output_proj_fp16) {
                row_fp32.resize(static_cast<size_t>(cols));
                for (int c = 0; c < cols; ++c) {
                    row_fp32[static_cast<size_t>(c)] = output_proj_elem_f32(weights_, r, c, cols);
                }
                src_row = row_fp32.data();
            } else {
                src_row = weights_.output_proj + static_cast<size_t>(r) * cols;
            }
            uint8_t*     dst_row = lm_head_q4_blocks_.data() + static_cast<size_t>(r) * row_bytes;
            for (int g = 0; g < n_groups; ++g) {
                const float* xg  = src_row + g * gs;
                uint8_t*     blk = dst_row + g * blk_sz;
                // amax over 128 elements
                __m256 vmax = _mm256_setzero_ps();
                const __m256 sign_mask = _mm256_set1_ps(-0.0f);
                for (int j = 0; j < gs; j += 8)
                    vmax = _mm256_max_ps(vmax, _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(xg+j)));
                __m128 lo128 = _mm256_castps256_ps128(vmax), hi128 = _mm256_extractf128_ps(vmax,1);
                lo128 = _mm_max_ps(lo128,hi128); lo128=_mm_max_ps(lo128,_mm_movehl_ps(lo128,lo128));
                lo128 = _mm_max_ps(lo128,_mm_movehdup_ps(lo128));
                float amax = _mm_cvtss_f32(lo128);
                if (amax < 1e-12f) { uint16_t z=0; std::memcpy(blk,&z,2); std::memset(blk+2,0x88,64); continue; }
                float scale = amax/7.0f;
                uint16_t sf16 = static_cast<uint16_t>(_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(scale),_MM_FROUND_TO_NEAREST_INT),0));
                std::memcpy(blk,&sf16,2);
                float inv = 7.0f/amax; uint8_t* nb = blk+2;
                for (int j=0;j<gs;j+=2) {
                    int q0=static_cast<int>(std::round(xg[j]*inv))+8, q1=static_cast<int>(std::round(xg[j+1]*inv))+8;
                    q0=q0<0?0:q0>15?15:q0; q1=q1<0?0:q1>15?15:q1;
                    nb[j/2]=static_cast<uint8_t>((q1<<4)|q0);
                }
            }
        }
        lm_head_q4_quantized_ = true;
    }

    int q_dim_c = config_.num_heads * config_.head_dim;
    int kv_dim_c = config_.num_kv_heads * config_.head_dim;
    int qkv_total_c = q_dim_c + 2 * kv_dim_c;
    size_t max_t = static_cast<size_t>(config_.max_seq_len);
    b_hidden_.resize(max_t * static_cast<size_t>(config_.hidden_size));
    b_residual_.resize(max_t * static_cast<size_t>(config_.hidden_size));
    b_qkv_out_.resize(max_t * static_cast<size_t>(qkv_total_c));
    b_att_out_.resize(max_t * static_cast<size_t>(config_.hidden_size));
    b_gu_out_.resize(max_t * static_cast<size_t>(2 * config_.intermediate_size));
    b_gate_batch_.resize(max_t * static_cast<size_t>(config_.intermediate_size));
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

void UnifiedEngine::rmsnorm_quantize_f32(const float* in, const float* weight, int8_t* out_q8, float* out_scales, int size, int group_size, float eps) {
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
    float inv_rms = 1.0f / std::sqrt(sum / static_cast<float>(size) + eps);
    __m256 inv_rms_v = _mm256_set1_ps(inv_rms);
    const int n_groups = size / group_size;
    for (int g = 0; g < n_groups; ++g) {
        const float* xg = in + g * group_size;
        const float* wg = weight + g * group_size;
        int8_t* xq = out_q8 + g * group_size;
        __m256 max_abs = _mm256_setzero_ps();
        int j = 0;
        for (; j + 8 <= group_size; j += 8) {
            __m256 x = _mm256_loadu_ps(xg + j);
            __m256 w = _mm256_loadu_ps(wg + j);
            __m256 n = _mm256_mul_ps(_mm256_mul_ps(x, inv_rms_v), w);
            __m256 a = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), n);
            max_abs = _mm256_max_ps(max_abs, a);
        }
        float amax = hmax256_ps(max_abs);
        for (; j < group_size; ++j) {
            float x = xg[j] * inv_rms * wg[j];
            amax = std::max(amax, std::fabs(x));
        }
        if (amax < 1e-9f) {
            std::memset(xq, 0, static_cast<size_t>(group_size));
            out_scales[g] = 1.0f;
            continue;
        }
        float scale = amax / 127.0f;
        float inv_scale = 127.0f / amax;
        out_scales[g] = scale;
        __m256 vscale = _mm256_set1_ps(inv_scale);
        j = 0;
        for (; j + 8 <= group_size; j += 8) {
            __m256 x = _mm256_loadu_ps(xg + j);
            __m256 w = _mm256_loadu_ps(wg + j);
            __m256 n = _mm256_mul_ps(_mm256_mul_ps(x, inv_rms_v), w);
            __m256 scaled = _mm256_mul_ps(n, vscale);
            __m256i int32 = _mm256_cvtps_epi32(_mm256_round_ps(scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m128i int16 = _mm_packs_epi32(_mm256_castsi256_si128(int32), _mm256_extracti128_si256(int32, 1));
            __m128i int8 = _mm_packs_epi16(int16, int16);
            _mm_storel_epi64((__m128i*)(xq + j), int8);
        }
        for (; j < group_size; ++j) {
            float x = xg[j] * inv_rms * wg[j];
            int q = static_cast<int>(std::nearbyint(x * inv_scale));
            q = std::max(-127, std::min(127, q));
            xq[j] = static_cast<int8_t>(q);
        }
    }
}

// Fused: vec_add(hidden, residual) + memcpy(residual, hidden) + rmsnorm_quantize(hidden → q8)
// Saves 2 full memory passes over the hidden vector per call (40KB saved per call)
void UnifiedEngine::residual_add_rmsnorm_quantize_f32(
    float* hidden, float* residual, const float* rms_weight,
    int8_t* out_q8, float* out_scales, int size, int group_size, float eps) {
    // Pass 1: hidden[i] += residual[i], residual[i] = hidden[i], accumulate sum_sq
    __m256 sum_v = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 h = _mm256_loadu_ps(hidden + i);
        __m256 r = _mm256_loadu_ps(residual + i);
        __m256 added = _mm256_add_ps(h, r);
        _mm256_storeu_ps(hidden + i, added);
        _mm256_storeu_ps(residual + i, added);
        sum_v = _mm256_fmadd_ps(added, added, sum_v);
    }
    float sum = hsum256_ps(sum_v);
    for (; i < size; ++i) {
        float v = hidden[i] + residual[i];
        hidden[i] = v;
        residual[i] = v;
        sum += v * v;
    }
    float inv_rms = 1.0f / std::sqrt(sum / static_cast<float>(size) + eps);

    // Passes 2-3: per-group max_abs + normalize + quantize (same as rmsnorm_quantize_f32)
    __m256 inv_rms_v = _mm256_set1_ps(inv_rms);
    const int n_groups = size / group_size;
    for (int g = 0; g < n_groups; ++g) {
        const float* xg = hidden + g * group_size;
        const float* wg = rms_weight + g * group_size;
        int8_t* xq = out_q8 + g * group_size;
        __m256 max_abs = _mm256_setzero_ps();
        int j = 0;
        for (; j + 8 <= group_size; j += 8) {
            __m256 x = _mm256_loadu_ps(xg + j);
            __m256 w = _mm256_loadu_ps(wg + j);
            __m256 n = _mm256_mul_ps(_mm256_mul_ps(x, inv_rms_v), w);
            __m256 a = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), n);
            max_abs = _mm256_max_ps(max_abs, a);
        }
        float amax = hmax256_ps(max_abs);
        for (; j < group_size; ++j) {
            float x = xg[j] * inv_rms * wg[j];
            amax = std::max(amax, std::fabs(x));
        }
        if (amax < 1e-9f) {
            std::memset(xq, 0, static_cast<size_t>(group_size));
            out_scales[g] = 1.0f;
            continue;
        }
        float scale = amax / 127.0f;
        float inv_scale = 127.0f / amax;
        out_scales[g] = scale;
        __m256 vscale = _mm256_set1_ps(inv_scale);
        j = 0;
        for (; j + 8 <= group_size; j += 8) {
            __m256 x = _mm256_loadu_ps(xg + j);
            __m256 w = _mm256_loadu_ps(wg + j);
            __m256 n = _mm256_mul_ps(_mm256_mul_ps(x, inv_rms_v), w);
            __m256 scaled = _mm256_mul_ps(n, vscale);
            __m256i int32 = _mm256_cvtps_epi32(_mm256_round_ps(scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m128i int16 = _mm_packs_epi32(_mm256_castsi256_si128(int32), _mm256_extracti128_si256(int32, 1));
            __m128i int8v = _mm_packs_epi16(int16, int16);
            _mm_storel_epi64((__m128i*)(xq + j), int8v);
        }
        for (; j < group_size; ++j) {
            float x = xg[j] * inv_rms * wg[j];
            int q = static_cast<int>(std::nearbyint(x * inv_scale));
            q = std::max(-127, std::min(127, q));
            xq[j] = static_cast<int8_t>(q);
        }
    }
}

// Residual add + RMSNorm in FP32 (no Q8 write — pair with fused preq GEMV).
void UnifiedEngine::residual_add_rmsnorm_f32(
    float* hidden, float* residual, const float* rms_weight, int size, float eps) {
    __m256 sum_v = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 h = _mm256_loadu_ps(hidden + i);
        __m256 r = _mm256_loadu_ps(residual + i);
        __m256 added = _mm256_add_ps(h, r);
        _mm256_storeu_ps(hidden + i, added);
        _mm256_storeu_ps(residual + i, added);
        sum_v = _mm256_fmadd_ps(added, added, sum_v);
    }
    float sum = hsum256_ps(sum_v);
    for (; i < size; ++i) {
        float v = hidden[i] + residual[i];
        hidden[i] = v;
        residual[i] = v;
        sum += v * v;
    }
    float inv_rms = 1.0f / std::sqrt(sum / static_cast<float>(size) + eps);
    __m256 inv_rms_v = _mm256_set1_ps(inv_rms);
    i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 x = _mm256_loadu_ps(hidden + i);
        __m256 w = _mm256_loadu_ps(rms_weight + i);
        _mm256_storeu_ps(hidden + i, _mm256_mul_ps(_mm256_mul_ps(x, inv_rms_v), w));
    }
    for (; i < size; ++i) {
        hidden[i] = hidden[i] * inv_rms * rms_weight[i];
    }
}

// Fused: swiglu_inplace(gate, up) + quantize_activation(gate → q8)
// Saves 1 full memory pass over the intermediate vector per call (70KB saved per call)
void UnifiedEngine::swiglu_quantize_inplace(
    float* gate, float* up, int8_t* out_q8, float* out_scales,
    int size, int group_size, float threshold) {
    const int n_groups = size / group_size;
    for (int g = 0; g < n_groups; ++g) {
        float* gg = gate + g * group_size;
        float* ug = up + g * group_size;
        int8_t* xq = out_q8 + g * group_size;

        // Pass 1: compute swiglu + find max_abs
        __m256 max_abs = _mm256_setzero_ps();
        int j = 0;
        for (; j + 8 <= group_size; j += 8) {
            __m256 gv = _mm256_loadu_ps(gg + j);
            __m256 uv = _mm256_loadu_ps(ug + j);
            __m256 neg_g = _mm256_sub_ps(_mm256_setzero_ps(), gv);
            __m256 exp_neg = fast_exp_avx2(neg_g);
            __m256 sigmoid = _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_add_ps(_mm256_set1_ps(1.0f), exp_neg));
            __m256 silu = _mm256_mul_ps(gv, sigmoid);
            __m256 val = _mm256_mul_ps(silu, uv);
            if (threshold > 0.0f) {
                __m256 abs_val = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), val);
                __m256 mask = _mm256_cmp_ps(abs_val, _mm256_set1_ps(threshold), _CMP_GE_OQ);
                val = _mm256_and_ps(val, mask);
            }
            _mm256_storeu_ps(gg + j, val);
            __m256 a = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), val);
            max_abs = _mm256_max_ps(max_abs, a);
        }
        for (; j < group_size; ++j) {
            float x = gg[j], sig = 1.0f / (1.0f + std::exp(-x));
            float v = x * sig * ug[j];
            if (threshold > 0.0f && std::abs(v) < threshold) v = 0.0f;
            gg[j] = v;
        }

        __m128 lo = _mm256_castps256_ps128(max_abs);
        __m128 hi = _mm256_extractf128_ps(max_abs, 1);
        lo = _mm_max_ps(lo, hi);
        lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
        lo = _mm_max_ps(lo, _mm_movehdup_ps(lo));
        float amax = _mm_cvtss_f32(lo);

        if (amax < 1e-9f) {
            std::memset(xq, 0, static_cast<size_t>(group_size));
            out_scales[g] = 1.0f;
            continue;
        }
        float scale = amax / 127.0f;
        float inv_scale = 127.0f / amax;
        out_scales[g] = scale;

        // Pass 2: quantize (read swiglu output → write int8)
        __m256 vscale = _mm256_set1_ps(inv_scale);
        j = 0;
        for (; j + 8 <= group_size; j += 8) {
            __m256 v = _mm256_loadu_ps(gg + j);
            __m256 scaled = _mm256_mul_ps(v, vscale);
            __m256i int32 = _mm256_cvtps_epi32(scaled);
            __m128i int16 = _mm_packs_epi32(_mm256_castsi256_si128(int32), _mm256_extracti128_si256(int32, 1));
            __m128i int8v = _mm_packs_epi16(int16, int16);
            _mm_storel_epi64((__m128i*)(xq + j), int8v);
        }
        for (; j < group_size; ++j) {
            int q = static_cast<int>(std::nearbyint(gg[j] * inv_scale));
            q = std::max(-127, std::min(127, q));
            xq[j] = static_cast<int8_t>(q);
        }
    }
}

void UnifiedEngine::rope_apply_inplace(float* q, float* k, const float* cos_table, const float* sin_table, int pos, int rotary_dim) {
    const int half_rotary = rotary_dim / 2;
    const float* cos_p = cos_table + pos * half_rotary;
    const float* sin_p = sin_table + pos * half_rotary;

    auto rotate_head = [&](float* head_row) {
        int i = 0;
        for (; i + 8 <= half_rotary; i += 8) {
            __m256 x_first = _mm256_loadu_ps(head_row + i);
            __m256 x_second = _mm256_loadu_ps(head_row + half_rotary + i);
            __m256 c = _mm256_loadu_ps(cos_p + i);
            __m256 s = _mm256_loadu_ps(sin_p + i);
            __m256 o0 = _mm256_fmsub_ps(x_first, c, _mm256_mul_ps(x_second, s));
            __m256 o1 = _mm256_fmadd_ps(x_first, s, _mm256_mul_ps(x_second, c));
            _mm256_storeu_ps(head_row + i, o0);
            _mm256_storeu_ps(head_row + half_rotary + i, o1);
        }
        for (; i < half_rotary; ++i) {
            float xf = head_row[i];
            float xs = head_row[half_rotary + i];
            float cc = cos_p[i];
            float ss = sin_p[i];
            head_row[i] = xf * cc - xs * ss;
            head_row[half_rotary + i] = xf * ss + xs * cc;
        }
    };

    for (int h = 0; h < config_.num_heads; ++h) {
        rotate_head(q + h * config_.head_dim);
    }
    for (int h = 0; h < config_.num_kv_heads; ++h) {
        rotate_head(k + h * config_.head_dim);
    }
}

inline __m256 fast_exp_avx2(__m256 x) {
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.3f));
    x = _mm256_min_ps(x, _mm256_set1_ps(87.3f));
    __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
    x = _mm256_mul_ps(x, log2e);
    __m256i i = _mm256_cvtps_epi32(x);
    __m256 fi = _mm256_cvtepi32_ps(i);
    __m256 fract = _mm256_sub_ps(x, fi);
    __m256 c1 = _mm256_set1_ps(0.6931471805599453f);
    __m256 c2 = _mm256_set1_ps(0.2402265069591007f);
    __m256 c3 = _mm256_set1_ps(0.05550410866482158f);
    __m256 c4 = _mm256_set1_ps(0.009618129107628477f);
    __m256 c5 = _mm256_set1_ps(0.001333838100523091f);
    __m256 p = _mm256_fmadd_ps(c5, fract, c4);
    p = _mm256_fmadd_ps(p, fract, c3);
    p = _mm256_fmadd_ps(p, fract, c2);
    p = _mm256_fmadd_ps(p, fract, c1);
    p = _mm256_fmadd_ps(p, fract, _mm256_set1_ps(1.0f));
    __m256i exponent = _mm256_add_epi32(i, _mm256_set1_epi32(127));
    exponent = _mm256_slli_epi32(exponent, 23);
    __m256 exp_i = _mm256_castsi256_ps(exponent);
    return _mm256_mul_ps(p, exp_i);
}

void UnifiedEngine::swiglu_inplace(float* gate, float* up, int size, float threshold) {
    int i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 g = _mm256_loadu_ps(gate + i);
        __m256 u = _mm256_loadu_ps(up + i);
        __m256 neg_g = _mm256_sub_ps(_mm256_setzero_ps(), g);
        __m256 exp_neg = fast_exp_avx2(neg_g);
        __m256 sigmoid = _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_add_ps(_mm256_set1_ps(1.0f), exp_neg));
        __m256 silu = _mm256_mul_ps(g, sigmoid);
        __m256 val = _mm256_mul_ps(silu, u);
        
        if (threshold > 0.0f) {
            __m256 abs_val = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), val);
            __m256 mask = _mm256_cmp_ps(abs_val, _mm256_set1_ps(threshold), _CMP_GE_OQ);
            val = _mm256_and_ps(val, mask);
        }
        
        _mm256_storeu_ps(gate + i, val);
    }
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
            int i = 0;
            for (; i + 8 <= len; i += 8) {
                __m256 vk = _mm256_loadu_ps(kh + off + i);
                __m256 vv = _mm256_loadu_ps(vh + off + i);
                __m256 ak = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), vk);
                __m256 av = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), vv);
                k_absmax = std::max(k_absmax, hmax256_ps(ak));
                v_absmax = std::max(v_absmax, hmax256_ps(av));
            }
            for (; i < len; ++i) {
                k_absmax = std::max(k_absmax, std::fabs(kh[off + i]));
                v_absmax = std::max(v_absmax, std::fabs(vh[off + i]));
            }
            float ks = (k_absmax > 1e-12f) ? (k_absmax / 127.0f) : 1e-12f;
            float vs = (v_absmax > 1e-12f) ? (v_absmax / 127.0f) : 1e-12f;
            k_cache_scales_[sb + b] = ks;
            v_cache_scales_[sb + b] = vs;

            float kinv = 1.0f / ks;
            float vinv = 1.0f / vs;
            __m256 kinv_v = _mm256_set1_ps(kinv);
            __m256 vinv_v = _mm256_set1_ps(vinv);
            i = 0;
            for (; i + 8 <= len; i += 8) {
                __m256 vk = _mm256_loadu_ps(kh + off + i);
                __m256 vv = _mm256_loadu_ps(vh + off + i);
                __m256i qki = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(vk, kinv_v), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                __m256i qvi = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(vv, vinv_v), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                qki = _mm256_min_epi32(_mm256_max_epi32(qki, _mm256_set1_epi32(-127)), _mm256_set1_epi32(127));
                qvi = _mm256_min_epi32(_mm256_max_epi32(qvi, _mm256_set1_epi32(-127)), _mm256_set1_epi32(127));
                // Vectorized int32→int8 narrowing: packs_epi32 → packs_epi16 → store 8 bytes.
                // Values are already clamped to [-127,127] so saturation is a no-op.
                int8_t* kdst = k_cache_q8_.data() + kb + off + i;
                int8_t* vdst = v_cache_q8_.data() + kb + off + i;
                __m128i k16 = _mm_packs_epi32(_mm256_castsi256_si128(qki), _mm256_extracti128_si256(qki, 1));
                _mm_storel_epi64((__m128i*)kdst, _mm_packs_epi16(k16, k16));
                __m128i v16 = _mm_packs_epi32(_mm256_castsi256_si128(qvi), _mm256_extracti128_si256(qvi, 1));
                _mm_storel_epi64((__m128i*)vdst, _mm_packs_epi16(v16, v16));
            }
            for (; i < len; ++i) {
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
    const int BLOCK_K = 64;

    auto run_head = [&](int h) {
        const int kv_h = h / groups;
        const float* qh = q + h * config_.head_dim;
        float* num = num_buf_.data() + h * config_.head_dim;
        std::memset(num, 0, config_.head_dim * sizeof(float));
        float m = -std::numeric_limits<float>::infinity();
        float l = 0.0f;

        for (int tk = 0; tk <= seq_pos; tk += BLOCK_K) {
            const int kend = std::min(seq_pos + 1, tk + BLOCK_K);
            const int span = kend - tk;
            float* scores = scores_buf_.data() + h * config_.max_seq_len + tk;
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
                    __m256 ks_v = _mm256_set1_ps(ks);
                    __m256 dot_acc = _mm256_setzero_ps();
                    int i = 0;
                    for (; i + 8 <= len; i += 8) {
                        __m128i k8 = _mm_loadl_epi64((const __m128i*)(kq + i));
                        __m256i k32 = _mm256_cvtepi8_epi32(k8);
                        __m256 kf = _mm256_mul_ps(_mm256_cvtepi32_ps(k32), ks_v);
                        dot_acc = _mm256_fmadd_ps(_mm256_loadu_ps(qh + off + i), kf, dot_acc);
                    }
                    dot += hsum256_ps(dot_acc);
                    for (; i < len; ++i) {
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
            float* tile_num = tile_num_buf_.data() + h * config_.head_dim;
            std::memset(tile_num, 0, config_.head_dim * sizeof(float));

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
                    __m256 w_v = _mm256_set1_ps(w);
                    __m256 vs_v = _mm256_set1_ps(vs);
                    int i = 0;
                    for (; i + 8 <= len; i += 8) {
                        __m128i v8 = _mm_loadl_epi64((const __m128i*)(vq + i));
                        __m256i v32 = _mm256_cvtepi8_epi32(v8);
                        __m256 vf = _mm256_mul_ps(_mm256_cvtepi32_ps(v32), vs_v);
                        __m256 cur = _mm256_loadu_ps(tile_num + off + i);
                        _mm256_storeu_ps(tile_num + off + i, _mm256_fmadd_ps(w_v, vf, cur));
                    }
                    for (; i < len; ++i) {
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
    };

    if (persistent_pool_enabled() && tl_active_pool != nullptr) {
        pool_.parallel_for(0, config_.num_heads, 1, [&](int h) {
            run_head(h);
        });
    } else {
        #pragma omp parallel for schedule(static)
        for (int h = 0; h < config_.num_heads; ++h) {
            run_head(h);
        }
    }
}


static void gemv_q8_q8_omp(const int8_t* W_q8, const float* W_scales, const int8_t* x_q8, const float* x_scales, float* y, int row_begin, int row_end, int cols, int group_size) {
    if (row_end <= row_begin) return;
    const int n_groups = cols / group_size;

    #pragma omp parallel for schedule(static)
    for (int i = row_begin; i < row_end; ++i) {
        const int8_t* w_row = W_q8 + (size_t)i * cols;
        const float* w_row_scales = W_scales + (size_t)i * n_groups;
        __m256 acc_f = _mm256_setzero_ps();
        float tail_sum = 0.0f;

        for (int g = 0; g < n_groups; g++) {
            const int8_t* xq = x_q8 + g * group_size;
            const int8_t* wq = w_row + g * group_size;
            float ws = w_row_scales[g];
            float xs = x_scales[g];
            __m256 scale_v = _mm256_set1_ps(ws * xs);

            __m256i acc = _mm256_setzero_si256();
            int j = 0;
            for (; j + 32 <= group_size; j += 32) {
                __m256i xw0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(xq + j)));
                __m256i xw1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(xq + j + 16)));
                __m256i ww0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(wq + j)));
                __m256i ww1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(wq + j + 16)));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(xw0, ww0));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(xw1, ww1));
            }
            int32_t dot_tail = 0;
            if (j < group_size) {
                for (; j < group_size; ++j) {
                    dot_tail += static_cast<int>(xq[j]) * static_cast<int>(wq[j]);
                }
            }

            acc_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc), scale_v, acc_f);
            tail_sum += static_cast<float>(dot_tail) * ws * xs;
        }
        y[i] = hsum256_ps(acc_f) + tail_sum;
    }
}

static void gemv_f32_f32_omp(const float* W, const float* x, float* y, int rows, int cols) {
    auto process_row = [&](int i) {
        const float* w_row = W + (size_t)i * cols;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        int j = 0;
        for (; j + 32 <= cols; j += 32) {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(w_row + j),      _mm256_loadu_ps(x + j),      acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(w_row + j + 8),  _mm256_loadu_ps(x + j + 8),  acc1);
            acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(w_row + j + 16), _mm256_loadu_ps(x + j + 16), acc2);
            acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(w_row + j + 24), _mm256_loadu_ps(x + j + 24), acc3);
        }
        for (; j + 8 <= cols; j += 8) {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(w_row + j), _mm256_loadu_ps(x + j), acc0);
        }
        acc0 = _mm256_add_ps(acc0, acc1);
        acc2 = _mm256_add_ps(acc2, acc3);
        acc0 = _mm256_add_ps(acc0, acc2);
        __m128 hi = _mm256_extractf128_ps(acc0, 1);
        __m128 lo = _mm256_castps256_ps128(acc0);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        float sum = _mm_cvtss_f32(lo);
        for (; j < cols; ++j) sum += w_row[j] * x[j];
        y[i] = sum;
    };

    if (persistent_pool_enabled() && tl_active_pool != nullptr) {
        asdsl::ThreadPool& pool = asdsl::ThreadPool::get_instance();
        const int n_threads = std::max(1, pool.thread_count() + 1);
        const int grain = std::max(1, (rows + n_threads - 1) / n_threads);
        pool.parallel_for(0, rows, grain, [&](int i) { process_row(i); });
    } else {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < rows; ++i) {
            process_row(i);
        }
    }
}



void UnifiedEngine::forward_batch(const int32_t* tokens, int num_tokens, int start_pos, float* out_logits, bool all_logits) {
    if (num_tokens > config_.max_seq_len) {
        return;
    }
    ScopedActivePool _sap(&pool_);
    float* b_hidden = b_hidden_.data();
    float* b_residual = b_residual_.data();

    int q_dim = config_.num_heads * config_.head_dim;
    int kv_dim = config_.num_kv_heads * config_.head_dim;
    int qkv_total = q_dim + 2 * kv_dim;
    float* b_qkv_out = b_qkv_out_.data();
    float* b_att_out = b_att_out_.data();
    float* b_gu_out = b_gu_out_.data();
    float* b_gate = b_gate_batch_.data();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_tokens; ++i) {
        load_embed_row_f32(
            weights_, tokens[i], config_.hidden_size,
            b_hidden + static_cast<size_t>(i) * config_.hidden_size);
    }

    for (int l = 0; l < config_.num_layers; ++l) {
        if (draft_skip_layers_ && config_.skip_layer[l]) {
            continue;
        }
        const LayerWeights& lw = weights_.layers.at(l);

        std::memcpy(b_residual, b_hidden, static_cast<size_t>(num_tokens) * config_.hidden_size * sizeof(float));

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < num_tokens; ++i) {
            rmsnorm_f32(b_hidden + static_cast<size_t>(i) * config_.hidden_size, b_hidden + static_cast<size_t>(i) * config_.hidden_size, lw.rms_att, config_.hidden_size, config_.rms_norm_eps);
        }

        if (config_.group_size == 128) {
            gemm_q4_128_q8_avx2(lw.qkv_proj, b_hidden, b_qkv_out,
                            qkv_total, config_.hidden_size, config_.group_size, num_tokens);
        } else {
            gemm_q4_32_q8_avx2(lw.qkv_proj, b_hidden, b_qkv_out,
                            qkv_total, config_.hidden_size, config_.group_size, num_tokens);
        }

        for (int i = 0; i < num_tokens; ++i) {
            int pos = start_pos + i;
            float* qkv_row = b_qkv_out + static_cast<size_t>(i) * qkv_total;
            float* q_p = qkv_row;
            float* k_p = qkv_row + q_dim;
            float* v_p = qkv_row + q_dim + kv_dim;
            rope_apply_inplace(q_p, k_p, weights_.cos_table, weights_.sin_table, pos, config_.rotary_dim);
            set_kv_cache(l, pos, k_p, v_p);
            compute_attention_flash_q8(b_att_out + static_cast<size_t>(i) * config_.hidden_size, q_p, l, pos);
        }

        if (config_.group_size == 128) {
            gemm_q4_128_q8_avx2(lw.o_proj, b_att_out, b_hidden,
                            config_.hidden_size, q_dim, config_.group_size, num_tokens);
        } else {
            gemm_q4_32_q8_avx2(lw.o_proj, b_att_out, b_hidden,
                            config_.hidden_size, q_dim, config_.group_size, num_tokens);
        }

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < num_tokens; ++i) {
            vec_add_inplace(b_hidden + static_cast<size_t>(i) * config_.hidden_size,
                            b_residual + static_cast<size_t>(i) * config_.hidden_size, config_.hidden_size);
        }

        std::memcpy(b_residual, b_hidden, static_cast<size_t>(num_tokens) * config_.hidden_size * sizeof(float));

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < num_tokens; ++i) {
            rmsnorm_f32(b_hidden + static_cast<size_t>(i) * config_.hidden_size, b_hidden + static_cast<size_t>(i) * config_.hidden_size, lw.rms_ffn, config_.hidden_size, config_.rms_norm_eps);
        }

        if (config_.group_size == 128) {
            gemm_q4_128_q8_avx2(lw.gate_up_proj, b_hidden, b_gu_out,
                            2 * config_.intermediate_size, config_.hidden_size, config_.group_size, num_tokens);
        } else {
            gemm_q4_32_q8_avx2(lw.gate_up_proj, b_hidden, b_gu_out,
                            2 * config_.intermediate_size, config_.hidden_size, config_.group_size, num_tokens);
        }

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < num_tokens; ++i) {
            float* gu_row = b_gu_out + static_cast<size_t>(i) * 2 * config_.intermediate_size;
            swiglu_inplace(gu_row, gu_row + config_.intermediate_size, config_.intermediate_size, lw.fatrelu_threshold);
        }

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < num_tokens; ++i) {
            std::memcpy(b_gate + static_cast<size_t>(i) * config_.intermediate_size,
                       b_gu_out + static_cast<size_t>(i) * 2 * config_.intermediate_size,
                       config_.intermediate_size * sizeof(float));
        }

        if (config_.group_size == 128) {
            gemm_q4_128_q8_avx2(lw.down_proj, b_gate, b_hidden,
                            config_.hidden_size, config_.intermediate_size, config_.group_size, num_tokens);
        } else {
            gemm_q4_32_q8_avx2(lw.down_proj, b_gate, b_hidden,
                            config_.hidden_size, config_.intermediate_size, config_.group_size, num_tokens);
        }

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < num_tokens; ++i) {
            vec_add_inplace(b_hidden + static_cast<size_t>(i) * config_.hidden_size,
                            b_residual + static_cast<size_t>(i) * config_.hidden_size, config_.hidden_size);
        }
    }

    kv_seq_len_ = std::max(kv_seq_len_, start_pos + num_tokens);

    if (out_logits) {
        if (all_logits) {
            for (int i = 0; i < num_tokens; ++i) {
                float* cur_hidden = b_hidden + static_cast<size_t>(i) * config_.hidden_size;
                float* cur_logits = out_logits + static_cast<size_t>(i) * config_.vocab_size;
                rmsnorm_f32(cur_hidden, cur_hidden, weights_.output_norm, config_.hidden_size, config_.rms_norm_eps);
                if (lm_head_q4_quantized_) {
                    quantize_activation_avx2(cur_hidden, hidden_q8_.data(), hidden_scales_.data(), config_.hidden_size, config_.group_size);
                    if (config_.group_size == 128)
                        gemv_q4_128_preq_avx2(lm_head_q4_blocks_.data(), hidden_q8_.data(), hidden_scales_.data(), cur_logits, config_.vocab_size, config_.hidden_size, config_.group_size);
                    else
                        gemv_q4_32_preq_avx2(lm_head_q4_blocks_.data(), hidden_q8_.data(), hidden_scales_.data(), cur_logits, config_.vocab_size, config_.hidden_size, config_.group_size);
                } else {
                    gemv_f32_f32_omp(weights_.output_proj, cur_hidden, cur_logits, config_.vocab_size, config_.hidden_size);
                }
            }
        } else {
            float* last_hidden = b_hidden + static_cast<size_t>(num_tokens - 1) * config_.hidden_size;
            rmsnorm_f32(last_hidden, last_hidden, weights_.output_norm, config_.hidden_size, config_.rms_norm_eps);
            if (lm_head_q4_quantized_) {
                quantize_activation_avx2(last_hidden, hidden_q8_.data(), hidden_scales_.data(), config_.hidden_size, config_.group_size);
                if (config_.group_size == 128)
                    gemv_q4_128_preq_avx2(lm_head_q4_blocks_.data(), hidden_q8_.data(), hidden_scales_.data(), out_logits, config_.vocab_size, config_.hidden_size, config_.group_size);
                else
                    gemv_q4_32_preq_avx2(lm_head_q4_blocks_.data(), hidden_q8_.data(), hidden_scales_.data(), out_logits, config_.vocab_size, config_.hidden_size, config_.group_size);
            } else {
                gemv_f32_f32_omp(weights_.output_proj, last_hidden, out_logits, config_.vocab_size, config_.hidden_size);
            }
        }
    }
}

void UnifiedEngine::zero_kv_from(int from_pos) {
    if (from_pos < 0) {
        from_pos = 0;
    }
    if (from_pos >= config_.max_seq_len) {
        return;
    }
    #pragma omp parallel for collapse(2) schedule(static)
    for (int layer = 0; layer < config_.num_layers; ++layer) {
        for (int pos = from_pos; pos < config_.max_seq_len; ++pos) {
            for (int h = 0; h < config_.num_kv_heads; ++h) {
                const size_t kb = kv_base(layer, pos, h);
                const size_t sb = scale_base(layer, pos, h);
                std::fill(k_cache_q8_.begin() + kb, k_cache_q8_.begin() + kb + config_.head_dim, static_cast<int8_t>(0));
                std::fill(v_cache_q8_.begin() + kb, v_cache_q8_.begin() + kb + config_.head_dim, static_cast<int8_t>(0));
                std::fill(k_cache_scales_.begin() + sb, k_cache_scales_.begin() + sb + blocks_per_head_, 1.0f);
                std::fill(v_cache_scales_.begin() + sb, v_cache_scales_.begin() + sb + blocks_per_head_, 1.0f);
            }
        }
    }
}

void UnifiedEngine::snapshot_kv() {
    kv_snapshot_stack_.push_back(kv_seq_len_);
}

void UnifiedEngine::restore_kv() {
    if (kv_snapshot_stack_.empty()) {
        return;
    }
    const int len = kv_snapshot_stack_.back();
    kv_snapshot_stack_.pop_back();
    truncate_kv(len);
}

void UnifiedEngine::truncate_kv(int new_len) {
    if (new_len < 0) {
        new_len = 0;
    }
    if (new_len > config_.max_seq_len) {
        new_len = config_.max_seq_len;
    }
    zero_kv_from(new_len);
    kv_seq_len_ = new_len;
}

void UnifiedEngine::set_skip_mask(const bool* mask, int n) {
    const int lim = std::min(n, 64);
    for (int i = 0; i < lim; ++i) {
        config_.skip_layer[i] = mask[i];
    }
    for (int i = lim; i < 64; ++i) {
        config_.skip_layer[i] = false;
    }
}

void UnifiedEngine::clear_skip_mask() {
    for (int i = 0; i < 64; ++i) {
        config_.skip_layer[i] = false;
    }
}

void UnifiedEngine::reset_session() {
    if (!k_cache_q8_.empty()) {
        std::fill(k_cache_q8_.begin(), k_cache_q8_.end(), 0);
        std::fill(v_cache_q8_.begin(), v_cache_q8_.end(), 0);
        std::fill(k_cache_scales_.begin(), k_cache_scales_.end(), 1.0f);
        std::fill(v_cache_scales_.begin(), v_cache_scales_.end(), 1.0f);
    }
    kv_seq_len_ = 0;
    kv_snapshot_stack_.clear();
    clear_skip_mask();
    draft_skip_layers_ = false;
    draft_q2_gemv_ = false;
}

void UnifiedEngine::forward_token(int token_id, int pos, float* out_logits) {
    ScopedActivePool _sap(&pool_);
    const bool do_prof = engine_profile_enabled();
    static int prof_decode_idx = 0;
    EngineRuntimeProfile rtp;
    Clock::time_point t_total;
    if (do_prof) {
        t_total = Clock::now();
        rtp.reset();
    }
    if (token_id < 0 || token_id >= config_.vocab_size) {
        throw std::out_of_range("forward_token: token_id out of vocab range");
    }
    if (pos < 0 || pos >= config_.max_seq_len) {
        throw std::out_of_range("forward_token: pos out of kv-cache range");
    }
    load_embed_row_f32(weights_, token_id, config_.hidden_size, hidden_.data());
    // std::cout << "[DB] emb[0]=" << hidden_[0] << " ";
    // std::cout << "[DB] emb ";

    int q_dim = config_.num_heads * config_.head_dim;
    int kv_dim = config_.num_kv_heads * config_.head_dim;
    int qkv_total = q_dim + 2 * kv_dim;
    const bool fused_preq = use_q4_32_preq_fused(config_);

    for (int l = 0; l < config_.num_layers; ++l) {
        if (draft_skip_layers_ && config_.skip_layer[l]) {
            continue;
        }
        const LayerWeights& lw = weights_.layers.at(l);
#ifdef ASDSL_PROFILE
        std::chrono::high_resolution_clock::time_point _ft_t;
#endif

        // Pre-attention RMSNorm (+ optional Q8 quant for legacy path)
#ifdef ASDSL_PROFILE
        _ft_t = std::chrono::high_resolution_clock::now();
#endif
        Clock::time_point t_act = do_prof ? Clock::now() : Clock::time_point{};
        std::memcpy(residual_.data(), hidden_.data(), config_.hidden_size * sizeof(float));
        if (fused_preq) {
            rmsnorm_f32(hidden_.data(), hidden_.data(), lw.rms_att, config_.hidden_size, config_.rms_norm_eps);
        } else {
            rmsnorm_quantize_f32(hidden_.data(), lw.rms_att, hidden_q8_.data(), hidden_scales_.data(), config_.hidden_size, config_.group_size, config_.rms_norm_eps);
        }
#ifdef ASDSL_PROFILE
        g_forward_prof.rmsnorm_quantize += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - _ft_t);
#endif
        if (do_prof) {
            if (fused_preq) {
                rtp.prep_fused_ms += ms_since(t_act);
            } else {
                rtp.activation_q8_ms += ms_since(t_act);
            }
            t_act = Clock::now();
        }

        // QKV
        Clock::time_point t_gemv = do_prof ? Clock::now() : Clock::time_point{};
        if (draft_q2_gemv_ && lw.has_q2 && lw.q2_qkv_proj && lw.q2_qkv_scales && lw.q2_qkv_biases) {
            gemv_q2_packed_impl(lw.q2_qkv_proj, hidden_.data(), lw.q2_qkv_scales, lw.q2_qkv_biases,
                qkv_out_.data(), qkv_total, config_.hidden_size, config_.group_size);
        } else if (lw.qkv_q4km) {
            gemv_q4km_q8_avx2(lw.qkv_proj, hidden_.data(), qkv_out_.data(),
                qkv_total, config_.hidden_size);
        } else if (c03_gemv_enabled() && lw.qkv_g128 && lw.qkv_proj_g128) {
            quantize_activation_avx2(
                hidden_.data(), hidden_q8_g128_.data(), hidden_scales_g128_.data(),
                config_.hidden_size, 128);
            gemv_q4_128_preq_avx2(
                lw.qkv_proj_g128, hidden_q8_g128_.data(), hidden_scales_g128_.data(),
                qkv_out_.data(), qkv_total, config_.hidden_size, 128);
        } else if (fused_preq) {
            fused_preq_gemv(lw.preq2_qkv, lw.qkv_proj, hidden_.data(), qkv_out_.data(),
                qkv_total, config_.hidden_size, config_.group_size);
        } else if (config_.weight_format == 1) {
            gemv_q4_s256_preq_avx2(lw.qkv_proj, hidden_q8_.data(), hidden_scales_.data(), qkv_out_.data(),
                            qkv_total, config_.hidden_size, config_.group_size);
        } else if (config_.group_size == 128) {
            gemv_q4_128_preq_avx2(lw.qkv_proj, hidden_q8_.data(), hidden_scales_.data(), qkv_out_.data(),
                            qkv_total, config_.hidden_size, config_.group_size);
        } else {
            gemv_q4_32_preq_avx2(lw.qkv_proj, hidden_q8_.data(), hidden_scales_.data(), qkv_out_.data(),
                            qkv_total, config_.hidden_size, config_.group_size);
        }
        if (do_prof) {
            rtp.gemv_qkv_ms += ms_since(t_gemv);
        }

        // RoPE & Cache & Attention
#ifdef ASDSL_PROFILE
        _ft_t = std::chrono::high_resolution_clock::now();
#endif
        Clock::time_point t_other = do_prof ? Clock::now() : Clock::time_point{};
        rope_apply_inplace(qkv_out_.data(), qkv_out_.data() + q_dim, weights_.cos_table, weights_.sin_table, pos, config_.rotary_dim);
#ifdef ASDSL_PROFILE
        g_forward_prof.rope += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - _ft_t);
#endif

#ifdef ASDSL_PROFILE
        _ft_t = std::chrono::high_resolution_clock::now();
#endif
        set_kv_cache(l, pos, qkv_out_.data() + q_dim, qkv_out_.data() + q_dim + kv_dim);
#ifdef ASDSL_PROFILE
        g_forward_prof.kv_cache += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - _ft_t);
#endif

#ifdef ASDSL_PROFILE
        _ft_t = std::chrono::high_resolution_clock::now();
#endif
        compute_attention_flash_q8(att_out_.data(), qkv_out_.data(), l, pos);
#ifdef ASDSL_PROFILE
        g_forward_prof.attention += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - _ft_t);
#endif

#ifdef ASDSL_PROFILE
        _ft_t = std::chrono::high_resolution_clock::now();
#endif
        Clock::time_point t_qatt = do_prof ? Clock::now() : Clock::time_point{};
        if (!fused_preq) {
            quantize_activation_avx2(att_out_.data(), hidden_q8_.data(), hidden_scales_.data(), config_.hidden_size, config_.group_size);
        }
#ifdef ASDSL_PROFILE
        g_forward_prof.quantize_att += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - _ft_t);
#endif
        if (do_prof) {
            rtp.other_ms += ms_since(t_other);
            if (!fused_preq) {
                rtp.activation_q8_ms += ms_since(t_qatt);
            }
            t_act = Clock::now();
        }

        // O proj
        t_gemv = do_prof ? Clock::now() : Clock::time_point{};
        if (draft_q2_gemv_ && lw.has_q2 && lw.q2_o_proj && lw.q2_o_scales && lw.q2_o_biases) {
            gemv_q2_packed_impl(lw.q2_o_proj, att_out_.data(), lw.q2_o_scales, lw.q2_o_biases,
                hidden_.data(), config_.hidden_size, q_dim, config_.group_size);
        } else if (lw.o_q4km) {
            gemv_q4km_q8_avx2(lw.o_proj, att_out_.data(), hidden_.data(),
                config_.hidden_size, q_dim);
        } else if (c03_gemv_enabled() && lw.o_g128 && lw.o_proj_g128) {
            quantize_activation_avx2(
                att_out_.data(), hidden_q8_g128_.data(), hidden_scales_g128_.data(),
                q_dim, 128);
            gemv_q4_128_preq_avx2(
                lw.o_proj_g128, hidden_q8_g128_.data(), hidden_scales_g128_.data(),
                hidden_.data(), config_.hidden_size, q_dim, 128);
        } else if (fused_preq) {
            fused_preq_gemv(lw.preq2_o, lw.o_proj, att_out_.data(), hidden_.data(),
                config_.hidden_size, q_dim, config_.group_size);
        } else if (config_.weight_format == 1) {
            gemv_q4_s256_preq_avx2(lw.o_proj, hidden_q8_.data(), hidden_scales_.data(), hidden_.data(),
                            config_.hidden_size, q_dim, config_.group_size);
        } else if (config_.group_size == 128) {
            gemv_q4_128_preq_avx2(lw.o_proj, hidden_q8_.data(), hidden_scales_.data(), hidden_.data(),
                            config_.hidden_size, q_dim, config_.group_size);
        } else {
            gemv_q4_32_preq_avx2(lw.o_proj, hidden_q8_.data(), hidden_scales_.data(), hidden_.data(),
                            config_.hidden_size, q_dim, config_.group_size);
        }
        if (do_prof) {
            rtp.gemv_o_ms += ms_since(t_gemv);
        }
        // Fused: vec_add(hidden, residual) + memcpy(residual, hidden) + rmsnorm_quantize
#ifdef ASDSL_PROFILE
        _ft_t = std::chrono::high_resolution_clock::now();
#endif
        t_act = do_prof ? Clock::now() : Clock::time_point{};
        if (fused_preq) {
            residual_add_rmsnorm_f32(hidden_.data(), residual_.data(), lw.rms_ffn,
                config_.hidden_size, config_.rms_norm_eps);
        } else {
            residual_add_rmsnorm_quantize_f32(hidden_.data(), residual_.data(), lw.rms_ffn,
                hidden_q8_.data(), hidden_scales_.data(), config_.hidden_size, config_.group_size, config_.rms_norm_eps);
        }
#ifdef ASDSL_PROFILE
        g_forward_prof.rmsnorm_quantize += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - _ft_t);
#endif
        if (do_prof && fused_preq) {
            rtp.prep_fused_ms += ms_since(t_act);
        } else if (do_prof) {
            rtp.activation_q8_ms += ms_since(t_act);
        }

        // Gate/Up
        t_gemv = do_prof ? Clock::now() : Clock::time_point{};
        if (draft_q2_gemv_ && lw.has_q2 && lw.q2_gate_up_proj && lw.q2_gate_up_scales && lw.q2_gate_up_biases) {
            gemv_q2_packed_impl(lw.q2_gate_up_proj, hidden_.data(), lw.q2_gate_up_scales, lw.q2_gate_up_biases,
                gu_out_.data(), 2 * config_.intermediate_size, config_.hidden_size, config_.group_size);
        } else if (lw.gate_up_q4km) {
            gemv_q4km_q8_avx2(lw.gate_up_proj, hidden_.data(), gu_out_.data(),
                2 * config_.intermediate_size, config_.hidden_size);
        } else if (c01_gemv_enabled() && lw.gate_up_g128 && lw.gate_up_proj_g128) {
            quantize_activation_avx2(
                hidden_.data(), hidden_q8_g128_.data(), hidden_scales_g128_.data(),
                config_.hidden_size, 128);
            gemv_q4_128_preq_avx2(
                lw.gate_up_proj_g128, hidden_q8_g128_.data(), hidden_scales_g128_.data(),
                gu_out_.data(), 2 * config_.intermediate_size, config_.hidden_size, 128);
        } else if (fused_preq) {
            fused_preq_gemv(lw.preq2_gate_up, lw.gate_up_proj, hidden_.data(), gu_out_.data(),
                2 * config_.intermediate_size, config_.hidden_size, config_.group_size);
        } else if (lw.gate_up_g128 && lw.gate_up_proj_g128) {
            quantize_activation_avx2(
                hidden_.data(), hidden_q8_g128_.data(), hidden_scales_g128_.data(),
                config_.hidden_size, 128);
            gemv_q4_128_preq_avx2(
                lw.gate_up_proj_g128, hidden_q8_g128_.data(), hidden_scales_g128_.data(),
                gu_out_.data(), 2 * config_.intermediate_size, config_.hidden_size, 128);
        } else if (config_.weight_format == 1) {
            gemv_q4_s256_preq_avx2(lw.gate_up_proj, hidden_q8_.data(), hidden_scales_.data(), gu_out_.data(),
                            2 * config_.intermediate_size, config_.hidden_size, config_.group_size);
        } else if (config_.group_size == 128) {
            gemv_q4_128_preq_avx2(lw.gate_up_proj, hidden_q8_.data(), hidden_scales_.data(), gu_out_.data(),
                            2 * config_.intermediate_size, config_.hidden_size, config_.group_size);
        } else {
            gemv_q4_32_preq_avx2(lw.gate_up_proj, hidden_q8_.data(), hidden_scales_.data(), gu_out_.data(),
                            2 * config_.intermediate_size, config_.hidden_size, config_.group_size);
        }
        if (do_prof) {
            rtp.gemv_gateup_ms += ms_since(t_gemv);
        }

        // Fused: swiglu + quantize (saves 1 memory pass over intermediate vector)
#ifdef ASDSL_PROFILE
        _ft_t = std::chrono::high_resolution_clock::now();
#endif
        t_act = do_prof ? Clock::now() : Clock::time_point{};
        if (fused_preq) {
            swiglu_inplace(gu_out_.data(), gu_out_.data() + config_.intermediate_size,
                config_.intermediate_size, lw.fatrelu_threshold);
        } else {
            swiglu_quantize_inplace(gu_out_.data(), gu_out_.data() + config_.intermediate_size,
                gate_q8_.data(), gate_scales_.data(), config_.intermediate_size, config_.group_size, lw.fatrelu_threshold);
        }
#ifdef ASDSL_PROFILE
        g_forward_prof.swiglu += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - _ft_t);
#endif
        if (do_prof && fused_preq) {
            rtp.prep_fused_ms += ms_since(t_act);
        } else if (do_prof) {
            rtp.activation_q8_ms += ms_since(t_act);
        }

        // Down proj
        t_gemv = do_prof ? Clock::now() : Clock::time_point{};
        if (draft_q2_gemv_ && lw.has_q2 && lw.q2_down_proj && lw.q2_down_scales && lw.q2_down_biases) {
            gemv_q2_packed_impl(lw.q2_down_proj, gu_out_.data(), lw.q2_down_scales, lw.q2_down_biases,
                hidden_.data(), config_.hidden_size, config_.intermediate_size, config_.group_size);
        } else if (lw.down_q4km) {
            gemv_q4km_q8_avx2(lw.down_proj, gu_out_.data(), hidden_.data(),
                config_.hidden_size, config_.intermediate_size);
        } else if (c01_gemv_enabled() && lw.down_g128 && lw.down_proj_g128) {
            quantize_activation_avx2(
                gu_out_.data(), gate_q8_g128_.data(), gate_scales_g128_.data(),
                config_.intermediate_size, 128);
            gemv_q4_128_preq_avx2(
                lw.down_proj_g128, gate_q8_g128_.data(), gate_scales_g128_.data(),
                hidden_.data(), config_.hidden_size, config_.intermediate_size, 128);
        } else if (fused_preq) {
            fused_preq_gemv(lw.preq2_down, lw.down_proj, gu_out_.data(), hidden_.data(),
                config_.hidden_size, config_.intermediate_size, config_.group_size);
        } else if (lw.down_g128 && lw.down_proj_g128) {
            quantize_activation_avx2(
                gu_out_.data(), gate_q8_g128_.data(), gate_scales_g128_.data(),
                config_.intermediate_size, 128);
            gemv_q4_128_preq_avx2(
                lw.down_proj_g128, gate_q8_g128_.data(), gate_scales_g128_.data(),
                hidden_.data(), config_.hidden_size, config_.intermediate_size, 128);
        } else if (config_.weight_format == 1) {
            gemv_q4_s256_preq_avx2(lw.down_proj, gate_q8_.data(), gate_scales_.data(), hidden_.data(),
                            config_.hidden_size, config_.intermediate_size, config_.group_size);
        } else if (config_.group_size == 128) {
            gemv_q4_128_preq_avx2(lw.down_proj, gate_q8_.data(), gate_scales_.data(), hidden_.data(),
                            config_.hidden_size, config_.intermediate_size, config_.group_size);
        } else {
            gemv_q4_32_preq_avx2(lw.down_proj, gate_q8_.data(), gate_scales_.data(), hidden_.data(),
                            config_.hidden_size, config_.intermediate_size, config_.group_size);
        }
        if (do_prof) {
            rtp.gemv_down_ms += ms_since(t_gemv);
        }

        vec_add_inplace(hidden_.data(), residual_.data(), config_.hidden_size);
    }

    kv_seq_len_ = std::max(kv_seq_len_, pos + 1);

    if (out_logits) {
        Clock::time_point t_lm = do_prof ? Clock::now() : Clock::time_point{};
        if (lm_head_preq2_ready_ || lm_head_q4_quantized_) {
            const int lm_gs = lm_head_group_size(config_);
            const bool lm_fused_g32 = fused_preq && lm_gs == 32;
            if (lm_fused_g32) {
                rmsnorm_f32(hidden_.data(), hidden_.data(), weights_.output_norm,
                    config_.hidden_size, config_.rms_norm_eps);
                if (preq2_gemv_enabled() && lm_head_preq2_ready_) {
                    gemv_preq2_fused_avx2(
                        lm_head_preq2_meta_.data(), lm_head_preq2_quant_.data(),
                        hidden_.data(), out_logits,
                        config_.vocab_size, config_.hidden_size, lm_gs);
                } else {
                    gemv_q4_32_preq_fused_avx2(lm_head_q4_blocks_.data(), hidden_.data(), out_logits,
                        config_.vocab_size, config_.hidden_size, lm_gs);
                }
            } else {
                rmsnorm_quantize_f32(hidden_.data(), weights_.output_norm, hidden_q8_.data(), hidden_scales_.data(),
                    config_.hidden_size, lm_gs, config_.rms_norm_eps);
                if (config_.weight_format == 1) {
                    gemv_q4_s256_preq_avx2(lm_head_q4_blocks_.data(), hidden_q8_.data(), hidden_scales_.data(),
                        out_logits, config_.vocab_size, config_.hidden_size, lm_gs);
                } else if (lm_gs == 128) {
                    gemv_q4_128_preq_avx2(lm_head_q4_blocks_.data(), hidden_q8_.data(), hidden_scales_.data(),
                        out_logits, config_.vocab_size, config_.hidden_size, 128);
                } else {
                    gemv_q4_32_preq_avx2(lm_head_q4_blocks_.data(), hidden_q8_.data(), hidden_scales_.data(),
                        out_logits, config_.vocab_size, config_.hidden_size, lm_gs);
                }
            }
        } else {
            rmsnorm_f32(hidden_.data(), hidden_.data(), weights_.output_norm, config_.hidden_size, config_.rms_norm_eps);
            gemv_f32_f32_omp(weights_.output_proj, hidden_.data(), out_logits, config_.vocab_size, config_.hidden_size);
        }
        if (do_prof) {
            rtp.lm_head_ms = ms_since(t_lm);
        }
    }

    if (do_prof) {
        prof_decode_idx++;
        const double total_ms = ms_since(t_total);
        const double gemv_ms = rtp.gemv_qkv_ms + rtp.gemv_o_ms + rtp.gemv_gateup_ms + rtp.gemv_down_ms;
        const double prep_ms = rtp.activation_q8_ms + rtp.prep_fused_ms;
        rtp.other_ms = std::max(0.0, total_ms - gemv_ms - prep_ms - rtp.lm_head_ms);
        if (prof_decode_idx == engine_profile_target_token()) {
            std::cerr << "[ASDSL_ENGINE_PROFILE] decode token " << prof_decode_idx
                      << " (ms, " << config_.num_layers << " layers):\n";
            std::cerr << "  gate_up_gemv " << rtp.gemv_gateup_ms << "\n";
            std::cerr << "  down_gemv " << rtp.gemv_down_ms << "\n";
            std::cerr << "  qkv_gemv " << rtp.gemv_qkv_ms << "\n";
            std::cerr << "  o_gemv " << rtp.gemv_o_ms << "\n";
            std::cerr << "  activation_q8 " << rtp.activation_q8_ms << "\n";
            std::cerr << "  prep_fused " << rtp.prep_fused_ms << "\n";
            std::cerr << "  lm_head " << rtp.lm_head_ms << "\n";
            std::cerr << "  other " << rtp.other_ms << "\n";
            std::cerr << "  total " << total_ms << "\n";
        }
    }
}

void UnifiedEngine::forward_token_fp32_lmhead(int token_id, int pos,
                                               const float* lm_head_fp32,
                                               float* out_logits) {
    // Run all transformer layers (populates hidden_ with the final residual).
    // Pass nullptr for out_logits to skip the internal Q4 / FP32 LM-head step.
    forward_token(token_id, pos, nullptr);

    // Apply output RMSNorm then FP32 GEMV against the caller-supplied weights.
    const int H = config_.hidden_size;
    const int V = config_.vocab_size;
    std::vector<float> normed(H);
    {
        float ss = 0.0f;
        for (int i = 0; i < H; ++i) ss += hidden_[i] * hidden_[i];
        float inv_rms = 1.0f / std::sqrt(ss / H + config_.rms_norm_eps);
        const float* w = weights_.output_norm;
        for (int i = 0; i < H; ++i) normed[i] = hidden_[i] * inv_rms * w[i];
    }
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < V; ++r) {
        const float* row = lm_head_fp32 + static_cast<size_t>(r) * H;
        float acc = 0.0f;
        for (int i = 0; i < H; ++i) acc += row[i] * normed[i];
        out_logits[r] = acc;
    }
}

std::vector<int32_t> UnifiedEngine::generate(
    const std::vector<int32_t>& prompt,
    int max_tokens,
    const std::vector<int32_t>& stop_tokens) {
    std::vector<int32_t> output = prompt;
    int current_pos = 0;

    if (!prompt.empty()) {
        const int prompt_n = (int)prompt.size();
        for (int i = 0; i < prompt_n; ++i) {
            forward_token(prompt[i], i, (i == prompt_n - 1) ? logits_.data() : nullptr);
        }
        current_pos += (int)prompt.size();
    }

#ifdef ASDSL_PROFILE
    g_forward_prof.reset();
    int profile_decode_tokens = 0;
#endif
    for (int t = 0; t < max_tokens; ++t) {

        int best_token = argmax_f32_avx2(logits_.data(), config_.vocab_size);

        output.push_back(best_token);
        bool is_stop = false;
        for (int32_t s : stop_tokens) {
            if (best_token == s) {
                is_stop = true;
                break;
            }
        }
        if (is_stop) {
            break;
        }

        forward_token(best_token, current_pos++, logits_.data());
#ifdef ASDSL_PROFILE
        profile_decode_tokens++;
#endif
    }
#ifdef ASDSL_PROFILE
    if (profile_decode_tokens > 0) {
        g_forward_prof.print(profile_decode_tokens);
    }
#endif

    return output;
}



void UnifiedEngine::forward_verify_serial(const int32_t* tokens, int n, int start_pos, float* out_logits) {
    for (int i = 0; i < n; ++i) {
        forward_token(tokens[i], start_pos + i, out_logits ? out_logits + static_cast<size_t>(i) * config_.vocab_size : nullptr);
    }
}

void UnifiedEngine::forward_verify_batch(const int32_t* tokens, int n, int start_pos, float* out_logits) {
    if (n <= 0) {
        return;
    }
    if (n == 1) {
        forward_token(tokens[0], start_pos, out_logits);
        return;
    }
    // Batched legacy GEMM does not dispatch preq2 fused weights; serial verify matches
    // forward_token numerics (required for lossless PLD under ASDSL_PREQ2=1).
    if (preq2_gemv_enabled() && use_q4_32_preq_fused(config_)) {
        forward_verify_serial(tokens, n, start_pos, out_logits);
        return;
    }
    forward_batch(tokens, n, start_pos, out_logits, true);
}
void UnifiedEngine::forward_token_draft(int token_id, int pos, float* out_logits) {
    draft_skip_layers_ = true;
    forward_token(token_id, pos, out_logits);
    draft_skip_layers_ = false;
}

void UnifiedEngine::forward_token_draft_q2(int token_id, int pos, float* out_logits) {
    draft_skip_layers_ = true;
    draft_q2_gemv_ = true;
    forward_token(token_id, pos, out_logits);
    draft_q2_gemv_ = false;
    draft_skip_layers_ = false;
}

namespace {
inline bool is_phi_eos_token(int32_t token) {
    return token == 199999 || token == 200020;
}
} // namespace

std::vector<int32_t> UnifiedEngine::generate_swift(const std::vector<int32_t>& prompt, int max_tokens, int draft_k) {
    std::vector<int32_t> output = prompt;
    int current_pos = 0;

    if (!prompt.empty()) {
        const int prompt_n = (int)prompt.size();
        for (int i = 0; i < prompt_n; ++i) {
            forward_token(prompt[i], i, (i == prompt_n - 1) ? logits_.data() : nullptr);
        }
        current_pos = prompt_n;
    }

    const int prompt_len = static_cast<int>(prompt.size());
    std::vector<float> verify_logits(static_cast<size_t>(draft_k + 1) * config_.vocab_size);

    while (static_cast<int>(output.size()) - prompt_len < max_tokens) {
        const int32_t current_token = static_cast<int32_t>(
            argmax_f32_avx2(logits_.data(), config_.vocab_size));
        if (is_phi_eos_token(current_token)) {
            break;
        }

        const int draft_start_pos = current_pos;

        snapshot_kv();
        std::vector<int32_t> draft_tokens;
        draft_tokens.reserve(static_cast<size_t>(draft_k));
        int32_t draft_tok = current_token;
        for (int k = 0; k < draft_k; ++k) {
            forward_token_draft(draft_tok, draft_start_pos + k, logits_.data());
            const int32_t next_draft = static_cast<int32_t>(
                argmax_f32_avx2(logits_.data(), config_.vocab_size));
            draft_tokens.push_back(next_draft);
            draft_tok = next_draft;
            if (is_phi_eos_token(next_draft)) {
                break;
            }
        }
        restore_kv();

        const int L = static_cast<int>(draft_tokens.size());
        std::vector<int32_t> verify_tokens;
        verify_tokens.reserve(static_cast<size_t>(std::max(L, 1)));
        if (L == 0) {
            verify_tokens.push_back(current_token);
        } else {
            verify_tokens.push_back(current_token);
            for (int i = 0; i < L - 1; ++i) {
                verify_tokens.push_back(draft_tokens[static_cast<size_t>(i)]);
            }
        }
        const int n_verify = static_cast<int>(verify_tokens.size());
        forward_verify_serial(
            verify_tokens.data(), n_verify, draft_start_pos, verify_logits.data());

        output.push_back(current_token);

        int accepted = 0;
        int32_t correction = -1;
        for (int k_idx = 0; k_idx < L; ++k_idx) {
            const int32_t ref_tok = static_cast<int32_t>(argmax_f32_avx2(
                verify_logits.data() + static_cast<size_t>(k_idx) * config_.vocab_size,
                config_.vocab_size));
            if (ref_tok == draft_tokens[static_cast<size_t>(k_idx)]) {
                accepted++;
            } else {
                correction = ref_tok;
                break;
            }
        }

        bool stop = is_phi_eos_token(current_token);
        for (int i = 0; i < accepted; ++i) {
            output.push_back(draft_tokens[static_cast<size_t>(i)]);
            if (is_phi_eos_token(draft_tokens[static_cast<size_t>(i)])) {
                stop = true;
                break;
            }
        }
        if (stop) {
            current_pos = draft_start_pos + 1 + accepted;
            truncate_kv(current_pos);
            break;
        }

        if (correction >= 0) {
            const int n_keep = 1 + accepted;
            current_pos = draft_start_pos + n_keep;
            truncate_kv(current_pos);
            forward_token(correction, current_pos, logits_.data());
            current_pos++;
            output.push_back(correction);
            if (is_phi_eos_token(correction)) {
                break;
            }
        } else if (L > 0 && accepted == L) {
            current_pos = draft_start_pos + n_verify;
            truncate_kv(current_pos);
            forward_token(draft_tokens.back(), current_pos, logits_.data());
            current_pos++;
            if (is_phi_eos_token(draft_tokens.back())) {
                break;
            }
        } else if (L == 0) {
            forward_token(current_token, draft_start_pos, logits_.data());
            current_pos = draft_start_pos + 1;
        }

        if (static_cast<int>(output.size()) - prompt_len >= max_tokens) {
            break;
        }
    }

    return output;
}


std::vector<int32_t> UnifiedEngine::generate_ahsd(
    const std::vector<int32_t>& prompt, int max_tokens, int draft_k, AhsdStats* stats_out) {
    AhsdStats local_stats;
    std::vector<int32_t> output = prompt;
    int current_pos = 0;

    if (!prompt.empty()) {
        const int prompt_n = (int)prompt.size();
        for (int i = 0; i < prompt_n; ++i) {
            forward_token(prompt[i], i, (i == prompt_n - 1) ? logits_.data() : nullptr);
        }
        current_pos = prompt_n;
    }

    const int prompt_len = static_cast<int>(prompt.size());
    std::vector<float> verify_logits(static_cast<size_t>(draft_k + 1) * config_.vocab_size);

    while (static_cast<int>(output.size()) - prompt_len < max_tokens) {
        const int32_t current_token = static_cast<int32_t>(
            argmax_f32_avx2(logits_.data(), config_.vocab_size));
        if (is_phi_eos_token(current_token)) {
            break;
        }

        const int draft_start_pos = current_pos;

        const auto t_draft0 = Clock::now();
        snapshot_kv();
        std::vector<int32_t> draft_tokens;
        draft_tokens.reserve(static_cast<size_t>(draft_k));
        int32_t draft_tok = current_token;
        for (int k = 0; k < draft_k; ++k) {
            forward_token_draft(draft_tok, draft_start_pos + k, logits_.data());
            const int32_t next_draft = static_cast<int32_t>(
                argmax_f32_avx2(logits_.data(), config_.vocab_size));
            draft_tokens.push_back(next_draft);
            draft_tok = next_draft;
            if (is_phi_eos_token(next_draft)) {
                break;
            }
        }
        restore_kv();
        local_stats.draft_ms += ms_since(t_draft0);
        local_stats.draft_tokens += static_cast<int>(draft_tokens.size());
        local_stats.speculative_cycles++;

        const int L = static_cast<int>(draft_tokens.size());
        std::vector<int32_t> verify_tokens;
        verify_tokens.reserve(static_cast<size_t>(std::max(L, 1)));
        if (L == 0) {
            verify_tokens.push_back(current_token);
        } else {
            verify_tokens.push_back(current_token);
            for (int i = 0; i < L - 1; ++i) {
                verify_tokens.push_back(draft_tokens[static_cast<size_t>(i)]);
            }
        }
        const int n_verify = static_cast<int>(verify_tokens.size());

        const auto t_verify0 = Clock::now();
        forward_verify_serial(
            verify_tokens.data(), n_verify, draft_start_pos, verify_logits.data());
        local_stats.verify_ms += ms_since(t_verify0);

        output.push_back(current_token);

        int accepted = 0;
        int32_t correction = -1;
        for (int k_idx = 0; k_idx < L; ++k_idx) {
            const int32_t ref_tok = static_cast<int32_t>(argmax_f32_avx2(
                verify_logits.data() + static_cast<size_t>(k_idx) * config_.vocab_size,
                config_.vocab_size));
            if (ref_tok == draft_tokens[static_cast<size_t>(k_idx)]) {
                accepted++;
            } else {
                correction = ref_tok;
                break;
            }
        }
        local_stats.accepted_tokens += accepted;

        bool stop = is_phi_eos_token(current_token);
        for (int i = 0; i < accepted; ++i) {
            output.push_back(draft_tokens[static_cast<size_t>(i)]);
            if (is_phi_eos_token(draft_tokens[static_cast<size_t>(i)])) {
                stop = true;
                break;
            }
        }
        if (stop) {
            current_pos = draft_start_pos + 1 + accepted;
            truncate_kv(current_pos);
            break;
        }

        if (correction >= 0) {
            const int n_keep = 1 + accepted;
            current_pos = draft_start_pos + n_keep;
            truncate_kv(current_pos);
            forward_token(correction, current_pos, logits_.data());
            current_pos++;
            output.push_back(correction);
            if (is_phi_eos_token(correction)) {
                break;
            }
        } else if (L > 0 && accepted == L) {
            current_pos = draft_start_pos + n_verify;
            truncate_kv(current_pos);
            forward_token(draft_tokens.back(), current_pos, logits_.data());
            current_pos++;
            if (is_phi_eos_token(draft_tokens.back())) {
                break;
            }
        } else if (L == 0) {
            forward_token(current_token, draft_start_pos, logits_.data());
            current_pos = draft_start_pos + 1;
        }

        if (static_cast<int>(output.size()) - prompt_len >= max_tokens) {
            break;
        }
    }

    if (local_stats.draft_tokens > 0) {
        local_stats.acceptance_rate = static_cast<double>(local_stats.accepted_tokens)
            / static_cast<double>(local_stats.draft_tokens);
    }
    if (stats_out) {
        *stats_out = local_stats;
    }
    return output;
}

std::vector<int32_t> UnifiedEngine::generate_sdqs(
    const std::vector<int32_t>& prompt, int max_tokens, int draft_k, AhsdStats* stats_out) {
    AhsdStats local_stats;
    std::vector<int32_t> output = prompt;
    int current_pos = 0;

    if (!prompt.empty()) {
        const int prompt_n = (int)prompt.size();
        for (int i = 0; i < prompt_n; ++i) {
            forward_token(prompt[i], i, (i == prompt_n - 1) ? logits_.data() : nullptr);
        }
        current_pos = prompt_n;
    }

    const int prompt_len = static_cast<int>(prompt.size());
    std::vector<float> verify_logits(static_cast<size_t>(draft_k + 1) * config_.vocab_size);

    while (static_cast<int>(output.size()) - prompt_len < max_tokens) {
        const int32_t current_token = static_cast<int32_t>(
            argmax_f32_avx2(logits_.data(), config_.vocab_size));
        if (is_phi_eos_token(current_token)) {
            break;
        }

        const int draft_start_pos = current_pos;
        std::vector<int32_t> draft_tokens;
        draft_tokens.reserve(static_cast<size_t>(draft_k));

        const auto t_draft0 = Clock::now();
        snapshot_kv();
        int32_t draft_tok = current_token;
        for (int k = 0; k < draft_k; ++k) {
            forward_token_draft_q2(draft_tok, draft_start_pos + k, logits_.data());
            const int32_t next_draft = static_cast<int32_t>(
                argmax_f32_avx2(logits_.data(), config_.vocab_size));
            draft_tokens.push_back(next_draft);
            draft_tok = next_draft;
            if (is_phi_eos_token(next_draft)) {
                break;
            }
        }
        restore_kv();
        local_stats.draft_ms += ms_since(t_draft0);
        local_stats.draft_tokens += static_cast<int>(draft_tokens.size());
        local_stats.speculative_cycles++;

        const int L = static_cast<int>(draft_tokens.size());
        std::vector<int32_t> verify_tokens;
        verify_tokens.reserve(static_cast<size_t>(std::max(L, 1)));
        if (L == 0) {
            verify_tokens.push_back(current_token);
        } else {
            verify_tokens.push_back(current_token);
            for (int i = 0; i < L - 1; ++i) {
                verify_tokens.push_back(draft_tokens[static_cast<size_t>(i)]);
            }
        }
        const int n_verify = static_cast<int>(verify_tokens.size());

        const auto t_verify0 = Clock::now();
        forward_verify_serial(
            verify_tokens.data(), n_verify, draft_start_pos, verify_logits.data());
        local_stats.verify_ms += ms_since(t_verify0);

        output.push_back(current_token);

        int accepted = 0;
        int32_t correction = -1;
        for (int k_idx = 0; k_idx < L; ++k_idx) {
            const int32_t ref_tok = static_cast<int32_t>(argmax_f32_avx2(
                verify_logits.data() + static_cast<size_t>(k_idx) * config_.vocab_size,
                config_.vocab_size));
            if (ref_tok == draft_tokens[static_cast<size_t>(k_idx)]) {
                accepted++;
            } else {
                correction = ref_tok;
                break;
            }
        }
        local_stats.accepted_tokens += accepted;

        bool stop = is_phi_eos_token(current_token);
        for (int i = 0; i < accepted; ++i) {
            output.push_back(draft_tokens[static_cast<size_t>(i)]);
            if (is_phi_eos_token(draft_tokens[static_cast<size_t>(i)])) {
                stop = true;
                break;
            }
        }
        if (stop) {
            current_pos = draft_start_pos + 1 + accepted;
            truncate_kv(current_pos);
            break;
        }

        if (correction >= 0) {
            const int n_keep = 1 + accepted;
            current_pos = draft_start_pos + n_keep;
            truncate_kv(current_pos);
            forward_token(correction, current_pos, logits_.data());
            current_pos++;
            output.push_back(correction);
            if (is_phi_eos_token(correction)) {
                break;
            }
        } else if (L > 0 && accepted == L) {
            current_pos = draft_start_pos + n_verify;
            truncate_kv(current_pos);
            forward_token(draft_tokens.back(), current_pos, logits_.data());
            current_pos++;
            if (is_phi_eos_token(draft_tokens.back())) {
                break;
            }
        } else if (L == 0) {
            forward_token(current_token, draft_start_pos, logits_.data());
            current_pos = draft_start_pos + 1;
        }

        if (static_cast<int>(output.size()) - prompt_len >= max_tokens) {
            break;
        }
    }

    if (local_stats.draft_tokens > 0) {
        local_stats.acceptance_rate = static_cast<double>(local_stats.accepted_tokens)
            / static_cast<double>(local_stats.draft_tokens);
    }
    if (stats_out) {
        *stats_out = local_stats;
    }
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
        .def_readwrite("lm_head_group_size", &asdsl::EngineConfig::lm_head_group_size)
        .def_readwrite("max_seq_len", &asdsl::EngineConfig::max_seq_len)
        .def_readwrite("rotary_dim", &asdsl::EngineConfig::rotary_dim)
        .def_readwrite("weight_format", &asdsl::EngineConfig::weight_format);

    return cls;
}

// A wrapper to safely initialize the C++ engine from numpy arrays:
class UnifiedEnginePy {
    std::unique_ptr<UnifiedEngine> engine_;
    EngineWeights weights_;

    // We hold references to py::array so memory isn't GC'd
    std::vector<py::array> keep_alive_;
    // Reused decode logits buffer (avoids ~vocab*4 B alloc per token).
    py::array_t<float> logits_out_;

    template<typename T>
    const T* get_ptr(py::array_t<T>& arr) {
        if (arr.size() == 0) return nullptr;
        keep_alive_.push_back(arr);
        return arr.data();
    }

    static void bind_embedding(py::array arr, asdsl::EngineWeights& weights, std::vector<py::array>& keep_alive) {
        keep_alive.push_back(arr);
        const auto req = arr.request();
        const std::string dtype = py::str(arr.attr("dtype"));
        if (dtype == "float16" || dtype == "<f2") {
            weights.token_embd_f16 = static_cast<const uint16_t*>(req.ptr);
            weights.embed_fp16 = true;
            weights.token_embd = nullptr;
            return;
        }
        weights.token_embd = static_cast<const float*>(req.ptr);
        weights.embed_fp16 = false;
        weights.token_embd_f16 = nullptr;
    }

    static void bind_output_proj(py::array arr, asdsl::EngineWeights& weights, std::vector<py::array>& keep_alive) {
        keep_alive.push_back(arr);
        const auto req = arr.request();
        const std::string dtype = py::str(arr.attr("dtype"));
        if (dtype == "float16" || dtype == "<f2") {
            weights.output_proj_f16 = static_cast<const uint16_t*>(req.ptr);
            weights.output_proj_fp16 = true;
            weights.output_proj = nullptr;
            return;
        }
        weights.output_proj = static_cast<const float*>(req.ptr);
        weights.output_proj_fp16 = false;
        weights.output_proj_f16 = nullptr;
    }

public:
    UnifiedEnginePy(
        EngineConfig config,
        py::array token_embd,
        py::array_t<float> output_norm,
        py::object output_proj,
        py::array_t<float> cos_table,
        py::array_t<float> sin_table,
        py::dict layers_dict,
        py::object lm_head_preq2_meta = py::none(),
        py::object lm_head_preq2_quant = py::none()
    ) {
        bind_embedding(token_embd, weights_, keep_alive_);
        weights_.output_norm = get_ptr(output_norm);
        if (!lm_head_preq2_meta.is_none() && !lm_head_preq2_quant.is_none()) {
            auto meta_arr = lm_head_preq2_meta.cast<py::array_t<uint8_t>>();
            auto quant_arr = lm_head_preq2_quant.cast<py::array_t<uint8_t>>();
            weights_.lm_head_preq2_meta_in = get_ptr(meta_arr);
            weights_.lm_head_preq2_quant_in = get_ptr(quant_arr);
            weights_.lm_head_preq2_meta_size = static_cast<size_t>(meta_arr.size());
            weights_.lm_head_preq2_quant_size = static_cast<size_t>(quant_arr.size());
            weights_.lm_head_preq2_from_cache = true;
        }
        if (!output_proj.is_none()) {
            bind_output_proj(output_proj.cast<py::array>(), weights_, keep_alive_);
        }
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

            if (l_dict.contains("gate_up_proj_g128")) {
                auto gu128 = l_dict["gate_up_proj_g128"].cast<py::array_t<uint8_t>>();
                lw.gate_up_proj_g128 = get_ptr(gu128);
                lw.gate_up_g128 = true;
            }
            if (l_dict.contains("down_proj_g128")) {
                auto dn128 = l_dict["down_proj_g128"].cast<py::array_t<uint8_t>>();
                lw.down_proj_g128 = get_ptr(dn128);
                lw.down_g128 = true;
            }
            if (l_dict.contains("qkv_proj_g128")) {
                auto qkv128 = l_dict["qkv_proj_g128"].cast<py::array_t<uint8_t>>();
                lw.qkv_proj_g128 = get_ptr(qkv128);
                lw.qkv_g128 = true;
            }
            if (l_dict.contains("o_proj_g128")) {
                auto o128 = l_dict["o_proj_g128"].cast<py::array_t<uint8_t>>();
                lw.o_proj_g128 = get_ptr(o128);
                lw.o_g128 = true;
            }

            if (l_dict.contains("fatrelu_threshold")) {
                lw.fatrelu_threshold = l_dict["fatrelu_threshold"].cast<float>();
            }
            if (l_dict.contains("qkv_q4km")) {
                lw.qkv_q4km = l_dict["qkv_q4km"].cast<bool>();
            }
            if (l_dict.contains("o_q4km")) {
                lw.o_q4km = l_dict["o_q4km"].cast<bool>();
            }
            if (l_dict.contains("gate_up_q4km")) {
                lw.gate_up_q4km = l_dict["gate_up_q4km"].cast<bool>();
            }
            if (l_dict.contains("down_q4km")) {
                lw.down_q4km = l_dict["down_q4km"].cast<bool>();
            }
            if (l_dict.contains("qkv_q5km")) {
                lw.qkv_q5km = l_dict["qkv_q5km"].cast<bool>();
            }

            auto bind_preq2 = [&](const char* prefix, asdsl::Preq2Weights& dst) {
                const py::str meta_key = py::str(std::string(prefix) + "_meta");
                const py::str quant_key = py::str(std::string(prefix) + "_quant");
                if (l_dict.contains(meta_key)) {
                    auto arr = l_dict[meta_key].cast<py::array_t<uint8_t>>();
                    dst.meta = get_ptr(arr);
                }
                if (l_dict.contains(quant_key)) {
                    auto arr = l_dict[quant_key].cast<py::array_t<uint8_t>>();
                    dst.quant = get_ptr(arr);
                }
            };
            bind_preq2("qkv_proj", lw.preq2_qkv);
            bind_preq2("o_proj", lw.preq2_o);
            bind_preq2("gate_up_proj", lw.preq2_gate_up);
            bind_preq2("down_proj", lw.preq2_down);

            if (l_dict.contains("q2_qkv_proj")) {
                auto arr = l_dict["q2_qkv_proj"].cast<py::array_t<uint8_t>>();
                lw.q2_qkv_proj = get_ptr(arr);
            }
            if (l_dict.contains("q2_o_proj")) {
                auto arr = l_dict["q2_o_proj"].cast<py::array_t<uint8_t>>();
                lw.q2_o_proj = get_ptr(arr);
            }
            if (l_dict.contains("q2_gate_up_proj")) {
                auto arr = l_dict["q2_gate_up_proj"].cast<py::array_t<uint8_t>>();
                lw.q2_gate_up_proj = get_ptr(arr);
            }
            if (l_dict.contains("q2_down_proj")) {
                auto arr = l_dict["q2_down_proj"].cast<py::array_t<uint8_t>>();
                lw.q2_down_proj = get_ptr(arr);
            }
            if (l_dict.contains("q2_qkv_scales")) {
                auto arr = l_dict["q2_qkv_scales"].cast<py::array_t<float>>();
                lw.q2_qkv_scales = get_ptr(arr);
            }
            if (l_dict.contains("q2_qkv_biases")) {
                auto arr = l_dict["q2_qkv_biases"].cast<py::array_t<float>>();
                lw.q2_qkv_biases = get_ptr(arr);
            }
            if (l_dict.contains("q2_o_scales")) {
                auto arr = l_dict["q2_o_scales"].cast<py::array_t<float>>();
                lw.q2_o_scales = get_ptr(arr);
            }
            if (l_dict.contains("q2_o_biases")) {
                auto arr = l_dict["q2_o_biases"].cast<py::array_t<float>>();
                lw.q2_o_biases = get_ptr(arr);
            }
            if (l_dict.contains("q2_gate_up_scales")) {
                auto arr = l_dict["q2_gate_up_scales"].cast<py::array_t<float>>();
                lw.q2_gate_up_scales = get_ptr(arr);
            }
            if (l_dict.contains("q2_gate_up_biases")) {
                auto arr = l_dict["q2_gate_up_biases"].cast<py::array_t<float>>();
                lw.q2_gate_up_biases = get_ptr(arr);
            }
            if (l_dict.contains("q2_down_scales")) {
                auto arr = l_dict["q2_down_scales"].cast<py::array_t<float>>();
                lw.q2_down_scales = get_ptr(arr);
            }
            if (l_dict.contains("q2_down_biases")) {
                auto arr = l_dict["q2_down_biases"].cast<py::array_t<float>>();
                lw.q2_down_biases = get_ptr(arr);
            }
            if (l_dict.contains("has_q2")) {
                lw.has_q2 = l_dict["has_q2"].cast<bool>();
            }
            if (l_dict.contains("down_q6km")) {
                lw.down_q6km = l_dict["down_q6km"].cast<bool>();
            }

            weights_.layers[layer_idx] = lw;
        }

        engine_ = std::make_unique<UnifiedEngine>(config, weights_);
        logits_out_ = py::array_t<float>(static_cast<py::ssize_t>(config.vocab_size));
    }

    std::vector<int32_t> generate(std::vector<int32_t> prompt, int max_tokens) {
        // Release GIL during generation process
        py::gil_scoped_release release;
        return engine_->generate(prompt, max_tokens);
    }

    std::vector<int32_t> generate_with_stops(
            std::vector<int32_t> prompt,
            int max_tokens,
            std::vector<int32_t> stop_tokens) {
        py::gil_scoped_release release;
        return engine_->generate(prompt, max_tokens, stop_tokens);
    }

    std::vector<int32_t> generate_swift(std::vector<int32_t> prompt, int max_tokens, int draft_k) {
        py::gil_scoped_release release;
        return engine_->generate_swift(prompt, max_tokens, draft_k);
    }

    void reset_session() {
        engine_->reset_session();
    }

    py::tuple export_lm_head_preq2() {
        if (!engine_->lm_head_preq2_ready()) {
            throw std::runtime_error("export_lm_head_preq2: lm_head preq2 not ready");
        }
        const auto& meta = engine_->lm_head_preq2_meta();
        const auto& quant = engine_->lm_head_preq2_quant();
        py::array_t<uint8_t> meta_arr(static_cast<py::ssize_t>(meta.size()));
        py::array_t<uint8_t> quant_arr(static_cast<py::ssize_t>(quant.size()));
        std::memcpy(meta_arr.mutable_data(), meta.data(), meta.size());
        std::memcpy(quant_arr.mutable_data(), quant.data(), quant.size());
        return py::make_tuple(meta_arr, quant_arr);
    }

    py::array_t<float> forward_token(int token, int pos) {
        float* ptr = static_cast<float*>(logits_out_.request().ptr);
        {
            py::gil_scoped_release release;
            engine_->forward_token(token, pos, ptr);
        }
        return logits_out_;
    }

    void forward_token_prefill(int token, int pos) {
        py::gil_scoped_release release;
        engine_->forward_token(token, pos, nullptr);
    }

    int forward_token_argmax(int token, int pos) {
        float* ptr = static_cast<float*>(logits_out_.request().ptr);
        {
            py::gil_scoped_release release;
            engine_->forward_token(token, pos, ptr);
            return argmax_f32_avx2(ptr, engine_->config_.vocab_size);
        }
    }

    py::array_t<float> forward_token_fp32_lmhead(
            int token, int pos,
            py::array_t<float, py::array::c_style | py::array::forcecast> lm_head_fp32) {
        const int V = engine_->config_.vocab_size;
        const int H = engine_->config_.hidden_size;
        auto lmb = lm_head_fp32.request();
        if (lmb.size != static_cast<py::ssize_t>(V) * H)
            throw std::invalid_argument("forward_token_fp32_lmhead: lm_head must be (vocab_size, hidden_size)");
        const float* lm_ptr = static_cast<const float*>(lmb.ptr);
        py::array_t<float> result(V);
        float* out = static_cast<float*>(result.request().ptr);
        {
            py::gil_scoped_release release;
            engine_->forward_token_fp32_lmhead(token, pos, lm_ptr, out);
        }
        return result;
    }
    
    py::array_t<float> forward_batch(std::vector<int32_t> prompt, int start_pos) {
        py::array_t<float> result(engine_->config_.vocab_size);
        float* ptr = static_cast<float*>(result.request().ptr);
        {
            py::gil_scoped_release release;
            engine_->forward_batch(prompt.data(), (int)prompt.size(), start_pos, ptr);
        }
        return result;
    }

    std::vector<int32_t> generate_ahsd(std::vector<int32_t> prompt, int max_tokens, int draft_k) {
        py::gil_scoped_release release;
        return engine_->generate_ahsd(prompt, max_tokens, draft_k, nullptr);
    }

    py::dict generate_ahsd_stats(std::vector<int32_t> prompt, int max_tokens, int draft_k) {
        asdsl::UnifiedEngine::AhsdStats stats;
        std::vector<int32_t> tokens;
        {
            py::gil_scoped_release release;
            tokens = engine_->generate_ahsd(prompt, max_tokens, draft_k, &stats);
        }
        py::dict out;
        out["tokens"] = tokens;
        out["acceptance_rate"] = stats.acceptance_rate;
        out["draft_tokens"] = stats.draft_tokens;
        out["accepted_tokens"] = stats.accepted_tokens;
        out["speculative_cycles"] = stats.speculative_cycles;
        out["draft_ms"] = stats.draft_ms;
        out["verify_ms"] = stats.verify_ms;
        return out;
    }

    std::vector<int32_t> generate_sdqs(std::vector<int32_t> prompt, int max_tokens, int draft_k) {
        py::gil_scoped_release release;
        return engine_->generate_sdqs(prompt, max_tokens, draft_k, nullptr);
    }

    py::dict generate_sdqs_stats(std::vector<int32_t> prompt, int max_tokens, int draft_k) {
        asdsl::UnifiedEngine::AhsdStats stats;
        std::vector<int32_t> tokens;
        {
            py::gil_scoped_release release;
            tokens = engine_->generate_sdqs(prompt, max_tokens, draft_k, &stats);
        }
        py::dict out;
        out["tokens"] = tokens;
        out["acceptance_rate"] = stats.acceptance_rate;
        out["draft_tokens"] = stats.draft_tokens;
        out["accepted_tokens"] = stats.accepted_tokens;
        out["speculative_cycles"] = stats.speculative_cycles;
        out["draft_ms"] = stats.draft_ms;
        out["verify_ms"] = stats.verify_ms;
        return out;
    }

    py::array_t<float> forward_token_draft(int token, int pos) {
        float* ptr = static_cast<float*>(logits_out_.request().ptr);
        {
            py::gil_scoped_release release;
            engine_->forward_token_draft(token, pos, ptr);
        }
        return logits_out_;
    }

    py::array_t<float> forward_batch_all_logits(std::vector<int32_t> prompt, int start_pos) {
        const int n = static_cast<int>(prompt.size());
        py::array_t<float> result(static_cast<py::ssize_t>(n) * engine_->config_.vocab_size);
        float* ptr = static_cast<float*>(result.request().ptr);
        {
            py::gil_scoped_release release;
            engine_->forward_verify_batch(prompt.data(), n, start_pos, ptr);
        }
        return result;
    }

    py::array_t<float> forward_verify_serial_all_logits(std::vector<int32_t> prompt, int start_pos) {
        const int n = static_cast<int>(prompt.size());
        py::array_t<float> result(static_cast<py::ssize_t>(n) * engine_->config_.vocab_size);
        float* ptr = static_cast<float*>(result.request().ptr);
        {
            py::gil_scoped_release release;
            engine_->forward_verify_serial(prompt.data(), n, start_pos, ptr);
        }
        return result;
    }

    py::array_t<float> forward_verify_batch_raw_all_logits(std::vector<int32_t> prompt, int start_pos) {
        const int n = static_cast<int>(prompt.size());
        py::array_t<float> result(static_cast<py::ssize_t>(n) * engine_->config_.vocab_size);
        float* ptr = static_cast<float*>(result.request().ptr);
        {
            py::gil_scoped_release release;
            if (n <= 1) {
                engine_->forward_token(prompt[0], start_pos, ptr);
            } else {
                engine_->forward_batch(prompt.data(), n, start_pos, ptr, true);
            }
        }
        return result;
    }

    int get_kv_seq_len() const { return engine_->get_kv_seq_len(); }

    void snapshot_kv() { engine_->snapshot_kv(); }

    void restore_kv() { engine_->restore_kv(); }

    void truncate_kv(int new_len) { engine_->truncate_kv(new_len); }

    void set_skip_mask(py::array_t<bool> mask) {
        auto req = mask.request();
        engine_->set_skip_mask(static_cast<const bool*>(req.ptr), static_cast<int>(req.size));
    }

    void clear_skip_mask() { engine_->clear_skip_mask(); }

};

}

PYBIND11_MODULE(_native_unified, m) {
    asdsl::register_config(m);

    // CPU topology helpers — safe to call before any OpenMP region is entered.
    // Returns logical CPU IDs for P-cores only (one per physical core, skips HT sibling).
    m.def("get_pcore_logical_ids", []() {
        return asdsl::ThreadPool::get_pcore_logical_ids();
    }, "One logical ID per physical P-core (skips HyperThreading sibling)");

    // Returns ALL logical CPU IDs for P-cores, including HyperThreading siblings.
    // Use this for OMP_PLACES to fully populate P-core execution ports.
    m.def("get_all_pcore_logical_ids", []() {
        return asdsl::ThreadPool::get_all_pcore_logical_ids();
    }, "All logical CPU IDs for P-cores including HyperThreading siblings");

    py::class_<asdsl::UnifiedEnginePy>(m, "UnifiedEngine")
        .def(py::init<
            asdsl::EngineConfig,
            py::array,
            py::array_t<float>,
            py::object,
            py::array_t<float>,
            py::array_t<float>,
            py::dict,
            py::object,
            py::object
        >(),
        py::arg("config"),
        py::arg("token_embd"),
        py::arg("output_norm"),
        py::arg("output_proj"),
        py::arg("cos_table"),
        py::arg("sin_table"),
        py::arg("layers_dict"),
        py::arg("lm_head_preq2_meta") = py::none(),
        py::arg("lm_head_preq2_quant") = py::none())
        .def("export_lm_head_preq2", &asdsl::UnifiedEnginePy::export_lm_head_preq2,
             "Copy lm_head preq2 meta+quant blobs for disk cache.")
        .def("generate", &asdsl::UnifiedEnginePy::generate)
        .def("generate_with_stops", &asdsl::UnifiedEnginePy::generate_with_stops,
             py::arg("prompt"), py::arg("max_tokens"), py::arg("stop_tokens"))
        .def("generate_swift", &asdsl::UnifiedEnginePy::generate_swift)
        .def("forward_token", &asdsl::UnifiedEnginePy::forward_token)
        .def("forward_token_prefill", &asdsl::UnifiedEnginePy::forward_token_prefill,
             "Forward one token updating KV; skip lm_head (prefill body).")
        .def("forward_token_argmax", &asdsl::UnifiedEnginePy::forward_token_argmax,
             "Single decode step; returns argmax token id (reuses internal logits buffer).")
        .def("forward_token_fp32_lmhead", &asdsl::UnifiedEnginePy::forward_token_fp32_lmhead,
             py::arg("token"), py::arg("pos"), py::arg("lm_head_fp32"))
        .def("forward_batch", &asdsl::UnifiedEnginePy::forward_batch)
        .def("generate_ahsd", &asdsl::UnifiedEnginePy::generate_ahsd)
        .def("generate_ahsd_stats", &asdsl::UnifiedEnginePy::generate_ahsd_stats)
        .def("generate_sdqs", &asdsl::UnifiedEnginePy::generate_sdqs)
        .def("generate_sdqs_stats", &asdsl::UnifiedEnginePy::generate_sdqs_stats)
        .def("forward_token_draft", &asdsl::UnifiedEnginePy::forward_token_draft)
        .def("forward_batch_all_logits", &asdsl::UnifiedEnginePy::forward_batch_all_logits)
        .def("forward_verify_serial_all_logits", &asdsl::UnifiedEnginePy::forward_verify_serial_all_logits,
             "Per-token verify oracle (lossless under preq2).")
        .def("forward_verify_batch_raw_all_logits", &asdsl::UnifiedEnginePy::forward_verify_batch_raw_all_logits,
             "Batched GEMM verify (may diverge from forward_token when preq2 is on).")
        .def("get_kv_seq_len", &asdsl::UnifiedEnginePy::get_kv_seq_len)
        .def("snapshot_kv", &asdsl::UnifiedEnginePy::snapshot_kv)
        .def("restore_kv", &asdsl::UnifiedEnginePy::restore_kv)
        .def("truncate_kv", &asdsl::UnifiedEnginePy::truncate_kv)
        .def("set_skip_mask", &asdsl::UnifiedEnginePy::set_skip_mask)
        .def("clear_skip_mask", &asdsl::UnifiedEnginePy::clear_skip_mask)
        .def("reset_session", &asdsl::UnifiedEnginePy::reset_session);
}
