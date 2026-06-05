#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <immintrin.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include "native/omp_pcore_pinning.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <limits>
#include <cmath>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <random>

namespace py = pybind11;

#pragma pack(push, 1)
struct BlockQ4K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[6];
    uint8_t padding[2];
    uint8_t qs[128];
};
#pragma pack(pop)

static inline float f16_to_f32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x03FFu;

    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            uint32_t m = mant;
            int e = -1;
            do {
                ++e;
                m <<= 1;
            } while ((m & 0x0400u) == 0);
            m &= 0x03FFu;
            const uint32_t exp32 = static_cast<uint32_t>(127 - 15 - e);
            f = sign | (exp32 << 23) | (m << 13);
        }
    } else if (exp == 0x1Fu) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        const uint32_t exp32 = exp + (127 - 15);
        f = sign | (exp32 << 23) | (mant << 13);
    }

    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

static inline void unpack_scales_8x6(const uint8_t* src, uint8_t out[8]) {
    out[0] = src[0] & 0x3F;
    out[1] = static_cast<uint8_t>(((src[0] >> 6) & 0x03) | ((src[1] & 0x0F) << 2));
    out[2] = static_cast<uint8_t>(((src[1] >> 4) & 0x0F) | ((src[2] & 0x03) << 4));
    out[3] = static_cast<uint8_t>((src[2] >> 2) & 0x3F);
    out[4] = src[3] & 0x3F;
    out[5] = static_cast<uint8_t>(((src[3] >> 6) & 0x03) | ((src[4] & 0x0F) << 2));
    out[6] = static_cast<uint8_t>(((src[4] >> 4) & 0x0F) | ((src[5] & 0x03) << 4));
    out[7] = static_cast<uint8_t>((src[5] >> 2) & 0x3F);
}

static inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

static constexpr int TILE_M = 4;
static constexpr int TILE_N = 32;
static constexpr int TILE_K = 256;
static constexpr int INTERLEAVE_ROWS = 4;

static inline const BlockQ4K* q4k_block_ptr(
    const BlockQ4K* base,
    int row,
    int block_idx,
    int blocks_per_row,
    bool interleaved4) {
    if (!interleaved4) {
        return base + static_cast<size_t>(row) * blocks_per_row + block_idx;
    }
    const int grp = row / INTERLEAVE_ROWS;
    const int lane = row % INTERLEAVE_ROWS;
    return base + ((static_cast<size_t>(grp) * blocks_per_row + block_idx) * INTERLEAVE_ROWS + lane);
}

// Q4_K_M GEMV: rows are parallelized with OpenMP.
// Each row is laid out as (in_dim / 256) BlockQ4K blocks.
void gemv_q4k_avx2_rows(const BlockQ4K* weights, const float* x, float* out, int out_dim, int in_dim, bool interleaved4) {
    if (in_dim % 256 != 0) {
        throw std::runtime_error("in_dim must be divisible by 256 for BlockQ4K");
    }
    const int blocks_per_row = in_dim / 256;
    const __m256i idx_even = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i idx_odd = _mm256_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15);
    const __m256i idx_even_hi = _mm256_setr_epi32(16, 18, 20, 22, 24, 26, 28, 30);
    const __m256i idx_odd_hi = _mm256_setr_epi32(17, 19, 21, 23, 25, 27, 29, 31);

    asdsl_omp_pinning::configure_openmp_for_pcores();
    #pragma omp parallel
    {
        asdsl_omp_pinning::bind_omp_thread_to_pcore_if_enabled();

        #pragma omp for schedule(static) nowait
        for (int r0 = 0; r0 < out_dim; r0 += INTERLEAVE_ROWS) {
            __m256 vacc[INTERLEAVE_ROWS] = {
                _mm256_setzero_ps(), _mm256_setzero_ps(),
                _mm256_setzero_ps(), _mm256_setzero_ps()
            };

            for (int b = 0; b < blocks_per_row; ++b) {
                const float* x_block = x + b * 256;

                for (int lane = 0; lane < INTERLEAVE_ROWS; ++lane) {
                    const int r = r0 + lane;
                    if (r >= out_dim) {
                        continue;
                    }
                    const BlockQ4K& blk = *q4k_block_ptr(weights, r, b, blocks_per_row, interleaved4);
                    const float d = f16_to_f32(blk.d);
                    const float dmin = f16_to_f32(blk.dmin);
                    uint8_t scales[8];
                    unpack_scales_8x6(blk.scales, scales);

                    for (int sb = 0; sb < 8; ++sb) {
                        const float sub_scale = (static_cast<float>(scales[sb]) * d) * (1.0f / 15.0f);
                        const __m256 v_scale = _mm256_set1_ps(sub_scale);
                        const __m256 v_bias = _mm256_set1_ps(dmin);

                        const uint8_t* qptr = blk.qs + sb * 16;
                        const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qptr));
                        const __m128i mask0f = _mm_set1_epi8(0x0F);
                        const __m128i lo_nib = _mm_and_si128(packed, mask0f);
                        const __m128i hi_nib = _mm_and_si128(_mm_srli_epi16(packed, 4), mask0f);

                        const __m128i lo0 = lo_nib;
                        const __m128i hi0 = hi_nib;
                        const __m128i lo1 = _mm_srli_si128(lo_nib, 8);
                        const __m128i hi1 = _mm_srli_si128(hi_nib, 8);

                        __m256i i_lo0 = _mm256_cvtepu8_epi32(lo0);
                        __m256 fq_lo0 = _mm256_cvtepi32_ps(i_lo0);
                        fq_lo0 = _mm256_fmadd_ps(fq_lo0, v_scale, v_bias);
                        __m256 vx0 = _mm256_i32gather_ps(x_block + sb * 32, idx_even, 4);
                        vacc[lane] = _mm256_fmadd_ps(fq_lo0, vx0, vacc[lane]);

                        __m256i i_hi0 = _mm256_cvtepu8_epi32(hi0);
                        __m256 fq_hi0 = _mm256_cvtepi32_ps(i_hi0);
                        fq_hi0 = _mm256_fmadd_ps(fq_hi0, v_scale, v_bias);
                        __m256 vx1 = _mm256_i32gather_ps(x_block + sb * 32, idx_odd, 4);
                        vacc[lane] = _mm256_fmadd_ps(fq_hi0, vx1, vacc[lane]);

                        __m256i i_lo1 = _mm256_cvtepu8_epi32(lo1);
                        __m256 fq_lo1 = _mm256_cvtepi32_ps(i_lo1);
                        fq_lo1 = _mm256_fmadd_ps(fq_lo1, v_scale, v_bias);
                        __m256 vx2 = _mm256_i32gather_ps(x_block + sb * 32, idx_even_hi, 4);
                        vacc[lane] = _mm256_fmadd_ps(fq_lo1, vx2, vacc[lane]);

                        __m256i i_hi1 = _mm256_cvtepu8_epi32(hi1);
                        __m256 fq_hi1 = _mm256_cvtepi32_ps(i_hi1);
                        fq_hi1 = _mm256_fmadd_ps(fq_hi1, v_scale, v_bias);
                        __m256 vx3 = _mm256_i32gather_ps(x_block + sb * 32, idx_odd_hi, 4);
                        vacc[lane] = _mm256_fmadd_ps(fq_hi1, vx3, vacc[lane]);
                    }
                }
            }

            for (int lane = 0; lane < INTERLEAVE_ROWS; ++lane) {
                const int r = r0 + lane;
                if (r < out_dim) {
                    out[r] = hsum256_ps(vacc[lane]);
                }
            }
        }
    }
}

// Prefill path: tiled GEMM over prompt batch X[T, in_dim] and quantized weights W[out_dim, in_dim].
// W is loaded once per (N, K) tile and reused for TILE_M prompt rows.
void gemm_q4k_tiled_prefill(const BlockQ4K* weights, const float* x, float* y, int t_rows, int out_dim, int in_dim, bool interleaved4) {
    if (in_dim % TILE_K != 0) {
        throw std::runtime_error("in_dim must be divisible by TILE_K (256)");
    }
    const int blocks_per_row = in_dim / TILE_K;
    std::memset(y, 0, static_cast<size_t>(t_rows) * out_dim * sizeof(float));

    asdsl_omp_pinning::configure_openmp_for_pcores();
    #pragma omp parallel
    {
        asdsl_omp_pinning::bind_omp_thread_to_pcore_if_enabled();

        #pragma omp for schedule(static) nowait
        for (int tile_idx = 0; tile_idx < ((t_rows + TILE_M - 1) / TILE_M) * ((out_dim + TILE_N - 1) / TILE_N); ++tile_idx) {
            const int tiles_n = (out_dim + TILE_N - 1) / TILE_N;
            const int tile_m_idx = tile_idx / tiles_n;
            const int tile_n_idx = tile_idx % tiles_n;
            const int m0 = tile_m_idx * TILE_M;
            const int n0 = tile_n_idx * TILE_N;

            const int m_lim = std::min(TILE_M, t_rows - m0);
            const int n_lim = std::min(TILE_N, out_dim - n0);
            float acc[TILE_M][TILE_N] = {};

            for (int k0 = 0; k0 < in_dim; k0 += TILE_K) {
                const int block_idx = k0 / TILE_K;

                for (int nn0 = 0; nn0 < n_lim; nn0 += INTERLEAVE_ROWS) {
                    const int row_cnt = std::min(INTERLEAVE_ROWS, n_lim - nn0);
                    const BlockQ4K* blks[INTERLEAVE_ROWS] = {nullptr, nullptr, nullptr, nullptr};
                    float d[INTERLEAVE_ROWS] = {0, 0, 0, 0};
                    float dmin[INTERLEAVE_ROWS] = {0, 0, 0, 0};
                    uint8_t scales[INTERLEAVE_ROWS][8] = {};

                    for (int rr = 0; rr < row_cnt; ++rr) {
                        const int out_row = n0 + nn0 + rr;
                        blks[rr] = q4k_block_ptr(weights, out_row, block_idx, blocks_per_row, interleaved4);
                        d[rr] = f16_to_f32(blks[rr]->d);
                        dmin[rr] = f16_to_f32(blks[rr]->dmin);
                        unpack_scales_8x6(blks[rr]->scales, scales[rr]);
                    }

                    for (int sb = 0; sb < 8; ++sb) {
                        float wtmp[INTERLEAVE_ROWS][32] = {};
                        for (int rr = 0; rr < row_cnt; ++rr) {
                            const float sub_scale = (static_cast<float>(scales[rr][sb]) * d[rr]) * (1.0f / 15.0f);
                            const uint8_t* qptr = blks[rr]->qs + sb * 16;
                            for (int i = 0; i < 16; ++i) {
                                const uint8_t p = qptr[i];
                                wtmp[rr][2 * i] = static_cast<float>(p & 0x0F) * sub_scale + dmin[rr];
                                wtmp[rr][2 * i + 1] = static_cast<float>((p >> 4) & 0x0F) * sub_scale + dmin[rr];
                            }
                        }

                        for (int mm = 0; mm < m_lim; ++mm) {
                            const float* xptr = x + static_cast<size_t>(m0 + mm) * in_dim + k0 + sb * 32;
                            __m256 vx0 = _mm256_loadu_ps(xptr + 0);
                            __m256 vx1 = _mm256_loadu_ps(xptr + 8);
                            __m256 vx2 = _mm256_loadu_ps(xptr + 16);
                            __m256 vx3 = _mm256_loadu_ps(xptr + 24);

                            for (int rr = 0; rr < row_cnt; ++rr) {
                                __m256 vsum = _mm256_setzero_ps();
                                __m256 vw0 = _mm256_loadu_ps(wtmp[rr] + 0);
                                __m256 vw1 = _mm256_loadu_ps(wtmp[rr] + 8);
                                __m256 vw2 = _mm256_loadu_ps(wtmp[rr] + 16);
                                __m256 vw3 = _mm256_loadu_ps(wtmp[rr] + 24);
                                vsum = _mm256_fmadd_ps(vw0, vx0, vsum);
                                vsum = _mm256_fmadd_ps(vw1, vx1, vsum);
                                vsum = _mm256_fmadd_ps(vw2, vx2, vsum);
                                vsum = _mm256_fmadd_ps(vw3, vx3, vsum);
                                acc[mm][nn0 + rr] += hsum256_ps(vsum);
                            }
                        }
                    }
                }
            }

            for (int mm = 0; mm < m_lim; ++mm) {
                float* yrow = y + static_cast<size_t>(m0 + mm) * out_dim + n0;
                for (int nn = 0; nn < n_lim; ++nn) {
                    yrow[nn] = acc[mm][nn];
                }
            }
        }
    }
}

class MmapWeights {
public:
    struct TensorInfo {
        uint8_t* ptr;
        int rows;
        int cols;
        std::string dtype;
        bool interleaved4;
    };

    MmapWeights(const std::string& filepath, py::dict metadata)
        : hFile(INVALID_HANDLE_VALUE), hMapping(nullptr), data(nullptr), size(0) {
        hFile = CreateFileA(filepath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (hFile == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("Failed to open file");
        }

        LARGE_INTEGER li;
        GetFileSizeEx(hFile, &li);
        size = static_cast<size_t>(li.QuadPart);

        hMapping = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!hMapping) {
            CloseHandle(hFile);
            throw std::runtime_error("Failed to create file mapping");
        }

        data = static_cast<uint8_t*>(MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0));
        if (!data) {
            CloseHandle(hMapping);
            CloseHandle(hFile);
            throw std::runtime_error("Failed to map file view");
        }

        for (auto item : metadata) {
            std::string key = py::cast<std::string>(item.first);
            py::dict info = py::cast<py::dict>(item.second);
            size_t offset = py::cast<size_t>(info["offset"]);
            int rows = 0;
            int cols = 0;
            std::string dtype = "";
            if (info.contains("shape")) {
                py::list shape = py::cast<py::list>(info["shape"]);
                if (shape.size() >= 2) {
                    rows = py::cast<int>(shape[0]);
                    cols = py::cast<int>(shape[1]);
                }
            }
            if (info.contains("dtype")) {
                dtype = py::cast<std::string>(info["dtype"]);
            }
            const bool interleaved4 = (dtype.find("i4") != std::string::npos) || (dtype.find("interleaved") != std::string::npos);
            tensors[key] = TensorInfo{data + offset, rows, cols, dtype, interleaved4};
        }

        prefetch_thread = std::thread(&MmapWeights::prefetch_worker, this);
    }

    ~MmapWeights() {
        {
            std::lock_guard<std::mutex> lg(prefetch_mu);
            prefetch_shutdown = true;
        }
        prefetch_cv.notify_all();
        if (prefetch_thread.joinable()) {
            prefetch_thread.join();
        }

        if (data) {
            UnmapViewOfFile(data);
        }
        if (hMapping) {
            CloseHandle(hMapping);
        }
        if (hFile != INVALID_HANDLE_VALUE) {
            CloseHandle(hFile);
        }
    }

    float test_gemv_q4(const std::string& key, py::array_t<float, py::array::c_style | py::array::forcecast> x) const {
        auto it = tensors.find(key);
        if (it == tensors.end()) {
            throw std::runtime_error("Weight key not found");
        }
        py::buffer_info xbuf = x.request();
        if (xbuf.ndim != 1) {
            throw std::runtime_error("x must be 1D float32");
        }

        const int in_dim = static_cast<int>(xbuf.shape[0]);
        if (in_dim % 256 != 0) {
            throw std::runtime_error("x length must be divisible by 256 for q4_k_m");
        }

        float out = 0.0f;
        gemv_q4k_avx2_rows(reinterpret_cast<const BlockQ4K*>(it->second.ptr), static_cast<const float*>(xbuf.ptr), &out, 1, in_dim, it->second.interleaved4);
        return out;
    }

    const TensorInfo* get_tensor(const std::string& key) const {
        auto it = tensors.find(key);
        if (it == tensors.end()) {
            return nullptr;
        }
        return &it->second;
    }

    void enqueue_prefetch_tensor(const std::string& key, size_t bytes_hint = 0) const {
        auto it = tensors.find(key);
        if (it == tensors.end()) {
            return;
        }
        size_t sz = bytes_hint;
        if (sz == 0 && it->second.rows > 0 && it->second.cols > 0) {
            const size_t blocks_per_row = static_cast<size_t>(it->second.cols) / 256;
            const size_t blocks = static_cast<size_t>(it->second.rows) * blocks_per_row;
            const size_t row_factor = it->second.interleaved4 ? 4 : 1;
            sz = (blocks * 140 * row_factor) / row_factor;
        }
        if (sz == 0) {
            return;
        }
        {
            std::lock_guard<std::mutex> lg(prefetch_mu);
            prefetch_queue.push_back({it->second.ptr, sz});
        }
        prefetch_cv.notify_one();
    }

private:
    struct PrefetchTask {
        void* addr;
        size_t size;
    };

    void prefetch_worker() {
        for (;;) {
            PrefetchTask task{nullptr, 0};
            {
                std::unique_lock<std::mutex> lk(prefetch_mu);
                prefetch_cv.wait(lk, [&] { return prefetch_shutdown || !prefetch_queue.empty(); });
                if (prefetch_shutdown && prefetch_queue.empty()) {
                    break;
                }
                task = prefetch_queue.front();
                prefetch_queue.erase(prefetch_queue.begin());
            }

#ifdef _WIN32
            WIN32_MEMORY_RANGE_ENTRY range;
            range.VirtualAddress = task.addr;
            range.NumberOfBytes = task.size;
            PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0);
#else
            volatile uint8_t s = 0;
            const uint8_t* p = reinterpret_cast<const uint8_t*>(task.addr);
            for (size_t i = 0; i < task.size; i += 4096) {
                s ^= p[i];
            }
            (void)s;
#endif
        }
    }

    HANDLE hFile;
    HANDLE hMapping;
    uint8_t* data;
    size_t size;
    std::unordered_map<std::string, TensorInfo> tensors;
    mutable std::mutex prefetch_mu;
    mutable std::condition_variable prefetch_cv;
    mutable std::vector<PrefetchTask> prefetch_queue;
    mutable bool prefetch_shutdown = false;
    mutable std::thread prefetch_thread;
};

struct KVCache {
    static constexpr int KV_QBLOCK = 64;

    KVCache(int layers, int max_seq_len, int num_kv_heads, int head_dim)
        : layers(layers), max_seq_len(max_seq_len), num_kv_heads(num_kv_heads), head_dim(head_dim) {
        blocks_per_head = (head_dim + KV_QBLOCK - 1) / KV_QBLOCK;
        const size_t kv_size = static_cast<size_t>(layers) * max_seq_len * num_kv_heads * head_dim;
        const size_t scale_size = static_cast<size_t>(layers) * max_seq_len * num_kv_heads * blocks_per_head;
        k_cache_q8.assign(kv_size, 0);
        v_cache_q8.assign(kv_size, 0);
        k_scales.assign(scale_size, 1.0f);
        v_scales.assign(scale_size, 1.0f);
    }

    inline size_t kv_base(int layer, int pos, int kv_head) const {
        return ((((static_cast<size_t>(layer) * max_seq_len) + pos) * num_kv_heads) + kv_head) * head_dim;
    }

    inline size_t scale_base(int layer, int pos, int kv_head) const {
        return ((((static_cast<size_t>(layer) * max_seq_len) + pos) * num_kv_heads) + kv_head) * blocks_per_head;
    }

    void set_history(int layer, int pos, const float* k, const float* v) {
        if (layer < 0 || layer >= layers || pos < 0 || pos >= max_seq_len) {
            return;
        }
        for (int h = 0; h < num_kv_heads; ++h) {
            const float* kh = k + static_cast<size_t>(h) * head_dim;
            const float* vh = v + static_cast<size_t>(h) * head_dim;
            const size_t kvb = kv_base(layer, pos, h);
            const size_t sb = scale_base(layer, pos, h);

            for (int b = 0; b < blocks_per_head; ++b) {
                const int off = b * KV_QBLOCK;
                const int len = std::min(KV_QBLOCK, head_dim - off);
                float k_absmax = 0.0f;
                float v_absmax = 0.0f;
                for (int i = 0; i < len; ++i) {
                    k_absmax = std::max(k_absmax, std::fabs(kh[off + i]));
                    v_absmax = std::max(v_absmax, std::fabs(vh[off + i]));
                }
                const float ks = (k_absmax > 1e-12f) ? (k_absmax / 127.0f) : 1e-12f;
                const float vs = (v_absmax > 1e-12f) ? (v_absmax / 127.0f) : 1e-12f;
                k_scales[sb + b] = ks;
                v_scales[sb + b] = vs;

                const float kinv = 1.0f / ks;
                const float vinv = 1.0f / vs;
                for (int i = 0; i < len; ++i) {
                    int qk = static_cast<int>(std::nearbyint(kh[off + i] * kinv));
                    int qv = static_cast<int>(std::nearbyint(vh[off + i] * vinv));
                    qk = std::max(-127, std::min(127, qk));
                    qv = std::max(-127, std::min(127, qv));
                    k_cache_q8[kvb + off + i] = static_cast<int8_t>(qk);
                    v_cache_q8[kvb + off + i] = static_cast<int8_t>(qv);
                }
            }
        }
    }

    int layers;
    int max_seq_len;
    int num_kv_heads;
    int head_dim;
    int blocks_per_head;
    std::vector<int8_t> k_cache_q8;
    std::vector<int8_t> v_cache_q8;
    std::vector<float> k_scales;
    std::vector<float> v_scales;
};

/** KV cache with 4-bit packed keys/values (2 nibbles/byte) and per-64-dim FP32 scales. */
struct KVCacheQ4 {
    static constexpr int KV_QBLOCK = 64;

    KVCacheQ4(int layers, int max_seq_len, int num_kv_heads, int head_dim)
        : layers(layers), max_seq_len(max_seq_len), num_kv_heads(num_kv_heads), head_dim(head_dim) {
        blocks_per_head = (head_dim + KV_QBLOCK - 1) / KV_QBLOCK;
        const size_t kv_size = static_cast<size_t>(layers) * max_seq_len * num_kv_heads * head_dim;
        const size_t pack_size = (kv_size + 1) / 2;
        const size_t scale_size = static_cast<size_t>(layers) * max_seq_len * num_kv_heads * blocks_per_head;
        k_pack.assign(pack_size, 0);
        v_pack.assign(pack_size, 0);
        k_scales.assign(scale_size, 1.0f);
        v_scales.assign(scale_size, 1.0f);
    }

    inline size_t kv_base(int layer, int pos, int kv_head) const {
        return ((((static_cast<size_t>(layer) * max_seq_len) + pos) * num_kv_heads) + kv_head) * head_dim;
    }

    inline size_t scale_base(int layer, int pos, int kv_head) const {
        return ((((static_cast<size_t>(layer) * max_seq_len) + pos) * num_kv_heads) + kv_head) * blocks_per_head;
    }

    static inline void pack_nibble(std::vector<uint8_t>& pack, size_t virt, int q) {
        const int v = std::max(0, std::min(15, q));
        const size_t bi = virt >> 1;
        if ((virt & 1) == 0) {
            pack[bi] = static_cast<uint8_t>((pack[bi] & 0xF0u) | static_cast<uint8_t>(v));
        } else {
            pack[bi] = static_cast<uint8_t>((pack[bi] & 0x0Fu) | (static_cast<uint8_t>(v) << 4));
        }
    }

    static inline float q4_to_float(const std::vector<uint8_t>& pack, size_t virt, float scale) {
        const size_t bi = virt >> 1;
        const int q = ((virt & 1) == 0) ? static_cast<int>(pack[bi] & 0xFu) : static_cast<int>((pack[bi] >> 4) & 0xFu);
        return (static_cast<float>(q) - 8.0f) * scale;
    }

    void set_history(int layer, int pos, const float* k, const float* v) {
        if (layer < 0 || layer >= layers || pos < 0 || pos >= max_seq_len) {
            return;
        }
        for (int h = 0; h < num_kv_heads; ++h) {
            const float* kh = k + static_cast<size_t>(h) * head_dim;
            const float* vh = v + static_cast<size_t>(h) * head_dim;
            const size_t kvb = kv_base(layer, pos, h);
            const size_t sb = scale_base(layer, pos, h);

            for (int b = 0; b < blocks_per_head; ++b) {
                const int off = b * KV_QBLOCK;
                const int len = std::min(KV_QBLOCK, head_dim - off);
                float k_absmax = 0.0f;
                float v_absmax = 0.0f;
                for (int i = 0; i < len; ++i) {
                    k_absmax = std::max(k_absmax, std::fabs(kh[off + i]));
                    v_absmax = std::max(v_absmax, std::fabs(vh[off + i]));
                }
                const float ks = (k_absmax > 1e-12f) ? (k_absmax / 8.0f) : 1e-12f;
                const float vs = (v_absmax > 1e-12f) ? (v_absmax / 8.0f) : 1e-12f;
                k_scales[sb + b] = ks;
                v_scales[sb + b] = vs;

                const float kinv = 1.0f / ks;
                const float vinv = 1.0f / vs;
                for (int i = 0; i < len; ++i) {
                    int qk = static_cast<int>(std::nearbyint(kh[off + i] * kinv + 8.0f));
                    int qv = static_cast<int>(std::nearbyint(vh[off + i] * vinv + 8.0f));
                    pack_nibble(k_pack, kvb + static_cast<size_t>(off + i), qk);
                    pack_nibble(v_pack, kvb + static_cast<size_t>(off + i), qv);
                }
            }
        }
    }

    int layers;
    int max_seq_len;
    int num_kv_heads;
    int head_dim;
    int blocks_per_head;
    std::vector<uint8_t> k_pack;
    std::vector<uint8_t> v_pack;
    std::vector<float> k_scales;
    std::vector<float> v_scales;
};

static constexpr int BLOCK_Q = 32;
static constexpr int BLOCK_K = 64;

static py::array_t<float> compute_attention_flash_q8(
    py::array_t<float, py::array::c_style | py::array::forcecast> q,
    py::array_t<float, py::array::c_style | py::array::forcecast> k,
    py::array_t<float, py::array::c_style | py::array::forcecast> v,
    int layer_id,
    int seq_pos,
    int num_heads,
    KVCache& cache) {
    py::buffer_info q_buf = q.request();
    py::buffer_info k_buf = k.request();
    py::buffer_info v_buf = v.request();

    if (q_buf.ndim != 2 || k_buf.ndim != 2 || v_buf.ndim != 2) {
        throw std::runtime_error("q, k, v must be 2D arrays");
    }

    const int head_dim = cache.head_dim;
    const int num_kv_heads = cache.num_kv_heads;
    const int q_heads = static_cast<int>(q_buf.shape[0]);
    if (q_heads != num_heads || q_buf.shape[1] != head_dim) {
        throw std::runtime_error("q shape mismatch");
    }
    if (k_buf.shape[0] != num_kv_heads || k_buf.shape[1] != head_dim ||
        v_buf.shape[0] != num_kv_heads || v_buf.shape[1] != head_dim) {
        throw std::runtime_error("k/v shape mismatch");
    }

    cache.set_history(layer_id, seq_pos, static_cast<const float*>(k_buf.ptr), static_cast<const float*>(v_buf.ptr));

    auto out = py::array_t<float>({num_heads, head_dim});
    py::buffer_info out_buf = out.request();
    float* out_ptr = static_cast<float*>(out_buf.ptr);
    const float* q_ptr = static_cast<const float*>(q_buf.ptr);

    const int groups = std::max(1, num_heads / num_kv_heads);
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(head_dim));

    asdsl_omp_pinning::configure_openmp_for_pcores();
#pragma omp parallel
    {
        asdsl_omp_pinning::bind_omp_thread_to_pcore_if_enabled();
#pragma omp for schedule(static)
        for (int h = 0; h < num_heads; ++h) {
        const int kv_h = h / groups;
        const float* qh = q_ptr + static_cast<size_t>(h) * head_dim;
        std::vector<float> num(head_dim, 0.0f);
        float m = -std::numeric_limits<float>::infinity();
        float l = 0.0f;

        for (int tk = 0; tk <= seq_pos; tk += BLOCK_K) {
            const int kend = std::min(seq_pos + 1, tk + BLOCK_K);
            const int span = kend - tk;
            float scores[BLOCK_K];
            float tile_max = -std::numeric_limits<float>::infinity();

            for (int p = 0; p < span; ++p) {
                const int pos = tk + p;
                const size_t kb = cache.kv_base(layer_id, pos, kv_h);
                const size_t sb = cache.scale_base(layer_id, pos, kv_h);
                float dot = 0.0f;
                for (int b = 0; b < cache.blocks_per_head; ++b) {
                    const int off = b * KVCache::KV_QBLOCK;
                    const int len = std::min(KVCache::KV_QBLOCK, head_dim - off);
                    const float ks = cache.k_scales[sb + b];
                    const int8_t* kq = cache.k_cache_q8.data() + kb + off;
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
            std::vector<float> tile_num(head_dim, 0.0f);

            for (int p = 0; p < span; ++p) {
                const int pos = tk + p;
                const float w = std::exp(scores[p] - new_m);
                tile_l += w;

                const size_t vb = cache.kv_base(layer_id, pos, kv_h);
                const size_t sb = cache.scale_base(layer_id, pos, kv_h);
                for (int b = 0; b < cache.blocks_per_head; ++b) {
                    const int off = b * KVCache::KV_QBLOCK;
                    const int len = std::min(KVCache::KV_QBLOCK, head_dim - off);
                    const float vs = cache.v_scales[sb + b];
                    const int8_t* vq = cache.v_cache_q8.data() + vb + off;
                    for (int i = 0; i < len; ++i) {
                        tile_num[off + i] += w * (static_cast<float>(vq[i]) * vs);
                    }
                }
            }

            for (int i = 0; i < head_dim; ++i) {
                num[i] = num[i] * old_scale + tile_num[i];
            }
            l = l * old_scale + tile_l;
            m = new_m;
        }

        const float inv_l = (l > 1e-30f) ? (1.0f / l) : 0.0f;
        float* oh = out_ptr + static_cast<size_t>(h) * head_dim;
        for (int i = 0; i < head_dim; ++i) {
            oh[i] = num[i] * inv_l;
        }
        }
    }

    return out;
}

static py::array_t<float> compute_attention_flash_q4(
    py::array_t<float, py::array::c_style | py::array::forcecast> q,
    py::array_t<float, py::array::c_style | py::array::forcecast> k,
    py::array_t<float, py::array::c_style | py::array::forcecast> v,
    int layer_id,
    int seq_pos,
    int num_heads,
    KVCacheQ4& cache) {
    py::buffer_info q_buf = q.request();
    py::buffer_info k_buf = k.request();
    py::buffer_info v_buf = v.request();

    if (q_buf.ndim != 2 || k_buf.ndim != 2 || v_buf.ndim != 2) {
        throw std::runtime_error("q, k, v must be 2D arrays");
    }

    const int head_dim = cache.head_dim;
    const int num_kv_heads = cache.num_kv_heads;
    const int q_heads = static_cast<int>(q_buf.shape[0]);
    if (q_heads != num_heads || q_buf.shape[1] != head_dim) {
        throw std::runtime_error("q shape mismatch");
    }
    if (k_buf.shape[0] != num_kv_heads || k_buf.shape[1] != head_dim ||
        v_buf.shape[0] != num_kv_heads || v_buf.shape[1] != head_dim) {
        throw std::runtime_error("k/v shape mismatch");
    }

    cache.set_history(layer_id, seq_pos, static_cast<const float*>(k_buf.ptr), static_cast<const float*>(v_buf.ptr));

    auto out = py::array_t<float>({num_heads, head_dim});
    py::buffer_info out_buf = out.request();
    float* out_ptr = static_cast<float*>(out_buf.ptr);
    const float* q_ptr = static_cast<const float*>(q_buf.ptr);

    const int groups = std::max(1, num_heads / num_kv_heads);
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(head_dim));

    asdsl_omp_pinning::configure_openmp_for_pcores();
#pragma omp parallel
    {
        asdsl_omp_pinning::bind_omp_thread_to_pcore_if_enabled();
#pragma omp for schedule(static)
        for (int h = 0; h < num_heads; ++h) {
        const int kv_h = h / groups;
        const float* qh = q_ptr + static_cast<size_t>(h) * head_dim;
        std::vector<float> num(head_dim, 0.0f);
        float m = -std::numeric_limits<float>::infinity();
        float l = 0.0f;

        for (int tk = 0; tk <= seq_pos; tk += BLOCK_K) {
            const int kend = std::min(seq_pos + 1, tk + BLOCK_K);
            const int span = kend - tk;
            float scores[BLOCK_K];
            float tile_max = -std::numeric_limits<float>::infinity();

            for (int p = 0; p < span; ++p) {
                const int pos = tk + p;
                const size_t kb = cache.kv_base(layer_id, pos, kv_h);
                const size_t sb = cache.scale_base(layer_id, pos, kv_h);
                float dot = 0.0f;
                for (int b = 0; b < cache.blocks_per_head; ++b) {
                    const int off = b * KVCacheQ4::KV_QBLOCK;
                    const int len = std::min(KVCacheQ4::KV_QBLOCK, head_dim - off);
                    const float ks = cache.k_scales[sb + b];
                    for (int i = 0; i < len; ++i) {
                        const size_t virt = kb + static_cast<size_t>(off + i);
                        dot += qh[off + i] * KVCacheQ4::q4_to_float(cache.k_pack, virt, ks);
                    }
                }
                const float s = dot * inv_sqrt_d;
                scores[p] = s;
                tile_max = std::max(tile_max, s);
            }

            const float new_m = std::max(m, tile_max);
            const float old_scale = (l > 0.0f) ? std::exp(m - new_m) : 0.0f;
            float tile_l = 0.0f;
            std::vector<float> tile_num(head_dim, 0.0f);

            for (int p = 0; p < span; ++p) {
                const int pos = tk + p;
                const float w = std::exp(scores[p] - new_m);
                tile_l += w;

                const size_t vb = cache.kv_base(layer_id, pos, kv_h);
                const size_t sb = cache.scale_base(layer_id, pos, kv_h);
                for (int b = 0; b < cache.blocks_per_head; ++b) {
                    const int off = b * KVCacheQ4::KV_QBLOCK;
                    const int len = std::min(KVCacheQ4::KV_QBLOCK, head_dim - off);
                    const float vs = cache.v_scales[sb + b];
                    for (int i = 0; i < len; ++i) {
                        const size_t virt = vb + static_cast<size_t>(off + i);
                        tile_num[off + i] += w * KVCacheQ4::q4_to_float(cache.v_pack, virt, vs);
                    }
                }
            }

            for (int i = 0; i < head_dim; ++i) {
                num[i] = num[i] * old_scale + tile_num[i];
            }
            l = l * old_scale + tile_l;
            m = new_m;
        }

        const float inv_l = (l > 1e-30f) ? (1.0f / l) : 0.0f;
        float* oh = out_ptr + static_cast<size_t>(h) * head_dim;
        for (int i = 0; i < head_dim; ++i) {
            oh[i] = num[i] * inv_l;
        }
        }
    }

    return out;
}

void py_kvc_q4_set_history(
    KVCacheQ4& cache,
    int layer,
    int pos,
    py::array_t<float, py::array::c_style | py::array::forcecast> k,
    py::array_t<float, py::array::c_style | py::array::forcecast> v) {
    py::buffer_info k_buf = k.request();
    py::buffer_info v_buf = v.request();
    if (k_buf.ndim != 2 || v_buf.ndim != 2) {
        throw std::runtime_error("k and v must be 2D");
    }
    if (k_buf.shape[0] != cache.num_kv_heads || k_buf.shape[1] != cache.head_dim ||
        v_buf.shape[0] != cache.num_kv_heads || v_buf.shape[1] != cache.head_dim) {
        throw std::runtime_error("k/v shape mismatch for Q4 cache");
    }
    cache.set_history(layer, pos, static_cast<const float*>(k_buf.ptr), static_cast<const float*>(v_buf.ptr));
}

void py_kvc_set_history(
    KVCache& cache,
    int layer,
    int pos,
    py::array_t<float, py::array::c_style | py::array::forcecast> k,
    py::array_t<float, py::array::c_style | py::array::forcecast> v) {
    py::buffer_info k_buf = k.request();
    py::buffer_info v_buf = v.request();
    if (k_buf.ndim != 2 || v_buf.ndim != 2) {
        throw std::runtime_error("k and v must be 2D");
    }
    if (k_buf.shape[0] != cache.num_kv_heads || k_buf.shape[1] != cache.head_dim ||
        v_buf.shape[0] != cache.num_kv_heads || v_buf.shape[1] != cache.head_dim) {
        throw std::runtime_error("k/v shape mismatch for cache");
    }
    cache.set_history(layer, pos, static_cast<const float*>(k_buf.ptr), static_cast<const float*>(v_buf.ptr));
}

py::array_t<float> gemv_q4k_row_py(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> row_bytes,
    py::array_t<float, py::array::c_style | py::array::forcecast> x,
    int out_dim,
    int in_dim) {
    py::buffer_info wbuf = row_bytes.request();
    py::buffer_info xbuf = x.request();

    if (wbuf.ndim != 1 || xbuf.ndim != 1) {
        throw std::runtime_error("row_bytes and x must be 1D arrays");
    }
    if (xbuf.shape[0] != in_dim) {
        throw std::runtime_error("x length must match in_dim");
    }

    const int expected_bytes = out_dim * (in_dim / 256) * static_cast<int>(sizeof(BlockQ4K));
    if (in_dim % 256 != 0 || wbuf.shape[0] != expected_bytes) {
        throw std::runtime_error("row_bytes has invalid size for out_dim/in_dim and BlockQ4K layout");
    }

    py::array_t<float> out(out_dim);
    py::buffer_info obuf = out.request();
    std::memset(obuf.ptr, 0, static_cast<size_t>(out_dim) * sizeof(float));

    gemv_q4k_avx2_rows(
        reinterpret_cast<const BlockQ4K*>(wbuf.ptr),
        static_cast<const float*>(xbuf.ptr),
        static_cast<float*>(obuf.ptr),
        out_dim,
        in_dim,
        false);

    return out;
}

int generate_token_stub(
    int token_id,
    int seq_pos,
    py::array_t<float, py::array::c_style | py::array::forcecast> token_emb,
    py::list layers_w,
    py::array_t<float, py::array::c_style | py::array::forcecast> final_rms,
    py::array_t<float, py::array::c_style | py::array::forcecast> lm_head,
    int num_layers,
    int dim,
    int hidden_dim,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int vocab_size,
    KVCache& cache) {
    (void)seq_pos;
    (void)token_emb;
    (void)layers_w;
    (void)final_rms;
    (void)lm_head;
    (void)num_layers;
    (void)dim;
    (void)hidden_dim;
    (void)num_heads;
    (void)num_kv_heads;
    (void)head_dim;
    (void)cache;
    if (vocab_size <= 0) {
        return 0;
    }
    int safe = token_id % vocab_size;
    return safe < 0 ? safe + vocab_size : safe;
}

int generate_token_mmap_stub(
    int token_id,
    int seq_pos,
    const MmapWeights& store,
    int num_layers,
    int dim,
    int hidden_dim,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int vocab_size,
    KVCache& cache) {
    
    if (token_id < 0 || token_id >= vocab_size) {
        return 0;
    }
    
    const int q_dim = num_heads * head_dim;
    const int kv_dim = num_kv_heads * head_dim;
    const int inter_dim = hidden_dim;
    
    std::vector<float> hidden(dim, 0.0f);
    std::vector<float> residual(dim, 0.0f);
    std::vector<float> qkv_out(q_dim + 2 * kv_dim, 0.0f);
    std::vector<float> q_arr(q_dim, 0.0f);
    std::vector<float> k_arr(kv_dim, 0.0f);
    std::vector<float> v_arr(kv_dim, 0.0f);
    std::vector<float> att_out(q_dim, 0.0f);
    std::vector<float> ffn_gate(inter_dim, 0.0f);
    std::vector<float> ffn_up(inter_dim, 0.0f);
    std::vector<float> ffn_combined(inter_dim, 0.0f);
    std::vector<float> logits(vocab_size, 0.0f);
    
    const MmapWeights::TensorInfo* embed_t = store.get_tensor("embed");
    if (!embed_t) {
        return 0;
    }
    
    const float* embed_data = reinterpret_cast<const float*>(embed_t->ptr);
    const size_t embed_offset = static_cast<size_t>(token_id) * dim;
    std::memcpy(hidden.data(), embed_data + embed_offset, dim * sizeof(float));
    
    for (int layer = 0; layer < num_layers; ++layer) {
        std::memcpy(residual.data(), hidden.data(), dim * sizeof(float));
        
        std::string rms1_key = "l" + std::to_string(layer) + "_rms1";
        const MmapWeights::TensorInfo* rms1_t = store.get_tensor(rms1_key);
        if (rms1_t) {
            const float* rms1_w = reinterpret_cast<const float*>(rms1_t->ptr);
            float rms = 0.0f;
            for (int i = 0; i < dim; ++i) {
                rms += hidden[i] * hidden[i];
            }
            rms = std::sqrt(rms / dim + 1e-5f);
            const float rms_inv = 1.0f / rms;
            for (int i = 0; i < dim; ++i) {
                hidden[i] = hidden[i] * rms_inv * rms1_w[i];
            }
        }
        
        std::string qkv_key = "l" + std::to_string(layer) + "_qkv";
        const MmapWeights::TensorInfo* qkv_t = store.get_tensor(qkv_key);
        if (!qkv_t) break;
        
        std::fill(qkv_out.begin(), qkv_out.end(), 0.0f);
        gemv_q4k_avx2_rows(
            reinterpret_cast<const BlockQ4K*>(qkv_t->ptr),
            hidden.data(),
            qkv_out.data(),
            q_dim + 2 * kv_dim,
            dim,
            qkv_t->interleaved4
        );
        
        std::memcpy(q_arr.data(), qkv_out.data(), q_dim * sizeof(float));
        std::memcpy(k_arr.data(), qkv_out.data() + q_dim, kv_dim * sizeof(float));
        std::memcpy(v_arr.data(), qkv_out.data() + q_dim + kv_dim, kv_dim * sizeof(float));
        
        cache.set_history(layer, seq_pos, k_arr.data(), v_arr.data());
        
        std::memcpy(att_out.data(), q_arr.data(), q_dim * sizeof(float));
        
        std::string o_key = "l" + std::to_string(layer) + "_o";
        const MmapWeights::TensorInfo* o_t = store.get_tensor(o_key);
        if (o_t) {
            std::vector<float> o_out(dim, 0.0f);
            gemv_q4k_avx2_rows(
                reinterpret_cast<const BlockQ4K*>(o_t->ptr),
                att_out.data(),
                o_out.data(),
                dim,
                q_dim,
                o_t->interleaved4
            );
            for (int i = 0; i < dim; ++i) {
                hidden[i] = o_out[i] + residual[i];
            }
        }
        
        std::memcpy(residual.data(), hidden.data(), dim * sizeof(float));
        
        std::string rms2_key = "l" + std::to_string(layer) + "_rms2";
        const MmapWeights::TensorInfo* rms2_t = store.get_tensor(rms2_key);
        if (rms2_t) {
            const float* rms2_w = reinterpret_cast<const float*>(rms2_t->ptr);
            float rms = 0.0f;
            for (int i = 0; i < dim; ++i) {
                rms += hidden[i] * hidden[i];
            }
            rms = std::sqrt(rms / dim + 1e-5f);
            const float rms_inv = 1.0f / rms;
            for (int i = 0; i < dim; ++i) {
                hidden[i] = hidden[i] * rms_inv * rms2_w[i];
            }
        }
        
        std::string gate_key = "l" + std::to_string(layer) + "_gate";
        std::string up_key = "l" + std::to_string(layer) + "_up";
        const MmapWeights::TensorInfo* gate_t = store.get_tensor(gate_key);
        const MmapWeights::TensorInfo* up_t = store.get_tensor(up_key);
        
        if (gate_t && up_t) {
            std::fill(ffn_gate.begin(), ffn_gate.end(), 0.0f);
            std::fill(ffn_up.begin(), ffn_up.end(), 0.0f);
            
            gemv_q4k_avx2_rows(
                reinterpret_cast<const BlockQ4K*>(gate_t->ptr),
                hidden.data(),
                ffn_gate.data(),
                inter_dim,
                dim,
                gate_t->interleaved4
            );
            
            gemv_q4k_avx2_rows(
                reinterpret_cast<const BlockQ4K*>(up_t->ptr),
                hidden.data(),
                ffn_up.data(),
                inter_dim,
                dim,
                up_t->interleaved4
            );
            
            for (int i = 0; i < inter_dim; ++i) {
                float up_val = ffn_up[i];
                float silu_val = up_val / (1.0f + std::exp(-up_val));
                ffn_combined[i] = ffn_gate[i] * silu_val;
            }
        }
        
        std::string down_key = "l" + std::to_string(layer) + "_down";
        const MmapWeights::TensorInfo* down_t = store.get_tensor(down_key);
        if (down_t) {
            std::fill(hidden.begin(), hidden.end(), 0.0f);
            gemv_q4k_avx2_rows(
                reinterpret_cast<const BlockQ4K*>(down_t->ptr),
                ffn_combined.data(),
                hidden.data(),
                dim,
                inter_dim,
                down_t->interleaved4
            );
            for (int i = 0; i < dim; ++i) {
                hidden[i] += residual[i];
            }
        }
    }
    
    int safe = token_id % vocab_size;
    return safe < 0 ? safe + vocab_size : safe;
}

void prefill_prompt_tokens_mmap(
    py::array_t<int, py::array::c_style | py::array::forcecast> prompt_ids,
    int seq_start,
    const MmapWeights& store,
    int num_layers,
    int dim,
    int hidden_dim,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int vocab_size,
    KVCache& cache) {
    (void)seq_start;
    (void)hidden_dim;
    (void)num_heads;
    (void)num_kv_heads;
    (void)head_dim;
    (void)vocab_size;
    (void)cache;

    py::buffer_info ids_buf = prompt_ids.request();
    if (ids_buf.ndim != 1) {
        throw std::runtime_error("prompt_ids must be 1D");
    }
    const int t_rows = static_cast<int>(ids_buf.shape[0]);
    if (t_rows <= 1 || dim <= 0) {
        return;
    }

    std::string key = "l0_o";
    const MmapWeights::TensorInfo* t = store.get_tensor(key);
    if (!t && num_layers > 0) {
        key = "l0_qkv";
        t = store.get_tensor(key);
    }
    if (!t || t->cols <= 0 || t->rows <= 0) {
        return;
    }

    const int in_dim = t->cols;
    const int out_dim = t->rows;
    if (in_dim % TILE_K != 0) {
        return;
    }

    std::vector<float> x(static_cast<size_t>(t_rows) * in_dim);
    const int* ids = static_cast<const int*>(ids_buf.ptr);
    for (int i = 0; i < t_rows; ++i) {
        const float v = static_cast<float>((ids[i] & 1023) - 512) * (1.0f / 512.0f);
        float* row = x.data() + static_cast<size_t>(i) * in_dim;
        for (int d = 0; d < in_dim; ++d) {
            row[d] = v;
        }
    }

    std::vector<float> y(static_cast<size_t>(t_rows) * out_dim);

    // Simulate per-layer prefill compute with tiled GEMM while reusing loaded row tiles.
    for (int l = 0; l < num_layers; ++l) {
        std::string lpfx = "l" + std::to_string(l) + "_o";
        const MmapWeights::TensorInfo* lw = store.get_tensor(lpfx);
        if (!lw || lw->cols != in_dim || lw->rows != out_dim) {
            lw = t;
        }

        if (l + 1 < num_layers) {
            std::string npfx = "l" + std::to_string(l + 1) + "_o";
            store.enqueue_prefetch_tensor(npfx);
        }

        gemm_q4k_tiled_prefill(
            reinterpret_cast<const BlockQ4K*>(lw->ptr),
            x.data(), y.data(), t_rows, out_dim, in_dim, lw->interleaved4);
        if (in_dim == out_dim) {
            x.swap(y);
        }
    }
}

py::array_t<float> gemm_q4k_prefill_py(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> weights_bytes,
    py::array_t<float, py::array::c_style | py::array::forcecast> x,
    int t_rows,
    int out_dim,
    int in_dim,
    bool interleaved4) {
    py::buffer_info wbuf = weights_bytes.request();
    py::buffer_info xbuf = x.request();
    if (xbuf.ndim != 2) {
        throw std::runtime_error("x must be 2D (T, in_dim)");
    }
    if (xbuf.shape[0] != t_rows || xbuf.shape[1] != in_dim) {
        throw std::runtime_error("x shape does not match (t_rows, in_dim)");
    }
    const int expected_bytes = out_dim * (in_dim / TILE_K) * static_cast<int>(sizeof(BlockQ4K));
    if (wbuf.ndim != 1 || wbuf.shape[0] != expected_bytes) {
        throw std::runtime_error("weights_bytes has invalid size for q4k tiled gemm");
    }

    py::array_t<float> out({t_rows, out_dim});
    py::buffer_info obuf = out.request();
    gemm_q4k_tiled_prefill(
        reinterpret_cast<const BlockQ4K*>(wbuf.ptr),
        static_cast<const float*>(xbuf.ptr),
        static_cast<float*>(obuf.ptr),
        t_rows,
        out_dim,
        in_dim,
        interleaved4);
    return out;
}

void pin_thread_to_core(int core_id) {
    DWORD_PTR mask = (1ULL << core_id);
    SetThreadAffinityMask(GetCurrentThread(), mask);
}

void pin_openmp_threads_to_pcores(bool enabled) {
    asdsl_omp_pinning::pin_openmp_pcores_enabled() = enabled;
    if (enabled) {
        asdsl_omp_pinning::configure_openmp_for_pcores();
    }
}

int detected_pcore_count() {
    return asdsl_omp_pinning::detected_pcore_count();
}

/** Multi-threaded STREAM Triad on float32 arrays — compiler vectorizes; use for DRAM roofline. */
static py::dict stream_triad_f32_py(int array_mb, int runs, int warmup_runs) {
    if (array_mb < 256) {
        throw std::runtime_error("array_mb must be >= 256 to exceed CPU caches");
    }
    if (runs < 1) {
        throw std::runtime_error("runs must be >= 1");
    }
    if (warmup_runs < 0) {
        throw std::runtime_error("warmup_runs must be >= 0");
    }

    const size_t n = (static_cast<size_t>(array_mb) * 1024ull * 1024ull) / sizeof(float);
    if (n == 0) {
        throw std::runtime_error("array_mb too small for float32 buffers");
    }

    std::vector<float> a(n);
    std::vector<float> b(n);
    std::vector<float> c(n);

    std::mt19937 rng(1234567);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) {
        b[i] = dist(rng);
        c[i] = dist(rng);
    }

    const float scalar = 3.0f;
    asdsl_omp_pinning::configure_openmp_for_pcores();

    auto run_triad = [&]() {
#pragma omp parallel
        {
            asdsl_omp_pinning::bind_omp_thread_to_pcore_if_enabled();
#pragma omp for schedule(static)
            for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
                a[static_cast<size_t>(i)] =
                    b[static_cast<size_t>(i)] + scalar * c[static_cast<size_t>(i)];
            }
        }
    };

    for (int w = 0; w < warmup_runs; ++w) {
        run_triad();
    }

    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < runs; ++r) {
        run_triad();
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();

    if (elapsed <= 0.0) {
        throw std::runtime_error("STREAM Triad timing produced non-positive elapsed time");
    }

    const double array_bytes = static_cast<double>(n) * static_cast<double>(sizeof(float));
    const double bandwidth_gb_s = (3.0 * array_bytes * static_cast<double>(runs)) / elapsed / 1e9;

    py::dict out;
    out["bandwidth_gb_s"] = bandwidth_gb_s;
    out["elapsed_sec"] = elapsed;
    out["runs"] = runs;
    out["warmup_runs"] = warmup_runs;
    out["array_bytes"] = static_cast<int64_t>(array_bytes);
    out["dtype"] = "float32";
    out["omp_max_threads"] = omp_get_max_threads();
    out["detected_pcores"] = detected_pcore_count();
    out["pin_openmp_pcores"] = asdsl_omp_pinning::pin_openmp_pcores_enabled();
    return out;
}

/** Multi-threaded STREAM Triad on int8 arrays (a = b + scalar * c) for DRAM bandwidth probing. */
static py::dict stream_triad_int8_py(int array_mb, int runs, int warmup_runs) {
    if (array_mb < 256) {
        throw std::runtime_error("array_mb must be >= 256 to exceed CPU caches");
    }
    if (runs < 1) {
        throw std::runtime_error("runs must be >= 1");
    }
    if (warmup_runs < 0) {
        throw std::runtime_error("warmup_runs must be >= 0");
    }

    const size_t n = static_cast<size_t>(array_mb) * 1024ull * 1024ull;
    std::vector<int8_t> a(n);
    std::vector<int8_t> b(n);
    std::vector<int8_t> c(n);

    std::mt19937 rng(1234567);
    std::uniform_int_distribution<int> dist(-127, 127);
    for (size_t i = 0; i < n; ++i) {
        b[i] = static_cast<int8_t>(dist(rng));
        c[i] = static_cast<int8_t>(dist(rng));
    }

    const int scalar = 3;
    // Match GEMV path: fixed thread count + per-thread affinity to P-cores before the timed region.
    asdsl_omp_pinning::configure_openmp_for_pcores();

    auto run_triad = [&]() {
#pragma omp parallel
        {
            asdsl_omp_pinning::bind_omp_thread_to_pcore_if_enabled();
#pragma omp for schedule(static)
            for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
                int v = static_cast<int>(b[static_cast<size_t>(i)])
                    + scalar * static_cast<int>(c[static_cast<size_t>(i)]);
                if (v > 127) {
                    v = 127;
                } else if (v < -128) {
                    v = -128;
                }
                a[static_cast<size_t>(i)] = static_cast<int8_t>(v);
            }
        }
    };

    for (int w = 0; w < warmup_runs; ++w) {
        run_triad();
    }

    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < runs; ++r) {
        run_triad();
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();

    if (elapsed <= 0.0) {
        throw std::runtime_error("STREAM Triad timing produced non-positive elapsed time");
    }

    const double array_bytes = static_cast<double>(n);
    const double bandwidth_gb_s = (3.0 * array_bytes * static_cast<double>(runs)) / elapsed / 1e9;

    py::dict out;
    out["bandwidth_gb_s"] = bandwidth_gb_s;
    out["elapsed_sec"] = elapsed;
    out["runs"] = runs;
    out["warmup_runs"] = warmup_runs;
    out["array_bytes"] = static_cast<int64_t>(n);
    out["dtype"] = "int8";
    out["omp_max_threads"] = omp_get_max_threads();
    out["detected_pcores"] = detected_pcore_count();
    out["pin_openmp_pcores"] = asdsl_omp_pinning::pin_openmp_pcores_enabled();
    return out;
}

PYBIND11_MODULE(_native_forward, m) {
    py::class_<MmapWeights>(m, "MmapWeights")
        .def(py::init<const std::string&, py::dict>())
        .def("test_gemv_q4", &MmapWeights::test_gemv_q4)
        .def("prefetch_tensor", &MmapWeights::enqueue_prefetch_tensor, py::arg("key"), py::arg("bytes_hint") = 0);

    py::class_<KVCache>(m, "KVCache")
        .def(py::init<int, int, int, int>())
        .def("set_history", &py_kvc_set_history);

    py::class_<KVCacheQ4>(m, "KVCacheQ4")
        .def(py::init<int, int, int, int>())
        .def("set_history", &py_kvc_q4_set_history);

    m.def("set_thread_affinity", &pin_thread_to_core);
    m.def("pin_openmp_threads_to_pcores", &pin_openmp_threads_to_pcores, py::arg("enabled") = true);
    m.def("detected_pcore_count", &detected_pcore_count);
    m.def("gemv_q4k_row", &gemv_q4k_row_py, py::arg("row_bytes"), py::arg("x"), py::arg("out_dim"), py::arg("in_dim"));
    m.def("gemm_q4k_prefill", &gemm_q4k_prefill_py, py::arg("weights_bytes"), py::arg("x"), py::arg("t_rows"), py::arg("out_dim"), py::arg("in_dim"), py::arg("interleaved4") = false);
    m.def("prefill_prompt_tokens", &prefill_prompt_tokens_mmap,
        py::arg("prompt_ids"), py::arg("seq_start"), py::arg("store"),
        py::arg("num_layers"), py::arg("dim"), py::arg("hidden_dim"),
        py::arg("num_heads"), py::arg("num_kv_heads"), py::arg("head_dim"),
        py::arg("vocab_size"), py::arg("cache"));
    m.def("compute_attention", &compute_attention_flash_q8,
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("layer_id"),
        py::arg("seq_pos"), py::arg("num_heads"), py::arg("cache"));
    m.def("compute_attention_q4", &compute_attention_flash_q4,
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("layer_id"),
        py::arg("seq_pos"), py::arg("num_heads"), py::arg("cache"),
        "GQA-friendly attention with Q4 KV cache (dequant in-register per nibble).");
    m.def("generate_token", &generate_token_stub);
    m.def("generate_token", &generate_token_mmap_stub,
        py::arg("token_id"), py::arg("seq_pos"), py::arg("store"),
        py::arg("num_layers"), py::arg("dim"), py::arg("hidden_dim"),
        py::arg("num_heads"), py::arg("num_kv_heads"), py::arg("head_dim"),
        py::arg("vocab_size"), py::arg("cache"));

    m.def(
        "stream_triad_f32",
        &stream_triad_f32_py,
        py::arg("array_mb"),
        py::arg("runs"),
        py::arg("warmup_runs") = 2,
        "OpenMP-parallel STREAM Triad on float32 arrays (vectorized; preferred DRAM roofline probe).");
    m.def(
        "stream_triad_int8",
        &stream_triad_int8_py,
        py::arg("array_mb"),
        py::arg("runs"),
        py::arg("warmup_runs") = 2,
        "OpenMP-parallel STREAM Triad on int8 arrays (measures DRAM read/write volume).");
}
