#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <immintrin.h>
#include <omp.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <limits>

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

static bool g_pin_openmp_pcores = true;

#ifdef _WIN32
static inline DWORD_PTR lowest_set_bit_mask(DWORD_PTR mask) {
    if (mask == 0) {
        return 0;
    }
    return mask & (~mask + 1);
}

static std::vector<DWORD_PTR> detect_windows_pcore_masks() {
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) {
        return {};
    }

    std::vector<uint8_t> buf(len);
    auto* base = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data());
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, base, &len)) {
        return {};
    }

    uint8_t min_eff = std::numeric_limits<uint8_t>::max();
    std::vector<std::pair<uint8_t, DWORD_PTR>> core_masks;

    uint8_t* ptr = buf.data();
    uint8_t* end = buf.data() + len;
    while (ptr < end) {
        auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(ptr);
        if (info->Relationship == RelationProcessorCore && info->Processor.GroupCount > 0) {
            // SetThreadAffinityMask is group-local. Keep group 0 masks for this process group.
            const GROUP_AFFINITY& ga = info->Processor.GroupMask[0];
            if (ga.Group == 0 && ga.Mask != 0) {
                uint8_t eff = info->Processor.EfficiencyClass;
                min_eff = std::min(min_eff, eff);
                core_masks.push_back({eff, lowest_set_bit_mask(ga.Mask)});
            }
        }
        ptr += info->Size;
    }

    std::vector<DWORD_PTR> pcores;
    for (const auto& e : core_masks) {
        if (e.first == min_eff && e.second != 0) {
            pcores.push_back(e.second);
        }
    }
    if (!pcores.empty()) {
        return pcores;
    }

    // Fallback: pin across all detected cores if efficiency classes were unavailable.
    for (const auto& e : core_masks) {
        if (e.second != 0) {
            pcores.push_back(e.second);
        }
    }
    return pcores;
}

static const std::vector<DWORD_PTR>& get_pcore_masks() {
    static std::vector<DWORD_PTR> masks = detect_windows_pcore_masks();
    return masks;
}

static void bind_omp_thread_to_pcore_if_enabled() {
    if (!g_pin_openmp_pcores) {
        return;
    }
    const auto& masks = get_pcore_masks();
    if (masks.empty()) {
        return;
    }
    const int tid = omp_get_thread_num();
    const DWORD_PTR mask = masks[static_cast<size_t>(tid) % masks.size()];
    if (mask != 0) {
        SetThreadAffinityMask(GetCurrentThread(), mask);
    }
}

static void configure_openmp_for_pcores() {
    if (!g_pin_openmp_pcores) {
        return;
    }
    const auto& masks = get_pcore_masks();
    if (masks.empty()) {
        return;
    }
    omp_set_dynamic(0);
    const int target_threads = static_cast<int>(masks.size());
    if (target_threads > 0) {
        omp_set_num_threads(target_threads);
    }
}
#else
static void bind_omp_thread_to_pcore_if_enabled() {}
static void configure_openmp_for_pcores() {}
#endif

// Q4_K_M GEMV: rows are parallelized with OpenMP.
// Each row is laid out as (in_dim / 256) BlockQ4K blocks.
void gemv_q4k_avx2_rows(const BlockQ4K* weights, const float* x, float* out, int out_dim, int in_dim) {
    if (in_dim % 256 != 0) {
        throw std::runtime_error("in_dim must be divisible by 256 for BlockQ4K");
    }
    const int blocks_per_row = in_dim / 256;
    const __m256i idx_even = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i idx_odd = _mm256_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15);
    const __m256i idx_even_hi = _mm256_setr_epi32(16, 18, 20, 22, 24, 26, 28, 30);
    const __m256i idx_odd_hi = _mm256_setr_epi32(17, 19, 21, 23, 25, 27, 29, 31);

    configure_openmp_for_pcores();
    #pragma omp parallel
    {
        bind_omp_thread_to_pcore_if_enabled();

        #pragma omp for schedule(static) nowait
        for (int r = 0; r < out_dim; ++r) {
            const BlockQ4K* row = weights + static_cast<size_t>(r) * blocks_per_row;
            __m256 vacc = _mm256_setzero_ps();

            for (int b = 0; b < blocks_per_row; ++b) {
                const BlockQ4K& blk = row[b];
                const float d = f16_to_f32(blk.d);
                const float dmin = f16_to_f32(blk.dmin);

                uint8_t scales[8];
                unpack_scales_8x6(blk.scales, scales);

                const float* x_block = x + b * 256;

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

                    // Keep decoded float lanes in-register throughout the MAC chain.
                    __m256i i_lo0 = _mm256_cvtepu8_epi32(lo0);
                    __m256 fq_lo0 = _mm256_cvtepi32_ps(i_lo0);
                    fq_lo0 = _mm256_fmadd_ps(fq_lo0, v_scale, v_bias);
                    __m256 vx0 = _mm256_i32gather_ps(x_block + sb * 32, idx_even, 4);
                    vacc = _mm256_fmadd_ps(fq_lo0, vx0, vacc);

                    __m256i i_hi0 = _mm256_cvtepu8_epi32(hi0);
                    __m256 fq_hi0 = _mm256_cvtepi32_ps(i_hi0);
                    fq_hi0 = _mm256_fmadd_ps(fq_hi0, v_scale, v_bias);
                    __m256 vx1 = _mm256_i32gather_ps(x_block + sb * 32, idx_odd, 4);
                    vacc = _mm256_fmadd_ps(fq_hi0, vx1, vacc);

                    __m256i i_lo1 = _mm256_cvtepu8_epi32(lo1);
                    __m256 fq_lo1 = _mm256_cvtepi32_ps(i_lo1);
                    fq_lo1 = _mm256_fmadd_ps(fq_lo1, v_scale, v_bias);
                    __m256 vx2 = _mm256_i32gather_ps(x_block + sb * 32, idx_even_hi, 4);
                    vacc = _mm256_fmadd_ps(fq_lo1, vx2, vacc);

                    __m256i i_hi1 = _mm256_cvtepu8_epi32(hi1);
                    __m256 fq_hi1 = _mm256_cvtepi32_ps(i_hi1);
                    fq_hi1 = _mm256_fmadd_ps(fq_hi1, v_scale, v_bias);
                    __m256 vx3 = _mm256_i32gather_ps(x_block + sb * 32, idx_odd_hi, 4);
                    vacc = _mm256_fmadd_ps(fq_hi1, vx3, vacc);
                }
            }

            out[r] = hsum256_ps(vacc);
        }
    }
}

// Prefill path: tiled GEMM over prompt batch X[T, in_dim] and quantized weights W[out_dim, in_dim].
// W is loaded once per (N, K) tile and reused for TILE_M prompt rows.
void gemm_q4k_tiled_prefill(const BlockQ4K* weights, const float* x, float* y, int t_rows, int out_dim, int in_dim) {
    if (in_dim % TILE_K != 0) {
        throw std::runtime_error("in_dim must be divisible by TILE_K (256)");
    }
    const int blocks_per_row = in_dim / TILE_K;
    std::memset(y, 0, static_cast<size_t>(t_rows) * out_dim * sizeof(float));

    configure_openmp_for_pcores();
    #pragma omp parallel
    {
        bind_omp_thread_to_pcore_if_enabled();

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

                for (int nn = 0; nn < n_lim; ++nn) {
                    const BlockQ4K& blk = weights[static_cast<size_t>(n0 + nn) * blocks_per_row + block_idx];
                    const float d = f16_to_f32(blk.d);
                    const float dmin = f16_to_f32(blk.dmin);
                    uint8_t scales[8];
                    unpack_scales_8x6(blk.scales, scales);

                    for (int sb = 0; sb < 8; ++sb) {
                        const float sub_scale = (static_cast<float>(scales[sb]) * d) * (1.0f / 15.0f);
                        float wtmp[32];
                        const uint8_t* qptr = blk.qs + sb * 16;
                        for (int i = 0; i < 16; ++i) {
                            const uint8_t p = qptr[i];
                            wtmp[2 * i] = static_cast<float>(p & 0x0F) * sub_scale + dmin;
                            wtmp[2 * i + 1] = static_cast<float>((p >> 4) & 0x0F) * sub_scale + dmin;
                        }

                        for (int mm = 0; mm < m_lim; ++mm) {
                            const float* xptr = x + static_cast<size_t>(m0 + mm) * in_dim + k0 + sb * 32;
                            __m256 vsum = _mm256_setzero_ps();
                            for (int j = 0; j < 32; j += 8) {
                                __m256 vw = _mm256_loadu_ps(wtmp + j);
                                __m256 vx = _mm256_loadu_ps(xptr + j);
                                vsum = _mm256_fmadd_ps(vw, vx, vsum);
                            }
                            acc[mm][nn] += hsum256_ps(vsum);
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
            if (info.contains("shape")) {
                py::list shape = py::cast<py::list>(info["shape"]);
                if (shape.size() >= 2) {
                    rows = py::cast<int>(shape[0]);
                    cols = py::cast<int>(shape[1]);
                }
            }
            tensors[key] = TensorInfo{data + offset, rows, cols};
        }
    }

    ~MmapWeights() {
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
        gemv_q4k_avx2_rows(reinterpret_cast<const BlockQ4K*>(it->second.ptr), static_cast<const float*>(xbuf.ptr), &out, 1, in_dim);
        return out;
    }

    const TensorInfo* get_tensor(const std::string& key) const {
        auto it = tensors.find(key);
        if (it == tensors.end()) {
            return nullptr;
        }
        return &it->second;
    }

private:
    HANDLE hFile;
    HANDLE hMapping;
    uint8_t* data;
    size_t size;
    std::unordered_map<std::string, TensorInfo> tensors;
};

struct KVCache {
    KVCache(int layers, int max_seq_len, int num_kv_heads, int head_dim)
        : layers(layers), max_seq_len(max_seq_len), num_kv_heads(num_kv_heads), head_dim(head_dim) {}

    int layers;
    int max_seq_len;
    int num_kv_heads;
    int head_dim;
};

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
        in_dim);

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
    (void)seq_pos;
    (void)store;
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
        gemm_q4k_tiled_prefill(reinterpret_cast<const BlockQ4K*>(lw->ptr), x.data(), y.data(), t_rows, out_dim, in_dim);
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
    int in_dim) {
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
        in_dim);
    return out;
}

void pin_thread_to_core(int core_id) {
    DWORD_PTR mask = (1ULL << core_id);
    SetThreadAffinityMask(GetCurrentThread(), mask);
}

void pin_openmp_threads_to_pcores(bool enabled) {
    g_pin_openmp_pcores = enabled;
    if (enabled) {
        configure_openmp_for_pcores();
    }
}

int detected_pcore_count() {
#ifdef _WIN32
    return static_cast<int>(get_pcore_masks().size());
#else
    return 0;
#endif
}

PYBIND11_MODULE(_native_forward, m) {
    py::class_<MmapWeights>(m, "MmapWeights")
        .def(py::init<const std::string&, py::dict>())
        .def("test_gemv_q4", &MmapWeights::test_gemv_q4);

    py::class_<KVCache>(m, "KVCache")
        .def(py::init<int, int, int, int>());

    m.def("set_thread_affinity", &pin_thread_to_core);
    m.def("pin_openmp_threads_to_pcores", &pin_openmp_threads_to_pcores, py::arg("enabled") = true);
    m.def("detected_pcore_count", &detected_pcore_count);
    m.def("gemv_q4k_row", &gemv_q4k_row_py, py::arg("row_bytes"), py::arg("x"), py::arg("out_dim"), py::arg("in_dim"));
    m.def("gemm_q4k_prefill", &gemm_q4k_prefill_py, py::arg("weights_bytes"), py::arg("x"), py::arg("t_rows"), py::arg("out_dim"), py::arg("in_dim"));
    m.def("prefill_prompt_tokens", &prefill_prompt_tokens_mmap,
        py::arg("prompt_ids"), py::arg("seq_start"), py::arg("store"),
        py::arg("num_layers"), py::arg("dim"), py::arg("hidden_dim"),
        py::arg("num_heads"), py::arg("num_kv_heads"), py::arg("head_dim"),
        py::arg("vocab_size"), py::arg("cache"));
    m.def("generate_token", &generate_token_stub);
    m.def("generate_token", &generate_token_mmap_stub,
        py::arg("token_id"), py::arg("seq_pos"), py::arg("store"),
        py::arg("num_layers"), py::arg("dim"), py::arg("hidden_dim"),
        py::arg("num_heads"), py::arg("num_kv_heads"), py::arg("head_dim"),
        py::arg("vocab_size"), py::arg("cache"));
}
