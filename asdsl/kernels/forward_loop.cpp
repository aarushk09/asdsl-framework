#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <windows.h>
#include <immintrin.h>
#include <thread>
#include <vector>
#include <iostream>
#include <cmath>
#include <omp.h>

namespace py = pybind11;

// Step 1: Strict Memory Layout
#pragma pack(push, 1)
struct BlockQ4_32 {
    uint16_t scale;       // 2 bytes (FP16 scale for the group)
    uint8_t weights[16];  // 16 bytes (32 weights * 4 bits = 128 bits)
};
#pragma pack(pop)

// Step 2: Zero-Copy mmap (Windows)
class MmapWeights {
    HANDLE hFile;
    HANDLE hMapping;
public:
    uint8_t* data;
    size_t size;
    std::unordered_map<std::string, uint8_t*> tensors;

    MmapWeights(const std::string& filepath, py::dict metadata) : data(nullptr), hMapping(NULL), hFile(INVALID_HANDLE_VALUE) {
        hFile = CreateFileA(filepath.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile == INVALID_HANDLE_VALUE) throw std::runtime_error("Failed to open file");
        
        LARGE_INTEGER li;
        GetFileSizeEx(hFile, &li);
        size = li.QuadPart;

        hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        if (!hMapping) throw std::runtime_error("Failed to create mapping");

        data = (uint8_t*)MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
        if (!data) throw std::runtime_error("Failed to map view");

        for (auto item : metadata) {
            std::string key = py::cast<std::string>(item.first);
            py::dict info = py::cast<py::dict>(item.second);
            size_t offset = py::cast<size_t>(info["offset"]);
            tensors[key] = data + offset;
        }
    }

    void test_gemv_q4() {}

    ~MmapWeights() {
        if (data) UnmapViewOfFile(data);
        if (hMapping) CloseHandle(hMapping);
        if (hFile != INVALID_HANDLE_VALUE) CloseHandle(hFile);
    }
};

void gemv_q4_avx2(const BlockQ4_32* weights, const float* activations, float* output, int num_blocks) {
    for (int b = 0; b < num_blocks; ++b) {
        const BlockQ4_32& block = weights[b];
    }
}

void gemv_f32(const float* weights, const float* activations, float* output, int out_dim, int in_dim) {
    #pragma omp parallel for
    for (int i = 0; i < out_dim; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < in_dim; ++j) {
            sum += weights[i * in_dim + j] * activations[j];
        }
        output[i] += sum;
    }
}

void apply_rmsnorm(float* x, const float* weight, int size, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < size; ++i) sum_sq += x[i] * x[i];
    float rms = std::sqrt(sum_sq / size + eps);
    float inv_rms = 1.0f / rms;
    for (int i = 0; i < size; ++i) {
        x[i] = x[i] * inv_rms * weight[i];
    }
}

void apply_rope(float* q, float* k, int seq_pos, int head_dim, int num_heads, int num_kv_heads, float theta) {
    int half_dim = head_dim / 2;
    for (int d = 0; d < half_dim; ++d) {
        float freq = seq_pos * (1.0f / std::pow(theta, (float)(2 * d) / head_dim));
        float cos_f = std::cos(freq);
        float sin_f = std::sin(freq);

        for (int h = 0; h < num_heads; ++h) {
            float* q_p = q + h * head_dim + d;
            float* q_p2 = q_p + half_dim;
            float q1 = *q_p;
            float q2 = *q_p2;
            *q_p = q1 * cos_f - q2 * sin_f;
            *q_p2 = q2 * cos_f + q1 * sin_f;
        }

        for (int h = 0; h < num_kv_heads; ++h) {
            float* k_p = k + h * head_dim + d;
            float* k_p2 = k_p + half_dim;
            float k1 = *k_p;
            float k2 = *k_p2;
            *k_p = k1 * cos_f - k2 * sin_f;
            *k_p2 = k2 * cos_f + k1 * sin_f;
        }
    }
}

void pin_thread_to_core(int core_id) {
    DWORD_PTR mask = (1ULL << core_id);
    SetThreadAffinityMask(GetCurrentThread(), mask);
}

struct KVCache {
    int num_layers;
    int max_seq_len;
    int num_kv_heads;
    int head_dim;
    std::vector<float> k_cache;
    std::vector<float> v_cache;

    KVCache(int layers, int seq_len, int kv_heads, int dim)
        : num_layers(layers), max_seq_len(seq_len), num_kv_heads(kv_heads), head_dim(dim) {
        size_t size = (size_t)layers * seq_len * kv_heads * dim;
        k_cache.resize(size, 0.0f);
        v_cache.resize(size, 0.0f);
    }

    inline size_t get_offset(int layer, int pos) const {
        return ((size_t)layer * max_seq_len + pos) * num_kv_heads * head_dim;
    }

    void set_history(int layer, int pos, const float* k, const float* v) {
        size_t off = get_offset(layer, pos);
        memcpy(&k_cache[off], k, num_kv_heads * head_dim * sizeof(float));
        memcpy(&v_cache[off], v, num_kv_heads * head_dim * sizeof(float));
    }
};

void compute_attention(float* out, const float* q, const float* k, const float* v, int layer_id, int seq_pos, int num_heads, KVCache& cache) {
    int num_kv_heads = cache.num_kv_heads;
    int head_dim = cache.head_dim;
    float scale = 1.0f / std::sqrt((float)head_dim);
    int groups = num_heads / num_kv_heads;

    size_t cache_off = cache.get_offset(layer_id, seq_pos);
    memcpy(&cache.k_cache[cache_off], k, num_kv_heads * head_dim * sizeof(float));      
    memcpy(&cache.v_cache[cache_off], v, num_kv_heads * head_dim * sizeof(float));      

    #pragma omp parallel for
    for (int h = 0; h < num_heads; ++h) {
        std::vector<float> scores(seq_pos + 1);
        int kv_h = h / groups;
        const float* q_h = q + h * head_dim;
        float* out_h = out + h * head_dim;
        float max_score = -1e9f;

        for (int p = 0; p <= seq_pos; ++p) {
            const float* k_p = &cache.k_cache[cache.get_offset(layer_id, p) + kv_h * head_dim];
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += q_h[d] * k_p[d];
            }
            score *= scale;
            scores[p] = score;
            if (score > max_score) max_score = score;
        }

        float exp_sum = 0.0f;
        for (int p = 0; p <= seq_pos; ++p) {
            scores[p] = std::exp(scores[p] - max_score);
            exp_sum += scores[p];
        }
        for (int p = 0; p <= seq_pos; ++p) {
            scores[p] /= exp_sum;
        }

        for (int d = 0; d < head_dim; ++d) {
            out_h[d] = 0.0f;
        }
        for (int p = 0; p <= seq_pos; ++p) {
            float s = scores[p];
            const float* v_p = &cache.v_cache[cache.get_offset(layer_id, p) + kv_h * head_dim];
            for (int d = 0; d < head_dim; ++d) {
                out_h[d] += s * v_p[d];
            }
        }
    }
}

void forward_layer(float* x, const float* rms1_w, const float* qkv_w, const float* o_w, const float* rms2_w, 
                  const float* gate_w, const float* up_w, const float* down_w,
                  int dim, int hidden_dim, int num_heads, int num_kv_heads, int head_dim, 
                  int layer_id, int seq_pos, KVCache& cache) {
    
    std::vector<float> res1(dim);
    memcpy(res1.data(), x, dim * sizeof(float));

    std::vector<float> h(dim);
    memcpy(h.data(), x, dim * sizeof(float));
    
    apply_rmsnorm(h.data(), rms1_w, dim, 1e-5f);

    int q_dim = num_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;
    int qkv_dim = q_dim + 2 * kv_dim;
    std::vector<float> qkv(qkv_dim, 0.0f);
    
    gemv_f32(qkv_w, h.data(), qkv.data(), qkv_dim, dim);
    
    float* q = qkv.data();
    float* k = qkv.data() + q_dim;
    float* v = qkv.data() + q_dim + kv_dim;

    apply_rope(q, k, seq_pos, head_dim, num_heads, num_kv_heads, 10000.0f);

    std::vector<float> attn_out(q_dim, 0.0f);
    compute_attention(attn_out.data(), q, k, v, layer_id, seq_pos, num_heads, cache);

    std::vector<float> o_out(dim, 0.0f);
    gemv_f32(o_w, attn_out.data(), o_out.data(), dim, q_dim);

    for(int i=0; i<dim; ++i) x[i] = res1[i] + o_out[i];
    
    std::vector<float> res2(dim);
    memcpy(res2.data(), x, dim * sizeof(float));

    memcpy(h.data(), x, dim * sizeof(float));
    apply_rmsnorm(h.data(), rms2_w, dim, 1e-5f);

    std::vector<float> gate(hidden_dim, 0.0f);
    std::vector<float> up(hidden_dim, 0.0f);

    gemv_f32(gate_w, h.data(), gate.data(), hidden_dim, dim);
    gemv_f32(up_w, h.data(), up.data(), hidden_dim, dim);

    #pragma omp parallel for
    for (int i=0; i<hidden_dim; ++i) {
        float x_gate = gate[i];
        float silu = x_gate / (1.0f + std::exp(-x_gate));
        gate[i] = silu * up[i];
    }

    std::vector<float> down(dim, 0.0f);
    gemv_f32(down_w, gate.data(), down.data(), dim, hidden_dim);

    for(int i=0; i<dim; ++i) x[i] = res2[i] + down[i];
}


// Pybind wrappers
void py_kvc_set_history(KVCache& cache, int layer, int pos, py::array_t<float> k, py::array_t<float> v) {
    py::buffer_info k_buf = k.request();
    py::buffer_info v_buf = v.request();
    cache.set_history(layer, pos, (const float*)k_buf.ptr, (const float*)v_buf.ptr);    
}

py::array_t<float> py_compute_attention(py::array_t<float> q, py::array_t<float> k, py::array_t<float> v, int layer_id, int seq_pos, int num_heads, KVCache& cache) {
    py::buffer_info q_buf = q.request();
    py::buffer_info k_buf = k.request();
    py::buffer_info v_buf = v.request();
    auto out = py::array_t<float>(q_buf.size);
    py::buffer_info out_buf = out.request();
    compute_attention((float*)out_buf.ptr, (const float*)q_buf.ptr, (const float*)k_buf.ptr, (const float*)v_buf.ptr, layer_id, seq_pos, num_heads, cache);
    return out;
}

void py_apply_rmsnorm(py::array_t<float> x, py::array_t<float> weight, float eps) {     
    py::buffer_info x_buf = x.request();
    py::buffer_info w_buf = weight.request();
    apply_rmsnorm((float*)x_buf.ptr, (const float*)w_buf.ptr, x_buf.shape[0], eps);
}

void py_apply_rope(py::array_t<float> q, py::array_t<float> k, int seq_pos, int head_dim, int num_heads, int num_kv_heads, float theta) {
    py::buffer_info q_buf = q.request();
    py::buffer_info k_buf = k.request();
    apply_rope((float*)q_buf.ptr, (float*)k_buf.ptr, seq_pos, head_dim, num_heads, num_kv_heads, theta);
}

void py_forward_layer(py::array_t<float> x, py::array_t<float> rms1_w, py::array_t<float> qkv_w, py::array_t<float> o_w, 
                     py::array_t<float> rms2_w, py::array_t<float> gate_w, py::array_t<float> up_w, py::array_t<float> down_w, 
                     int dim, int hidden_dim, int num_heads, int num_kv_heads, int head_dim, 
                     int layer_id, int seq_pos, KVCache& cache) {
    py::buffer_info x_buf = x.request();
    py::buffer_info rms1_buf = rms1_w.request();
    py::buffer_info qkv_buf = qkv_w.request();
    py::buffer_info o_buf = o_w.request();
    py::buffer_info rms2_buf = rms2_w.request();
    py::buffer_info gate_buf = gate_w.request();
    py::buffer_info up_buf = up_w.request();
    py::buffer_info down_buf = down_w.request();

    forward_layer(
        (float*)x_buf.ptr, (const float*)rms1_buf.ptr, (const float*)qkv_buf.ptr, (const float*)o_buf.ptr,
        (const float*)rms2_buf.ptr, (const float*)gate_buf.ptr, (const float*)up_buf.ptr, (const float*)down_buf.ptr,
        dim, hidden_dim, num_heads, num_kv_heads, head_dim, layer_id, seq_pos, cache
    );
}


int generate_token(int token_id, int seq_pos, 
                   const float* token_emb,
                   const std::vector<const float*>& rms1_ws,
                   const std::vector<const float*>& qkv_ws,
                   const std::vector<const float*>& o_ws,
                   const std::vector<const float*>& rms2_ws,
                   const std::vector<const float*>& gate_ws,
                   const std::vector<const float*>& up_ws,
                   const std::vector<const float*>& down_ws,
                   const float* final_rms_w,
                   const float* lm_head_w,
                   int num_layers, int dim, int hidden_dim, int num_heads, int num_kv_heads, int head_dim, int vocab_size,
                   KVCache& cache) {
    
    std::vector<float> x(dim);
    memcpy(x.data(), token_emb + (size_t)token_id * dim, dim * sizeof(float));

    for (int l = 0; l < num_layers; ++l) {
        forward_layer(x.data(), rms1_ws[l], qkv_ws[l], o_ws[l], rms2_ws[l], gate_ws[l], up_ws[l], down_ws[l],
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

int py_generate_token(int token_id, int seq_pos, 
                      py::array_t<float> token_emb,
                      py::list layers_weights,
                      py::array_t<float> final_rms_w,
                      py::array_t<float> lm_head_w,
                      int num_layers, int dim, int hidden_dim, int num_heads, int num_kv_heads, int head_dim, int vocab_size,
                      KVCache& cache) {
    
    py::buffer_info emb_buf = token_emb.request();
    py::buffer_info frms_buf = final_rms_w.request();
    py::buffer_info lm_buf = lm_head_w.request();

    std::vector<const float*> rms1_ws, qkv_ws, o_ws, rms2_ws, gate_ws, up_ws, down_ws;
    std::vector<py::array_t<float>> refs;

    for (int l = 0; l < num_layers; ++l) {
        py::tuple layer = layers_weights[l].cast<py::tuple>();
        refs.push_back(layer[0].cast<py::array_t<float>>()); rms1_ws.push_back((const float*)refs.back().request().ptr);
        refs.push_back(layer[1].cast<py::array_t<float>>()); qkv_ws.push_back((const float*)refs.back().request().ptr);
        refs.push_back(layer[2].cast<py::array_t<float>>()); o_ws.push_back((const float*)refs.back().request().ptr);
        refs.push_back(layer[3].cast<py::array_t<float>>()); rms2_ws.push_back((const float*)refs.back().request().ptr);
        refs.push_back(layer[4].cast<py::array_t<float>>()); gate_ws.push_back((const float*)refs.back().request().ptr);
        refs.push_back(layer[5].cast<py::array_t<float>>()); up_ws.push_back((const float*)refs.back().request().ptr);
        refs.push_back(layer[6].cast<py::array_t<float>>()); down_ws.push_back((const float*)refs.back().request().ptr);
    }

    return generate_token(token_id, seq_pos, 
                          (const float*)emb_buf.ptr,
                          rms1_ws, qkv_ws, o_ws, rms2_ws, gate_ws, up_ws, down_ws,
                          (const float*)frms_buf.ptr, (const float*)lm_buf.ptr,
                          num_layers, dim, hidden_dim, num_heads, num_kv_heads, head_dim, vocab_size,
                          cache);
}

PYBIND11_MODULE(_native_forward, m) {
    py::class_<MmapWeights>(m, "MmapWeights")
        .def(py::init<const std::string&, py::dict>())
        .def("test_gemv_q4", &MmapWeights::test_gemv_q4);

    py::class_<KVCache>(m, "KVCache")
        .def(py::init<int, int, int, int>())
        .def("set_history", &py_kvc_set_history);

    m.def("compute_attention", &py_compute_attention, "Compute SDPA");
    m.def("set_thread_affinity", &pin_thread_to_core);
    m.def("apply_rmsnorm", &py_apply_rmsnorm, "Apply RMSNorm in-place",
          py::arg("x"), py::arg("weight"), py::arg("eps") = 1e-5f);
    m.def("apply_rope", &py_apply_rope, "Apply RoPE in-place",
          py::arg("q"), py::arg("k"), py::arg("seq_pos"), py::arg("head_dim"),  
          py::arg("num_heads"), py::arg("num_kv_heads"), py::arg("theta") = 10000.0f);
    m.def("forward_layer", &py_forward_layer, "Forward layer execution");
    m.def("generate_token", &py_generate_token, "Generate a single token");
}