#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <windows.h>
#include <immintrin.h>
#include <thread>
#include <vector>
#include <iostream>

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

    MmapWeights(const std::string& filepath, py::dict metadata) {
        hFile = CreateFileA(filepath.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile == INVALID_HANDLE_VALUE) throw std::runtime_error("Failed to open file");
        
        LARGE_INTEGER li;
        GetFileSizeEx(hFile, &li);
        size = li.QuadPart;
        
        hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        if (!hMapping) throw std::runtime_error("Failed to create mapping");
        
        data = (uint8_t*)MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
        if (!data) throw std::runtime_error("Failed to map view");

        // Parse offsets from the metadata JSON passed as a Python dict
        for (auto item : metadata) {
            std::string key = py::cast<std::string>(item.first);
            py::dict info = py::cast<py::dict>(item.second);
            size_t offset = py::cast<size_t>(info["offset"]);
            tensors[key] = data + offset;
        }

        // --- DEBUG PRINT ---
        std::string test_key = "model.layers.0.self_attn.qkv_proj.base_layer.weight";
        if (tensors.find(test_key) != tensors.end()) {
            std::cout << "[C++ MMAP DEBUG] First 5 bytes of " << test_key << ": ";
            uint8_t* ptr = tensors[test_key];
            for (int i = 0; i < 5; i++) {
                printf("%02X ", ptr[i]);
            }
            std::cout << std::endl;
        } else {
            std::cout << "[C++ MMAP DEBUG] Key " << test_key << " not found in metadata!" << std::endl;
        }
    }
    
    ~MmapWeights() {
        if (data) UnmapViewOfFile(data);
        if (hMapping) CloseHandle(hMapping);
        if (hFile != INVALID_HANDLE_VALUE) CloseHandle(hFile);
    }
};

// Step 3: AVX2 Unpacking and MatVec (Simplified core loop)
void gemv_q4_avx2(const BlockQ4_32* weights, const float* activations, float* output, int num_blocks) {
    // Note: FP16 to FP32 conversion requires F16C extension (_mm256_cvtph_ps)
    for (int b = 0; b < num_blocks; ++b) {
        const BlockQ4_32& block = weights[b];
        
        // Load half-precision scale (convert to float format logic here - omitted for brevity)
        float scale = 1.0f; // placeholder for _cvtsh_ss(block.scale)

        // Load 16 bytes of 4-bit weights
        __m128i raw_w = _mm_loadu_si128((const __m128i*)block.weights);
        
        // Extract lower 4 bits
        __m128i mask = _mm_set1_epi8(0x0F);
        __m128i lower = _mm_and_si128(raw_w, mask);
        
        // Extract upper 4 bits
        __m128i upper = _mm_and_si128(_mm_srli_epi16(raw_w, 4), mask);
        
        // (Convert lower/upper to float and perform 32-element dot product with activations... omitted full unroll for brevity)
    }
}

// Step 4: Strangle Thread Pool to Physical Cores
void pin_thread_to_core(int core_id) {
    DWORD_PTR mask = (1ULL << core_id);
    SetThreadAffinityMask(GetCurrentThread(), mask);
}

// Step 5: C++ KV Cache
struct KVCache {
    std::vector<float> keys;
    std::vector<float> values;
    int max_seq_len;
    int head_dim;
    int num_heads;
    int current_seq_len;
    
    KVCache(int max_len, int heads, int dim) 
        : max_seq_len(max_len), num_heads(heads), head_dim(dim), current_seq_len(0) {
        keys.resize(max_len * heads * dim);
        values.resize(max_len * heads * dim);
    }
    
    void append(const float* k, const float* v) {
        if (current_seq_len >= max_seq_len) return; // overflow
        size_t offset = current_seq_len * num_heads * head_dim;
        memcpy(keys.data() + offset, k, num_heads * head_dim * sizeof(float));
        memcpy(values.data() + offset, v, num_heads * head_dim * sizeof(float));
        current_seq_len++;
    }
};

PYBIND11_MODULE(_native_forward, m) {
    py::class_<MmapWeights>(m, "MmapWeights")
        .def(py::init<const std::string&, py::dict>());

    py::class_<KVCache>(m, "KVCache")
        .def(py::init<int, int, int>());
        
    m.def("set_thread_affinity", &pin_thread_to_core);
}
