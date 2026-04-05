#include <immintrin.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>

// Mock of ASB Header
struct __attribute__((packed)) ASBHeader {
    uint8_t bit_width;
    uint8_t group_size; // typically 32
    uint16_t scale;     // fp16 mock
    uint16_t zero;      // fp16 mock
    uint16_t reserved;
};

// Convert fp16->fp32
static inline float fp16_to_float32(uint16_t h) {
#if defined(__F16C__)
    return _cvtsh_ss(h);
#else
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    uint32_t exp = (static_cast<uint32_t>(h) >> 10) & 0x1Fu;
    uint32_t mant = static_cast<uint32_t>(h) & 0x03FFu;
    if (exp == 0) return 0.0f;
    uint32_t exp32 = exp + (127 - 15);
    uint32_t bits = sign | (exp32 << 23) | (mant << 13);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
#endif
}

// Random initialization
void init_asb_buffer(std::vector<uint8_t>& buffer, int num_groups, bool sorted) {
    buffer.clear();
    std::vector<uint8_t> widths = {2, 3, 4, 8};
    std::mt19937 gen(42);
    
    std::vector<uint8_t> group_widths(num_groups);
    for(int i=0; i<num_groups; ++i) {
        group_widths[i] = widths[gen() % 4];
    }
    
    if (sorted) {
        std::sort(group_widths.begin(), group_widths.end());
    }
    
    for(int i=0; i<num_groups; ++i) {
        ASBHeader h;
        h.bit_width = group_widths[i];
        h.group_size = 32;
        h.scale = 0x3C00; // 1.0 in fp16
        h.zero = 0;
        h.reserved = 0;
        
        const uint8_t* h_ptr = reinterpret_cast<const uint8_t*>(&h);
        buffer.insert(buffer.end(), h_ptr, h_ptr + sizeof(ASBHeader));
        
        int bytes = (32 * h.bit_width) / 8;
        for(int b=0; b<bytes; ++b) {
            buffer.push_back(static_cast<uint8_t>(gen() % 255));
        }
    }
}

// Mock sum dot product
float benchmark_asb_dispatch(const std::vector<uint8_t>& buffer, const std::vector<float>& x) {
    float acc = 0.0f;
    int offset = 0;
    int group_idx = 0;
    
    int size = buffer.size();
    while (offset < size) {
        const ASBHeader* h = reinterpret_cast<const ASBHeader*>(buffer.data() + offset);
        offset += sizeof(ASBHeader);
        
        float scale = fp16_to_float32(h->scale);
        
        // Emulate the dispatch overhead + minimal scalar work representing the AVX2 payload
        switch(h->bit_width) {
            case 2: {
                int bytes = (h->group_size * 2) / 8;
                for(int i=0; i<bytes; ++i) acc += buffer[offset+i] * x[group_idx*32 + i] * scale;
                offset += bytes;
                break;
            }
            case 3: {
                // not aligned to byte perfectly, just mock bytes used roughly 12 bytes
                int bytes = 12;
                for(int i=0; i<bytes; ++i) acc += buffer[offset+i] * x[group_idx*32 + i] * scale;
                offset += bytes;
                break;
            }
            case 4: {
                int bytes = (h->group_size * 4) / 8;
                for(int i=0; i<bytes; ++i) acc += buffer[offset+i] * x[group_idx*32 + i] * scale;
                offset += bytes;
                break;
            }
            case 8: {
                int bytes = (h->group_size * 8) / 8;
                for(int i=0; i<bytes; ++i) acc += buffer[offset+i] * x[group_idx*32 + i] * scale;
                offset += bytes;
                break;
            }
            default:
                std::cout << "ERR " << (int)h->bit_width << std::endl;
                break;
        }
        group_idx++;
    }
    return acc;
}

int main() {
    int num_groups = 1000000; // About 32M weights (~32MB)
    std::vector<float> x(num_groups * 32, 1.0f);
    
    std::vector<uint8_t> buffer_random;
    init_asb_buffer(buffer_random, num_groups, false);
    
    std::vector<uint8_t> buffer_sorted;
    init_asb_buffer(buffer_sorted, num_groups, true);
    
    // Warmup
    benchmark_asb_dispatch(buffer_random, x);
    
    // Random Benchmark
    auto t1 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<100; ++i) benchmark_asb_dispatch(buffer_random, x);
    auto t2 = std::chrono::high_resolution_clock::now();
    double ms_rand = std::chrono::duration<double, std::milli>(t2 - t1).count();
    
    // Sorted Benchmark
    auto t3 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<100; ++i) benchmark_asb_dispatch(buffer_sorted, x);
    auto t4 = std::chrono::high_resolution_clock::now();
    double ms_sort = std::chrono::duration<double, std::milli>(t4 - t3).count();
    
    std::cout << "ASB Dispatch Benchmark (100 runs over 32M weights):" << std::endl;
    std::cout << "Random (Unpredictable Branches): " << ms_rand << " ms" << std::endl;
    std::cout << "Sorted (Predictable Branches):   " << ms_sort << " ms" << std::endl;
    std::cout << "Penalty for random layout:       " << (ms_rand / ms_sort - 1.0) * 100.0 << "%" << std::endl;
    return 0;
}
