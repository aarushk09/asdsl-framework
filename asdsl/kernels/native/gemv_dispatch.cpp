#include <cpuid.h>
#include <cstdint>

bool cpu_has_avx512_vnni() {
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(7, &eax, &ebx, &ecx, &edx) == 0) return false;
    return (ecx >> 11) & 1;
}

bool cpu_has_avx2_vnni() {
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(7, &eax, &ebx, &ecx, &edx) == 0) return false;
    return (eax >> 4) & 1;
}

// Fallback/dummy scalar kernel
extern "C" void gemv_int8_scalar(const int8_t* a, const int8_t* b, int32_t* c, int m, int n) {}
extern "C" void gemv_int8_avx2_vnni(const int8_t* a, const int8_t* b, int32_t* c, int m, int n) {}
extern "C" void gemv_int8_avx512_vnni(const int8_t* a, const int8_t* b, int32_t* c, int m, int n) {}

typedef void (*GemvKernelFn)(const int8_t*, const int8_t*, int32_t*, int, int);

extern "C" GemvKernelFn select_gemv_kernel() {
    if (cpu_has_avx512_vnni()) return &gemv_int8_avx512_vnni;
    if (cpu_has_avx2_vnni())   return &gemv_int8_avx2_vnni;
    return &gemv_int8_scalar; 
}
