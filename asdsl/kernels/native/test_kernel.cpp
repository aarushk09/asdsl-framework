/**
 * Standalone verification for gemv_q4_avx2 (group size 64).
 *
 * Build (GCC/Clang, from repo root):
 *   g++ -std=c++17 -O2 -mavx2 -mfma -I asdsl/kernels/native \
 *       asdsl/kernels/native/test_kernel.cpp \
 *       asdsl/kernels/native/gemv_q4_kernel.cpp -o test_kernel
 *
 * Build (MSVC x64, from repo root):
 *   cl /std:c++17 /arch:AVX2 /O2 /EHsc /I asdsl/kernels/native ^
 *      asdsl/kernels/native/test_kernel.cpp asdsl/kernels/native/gemv_q4_kernel.cpp
 *
 * Run: ./test_kernel  (exit 0 on success)
 *
 * Case: 1 row x 64 columns, every packed byte 0x77 (both nibbles 7), scale 1.0,
 *       x all 1.0 -> dot = 64 * 7 = 448.0
 */

#include "gemv_q4_kernel.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

int main() {
    alignas(32) uint8_t w_packed[32];
    std::memset(w_packed, 0x77, sizeof(w_packed));

    alignas(32) float x[64];
    for (int i = 0; i < 64; ++i) {
        x[i] = 1.0f;
    }

    float scales[1] = {1.0f};
    float out[1] = {0.0f};

    gemv_q4_avx2(w_packed, x, scales, out, 1, 64);

    const float expected = 448.0f;
    if (out[0] != expected) {
        std::printf("FAIL: out[0] = %.8g (expected exactly %g)\n",
                    static_cast<double>(out[0]),
                    static_cast<double>(expected));
        return 1;
    }

    std::printf("PASS: gemv_q4_avx2 1x64 packed 0x77, scale 1, x=1 -> %g\n",
                static_cast<double>(out[0]));
    return 0;
}
