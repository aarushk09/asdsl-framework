#pragma once

#include <cstdlib>

namespace asdsl {

/** ASDSL_PERSISTENT_POOL=1: use pinned ThreadPool instead of per-op OpenMP teams. */
inline bool persistent_pool_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* s = std::getenv("ASDSL_PERSISTENT_POOL");
        v = (s && s[0] == '1') ? 1 : 0;
    }
    return v != 0;
}

/** ASDSL_C03=1: extend g128 byte diet to qkv_proj and o_proj (requires C0.1 gate_up/down). */
inline bool c03_gemv_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* s = std::getenv("ASDSL_C03");
        v = (s && s[0] == '1') ? 1 : 0;
    }
    return v != 0;
}

}  // namespace asdsl
