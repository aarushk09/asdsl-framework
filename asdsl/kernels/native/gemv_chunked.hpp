/**
 * Atomic chunked OpenMP work distribution (ggml-style current_chunk).
 * Used by preq / preq2 GEMV when ASDSL_CHUNKED_GEMV=1.
 */
#pragma once

#include <atomic>
#include <cstdlib>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace asdsl_chunked {

inline bool chunked_gemv_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char* v = std::getenv("ASDSL_CHUNKED_GEMV");
        if (!v || v[0] == '\0') {
            enabled = 0;
        } else if (v[0] == '0') {
            enabled = 0;
        } else {
            enabled = 1;
        }
    }
    return enabled != 0;
}

inline int chunk_divisor() {
    static int div = -1;
    if (div < 0) {
        const char* v = std::getenv("ASDSL_GEMV_CHUNK_DIV");
        if (!v || v[0] == '\0') {
            div = 4;
        } else {
            div = std::atoi(v);
            if (div < 1) {
                div = 1;
            }
        }
    }
    return div;
}

inline int compute_chunk_size(int out_features, int row_tile, int n_threads) {
    if (n_threads < 1) {
        n_threads = 1;
    }
    const int div = chunk_divisor();
    int chunk = out_features / (n_threads * div);
    if (chunk < row_tile) {
        chunk = row_tile;
    }
    return (chunk / row_tile) * row_tile;
}

template <typename Fn>
inline void parallel_row_chunks(int out_features, int row_tile, Fn&& body) {
#ifdef _OPENMP
    const int n_threads = omp_get_max_threads();
    const int chunk_size = compute_chunk_size(out_features, row_tile, n_threads);
    const int n_chunks = (out_features + chunk_size - 1) / chunk_size;
    std::atomic<int> next_chunk{0};

    #pragma omp parallel
    {
        for (;;) {
            const int chunk = next_chunk.fetch_add(1, std::memory_order_relaxed);
            if (chunk >= n_chunks) {
                break;
            }
            const int row0 = chunk * chunk_size;
            const int row1 = row0 + chunk_size;
            const int row_end = (row1 < out_features) ? row1 : out_features;
            body(row0, row_end);
        }
    }
#else
    body(0, out_features);
#endif
}

}  // namespace asdsl_chunked
