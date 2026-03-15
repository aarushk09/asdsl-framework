/**
 * Native AVX2 LUT Builder + MatVec Kernel for ASDSL Framework
 *
 * Replaces the Python-loop LUT table construction and matrix-vector
 * lookup with C++ code accelerated by AVX2 intrinsics.
 *
 * Two main entry points:
 *   1. lut_build_tables:  precompute all partial-sum tables for one
 *      weight-matrix × activation-vector pair.
 *   2. lut_matvec:  perform the actual matrix-vector multiply by
 *      indexing into the precomputed tables.
 *
 * The tables are returned and consumed as flat float32 numpy arrays
 * (all per-group tables concatenated), avoiding per-table Python
 * object overhead.
 *
 * Build flags:
 *   MSVC:      /arch:AVX2 /O2 /fp:fast /openmp
 *   GCC/Clang: -mavx2 -mfma -O3 -ffast-math -fopenmp
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <immintrin.h>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

/* ===================================================================
 * AVX2 Utility
 * =================================================================== */

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

/* ===================================================================
 * LUT Table Builder  (core C++ implementation)
 *
 * For each LUT group (group_width consecutive weights), we enumerate
 * all (1 << bits)^group_width combinations and compute
 *     partial_sum = sum_w  (qval_w - half) * scale * activation[w]
 * The resulting table is stored as float32.
 *
 * For small group_width (2 or 4) and bits (2-4), the number of entries
 * is 16 to 65536 — all fit comfortably in L2 cache.
 * =================================================================== */

static void lut_build_tables_impl(
    const uint8_t* __restrict weights,   // unpacked uint8, size = out_size * in_size
    const float*   __restrict activation,// size = in_size
    const float*   __restrict scales,    // per quantisation group
    int bits,
    int group_width,
    int output_size,
    int input_size,
    int quant_group_size,                // quantisation group size for scale lookup
    float* __restrict out_tables         // output: flat table array
) {
    const int num_values = 1 << bits;                // e.g. 16 for 4-bit
    int entries_per_table = 1;
    for (int i = 0; i < group_width; ++i)
        entries_per_table *= num_values;             // num_values ^ group_width
    const int half = num_values / 4;                 // matches Python convention

    const int lut_groups_per_row = input_size / group_width;
    const int total_tables = output_size * lut_groups_per_row;

    // Precompute dequant offsets for each combination index.
    // combo_qvals[entry][w] = integer qval for position w at that entry.
    // This avoids repeated mod/div in the hot loop.
    std::vector<std::vector<int>> combo_qvals(entries_per_table,
                                               std::vector<int>(group_width));
    for (int idx = 0; idx < entries_per_table; ++idx) {
        int remaining = idx;
        for (int w = 0; w < group_width; ++w) {
            combo_qvals[idx][w] = remaining % num_values;
            remaining /= num_values;
        }
    }

    // Build all tables — parallelise over output rows.
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < output_size; ++row) {
        for (int g = 0; g < lut_groups_per_row; ++g) {
            const int col_start = g * group_width;

            // Determine scale for this quantisation group
            const int qg_idx = col_start / quant_group_size;
            const int scale_idx = row * (input_size / quant_group_size) + qg_idx;
            const float scale = scales[scale_idx];

            const int table_idx = row * lut_groups_per_row + g;
            float* table = out_tables + static_cast<size_t>(table_idx) * entries_per_table;

            // Fetch activation segment for this group
            float act_seg[16];  // group_width <= 16 for any practical config
            for (int w = 0; w < group_width; ++w)
                act_seg[w] = activation[col_start + w];

            // Enumerate all combinations and compute partial sums.
            // For group_width=2, bits=2: 16 entries  (tiny)
            // For group_width=2, bits=4: 256 entries (still small)
            // For group_width=4, bits=2: 256 entries
            // For group_width=4, bits=4: 65536 entries
            for (int idx = 0; idx < entries_per_table; ++idx) {
                float partial_sum = 0.0f;
                for (int w = 0; w < group_width; ++w) {
                    float float_val = (float)(combo_qvals[idx][w] - half) * scale;
                    partial_sum += float_val * act_seg[w];
                }
                table[idx] = partial_sum;
            }
        }
    }
}

/* ===================================================================
 * LUT MatVec  (core C++ implementation)
 *
 * For each output row, iterate over LUT groups, compute the table key
 * from the weight indices, look up the precomputed partial sum, and
 * accumulate.   Hot path is table[key] gather + float add.
 * =================================================================== */

static void lut_matvec_impl(
    const float*   __restrict tables,   // flat table array
    const uint8_t* __restrict weights,  // unpacked uint8, size = out_size * in_size
    int bits,
    int group_width,
    int output_size,
    int input_size,
    float* __restrict output            // size = output_size
) {
    const int num_values = 1 << bits;
    int entries_per_table = 1;
    for (int i = 0; i < group_width; ++i)
        entries_per_table *= num_values;

    const int lut_groups_per_row = input_size / group_width;

    // Precompute power table for key computation
    int powers[16];  // group_width <= 16
    powers[0] = 1;
    for (int w = 1; w < group_width; ++w)
        powers[w] = powers[w - 1] * num_values;

    #pragma omp parallel for schedule(static)
    for (int row = 0; row < output_size; ++row) {
        const uint8_t* row_weights = weights + static_cast<size_t>(row) * input_size;
        float row_sum = 0.0f;

        // Process LUT groups.  For group_width=2 with 4-bit:
        //   key = qval[0] + 16*qval[1], then table[key] lookup.
        for (int g = 0; g < lut_groups_per_row; ++g) {
            const int col_start = g * group_width;

            // Compute table key from weight indices
            int table_key = 0;
            for (int w = 0; w < group_width; ++w) {
                int qval = static_cast<int>(row_weights[col_start + w]);
                table_key += qval * powers[w];
            }

            // Table lookup
            const int table_idx = row * lut_groups_per_row + g;
            const float* table_ptr = tables +
                static_cast<size_t>(table_idx) * entries_per_table;

            if (table_key < entries_per_table) {
                row_sum += table_ptr[table_key];
            }
        }

        output[row] = row_sum;
    }
}

/* ===================================================================
 * Optimised LUT MatVec with AVX2 accumulation
 *
 * When group_width == 2 (the common case), the table key is just
 *   key = w0 + num_values * w1
 * which can be computed very cheaply.  We batch 8 groups and
 * accumulate partial sums using 256-bit gather + add.
 * =================================================================== */

static void lut_matvec_gw2_avx2(
    const float*   __restrict tables,
    const uint8_t* __restrict weights,
    int bits,
    int output_size,
    int input_size,
    float* __restrict output
) {
    const int num_values = 1 << bits;
    const int entries_per_table = num_values * num_values;
    const int group_width = 2;
    const int lut_groups_per_row = input_size / group_width;

    #pragma omp parallel for schedule(static)
    for (int row = 0; row < output_size; ++row) {
        const uint8_t* row_w = weights + static_cast<size_t>(row) * input_size;
        float row_sum = 0.0f;

        int g = 0;

        // AVX2 path: process 8 LUT groups at a time using VGATHERDPS
        const int avx_limit = lut_groups_per_row - 7;
        for (; g < avx_limit; g += 8) {
            // Compute 8 table keys
            alignas(32) int keys[8];
            for (int i = 0; i < 8; ++i) {
                int col = (g + i) * group_width;
                keys[i] = static_cast<int>(row_w[col])
                         + num_values * static_cast<int>(row_w[col + 1]);
            }

            // Compute flat table offsets (table_idx * entries_per_table + key)
            alignas(32) int offsets[8];
            for (int i = 0; i < 8; ++i) {
                int ti = row * lut_groups_per_row + g + i;
                offsets[i] = ti * entries_per_table + keys[i];
            }

            // VGATHERDPS: gather 8 float32 values from non-contiguous addresses
            __m256i vidx = _mm256_load_si256(
                reinterpret_cast<const __m256i*>(offsets));
            __m256 gathered = _mm256_i32gather_ps(tables, vidx, 4);

            // Horizontal sum of gathered values
            row_sum += hsum256_ps(gathered);
        }

        // Scalar tail
        for (; g < lut_groups_per_row; ++g) {
            int col = g * group_width;
            int key = static_cast<int>(row_w[col])
                     + num_values * static_cast<int>(row_w[col + 1]);
            int ti = row * lut_groups_per_row + g;
            row_sum += tables[static_cast<size_t>(ti) * entries_per_table + key];
        }

        output[row] = row_sum;
    }
}

/* ===================================================================
 * pybind11 Bindings
 * =================================================================== */

static py::array_t<float> py_lut_build_tables(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> weights,
    py::array_t<float,   py::array::c_style | py::array::forcecast> activation,
    py::array_t<float,   py::array::c_style | py::array::forcecast> scales,
    int bits,
    int group_width,
    int output_size,
    int input_size,
    int quant_group_size
) {
    auto wb = weights.request();
    auto ab = activation.request();
    auto sb = scales.request();

    if (bits < 1 || bits > 8)
        throw std::invalid_argument("bits must be in [1, 8]");
    if (group_width < 1 || group_width > 16)
        throw std::invalid_argument("group_width must be in [1, 16]");
    if (output_size <= 0 || input_size <= 0)
        throw std::invalid_argument("output_size and input_size must be positive");
    if (input_size % group_width != 0)
        throw std::invalid_argument("input_size must be divisible by group_width");

    int64_t expected_w = static_cast<int64_t>(output_size) * input_size;
    if (wb.size != expected_w)
        throw std::invalid_argument("weights size mismatch");
    if (ab.size != input_size)
        throw std::invalid_argument("activation size mismatch");

    int num_values = 1 << bits;
    int entries_per_table = 1;
    for (int i = 0; i < group_width; ++i)
        entries_per_table *= num_values;

    int lut_groups_per_row = input_size / group_width;
    int64_t total_tables = static_cast<int64_t>(output_size) * lut_groups_per_row;
    int64_t total_floats = total_tables * entries_per_table;

    auto result = py::array_t<float>(total_floats);
    auto rb = result.request();

    {
        py::gil_scoped_release release;
        lut_build_tables_impl(
            static_cast<const uint8_t*>(wb.ptr),
            static_cast<const float*>(ab.ptr),
            static_cast<const float*>(sb.ptr),
            bits, group_width, output_size, input_size, quant_group_size,
            static_cast<float*>(rb.ptr)
        );
    }

    return result;
}


static py::array_t<float> py_lut_matvec(
    py::array_t<float,   py::array::c_style | py::array::forcecast> tables,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> weights,
    int bits,
    int group_width,
    int output_size,
    int input_size
) {
    auto tb = tables.request();
    auto wb = weights.request();

    if (output_size <= 0 || input_size <= 0)
        throw std::invalid_argument("output_size and input_size must be positive");
    if (input_size % group_width != 0)
        throw std::invalid_argument("input_size must be divisible by group_width");

    auto result = py::array_t<float>(output_size);
    auto rb = result.request();

    {
        py::gil_scoped_release release;

        // Use specialised AVX2 gather path for group_width == 2
        if (group_width == 2) {
            lut_matvec_gw2_avx2(
                static_cast<const float*>(tb.ptr),
                static_cast<const uint8_t*>(wb.ptr),
                bits, output_size, input_size,
                static_cast<float*>(rb.ptr)
            );
        } else {
            lut_matvec_impl(
                static_cast<const float*>(tb.ptr),
                static_cast<const uint8_t*>(wb.ptr),
                bits, group_width, output_size, input_size,
                static_cast<float*>(rb.ptr)
            );
        }
    }

    return result;
}


PYBIND11_MODULE(_native_lut, m) {
    m.doc() = R"doc(
Native AVX2 LUT builder and matrix-vector multiply for ASDSL.

Provides two entry points:
  - lut_build_tables:  build all partial-sum LUT tables for a
    weight matrix + activation vector pair.
  - lut_matvec:  matrix-vector multiply via table lookups.

Both operate on flat numpy arrays to avoid Python object overhead.
)doc";

    m.def("lut_build_tables", &py_lut_build_tables,
        R"doc(
Build all LUT tables for a weight matrix × activation vector pair.

Args:
    weights:          Unpacked uint8 weight values, shape (output_size * input_size,).
    activation:       Input activation vector, shape (input_size,), float32.
    scales:           Per-quantisation-group scale factors, float32.
    bits:             Weight bit-width (2, 3, or 4).
    group_width:      Number of weights per LUT entry (typically 2 or 4).
    output_size:      Number of output features (rows).
    input_size:       Number of input features (columns).
    quant_group_size: Quantisation group size (for scale lookup).

Returns:
    Flat float32 array of all tables concatenated.
    Total size = output_size * (input_size / group_width) * (2^bits)^group_width.
)doc",
        py::arg("weights"), py::arg("activation"), py::arg("scales"),
        py::arg("bits"), py::arg("group_width"),
        py::arg("output_size"), py::arg("input_size"),
        py::arg("quant_group_size"));

    m.def("lut_matvec", &py_lut_matvec,
        R"doc(
LUT-based matrix-vector multiply using precomputed tables.

Args:
    tables:      Flat float32 table array (from lut_build_tables).
    weights:     Unpacked uint8 weight values, shape (output_size * input_size,).
    bits:        Weight bit-width.
    group_width: Number of weights per LUT entry.
    output_size: Number of output features.
    input_size:  Number of input features.

Returns:
    Output vector, shape (output_size,), float32.
)doc",
        py::arg("tables"), py::arg("weights"),
        py::arg("bits"), py::arg("group_width"),
        py::arg("output_size"), py::arg("input_size"));

#ifdef _OPENMP
    m.attr("has_openmp") = true;
#else
    m.attr("has_openmp") = false;
#endif
}
