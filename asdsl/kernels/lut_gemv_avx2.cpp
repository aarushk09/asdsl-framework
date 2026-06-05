/**

 * Phase 2/3 LUT GEMV: F16C gather + AVX2 FMA for prebuilt dequant tables.

 *

 * T_tile[g][q][i] = (q - zero[g]) * scale[g] as float16 (Phase 1 layout).

 * Partial dot: sum_{g,j} T[g, q_vals[g,j], j] * x[g,j] — no bias*sum(x).

 *

 * Build: python asdsl/kernels/setup_lut.py build_ext --inplace

 */



#include <pybind11/pybind11.h>

#include <pybind11/numpy.h>



#include <immintrin.h>

#include <algorithm>

#include <cstdint>

#include <cstddef>

#include <stdexcept>

#include <string>

#include <vector>



#if defined(_MSC_VER)

#include <intrin.h>

#elif defined(__GNUC__) || defined(__clang__)

#include <cpuid.h>

#endif



#ifdef _OPENMP

#include <omp.h>

#endif



namespace py = pybind11;



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



static bool check_avx2_support() {

#if defined(_MSC_VER)

    int info[4];

    __cpuidex(info, 7, 0);

    return (info[1] & (1 << 5)) != 0;

#elif defined(__GNUC__) || defined(__clang__)

    unsigned int eax, ebx, ecx, edx;

    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {

        return (ebx & (1 << 5)) != 0;

    }

    return false;

#else

    return false;

#endif

}



static bool check_f16c_support() {

#if defined(_MSC_VER)

    int info[4];

    __cpuid(info, 1);

    return (info[2] & (1 << 29)) != 0;

#elif defined(__GNUC__) || defined(__clang__)

    unsigned int eax, ebx, ecx, edx;

    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {

        return (ecx & (1 << 29)) != 0;

    }

    return false;

#else

    return false;

#endif

}



static void require_simd() {

    if (!check_avx2_support() || !check_f16c_support()) {

        throw std::runtime_error(

            "CPU lacks AVX2 and/or F16C required for asdsl_lut_avx2");

    }

}



static float lut_gemv_avx2_tile_impl(

    const uint16_t* T,

    const uint8_t* q_vals,

    const float* x_tile,

    int n_groups,

    int group_size,

    int lut_levels

) {

    const int stride_T_g = lut_levels * group_size;

    const int stride_T_q = group_size;

    __m256 acc = _mm256_setzero_ps();



    for (int g = 0; g < n_groups; ++g) {

        const uint16_t* Tg = T + static_cast<size_t>(g) * stride_T_g;

        const uint8_t* qg = q_vals + static_cast<size_t>(g) * group_size;

        const float* xg = x_tile + static_cast<size_t>(g) * group_size;



        for (int j = 0; j < group_size; j += 8) {

            __m256 xv = _mm256_loadu_ps(xg + j);

            alignas(16) uint16_t h8[8];

            for (int k = 0; k < 8; ++k) {

                const int col = j + k;

                const uint8_t q = static_cast<uint8_t>(qg[col] & 0x0F);

                h8[k] = Tg[static_cast<size_t>(q) * stride_T_q + col];

            }

            const __m128i hvec = _mm_loadu_si128(

                reinterpret_cast<const __m128i*>(h8));

            const __m256 tv = _mm256_cvtph_ps(hvec);

            acc = _mm256_fmadd_ps(tv, xv, acc);

        }

    }

    return hsum256_ps(acc);

}



static void validate_tile_shapes(

    const py::buffer_info& Tb,

    const py::buffer_info& qb,

    const py::buffer_info& xb,

    int expected_groups

) {

    if (Tb.ndim != 3) {

        throw std::invalid_argument("T_tile must be 3-D float16 [G, 16, group_size]");

    }

    if (qb.ndim != 2 || xb.ndim != 2) {

        throw std::invalid_argument("q_vals and x_tile must be 2-D [G, group_size]");

    }

    const int G = static_cast<int>(Tb.shape[0]);

    const int levels = static_cast<int>(Tb.shape[1]);

    const int gs_t = static_cast<int>(Tb.shape[2]);

    const int gs_q = static_cast<int>(qb.shape[1]);

    const int gs_x = static_cast<int>(xb.shape[1]);



    if (G != qb.shape[0] || G != xb.shape[0]) {

        throw std::invalid_argument("T_tile, q_vals, x_tile group count mismatch");

    }

    if (levels != 16) {

        throw std::invalid_argument("T_tile second dimension must be 16 (Q4 levels)");

    }

    if (gs_t != gs_q || gs_t != gs_x) {

        throw std::invalid_argument("group_size mismatch across T_tile, q_vals, x_tile");

    }

    if (expected_groups > 0 && (G < expected_groups || qb.shape[0] < expected_groups)) {

        throw std::invalid_argument("n_groups exceeds tensor leading dimension");

    }

    if (gs_t != 32) {

        throw std::invalid_argument("only group_size=32 is supported");

    }

    const py::ssize_t stride_g = Tb.strides[0];

    const py::ssize_t stride_q = Tb.strides[1];

    const py::ssize_t stride_i = Tb.strides[2];

    const py::ssize_t expect_i = static_cast<py::ssize_t>(sizeof(uint16_t));

    const py::ssize_t expect_q = static_cast<py::ssize_t>(gs_t * expect_i);

    const py::ssize_t expect_g = static_cast<py::ssize_t>(levels * expect_q);

    if (stride_i != expect_i || stride_q != expect_q || stride_g != expect_g) {

        throw std::invalid_argument(

            "T_tile must be C-contiguous float16 [G, 16, group_size]");

    }

    if (qb.strides[1] != 1 || xb.strides[1] != static_cast<py::ssize_t>(sizeof(float))) {

        throw std::invalid_argument(

            "q_vals and x_tile must be C-contiguous [G, group_size]");

    }

}



static float py_lut_gemv_avx2_tile(

    py::array T_tile,

    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> q_vals,

    py::array_t<float, py::array::c_style | py::array::forcecast> x_tile,

    int n_groups

) {

    require_simd();

    auto Tb = T_tile.request();

    if (Tb.itemsize != 2) {

        throw std::invalid_argument("T_tile must be float16 (itemsize 2)");

    }

    auto qb = q_vals.request();

    auto xb = x_tile.request();

    validate_tile_shapes(Tb, qb, xb, n_groups > 0 ? n_groups : -1);



    const int G = static_cast<int>(Tb.shape[0]);

    const int gs = static_cast<int>(Tb.shape[2]);

    const int n_active = (n_groups > 0) ? n_groups : G;

    if (n_active > G) {

        throw std::invalid_argument("n_groups exceeds T_tile leading dimension");

    }

    float out = 0.0f;

    {

        py::gil_scoped_release release;

        out = lut_gemv_avx2_tile_impl(

            reinterpret_cast<const uint16_t*>(Tb.ptr),

            static_cast<const uint8_t*>(qb.ptr),

            static_cast<const float*>(xb.ptr),

            n_active,

            gs,

            16);

    }

    return out;

}



static inline uint8_t unpack_nibble(const uint8_t* row_packed, int k) {

    const int byte_idx = k / 2;

    const uint8_t b = row_packed[byte_idx];

    return (k & 1) == 0 ? static_cast<uint8_t>(b & 0x0F)

                        : static_cast<uint8_t>((b >> 4) & 0x0F);

}



static void build_tile_f16(

    uint16_t* T,

    const float* scales,

    const float* biases,

    const float* zeros,

    int tile_group_start,

    int n_groups,

    int tile_groups_cap,

    int group_size

) {

    const int stride_T_g = 16 * group_size;

    for (int g_local = 0; g_local < tile_groups_cap; ++g_local) {

        uint16_t* Tg = T + static_cast<size_t>(g_local) * stride_T_g;

        if (g_local >= n_groups) {

            for (int q = 0; q < 16 * group_size; ++q) {

                Tg[q] = 0;

            }

            continue;

        }

        const int gidx = tile_group_start + g_local;

        const float scale_g = scales[gidx];

        float zero_g;

        if (zeros != nullptr) {

            zero_g = zeros[gidx];

        } else {

            const float s = scale_g;

            zero_g = (s == 0.0f) ? 0.0f : (-biases[gidx] / s);

        }

        for (int q = 0; q < 16; ++q) {

            const float val = (static_cast<float>(q) - zero_g) * scale_g;

            const __m128i ph = _mm_cvtps_ph(_mm_set_ss(val), 0);

            const uint16_t h = static_cast<uint16_t>(_mm_extract_epi16(ph, 0));

            uint16_t* row = Tg + static_cast<size_t>(q) * group_size;

            for (int j = 0; j < group_size; ++j) {

                row[j] = h;

            }

        }

    }

}



static void lut_gemv_projection_impl(

    const uint8_t* wp,

    const float* sc,

    const float* bi,

    const float* xp,

    float* yp,

    const uint8_t* q_prebuilt,

    const float* zeros_ptr,

    int M,

    int K,

    int group_size,

    int tile_groups,

    int groups_per_row,

    int packed_per_row,

    int num_tiles

) {

#ifdef _OPENMP

#pragma omp parallel

#endif

    {

#ifdef _OPENMP

        std::vector<uint16_t> T_buf(

            static_cast<size_t>(tile_groups) * 16 * group_size);

        std::vector<uint8_t> q_buf(

            static_cast<size_t>(tile_groups) * group_size);

#pragma omp for schedule(static)

#else

        std::vector<uint16_t> T_buf(

            static_cast<size_t>(tile_groups) * 16 * group_size);

        std::vector<uint8_t> q_buf(

            static_cast<size_t>(tile_groups) * group_size);

#endif

    for (int m = 0; m < M; ++m) {

        const uint8_t* row_packed = wp + static_cast<size_t>(m) * packed_per_row;

        const uint8_t* q_row =

            (q_prebuilt != nullptr)

                ? q_prebuilt + static_cast<size_t>(m) * groups_per_row * group_size

                : nullptr;

        float acc = 0.0f;

        for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {

            const int g_start = tile_idx * tile_groups;

            const int n_groups = std::min(tile_groups, groups_per_row - g_start);

            if (n_groups <= 0) {

                break;

            }

            const int base = m * groups_per_row + g_start;

            build_tile_f16(

                T_buf.data(),

                sc + base,

                bi + base,

                zeros_ptr != nullptr ? zeros_ptr + base : nullptr,

                0,

                n_groups,

                tile_groups,

                group_size);



            const int k0 = g_start * group_size;

            for (int g_local = 0; g_local < n_groups; ++g_local) {

                uint8_t* qg =

                    q_buf.data() + static_cast<size_t>(g_local) * group_size;

                if (q_row != nullptr) {

                    const uint8_t* src =

                        q_row + static_cast<size_t>(g_start + g_local) * group_size;

                    for (int j = 0; j < group_size; ++j) {

                        qg[j] = src[j];

                    }

                } else {

                    const int k_base = k0 + g_local * group_size;

                    for (int j = 0; j < group_size; ++j) {

                        qg[j] = unpack_nibble(row_packed, k_base + j);

                    }

                }

            }



            acc += lut_gemv_avx2_tile_impl(

                T_buf.data(),

                q_buf.data(),

                xp + k0,

                n_groups,

                group_size,

                16);

        }

        yp[m] = acc;

    }

    }

}



static py::array_t<float> py_lut_gemv_avx2_projection(

    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> w_packed,

    py::array_t<float, py::array::c_style | py::array::forcecast> scales,

    py::array_t<float, py::array::c_style | py::array::forcecast> biases,

    py::array_t<float, py::array::c_style | py::array::forcecast> x,

    int M,

    int K,

    int group_size,

    py::object zeros_obj,

    int tile_groups

) {

    require_simd();

    if (group_size != 32) {

        throw std::invalid_argument("lut_gemv_avx2_projection requires group_size=32");

    }

    if (K % group_size != 0) {

        throw std::invalid_argument("K must be divisible by group_size");

    }



    auto wb = w_packed.request();

    auto sb = scales.request();

    auto bb = biases.request();

    auto xb = x.request();



    const int groups_per_row = K / group_size;

    const int packed_per_row = K / 2;

    const int num_tiles = (groups_per_row + tile_groups - 1) / tile_groups;



    const float* zeros_ptr = nullptr;

    if (!zeros_obj.is_none()) {

        auto zeros_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(

            zeros_obj);

        zeros_ptr = static_cast<const float*>(zeros_arr.request().ptr);

    }



    auto y = py::array_t<float>(M);

    auto yb = y.request();

    float* yp = static_cast<float*>(yb.ptr);



    const uint8_t* wp = static_cast<const uint8_t*>(wb.ptr);

    const float* sc = static_cast<const float*>(sb.ptr);

    const float* bi = static_cast<const float*>(bb.ptr);

    const float* xp = static_cast<const float*>(xb.ptr);



    {

        py::gil_scoped_release release;

        lut_gemv_projection_impl(

            wp, sc, bi, xp, yp, nullptr, zeros_ptr,

            M, K, group_size, tile_groups,

            groups_per_row, packed_per_row, num_tiles);

    }

    return y;

}



static py::array_t<float> py_lut_gemv_full(

    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> w_packed,

    py::array_t<float, py::array::c_style | py::array::forcecast> scales,

    py::array_t<float, py::array::c_style | py::array::forcecast> biases,

    py::array_t<float, py::array::c_style | py::array::forcecast> x,

    int M,

    int K,

    py::object zeros_obj,

    py::object q_vals_obj,

    int tile_groups

) {

    require_simd();

    const int group_size = 32;

    if (K % group_size != 0) {

        throw std::invalid_argument("K must be divisible by 32");

    }



    auto wb = w_packed.request();

    auto sb = scales.request();

    auto bb = biases.request();

    auto xb = x.request();



    const int groups_per_row = K / group_size;

    const int packed_per_row = K / 2;

    const int num_tiles = (groups_per_row + tile_groups - 1) / tile_groups;



    const float* zeros_ptr = nullptr;

    if (!zeros_obj.is_none()) {

        auto zeros_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(

            zeros_obj);

        zeros_ptr = static_cast<const float*>(zeros_arr.request().ptr);

    }



    const uint8_t* q_prebuilt = nullptr;

    if (!q_vals_obj.is_none()) {

        auto q_arr = py::cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>(

            q_vals_obj);

        auto qb = q_arr.request();

        if (qb.ndim != 3) {

            throw std::invalid_argument("q_vals must be [M, groups_per_row, 32]");

        }

        if (static_cast<int>(qb.shape[0]) != M

            || static_cast<int>(qb.shape[1]) != groups_per_row

            || static_cast<int>(qb.shape[2]) != group_size) {

            throw std::invalid_argument("q_vals shape mismatch for M, K");

        }

        if (qb.strides[2] != 1

            || qb.strides[1] != static_cast<py::ssize_t>(group_size)

            || qb.strides[0] != static_cast<py::ssize_t>(groups_per_row * group_size)) {

            throw std::runtime_error(

                "q_vals must be C-contiguous [M, groups_per_row, 32] "

                "(strides [M*groups_per_row*32, 32, 1])");

        }

        q_prebuilt = static_cast<const uint8_t*>(qb.ptr);

    }



    auto y = py::array_t<float>(M);

    auto yb = y.request();

    float* yp = static_cast<float*>(yb.ptr);



    const uint8_t* wp = static_cast<const uint8_t*>(wb.ptr);

    const float* sc = static_cast<const float*>(sb.ptr);

    const float* bi = static_cast<const float*>(bb.ptr);

    const float* xp = static_cast<const float*>(xb.ptr);



    {

        py::gil_scoped_release release;

        lut_gemv_projection_impl(

            wp, sc, bi, xp, yp, q_prebuilt, zeros_ptr,

            M, K, group_size, tile_groups,

            groups_per_row, packed_per_row, num_tiles);

    }

    return y;

}



static py::array_t<float> py_lut_gemv_full_batched(

    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> w_packed,

    py::array_t<float, py::array::c_style | py::array::forcecast> scales,

    py::array_t<float, py::array::c_style | py::array::forcecast> biases,

    py::array_t<float, py::array::c_style | py::array::forcecast> x_batch,

    int M,

    int K,

    py::object zeros_obj,

    py::object q_vals_obj,

    int tile_groups

) {

    require_simd();

    const int group_size = 32;

    if (K % group_size != 0) {

        throw std::invalid_argument("K must be divisible by 32");

    }



    auto xb = x_batch.request();

    if (xb.ndim != 2) {

        throw std::invalid_argument("x_batch must be 2-D [B, K]");

    }

    const int B = static_cast<int>(xb.shape[0]);

    if (static_cast<int>(xb.shape[1]) != K) {

        throw std::invalid_argument("x_batch second dim must equal K");

    }



    auto wb = w_packed.request();

    auto sb = scales.request();

    auto bb = biases.request();



    const int groups_per_row = K / group_size;

    const int packed_per_row = K / 2;

    const int num_tiles = (groups_per_row + tile_groups - 1) / tile_groups;



    const float* zeros_ptr = nullptr;

    if (!zeros_obj.is_none()) {

        auto zeros_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(

            zeros_obj);

        zeros_ptr = static_cast<const float*>(zeros_arr.request().ptr);

    }



    const uint8_t* q_prebuilt = nullptr;

    if (!q_vals_obj.is_none()) {

        auto q_arr = py::cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>(

            q_vals_obj);

        auto qb = q_arr.request();

        if (qb.ndim != 3

            || static_cast<int>(qb.shape[0]) != M

            || static_cast<int>(qb.shape[1]) != groups_per_row

            || static_cast<int>(qb.shape[2]) != group_size) {

            throw std::invalid_argument("q_vals must be [M, groups_per_row, 32]");

        }

        q_prebuilt = static_cast<const uint8_t*>(qb.ptr);

    }



    auto out = py::array_t<float>({B, M});

    auto ob = out.request();

    float* outp = static_cast<float*>(ob.ptr);



    const uint8_t* wp = static_cast<const uint8_t*>(wb.ptr);

    const float* sc = static_cast<const float*>(sb.ptr);

    const float* bi = static_cast<const float*>(bb.ptr);

    const float* xp_base = static_cast<const float*>(xb.ptr);



    {

        py::gil_scoped_release release;

#ifdef _OPENMP

#pragma omp parallel

#endif

        {

#ifdef _OPENMP

            std::vector<uint16_t> T_buf(

                static_cast<size_t>(tile_groups) * 16 * group_size);

            std::vector<uint8_t> q_buf(

                static_cast<size_t>(tile_groups) * group_size);

#pragma omp for collapse(2) schedule(static)

#else

            std::vector<uint16_t> T_buf(

                static_cast<size_t>(tile_groups) * 16 * group_size);

            std::vector<uint8_t> q_buf(

                static_cast<size_t>(tile_groups) * group_size);

#endif

        for (int b = 0; b < B; ++b) {

            for (int m = 0; m < M; ++m) {

                const uint8_t* row_packed =

                    wp + static_cast<size_t>(m) * packed_per_row;

                const uint8_t* q_row =

                    (q_prebuilt != nullptr)

                        ? q_prebuilt

                              + static_cast<size_t>(m) * groups_per_row * group_size

                        : nullptr;

                const float* xp = xp_base + static_cast<size_t>(b) * K;

                float acc = 0.0f;

                for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {

                    const int g_start = tile_idx * tile_groups;

                    const int n_groups =

                        std::min(tile_groups, groups_per_row - g_start);

                    if (n_groups <= 0) {

                        break;

                    }

                    const int base = m * groups_per_row + g_start;

                    build_tile_f16(

                        T_buf.data(),

                        sc + base,

                        bi + base,

                        zeros_ptr != nullptr ? zeros_ptr + base : nullptr,

                        0,

                        n_groups,

                        tile_groups,

                        group_size);



                    const int k0 = g_start * group_size;

                    for (int g_local = 0; g_local < n_groups; ++g_local) {

                        uint8_t* qg =

                            q_buf.data()

                            + static_cast<size_t>(g_local) * group_size;

                        if (q_row != nullptr) {

                            const uint8_t* src =

                                q_row

                                + static_cast<size_t>(g_start + g_local)

                                      * group_size;

                            for (int j = 0; j < group_size; ++j) {

                                qg[j] = src[j];

                            }

                        } else {

                            const int k_base = k0 + g_local * group_size;

                            for (int j = 0; j < group_size; ++j) {

                                qg[j] = unpack_nibble(row_packed, k_base + j);

                            }

                        }

                    }



                    acc += lut_gemv_avx2_tile_impl(

                        T_buf.data(),

                        q_buf.data(),

                        xp + k0,

                        n_groups,

                        group_size,

                        16);

                }

                outp[static_cast<size_t>(b) * M + m] = acc;

            }

        }

        }

    }

    return out;

}



PYBIND11_MODULE(asdsl_lut_avx2, m) {

    m.doc() = "Phase 2/3 F16C LUT gather GEMV (AVX2 + FMA)";



    m.def(

        "lut_gemv_avx2_tile",

        &py_lut_gemv_avx2_tile,

        py::arg("T_tile"),

        py::arg("q_vals"),

        py::arg("x_tile"),

        py::arg("n_groups") = 0,

        "Partial dot for one K-tile: T float16 [G,16,32], q uint8 [G,32], x float32 [G,32].");



    m.def(

        "lut_gemv_avx2_projection",

        &py_lut_gemv_avx2_projection,

        py::arg("w_packed"),

        py::arg("scales"),

        py::arg("biases"),

        py::arg("x"),

        py::arg("M"),

        py::arg("K"),

        py::arg("group_size"),

        py::arg("zeros") = py::none(),

        py::arg("tile_groups") = 128,

        "Full projection GEMV with OpenMP over rows (baseline).");



    m.def(

        "lut_gemv_full",

        &py_lut_gemv_full,

        py::arg("w_packed"),

        py::arg("scales"),

        py::arg("biases"),

        py::arg("x"),

        py::arg("M"),

        py::arg("K"),

        py::arg("zeros") = py::none(),

        py::arg("q_vals") = py::none(),

        py::arg("tile_groups") = 128,

        "Optimized projection GEMV; optional prebuilt q_vals [M,G,32].");



    m.def(

        "lut_gemv_full_batched",

        &py_lut_gemv_full_batched,

        py::arg("w_packed"),

        py::arg("scales"),

        py::arg("biases"),

        py::arg("x_batch"),

        py::arg("M"),

        py::arg("K"),

        py::arg("zeros") = py::none(),

        py::arg("q_vals") = py::none(),

        py::arg("tile_groups") = 128,

        "Batched GEMV: x [B,K] -> out [B,M].");



    m.def("check_avx2", &check_avx2_support, "Runtime AVX2 support.");

    m.def("check_f16c", &check_f16c_support, "Runtime F16C support.");

}


