/**

 * Sparse GEMV on dequant float16 weights: y = W[:, cols] @ x[cols].

 * Build: python asdsl/kernels/setup_sparse.py build_ext --inplace

 */



#include <pybind11/pybind11.h>

#include <pybind11/numpy.h>



#include <immintrin.h>

#include <cstdint>

#include <cstddef>
#include <cstring>



#ifdef _OPENMP

#include <omp.h>

#endif



namespace py = pybind11;

/** Portable IEEE-754 binary16 -> binary32 (F16C intrinsics mis-read on some MSVC builds). */
static inline float half_to_float(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h & 0x7C00u) >> 10;
    uint32_t mant = h & 0x03FFu;
    if (exp == 0) {
        if (mant == 0) {
            uint32_t bits = sign;
            float out;
            std::memcpy(&out, &bits, sizeof(out));
            return out;
        }
        while ((mant & 0x0400u) == 0) {
            mant <<= 1;
            --exp;
        }
        ++exp;
        mant &= 0x03FFu;
    } else if (exp == 31) {
        uint32_t bits = sign | 0x7F800000u | (mant << 13);
        float out;
        std::memcpy(&out, &bits, sizeof(out));
        return out;
    }
    exp = exp + (127 - 15);
    uint32_t bits = sign | (exp << 23) | (mant << 13);
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

static py::array_t<float> py_sparse_gemv_f16(

    py::array_t<uint16_t, py::array::c_style | py::array::forcecast> w_f16,

    py::array_t<float, py::array::c_style | py::array::forcecast> x,

    py::array_t<int32_t, py::array::c_style | py::array::forcecast> active_cols,

    int M,

    int K

) {

    auto wb = w_f16.request();

    auto xb = x.request();

    auto cb = active_cols.request();



    if (wb.ndim != 2) {

        throw std::invalid_argument("w_f16 must be [M, K] float16");

    }

    if (static_cast<int>(wb.shape[0]) != M || static_cast<int>(wb.shape[1]) != K) {

        throw std::invalid_argument("w_f16 shape mismatch with M, K");

    }

    if (static_cast<int>(xb.shape[0]) != K) {

        throw std::invalid_argument("x length must equal K");

    }



    const int n_active = static_cast<int>(cb.shape[0]);

    const uint16_t* wp = static_cast<const uint16_t*>(wb.ptr);

    const float* xp = static_cast<const float*>(xb.ptr);

    const int32_t* cols = static_cast<const int32_t*>(cb.ptr);



    auto y = py::array_t<float>(M);

    auto yb = y.request();

    float* yp = static_cast<float*>(yb.ptr);



    {

        py::gil_scoped_release release;

        for (int m = 0; m < M; ++m) {

            const uint16_t* row = wp + static_cast<size_t>(m) * K;

            __m256 acc = _mm256_setzero_ps();

            if (n_active == 0) {

                yp[m] = 0.0f;

                continue;

            }

            int j = 0;

            for (; j + 8 <= n_active; j += 8) {

                alignas(32) float x8[8] = {};

                alignas(32) float w8[8] = {};

                for (int k = 0; k < 8; ++k) {

                    const int col = cols[j + k];

                    if (col < 0 || col >= K) {

                        continue;

                    }

                    x8[k] = xp[col];

                    w8[k] = half_to_float(row[col]);

                }

                __m256 xv = _mm256_load_ps(x8);

                __m256 wv = _mm256_load_ps(w8);

                acc = _mm256_fmadd_ps(wv, xv, acc);

            }

            alignas(32) float acc_lane[8];

            _mm256_store_ps(acc_lane, acc);

            double sum = 0.0;

            for (int k = 0; k < 8; ++k) {

                sum += static_cast<double>(acc_lane[k]);

            }

            for (; j < n_active; ++j) {

                const int col = cols[j];

                if (col < 0 || col >= K) {

                    continue;

                }

                sum += static_cast<double>(half_to_float(row[col]))

                    * static_cast<double>(xp[col]);

            }

            yp[m] = static_cast<float>(sum);

        }

    }

    return y;

}



PYBIND11_MODULE(asdsl_sparse_gemv, m) {

    m.doc() = "Sparse GEMV on dequant float16 weight matrix";

    m.def(

        "sparse_gemv_f16",

        &py_sparse_gemv_f16,

        py::arg("w_f16"),

        py::arg("x"),

        py::arg("active_cols"),

        py::arg("M"),

        py::arg("K"),

        "y = W_f16[:, active_cols] @ x[active_cols]");

}


