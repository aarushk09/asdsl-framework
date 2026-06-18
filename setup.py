"""Build configuration for the ASDSL native C++ extensions.

This setup.py is used alongside pyproject.toml. It only defines the
C++ extension modules; all other metadata comes from pyproject.toml.

The native extensions are optional: if pybind11 is not installed or
the build fails, ASDSL falls back to pure-Python/NumPy implementations.

Non-Windows builds use ``-O3 -mavx2 -mfma -ffast-math -fopenmp``. On macOS,
Apple's linker may require Homebrew OpenMP (``brew install libomp``) and
``CPPFLAGS``/``LDFLAGS`` pointing at libomp if ``-fopenmp`` fails to link.

Build manually:
    python setup.py build_ext --inplace

Or via pip (reads pyproject.toml for deps, then runs this for extensions):
    pip install -e ".[dev]"
"""

import os
import sys
import platform

from setuptools import setup


def get_native_extensions():
    """Build the list of C++ extension modules, if pybind11 is available."""
    try:
        from pybind11.setup_helpers import Pybind11Extension, build_ext
    except ImportError:
        print(
            "WARNING: pybind11 not found — native GEMV kernels will not be built.\n"
            "Install pybind11 first:  pip install pybind11>=2.11"
        )
        return [], {}

    # Aggressive optimization for memory-bandwidth-heavy GEMV / inference kernels.
    # MSVC: /openmp is compile-only; the linker pulls the OpenMP runtime automatically.
    # Unix (Linux/macOS): -fopenmp on link as well. macOS may need Homebrew libomp
    # (see https://brew.sh/formula/libomp) and CPPFLAGS/LDFLAGS if the linker cannot find -lomp.
    extra_compile_args = []
    extra_link_args = []

    if sys.platform == "win32":
        extra_compile_args = [
            "/O2",
            "/Ob2",
            "/Oi",
            "/arch:AVX2",
            "/fp:fast",
            "/openmp",
            "/EHsc",
        ]
    else:
        extra_compile_args = [
            "-O3",
            "-mavx2",
            "-mfma",
            "-mf16c",
            "-ffast-math",
            "-std=c++17",
            "-fopenmp",
        ]
        extra_link_args = ["-fopenmp"]

    ext_modules = [
        Pybind11Extension(
            "asdsl.kernels._native_forward",
            ["asdsl/kernels/forward_loop.cpp"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            language="c++",
        ),

        Pybind11Extension(
            "asdsl.kernels._native_gemv",
            [
                "asdsl/kernels/native/gemv_q4_avx2.cpp",
                "asdsl/kernels/native/gemv_q4_preq_restored.cpp",
                "asdsl/kernels/native/gemv_preq2_avx2.cpp",
                "asdsl/kernels/native/gemv_q4_128.cpp",
                "asdsl/kernels/native/gemv_q4_kernel.cpp",
                "asdsl/kernels/native/lm_head_avx2.cpp",
            ],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            define_macros=[("PYBIND11_DETAILED_ERROR_MESSAGES", "1")],
            language="c++",
        ),
        Pybind11Extension(
            "asdsl.kernels._native_gemv_q8",
            ["asdsl/kernels/native/gemv_q8_avx2.cpp"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            define_macros=[("PYBIND11_DETAILED_ERROR_MESSAGES", "1")],
            language="c++",
        ),
        Pybind11Extension(
            "asdsl.kernels._native_gemv_q3",
            ["asdsl/kernels/native/gemv_q3_avx2.cpp"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            define_macros=[("PYBIND11_DETAILED_ERROR_MESSAGES", "1")],
            language="c++",
        ),
        Pybind11Extension(
            "asdsl.kernels._native_gemv_q2",
            ["asdsl/kernels/native/gemv_q2_avx2.cpp"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            define_macros=[("PYBIND11_DETAILED_ERROR_MESSAGES", "1"), ("ASDSL_GEMV_Q2_PYEXT", "1")],
            language="c++",
        ),
        Pybind11Extension(
            "asdsl.kernels._native_sparse_gemv",
            ["asdsl/kernels/native/gemv_sparse_avx2.cpp"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            define_macros=[("PYBIND11_DETAILED_ERROR_MESSAGES", "1")],
            language="c++",
        ),
        Pybind11Extension(
            "asdsl.kernels._native_lut",
            ["asdsl/kernels/native/lut_avx2.cpp"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            define_macros=[("PYBIND11_DETAILED_ERROR_MESSAGES", "1")],
            language="c++",
        ),
        Pybind11Extension(
            "asdsl.kernels._native_inference",
            ["asdsl/kernels/native/inference_engine.cpp"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            define_macros=[("PYBIND11_DETAILED_ERROR_MESSAGES", "1")],
            language="c++",
        ),
        Pybind11Extension(
            "asdsl.kernels._native_ops",
            ["asdsl/kernels/native/ops_avx2.cpp"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            define_macros=[("PYBIND11_DETAILED_ERROR_MESSAGES", "1")],
            language="c++",
        ),
        Pybind11Extension(
            "asdsl.kernels._native_engine",
            ["asdsl/kernels/native/engine.cpp"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            define_macros=[("PYBIND11_DETAILED_ERROR_MESSAGES", "1")],
            language="c++",
        ),
        Pybind11Extension(
            "asdsl.kernels._native_unified",
            [
                "asdsl/kernels/native/unified_engine.cpp",
                "asdsl/kernels/native/gemm_batch.cpp",
                "asdsl/kernels/native/gemv_q4_avx2.cpp",
                "asdsl/kernels/native/gemv_q4_preq_restored.cpp",
                "asdsl/kernels/native/gemv_preq2_avx2.cpp",
                "asdsl/kernels/native/gemv_q4_128.cpp",
                "asdsl/kernels/native/gemv_q4_s256.cpp",
                "asdsl/kernels/native/gemv_q4_kernel.cpp",
                "asdsl/kernels/native/gemv_q8_avx2.cpp",
                "asdsl/kernels/native/lm_head_avx2.cpp",
                "asdsl/kernels/native/gemv_q2_avx2.cpp",
            ],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            define_macros=[("PYBIND11_DETAILED_ERROR_MESSAGES", "1")],
            language="c++",
        ),

        # HEP: AES-NI weight synthesis kernel (AVX2 + AES-NI + OpenMP)
        # On Windows, AES-NI is enabled automatically with /arch:AVX2 on Raptor Lake.
        # On Linux/macOS, we explicitly add -maes to enable AES-NI intrinsics.
        Pybind11Extension(
            "asdsl.kernels._native_hep",
            ["asdsl/kernels/native/gemv_hep_aesni.cpp"],
            extra_compile_args=extra_compile_args + ([] if sys.platform == "win32" else ["-maes"]),
            extra_link_args=extra_link_args,
            define_macros=[("PYBIND11_DETAILED_ERROR_MESSAGES", "1")],
            language="c++",
        ),
    ]

    return ext_modules, {"build_ext": build_ext}


ext_modules, cmdclass = get_native_extensions()

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)

