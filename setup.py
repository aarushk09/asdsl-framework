"""Build configuration for the ASDSL native C++ extensions.

This setup.py is used alongside pyproject.toml. It only defines the
C++ extension modules; all other metadata comes from pyproject.toml.

The native extensions are optional: if pybind11 is not installed or
the build fails, ASDSL falls back to pure-Python/NumPy implementations.

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

    extra_compile_args = []
    extra_link_args = []

    if sys.platform == "win32":
        extra_compile_args = ["/arch:AVX2", "/O2", "/fp:fast", "/EHsc"]
        # MSVC OpenMP
        extra_compile_args.append("/openmp")
    elif sys.platform == "darwin":
        extra_compile_args = ["-mavx2", "-mfma", "-O3", "-ffast-math"]
        extra_compile_args.append("-std=c++17")
    else:
        extra_compile_args = [
            "-mavx2", "-mfma", "-O3", "-ffast-math", "-std=c++17",
        ]
        extra_compile_args.append("-fopenmp")
        extra_link_args.append("-fopenmp")

    ext_modules = [
        Pybind11Extension(
            "asdsl.kernels._native_gemv",
            ["asdsl/kernels/native/gemv_q4_avx2.cpp"],
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
            define_macros=[("PYBIND11_DETAILED_ERROR_MESSAGES", "1")],
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
    ]

    return ext_modules, {"build_ext": build_ext}


ext_modules, cmdclass = get_native_extensions()

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
