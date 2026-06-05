"""Standalone build for Phase 3 sparse dequant GEMV (asdsl_sparse_gemv)."""



from __future__ import annotations



import shutil

import sys

from pathlib import Path



from setuptools import setup



try:

    from pybind11.setup_helpers import Pybind11Extension, build_ext

except ImportError as exc:

    raise SystemExit("pybind11 is required: pip install pybind11>=2.11") from exc



KERN = Path(__file__).resolve().parent

PROJ = KERN.parent.parent



extra_compile_args: list[str] = []

extra_link_args: list[str] = []



if sys.platform == "win32":

    extra_compile_args = [

        "/O2",

        "/Ob2",

        "/Oi",

        "/arch:AVX2",

        "/fp:fast",

        "/openmp",

        "/EHsc",

        "/std:c++17",

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

        "asdsl.kernels.asdsl_sparse_gemv",

        [str(KERN / "sparse_gemv_avx2.cpp")],

        extra_compile_args=extra_compile_args,

        extra_link_args=extra_link_args,

        define_macros=[("PYBIND11_DETAILED_ERROR_MESSAGES", "1")],

        language="c++",

    ),

]





class BuildExtInplace(build_ext):

    def run(self) -> None:

        super().run()

        if Path.cwd().resolve() != KERN.resolve():

            return

        for src in self.get_outputs():

            src_path = Path(src)

            if src_path.is_file():

                dest = KERN / src_path.name

                if src_path.resolve() != dest.resolve():

                    shutil.copy2(src_path, dest)





setup(

    name="asdsl_sparse_gemv",

    ext_modules=ext_modules,

    cmdclass={"build_ext": BuildExtInplace},

    package_dir={"asdsl": str(PROJ / "asdsl")},

    packages=["asdsl.kernels"],

)


