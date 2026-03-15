"""Low-level C/intrinsics kernel interfaces.

Provides SIMD kernel stubs (simd.py) and the fused 4-bit GEMV
extension (gemv_q4.py) with optional native AVX2+FMA acceleration.
"""

from asdsl.kernels.gemv_q4 import (
    gemv_q4,
    gemv_q4_packed,
    gemv_q4_unpacked,
    has_native_kernel,
)
from asdsl.kernels.gemv_q8 import (
    gemv_q8_unpacked,
    has_native_kernel as has_native_q8_kernel,
)

__all__ = [
    "gemv_q4",
    "gemv_q4_packed",
    "gemv_q4_unpacked",
    "has_native_kernel",
    "gemv_q8_unpacked",
    "has_native_q8_kernel",
]
