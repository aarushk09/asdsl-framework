"""Low-level C/intrinsics kernel interfaces.

Provides SIMD kernel stubs (simd.py) and fused GEMV extensions
(4-bit, 8-bit, 3-bit, 2-bit, sparse) with optional native AVX2+FMA
acceleration.
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
from asdsl.kernels.gemv_q3 import (
    gemv_q3_unpacked,
    has_native_kernel as has_native_q3_kernel,
)
from asdsl.kernels.gemv_q2 import (
    gemv_q2_unpacked,
    gemv_q2_packed,
    has_native_kernel as has_native_q2_kernel,
)
from asdsl.kernels.gemv_sparse import (
    compute_activation_bitmask,
    gemv_sparse_unpacked,
    gemv_sparse_with_indices,
    has_native_sparse_kernel,
)

__all__ = [
    "gemv_q4",
    "gemv_q4_packed",
    "gemv_q4_unpacked",
    "has_native_kernel",
    "gemv_q8_unpacked",
    "has_native_q8_kernel",
    "gemv_q3_unpacked",
    "has_native_q3_kernel",
    "gemv_q2_unpacked",
    "gemv_q2_packed",
    "has_native_q2_kernel",
    "compute_activation_bitmask",
    "gemv_sparse_unpacked",
    "gemv_sparse_with_indices",
    "has_native_sparse_kernel",
]
