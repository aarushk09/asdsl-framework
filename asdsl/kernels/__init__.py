"""Low-level C/intrinsics kernel interfaces.

Provides SIMD kernel stubs (simd.py) and fused GEMV extensions
(4-bit, 8-bit, 3-bit, 2-bit, sparse) with optional native AVX2+FMA
acceleration.
"""

try:
    import cpufeature
    if cpufeature.CPUFeature.get('AVX512f', False):
        KERNEL_TIER = 'AVX512'
    elif cpufeature.CPUFeature.get('AVX2', False):
        KERNEL_TIER = 'AVX2'
    else:
        KERNEL_TIER = 'BASIC'
except ImportError:
    KERNEL_TIER = 'BASIC'

from asdsl.kernels.gemv_q4 import (
    gemv_q4,
    gemv_q4_packed,
    gemv_q4_unpacked,
    gemv_q4km_q8,
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


def has_avx2_lut() -> bool:
    """Return True if Phase 2 LUT GEMV (F16C gather) extension is available."""
    try:
        from asdsl.kernels import asdsl_lut_avx2 as _lut_avx2

        return _lut_avx2.check_avx2() and _lut_avx2.check_f16c()
    except ImportError:
        return False


__all__ = [
    "KERNEL_TIER",
    "gemv_q4",
    "gemv_q4_packed",
    "gemv_q4_unpacked",
    "gemv_q4km_q8",
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
    "has_avx2_lut",
]
