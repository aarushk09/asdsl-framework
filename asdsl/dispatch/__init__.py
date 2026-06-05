"""Phase 3 dynamic kernel dispatch (LUT / AVX2 / SPARSE)."""

from asdsl.dispatch.calibrate import calibrate, build_profiles_from_store
from asdsl.dispatch.policy import (
    DispatchPolicy,
    KernelTag,
    PHI4_PROJECTIONS,
    ProjectionProfile,
    l2_budget_bytes,
    sparse_min_size,
)

__all__ = [
    "calibrate",
    "build_profiles_from_store",
    "DispatchPolicy",
    "KernelTag",
    "PHI4_PROJECTIONS",
    "ProjectionProfile",
    "l2_budget_bytes",
    "sparse_min_size",
]
