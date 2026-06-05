"""Dispatch policy for Phase 1 LUT-native vs legacy GEMV paths."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from asdsl.lut.lut_gemv_kernel import LUTGEMVKernel

if TYPE_CHECKING:
    from asdsl.lut.lut_table_builder import LUTProjectionCache

logger = logging.getLogger(__name__)


def should_use_lut(bits: int, group_size: int, M: int, K: int) -> bool:
    """Return True when the prebuilt-table LUT path applies."""
    return bits == 4 and group_size == 32 and K % 32 == 0 and M > 0


class LUTKernelDispatcher:
    """Route ``gemv_q4_packed(use_lut=True)`` to LUT, legacy LUT, or FMA."""

    _kernel = LUTGEMVKernel()

    @classmethod
    def dispatch(
        cls,
        w_packed: np.ndarray,
        x: np.ndarray,
        scales: np.ndarray,
        biases: np.ndarray,
        M: int,
        K: int,
        group_size: int,
        *,
        lut_cache: LUTProjectionCache | None = None,
        bits: int = 4,
        use_lut: bool = True,
        _try_import_lut=None,
        _native_available: bool = False,
        _native=None,
        _gemv_numpy=None,
    ) -> np.ndarray:
        """Dispatch GEMV; falls back when LUT cache or native extensions unavailable."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim != 1:
            raise ValueError("LUT dispatch requires 1-D activation vector")

        if use_lut and should_use_lut(bits, group_size, M, K):
            if lut_cache is not None:
                return cls._kernel.gemv(lut_cache, x)

            if _try_import_lut is not None:
                nl = _try_import_lut()
                if nl is not None:
                    try:
                        return np.asarray(
                            nl.gemv_lut_q4_tiled(
                                w_packed, scales, biases, x, M, K, group_size
                            )
                        )
                    except Exception as e:
                        logger.warning(
                            "Legacy LUT kernel failed (%s), falling back to FMA", e
                        )

        if _native_available and _native is not None:
            return np.asarray(
                _native.gemv_q4_packed(w_packed, x, scales, biases, M, K, group_size)
            )
        if _gemv_numpy is not None:
            return _gemv_numpy(w_packed, x, scales, biases, M, K, group_size)
        raise RuntimeError("No GEMV backend available")
