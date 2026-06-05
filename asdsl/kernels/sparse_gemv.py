"""Dequant float16 sparse GEMV: y = W_deq[:, active] @ x[active].



Distinct from ``gemv_sparse.py`` (quantized activation-sparse path).

"""



from __future__ import annotations



import numpy as np



try:

    from asdsl.kernels import asdsl_sparse_gemv as _sparse_native



    _HAS_NATIVE = True

except ImportError:

    _sparse_native = None  # type: ignore[assignment]

    _HAS_NATIVE = False





def active_columns(x: np.ndarray, threshold: float = 0.01) -> np.ndarray:

    """Return column indices where ``|x| >= threshold``."""

    flat = np.ascontiguousarray(x, dtype=np.float32).ravel()

    return np.flatnonzero(np.abs(flat) >= threshold).astype(np.int64)





def sparse_gemv_dequant_f16(

    w_deq_f16: np.ndarray,

    x: np.ndarray,

    threshold: float = 0.01,

) -> np.ndarray:

    """Reference sparse GEMV on dequantized float16 weights ``[M, K]``."""

    x = np.ascontiguousarray(x, dtype=np.float32).ravel()

    w = np.ascontiguousarray(w_deq_f16)

    if w.ndim != 2:

        raise ValueError("w_deq_f16 must be 2-D [M, K]")

    M, K = w.shape

    if x.shape[0] != K:

        raise ValueError(f"x length {x.shape[0]} != K={K}")



    cols = active_columns(x, threshold)

    if cols.size == 0:

        return np.zeros(M, dtype=np.float32)



    w_block = w[:, cols].astype(np.float32)

    x_sel = x[cols]

    return w_block @ x_sel





def sparse_gemv_dequant_f16_avx2(

    w_deq_f16: np.ndarray,

    x: np.ndarray,

    threshold: float = 0.01,

) -> np.ndarray:

    """AVX2 sparse GEMV when extension is built; else Python reference."""

    if _HAS_NATIVE and _sparse_native is not None:

        x = np.ascontiguousarray(x, dtype=np.float32).ravel()

        w = np.ascontiguousarray(w_deq_f16)

        M, K = w.shape

        cols = active_columns(x, threshold)

        if cols.size == 0:

            return np.zeros(M, dtype=np.float32)

        # Bitwise f16 payload; pybind forcecast from float16 would truncate values.
        w_bits = w.view(np.uint16)

        return np.asarray(

            _sparse_native.sparse_gemv_f16(

                w_bits, x, cols.astype(np.int32), M, K

            ),

            dtype=np.float32,

        )

    return sparse_gemv_dequant_f16(w_deq_f16, x, threshold)





def has_sparse_dequant_kernel() -> bool:

    return _HAS_NATIVE


