"""LUT GEMV reference helpers and correctness checks."""



from __future__ import annotations



import numpy as np



from asdsl.quantization.core import dequantize_weights, quantize_weights





def make_test_case(M: int, K: int, group_size: int = 32, seed: int = 0):

    rng = np.random.default_rng(seed)

    w = rng.standard_normal((M, K)).astype(np.float32) * 0.05

    qt = quantize_weights(w, bits=4, group_size=group_size, symmetric=False)

    n_groups = M * (K // group_size)

    scales = qt.scales[:n_groups].astype(np.float32)

    if qt.zeros is not None:

        zeros = qt.zeros[:n_groups].astype(np.float32)

        biases = (-zeros * scales).astype(np.float32)

    else:

        biases = (-7.5 * scales).astype(np.float32)

    x = rng.standard_normal(K).astype(np.float32)

    return qt.data, scales, biases, x





def dequant_q4_ref(

    w_packed: np.ndarray,

    scales: np.ndarray,

    biases: np.ndarray,

    x: np.ndarray,

    M: int,

    K: int,

    group_size: int,

) -> np.ndarray:

    """Reference GEMV via full dequant matvec."""

    from asdsl.quantization.core import QuantizedTensor

    n_groups = M * (K // group_size)

    zeros = (-biases / np.maximum(scales, 1e-12)).astype(np.float32)

    qt = QuantizedTensor(

        data=np.ascontiguousarray(w_packed, dtype=np.uint8),

        scales=scales[:n_groups],

        zeros=zeros[:n_groups],

        bits=4,

        group_size=group_size,

        shape=(M, K),

        is_symmetric=False,

    )

    w = dequantize_weights(qt).reshape(M, K)

    return (w.astype(np.float32) @ np.asarray(x, dtype=np.float32).ravel()).astype(

        np.float32

    )


