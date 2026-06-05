"""Type stubs for optional ``asdsl.kernels.asdsl_lut_avx2`` extension."""



from __future__ import annotations



from typing import Optional



import numpy as np

from numpy.typing import NDArray



def lut_gemv_avx2_tile(

    T_tile: NDArray[np.uint16],

    q_vals: NDArray[np.uint8],

    x_tile: NDArray[np.float32],

    n_groups: int = 0,

) -> float: ...



def lut_gemv_avx2_projection(

    w_packed: NDArray[np.uint8],

    scales: NDArray[np.float32],

    biases: NDArray[np.float32],

    x: NDArray[np.float32],

    M: int,

    K: int,

    group_size: int,

    zeros: Optional[NDArray[np.float32]] = ...,

    tile_groups: int = ...,

) -> NDArray[np.float32]: ...



def lut_gemv_full(

    w_packed: NDArray[np.uint8],

    scales: NDArray[np.float32],

    biases: NDArray[np.float32],

    x: NDArray[np.float32],

    M: int,

    K: int,

    zeros: Optional[NDArray[np.float32]] = ...,

    q_vals: Optional[NDArray[np.uint8]] = ...,

    tile_groups: int = ...,

) -> NDArray[np.float32]: ...



def lut_gemv_full_batched(

    w_packed: NDArray[np.uint8],

    scales: NDArray[np.float32],

    biases: NDArray[np.float32],

    x_batch: NDArray[np.float32],

    M: int,

    K: int,

    zeros: Optional[NDArray[np.float32]] = ...,

    q_vals: Optional[NDArray[np.uint8]] = ...,

    tile_groups: int = ...,

) -> NDArray[np.float32]: ...



def check_avx2() -> bool: ...

def check_f16c() -> bool: ...


