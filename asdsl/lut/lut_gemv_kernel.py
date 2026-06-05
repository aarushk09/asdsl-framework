"""Python reference LUT-native GEMV: gather from prebuilt T tables + bias correction."""



from __future__ import annotations



import os

import numpy as np



from asdsl.lut.lut_table_builder import LUTProjectionCache, LUTTableBuilder



try:

    from asdsl.kernels import asdsl_lut_avx2 as _lut_avx2



    _HAS_AVX2_LUT = bool(_lut_avx2.check_avx2() and _lut_avx2.check_f16c())

except ImportError:

    _lut_avx2 = None  # type: ignore[assignment]

    _HAS_AVX2_LUT = False





class LUTGEMVKernel:

    """Weight-only LUT GEMV: ``y += dot(q,x)*scale + bias*sum(x)`` via table gather."""



    def __init__(self, tile_groups: int = LUTTableBuilder.DEFAULT_TILE_GROUPS):

        self.tile_groups = tile_groups



    def gemv(

        self,

        cache: LUTProjectionCache,

        x: np.ndarray,

        *,

        tile_groups: int | None = None,

        use_avx2: bool = True,

    ) -> np.ndarray:

        """Compute ``y = dequant(W) @ x`` using LUT tables for one projection."""

        meta = cache.meta

        M, K = meta.rows, meta.cols

        gs = meta.group_size

        tg = tile_groups if tile_groups is not None else self.tile_groups

        x = np.ascontiguousarray(x, dtype=np.float32).ravel()

        if x.shape[0] != K:

            raise ValueError(f"x length {x.shape[0]} != K={K}")



        if use_avx2 and _HAS_AVX2_LUT and _lut_avx2 is not None:

            return self._compute_avx2(cache, x, M, K, gs, tg)

        return self._compute_python(cache, x, M, K, gs, tg)



    def _compute_python(

        self,

        cache: LUTProjectionCache,

        x: np.ndarray,

        M: int,

        K: int,

        gs: int,

        tg: int,

    ) -> np.ndarray:

        y = np.zeros(M, dtype=np.float32)

        packed_per_row = K // 2

        groups_per_row = cache.meta.groups_per_row

        num_tiles = LUTTableBuilder.num_k_tiles(groups_per_row, tg)



        for m in range(M):

            row_off = m * packed_per_row

            row_packed = cache.w_packed[row_off : row_off + packed_per_row]

            acc = 0.0

            for tile_idx in range(num_tiles):

                acc += self._tile_partial_python(

                    cache, m, tile_idx, row_packed, x, gs, tg, groups_per_row

                )

            y[m] = acc

        return y



    def _compute_avx2(

        self,

        cache: LUTProjectionCache,

        x: np.ndarray,

        M: int,

        K: int,

        gs: int,

        tg: int,

    ) -> np.ndarray:

        assert _lut_avx2 is not None

        zeros = cache.zeros

        if hasattr(_lut_avx2, "lut_gemv_full"):

            w_packed = np.ascontiguousarray(cache.w_packed, dtype=np.uint8)

            scales = np.ascontiguousarray(cache.scales, dtype=np.float32)

            biases = np.ascontiguousarray(cache.biases, dtype=np.float32)

            x_c = np.ascontiguousarray(x, dtype=np.float32)

            q_arg = cache.q_packed

            if q_arg is not None:

                q_arg = np.ascontiguousarray(q_arg, dtype=np.uint8)

            if os.environ.get("ASDSL_LUT_DEBUG", "").strip() == "1":

                for name, arr in (

                    ("w_packed", w_packed),

                    ("scales", scales),

                    ("biases", biases),

                    ("x", x_c),

                ):

                    assert arr.flags["C_CONTIGUOUS"], f"{name} not C-contiguous"

                if q_arg is not None:

                    assert q_arg.flags["C_CONTIGUOUS"], "q_packed not C-contiguous"

                    gpr = cache.meta.groups_per_row

                    gs = cache.meta.group_size

                    assert q_arg.shape == (M, gpr, gs), (

                        f"q_packed shape {q_arg.shape} != ({M}, {gpr}, {gs})"

                    )

                    assert q_arg.strides == (gpr * gs, gs, 1), (

                        f"q_packed strides {q_arg.strides}"

                    )

            return _lut_avx2.lut_gemv_full(

                w_packed,

                scales,

                biases,

                x_c,

                M,

                K,

                zeros=zeros,

                q_vals=q_arg,

                tile_groups=tg,

            )

        return _lut_avx2.lut_gemv_avx2_projection(

            cache.w_packed,

            cache.scales,

            cache.biases,

            x,

            M,

            K,

            gs,

            zeros=zeros,

            tile_groups=tg,

        )



    def _tile_partial_python(

        self,

        cache: LUTProjectionCache,

        m: int,

        tile_idx: int,

        row_packed: np.ndarray,

        x: np.ndarray,

        gs: int,

        tg: int,

        groups_per_row: int,

    ) -> float:

        g_start = tile_idx * tg

        n_groups = min(tg, groups_per_row - g_start)

        if n_groups <= 0:

            return 0.0

        k0 = g_start * gs

        k1 = k0 + n_groups * gs



        T_tile = self._get_tile(cache, m, tile_idx, g_start, tg)

        q_vals = self._unpack_row_tile(row_packed, g_start, n_groups, gs)

        x_tile = x[k0:k1].reshape(n_groups, gs)



        dequant = LUTTableBuilder.dequant_from_table(T_tile[:n_groups], q_vals)

        return float((dequant * x_tile).sum())



    def _tile_partial_avx2(

        self,

        cache: LUTProjectionCache,

        m: int,

        tile_idx: int,

        row_packed: np.ndarray,

        x: np.ndarray,

        gs: int,

        tg: int,

        groups_per_row: int,

    ) -> float:

        g_start = tile_idx * tg

        n_groups = min(tg, groups_per_row - g_start)

        if n_groups <= 0:

            return 0.0

        k0 = g_start * gs

        k1 = k0 + n_groups * gs



        T_tile = np.ascontiguousarray(

            self._get_tile(cache, m, tile_idx, g_start, tg)[:n_groups],

            dtype=np.float16,

        )

        q_vals = self._unpack_row_tile(row_packed, g_start, n_groups, gs)

        x_tile = np.ascontiguousarray(x[k0:k1].reshape(n_groups, gs), dtype=np.float32)

        return float(

            _lut_avx2.lut_gemv_avx2_tile(T_tile, q_vals, x_tile, n_groups)

        )



    def _get_tile(

        self,

        cache: LUTProjectionCache,

        row: int,

        tile_idx: int,

        g_start: int,

        tile_groups: int,

    ) -> np.ndarray:

        key = (row, tile_idx)

        if cache.tiles is not None and key in cache.tiles:

            return np.ascontiguousarray(cache.tiles[key], dtype=np.float16)

        return LUTTableBuilder.build_row_tile(

            row,

            tile_idx,

            cache.scales,

            cache.biases,

            cache.meta.groups_per_row,

            cache.meta.group_size,

            zeros=cache.zeros,

            tile_groups=tile_groups,

        )



    @staticmethod

    def _unpack_row_tile(

        row_packed: np.ndarray,

        g_start: int,

        n_groups: int,

        group_size: int,

    ) -> np.ndarray:

        """Unpack nibbles for ``n_groups`` groups starting at group ``g_start``."""

        k0 = g_start * group_size

        byte0 = k0 // 2

        n_elems = n_groups * group_size

        nbytes = (n_elems + 1) // 2

        chunk = row_packed[byte0 : byte0 + nbytes]

        low = (chunk & 0x0F).astype(np.uint8)

        high = ((chunk >> 4) & 0x0F).astype(np.uint8)

        flat = np.stack([low, high], axis=-1).reshape(-1)[:n_elems]

        return flat.reshape(n_groups, group_size)

