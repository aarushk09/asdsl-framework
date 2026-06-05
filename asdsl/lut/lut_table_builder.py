"""Precomputed dequant lookup tables for Phase 1 LUT-native GEMV.

Tables are tiled along K (quantization groups), not output rows. Each K-tile
holds ``T[g][q][i] = (q - zero[g]) * scale[g]`` as float16 for asymmetric Q4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LUTProjectionMeta:
    """Metadata for one weight projection's K-group tiling."""

    rows: int
    cols: int
    group_size: int
    groups_per_row: int
    num_k_tiles: int
    tile_groups: int = 128

    @property
    def tile_footprint_bytes(self) -> int:
        return LUTTableBuilder.tile_working_set_bytes(
            self.cols, self.group_size, self.tile_groups
        )


@dataclass
class LUTProjectionCache:
    """Per-projection LUT context: weights + affine params + tiling metadata."""

    meta: LUTProjectionMeta
    w_packed: np.ndarray
    scales: np.ndarray
    biases: np.ndarray
    zeros: np.ndarray | None = None
    # Optional pre-unpacked nibbles [M, groups_per_row, group_size] for C++ fast path
    q_packed: np.ndarray | None = None
    # Optional eager tiles: (row, tile_idx) -> T_tile float16 [tile_groups, 16, gs]
    tiles: dict[tuple[int, int], np.ndarray] | None = None


class LUTTableBuilder:
    """Build float16 dequant tables ``T[g][q][i]`` for K-group tiles."""

    DEFAULT_TILE_GROUPS = 128

    @staticmethod
    def footprint_bytes(tile_groups: int = 128, group_size: int = 32) -> int:
        """Bytes for one K-tile table ``[tile_groups, 16, group_size]`` float16."""
        return tile_groups * 16 * group_size * 2

    @staticmethod
    def tile_working_set_bytes(
        cols: int,
        group_size: int = 32,
        tile_groups: int = 64,
    ) -> int:
        """L2 budget gate: worst-case LUT working set for one output row.

        ``tile_groups * (cols/group_size) * 16 * group_size`` float16 cells.
        Used by dispatch policy — not the full ``rows * G * 16 * gs`` table.
        """
        g = cols // group_size
        return tile_groups * g * 16 * group_size * 2

    @staticmethod
    def full_table_bytes(rows: int, cols: int, group_size: int = 32) -> int:
        """Total float16 LUT storage if all rows were materialized (diagnostics only)."""
        g = cols // group_size
        return rows * g * 16 * group_size * 2

    @staticmethod
    def num_k_tiles(groups_per_row: int, tile_groups: int = DEFAULT_TILE_GROUPS) -> int:
        return (groups_per_row + tile_groups - 1) // tile_groups

    @classmethod
    def projection_meta(
        cls,
        rows: int,
        cols: int,
        group_size: int,
        tile_groups: int = DEFAULT_TILE_GROUPS,
    ) -> LUTProjectionMeta:
        groups_per_row = cols // group_size
        return LUTProjectionMeta(
            rows=rows,
            cols=cols,
            group_size=group_size,
            groups_per_row=groups_per_row,
            num_k_tiles=cls.num_k_tiles(groups_per_row, tile_groups),
            tile_groups=tile_groups,
        )

    @classmethod
    def build_q_packed(
        cls,
        w_packed: np.ndarray,
        rows: int,
        cols: int,
        group_size: int,
    ) -> np.ndarray:
        """Unpack all Q4 nibbles to ``[rows, groups_per_row, group_size]`` uint8."""
        groups_per_row = cols // group_size
        packed_per_row = cols // 2
        q = np.empty((rows, groups_per_row, group_size), dtype=np.uint8)
        for m in range(rows):
            row_off = m * packed_per_row
            row_packed = w_packed[row_off : row_off + packed_per_row]
            for g in range(groups_per_row):
                k0 = g * group_size
                byte0 = k0 // 2
                n_elems = group_size
                nbytes = (n_elems + 1) // 2
                chunk = row_packed[byte0 : byte0 + nbytes]
                low = (chunk & 0x0F).astype(np.uint8)
                high = ((chunk >> 4) & 0x0F).astype(np.uint8)
                flat = np.stack([low, high], axis=-1).reshape(-1)[:n_elems]
                q[m, g] = flat
        return np.ascontiguousarray(q)

    @classmethod
    def build_projection(
        cls,
        w_packed: np.ndarray,
        scales: np.ndarray,
        biases: np.ndarray,
        rows: int,
        cols: int,
        group_size: int,
        zeros: np.ndarray | None = None,
        tile_groups: int = DEFAULT_TILE_GROUPS,
        eager_max_k_tiles: int = 0,
        eager_max_rows: int = 256,
        build_q_packed: bool = False,
    ) -> LUTProjectionCache:
        """Create projection cache; optionally prebuild tiles for tiny projections only."""
        meta = cls.projection_meta(rows, cols, group_size, tile_groups)
        cache = LUTProjectionCache(
            meta=meta,
            w_packed=np.ascontiguousarray(w_packed, dtype=np.uint8),
            scales=np.ascontiguousarray(scales, dtype=np.float32),
            biases=np.ascontiguousarray(biases, dtype=np.float32),
            zeros=(
                np.ascontiguousarray(zeros, dtype=np.float32)
                if zeros is not None
                else None
            ),
        )
        if (
            meta.num_k_tiles <= eager_max_k_tiles
            and eager_max_k_tiles > 0
            and rows <= eager_max_rows
        ):
            tiles: dict[tuple[int, int], np.ndarray] = {}
            for row in range(rows):
                for tile_idx in range(meta.num_k_tiles):
                    tiles[(row, tile_idx)] = cls.build_row_tile(
                        row,
                        tile_idx,
                        cache.scales,
                        cache.biases,
                        meta.groups_per_row,
                        meta.group_size,
                        zeros=cache.zeros,
                        tile_groups=tile_groups,
                    )
            cache.tiles = {
                k: np.ascontiguousarray(v) for k, v in tiles.items()
            }
        if build_q_packed:
            cache.q_packed = cls.build_q_packed(
                cache.w_packed, rows, cols, group_size
            )
        return cache

    @staticmethod
    def _zero_for_group(
        gidx: int,
        scales: np.ndarray,
        zeros: np.ndarray | None,
        biases: np.ndarray,
    ) -> float:
        if zeros is not None:
            return float(zeros[gidx])
        s = float(scales[gidx])
        if s == 0.0:
            return 0.0
        return -float(biases[gidx]) / s

    @classmethod
    def build_tile(
        cls,
        scales: np.ndarray,
        biases: np.ndarray,
        tile_group_start: int,
        tile_groups: int,
        group_size: int,
        zeros: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build ``T`` for a contiguous run of global group indices (single row slice)."""
        g_end = min(tile_group_start + tile_groups, scales.shape[0])
        n = g_end - tile_group_start
        q = np.arange(16, dtype=np.float32)[:, None]
        T = np.empty((tile_groups, 16, group_size), dtype=np.float16)
        for g_local in range(n):
            gidx = tile_group_start + g_local
            scale_g = float(scales[gidx])
            zero_g = cls._zero_for_group(gidx, scales, zeros, biases)
            T[g_local] = ((q - zero_g) * scale_g).astype(np.float16)
        if n < tile_groups:
            T[n:] = 0
        return np.ascontiguousarray(T)

    @classmethod
    def build_row_tile(
        cls,
        row: int,
        tile_idx: int,
        scales: np.ndarray,
        biases: np.ndarray,
        groups_per_row: int,
        group_size: int,
        zeros: np.ndarray | None = None,
        tile_groups: int = DEFAULT_TILE_GROUPS,
    ) -> np.ndarray:
        """Build ``T`` for one output row and one K-tile index."""
        tile_group_start = tile_idx * tile_groups
        base = row * groups_per_row + tile_group_start
        row_scales = scales[base : base + tile_groups]
        row_biases = biases[base : base + tile_groups]
        row_zeros = zeros[base : base + tile_groups] if zeros is not None else None
        return cls.build_tile(
            row_scales,
            row_biases,
            0,
            tile_groups,
            group_size,
            zeros=row_zeros,
        )

    @classmethod
    def dequant_from_table(
        cls,
        T_tile: np.ndarray,
        q_vals: np.ndarray,
    ) -> np.ndarray:
        """Gather dequantized weights from ``T_tile`` and integer codes ``q_vals``."""
        q_idx = q_vals.astype(np.int64)[:, np.newaxis, :]
        return np.take_along_axis(
            T_tile.astype(np.float32), q_idx, axis=1
        ).squeeze(1)
