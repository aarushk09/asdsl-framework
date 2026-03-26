"""Mixed Q4/Q3 weight packer driven by an importance matrix (imatrix)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from asdsl.quantization.imatrix import (
    assign_mixed_bits_from_groups,
    group_importance_from_imatrix,
)


def _quantize_group_affine(wg: np.ndarray, n_levels: int) -> tuple[np.ndarray, float, float]:
    """Integer codes 0..n_levels-1 with affine ``w ≈ scale * q + bias`` (per group)."""
    wg = np.asarray(wg, dtype=np.float32).reshape(-1)
    lo = float(wg.min())
    hi = float(wg.max())
    if hi - lo < 1e-8:
        q = np.zeros(wg.shape, dtype=np.uint8)
        return q, 1.0, lo
    scale = (hi - lo) / float(max(n_levels - 1, 1))
    q = np.clip(np.round((wg - lo) / scale), 0, n_levels - 1).astype(np.uint8)
    return q, scale, lo


def _pack_group_q4_nibbles(q: np.ndarray) -> bytes:
    q = np.asarray(q, dtype=np.uint8).reshape(-1)
    assert q.size % 2 == 0
    out = bytearray(q.size // 2)
    for i in range(0, q.size, 2):
        out[i // 2] = (q[i] & 0x0F) | ((q[i + 1] & 0x0F) << 4)
    return bytes(out)


def _pack_group_q3_words(q: np.ndarray) -> bytes:
    """10×3-bit in little-endian uint32 words (matches native unpack)."""
    q = np.asarray(q, dtype=np.uint8).reshape(-1)
    out = bytearray()
    i = 0
    gs = q.size
    while i < gs:
        word = 0
        nvals = min(10, gs - i)
        for j in range(nvals):
            word |= (int(q[i + j]) & 0x07) << (3 * j)
        out.extend(word.to_bytes(4, "little"))
        i += nvals
    return bytes(out)


@dataclass
class MixedQ34Packed:
    """Packed mixed Q4/Q3 weights for :func:`asdsl.kernels.gemv_q3.gemv_q3_mixed`."""

    w_bytes: np.ndarray  # uint8 flat
    group_offsets: np.ndarray  # uint32, length M * G + 1
    bits_per_group: np.ndarray  # uint8, length M * G
    scales: np.ndarray
    biases: np.ndarray
    m: int
    k: int
    group_size: int

    @property
    def groups_per_row(self) -> int:
        return self.k // self.group_size

    def weight_bytes(self) -> int:
        return int(self.w_bytes.nbytes)


def pack_linear_mixed_q34(
    w: np.ndarray,
    imatrix_k: np.ndarray,
    *,
    group_size: int,
    q4_group_fraction: float,
) -> MixedQ34Packed:
    """Pack ``W`` (M, K) with per-group 4-bit (important) or 3-bit columns.

    Uses group-wise affine quantization matching the native GEMV:
    ``w_float ≈ scales * q_int + biases`` with one scale/bias per group.
    """
    w = np.asarray(w, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError("w must be 2D (M, K)")
    m, k = int(w.shape[0]), int(w.shape[1])
    if group_size % 2 != 0:
        raise ValueError("group_size must be even for Q4 nibble packing")
    if k % group_size != 0:
        raise ValueError("K must be divisible by group_size")

    g_per = k // group_size
    g_scores = group_importance_from_imatrix(imatrix_k, k, group_size)
    bits_template = assign_mixed_bits_from_groups(g_scores, q4_group_fraction=q4_group_fraction)

    blobs: list[bytes] = []
    scales_list: list[float] = []
    biases_list: list[float] = []
    bits_out: list[int] = []

    for row in range(m):
        for gi in range(g_per):
            wg = w[row, gi * group_size : (gi + 1) * group_size]
            b = int(bits_template[gi])
            if b == 4:
                q, scale, bias = _quantize_group_affine(wg, 16)
                blobs.append(_pack_group_q4_nibbles(q))
            else:
                q, scale, bias = _quantize_group_affine(wg, 8)
                blobs.append(_pack_group_q3_words(q))
            scales_list.append(scale)
            biases_list.append(bias)
            bits_out.append(b)

    w_flat = b"".join(blobs)
    group_offsets = np.zeros(m * g_per + 1, dtype=np.uint32)
    for i, blob in enumerate(blobs):
        group_offsets[i + 1] = group_offsets[i] + len(blob)
    w_bytes = np.frombuffer(w_flat, dtype=np.uint8).copy()
    return MixedQ34Packed(
        w_bytes=w_bytes,
        group_offsets=group_offsets,
        bits_per_group=np.asarray(bits_out, dtype=np.uint8),
        scales=np.asarray(scales_list, dtype=np.float32),
        biases=np.asarray(biases_list, dtype=np.float32),
        m=m,
        k=k,
        group_size=group_size,
    )


def q4_baseline_weight_bytes(m: int, k: int) -> int:
    """Packed 4-bit row-major storage size (nibbles), for footprint comparisons."""
    return (m * k) // 2


def reference_mixed_gemv_numpy(p: MixedQ34Packed, x: np.ndarray) -> np.ndarray:
    """Dequantize to float in memory and compute ``y = W @ x`` (test reference)."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size != p.k:
        raise ValueError("x length mismatch")
    m, k, gs = p.m, p.k, p.group_size
    g_per = k // gs
    w_deq = np.zeros((m, k), dtype=np.float32)
    idx = 0
    for row in range(m):
        for gi in range(g_per):
            gidx = row * g_per + gi
            b0 = int(p.group_offsets[gidx])
            b1 = int(p.group_offsets[gidx + 1])
            blob = p.w_bytes[b0:b1]
            bits = int(p.bits_per_group[gidx])
            s = float(p.scales[gidx])
            bias = float(p.biases[gidx])
            k0 = gi * gs
            if bits == 4:
                q = np.zeros(gs, dtype=np.float32)
                for j in range(0, gs, 2):
                    byte = blob[j // 2]
                    q[j] = float(byte & 0x0F)
                    q[j + 1] = float(byte >> 4)
            else:
                q = np.zeros(gs, dtype=np.float32)
                off = 0
                t = 0
                while t < gs:
                    word = int.from_bytes(blob[off : off + 4], "little")
                    off += 4
                    for j in range(min(10, gs - t)):
                        q[t + j] = float((word >> (3 * j)) & 0x07)
                    t += min(10, gs - t)
            w_deq[row, k0 : k0 + gs] = s * q + bias
    return w_deq @ x
