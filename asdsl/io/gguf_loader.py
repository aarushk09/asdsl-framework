"""Minimal GGUF tensor reader and Q4_K block helpers.

This module provides a lightweight API for reading tensor metadata and payloads
from GGUF files, with explicit support for Q4_K tensors used by llama.cpp
Q4_K_M models.
"""

from __future__ import annotations

import os
import struct
from typing import Any

import numpy as np

GGUF_MAGIC = 0x46554747  # b"GGUF"
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_K = 12
GGML_TYPE_BF16 = 30

Q4_K_BLOCK_SIZE = 144
Q4_K_N_PER_BLOCK = 256


def read_gguf_tensors(gguf_path: str) -> dict[str, dict[str, Any]]:
    """Read GGUF tensor payloads into a metadata dictionary.

    Returns:
        Dict mapping tensor name to:
        - type: canonical string (q4_k, f16, f32, bf16, unknown_<id>)
        - ggml_type: integer GGML tensor type id
        - shape: logical tensor shape tuple (rows, cols, ...)
        - data: numpy array (contiguous copy)
        - row_bytes: bytes per row for block-quantized tensors (when available)
    """
    if not os.path.exists(gguf_path):
        raise FileNotFoundError(f"GGUF not found: {gguf_path}")

    try:
        from gguf import GGUFReader  # type: ignore

        reader = GGUFReader(gguf_path)
        out: dict[str, dict[str, Any]] = {}
        for t in reader.tensors:
            ggml_type = int(t.tensor_type)
            # gguf-py stores GGML dims in reverse order for display; reverse to row-major.
            logical_shape = tuple(int(x) for x in reversed(t.shape.tolist()))
            arr = np.asarray(t.data)

            if ggml_type == GGML_TYPE_Q4_K:
                out[t.name] = {
                    "type": "q4_k",
                    "ggml_type": ggml_type,
                    "shape": logical_shape,
                    "row_bytes": int(arr.shape[1]) if arr.ndim >= 2 else int(arr.size),
                    "data": np.ascontiguousarray(arr, dtype=np.uint8).reshape(-1),
                }
            elif ggml_type == GGML_TYPE_F16:
                out[t.name] = {
                    "type": "f16",
                    "ggml_type": ggml_type,
                    "shape": logical_shape,
                    "data": np.ascontiguousarray(arr, dtype=np.float16).reshape(-1),
                }
            elif ggml_type == GGML_TYPE_BF16:
                # gguf-py exposes bf16 tensors as uint16 bit patterns.
                raw_u16 = np.ascontiguousarray(arr, dtype=np.uint16).reshape(-1)
                f32 = (raw_u16.astype(np.uint32) << 16).view(np.float32)
                out[t.name] = {
                    "type": "bf16",
                    "ggml_type": ggml_type,
                    "shape": logical_shape,
                    "data": f32,
                }
            elif ggml_type == GGML_TYPE_F32:
                out[t.name] = {
                    "type": "f32",
                    "ggml_type": ggml_type,
                    "shape": logical_shape,
                    "data": np.ascontiguousarray(arr, dtype=np.float32).reshape(-1),
                }
            else:
                out[t.name] = {
                    "type": f"unknown_{ggml_type}",
                    "ggml_type": ggml_type,
                    "shape": logical_shape,
                    "data": None,
                }
        return out
    except ImportError:
        # Fallback parser for environments where gguf package is unavailable.
        return _read_gguf_tensors_fallback(gguf_path)


def _read_gguf_tensors_fallback(gguf_path: str) -> dict[str, dict[str, Any]]:
    """Minimal binary GGUF parser used only if gguf-py is unavailable."""
    tensors: dict[str, dict[str, Any]] = {}
    with open(gguf_path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file: magic={magic:#x}")

        version = struct.unpack("<I", f.read(4))[0]
        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version: {version}")

        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        for _ in range(n_kv):
            key_len = struct.unpack("<Q", f.read(8))[0]
            f.read(key_len)
            value_type = struct.unpack("<I", f.read(4))[0]
            _skip_gguf_value(f, value_type)

        tensor_infos: list[tuple[str, tuple[int, ...], int, int]] = []
        for _ in range(n_tensors):
            name_len = struct.unpack("<Q", f.read(8))[0]
            name = f.read(name_len).decode("utf-8")
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims))
            ggml_type = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            # GGUF stores dims in reverse order relative to row-major usage.
            shape = tuple(int(x) for x in reversed(dims))
            tensor_infos.append((name, shape, ggml_type, int(offset)))

        alignment = 32
        data_start = (f.tell() + alignment - 1) // alignment * alignment

        for name, shape, ggml_type, offset in tensor_infos:
            f.seek(data_start + offset)
            n_elements = int(np.prod(shape, dtype=np.int64))

            if ggml_type == GGML_TYPE_Q4_K:
                cols = int(shape[-1]) if len(shape) >= 2 else int(shape[0])
                if cols % Q4_K_N_PER_BLOCK != 0:
                    raise ValueError(f"Q4_K tensor {name} has non-divisible cols={cols}")
                row_blocks = cols // Q4_K_N_PER_BLOCK
                row_bytes = row_blocks * Q4_K_BLOCK_SIZE
                rows = int(np.prod(shape[:-1], dtype=np.int64)) if len(shape) > 1 else 1
                byte_count = rows * row_bytes
                raw = np.frombuffer(f.read(byte_count), dtype=np.uint8).copy()
                tensors[name] = {
                    "type": "q4_k",
                    "ggml_type": ggml_type,
                    "shape": shape,
                    "row_bytes": row_bytes,
                    "data": raw,
                }
            elif ggml_type == GGML_TYPE_F16:
                raw = np.frombuffer(f.read(n_elements * 2), dtype=np.float16).copy()
                tensors[name] = {
                    "type": "f16",
                    "ggml_type": ggml_type,
                    "shape": shape,
                    "data": raw,
                }
            elif ggml_type == GGML_TYPE_BF16:
                raw_u16 = np.frombuffer(f.read(n_elements * 2), dtype=np.uint16).copy()
                f32 = (raw_u16.astype(np.uint32) << 16).view(np.float32)
                tensors[name] = {
                    "type": "bf16",
                    "ggml_type": ggml_type,
                    "shape": shape,
                    "data": f32,
                }
            elif ggml_type == GGML_TYPE_F32:
                raw = np.frombuffer(f.read(n_elements * 4), dtype=np.float32).copy()
                tensors[name] = {
                    "type": "f32",
                    "ggml_type": ggml_type,
                    "shape": shape,
                    "data": raw,
                }
            else:
                tensors[name] = {
                    "type": f"unknown_{ggml_type}",
                    "ggml_type": ggml_type,
                    "shape": shape,
                    "data": None,
                }

    return tensors


def _skip_gguf_value(fobj, val_type: int) -> None:
    """Skip one GGUF KV value by type id."""
    type_sizes = {
        0: 1,   # uint8
        1: 1,   # int8
        2: 2,   # uint16
        3: 2,   # int16
        4: 4,   # uint32
        5: 4,   # int32
        6: 8,   # float32 (stored as 4 in gguf spec variants, guarded below)
        7: 8,   # uint64
        10: 4,  # float32
        11: 8,  # float64
    }
    if val_type in type_sizes:
        fobj.read(type_sizes[val_type])
        return
    if val_type == 8:  # string
        slen = struct.unpack("<Q", fobj.read(8))[0]
        fobj.read(slen)
        return
    if val_type == 9:  # array
        arr_type = struct.unpack("<I", fobj.read(4))[0]
        arr_len = struct.unpack("<Q", fobj.read(8))[0]
        for _ in range(arr_len):
            _skip_gguf_value(fobj, arr_type)
        return
    raise ValueError(f"Unknown GGUF value type: {val_type}")


def q4k_unpack_6bit(packed12: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unpack 8 sub-scales and 8 sub-mins from a 12-byte Q4_K field."""
    if packed12.size != 12:
        raise ValueError("q4k_unpack_6bit expects exactly 12 bytes")
    out = np.zeros(16, dtype=np.uint8)
    for idx in range(16):
        bit_pos = idx * 6
        byte_idx = bit_pos // 8
        bit_off = bit_pos % 8
        if bit_off <= 2:
            val = (int(packed12[byte_idx]) >> bit_off) & 0x3F
        else:
            lo = int(packed12[byte_idx]) >> bit_off
            hi = int(packed12[byte_idx + 1]) << (8 - bit_off)
            val = (lo | hi) & 0x3F
        out[idx] = val
    return out[:8], out[8:]


def q4k_dequantize_to_fp32(raw_blocks: np.ndarray, n_elements: int) -> np.ndarray:
    """Dequantize Q4_K packed blocks to a flat float32 vector."""
    raw = np.ascontiguousarray(raw_blocks, dtype=np.uint8).reshape(-1, Q4_K_BLOCK_SIZE)
    n_blocks = raw.shape[0]
    out = np.zeros(n_blocks * Q4_K_N_PER_BLOCK, dtype=np.float32)

    for b in range(n_blocks):
        block = raw[b]
        d = np.frombuffer(block[0:2].tobytes(), dtype=np.float16)[0].astype(np.float32)
        dmin = np.frombuffer(block[2:4].tobytes(), dtype=np.float16)[0].astype(np.float32)
        sub_scales_u6, sub_mins_u6 = q4k_unpack_6bit(block[4:16])
        sub_scales = d * sub_scales_u6.astype(np.float32)
        sub_mins = dmin * sub_mins_u6.astype(np.float32)

        qs = block[16:144]
        base = b * Q4_K_N_PER_BLOCK
        for sb in range(8):
            q16 = qs[sb * 16 : (sb + 1) * 16]
            lo = (q16 & 0x0F).astype(np.float32)
            hi = (q16 >> 4).astype(np.float32)
            vals = np.empty(32, dtype=np.float32)
            vals[0::2] = lo
            vals[1::2] = hi
            out[base + sb * 32 : base + (sb + 1) * 32] = sub_scales[sb] * vals - sub_mins[sb]

    return out[:n_elements]
