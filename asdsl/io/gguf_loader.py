"""GGUF tensor reader with stable GGML type IDs and ASDSL name mapping."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# GGML type IDs (ggml.h) — extend when gguf-py lags behind llama.cpp.
GGML_TYPE_NAMES: dict[int, str] = {
    0: "f32",
    1: "f16",
    2: "q4_0",
    3: "q4_1",
    6: "q5_0",
    7: "q5_1",
    8: "q8_0",
    9: "q8_1",
    10: "q2_k",
    11: "q3_k",
    12: "q4_k",
    13: "q5_k",
    14: "q6_k",
    15: "q8_k",
    16: "iq2_xxs",
    17: "iq2_xs",
    18: "iq3_xxs",
    19: "iq1_s",
    20: "iq4_nl",
    21: "iq3_s",
    22: "iq2_s",
    23: "iq4_xs",
    24: "iq1_m",
    25: "bf16",
}

# K-quant block sizes (bytes per 256 weights along the K / input axis).
_K_BLOCK_BYTES: dict[str, int] = {
    "q2_k": 84,
    "q3_k": 110,
    "q4_k": 144,
    "q5_k": 176,
    "q6_k": 210,
    "q8_k": 256,
}

# GGUF tensor suffix → (asdsl_projection, module).
GGUF_TO_ASDSL: dict[str, tuple[str, str]] = {
    "attn_qkv.weight": ("qkv_proj", "attn"),
    "attn_output.weight": ("o_proj", "attn"),
    "ffn_gate_up.weight": ("gate_up_proj", "ffn"),
    "ffn_gate.weight": ("gate_proj", "ffn"),
    "ffn_up.weight": ("up_proj", "ffn"),
    "ffn_down.weight": ("down_proj", "ffn"),
    "attn_norm.weight": ("attn_norm", "norm"),
    "ffn_norm.weight": ("ffn_norm", "norm"),
    "token_embd.weight": ("token_embd", "embed"),
    "output_norm.weight": ("output_norm", "norm"),
    "output.weight": ("lm_head", "output"),
}

_QUANT_TYPES = frozenset(_K_BLOCK_BYTES.keys()) | frozenset(
    {"q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q8_1"}
)


def ggml_type_name(type_id: int) -> str:
    if type_id in GGML_TYPE_NAMES:
        return GGML_TYPE_NAMES[type_id]
    return f"unknown_{type_id}"


def row_bytes_for_type(type_name: str, in_features: int) -> int:
    """Bytes per GGUF storage row (K axis = in_features for logical W @ x)."""
    t = type_name.lower()
    if t in _K_BLOCK_BYTES:
        return (in_features // 256) * _K_BLOCK_BYTES[t]
    raise ValueError(f"row_bytes_for_type: unsupported type {type_name}")


def logical_shape(file_shape: tuple[int, ...], type_name: str) -> tuple[int, int]:
    """Map on-disk GGUF 2-D shape to logical (out_features, in_features)."""
    if len(file_shape) != 2:
        raise ValueError(f"Expected rank-2 weight, got {file_shape}")
    n0, n1 = int(file_shape[0]), int(file_shape[1])
    t = type_name.lower()
    if t in _K_BLOCK_BYTES:
        # K-quants: file is [in_features, out_features]; dequant yields (out, in).
        return n1, n0
    # Dense: same transpose convention as llama.cpp GGUF export.
    return n1, n0


def _tensor_payload(reader_tensor: Any) -> np.ndarray:
    data = reader_tensor.data
    if hasattr(data, "numpy"):
        return np.asarray(data.numpy(), dtype=np.uint8)
    return np.ascontiguousarray(np.asarray(data), dtype=np.uint8)


def read_gguf_tensors(
    path: str | Path,
    *,
    name_filter: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Read all tensors from a GGUF file.

    Returns dict[name] → {type, type_id, shape, file_shape, logical_shape,
    data, row_bytes, n_bytes}.
    """
    import gguf

    reader = gguf.GGUFReader(str(path))
    out: dict[str, dict[str, Any]] = {}
    for t in reader.tensors:
        name = t.name
        if name_filter and name_filter not in name:
            continue
        type_id = int(t.tensor_type)
        type_name = ggml_type_name(type_id)
        file_shape = tuple(int(x) for x in t.shape)
        payload = _tensor_payload(t)
        info: dict[str, Any] = {
            "type": type_name,
            "type_id": type_id,
            "shape": file_shape,
            "file_shape": file_shape,
            "data": payload,
            "n_bytes": int(payload.nbytes),
        }
        if len(file_shape) == 2:
            log = logical_shape(file_shape, type_name)
            info["logical_shape"] = log
            if type_name.lower() in _K_BLOCK_BYTES:
                info["row_bytes"] = row_bytes_for_type(type_name, log[1])
        out[name] = info
    return out


def dequant_tensor(info: dict[str, Any]) -> np.ndarray:
    """Dequantize one tensor to float32 with shape (out_features, in_features)."""
    from gguf import quants

    tname = info["type"].lower()
    if tname in ("f32",):
        arr = np.asarray(info["data"], dtype=np.float32)
        if arr.ndim == 1:
            return arr
        return np.ascontiguousarray(arr.reshape(logical_shape(info["shape"], tname)), dtype=np.float32)
    if tname in ("f16", "bf16"):
        from gguf import GGMLQuantizationType

        qt = GGMLQuantizationType.F16 if tname == "f16" else GGMLQuantizationType.BF16
        arr = quants.dequantize(info["data"], qt)
        return np.ascontiguousarray(arr, dtype=np.float32)
    if tname in _K_BLOCK_BYTES or tname in _QUANT_TYPES:
        import gguf

        type_id = info.get("type_id")
        if type_id is None:
            rev = {v: k for k, v in GGML_TYPE_NAMES.items()}
            type_id = rev.get(tname, info["type_id"])
        qt = gguf.GGMLQuantizationType(type_id)
        arr = quants.dequantize(info["data"], qt)
        return np.ascontiguousarray(arr, dtype=np.float32)
    raise ValueError(f"dequant_tensor: unsupported type {info.get('type')}")


def k_blocks_rowmajor(info: dict[str, Any], type_name: str | None = None) -> np.ndarray:
    """Flatten GGUF K-quant tensor to row-major bytes for native GEMV kernels."""
    tname = (type_name or str(info.get("type", ""))).lower()
    if tname not in _K_BLOCK_BYTES:
        raise ValueError(f"k_blocks_rowmajor: unsupported type {tname}")
    data = np.ascontiguousarray(info["data"], dtype=np.uint8)
    out, inn = info["logical_shape"]
    bsz = _K_BLOCK_BYTES[tname]
    expected = out * (inn // 256) * bsz
    flat = data.reshape(-1)
    if flat.size != expected:
        raise ValueError(
            f"k_blocks_rowmajor({tname}): byte count {flat.size} != expected {expected} "
            f"for logical {out}x{inn}"
        )
    return flat


def q4k_blocks_rowmajor(info: dict[str, Any]) -> np.ndarray:
    """Flatten GGUF Q4_K tensor to row-major [out_rows × (in/256 × 144)] bytes for gemv_q4km."""
    return k_blocks_rowmajor(info, "q4_k")


def q5k_blocks_rowmajor(info: dict[str, Any]) -> np.ndarray:
    return k_blocks_rowmajor(info, "q5_k")


def q6k_blocks_rowmajor(info: dict[str, Any]) -> np.ndarray:
    return k_blocks_rowmajor(info, "q6_k")


def summarize_tensors(path: str | Path, limit: int = 10) -> list[dict[str, Any]]:
    """Return a JSON-serializable summary of tensors (for phase results)."""
    tensors = read_gguf_tensors(path)
    rows: list[dict[str, Any]] = []
    for name in sorted(tensors.keys()):
        info = tensors[name]
        row = {
            "name": name,
            "type": info["type"],
            "type_id": info.get("type_id"),
            "shape": list(info["shape"]),
        }
        if "logical_shape" in info:
            row["logical_shape"] = list(info["logical_shape"])
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows
