"""Disk cache for imatrix-lite per-projection importance vectors (Phase 4 / C0.1)."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

IMATRIX_CACHE_FORMAT = "phi4_imatrix_lite_v1"
WEIGHT_CACHE_FORMAT = "phi4_cpu_weights_v1"
WEIGHT_CACHE_DIRNAME = "phi4_weight_cache"


def _resolve_model_dir() -> Path:
    override = os.environ.get("ASDSL_MODEL_DIR", "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "models" / "phi4"


def _weight_cache_digest(store: Any) -> str:
    wc_path = getattr(store, "_weight_cache_path", None)
    if wc_path is not None:
        return Path(wc_path).stem.replace("phi4_cpu_", "")
    model_dir = _resolve_model_dir()
    index_file = model_dir / "model.safetensors.index.json"
    mtime = str(index_file.stat().st_mtime_ns) if index_file.is_file() else ""
    payload = "|".join(
        [
            str(model_dir.resolve()),
            mtime,
            str(store.bits),
            str(store.group_size),
            str(getattr(store, "_enable_qcsd", False)),
            str(getattr(store, "_draft_bits", 2)),
            str(getattr(store, "_draft_group_size", 32)),
            str(getattr(store, "_enable_sparse", False)),
            f"{getattr(store, '_sparsity_threshold', 0.01):.8g}",
            str(getattr(store, "_symmetric", False)),
            str(getattr(store, "_optimize_clips", False)),
            WEIGHT_CACHE_FORMAT,
        ]
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def imatrix_cache_path_for_store(store: Any) -> Path:
    digest = _weight_cache_digest(store)
    wc_path = getattr(store, "_weight_cache_path", None)
    cache_dir = Path(wc_path).parent if wc_path is not None else _resolve_model_dir().parent / WEIGHT_CACHE_DIRNAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"phi4_imatrix_lite_{digest}.npz"


def _tensor_key(layer: int, name: str) -> str:
    return f"L{layer}_{name}"


def try_load_imatrix_cache(store: Any, path: Path) -> dict[tuple[int, str], np.ndarray] | None:
    if not path.is_file():
        return None
    meta = np.load(path, allow_pickle=True)
    if str(meta.get("format", "")) != IMATRIX_CACHE_FORMAT:
        return None
    out: dict[tuple[int, str], np.ndarray] = {}
    for key in meta.files:
        if key in ("format", "n_samples"):
            continue
        if not key.startswith("L"):
            continue
        layer_s, _, name = key[1:].partition("_")
        out[(int(layer_s), name)] = np.asarray(meta[key], dtype=np.float32)
    if not out:
        return None
    print(f"  imatrix-lite cache restored: {len(out)} tensors ({path.name})", flush=True)
    return out


def save_imatrix_cache(
    store: Any,
    path: Path,
    importance: dict[tuple[int, str], np.ndarray],
    *,
    n_samples: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "format": np.array(IMATRIX_CACHE_FORMAT),
        "n_samples": np.array(n_samples, dtype=np.int64),
    }
    for (layer, name), vec in importance.items():
        payload[_tensor_key(layer, name)] = np.ascontiguousarray(vec, dtype=np.float32)
    t0 = time.perf_counter()
    np.savez_compressed(str(path), **payload)
    dt = time.perf_counter() - t0
    print(f"  imatrix-lite cache saved: {path.name} ({len(importance)} tensors, {dt:.2f}s)", flush=True)
