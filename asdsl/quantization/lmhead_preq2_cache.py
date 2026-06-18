"""Disk cache for lm_head preq2 meta+quant blobs (Phase 3)."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from asdsl.quantization.repack_preq2 import preq2_flat_sizes

if TYPE_CHECKING:
    pass

LMHEAD_PREQ2_FORMAT = "phi4_lmhead_preq2_v1"
WEIGHT_CACHE_FORMAT = "phi4_cpu_weights_v1"
WEIGHT_CACHE_DIRNAME = "phi4_weight_cache"


def _lmhead_cache_enabled() -> bool:
    v = os.environ.get("PHI4_NO_LMHEAD_PREQ2_CACHE", "").strip().lower()
    return v not in ("1", "true", "yes")


def _resolve_model_dir() -> Path:
    override = os.environ.get("ASDSL_MODEL_DIR", "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "models" / "phi4"


def _weight_cache_digest(store: Any) -> str:
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


def lmhead_preq2_cache_path_for_store(store: Any) -> Path:
    wc_path = getattr(store, "_weight_cache_path", None)
    if wc_path is not None:
        digest = Path(wc_path).stem.replace("phi4_cpu_", "")
        cache_dir = Path(wc_path).parent
    else:
        digest = _weight_cache_digest(store)
        cache_dir = _resolve_model_dir().parent / WEIGHT_CACHE_DIRNAME
    lm_gs = int(os.environ.get("ASDSL_LMHEAD_GS", str(store.group_size)))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"phi4_lmhead_preq2_{digest}_gs{lm_gs}.safetensors"


def _expected_sizes(store: Any) -> tuple[int, int, int, int, int]:
    hidden = 3072
    vocab_default = 200064
    vocab = int(store.lm_head.shape[0]) if getattr(store, "lm_head", None) is not None else vocab_default
    hidden = int(store.lm_head.shape[1]) if getattr(store, "lm_head", None) is not None else hidden
    lm_gs = int(os.environ.get("ASDSL_LMHEAD_GS", str(store.group_size)))
    meta_bytes, quant_bytes = preq2_flat_sizes(vocab, hidden, lm_gs)
    return vocab, hidden, lm_gs, meta_bytes, quant_bytes


def try_restore_lmhead_preq2_cache(
    store: Any, path: Path
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (meta, quant) uint8 arrays if cache is valid."""
    if not _lmhead_cache_enabled() or not path.is_file():
        return None

    from safetensors import safe_open
    import safetensors.numpy as st_np

    with safe_open(str(path), framework="pt", device="cpu") as f:
        md = f.metadata()
        if md.get("format") != LMHEAD_PREQ2_FORMAT:
            return None
        vocab, hidden, lm_gs, meta_bytes, quant_bytes = _expected_sizes(store)
        if int(md.get("vocab_size", "0")) != vocab:
            return None
        if int(md.get("hidden_size", "0")) != hidden:
            return None
        if int(md.get("group_size", "0")) != lm_gs:
            return None
        if int(md.get("meta_bytes", "0")) != meta_bytes:
            return None
        if int(md.get("quant_bytes", "0")) != quant_bytes:
            return None

    t0 = time.perf_counter()
    tensors = st_np.load_file(str(path))
    meta = np.ascontiguousarray(tensors["lm_head_meta"], dtype=np.uint8)
    quant = np.ascontiguousarray(tensors["lm_head_quant"], dtype=np.uint8)
    if meta.size != meta_bytes or quant.size != quant_bytes:
        return None
    dt = time.perf_counter() - t0
    print(
        f"  lm_head preq2 cache restored: meta={meta_bytes // 1024}KB "
        f"quant={quant_bytes // (1024 * 1024)}MB ({dt:.2f}s)",
        flush=True,
    )
    return meta, quant


def save_lmhead_preq2_cache(
    store: Any,
    path: Path,
    meta: np.ndarray,
    quant: np.ndarray,
) -> None:
    import safetensors.numpy as st_np

    vocab, hidden, lm_gs, meta_bytes, quant_bytes = _expected_sizes(store)
    meta = np.ascontiguousarray(meta.reshape(-1), dtype=np.uint8)
    quant = np.ascontiguousarray(quant.reshape(-1), dtype=np.uint8)
    if meta.size != meta_bytes or quant.size != quant_bytes:
        raise ValueError(
            f"lm_head preq2 size mismatch: meta {meta.size} vs {meta_bytes}, "
            f"quant {quant.size} vs {quant_bytes}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    md = {
        "format": LMHEAD_PREQ2_FORMAT,
        "vocab_size": str(vocab),
        "hidden_size": str(hidden),
        "group_size": str(lm_gs),
        "meta_bytes": str(meta_bytes),
        "quant_bytes": str(quant_bytes),
    }
    t0 = time.perf_counter()
    st_np.save_file(
        {"lm_head_meta": meta, "lm_head_quant": quant},
        str(path),
        metadata=md,
    )
    dt = time.perf_counter() - t0
    print(f"  lm_head preq2 cache saved: {path.name} ({dt:.2f}s)", flush=True)
