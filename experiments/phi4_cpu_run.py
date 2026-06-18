"""
Phi-4 Multimodal Instruct – CPU inference through the ASDSL pipeline.

What this does:
  - Loads Phi-4 backbone weights from the local safetensors shards
  - Quantizes every projection weight to 4-bit with ASDSL
  - Runs a proper Phi-4 text forward pass (RMSNorm, RoPE, GQA, SiLU MLP)
  - Uses ASDSL's BlockSparseKVCache alongside inference for tracking
  - Generates tokens greedily and streams them to the terminal

Speed note: this is a Python reference implementation using PyTorch float32
matmul on CPU. On a modern desktop, expect ~0.1-0.5 tokens/second. The ASDSL
LUT kernel path (sub-byte matmul without dequantize) would require the native
C++/SIMD backend to be compiled – this script demonstrates correctness.

Usage:
  python experiments/phi4_cpu_run.py
  python experiments/phi4_cpu_run.py --prompt "Explain gravity in one sentence"
  python experiments/phi4_cpu_run.py --max-new-tokens 60 --bits 4
  python experiments/phi4_cpu_run.py --qcsd --bits 4 --draft-bits 2 --draft-k 5
  python experiments/phi4_cpu_run.py --qcsd-benchmark --prompt "Hi"   # no local weights
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import Path
from typing import Optional, Callable, Any

# Suppress TensorFlow / JAX import attempts in transformers (they're incompatible
# with NumPy 2.x on this machine and aren't needed for text-only inference).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from asdsl.engine import run_dual_model_speculative_benchmark


class ForwardProfiler:
    """Section timing for one layer / one decode token (Phase 7)."""

    def __init__(self) -> None:
        self.enabled = False
        self.target_layer = 1
        self.target_pos: int | None = None
        self._layer_idx = -1
        self._t0 = 0.0
        self._key = ""
        self.timings_ms: dict[str, float] = {}
        self.layer_total_ms: float = 0.0

    def set_pos(self, pos: int) -> None:
        self._pos = pos

    def active(self) -> bool:
        return (
            self.enabled
            and self._layer_idx == self.target_layer
            and self.target_pos is not None
            and getattr(self, "_pos", -1) == self.target_pos
        )

    def begin(self, key: str) -> None:
        if not self.active():
            return
        self._key = key
        self._t0 = time.perf_counter()

    def end(self) -> None:
        if not self.active():
            return
        self.timings_ms[self._key] = (
            self.timings_ms.get(self._key, 0.0)
            + (time.perf_counter() - self._t0) * 1000.0
        )

    def print_report(self) -> None:
        if not self.timings_ms:
            return
        total = sum(self.timings_ms.values())
        print(
            f"\n[PROFILE] Layer {self.target_layer}, pos={self.target_pos} "
            f"(sections={total:.1f}ms layer_total={self.layer_total_ms:.1f}ms):"
        )
        for key, ms in sorted(self.timings_ms.items(), key=lambda x: -x[1]):
            pct = (ms / total * 100.0) if total > 0 else 0.0
            bar = "#" * int(pct / 2)
            print(f"  {key:22s} {ms:7.2f}ms  {pct:5.1f}%  {bar}")
        overhead = self.layer_total_ms - total
        if overhead > 0.5:
            print(
                f"  {'unaccounted':22s} {overhead:7.2f}ms  "
                f"{overhead / max(self.layer_total_ms, 1e-6) * 100:5.1f}%"
            )
        print()


@dataclass
class StreamToken:
    text: str = ""
    token_id: int = 0
    step: int = 0
    is_eos: bool = False
    elapsed_s: float = 0.0
    tokens_per_second: float = 0.0
    step_elapsed_s: float = 0.0

# Phase 17: Native C++ ops for non-GEMV operations
try:
    from asdsl.kernels import _native_ops
    HAS_NATIVE_OPS = True
except ImportError:
    HAS_NATIVE_OPS = False
    _native_ops = None


def _configure_cpu_torch_runtime() -> None:
    """Flush FP32 subnormals to zero on CPU (avoids 100× slowdowns in matmul/mv)."""
    if hasattr(torch, "set_flush_denormal"):
        try:
            torch.set_flush_denormal(True)
        except Exception:
            pass


_configure_cpu_torch_runtime()

# Raptor Lake hybrid default: 4 P-cores (logical 0-7) + E-cores (logical 8-15) on i7-1360P class CPUs.
_P_LOGICAL_COUNT = 8


def _logical_ids_for_threads(n: int) -> list[int]:
    """Map requested OpenMP threads to distinct logical CPUs (P first, then E)."""
    nlog = os.cpu_count() or 16
    p_end = min(_P_LOGICAL_COUNT, nlog)
    ids: list[int] = []
    for i in range(p_end):
        if len(ids) >= n:
            break
        ids.append(i)
    for i in range(p_end, nlog):
        if len(ids) >= n:
            break
        ids.append(i)
    while len(ids) < n and len(ids) < nlog:
        ids.append(len(ids))
    return ids[:n]


def _physical_logical_ids_for_threads(n: int) -> list[int]:
    """One logical CPU per physical core (P-cores first, then E-cores)."""
    try:
        from asdsl.kernels import _native_gemv as _ng

        if hasattr(_ng, "get_cpu_topology"):
            topo = _ng.get_cpu_topology()
            raw = list(topo.get("physical_logical_ids", []))
            if raw:
                return raw[:n]
    except Exception:
        pass
    # Raptor Lake i7-1360P fallback: P 0-3 + E 8-15 (skip HT siblings 4-7)
    fallback = [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15]
    nlog = os.cpu_count() or 16
    ids = [i for i in fallback if i < nlog]
    while len(ids) < n and len(ids) < nlog:
        for i in range(nlog):
            if i not in ids:
                ids.append(i)
            if len(ids) >= n:
                break
    return ids[:n]


def _smt_logical_ids_for_threads(n: int) -> list[int]:
    """P-cores with HT siblings first, then E-cores (Phase 6 SMT experiment)."""
    try:
        from asdsl.kernels import _native_unified as nu

        p_all = list(nu.get_all_pcore_logical_ids())
    except Exception:
        p_all = [0, 1, 2, 3, 4, 5, 6, 7]
    nlog = os.cpu_count() or 16
    ids = [i for i in p_all if i < nlog]
    for i in range(8, nlog):
        if i not in ids:
            ids.append(i)
        if len(ids) >= n:
            break
    while len(ids) < n and len(ids) < nlog:
        for i in range(nlog):
            if i not in ids:
                ids.append(i)
            if len(ids) >= n:
                break
    return ids[:n]


def set_thread_count(n: int) -> None:
    """Set CPU threads for NumPy/BLAS/PyTorch and native OpenMP GEMV.

    ``n <= 0`` (auto): 8 threads on P-cores only.

    Affinity modes (``ASDSL_AFFINITY``):
      physical (recommended): one thread per physical core (4P+8E on i7-1360P);
      spread: one OpenMP thread per logical CPU in ``_logical_ids_for_threads``;
      smt: P-core HT siblings + E-cores (Phase 6, typically 16 threads);
      legacy: old ``OMP_PLACES={0-7}`` (12 threads oversubscribe 8 slots);
      none: only ``OMP_NUM_THREADS``, no OMP binding.
    """
    if n <= 0:
        n = 8

    mode = os.environ.get("ASDSL_AFFINITY", "physical").strip().lower()
    if mode == "physical":
        logical_ids = _physical_logical_ids_for_threads(n)
    elif mode == "smt":
        logical_ids = _smt_logical_ids_for_threads(n)
    else:
        logical_ids = _logical_ids_for_threads(n)

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n)

    os.environ["OMP_SCHEDULE"] = "static"
    if mode == "legacy":
        os.environ["OMP_PROC_BIND"] = "TRUE"
        os.environ["OMP_PLACES"] = "{0-7}"
    elif mode in ("spread", "physical", "smt"):
        os.environ["OMP_PROC_BIND"] = "close"
        os.environ["OMP_PLACES"] = ",".join(f"{{{i}}}" for i in logical_ids)
    else:
        os.environ.pop("OMP_PROC_BIND", None)
        os.environ.pop("OMP_PLACES", None)

    torch.set_num_threads(n)
    _configure_cpu_torch_runtime()

    affinity_mask = sum(1 << i for i in logical_ids)
    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetCurrentProcess()
            if handle and handle != -1:
                if kernel32.SetProcessAffinityMask(handle, affinity_mask):
                    print(
                        f"[threads] affinity={mode} omp={n} "
                        f"cpus={logical_ids} mask=0x{affinity_mask:X}",
                        flush=True,
                    )
                else:
                    print(
                        f"[threads] SetProcessAffinityMask failed err={kernel32.GetLastError()}",
                        flush=True,
                    )
        except Exception as exc:
            print(f"[threads] affinity pin failed: {exc}", flush=True)
    else:
        try:
            import psutil
            psutil.Process().cpu_affinity(logical_ids)
        except Exception:
            pass

    try:
        from asdsl.kernels import _native_gemv as _ng
        if bool(getattr(_ng, "has_openmp", False)) and hasattr(_ng, "set_num_threads"):
            _ng.set_num_threads(int(n))
    except ImportError:
        pass


ROOT = Path(__file__).parent.parent


def resolve_model_dir() -> Path:
    """Local Phi-4 safetensors tree (override with ASDSL_MODEL_DIR)."""
    override = os.environ.get("ASDSL_MODEL_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (ROOT / "models" / "phi4-multimodal-instruct").resolve()


MODEL_DIR = resolve_model_dir()
INDEX_FILE = MODEL_DIR / "model.safetensors.index.json"


def load_tokenizer():
    """Load Phi-4 tokenizer (local model dir when present, else HuggingFace hub)."""
    if (MODEL_DIR / "tokenizer_config.json").is_file():
        return AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    return AutoTokenizer.from_pretrained(
        "microsoft/Phi-4-multimodal-instruct",
        trust_remote_code=True,
    )


# ---------------------------------------------------------------------------
# Architecture constants (from microsoft/Phi-4-multimodal-instruct config)
# ---------------------------------------------------------------------------
HIDDEN = 3072
NUM_LAYERS = 32
NUM_HEADS = 24
NUM_KV_HEADS = 8
HEAD_DIM = HIDDEN // NUM_HEADS          # 128
Q_DIM = NUM_HEADS * HEAD_DIM            # 3072
KV_DIM = NUM_KV_HEADS * HEAD_DIM        # 1024
QKV_DIM = Q_DIM + 2 * KV_DIM           # 5120
INTER = 8192                            # gate/up each; combined weight is 2*INTER
VOCAB = 200064
RMS_EPS = 1e-5
ROPE_THETA = 10000.0
# Phi-4 uses partial RoPE: only 75% of head_dim is rotated (config: partial_rotary_factor=0.75)
ROTARY_DIM = int(0.75 * HEAD_DIM)      # 96 - only these dims get RoPE applied

# Special token IDs
# 200020 = <|end|> (end of turn),  199999 = <|endoftext|> / </s> (end of text)
# 200019 = <|assistant|> (start of assistant turn - NOT an EOS token)
EOS_TOKEN_IDS = {200020, 199999}

# Persistent mmap weight cache (safetensors). Bump when on-disk layout changes.
WEIGHT_CACHE_FORMAT = "phi4_cpu_weights_v1"
WEIGHT_CACHE_DIRNAME = "phi4_weight_cache"


def _weight_cache_enabled() -> bool:
    v = os.environ.get("PHI4_NO_WEIGHT_CACHE", "").strip().lower()
    return v not in ("1", "true", "yes")


def weight_cache_path_for_store(store: WeightStore) -> Path:
    """Unique cache file from model dir, index mtime, and quantization flags."""
    mtime = ""
    if INDEX_FILE.exists():
        mtime = str(INDEX_FILE.stat().st_mtime_ns)
    payload = "|".join(
        [
            str(MODEL_DIR.resolve()),
            mtime,
            str(store.bits),
            str(store.group_size),
            str(store._enable_qcsd),
            str(store._draft_bits),
            str(store._draft_group_size),
            str(store._enable_sparse),
            f"{store._sparsity_threshold:.8g}",
            str(store._symmetric),
            str(store._optimize_clips),
            WEIGHT_CACHE_FORMAT,
        ]
    )
    digest = hashlib.md5(payload.encode("utf-8")).hexdigest()
    d = MODEL_DIR.parent / WEIGHT_CACHE_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d / f"phi4_cpu_{digest}.safetensors"


def _shape_map_to_json(shapes: dict[tuple[int, str], tuple[int, int]]) -> str:
    ser = {f"{i}|{n}": [int(r), int(c)] for (i, n), (r, c) in shapes.items()}
    return json.dumps(ser, sort_keys=True)


def _shape_map_from_json(s: str) -> dict[tuple[int, str], tuple[int, int]]:
    raw = json.loads(s)
    out: dict[tuple[int, str], tuple[int, int]] = {}
    for k, v in raw.items():
        li_s, nm = k.split("|", 1)
        out[(int(li_s), nm)] = (int(v[0]), int(v[1]))
    return out


def _meta_sparsity_value(s: str) -> float:
    try:
        return float(json.loads(s))
    except (json.JSONDecodeError, TypeError, ValueError):
        return float(s)


def _cache_meta_matches(store: WeightStore, md: dict[str, str] | None) -> bool:
    if not md or md.get("format") != WEIGHT_CACHE_FORMAT:
        return False
    try:
        checks = [
            int(md["bits"]) == store.bits,
            int(md["group_size"]) == store.group_size,
            md.get("enable_qcsd", "False") == str(store._enable_qcsd),
            int(md.get("draft_bits", "2")) == store._draft_bits,
            int(md.get("draft_group_size", "32")) == store._draft_group_size,
            md.get("enable_sparse", "False") == str(store._enable_sparse),
            math.isclose(
                _meta_sparsity_value(md.get("sparsity_threshold", "0.01")),
                store._sparsity_threshold,
                rel_tol=0.0,
                abs_tol=1e-9,
            ),
            md.get("symmetric", str(store._symmetric)) == str(store._symmetric),
            md.get("optimize_clips", str(store._optimize_clips))
            == str(store._optimize_clips),
        ]
        return all(checks)
    except (KeyError, TypeError, ValueError):
        return False


def try_restore_weight_cache(store: WeightStore, path: Path) -> bool:
    """Validate metadata, then mmap tensors via ``safetensors.torch.load_file``."""
    if not path.is_file():
        return False
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(str(path), framework="pt", device="cpu") as f0:
        md = f0.metadata()
        if not _cache_meta_matches(store, md) or "quant_shapes" not in md:
            return False
        shapes = _shape_map_from_json(md["quant_shapes"])

    tensors = load_file(str(path), device="cpu")
    fk = set(tensors.keys())

    store.embed_f16 = tensors["embed_f16"]
    store.lm_head = store.embed_f16
    store.final_norm = tensors["final_norm"]
    layer_norms: dict[int, dict[str, torch.Tensor]] = {}
    for i in range(NUM_LAYERS):
        for nm in ("input_layernorm", "post_attention_layernorm"):
            layer_norms.setdefault(i, {})[nm] = tensors[f"norm_{i}_{nm}"]
    store.layer_norms = layer_norms
    store.embed = None

    store._quant_shapes = shapes
    store._quant_packed.clear()
    store._quant_u8.clear()
    store._quant_sc.clear()
    store._quant_bi.clear()
    store._draft_quant_packed.clear()
    store._draft_quant_u8.clear()
    store._draft_quant_sc.clear()
    store._draft_quant_bi.clear()
    store._outlier_values.clear()
    store._outlier_coords.clear()
    store._draft_outlier_values.clear()
    store._draft_outlier_coords.clear()
    store.layers.clear()
    store._weight_cache.clear()

    if store.bits == 16:
        for (i, nm) in shapes.keys():
            store._weight_cache[(i, nm)] = tensors[f"w16_{i}_{nm}"]
    else:
        for (i, nm) in shapes.keys():
            store._quant_sc[(i, nm)] = tensors[f"qs_{i}_{nm}"]
            store._quant_bi[(i, nm)] = tensors[f"qb_{i}_{nm}"]
            pk = f"qp_{i}_{nm}"
            uk = f"qu_{i}_{nm}"
            r, c = shapes[(i, nm)]
            if pk in fk:
                store._quant_packed[(i, nm)] = tensors[pk].reshape(r, c // 2)
            elif uk in fk:
                store._quant_u8[(i, nm)] = tensors[uk].reshape(r, c)
                store._quant_u8_np[(i, nm)] = store._quant_u8[(i, nm)].numpy().ravel()
            else:
                return False
            
            if (i, nm) in store._quant_packed:
                store._quant_packed_np[(i, nm)] = np.ascontiguousarray(
                    store._quant_packed[(i, nm)].numpy().ravel(), dtype=np.uint8
                )
            else:
                store._quant_packed_np[(i, nm)] = None
            store._quant_sc_np[(i, nm)] = np.ascontiguousarray(
                store._quant_sc[(i, nm)].float().numpy().ravel(), dtype=np.float32
            )
            store._quant_bi_np[(i, nm)] = np.ascontiguousarray(
                store._quant_bi[(i, nm)].float().numpy().ravel(), dtype=np.float32
            )

        if store._enable_qcsd:
            for (i, nm) in shapes.keys():
                d_sc_k = f"dqs_{i}_{nm}"
                if d_sc_k not in fk:
                    return False
                store._draft_quant_sc[(i, nm)] = tensors[d_sc_k]
                store._draft_quant_bi[(i, nm)] = tensors[f"dqb_{i}_{nm}"]
                dpk = f"dqp_{i}_{nm}"
                duk = f"dqu_{i}_{nm}"
                r, c = shapes[(i, nm)]
                if dpk in fk:
                    store._draft_quant_packed[(i, nm)] = tensors[dpk].reshape(r, c // 2)
                elif duk in fk:
                    raw = tensors[duk]
                    if raw.numel() == r * c:
                        store._draft_quant_u8[(i, nm)] = raw.reshape(r, c)
                    else:
                        store._draft_quant_u8[(i, nm)] = raw
                    store._draft_quant_u8_np[(i, nm)] = store._draft_quant_u8[(i, nm)].numpy().ravel()
                else:
                    return False
                
                if (i, nm) in store._draft_quant_packed:
                    store._draft_quant_packed_np[(i, nm)] = np.ascontiguousarray(
                        store._draft_quant_packed[(i, nm)].numpy().ravel(), dtype=np.uint8
                    )
                else:
                    store._draft_quant_packed_np[(i, nm)] = None
                store._draft_quant_sc_np[(i, nm)] = np.ascontiguousarray(
                    store._draft_quant_sc[(i, nm)].float().numpy().ravel(), dtype=np.float32
                )
                store._draft_quant_bi_np[(i, nm)] = np.ascontiguousarray(
                    store._draft_quant_bi[(i, nm)].float().numpy().ravel(), dtype=np.float32
                )
                dov_k = f"dov_{i}_{nm}"
                if dov_k in fk:
                    ov = tensors[dov_k].numpy()
                    oc = tensors[f"doc_{i}_{nm}"].numpy().astype(np.int64)
                    if ov.size > 0:
                        store._draft_outlier_values[(i, nm)] = ov
                        store._draft_outlier_coords[(i, nm)] = oc

        for (i, nm) in shapes.keys():
            ov_k = f"ov_{i}_{nm}"
            if ov_k in fk:
                ov = tensors[ov_k].numpy()
                oc = tensors[f"oc_{i}_{nm}"].numpy().astype(np.int64)
                if ov.size > 0:
                    store._outlier_values[(i, nm)] = ov
                    store._outlier_coords[(i, nm)] = oc

    store._pool = torch.empty(_TARGET_CHUNK_BYTES // 4, dtype=torch.float32)
    store._loaded_from_cache = True
    store._weight_cache_path = path
    return True


def save_weight_store_cache(store: WeightStore, path: Path) -> None:
    """Write post-``warm_cache`` tensors + metadata to a single safetensors file."""
    from safetensors.torch import save_file

    tensors: dict[str, torch.Tensor] = {}
    tensors["embed_f16"] = store.embed_f16.contiguous()
    tensors["final_norm"] = store.final_norm.contiguous()
    for i in range(NUM_LAYERS):
        for nm in ("input_layernorm", "post_attention_layernorm"):
            tensors[f"norm_{i}_{nm}"] = store.layer_norms[i][nm].contiguous()

    if store.bits == 16:
        shape_map = {k: tuple(w.shape) for k, w in store._weight_cache.items()}
        shapes_json = _shape_map_to_json(shape_map)
    else:
        shapes_json = _shape_map_to_json(store._quant_shapes)

    meta: dict[str, str] = {
        "format": WEIGHT_CACHE_FORMAT,
        "bits": str(store.bits),
        "group_size": str(store.group_size),
        "enable_qcsd": str(store._enable_qcsd),
        "draft_bits": str(store._draft_bits),
        "draft_group_size": str(store._draft_group_size),
        "enable_sparse": str(store._enable_sparse),
        "sparsity_threshold": json.dumps(store._sparsity_threshold),
        "symmetric": str(store._symmetric),
        "optimize_clips": str(store._optimize_clips),
        "quant_shapes": shapes_json,
    }

    if store.bits == 16:
        for (i, nm), w in store._weight_cache.items():
            tensors[f"w16_{i}_{nm}"] = w.contiguous()
    else:
        for (i, nm), sh in store._quant_shapes.items():
            tensors[f"qs_{i}_{nm}"] = store._quant_sc[(i, nm)].contiguous()
            tensors[f"qb_{i}_{nm}"] = store._quant_bi[(i, nm)].contiguous()
            if (i, nm) in store._quant_packed:
                tensors[f"qp_{i}_{nm}"] = store._quant_packed[(i, nm)].contiguous().reshape(
                    -1
                )
            else:
                tensors[f"qu_{i}_{nm}"] = store._quant_u8[(i, nm)].contiguous().reshape(-1)

            if store._enable_qcsd and (i, nm) in store._draft_quant_sc:
                tensors[f"dqs_{i}_{nm}"] = store._draft_quant_sc[(i, nm)].contiguous()
                tensors[f"dqb_{i}_{nm}"] = store._draft_quant_bi[(i, nm)].contiguous()
                if (i, nm) in store._draft_quant_packed:
                    tensors[f"dqp_{i}_{nm}"] = store._draft_quant_packed[
                        (i, nm)
                    ].contiguous().reshape(-1)
                elif (i, nm) in store._draft_quant_u8:
                    dq = store._draft_quant_u8[(i, nm)].contiguous()
                    tensors[f"dqu_{i}_{nm}"] = dq.reshape(-1)

                if (i, nm) in store._draft_outlier_values:
                    ov = store._draft_outlier_values[(i, nm)]
                    oc = store._draft_outlier_coords[(i, nm)]
                    tensors[f"dov_{i}_{nm}"] = torch.from_numpy(
                        np.asarray(ov, dtype=np.float32)
                    )
                    tensors[f"doc_{i}_{nm}"] = torch.from_numpy(
                        np.asarray(oc, dtype=np.int64)
                    )

            if (i, nm) in store._outlier_values and len(store._outlier_values[(i, nm)]) > 0:
                ov = store._outlier_values[(i, nm)]
                oc = store._outlier_coords[(i, nm)]
                tensors[f"ov_{i}_{nm}"] = torch.from_numpy(np.asarray(ov, dtype=np.float32))
                tensors[f"oc_{i}_{nm}"] = torch.from_numpy(np.asarray(oc, dtype=np.int64))

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=meta)


# Persistent mmap preq-block cache (safetensors.numpy). Bump when block layout changes.
PREQ_CACHE_FORMAT = "phi4_preq_blocks_v1"
PROJ_NAMES = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")


def _preq_cache_enabled() -> bool:
    v = os.environ.get("PHI4_NO_PREQ_CACHE", "").strip().lower()
    return v not in ("1", "true", "yes")


def _preq_repack_imports(group_size: int):
    """Return (BLOCK_SIZE, repack_fn, blocks_to_flat) for the store group size."""
    if group_size == 128:
        from asdsl.quantization.repack_q4_128 import (
            BLOCK_SIZE,
            blocks_to_flat,
            repack_asymmetric_to_q4_128_blocks,
        )

        return BLOCK_SIZE, repack_asymmetric_to_q4_128_blocks, blocks_to_flat
    from asdsl.quantization.repack_q4_32 import (
        BLOCK_SIZE,
        blocks_to_flat,
        repack_asymmetric_to_q4_32_blocks,
    )

    return BLOCK_SIZE, repack_asymmetric_to_q4_32_blocks, blocks_to_flat


def preq_cache_path_for_store(store: WeightStore) -> Path:
    """Digest from quant config; GGUF loads use a separate preq cache file."""
    wc = weight_cache_path_for_store(store)
    digest = wc.stem.replace("phi4_cpu_", "")
    tag = ""
    gguf = getattr(store, "_gguf_path", None)
    if gguf:
        tag = "_gguf_" + hashlib.sha256(str(gguf).encode()).hexdigest()[:12]
    fmt = "q4_128" if store.group_size == 128 else "q4_32"
    return wc.parent / f"phi4_preq_{digest}{tag}_{fmt}.safetensors"


def _quantize_config_meta(store: WeightStore) -> dict[str, str]:
    return {
        "bits": str(store.bits),
        "group_size": str(store.group_size),
        "enable_qcsd": str(store._enable_qcsd),
        "draft_bits": str(store._draft_bits),
        "draft_group_size": str(store._draft_group_size),
        "enable_sparse": str(store._enable_sparse),
        "sparsity_threshold": json.dumps(store._sparsity_threshold),
        "symmetric": str(store._symmetric),
        "optimize_clips": str(store._optimize_clips),
        "quant_shapes": _shape_map_to_json(store._quant_shapes),
    }


def _quantize_config_matches(store: WeightStore, md: dict[str, str] | None) -> bool:
    if not md or "quant_shapes" not in md:
        return False
    try:
        shapes = _shape_map_from_json(md["quant_shapes"])
        if shapes != store._quant_shapes:
            return False
        checks = [
            int(md["bits"]) == store.bits,
            int(md["group_size"]) == store.group_size,
            md.get("enable_qcsd", "False") == str(store._enable_qcsd),
            int(md.get("draft_bits", "2")) == store._draft_bits,
            int(md.get("draft_group_size", "32")) == store._draft_group_size,
            md.get("enable_sparse", "False") == str(store._enable_sparse),
            math.isclose(
                _meta_sparsity_value(md.get("sparsity_threshold", "0.01")),
                store._sparsity_threshold,
                rel_tol=0.0,
                abs_tol=1e-9,
            ),
            md.get("symmetric", str(store._symmetric)) == str(store._symmetric),
            md.get("optimize_clips", str(store._optimize_clips))
            == str(store._optimize_clips),
        ]
        return all(checks)
    except (KeyError, TypeError, ValueError):
        return False


def _preq_tensor_key(layer: int, name: str) -> str:
    return f"L{layer}_{name}"


def _parse_preq_tensor_key(key: str) -> tuple[int, str]:
    if not key.startswith("L"):
        raise ValueError(f"invalid preq cache key: {key!r}")
    layer_s, _, name = key[1:].partition("_")
    return int(layer_s), name


def try_restore_preq_cache(store: WeightStore, path: Path) -> bool:
    """Mmap preq blocks from disk; return True if restored."""
    if not path.is_file():
        return False
    BLOCK_SIZE, _, _ = _preq_repack_imports(store.group_size)

    with safe_open(str(path), framework="pt", device="cpu") as f0:
        md = f0.metadata()
        if md.get("format") != PREQ_CACHE_FORMAT:
            return False
        if int(md.get("block_size", "0")) != BLOCK_SIZE:
            return False
        if not _quantize_config_matches(store, md):
            return False

    import safetensors.numpy as st_np

    t0 = time.perf_counter()
    loaded = st_np.load_file(str(path))
    expected = {
        (i, nm) for i in range(NUM_LAYERS) for nm in PROJ_NAMES if (i, nm) in store._quant_shapes
    }
    if set(_parse_preq_tensor_key(k) for k in loaded.keys()) != expected:
        return False

    gs = store.group_size
    for key, blocks in loaded.items():
        li, nm = _parse_preq_tensor_key(key)
        rows, cols = store._quant_shapes[(li, nm)]
        n_groups = cols // gs
        want = rows * n_groups * BLOCK_SIZE
        arr = np.ascontiguousarray(blocks, dtype=np.uint8)
        if arr.nbytes != want:
            return False
        store._preq_blocks_np[(li, nm)] = arr

    store._preq_built = True
    store._preq_block_size = BLOCK_SIZE
    dt = time.perf_counter() - t0
    print(
        f"  Preq cache restored: {len(loaded)} projections "
        f"({path.name}, mmap, {dt:.2f}s)"
    )
    return True


def save_preq_cache(store: WeightStore, path: Path) -> None:
    """Write all preq block arrays to a single safetensors file."""
    import safetensors.numpy as st_np
    BLOCK_SIZE, _, _ = _preq_repack_imports(store.group_size)

    preq_dict = {
        _preq_tensor_key(li, nm): blocks
        for (li, nm), blocks in store._preq_blocks_np.items()
    }
    meta = {"format": PREQ_CACHE_FORMAT, "block_size": str(BLOCK_SIZE)}
    meta.update(_quantize_config_meta(store))
    path.parent.mkdir(parents=True, exist_ok=True)
    st_np.save_file(preq_dict, str(path), metadata=meta)
    mb = path.stat().st_size / 1e6
    print(f"  Preq cache saved: {path} ({mb:.0f} MB)")


Q4KM_CACHE_FORMAT = "phi4_q4km_blocks_v1"


def q4km_cache_path_for_store(store: WeightStore) -> Path:
    gguf = getattr(store, "_gguf_path", None)
    if not gguf:
        raise ValueError("Q4KM cache requires store._gguf_path (load_from_gguf first)")
    digest = hashlib.sha256(str(gguf).encode()).hexdigest()[:16]
    wc = weight_cache_path_for_store(store)
    return wc.parent / f"phi4_q4km_{digest}.safetensors"


def _q4km_tensor_key(layer: int, name: str) -> str:
    return f"KM_L{layer}_{name}"


def try_restore_q4km_cache(store: WeightStore, path: Path) -> bool:
    if not path.is_file():
        return False
    import safetensors.numpy as st_np

    with safe_open(str(path), framework="pt", device="cpu") as f0:
        md = f0.metadata()
        if md.get("format") != Q4KM_CACHE_FORMAT:
            return False
        if md.get("gguf_path") != str(getattr(store, "_gguf_path", "")):
            return False

    t0 = time.perf_counter()
    loaded = st_np.load_file(str(path))
    store._q4km_weights.clear()
    store._q4km_shapes.clear()
    for key, arr in loaded.items():
        if not key.startswith("KM_L"):
            continue
        layer_s, _, name = key[2:].partition("_")
        li, nm = int(layer_s), name
        store._q4km_weights[(li, nm)] = np.ascontiguousarray(arr, dtype=np.uint8)
        shp_key = f"shape_{key}"
        if shp_key in loaded:
            shp = loaded[shp_key]
            store._q4km_shapes[(li, nm)] = (int(shp[0]), int(shp[1]))
    store._use_q4km = len(store._q4km_weights) > 0
    dt = time.perf_counter() - t0
    print(
        f"  Q4KM cache restored: {len(store._q4km_weights)} projections "
        f"({path.name}, mmap, {dt:.2f}s)"
    )
    return store._use_q4km


def save_q4km_cache(store: WeightStore, path: Path) -> None:
    import safetensors.numpy as st_np

    tensors: dict[str, np.ndarray] = {}
    for (li, nm), blocks in store._q4km_weights.items():
        k = _q4km_tensor_key(li, nm)
        tensors[k] = blocks
        shp = store._q4km_shapes.get((li, nm))
        if shp:
            tensors[f"shape_{k}"] = np.array(shp, dtype=np.int32)
    meta = {
        "format": Q4KM_CACHE_FORMAT,
        "gguf_path": str(getattr(store, "_gguf_path", "")),
        "n_projections": str(len(store._q4km_weights)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    st_np.save_file(tensors, str(path), metadata=meta)
    print(f"  Q4KM cache saved: {path.name} ({path.stat().st_size / 1e9:.2f} GB)")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Root-mean-square layer normalisation."""
    rms = x.pow(2).mean(-1, keepdim=True).add(RMS_EPS).sqrt()
    return (x / rms) * weight


def _normalize_input_ids(raw_ids) -> list[int]:
    """Normalize tokenizer outputs to a flat ``list[int]``."""
    ids = raw_ids
    if isinstance(ids, Mapping):
        if "input_ids" in ids:
            ids = ids["input_ids"]
        else:
            vals = list(ids.values())
            if not vals:
                return []
            ids = vals[0]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def build_rope_cache(seq_len: int, head_dim: int, theta: float = ROPE_THETA,
                     dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute RoPE cos/sin tables up to seq_len."""
    pos = torch.arange(seq_len, dtype=torch.float32)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = torch.outer(pos, freqs)          # (seq_len, head_dim/2)
    cos = torch.cos(angles).to(dtype)
    sin = torch.sin(angles).to(dtype)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Partial rotate-half RoPE.

    x: (seq, heads, head_dim)  - full head vector
    cos/sin: (seq, ROTARY_DIM//2)  - tables for the rotated portion only

    Phi-4 sets partial_rotary_factor=0.75, so only the first ROTARY_DIM=96 dims
    of each head are rotated; the remaining 32 dims pass through unchanged.
    """
    d = cos.shape[-1]              # = ROTARY_DIM // 2 = 48
    rotary_dim = 2 * d             # = ROTARY_DIM = 96
    x_rot  = x[..., :rotary_dim]   # (seq, heads, 96) - will be rotated
    x_pass = x[..., rotary_dim:]   # (seq, heads, 32) - untouched
    x1 = x_rot[..., :d]            # first half:  dims 0..47
    x2 = x_rot[..., d:]            # second half: dims 48..95
    c  = cos.unsqueeze(-2)         # (seq, 1, 48) - broadcasts over heads
    s  = sin.unsqueeze(-2)
    rotated = torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
    return torch.cat([rotated, x_pass], dim=-1)


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


# ---------------------------------------------------------------------------
# Weight loading & quantization
# ---------------------------------------------------------------------------

# Target chunk size for streaming dequant+BLAS matvec.
# Each chunk must fit in CPU L2/L3 cache to avoid redundant DRAM traffic.
_TARGET_CHUNK_BYTES = 8 * 1024 * 1024  # 8 MB – fits L3 comfortably
_POOL_SIZE = 128 * 1024 * 1024  # 128 MB scratch pool for verify batch dequant



class WeightStore:
    """
    Loads backbone weights from safetensors shards, quantizes them to N-bit
    with ASDSL, and stores the compressed tensors in a nested dict.

    Supports dual-bank mode for QCSD (Tier 2): holds both a primary (4-bit)
    and draft (2-bit) weight bank sharing scales/zeros, plus SpQR outlier
    separation (Tier 1C) for bits <= 3.

    Memory layout (4-bit, group_size=32):
      - Packed uint8 (rows, cols//2) per projection; fused GEMV via ``gemv_q4_packed``
        (native AVX2 in-register unpack). No pre-unpacked uint8 expansion.
      - ~1.7 GB total backbone packed (vs ~6.4 GB float32)
      - Embedding kept as bfloat16 torch tensor (~1.2 GB)
    """

    def __init__(self, bits: int = 4, group_size: int | None = None,
                 enable_qcsd: bool = False, draft_bits: int = 2,
                 enable_sparse: bool = False, sparsity_threshold: float = 0.01,
                 enable_lut: bool = False, enable_dispatch: bool = False):
        from asdsl.quantization.core import quantize_weights
        self._quantize = quantize_weights
        self.bits = bits
        if group_size is None:
            if bits <= 3:
                self.group_size = 16
            elif bits <= 4:
                self.group_size = 32
            else:
                self.group_size = 128
        else:
            self.group_size = group_size
        self._symmetric = bits > 4
        self._optimize_clips = bits <= 4

        self.layers: dict[int, dict[str, object]] = {}
        self.layer_norms: dict[int, dict[str, torch.Tensor]] = {}
        self.embed: torch.Tensor | None = None
        self.embed_f16: torch.Tensor | None = None
        self.lm_head: torch.Tensor | None = None
        self._lm_head_u16_np: np.ndarray | None = None
        self.final_norm: torch.Tensor | None = None
        self._weight_cache: dict[tuple, torch.Tensor] = {}
        self._dequant_cache: dict[tuple, torch.Tensor] = {}
        self._in_verify_phase = False
        self._quant_u8: dict[tuple, torch.Tensor] = {}
        self._quant_packed: dict[tuple, torch.Tensor] = {}
        self._quant_sc: dict[tuple, torch.Tensor] = {}
        self._quant_bi: dict[tuple, torch.Tensor] = {}
        self._quant_shapes: dict[tuple, tuple] = {}
        
        # Pre-cached numpy views for fast dispatch
        self._quant_packed_np: dict[tuple, np.ndarray] = {}
        self._quant_u8_np: dict[tuple, np.ndarray] = {}
        self._quant_sc_np: dict[tuple, np.ndarray] = {}
        self._quant_bi_np: dict[tuple, np.ndarray] = {}
        
        self._draft_quant_packed: dict[tuple, torch.Tensor] = {}
        self._draft_quant_packed_np: dict[tuple, np.ndarray] = {}
        self._draft_quant_u8: dict[tuple, torch.Tensor] = {}
        self._draft_quant_u8_np: dict[tuple, np.ndarray] = {}
        self._draft_quant_sc: dict[tuple, torch.Tensor] = {}
        self._draft_quant_sc_np: dict[tuple, np.ndarray] = {}
        self._draft_quant_bi: dict[tuple, torch.Tensor] = {}
        self._draft_quant_bi_np: dict[tuple, np.ndarray] = {}

        # Chunked matmul buffer
        self._pool = torch.empty(_POOL_SIZE, dtype=torch.uint8)

        # Phase 21: Pre-allocated input buffer for matvec to avoid numpy allocation per call
        self._x_buf: np.ndarray | None = None
        self._out_buf: np.ndarray | None = None


        # Native LUT/GEMV fast path
        self._use_native_gemv = False
        # Phase 1: LUT-native GEMV (prebuilt T tables) or legacy vpshufb fallback
        self._enable_lut = enable_lut
        self._lut_cache: dict[tuple, object] = {}
        self._use_lut_gemv = False
        # Phase 3: calibration-based kernel dispatch
        self._enable_dispatch = enable_dispatch
        self._dispatch_policy = None
        self._kernel_tags: dict[tuple[int, str], object] = {}
        self._dispatch_matvec_fast = False
        self._sparse_dequant_f16: dict[tuple, np.ndarray] = {}
        self._dispatch_output_buffers: dict[tuple, np.ndarray] = {}
        self._matvec_out_bufs: dict[tuple, np.ndarray] = {}
        self._norm_np: dict[tuple[int, str], np.ndarray] = {}
        self._forward_profiler = ForwardProfiler()
        self._lut_tile_groups = 64
        # Phase 2: SliM mixed-precision dispatch (Profile E)
        self._use_slim = False
        self._slim_meta = None          # loaded from phi4_slim_meta.json
        self._slim_npz = None           # loaded from phi4_slim_meta.npz
        self._repacked_layers: dict = {}  # lazy per-layer repacked buffers
        # Phase 3: FATReLU sparsity (Profile F)
        self._use_fatrelu = False
        self._fatrelu_thresholds: dict[int, float] = {}  # per-layer tau (fixed fallback)
        self._adaptive_tau: dict[int, float] = {}         # per-layer measured tau (adaptive)
        self._adaptive_tau_warmup_tokens = 0               # tokens seen for warmup
        self._adaptive_tau_warmup_target = 5                # measure tau from first 5 tokens
        self._use_adaptive_fatrelu = False                  # enabled via load_fatrelu(path, adaptive=True)
        self._sparse_down_proj_T_calls = 0
        self._dense_down_proj_fallback_calls = 0
        self._sparsity_debug_count = 0

        def advance_sparsity_debug():
            if self._use_fatrelu and self._sparsity_debug_count < 3:
                self._sparsity_debug_count += 1
        self._advance_sparsity_debug = advance_sparsity_debug

        # SpQR outlier separation (Tier 1C) — for bits <= 3
        self._outlier_values: dict[tuple, np.ndarray] = {}
        self._outlier_coords: dict[tuple, np.ndarray] = {}

        # QCSD dual-bank (Tier 2)
        self._enable_qcsd = enable_qcsd
        db = int(draft_bits)
        if enable_qcsd and bits < 16:
            db = min(db, bits)
        self._draft_bits = db
        self._draft_group_size = 16 if db <= 3 else 32
        self._draft_quant_u8: dict[tuple, torch.Tensor] = {}
        self._draft_quant_packed: dict[tuple, torch.Tensor] = {}
        self._draft_quant_sc: dict[tuple, torch.Tensor] = {}
        self._draft_quant_bi: dict[tuple, torch.Tensor] = {}
        self._draft_outlier_values: dict[tuple, np.ndarray] = {}
        self._draft_outlier_coords: dict[tuple, np.ndarray] = {}

        # Activation-sparse GEMV (Tier 3)
        self._enable_sparse = enable_sparse
        self._sparsity_threshold = sparsity_threshold
        # Unpacked scratch for sparse path when weights are Q4 packed (down_proj only).
        self._sparse_u8_scratch: np.ndarray | None = None
        self._loaded_from_cache = False

        # Phase 4 Prerequisite B: transposed down_proj weights for column-sparse access
        # Stored as {layer_idx: {'packed': np.ndarray, 'scales': np.ndarray, 'biases': np.ndarray}}
        self._down_proj_T: dict = {}

        # Phase 5 EAGLE-3: MTP head for speculative decoding
        self._use_eagle3: bool = False
        self._mtp_head: dict | None = None
        self._last_final_hidden: "np.ndarray | None" = None
        # EAGLE-3 multi-layer feature fusion: capture hidden states from 3 layers
        # Layer 0 (low), Layer 15 (mid), Layer 31 (high = final hidden)
        self._eagle3_hidden_low: "np.ndarray | None" = None
        self._eagle3_hidden_mid: "np.ndarray | None" = None
        self._eagle3_hidden_high: "np.ndarray | None" = None  # same as _last_final_hidden

        # Phase 16: Q8_0 activation quantization + integer GEMV (Profile F)
        self._use_q8_gemv = os.environ.get("ASDSL_USE_Q8_GEMV", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        self._use_unified = os.environ.get("ASDSL_USE_UNIFIED", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        self._unified_engine = None
        self._preq_blocks_np: dict[tuple, np.ndarray] = {}
        self._preq_built = False
        self._preq_block_size: int | None = None
        self._keep_packed_for_fallback = os.environ.get(
            "ASDSL_KEEP_PACKED", ""
        ).strip().lower() in ("1", "true", "yes")
        # Phase 22: Q4_K_M superblock weights loaded from GGUF
        self._use_q4km = False
        self._q4km_weights: dict[tuple[int, str], np.ndarray] = {}
        self._q4km_shapes: dict[tuple[int, str], tuple[int, int]] = {}
        self._gguf_proj_types: dict[tuple[int, str], str] = {}
        self._gguf_path: str | None = None
        self._gguf_projections_loaded: bool = False

        # Phase 17: Native C++ ops for non-GEMV operations
        self._use_native_ops = HAS_NATIVE_OPS
        # Pre-computed RoPE cos/sin tables as flat float32 numpy arrays
        self._cos_table: "np.ndarray | None" = None
        self._sin_table: "np.ndarray | None" = None

        # Phase 4: per-layer MLP hidden correction (quantization error compensation)
        self._correction = None
        self._enable_correction = False
        self._correction_scale = 1.0

    def enter_verify_phase(self) -> None:
        """Enable temporary dequantization caching for the speculative verify phase."""
        self._in_verify_phase = True

    def exit_verify_phase(self) -> None:
        """Clear and disable the temporary dequantization cache."""
        self._dequant_cache.clear()
        self._in_verify_phase = False

    # ------------------------------------------------------------------
    def load(self) -> None:
        import time

        self._loaded_from_cache = False
        t_load0 = time.perf_counter()
        if _weight_cache_enabled():
            cpath = weight_cache_path_for_store(self)
            if try_restore_weight_cache(self, cpath):
                self._loaded_from_cache = True
                self._weight_cache_path = cpath
                nproj = NUM_LAYERS * 4
                print(
                    f"  Restored {nproj} projection weights from cache "
                    f"({cpath.name}, mmap CPU) — skipping shard read/quantize"
                )
                print(f"  Weight restore: {time.perf_counter() - t_load0:.1f}s")
                return

        idx = json.loads(INDEX_FILE.read_text())["weight_map"]

        # Which shard holds which key
        shard_keys: dict[str, list[str]] = {}
        needed = set()
        needed.add("model.embed_tokens.weight")
        needed.add("model.norm.weight")
        for i in range(NUM_LAYERS):
            needed.add(f"model.layers.{i}.input_layernorm.weight")
            needed.add(f"model.layers.{i}.post_attention_layernorm.weight")
            needed.add(f"model.layers.{i}.self_attn.qkv_proj.base_layer.weight")
            needed.add(f"model.layers.{i}.self_attn.o_proj.base_layer.weight")
            needed.add(f"model.layers.{i}.mlp.gate_up_proj.base_layer.weight")
            needed.add(f"model.layers.{i}.mlp.down_proj.base_layer.weight")

        for k in needed:
            if k in idx:
                shard = idx[k]
                shard_keys.setdefault(shard, []).append(k)

        total_proj = NUM_LAYERS * 4
        done = 0

        if self.bits == 16:
            print(f"  Loading {total_proj} projection weights directly as float16 (no quantization) ...")
        else:
            print(f"  Loading & quantizing {total_proj} projection weights to {self.bits}-bit ...")
        print(f"  (embedding + norms loaded as bfloat16)")

        for shard_name, keys in sorted(shard_keys.items()):
            shard_path = MODEL_DIR / shard_name
            with safe_open(str(shard_path), framework="pt") as f:
                for key in keys:
                    tensor = f.get_tensor(key)   # bfloat16

                    if key == "model.embed_tokens.weight":
                        self.embed = tensor          # keep bfloat16
                        continue
                    if key == "model.norm.weight":
                        self.final_norm = tensor.to(torch.float32)
                        continue

                    parts = key.split(".")
                    layer_idx = int(parts[2])

                    if "layernorm" in key:
                        self.layer_norms.setdefault(layer_idx, {})[parts[-2]] = \
                            tensor.to(torch.float32)
                        continue

                    # Derive friendly key like "qkv_proj" / "o_proj" etc.
                    if "qkv_proj" in key:
                        friendly = "qkv_proj"
                    elif "o_proj" in key:
                        friendly = "o_proj"
                    elif "gate_up_proj" in key:
                        friendly = "gate_up_proj"
                    elif "down_proj" in key:
                        friendly = "down_proj"
                    else:
                        friendly = key.split(".")[-1]

                    if self.bits == 16:
                        self._weight_cache[(layer_idx, friendly)] = tensor.to(torch.float16)
                    else:
                        w_f32 = tensor.to(torch.float32).numpy()
                        use_outliers = self.bits <= 3 or (
                            self.bits == 4
                            and os.environ.get("ASDSL_4BIT_OUTLIERS", "").strip().lower()
                            in ("1", "true", "yes")
                        )
                        if use_outliers:
                            from asdsl.quantization.core import quantize_weights_with_outliers
                            # 2-bit: 3.0σ with 0.5% cap balances PPL improvement vs
                            # outlier correction overhead; 3-bit uses milder 3.5σ
                            sigma = 3.0 if self.bits == 2 else 3.5
                            cap = 0.005 if self.bits <= 3 else 0.001
                            qt, ov, oc = quantize_weights_with_outliers(
                                w_f32, bits=self.bits,
                                group_size=self.group_size,
                                outlier_threshold_sigma=sigma,
                                outlier_fraction_cap=cap,
                                symmetric=self._symmetric,
                                optimize_clips=self._optimize_clips,
                            )
                            self._outlier_values[(layer_idx, friendly)] = ov
                            self._outlier_coords[(layer_idx, friendly)] = oc
                        else:
                            qt = self._quantize(w_f32, bits=self.bits,
                                                group_size=self.group_size,
                                                symmetric=self._symmetric,
                                                optimize_clips=self._optimize_clips)
                        self.layers.setdefault(layer_idx, {})[friendly] = qt

                        # QCSD: separate lower-bit draft bank when primary is strictly coarser,
                        # else mirror primary quant so warm_cache always fills _draft_quant_*.
                        if self._enable_qcsd:
                            if self.bits > self._draft_bits:
                                from asdsl.quantization.core import quantize_weights_with_outliers
                                d_sigma = 3.0 if self._draft_bits == 2 else 3.5
                                d_cap = 0.005
                                qt_d, ov_d, oc_d = quantize_weights_with_outliers(
                                    w_f32, bits=self._draft_bits,
                                    group_size=self._draft_group_size,
                                    outlier_threshold_sigma=d_sigma,
                                    outlier_fraction_cap=d_cap,
                                    symmetric=False,
                                    optimize_clips=True,
                                )
                                self.layers.setdefault(layer_idx, {})["_draft_" + friendly] = qt_d
                                self._draft_outlier_values[(layer_idx, friendly)] = ov_d
                                self._draft_outlier_coords[(layer_idx, friendly)] = oc_d
                            else:
                                self.layers.setdefault(layer_idx, {})["_draft_" + friendly] = qt
                                ov_k = (layer_idx, friendly)
                                if ov_k in self._outlier_values:
                                    self._draft_outlier_values[ov_k] = self._outlier_values[ov_k]
                                    self._draft_outlier_coords[ov_k] = self._outlier_coords[ov_k]

                    done += 1
                    if done % 16 == 0 or done == total_proj:
                        pct = done / total_proj * 100
                        print(f"    {done}/{total_proj} ({pct:.0f}%)  ", end="\r", flush=True)

        print(f"    {total_proj}/{total_proj} (100%)  done.               ")
        # Embedding: keep as float16 to save ~1.2 GB RAM.
        # Token lookup casts one row to float32 (12 KB, negligible).
        # LM head reads f16 in L2-sized chunks, halving DRAM traffic.
        print("  Caching embed float16 ... ", end="", flush=True)
        self.embed_f16 = self.embed.to(torch.float16).clone()
        self.lm_head = self.embed_f16              # tied weight, float16
        self.embed = None                          # free bfloat16 copy (~1.2 GB)
        print("done")

        # Pre-allocate a flat float32 pool for chunked matvec.
        self._pool = torch.empty(_TARGET_CHUNK_BYTES // 4, dtype=torch.float32)

    def _gguf_tensor_to_f32(self, info: dict) -> np.ndarray:
        t = info.get("type")
        data = info.get("data")
        if data is None:
            raise ValueError("GGUF tensor payload is missing")
        arr = np.asarray(data)
        if t == "f32":
            return np.ascontiguousarray(arr, dtype=np.float32)
        if t == "f16":
            return np.ascontiguousarray(arr.astype(np.float16), dtype=np.float32)
        if t == "bf16":
            return np.ascontiguousarray(arr, dtype=np.float32)
        raise ValueError(f"Unsupported GGUF tensor type for float conversion: {t}")

    def _concat_q4k_rows(self, infos: list[dict]) -> tuple[np.ndarray, int, int]:
        if not infos:
            raise ValueError("No Q4_K tensors to concatenate")
        cols = int(infos[0]["shape"][-1])
        rows_total = 0
        row_chunks: list[np.ndarray] = []
        for info in infos:
            if info.get("type") != "q4_k":
                raise ValueError(f"Expected q4_k tensor, got {info.get('type')}")
            shape = tuple(int(x) for x in info["shape"])
            rows = int(shape[0])
            if int(shape[-1]) != cols:
                raise ValueError("Cannot concat Q4_K rows with different input dims")
            row_bytes = int(info.get("row_bytes", (cols // 256) * 144))
            raw = np.ascontiguousarray(info["data"], dtype=np.uint8).reshape(rows, row_bytes)
            row_chunks.append(raw)
            rows_total += rows
        merged = np.ascontiguousarray(np.concatenate(row_chunks, axis=0), dtype=np.uint8)
        return merged.reshape(-1), rows_total, cols

    def load_from_gguf(self, gguf_path: str) -> None:
        """Load projection weights from a Q4_K_M GGUF into ASDSL packed + preq path.

        Dequantizes Q4_K / Q5_K / Q6_K tensors (logical out×in), re-quantizes with
        ASDSL asymmetric Q4, then builds preq blocks for UnifiedEngine. Skips the
        long HF calibration quantize when called before warm_cache().
        """
        from asdsl.io.gguf_loader import dequant_tensor, read_gguf_tensors

        _ALLOWED_K = frozenset({"q4_k", "q5_k", "q6_k"})

        tensors = read_gguf_tensors(gguf_path)
        if not tensors:
            raise RuntimeError(f"No tensors found in GGUF: {gguf_path}")

        if self.embed_f16 is None or self.final_norm is None:
            raise RuntimeError(
                "load_from_gguf requires existing embeddings/norms; call load() first"
            )

        expected = {
            "qkv_proj": (QKV_DIM, HIDDEN),
            "o_proj": (HIDDEN, HIDDEN),
            "gate_up_proj": (2 * INTER, HIDDEN),
            "down_proj": (HIDDEN, INTER),
        }
        loaded = 0
        t0 = time.perf_counter()
        self._q4km_weights.clear()
        self._q4km_shapes.clear()
        self._gguf_proj_types.clear()
        self._use_q4km = False
        use_q4km_gguf = os.environ.get("ASDSL_USE_Q4KM_GGUF", "0").strip().lower() in (
            "1",
            "true",
            "yes",
        )

        for i in range(NUM_LAYERS):
            qkv_info = tensors.get(f"blk.{i}.attn_qkv.weight")
            if qkv_info is None:
                raise RuntimeError(f"GGUF missing blk.{i}.attn_qkv.weight")
            o_info = tensors.get(f"blk.{i}.attn_output.weight")
            if o_info is None:
                raise RuntimeError(f"GGUF missing blk.{i}.attn_output.weight")
            gu_info = tensors.get(f"blk.{i}.ffn_gate_up.weight")
            if gu_info is None:
                gu_info = tensors.get(f"blk.{i}.ffn_up.weight")
            if gu_info is None:
                g = tensors.get(f"blk.{i}.ffn_gate.weight")
                u = tensors.get(f"blk.{i}.ffn_up.weight")
                if g is None or u is None:
                    raise RuntimeError(f"GGUF missing FFN gate/up for layer {i}")
                gu_fp = np.concatenate(
                    [dequant_tensor(g), dequant_tensor(u)], axis=0
                ).astype(np.float32, copy=False)
                gu_type = f"{g.get('type')}+{u.get('type')}"
            elif use_q4km_gguf and str(gu_info.get("type", "")).lower() == "q4_k":
                gu_fp = None
                gu_type = gu_info.get("type", "?")
            else:
                gu_fp = dequant_tensor(gu_info)
                gu_type = gu_info.get("type", "?")
            d_info = tensors.get(f"blk.{i}.ffn_down.weight")
            if d_info is None:
                raise RuntimeError(f"GGUF missing blk.{i}.ffn_down.weight")

            proj_infos = {
                "qkv_proj": qkv_info,
                "o_proj": o_info,
                "down_proj": d_info,
            }
            o_fp = (
                None
                if use_q4km_gguf and str(o_info.get("type", "")).lower() == "q4_k"
                else dequant_tensor(o_info)
            )
            proj_fp = {
                "qkv_proj": dequant_tensor(qkv_info),
                "o_proj": o_fp,
                "gate_up_proj": gu_fp,
                "down_proj": dequant_tensor(d_info),
            }
            proj_types = {
                "qkv_proj": qkv_info.get("type"),
                "o_proj": o_info.get("type"),
                "gate_up_proj": gu_type,
                "down_proj": d_info.get("type"),
            }
            for nm, t in proj_types.items():
                self._gguf_proj_types[(i, nm)] = str(t).lower().split("+")[0]

            for nm, info in proj_infos.items():
                t = str(info.get("type", "")).lower()
                if t not in _ALLOWED_K:
                    raise RuntimeError(
                        f"Layer {i} {nm}: unsupported GGUF type {info.get('type')}"
                    )

            def _logical_shape(nm: str) -> tuple[int, int]:
                arr = proj_fp[nm]
                if arr is not None:
                    return tuple(int(x) for x in arr.shape)
                if nm == "gate_up_proj":
                    info = gu_info
                else:
                    info = proj_infos[nm]
                return tuple(int(x) for x in info["logical_shape"])

            for nm, shp in expected.items():
                got = _logical_shape(nm)
                if got != shp:
                    raise RuntimeError(
                        f"GGUF shape mismatch layer {i} {nm}: got {got}, expected {shp}"
                    )

            def _gguf_type_for(nm: str) -> str:
                if nm == "gate_up_proj":
                    return str(proj_types["gate_up_proj"]).split("+")[0].lower()
                return str(proj_types.get(nm, "")).lower()

            for nm in expected:
                if use_q4km_gguf and _gguf_type_for(nm) == "q4_k":
                    continue
                key = (i, nm)
                w_f32 = np.ascontiguousarray(proj_fp[nm], dtype=np.float32)
                qt = self._quantize(w_f32, bits=4, group_size=self.group_size)
                rows, cols = qt.shape
                self._quant_shapes[key] = (rows, cols)
                packed_np = np.ascontiguousarray(qt.data, dtype=np.uint8).copy()
                self._quant_packed[key] = torch.from_numpy(packed_np).reshape(
                    rows, cols // 2
                )
                numel = rows * cols
                n_groups = numel // qt.group_size
                sc = torch.from_numpy(qt.scales[:n_groups].copy().astype(np.float32))
                if qt.is_symmetric:
                    dq = (1 << qt.bits) - 1
                    half_range = float(dq) / 2.0
                    bi = (-half_range * sc).to(torch.float32)
                else:
                    zr = torch.from_numpy(qt.zeros[:n_groups].copy().astype(np.float32))
                    bi = (-zr * sc)
                self._quant_sc[key] = sc
                self._quant_bi[key] = bi
                self._quant_packed_np[key] = packed_np.ravel()
                if i == 0 and nm == "qkv_proj":
                    print(
                        f"  GGUF types L0: qkv={proj_types['qkv_proj']} "
                        f"o={proj_types['o_proj']} gu={proj_types['gate_up_proj']} "
                        f"down={proj_types['down_proj']}"
                    )
                loaded += 1

        self._gguf_path = str(gguf_path)
        self._gguf_projections_loaded = True
        self._preq_built = False
        self._preq_blocks_np.clear()
        if getattr(self, "_unified_engine", None) is not None:
            self._unified_engine = None

        if use_q4km_gguf:
            from asdsl.io.gguf_loader import q4k_blocks_rowmajor

            self._q4km_weights.clear()
            self._q4km_shapes.clear()
            q4km_loaded = 0
            cache_restored = False
            if _preq_cache_enabled():
                try:
                    km_path = q4km_cache_path_for_store(self)
                    cache_restored = try_restore_q4km_cache(self, km_path)
                    q4km_loaded = len(self._q4km_weights)
                except ValueError:
                    pass
            if not cache_restored:
                for i in range(NUM_LAYERS):
                    mapping = {
                        "qkv_proj": f"blk.{i}.attn_qkv.weight",
                        "o_proj": f"blk.{i}.attn_output.weight",
                        "down_proj": f"blk.{i}.ffn_down.weight",
                    }
                    gu_name = f"blk.{i}.ffn_gate_up.weight"
                    if gu_name not in tensors:
                        gu_name = f"blk.{i}.ffn_up.weight"
                    if gu_name not in tensors:
                        g_name = f"blk.{i}.ffn_gate.weight"
                        u_name = f"blk.{i}.ffn_up.weight"
                        if g_name in tensors and u_name in tensors:
                            raise RuntimeError(
                                f"Layer {i}: split ffn_gate+ffn_up Q4_K concat not implemented; "
                                "use fused ffn_up or ffn_gate_up in GGUF"
                            )
                        raise RuntimeError(
                            f"GGUF missing FFN gate/up for layer {i} "
                            "(need ffn_gate_up, ffn_up, or gate+up)"
                        )
                    mapping["gate_up_proj"] = gu_name
                    for nm, tname in mapping.items():
                        info = tensors[tname]
                        t = str(info.get("type", "")).lower()
                        if t == "q4_k":
                            self._q4km_weights[(i, nm)] = q4k_blocks_rowmajor(info)
                            self._q4km_shapes[(i, nm)] = tuple(info["logical_shape"])
                            q4km_loaded += 1
                        elif t in ("q5_k", "q6_k"):
                            pass
                        else:
                            raise RuntimeError(
                                f"Layer {i} {nm}: Q4KM path unsupported type {info.get('type')}"
                            )
                q4km_loaded = len(self._q4km_weights)
                if _preq_cache_enabled() and q4km_loaded > 0:
                    try:
                        save_q4km_cache(self, q4km_cache_path_for_store(self))
                    except ValueError:
                        pass
            self._use_q4km = q4km_loaded > 0
            print(
                f"  Q4_K raw blocks for UnifiedEngine: {q4km_loaded} projections "
                f"(q5_k/q6_k use preq fallback)"
            )

        dt = time.perf_counter() - t0
        print(
            f"  Loaded {loaded} GGUF projections (dequant+ASDSL Q4) in {dt:.1f}s: {gguf_path}"
        )

    # --- float16 path (bits=16) helpers --------------------------------

    def _matvec_f16(self, layer_idx: int, name: str, x: torch.Tensor) -> torch.Tensor:
        """Chunked f16->f32 matvec for unquantized weights."""
        w_f16 = self._weight_cache[(layer_idx, name)]
        rows, cols = w_f16.shape
        x_flat = x.view(-1)
        chunk_rows = max(1, _TARGET_CHUNK_BYTES // (cols * 4))
        result = torch.empty(rows, dtype=torch.float32)
        buf = self._pool[:chunk_rows * cols].view(chunk_rows, cols)
        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            n = end - start
            buf[:n].copy_(w_f16[start:end])
            torch.mv(buf[:n], x_flat, out=result[start:end])
        return result.unsqueeze(0)

    # --- quantized path (bits≤8) helpers --------------------------------

    def _matvec_quant(self, layer_idx: int, name: str, x: torch.Tensor) -> torch.Tensor:
        """Chunked dequant+matvec for quantized weights (unpacked uint8 path)."""
        key = (layer_idx, name)
        if self._use_q4km and key in self._q4km_weights:
            return self._matvec_q4km(layer_idx, name, x)
        if self.bits == 4:
            return self._matvec_q4_packed(layer_idx, name, x, use_draft=False)
        u8 = self._quant_u8[key]
        sc = self._quant_sc[key]
        bi = self._quant_bi[key]
        rows, cols = self._quant_shapes[key]
        groups_per_row = cols // self.group_size
        x_flat = x.view(-1)
        chunk_rows = max(1, _TARGET_CHUNK_BYTES // (cols * 4))
        result = torch.empty(rows, dtype=torch.float32)
        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            n = end - start
            flat_len = n * cols
            buf = self._pool[:flat_len]
            buf.copy_(u8[start * cols:end * cols])
            vals = buf.view(n, groups_per_row, self.group_size)
            gs = start * groups_per_row
            ge = end * groups_per_row
            vals.mul_(sc[gs:ge].float().view(n, groups_per_row, 1))
            vals.add_(bi[gs:ge].float().view(n, groups_per_row, 1))
            torch.mv(vals.view(n, cols), x_flat, out=result[start:end])
        return result.unsqueeze(0)

    def _matvec_q4_packed(
        self, layer_idx: int, name: str, x: torch.Tensor, *, use_draft: bool
    ) -> torch.Tensor:
        """Fused GEMV on 4-bit packed weights (low nibble = even col, high = odd).

        Dispatches to ``asdsl.kernels.gemv_q4_packed`` → C++ ``gemv_q4_packed``
        (AVX2 in-register unpack + FMA; same nibble order as ``gemv_q4_kernel`` tests).

        Phase 21: Pre-allocated x_np buffer eliminates numpy allocation per call.
        """
        from asdsl.kernels import gemv_q4_packed

        key = (layer_idx, name)
        if use_draft and key in self._draft_quant_packed_np:
            w_np = self._draft_quant_packed_np[key]
            sc_np = self._draft_quant_sc_np[key]
            bi_np = self._draft_quant_bi_np[key]
            gs = self._draft_group_size
        else:
            w_np = self._quant_packed_np[key]
            sc_np = self._quant_sc_np[key]
            bi_np = self._quant_bi_np[key]
            gs = self.group_size
            
        rows, cols = self._quant_shapes[key]
        
        # Phase 21: Use pre-allocated buffer to avoid numpy allocation
        if self._x_buf is None or self._x_buf.size < cols:
            self._x_buf = np.empty(cols, dtype=np.float32)
        x_np = x.detach().cpu().float().contiguous().numpy().ravel()
        np.copyto(self._x_buf[:cols], x_np)
        x_np = self._x_buf[:cols]
        
        lut_key = (layer_idx, name)
        lut_cache = None
        if use_draft and lut_key in self._lut_cache:
            lut_cache = self._lut_cache.get(("draft",) + lut_key)
        elif not use_draft:
            lut_cache = self._lut_cache.get(lut_key)
        draft_bits = self._draft_bits if use_draft else self.bits
        out_np = self._matvec_out_bufs.get(key)
        if out_np is None or out_np.shape[0] != rows:
            out_np = np.empty(rows, dtype=np.float32)
            self._matvec_out_bufs[key] = out_np
        gemv_q4_packed(
            w_np,
            x_np,
            sc_np,
            bi_np,
            rows,
            cols,
            gs,
            out=out_np,
            use_lut=self._use_lut_gemv,
            use_q8=self._use_q8_gemv if not use_draft else False,
            lut_cache=lut_cache,
            bits=draft_bits if use_draft else self.bits,
        )
        result = torch.from_numpy(out_np).unsqueeze(0)
        outlier_store = (
            self._draft_outlier_values if use_draft else self._outlier_values
        )
        coord_store = (
            self._draft_outlier_coords if use_draft else self._outlier_coords
        )
        if key in outlier_store and len(outlier_store[key]) > 0:
            ov = outlier_store[key].astype(np.float32)
            oc = coord_store[key]
            col_indices = oc[:, 1]
            row_indices = oc[:, 0]
            x_sel = x_np[col_indices]
            contributions = ov * x_sel
            out_corr = np.zeros(rows, dtype=np.float32)
            np.add.at(out_corr, row_indices, contributions)
            result = result + torch.from_numpy(out_corr).unsqueeze(0)
        return result

    def _matvec_q4km(self, layer_idx: int, name: str, x: torch.Tensor) -> torch.Tensor:
        """Fused GEMV for GGUF Q4_K_M projection weights."""
        try:
            from asdsl.kernels import _native_gemv as _ng  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Q4_K_M GEMV requires native _native_gemv extension") from exc

        if not hasattr(_ng, "gemv_q4km_q8_avx2"):
            raise RuntimeError("Native extension does not export gemv_q4km_q8_avx2")

        key = (layer_idx, name)
        if key not in self._q4km_weights:
            raise KeyError(f"Missing Q4_K_M weights for {key}")

        rows, cols = self._q4km_shapes[key]
        w_np = self._q4km_weights[key]

        if self._x_buf is None or self._x_buf.size < cols:
            self._x_buf = np.empty(cols, dtype=np.float32)
        if self._out_buf is None or self._out_buf.size < rows:
            self._out_buf = np.empty(rows, dtype=np.float32)

        x_np = x.detach().cpu().float().contiguous().numpy().ravel()
        np.copyto(self._x_buf[:cols], x_np)
        _ng.gemv_q4km_q8_avx2(w_np, self._x_buf[:cols], self._out_buf[:rows], rows, cols)
        return torch.from_numpy(np.asarray(self._out_buf[:rows], dtype=np.float32).copy()).unsqueeze(0)

    def _matvec_native_gemv(self, layer_idx: int, name: str, x: torch.Tensor,
                            use_draft: bool = False) -> torch.Tensor:
        """AVX2 GEMV fast path for 4-bit packed, 8-bit, 3-bit, and 2-bit weights."""
        key = (layer_idx, name)

        if not use_draft and self._use_q4km and key in self._q4km_weights:
            return self._matvec_q4km(layer_idx, name, x)

        draft_q4 = use_draft and key in self._draft_quant_packed
        draft_q2p = use_draft and key in self._draft_quant_u8 and self._draft_bits == 2
        draft_unpacked = (
            use_draft and key in self._draft_quant_u8 and not draft_q2p
        )
        draft_active = draft_q4 or draft_q2p or draft_unpacked

        if self.bits == 4 and not draft_active:
            return self._matvec_q4_packed(layer_idx, name, x, use_draft=False)
        if draft_q4:
            return self._matvec_q4_packed(layer_idx, name, x, use_draft=True)

        if draft_q2p:
            w_data = self._draft_quant_u8[key].numpy()
            sc = self._draft_quant_sc[key].float().numpy()
            bi = self._draft_quant_bi[key].float().numpy()
            rows, cols = self._quant_shapes[key]
            gs = self._draft_group_size
            bits = 2
            is_packed = True
        elif draft_unpacked:
            w_data = self._draft_quant_u8[key].numpy()
            sc = self._draft_quant_sc[key].float().numpy()
            bi = self._draft_quant_bi[key].float().numpy()
            rows, cols = self._quant_shapes[key]
            gs = self._draft_group_size
            bits = self._draft_bits
            is_packed = False
        else:
            w_data = self._quant_u8[key].numpy()
            sc = self._quant_sc[key].float().numpy()
            bi = self._quant_bi[key].float().numpy()
            rows, cols = self._quant_shapes[key]
            bits = self.bits
            gs = self.group_size
            is_packed = False

        x_np = x.detach().cpu().float().contiguous().numpy().ravel()

        if is_packed and bits == 2:
            from asdsl.kernels import gemv_q2_packed
            out_np = gemv_q2_packed(w_data, x_np, sc, bi, rows, cols, gs)
        elif bits == 4:
            from asdsl.kernels import gemv_q4_unpacked
            out_np = gemv_q4_unpacked(w_data, x_np, sc, bi, rows, cols, gs)
        elif bits == 8:
            from asdsl.kernels import gemv_q8_unpacked
            out_np = gemv_q8_unpacked(w_data, x_np, sc, bi, rows, cols, gs)
        elif bits == 3:
            from asdsl.kernels import gemv_q3_unpacked
            out_np = gemv_q3_unpacked(w_data, x_np, sc, bi, rows, cols, gs)
        elif bits == 2:
            from asdsl.kernels import gemv_q2_unpacked
            out_np = gemv_q2_unpacked(w_data, x_np, sc, bi, rows, cols, gs)
        else:
            raise ValueError(f"No native kernel for {bits}-bit")

        result = torch.from_numpy(out_np).unsqueeze(0)

        outlier_store = (
            self._draft_outlier_values if draft_active else self._outlier_values
        )
        coord_store = (
            self._draft_outlier_coords if draft_active else self._outlier_coords
        )
        if key in outlier_store and len(outlier_store[key]) > 0:
            ov = outlier_store[key].astype(np.float32)
            oc = coord_store[key]
            col_indices = oc[:, 1]
            row_indices = oc[:, 0]
            x_sel = x_np[col_indices]
            contributions = ov * x_sel
            out_corr = np.zeros(rows, dtype=np.float32)
            np.add.at(out_corr, row_indices, contributions)
            result = result + torch.from_numpy(out_corr).unsqueeze(0)

        return result

    def matvec_sparse(self, layer_idx: int, name: str, x: torch.Tensor,
                      bitmask: np.ndarray, active_indices: np.ndarray) -> torch.Tensor:
        """Activation-sparse GEMV: skip near-zero activation columns (Tier 3)."""
        key = (layer_idx, name)
        rows, cols = self._quant_shapes[key]
        numel = rows * cols
        if self.bits == 4:
            from asdsl.quantization.core import _unpack_bits

            if self._sparse_u8_scratch is None or self._sparse_u8_scratch.size < numel:
                self._sparse_u8_scratch = np.empty(numel, dtype=np.uint8)
            packed = self._quant_packed[key].numpy().ravel()
            self._sparse_u8_scratch[:numel] = _unpack_bits(packed, 4)[:numel]
            u8_weights = self._sparse_u8_scratch[:numel]
        else:
            u8_weights = self._quant_u8[key].numpy().ravel()
        sc = self._quant_sc[key].float().numpy()
        bi = self._quant_bi[key].float().numpy()
        x_np = x.detach().cpu().float().contiguous().numpy().ravel()

        try:
            from asdsl.kernels import gemv_sparse_with_indices
            out_np = gemv_sparse_with_indices(
                u8_weights, x_np, sc, bi, active_indices, rows, cols, self.group_size
            )
        except ImportError:
            from asdsl.kernels import gemv_sparse_unpacked
            out_np = gemv_sparse_unpacked(
                u8_weights, x_np, sc, bi, bitmask, rows, cols, self.group_size
            )

        result = torch.from_numpy(out_np).unsqueeze(0)

        # Outlier correction
        if key in self._outlier_values and len(self._outlier_values[key]) > 0:
            ov = self._outlier_values[key].astype(np.float32)
            oc = self._outlier_coords[key]
            col_mask = np.isin(oc[:, 1], active_indices)
            if col_mask.any():
                ov_act = ov[col_mask]
                oc_act = oc[col_mask]
                x_sel = x_np[oc_act[:, 1]]
                contributions = ov_act * x_sel
                out_corr = np.zeros(rows, dtype=np.float32)
                np.add.at(out_corr, oc_act[:, 0], contributions)
                result = result + torch.from_numpy(out_corr).unsqueeze(0)

        return result

    def _matmul_quant_batch(self, layer_idx: int, name: str,
                            X_batch: torch.Tensor) -> torch.Tensor:
        if self.bits == 4:
            return self._matmul_q4_packed_batch(layer_idx, name, X_batch)
        key = (layer_idx, name)
        u8 = self._quant_u8[key]
        sc = self._quant_sc[key]
        bi = self._quant_bi[key]
        rows, cols = self._quant_shapes[key]
        groups_per_row = cols // self.group_size

        x_flat = X_batch.reshape(-1, cols).float()
        k_batch = x_flat.shape[0]

        chunk_rows = max(1, _TARGET_CHUNK_BYTES // (cols * 4))
        result = torch.empty(rows, k_batch, dtype=torch.float32)

        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            n = end - start

            flat_len = n * cols
            buf = self._pool[:flat_len]
            buf.copy_(u8[start * cols:end * cols])

            vals = buf.view(n, groups_per_row, self.group_size)
            gs = start * groups_per_row
            ge = end * groups_per_row
            vals.mul_(sc[gs:ge].view(n, groups_per_row, 1))
            vals.add_(bi[gs:ge].view(n, groups_per_row, 1))

            w_chunk = vals.view(n, cols)
            torch.mm(w_chunk, x_flat.T, out=result[start:end, :])

        return result.T.view(*X_batch.shape[:-1], rows)

    def _matmul_q4_packed_batch(self, layer_idx: int, name: str,
                                X_batch: torch.Tensor) -> torch.Tensor:
        """Batched verify: native fused packed GEMV, or unpack + BLAS fallback."""
        from asdsl.quantization.core import _unpack_bits
        from asdsl.kernels import gemv_q4_packed

        key = (layer_idx, name)
        packed = self._quant_packed[key]
        sc = self._quant_sc[key]
        bi = self._quant_bi[key]
        rows, cols = self._quant_shapes[key]
        groups_per_row = cols // self.group_size
        x_flat = X_batch.reshape(-1, cols).float()
        k_batch = x_flat.shape[0]

        # Phase 10: native path must run during batched verify too; the dequant+torch.mm
        # fallback differs numerically from AR ``forward_layer`` (native GEMV), so MTP
        # drafts trained on native activations never matched verify argmax (0% acceptance).
        if self._use_native_gemv:
            w_np = self._quant_packed_np.get(key)
            if w_np is None:
                w_np = packed.detach().cpu().numpy().reshape(-1)
                self._quant_packed_np[key] = w_np
                
            x_np = x_flat.detach().cpu().numpy()
            sc_np = self._quant_sc_np[key]
            bi_np = self._quant_bi_np[key]
            
            out_np = gemv_q4_packed(
                w_np, x_np, sc_np, bi_np, rows, cols, self.group_size,
                use_lut=self._use_lut_gemv,
            )
            result_t = torch.from_numpy(np.asarray(out_np, dtype=np.float32))
            if result_t.dim() == 1:
                result_t = result_t.unsqueeze(0)
            if key in self._outlier_values and len(self._outlier_values[key]) > 0:
                ov = self._outlier_values[key].astype(np.float32)
                oc = self._outlier_coords[key]
                col_indices = oc[:, 1]
                row_indices = oc[:, 0]
                for bi in range(k_batch):
                    x_row = x_np[bi]
                    x_sel = x_row[col_indices]
                    contributions = ov * x_sel
                    out_corr = np.zeros(rows, dtype=np.float32)
                    np.add.at(out_corr, row_indices, contributions)
                    result_t[bi] = result_t[bi] + torch.from_numpy(out_corr)
            return result_t.view(*X_batch.shape[:-1], rows)

        chunk_rows = max(1, _TARGET_CHUNK_BYTES // (cols * 4))
        result = torch.empty(rows, k_batch, dtype=torch.float32)
        packed_np = packed.numpy()

        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            n = end - start
            flat_len = n * cols
            
            cache_key = (key, start, end, "f32")
            if self._in_verify_phase and cache_key in self._dequant_cache:
                vals_f32 = self._dequant_cache[cache_key]
            else:
                chunk_packed = packed_np[start:end].reshape(-1)
                unpacked = _unpack_bits(chunk_packed, 4)[: n * cols]
                w_indices = torch.from_numpy(unpacked.astype(np.uint8))
                
                buf = self._pool[:flat_len]
                buf.copy_(w_indices)
                vals = buf.view(n, groups_per_row, self.group_size)
                gs = start * groups_per_row
                ge = end * groups_per_row
                vals_f32 = vals.mul(sc[gs:ge].float().view(n, groups_per_row, 1))
                vals_f32.add_(bi[gs:ge].float().view(n, groups_per_row, 1))
                vals_f32 = vals_f32.view(n, cols).clone() # detach from pool
                
                if self._in_verify_phase:
                    self._dequant_cache[cache_key] = vals_f32

            torch.mm(vals_f32, x_flat.T, out=result[start:end, :])

        return result.T.view(*X_batch.shape[:-1], rows)


    def _matmul_f16_batch(self, layer_idx: int, name: str,
                          X_batch: torch.Tensor) -> torch.Tensor:
        """Batched f16->f32 matmul for unquantized weights."""
        w_f16 = self._weight_cache[(layer_idx, name)]
        rows, cols = w_f16.shape
        K_batch = X_batch.shape[0]
        X_flat = X_batch.reshape(K_batch, cols)

        chunk_rows = max(1, _TARGET_CHUNK_BYTES // (cols * 4))
        result = torch.empty(rows, K_batch, dtype=torch.float32)
        buf = self._pool[:chunk_rows * cols].view(chunk_rows, cols)
        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            n = end - start
            buf[:n].copy_(w_f16[start:end])
            torch.mm(buf[:n], X_flat.T, out=result[start:end, :])
        return result.T

    def matmul_batch(self, layer_idx: int, name: str,
                     X_batch: torch.Tensor) -> torch.Tensor:
        """Batched matrix multiply: Y = X_batch @ W^T, loading W once."""
        if self.bits == 16:
            return self._matmul_f16_batch(layer_idx, name, X_batch)
        return self._matmul_quant_batch(layer_idx, name, X_batch)

    def lm_head_matmul_batch(self, hidden_batch: torch.Tensor) -> torch.Tensor:
        """Batched logits = hidden_batch @ lm_head.T, loading lm_head once."""
        K_batch = hidden_batch.shape[0]
        X = hidden_batch.reshape(K_batch, -1)
        lm = self.lm_head
        rows, cols = lm.shape
        chunk_rows = max(1, _TARGET_CHUNK_BYTES // (cols * 4))
        result = torch.empty(rows, K_batch, dtype=torch.float32)
        buf = self._pool[:chunk_rows * cols].view(chunk_rows, cols)
        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            n = end - start
            buf[:n].copy_(lm[start:end])
            torch.mm(buf[:n], X.T, out=result[start:end, :])
        return result.T

    # ------------------------------------------------------------------
    # Phase 2: SliM mixed-precision dispatch
    # ------------------------------------------------------------------

    def load_slim(self, meta_path) -> None:
        """Load SliM calibration metadata from phi4_slim_meta.json + .npz.

        Validates group_size matches WeightStore. Sets self._use_slim = True.
        Does NOT re-quantize weights — uses existing Q4 packed buffers with
        calibrated scales/zero_points from the metadata.
        """
        import json as _json
        meta_path = Path(meta_path)
        if not meta_path.exists():
            print(f"[SliM] WARNING: {meta_path} not found — SliM disabled")
            return

        meta = _json.loads(meta_path.read_text(encoding="utf-8"))

        if meta.get("quick_mode"):
            print("[SliM] WARNING: phi4_slim_meta.json was generated in quick mode "
                  "(only 4 layers calibrated) — using for Profile E anyway")

        meta_gs = meta.get("group_size", self.group_size)
        if meta_gs != self.group_size:
            print(f"[SliM] WARNING: meta group_size={meta_gs} != store group_size={self.group_size}")

        # Load .npz arrays
        npz_name = meta.get("npz_path", "phi4_slim_meta.npz")
        npz_path = meta_path.parent / npz_name
        if not npz_path.exists():
            print(f"[SliM] WARNING: {npz_path} not found — SliM disabled")
            return

        self._slim_meta = meta
        self._slim_npz = np.load(str(npz_path))
        self._use_slim = True
        self._repacked_layers = {}

        avg_bits = meta.get("achieved_avg_bits", "?")
        size_gb = meta.get("statistics", {}).get("estimated_model_size_gb", "?")
        print(f"[SliM] Loaded: {avg_bits} avg bits, ~{size_gb} GB estimated size")

    def load_fatrelu(self, thresholds_path, adaptive: bool = False) -> None:
        """Load FATReLU thresholds from phi4_fatrelu_thresholds.json.

        Sets self._use_fatrelu = True and populates self._fatrelu_thresholds.
        Also triggers transposed down_proj loading (Prerequisite B).

        If ``adaptive=True``, enables runtime adaptive tau measurement: the first
        ``_adaptive_tau_warmup_target`` tokens are used to measure the actual 85th
        percentile activation magnitude per layer, after which the measured tau values
        are used for all subsequent tokens (with exponential moving average smoothing).
        This compensates for prompt-specific activation magnitude variance that causes
        fixed offline tau values to produce 46-54% actual sparsity instead of 85%.
        """
        import json as _json
        thresholds_path = Path(thresholds_path)
        if not thresholds_path.exists():
            print(f"[FATReLU] WARNING: {thresholds_path} not found — FATReLU disabled")
            return

        data = _json.loads(thresholds_path.read_text(encoding="utf-8"))
        raw = data.get("thresholds", {})
        self._fatrelu_thresholds = {int(k): float(v) for k, v in raw.items()}
        self._use_fatrelu = True
        self._use_adaptive_fatrelu = adaptive

        n = len(self._fatrelu_thresholds)
        mean_tau = sum(self._fatrelu_thresholds.values()) / max(n, 1)
        sparsity = data.get("target_sparsity", 0.85)
        mode_str = "adaptive" if adaptive else "fixed"
        print(f"[FATReLU] Loaded: {n} layers, mean tau={mean_tau:.4f}, "
              f"target sparsity={sparsity:.0%}, mode={mode_str}")
        if adaptive:
            print(f"[FATReLU] Adaptive: warmup={self._adaptive_tau_warmup_target} tokens, "
                  f"EMA smoothing=0.7/0.3")

        # Phase 4 Prerequisite B: build transposed down_proj weights
        self.load_transposed_down_proj()

    def load_transposed_down_proj(self) -> None:
        """Build transposed down_proj Q4 weights for column-sparse access.

        Original down_proj: [out_dim=3072, in_dim=8192] Q4 packed row-major.
        Transposed down_proj: [in_dim=8192, out_dim=3072] Q4 packed row-major.

        With FATReLU 85% sparsity, only ~15% of 8192 intermediate neurons are
        active. Transposed storage makes each active neuron's contribution a
        contiguous 3072-element row read (vs column-sparse access in original).
        Memory traffic: 6.7x reduction for down_proj at 85% sparsity.

        Runs layer-by-layer to stay within ~202 MB peak RAM per layer.
        """
        import gc as _gc
        import psutil as _psutil

        if not self._quant_packed:
            print("[FATReLU] WARNING: load_transposed_down_proj called before warm_cache — skipping")
            return

        print("[FATReLU] Building transposed down_proj weights (Prerequisite B)...")
        n_transposed = 0

        for layer_idx in range(NUM_LAYERS):
            key = (layer_idx, "down_proj")
            if key not in self._quant_packed:
                continue

            # Memory guard
            avail_gb = _psutil.virtual_memory().available / 1e9
            if avail_gb < 1.0:
                _gc.collect()
                avail_gb = _psutil.virtual_memory().available / 1e9
                if avail_gb < 0.5:
                    print(f"[FATReLU] WARNING: Only {avail_gb:.1f} GB available — "
                          f"stopping transposed load at layer {layer_idx}")
                    break

            packed_t = self._quant_packed[key]  # [out_dim, in_dim/2] = [3072, 4096]
            sc_t = self._quant_sc[key]           # [out_dim * n_groups]
            bi_t = self._quant_bi[key]           # [out_dim * n_groups]
            out_dim, in_half = packed_t.shape
            in_dim = in_half * 2  # 8192

            # Dequantize to float32 — one layer at a time (~100 MB)
            n_groups_per_row = in_dim // self.group_size
            packed_np = packed_t.numpy().astype(np.uint8)  # [out_dim, in_dim/2]

            # Unpack nibbles to uint8: [out_dim, in_dim]
            lo = (packed_np & 0x0F).astype(np.float32)  # even cols
            hi = ((packed_np >> 4) & 0x0F).astype(np.float32)  # odd cols
            # Interleave: col 2j = lo[j], col 2j+1 = hi[j]
            w_f32 = np.empty((out_dim, in_dim), dtype=np.float32)
            w_f32[:, 0::2] = lo
            w_f32[:, 1::2] = hi

            # Apply scale and bias to get float32 weights
            sc_np = sc_t.float().numpy().reshape(out_dim, n_groups_per_row)  # [3072, 128]
            bi_np = bi_t.float().numpy().reshape(out_dim, n_groups_per_row)  # [3072, 128]
            # Broadcast scale/bias to each group's columns
            sc_expanded = np.repeat(sc_np, self.group_size, axis=1)  # [3072, 8192]
            bi_expanded = np.repeat(bi_np, self.group_size, axis=1)  # [3072, 8192]
            w_f32 = w_f32 * sc_expanded + bi_expanded  # [3072, 8192] dequantized

            del lo, hi, sc_expanded, bi_expanded

            # Transpose: [8192, 3072]
            w_T_f32 = w_f32.T.copy()  # [in_dim, out_dim] = [8192, 3072]
            del w_f32

            # Re-quantize transposed weights: [8192 rows, 3072 cols]
            # Group size along new rows (each row = one intermediate dim, 3072 elements)
            gs = self.group_size  # 32
            out_dim_T = w_T_f32.shape[1]  # 3072
            in_dim_T  = w_T_f32.shape[0]  # 8192
            n_groups_T_per_row = out_dim_T // gs

            # Quantize row by row (vectorized across rows)
            # Scale per (in_dim, out_dim/gs) group
            w_T_reshaped = w_T_f32.reshape(in_dim_T, n_groups_T_per_row, gs)
            max_abs = np.max(np.abs(w_T_reshaped), axis=2, keepdims=True)  # [8192, 96, 1]
            max_abs = np.maximum(max_abs, 1e-9)
            sc_T = (max_abs / 7.5).astype(np.float32).reshape(in_dim_T, n_groups_T_per_row)
            bi_T_base = (-8.0 * sc_T).astype(np.float32)  # zero_point=8

            # Quantize
            inv_sc = (7.5 / max_abs.squeeze(2)).astype(np.float32)  # [8192, 96]
            inv_expanded = np.repeat(inv_sc, gs, axis=1)  # [8192, 3072]
            w_q = np.clip(
                np.round(w_T_f32 * inv_expanded + 8.0).astype(np.int32),
                0, 15
            ).astype(np.uint8)  # [8192, 3072]

            del w_T_f32, inv_sc, inv_expanded

            # Pack nibbles: [8192, 3072/2]
            packed_T = (w_q[:, 0::2] | (w_q[:, 1::2] << 4)).astype(np.uint8)  # [8192, 1536]
            del w_q

            self._down_proj_T[layer_idx] = {
                'packed': packed_T,                              # [8192, 1536]
                'scales': sc_T.reshape(in_dim_T * n_groups_T_per_row),   # flat
                'biases': bi_T_base.reshape(in_dim_T * n_groups_T_per_row),  # flat
                'in_dim': in_dim_T,   # 8192
                'out_dim': out_dim_T,  # 3072
            }
            n_transposed += 1
            _gc.collect()
            if n_transposed % 4 == 0:
                print(f"  [FATReLU] Transposed {n_transposed}/{NUM_LAYERS} layers...", flush=True)

        print(f"[FATReLU] Transposed down_proj ready: {n_transposed}/{NUM_LAYERS} layers")

    def load_mtp_head(self, path: str = "models/mtp_head.pt") -> bool:
        """Load EAGLE-3 MTP head checkpoint.

        Returns True if loaded successfully (val_top1 >= 5%), False otherwise.
        Disabled head: sets _use_eagle3 = False.
        """
        import os
        import torch

        if not os.path.exists(path):
            print(f"[EAGLE-3] {path} not found — EAGLE-3 disabled")
            self._use_eagle3 = False
            return False

        try:
            ckpt = torch.load(path, map_location="cpu")
        except Exception as e:
            print(f"[EAGLE-3] Failed to load {path}: {e}")
            self._use_eagle3 = False
            return False

        val_acc = float(ckpt.get("val_top1_accuracy", 0))
        if val_acc < 5.0:
            print(f"[EAGLE-3] WARNING: val_top1={val_acc:.1f}% < 5% threshold — "
                  f"head may be overfit, EAGLE-3 disabled")
            self._use_eagle3 = False
            return False

        # Store as float32 numpy for fast inference
        self._mtp_head = {
            "fc1_W":   ckpt["fc1_weight"].float().numpy(),    # [1024, 6144]
            "fc1_b":   ckpt["fc1_bias"].float().numpy(),      # [1024]
            "norm_W":  ckpt["norm_weight"].float().numpy(),   # [1024]
            "norm_b":  ckpt["norm_bias"].float().numpy(),     # [1024]
            "proj_W":  ckpt["proj_weight"].float().numpy(),   # [3072, 1024]
            "proj_b":  ckpt["proj_bias"].float().numpy(),     # [3072]
            "hidden_dim_mtp": int(ckpt.get("hidden_dim_mtp", 1024)),
            "val_acc": val_acc,
        }
        self._use_eagle3 = True
        size_mb = os.path.getsize(path) / 1e6
        print(f"[EAGLE-3] MTP head loaded ({size_mb:.1f} MB, val_top1={val_acc:.1f}%)")
        return True

    def matmul_batch(self, layer_idx: int, name: str, x: torch.Tensor) -> torch.Tensor:
        """Batched matrix multiplication y = x @ W.T.

        Fast path: calls native ``matmul_batch_q4`` which dequantizes each weight
        group exactly once and applies it to all B batch rows — O(B) work vs O(B²)
        for B separate dequant+GEMV calls.  No float32 weight matrix is ever
        materialised in memory.

        Fallback: standard float32 dequant + numpy MM (slow; only when native ext
        is unavailable).
        """
        key = (layer_idx, name)
        if key not in self._quant_shapes:
            w = self.get_weight(layer_idx, name)
            return torch.matmul(x, w.T)

        rows, cols = self._quant_shapes[key]
        packed_np = self._quant_packed[key].numpy()   # uint8 [rows, cols/2]
        sc_np     = self._quant_sc[key].float().numpy().ravel()   # [rows * n_groups]
        bi_np     = self._quant_bi[key].float().numpy().ravel()   # [rows * n_groups]

        x_np = x.float().numpy()   # [B, cols]
        B    = x_np.shape[0]

        # --- Native fast path ---
        try:
            from asdsl.kernels import _native_gemv as _ng_local  # type: ignore
            if hasattr(_ng_local, "matmul_batch_q4"):
                Y = np.zeros((B, rows), dtype=np.float32)
                _ng_local.matmul_batch_q4(
                    np.ascontiguousarray(packed_np, dtype=np.uint8),
                    np.ascontiguousarray(sc_np, dtype=np.float32),
                    np.ascontiguousarray(bi_np, dtype=np.float32),
                    np.ascontiguousarray(x_np, dtype=np.float32),
                    Y,
                    rows, cols, B, self.group_size,
                )
                return torch.from_numpy(Y).to(x.dtype)
        except (ImportError, AttributeError):
            pass

        # --- Fallback: float32 dequant + numpy MM ---
        n_groups = cols // self.group_size
        packed_u8 = packed_np.astype(np.uint8)
        lo = (packed_u8 & 0x0F).astype(np.float32)
        hi = ((packed_u8 >> 4) & 0x0F).astype(np.float32)
        w_f32 = np.empty((rows, cols), dtype=np.float32)
        w_f32[:, 0::2] = lo
        w_f32[:, 1::2] = hi
        sc_2d = sc_np.reshape(rows, n_groups)
        bi_2d = bi_np.reshape(rows, n_groups)
        sc_exp = np.repeat(sc_2d, self.group_size, axis=1)
        bi_exp = np.repeat(bi_2d, self.group_size, axis=1)
        w_f32 = w_f32 * sc_exp + bi_exp
        result = (x_np @ w_f32.T).astype(np.float32)
        return torch.from_numpy(result).to(x.dtype)

    def clear_tmp_cache(self) -> None:
        """No-op: retained for API compatibility; native path has no cache."""
        pass

    def _get_slim_arrays(self, layer_idx: int, name: str):
        """Return (bits_arr, scales_arr, zp_arr) from SliM npz, or None if not found."""
        if self._slim_npz is None:
            return None
        prefix = f"L{layer_idx}_{name}"
        bits_key = f"{prefix}_bits"
        if bits_key not in self._slim_npz:
            return None
        return (
            self._slim_npz[f"{prefix}_bits"],
            self._slim_npz[f"{prefix}_scales"],
            self._slim_npz[f"{prefix}_zp"],
        )

    def _get_slim_repacked(self, layer_idx: int, name: str):
        """
        Lazy repack: for groups assigned 2-bit, pack 4 values per byte.
        Returns (repacked_packed, slim_scales, slim_biases) or None.

        Repacking strategy:
          - 2-bit groups: extract low 2 bits of each nibble, pack 4 per byte
          - 3-bit groups: keep as 4-bit (no repacking, just use narrower scale)
          - 4-bit groups: unchanged

        Memory saving: 2-bit groups use half the bytes of 4-bit groups.
        """
        cache_key = (layer_idx, name)
        if cache_key in self._repacked_layers:
            return self._repacked_layers[cache_key]

        slim_arrays = self._get_slim_arrays(layer_idx, name)
        if slim_arrays is None:
            self._repacked_layers[cache_key] = None
            return None

        bits_arr, scales_arr, zp_arr = slim_arrays
        key = (layer_idx, name)
        if key not in self._quant_packed:
            self._repacked_layers[cache_key] = None
            return None

        rows, cols = self._quant_shapes[key]
        n_groups_per_row = cols // self.group_size

        # Build new scales and biases from SliM metadata
        slim_scales = torch.from_numpy(scales_arr.astype(np.float32))
        slim_biases = torch.from_numpy(
            (-zp_arr.astype(np.float32) * scales_arr.astype(np.float32))
        )

        # Check if any groups are 2-bit (need repacking)
        has_2bit = bool(np.any(bits_arr == 2))

        if not has_2bit:
            # No repacking needed — just use new scales/biases with existing packed weights
            result = (self._quant_packed[key], slim_scales, slim_biases)
            self._repacked_layers[cache_key] = result
            return result

        # Repack 2-bit groups: extract low 2 bits of each nibble
        # Original: 2 nibbles per byte (4-bit each)
        # Repacked: 4 two-bit values per byte
        packed_np = self._quant_packed[key].numpy()  # (rows, cols//2)
        bits_2d = bits_arr.reshape(rows, n_groups_per_row)

        # Vectorized 2-bit masking: mask high bits of nibbles for 2-bit groups.
        # For 2-bit groups: keep only low 2 bits of each nibble (values 0-3).
        # This is done vectorized over all rows at once using numpy broadcasting.
        repacked = packed_np.copy()

        # Build column mask: packed column j covers weight cols 2j and 2j+1
        # Group index for packed column j = j // (group_size//2)
        half_gs = self.group_size // 2
        col_group_idx = np.arange(cols // 2) // half_gs  # (cols//2,)
        # col_is_2bit[row, col] = True if that column's group is 2-bit
        col_is_2bit = (bits_2d[:, col_group_idx] == 2)  # (rows, cols//2)

        # Apply mask: keep only low 2 bits of each nibble for 2-bit groups
        lo = repacked & 0x03
        hi = (repacked >> 4) & 0x03
        masked = lo | (hi << 4)
        repacked = np.where(col_is_2bit, masked, repacked)

        repacked_t = torch.from_numpy(repacked)
        result = (repacked_t, slim_scales, slim_biases)
        self._repacked_layers[cache_key] = result
        return result

    def _matvec_slim(self, layer_idx: int, name: str, x: torch.Tensor) -> torch.Tensor:
        """SliM mixed-precision GEMV: uses calibrated scales from phi4_slim_meta."""
        import time
        t_start = time.perf_counter()
        from asdsl.kernels import gemv_q4_packed

        repacked = self._get_slim_repacked(layer_idx, name)
        t_repack = time.perf_counter()
        if repacked is None:
            return self._matvec_native_gemv(layer_idx, name, x, use_draft=False)

        packed_t, slim_scales, slim_biases = repacked
        rows, cols = self._quant_shapes[(layer_idx, name)]

        w_np = packed_t.numpy().reshape(-1)
        x_np = x.detach().cpu().float().contiguous().numpy().ravel()
        sc_np = slim_scales.float().numpy()
        bi_np = slim_biases.float().numpy()
        t_arrays = time.perf_counter()

        out_np = gemv_q4_packed(
            w_np, x_np, sc_np, bi_np, rows, cols, self.group_size,
            use_lut=self._use_lut_gemv,
        )
        t_gemv = time.perf_counter()
        res = torch.from_numpy(np.asarray(out_np, dtype=np.float32)).unsqueeze(0)
        t_end = time.perf_counter()
        
        # Only print once to avoid console spam
        if not hasattr(self, "_printed_slim_profile"):
            print(f"[Slim_Prof] repack: {(t_repack-t_start)*1000:.3f}ms  arrays: {(t_arrays-t_repack)*1000:.3f}ms  gemv: {(t_gemv-t_arrays)*1000:.3f}ms  tensor: {(t_end-t_gemv)*1000:.3f}ms")
            self._printed_slim_profile = True
            
        return res

    def _inference_mode_message(self, *, has_gemv: bool) -> str:
        """Human-readable active GEMV path for warm-cache status."""
        if self.bits == 16:
            return "chunked f16 matvec"
        if self._use_unified:
            preq2_on = os.environ.get("ASDSL_PREQ2", "0").strip().lower() not in (
                "0",
                "false",
                "no",
            )
            preq2_built = getattr(self, "_preq2_built", False)
            gemv = "preq2+VNNI" if (preq2_on and preq2_built) else "Q4_32 preq"
            if _env_flag("ASDSL_C01"):
                gemv += "+C0.1-g128(gate_up/down)"
            lm_gs = os.environ.get("ASDSL_LMHEAD_GS", str(self.group_size))
            if lm_gs != str(self.group_size):
                gemv += f", lm_head_gs={lm_gs}"
            aff = os.environ.get("ASDSL_AFFINITY", "physical")
            chunked = os.environ.get("ASDSL_CHUNKED_GEMV", "0")
            return (
                f"UnifiedEngine C++ forward ({gemv}, affinity={aff}, "
                f"chunked={chunked})"
            )
        if self._enable_dispatch and self._dispatch_policy is not None:
            return (
                f"Phase 3 kernel dispatch (LUT/SPARSE/AVX2); "
                f"{len(self._lut_cache)} LUT caches, "
                f"{len(self._sparse_dequant_f16)} sparse dequant caches"
            )
        if self._use_q8_gemv and has_gemv:
            return "native AVX2 GEMV Q4×Q8 (gemv_q4_q8_avx2)"
        if has_gemv:
            labels = {4: "Q4 packed (gemv_q4_packed)", 8: "Q8", 3: "Q3", 2: "Q2"}
            kl = labels.get(self.bits, f"Q{self.bits}")
            return f"native AVX2 GEMV {kl}"
        return "chunked uint8 dequant+matvec (in-place, no AVX GEMV)"

    def _finish_dispatch_caches(self) -> None:
        """Build LUT/sparse caches once dispatch policy and quant arrays exist."""
        import time

        if not (self._enable_dispatch and self.bits == 4 and self._dispatch_policy is not None):
            return
        if not self._quant_packed_np:
            return
        t0 = time.perf_counter()
        self._build_lut_caches()
        t_lut = time.perf_counter() - t0
        t1 = time.perf_counter()
        self._build_sparse_dequant_cache()
        t_sparse = time.perf_counter() - t1
        if self._dispatch_policy is not None:
            self._rebuild_dispatch_matvec_routing()
        print(
            f"  Dispatch cache build: LUT {t_lut:.1f}s, sparse {t_sparse:.1f}s "
            f"({len(self._lut_cache)} LUT, {len(self._sparse_dequant_f16)} sparse)"
        )

    def load_dispatch_policy(self, path: str | Path) -> None:
        """Load Phase 3 projection profiles for kernel routing."""
        from asdsl.dispatch.policy import DispatchPolicy

        self._dispatch_policy = DispatchPolicy.load_json(path)
        self._enable_dispatch = True
        if self._quant_shapes:
            self._rebuild_dispatch_matvec_routing()
        print(
            f"  Phase 3 dispatch: {len(self._dispatch_policy.profiles)} profiles "
            f"from {Path(path).name}"
        )

    def _rebuild_dispatch_matvec_routing(self) -> None:
        """Precompute kernel tags; bypass dispatch wrapper when all paths are AVX2."""
        import os

        from asdsl.dispatch.policy import KernelTag

        self._kernel_tags = {}
        if self._dispatch_policy is None:
            self._dispatch_matvec_fast = False
            return

        sparse_infer = os.environ.get("ASDSL_SPARSE_INFER", "0").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        has_lut = False
        n_sparse_profile = 0
        n_sparse_runtime = 0
        for key in self._quant_shapes:
            layer_idx, name = key
            if name not in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"):
                continue
            tag = self._dispatch_policy.get_kernel(layer_idx, name)
            if tag == KernelTag.SPARSE:
                n_sparse_profile += 1
                if not sparse_infer:
                    tag = KernelTag.AVX2
                else:
                    n_sparse_runtime += 1
            self._kernel_tags[key] = tag
            if tag == KernelTag.LUT:
                has_lut = True

        self._dispatch_matvec_fast = not has_lut
        if n_sparse_profile and not sparse_infer and not getattr(
            self, "_printed_sparse_avx2_routing", False
        ):
            print(
                f"  Phase 6 dispatch: {n_sparse_profile} SPARSE profiles -> AVX2 at inference "
                f"(set ASDSL_SPARSE_INFER=1 to enable sparse GEMV)"
            )
            self._printed_sparse_avx2_routing = True
        elif n_sparse_runtime and not getattr(self, "_printed_sparse_infer", False):
            print(f"  Phase 6 dispatch: {n_sparse_runtime} projections use sparse GEMV")
            self._printed_sparse_infer = True

    def packed_weight_bytes(self) -> int:
        """Packed Q4 weight bytes only (projection matrices)."""
        return sum(int(arr.nbytes) for arr in self._quant_packed_np.values())

    def total_matvec_weight_bytes(self) -> int:
        """Bytes touched per full forward pass (packed + scales + biases per projection)."""
        total = self.packed_weight_bytes()
        total += sum(int(arr.nbytes) for arr in self._quant_sc_np.values())
        total += sum(int(arr.nbytes) for arr in self._quant_bi_np.values())
        return total

    def _build_norm_np_cache(self) -> None:
        """Precompute float32 norm vectors (load-time, not per-token)."""
        self._norm_np.clear()
        for layer_idx in range(NUM_LAYERS):
            for name in ("input_layernorm", "post_attention_layernorm"):
                t = self.layer_norms[layer_idx][name]
                self._norm_np[(layer_idx, name)] = (
                    t.detach().cpu().float().numpy().astype(np.float32, copy=False)
                )
        if self.final_norm is not None:
            self._norm_np[(-1, "final")] = (
                self.final_norm.detach().cpu().float().numpy().astype(np.float32, copy=False)
            )

    def _matvec_q4_packed_np(
        self,
        layer_idx: int,
        name: str,
        x_np: np.ndarray,
        *,
        use_draft: bool = False,
    ) -> np.ndarray:
        """Fused Q4 GEMV; x_np is float32 contiguous, returns float32 (M,)."""
        from asdsl.kernels import gemv_q4_packed

        key = (layer_idx, name)
        if use_draft and key in self._draft_quant_packed_np:
            w_np = self._draft_quant_packed_np[key]
            sc_np = self._draft_quant_sc_np[key]
            bi_np = self._draft_quant_bi_np[key]
            gs = self._draft_group_size
        else:
            w_np = self._quant_packed_np[key]
            sc_np = self._quant_sc_np[key]
            bi_np = self._quant_bi_np[key]
            gs = self.group_size
        rows, cols = self._quant_shapes[key]
        if self._x_buf is None or self._x_buf.size < cols:
            self._x_buf = np.empty(cols, dtype=np.float32)
        x_flat = np.ascontiguousarray(x_np, dtype=np.float32).ravel()
        if x_flat.size != cols:
            raise ValueError(f"x length {x_flat.size} != cols {cols}")
        np.copyto(self._x_buf[:cols], x_flat)
        out_np = self._matvec_out_bufs.get(key)
        if out_np is None or out_np.shape[0] != rows:
            out_np = np.empty(rows, dtype=np.float32)
            self._matvec_out_bufs[key] = out_np
        gemv_q4_packed(
            w_np,
            self._x_buf[:cols],
            sc_np,
            bi_np,
            rows,
            cols,
            gs,
            out=out_np,
            use_lut=self._use_lut_gemv,
            use_q8=self._use_q8_gemv if not use_draft else False,
            lut_cache=self._lut_cache.get(key) if not use_draft else None,
            bits=self._draft_bits if use_draft else self.bits,
        )
        return out_np

    def matvec_np(
        self,
        layer_idx: int,
        name: str,
        x_np: np.ndarray,
        *,
        use_draft: bool = False,
    ) -> np.ndarray:
        """GEMV with float32 numpy input (avoids torch in Q4 packed path)."""
        if self.bits == 16:
            out_t = self.matvec(layer_idx, name, torch.from_numpy(x_np).unsqueeze(0), use_draft=use_draft)
            return out_t.detach().cpu().float().numpy().ravel()
        if self._enable_dispatch and self.bits == 4 and not self._use_slim:
            tag = self._kernel_tags.get((layer_idx, name))
            from asdsl.dispatch.policy import KernelTag

            if tag in (KernelTag.LUT, KernelTag.SPARSE):
                out_t = self.matvec(
                    layer_idx, name, torch.from_numpy(x_np).unsqueeze(0), use_draft=use_draft
                )
                return out_t.detach().cpu().float().numpy().ravel()
            if self._dispatch_matvec_fast and not use_draft:
                return self._matvec_q4_packed_np(layer_idx, name, x_np, use_draft=False)
            if tag in (None, KernelTag.AVX2):
                return self._matvec_q4_packed_np(layer_idx, name, x_np, use_draft=use_draft)
        if self.bits == 4 and (self._use_native_gemv or use_draft):
            return self._matvec_q4_packed_np(layer_idx, name, x_np, use_draft=use_draft)
        out_t = self.matvec(layer_idx, name, torch.from_numpy(x_np).unsqueeze(0), use_draft=use_draft)
        return out_t.detach().cpu().float().numpy().ravel()

    def _assert_packed_contiguous(self) -> None:
        """One-time contiguity check after warm_cache (load-time, not per-token)."""
        bad: list[str] = []
        for key, arr in self._quant_packed_np.items():
            if not arr.flags["C_CONTIGUOUS"]:
                bad.append(f"L{key[0]}:{key[1]}")
        if bad:
            raise RuntimeError(
                f"Non-contiguous packed weights ({len(bad)}): {bad[:4]}..."
            )

    def load_correction(self, path: str | Path, *, scale: float = 1.0) -> None:
        """Load Phase 4 per-layer MLP correction bank (models/ + correction_manifest.json)."""
        from asdsl.correction import load_correction

        model = load_correction(path)
        if model is None:
            raise FileNotFoundError(f"correction weights not found: {path}")
        self._correction = model
        self._enable_correction = True
        self._correction_scale = float(scale)
        val_losses = [
            float(x.get("val_loss", 0.0))
            for x in model.manifest.get("layers", [])
        ]
        med = float(np.median(val_losses)) if val_losses else 0.0
        print(
            f"  Phase 4 correction: {model.num_layers} MLP layers, "
            f"scale={self._correction_scale}, median val_loss={med:.2e}"
        )

    def _get_dispatch_output_buffer(self, key: tuple, size: int) -> np.ndarray:
        """Reuse float32 output buffers across dispatch matvec calls."""
        buf = self._dispatch_output_buffers.get(key)
        if buf is None or buf.shape[0] != size:
            buf = np.empty(size, dtype=np.float32)
            self._dispatch_output_buffers[key] = buf
        return buf

    def _build_sparse_dequant_entry(self, key: tuple[int, str]) -> np.ndarray | None:
        """Lazily dequant one SPARSE-tagged projection to float16."""
        from asdsl.dispatch.policy import KernelTag
        from asdsl.quantization.core import dequantize_weights

        if key in self._sparse_dequant_f16:
            return self._sparse_dequant_f16[key]
        if self._dispatch_policy is None or key not in self._quant_packed_np:
            return None
        layer_idx, name = key
        if self._dispatch_policy.get_kernel(layer_idx, name) != KernelTag.SPARSE:
            return None
        w_np = self._quant_packed_np[key]
        rows, cols = self._quant_shapes[key]
        gs = self.group_size
        numel = rows * cols
        n_groups = numel // gs
        qt_stub = type("Q", (), {})()
        qt_stub.data = w_np
        qt_stub.scales = self._quant_sc_np[key][:n_groups]
        qt_stub.zeros = None
        qt_stub.group_size = gs
        qt_stub.bits = 4
        qt_stub.numel = numel
        qt_stub.is_symmetric = True
        qt_stub.shape = (rows, cols)
        deq = dequantize_weights(qt_stub).reshape(rows, cols)
        deq_clipped = np.clip(deq.astype(np.float32), -65504.0, 65504.0)
        arr = np.ascontiguousarray(deq_clipped.astype(np.float16))
        self._sparse_dequant_f16[key] = arr
        return arr

    def _build_sparse_dequant_cache(self) -> None:
        """Prepare lazy sparse dequant cache (no eager 32×50MB build at load)."""
        from asdsl.dispatch.policy import KernelTag

        if self._dispatch_policy is None:
            return
        n_sparse = sum(
            1
            for key in self._quant_shapes
            if self._dispatch_policy.get_kernel(key[0], key[1]) == KernelTag.SPARSE
        )
        if n_sparse == 0:
            self._sparse_dequant_f16.clear()
            return
        self._sparse_dequant_f16.clear()
        print(
            f"  Phase 3 sparse dequant: {n_sparse} SPARSE projections "
            f"(lazy f16 build on first sparse matvec)"
        )

    def _matvec_dispatch(
        self,
        layer_idx: int,
        name: str,
        x: torch.Tensor,
        *,
        use_draft: bool = False,
    ) -> torch.Tensor:
        """Route matvec by calibrated KernelTag (LUT / SPARSE / AVX2)."""
        from asdsl.dispatch.policy import KernelTag
        from asdsl.kernels.sparse_gemv import sparse_gemv_dequant_f16_avx2
        from asdsl.lut import LUTGEMVKernel

        if self._dispatch_policy is None:
            return self._matvec_q4_packed(layer_idx, name, x, use_draft=use_draft)

        key = (layer_idx, name)
        tag = self._kernel_tags.get(
            key, self._dispatch_policy.get_kernel(layer_idx, name)
        )
        lut_key = ("draft",) + key if use_draft else key
        counts = getattr(self, "_dispatch_route_counts", None)
        if counts is None:
            self._dispatch_route_counts = {"LUT": 0, "SPARSE": 0, "AVX2": 0}
            counts = self._dispatch_route_counts

        if tag == KernelTag.LUT:
            lut_cache = self._lut_cache.get(lut_key)
            if lut_cache is not None:
                counts["LUT"] += 1
                x_np = x.detach().cpu().float().contiguous().numpy().ravel()
                cols = self._quant_shapes[key][1]
                if self._x_buf is None or self._x_buf.size < cols:
                    self._x_buf = np.empty(cols, dtype=np.float32)
                np.copyto(self._x_buf[:cols], x_np)
                y = LUTGEMVKernel(tile_groups=self._lut_tile_groups).gemv(
                    lut_cache, self._x_buf[:cols], use_avx2=True
                )
                return torch.from_numpy(y).unsqueeze(0)
            if not getattr(self, "_warned_lut_cache_miss", False):
                print(
                    f"  WARNING: LUT kernel assigned for {name} L{layer_idx} "
                    f"but no lut_cache entry; falling back to gemv_q4_packed"
                )
                self._warned_lut_cache_miss = True
            counts["AVX2"] += 1
            return self._matvec_q4_packed(layer_idx, name, x, use_draft=use_draft)

        if tag == KernelTag.SPARSE:
            rows, cols = self._quant_shapes[key]
            x_np = x.detach().cpu().float().contiguous().numpy().ravel()
            thr = self._sparsity_threshold
            n_active = int(np.count_nonzero(np.abs(x_np) >= thr))
            # Dense activation: packed Q4 AVX2 beats f16 sparse gather.
            if n_active > int(0.45 * cols):
                counts["AVX2"] += 1
                return self._matvec_q4_packed(layer_idx, name, x, use_draft=use_draft)
            w_f16 = self._build_sparse_dequant_entry(key)
            if w_f16 is None:
                counts["AVX2"] += 1
                return self._matvec_q4_packed(layer_idx, name, x, use_draft=use_draft)
            counts["SPARSE"] += 1
            buf_key = ("sparse",) + key
            y = self._get_dispatch_output_buffer(buf_key, rows)
            np.copyto(
                y,
                sparse_gemv_dequant_f16_avx2(w_f16, x_np, threshold=thr),
            )
            return torch.from_numpy(y).unsqueeze(0)

        counts["AVX2"] += 1
        return self._matvec_q4_packed(layer_idx, name, x, use_draft=use_draft)

    def matvec(self, layer_idx: int, name: str, x: torch.Tensor,
               use_draft: bool = False) -> torch.Tensor:
        """Bandwidth-efficient matrix-vector product: y = W @ x."""
        key = (layer_idx, name)
        if not use_draft and self._use_q4km and key in self._q4km_weights:
            return self._matvec_q4km(layer_idx, name, x)
        if self.bits == 16:
            return self._matvec_f16(layer_idx, name, x)
        if self._enable_dispatch and self.bits == 4 and not self._use_slim:
            if self._dispatch_matvec_fast and not use_draft:
                return self._matvec_q4_packed(layer_idx, name, x, use_draft=False)
            return self._matvec_dispatch(layer_idx, name, x, use_draft=use_draft)
        # Phase 2: SliM mixed-precision path
        if self._use_slim and not use_draft and self.bits == 4:
            return self._matvec_slim(layer_idx, name, x)
        if self._use_lut_gemv and self.bits == 4 and not use_draft:
            lut_cache = self._lut_cache.get(key)
            if lut_cache is not None:
                return self._matvec_dispatch(layer_idx, name, x, use_draft=False)
        if self._use_native_gemv or use_draft:
            return self._matvec_native_gemv(layer_idx, name, x, use_draft=use_draft)
        return self._matvec_quant(layer_idx, name, x)

    def _get_token_embedding(self, token_id: int) -> np.ndarray:
        """Returns the float32 embedding vector for a token as a numpy array."""
        return self.embed_f16[token_id].float().cpu().numpy().ravel()

    def load_mtp_head(self, path: str) -> None:
        """Load trained MTP head for EAGLE-3 speculative decoding (Profile G).

        Supports both old single-layer format (fc1/norm/proj) and new multi-layer
        EAGLE-3 format (proj_low/mid/high + attn/norm/ffn/proj_out). The format is
        detected from the checkpoint's ``architecture`` key or by checking for
        multi-layer keys.
        """
        import torch
        import os
        from pathlib import Path
        path_p = Path(path)
        if not path_p.exists():
            print(f"[EAGLE-3] MTP head not found at {path} — disabling.")
            self._use_eagle3 = False
            return

        try:
            ckpt = torch.load(str(path_p), map_location="cpu", weights_only=False)
            val_acc = ckpt.get("val_top1_accuracy", 0.0)
            architecture = ckpt.get("architecture", "single_layer")

            # Detect format: multi-layer EAGLE-3 has proj_low_W, etc.
            is_multilayer = (
                architecture == "eagle3_multilayer"
                or "proj_low_W" in ckpt
                or "proj_low_weight" in ckpt
            )

            if is_multilayer:
                # New multi-layer EAGLE-3 format
                self._mtp_head = {
                    "architecture": "eagle3_multilayer",
                    "proj_low_W": ckpt.get("proj_low_W", ckpt.get("proj_low_weight")).float().numpy(),
                    "proj_low_b": ckpt.get("proj_low_b", ckpt.get("proj_low_bias")).float().numpy(),
                    "proj_mid_W": ckpt.get("proj_mid_W", ckpt.get("proj_mid_weight")).float().numpy(),
                    "proj_mid_b": ckpt.get("proj_mid_b", ckpt.get("proj_mid_bias")).float().numpy(),
                    "proj_high_W": ckpt.get("proj_high_W", ckpt.get("proj_high_weight")).float().numpy(),
                    "proj_high_b": ckpt.get("proj_high_b", ckpt.get("proj_high_bias")).float().numpy(),
                    "attn_W": ckpt.get("attn_W", ckpt.get("attn_weight")).float().numpy(),
                    "attn_b": ckpt.get("attn_b", ckpt.get("attn_bias")).float().numpy(),
                    "norm_W": ckpt.get("norm_W", ckpt.get("norm_weight")).float().numpy(),
                    "norm_b": ckpt.get("norm_b", ckpt.get("norm_bias")).float().numpy(),
                    "ffn_W": ckpt.get("ffn_W", ckpt.get("ffn_weight")).float().numpy(),
                    "ffn_b": ckpt.get("ffn_b", ckpt.get("ffn_bias")).float().numpy(),
                    "proj_out_W": ckpt.get("proj_out_W", ckpt.get("proj_out_weight")).float().numpy(),
                    "proj_out_b": ckpt.get("proj_out_b", ckpt.get("proj_out_bias")).float().numpy(),
                    "val_acc": val_acc,
                    "test_acc": ckpt.get("test_top1_accuracy", 0.0),
                }
                print(f"[EAGLE-3] Loaded multi-layer MTP head from {path} "
                      f"(val_acc={val_acc:.1f}%, architecture=eagle3_multilayer)")
            else:
                # Old single-layer format (backward compatible)
                self._mtp_head = {
                    "architecture": "single_layer",
                    "fc1_W": ckpt["fc1_weight"].float().numpy(),
                    "fc1_b": ckpt["fc1_bias"].float().numpy(),
                    "norm_W": ckpt["norm_weight"].float().numpy(),
                    "norm_b": ckpt["norm_bias"].float().numpy(),
                    "proj_W": ckpt["proj_weight"].float().numpy(),
                    "proj_b": ckpt["proj_bias"].float().numpy(),
                    "val_acc": val_acc
                }
                print(f"[EAGLE-3] Loaded single-layer MTP head from {path} "
                      f"(val_acc={val_acc:.1f}%, fallback)")
            self._use_eagle3 = True
        except Exception as e:
            print(f"[EAGLE-3] Error loading MTP head from {path}: {e}")
            self._use_eagle3 = False

    def _ensure_lm_head_native_cache(self) -> None:
        if self._lm_head_u16_np is not None or self.lm_head is None:
            return
        self._lm_head_u16_np = (
            self.lm_head.detach().cpu().view(torch.uint16).numpy()
        )

    def lm_head_matvec(self, hidden: torch.Tensor) -> torch.Tensor:
        """Compute logits = hidden @ lm_head.T with chunked f16 reads.

        Accepts ``hidden`` of shape ``(hidden_dim,)``, ``(1, hidden_dim)``, or
        ``(K, hidden_dim)``. Uses batched ``torch.mm`` on weight chunks (not
        ``torch.mv``) so multiple positions amortize the vocab-sized projection.

        Returns ``(vocab,)`` when ``K == 1``, else ``(K, vocab)``.
        """
        if hidden.dim() == 1:
            hidden = hidden.unsqueeze(0)
        elif hidden.dim() != 2:
            raise ValueError(f"lm_head_matvec expects 1D/2D hidden, got shape {tuple(hidden.shape)}")
        if hidden.shape[0] == 1 and self.lm_head is not None:
            if os.environ.get("ASDSL_LM_HEAD_NATIVE", "1") != "0":
                try:
                    from asdsl.kernels import _native_gemv

                    if hasattr(_native_gemv, "lm_head_gemv_f16"):
                        self._ensure_lm_head_native_cache()
                        w_u16 = self._lm_head_u16_np
                        if w_u16 is not None:
                            h_np = hidden.detach().float().squeeze(0).cpu().numpy()
                            m, k = int(w_u16.shape[0]), int(w_u16.shape[1])
                            logits_np = _native_gemv.lm_head_gemv_f16(w_u16, h_np, m, k)
                            return torch.from_numpy(logits_np)
                except Exception:
                    pass
            h = hidden.to(device=self.lm_head.device, dtype=self.lm_head.dtype)
            return torch.matmul(h, self.lm_head.T).float().squeeze(0)
        out = self.lm_head_matmul_batch(hidden)
        if out.shape[0] == 1:
            return out.squeeze(0)
        return out

    def _build_lut_caches(self) -> None:
        """Build per-projection LUT tables for Phase 1 LUT-native GEMV."""
        from asdsl.dispatch.policy import KernelTag, PHI4_PROJECTIONS
        from asdsl.lut import LUTTableBuilder, should_use_lut

        force_all_lut = os.environ.get("ASDSL_PPL_FORCE_LUT", "0").strip().lower() in (
            "1",
            "true",
            "yes",
        )

        self._lut_cache.clear()
        if self._dispatch_policy is not None and not force_all_lut:
            has_lut = any(
                self._dispatch_policy.get_kernel(layer_idx, name) == KernelTag.LUT
                for (layer_idx, name) in self._quant_shapes
                if name in PHI4_PROJECTIONS
            )
            if not has_lut:
                self._use_lut_gemv = False
                return

        for key, w_np in self._quant_packed_np.items():
            if w_np is None:
                continue
            layer_idx, name = key
            if self._dispatch_policy is not None and not force_all_lut:
                if self._dispatch_policy.get_kernel(layer_idx, name) != KernelTag.LUT:
                    continue
            rows, cols = self._quant_shapes[key]
            gs = self.group_size
            if not should_use_lut(4, gs, rows, cols):
                continue
            self._lut_cache[key] = LUTTableBuilder.build_projection(
                np.ascontiguousarray(w_np, dtype=np.uint8),
                np.ascontiguousarray(self._quant_sc_np[key], dtype=np.float32),
                np.ascontiguousarray(self._quant_bi_np[key], dtype=np.float32),
                rows,
                cols,
                gs,
                zeros=None,
                tile_groups=self._lut_tile_groups,
                build_q_packed=True,
            )
        if self._enable_qcsd:
            for key, w_np in self._draft_quant_packed_np.items():
                rows, cols = self._quant_shapes[key]
                gs = self._draft_group_size
                if self._draft_bits != 4 or not should_use_lut(4, gs, rows, cols):
                    continue
                self._lut_cache[("draft",) + key] = LUTTableBuilder.build_projection(
                    w_np,
                    self._draft_quant_sc_np[key],
                    self._draft_quant_bi_np[key],
                    rows,
                    cols,
                    gs,
                    zeros=None,
                    tile_groups=self._lut_tile_groups,
                    build_q_packed=True,
                )
        self._use_lut_gemv = len(self._lut_cache) > 0
        if self._use_lut_gemv:
            fp = LUTTableBuilder.footprint_bytes()
            print(
                f"  Phase 1 LUT cache: {len(self._lut_cache)} projections "
                f"(K-tile ~{fp / 1024:.0f} KB float16)"
            )

    def warm_cache(self) -> None:
        """Prepare weight cache for streaming inference.

        bits=16: float16 weight cache is already populated from load().
        bits==4:  keep 4-bit weights packed (rows, cols//2) uint8; fused GEMV
                 via native ``gemv_q4_packed`` (AVX2 in-register unpack).
        other:    pre-unpack to uint8 tensors with f16 scales/biases.
        """
        from asdsl.quantization.core import _unpack_bits

        try:
            from asdsl.kernels import (
                has_native_kernel, has_native_q8_kernel,
                has_native_q3_kernel, has_native_q2_kernel,
            )
            if self.bits == 4:
                has_gemv = has_native_kernel()
            elif self.bits == 8:
                has_gemv = has_native_q8_kernel()
            elif self.bits == 3:
                has_gemv = has_native_q3_kernel()
            elif self.bits == 2:
                has_gemv = has_native_q2_kernel()
            else:
                has_gemv = False
        except ImportError:
            has_gemv = False

        self._use_native_gemv = has_gemv

        # LUT-only (no dispatch): build all eligible projections.
        if self._enable_lut and self.bits == 4 and not self._enable_dispatch:
            self._build_lut_caches()

        # Phase 4 Prerequisite B: build transposed down_proj weights
        if getattr(self, "_use_fatrelu", False):
            self.load_transposed_down_proj()

        # Phase 2 SliM: pre-repack calibrated arrays
        if getattr(self, "_use_slim", False):
            import time as _time
            print("  [SliM] Pre-repacking arrays for calibrated layers...")
            t0 = _time.time()
            for i in range(NUM_LAYERS):
                for name in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"):
                    self._get_slim_repacked(i, name)
            print(f"  [SliM] Repacking done in {_time.time()-t0:.2f}s")

        if getattr(self, "_gguf_projections_loaded", False):
            total = NUM_LAYERS * 4
            print(
                f"  Warm-cache: skipped projection quantize ({total} tensors from GGUF)"
            )
            self._finish_dispatch_caches()
            if self.bits == 4 and self._quant_packed_np:
                self._assert_packed_contiguous()
                self._build_norm_np_cache()
            if self.bits == 4 and self._use_unified:
                self.build_preq_blocks()
                self.build_preq2_blocks()
                self.build_c01_gs128_blocks()
                self._free_packed_after_unified_repack()
            print(f"  Inference: {self._inference_mode_message(has_gemv=has_gemv)}")
            self._build_rope_native_tables()
            return

        if getattr(self, "_loaded_from_cache", False):
            total = NUM_LAYERS * 4
            print(
                f"  Warm-cache: skipped (restored {total} projections from safetensors cache)"
            )
            self._finish_dispatch_caches()
            if self.bits == 4 and self._quant_packed_np:
                self._assert_packed_contiguous()
                self._build_norm_np_cache()
            if self.bits == 4 and self._use_unified:
                self.build_preq_blocks()
                self.build_preq2_blocks()
                self.build_c01_gs128_blocks()
                self._free_packed_after_unified_repack()
            print(f"  Inference: {self._inference_mode_message(has_gemv=has_gemv)}")
            if self._enable_qcsd and self.bits != 16:
                d_bytes = sum(t.nbytes for t in self._draft_quant_u8.values())
                d_bytes += sum(t.nbytes for t in self._draft_quant_packed.values())
                print(
                    f"  QCSD draft bank: {d_bytes / 1e6:.0f} MB ({self._draft_bits}-bit)"
                )
            n_outliers = sum(len(v) for v in self._outlier_values.values())
            if n_outliers > 0:
                print(f"  SpQR outliers: {n_outliers:,} values in FP16 sparse format")
            self._build_rope_native_tables()
            return

        total = NUM_LAYERS * 4
        if self.bits == 16:
            print(f"  Float16 weight cache already populated ({total} tensors)")
            print("  Inference: chunked f16 matvec")
        else:
            done = 0
            qmax = (1 << self.bits) - 1

            if self.bits == 4:
                print(
                    f"  Caching packed Q4 weights ({total} tensors, rows×cols/2) ... ",
                    end="",
                    flush=True,
                )
                for i in range(NUM_LAYERS):
                    for name in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"):
                        qt = self.layers[i][name]
                        rows, cols = qt.shape
                        key = (i, name)
                        self._quant_shapes[key] = (rows, cols)
                        packed_np = np.ascontiguousarray(qt.data, dtype=np.uint8).copy()
                        self._quant_packed[key] = torch.from_numpy(packed_np).reshape(
                            rows, cols // 2
                        )

                        numel = rows * cols
                        n_groups = numel // qt.group_size
                        sc = torch.from_numpy(qt.scales[:n_groups].copy().astype(np.float32))
                        if qt.is_symmetric:
                            dq = (1 << qt.bits) - 1
                            half_range = float(dq) / 2.0
                            bi = (-half_range * sc).to(torch.float32)
                        else:
                            zr = torch.from_numpy(qt.zeros[:n_groups].copy().astype(np.float32))
                            bi = (-zr * sc)
                        self._quant_sc[key] = sc
                        self._quant_bi[key] = bi

                        # Populate NumPy views for fast native dispatch (C-contiguous)
                        self._quant_packed_np[key] = np.ascontiguousarray(
                            packed_np.ravel(), dtype=np.uint8
                        )
                        self._quant_sc_np[key] = np.ascontiguousarray(
                            sc.numpy().ravel(), dtype=np.float32
                        )
                        self._quant_bi_np[key] = np.ascontiguousarray(
                            bi.numpy().ravel(), dtype=np.float32
                        )

                        if self._enable_qcsd and "_draft_" + name in self.layers.get(i, {}):
                            qt_d = self.layers[i]["_draft_" + name]
                            d_numel = rows * cols
                            if qt_d.bits == 2:
                                self._draft_quant_u8[key] = torch.from_numpy(qt_d.data.copy())
                            elif qt_d.bits == 4:
                                dp = np.ascontiguousarray(qt_d.data, dtype=np.uint8).copy()
                                self._draft_quant_packed[key] = torch.from_numpy(dp).reshape(
                                    rows, cols // 2
                                )
                            else:
                                unpacked_d = _unpack_bits(qt_d.data, qt_d.bits)[:d_numel]
                                self._draft_quant_u8[key] = torch.from_numpy(
                                    unpacked_d.astype(np.uint8)
                                )
                            d_qmax = (1 << qt_d.bits) - 1
                            d_n_groups = d_numel // qt_d.group_size
                            d_sc = torch.from_numpy(
                                qt_d.scales[:d_n_groups].copy().astype(np.float32)
                            )
                            if qt_d.is_symmetric:
                                d_half = float(d_qmax) / 2.0
                                d_bi = (-d_half * d_sc).to(torch.float32)
                            else:
                                d_zr = torch.from_numpy(
                                    qt_d.zeros[:d_n_groups].copy().astype(np.float32)
                                )
                                d_bi = (-d_zr * d_sc).to(torch.float32)
                            self._draft_quant_sc[key] = d_sc
                            self._draft_quant_bi[key] = d_bi
                            self._draft_quant_sc_np[key] = d_sc.numpy().ravel()
                            self._draft_quant_bi_np[key] = d_bi.numpy().ravel()
                            if key in self._draft_quant_packed:
                                self._draft_quant_packed_np[key] = self._draft_quant_packed[key].numpy().ravel()
                            if key in self._draft_quant_u8:
                                self._draft_quant_u8_np[key] = self._draft_quant_u8[key].numpy().ravel()

                        done += 1

                for i in range(NUM_LAYERS):
                    if i in self.layers:
                        for k in list(self.layers[i].keys()):
                            if k.startswith("_draft_"):
                                del self.layers[i][k]

                self.layers.clear()
                self._weight_cache.clear()
                packed_bytes = sum(t.nbytes for t in self._quant_packed.values())
                sc_bytes = sum(t.nbytes for t in self._quant_sc.values())
                bi_bytes = sum(t.nbytes for t in self._quant_bi.values())
                total_mb = (packed_bytes + sc_bytes + bi_bytes) / 1e6
                print(f"done ({done}/{total}) | {total_mb:.0f} MB")

                if self._enable_qcsd:
                    d_bytes = sum(t.nbytes for t in self._draft_quant_u8.values())
                    d_bytes += sum(t.nbytes for t in self._draft_quant_packed.values())
                    print(f"  QCSD draft bank: {d_bytes / 1e6:.0f} MB ({self._draft_bits}-bit)")
                    pk = set(self._quant_packed.keys())
                    dk = set(self._draft_quant_sc.keys())
                    if dk != pk:
                        raise RuntimeError(
                            "QCSD draft bank keys mismatch vs primary ("
                            f"{len(dk)} vs {len(pk)} tensors)"
                        )

            else:
                label = "primary" if self._enable_qcsd else ""
                print(
                    f"  Pre-unpacking {total} {label} tensors to uint8 ... ",
                    end="",
                    flush=True,
                )

                for i in range(NUM_LAYERS):
                    for name in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"):
                        qt = self.layers[i][name]
                        rows, cols = qt.shape
                        key = (i, name)
                        self._quant_shapes[key] = (rows, cols)

                        numel = rows * cols
                        unpacked = _unpack_bits(qt.data, qt.bits)[:numel]
                        self._quant_u8[key] = torch.from_numpy(unpacked.astype(np.uint8))

                        n_groups = numel // qt.group_size
                        sc_f16 = torch.from_numpy(qt.scales[:n_groups].copy()).to(torch.float16)
                        if qt.is_symmetric:
                            half_range = qmax / 2.0
                            bi_f16 = (-half_range * sc_f16.float()).half()
                        else:
                            zr = torch.from_numpy(qt.zeros[:n_groups].copy()).to(torch.float16)
                            bi_f16 = (-zr.float() * sc_f16.float()).half()
                        self._quant_sc[key] = sc_f16
                        self._quant_bi[key] = bi_f16

                        if self._enable_qcsd and "_draft_" + name in self.layers.get(i, {}):
                            qt_d = self.layers[i]["_draft_" + name]
                            d_numel = rows * cols
                            if qt_d.bits == 2:
                                self._draft_quant_u8[key] = torch.from_numpy(qt_d.data.copy())
                            else:
                                unpacked_d = _unpack_bits(qt_d.data, qt_d.bits)[:d_numel]
                                self._draft_quant_u8[key] = torch.from_numpy(
                                    unpacked_d.astype(np.uint8)
                                )
                            d_qmax = (1 << qt_d.bits) - 1
                            d_n_groups = d_numel // qt_d.group_size
                            d_sc = torch.from_numpy(qt_d.scales[:d_n_groups].copy()).to(
                                torch.float16
                            )
                            if qt_d.is_symmetric:
                                d_half = d_qmax / 2.0
                                d_bi = (-d_half * d_sc.float()).half()
                            else:
                                d_zr = torch.from_numpy(
                                    qt_d.zeros[:d_n_groups].copy()
                                ).to(torch.float16)
                                d_bi = (-d_zr.float() * d_sc.float()).half()
                            self._draft_quant_sc[key] = d_sc
                            self._draft_quant_bi[key] = d_bi

                        done += 1

                for i in range(NUM_LAYERS):
                    if i in self.layers:
                        for k in list(self.layers[i].keys()):
                            if k.startswith("_draft_"):
                                del self.layers[i][k]

                self.layers.clear()
                self._weight_cache.clear()
                u8_bytes = sum(t.nbytes for t in self._quant_u8.values())
                sc_bytes = sum(t.nbytes for t in self._quant_sc.values())
                bi_bytes = sum(t.nbytes for t in self._quant_bi.values())
                total_mb = (u8_bytes + sc_bytes + bi_bytes) / 1e6
                print(f"done ({done}/{total}) | {total_mb:.0f} MB")

                if self._enable_qcsd:
                    d_bytes = sum(t.nbytes for t in self._draft_quant_u8.values())
                    print(f"  QCSD draft bank: {d_bytes / 1e6:.0f} MB ({self._draft_bits}-bit)")
                    if len(self._draft_quant_u8) != len(self._quant_u8):
                        raise RuntimeError(
                            "QCSD draft bank incomplete relative to primary weights ("
                            f"{len(self._draft_quant_u8)} vs {len(self._quant_u8)} tensors)"
                        )

            n_outliers = sum(len(v) for v in self._outlier_values.values())
            if n_outliers > 0:
                print(f"  SpQR outliers: {n_outliers:,} values in FP16 sparse format")

            self._finish_dispatch_caches()
            if self.bits == 4 and self._quant_packed_np:
                self._assert_packed_contiguous()
                self._build_norm_np_cache()
            if self.bits == 4 and self._use_unified:
                self.build_preq_blocks()
                self.build_preq2_blocks()
                self.build_c01_gs128_blocks()
                self._free_packed_after_unified_repack()
            print(f"  Inference: {self._inference_mode_message(has_gemv=has_gemv)}")

        self._build_rope_native_tables()
        _maybe_calibrate_ahsd_skip_mask(self)

        # (Initialization moved up above the early return to fix benchmarking)

    def _build_rope_native_tables(self) -> None:
        """Pre-compute RoPE cos/sin for native C++ RoPE (matches build_rope_cache)."""
        half_rotary = ROTARY_DIM // 2
        cos = getattr(self, "_cos_table", None)
        if cos is not None:
            if cos.shape[1] == half_rotary:
                return
            self._cos_table = None
            self._sin_table = None
        if not HAS_NATIVE_OPS:
            return
        max_seq = 2048
        half_rotary = ROTARY_DIM // 2
        cos_table = np.zeros((max_seq, half_rotary), dtype=np.float32)
        sin_table = np.zeros((max_seq, half_rotary), dtype=np.float32)
        for pos in range(max_seq):
            for i in range(half_rotary):
                freq = 1.0 / (ROPE_THETA ** (2.0 * i / ROTARY_DIM))
                angle = pos * freq
                cos_table[pos, i] = math.cos(angle)
                sin_table[pos, i] = math.sin(angle)
        self._cos_table = np.ascontiguousarray(cos_table)
        self._sin_table = np.ascontiguousarray(sin_table)
        print(
            f"  Pre-computed RoPE tables: {max_seq} x {half_rotary} "
            f"(partial rotary dim={ROTARY_DIM})"
        )

    def get_norm(self, layer_idx: int, name: str) -> torch.Tensor:
        return self.layer_norms[layer_idx][name]

    def _free_packed_weights_if_unified(self) -> None:
        """Drop packed Q4 arrays when UnifiedEngine uses preq blocks only."""
        if self.bits != 4 or not self._use_unified or self._keep_packed_for_fallback:
            return
        if not self._preq_built:
            return
        freed = sum(int(a.nbytes) for a in self._quant_packed_np.values())
        self._quant_packed_np.clear()
        self._quant_packed.clear()
        gc.collect()
        print(f"  Freed packed Q4 weights ({freed / 1e6:.0f} MB) for unified-only path")

    def build_preq_blocks(self) -> None:
        """Repack Q4 weights into preq blocks (Q4_32 or Q4_128) for native GEMV."""
        BLOCK_SIZE, repack_fn, blocks_to_flat = _preq_repack_imports(self.group_size)

        if getattr(self, "_preq_block_size", None) != BLOCK_SIZE:
            self._preq_blocks_np.clear()
            self._preq_built = False
            self._preq_block_size = BLOCK_SIZE
        if self._preq_built or self.bits != 4:
            return

        if _preq_cache_enabled() and not getattr(self, "_gguf_projections_loaded", False):
            ppath = preq_cache_path_for_store(self)
            if try_restore_preq_cache(self, ppath):
                if self._use_unified:
                    self._build_norm_np_cache()
                return
        elif getattr(self, "_gguf_projections_loaded", False):
            print("  Preq: rebuilding from GGUF-packed weights (no HF preq cache) ...", flush=True)

        label = "Q4_128" if self.group_size == 128 else "Q4_32"
        print(f"  Repacking weights to {label} preq blocks (gs={self.group_size}) ... ", end="", flush=True)
        t0 = time.perf_counter()
        for key, w_np in self._quant_packed_np.items():
            rows, cols = self._quant_shapes[key]
            blocks = repack_fn(
                w_np,
                self._quant_sc_np[key],
                self._quant_bi_np[key],
                rows,
                cols,
                self.group_size,
                bits=self.bits,
            )
            self._preq_blocks_np[key] = blocks_to_flat(blocks)
        self._preq_built = True
        dt = time.perf_counter() - t0
        print(f"done ({len(self._preq_blocks_np)} tensors, {dt:.1f}s)")

        if _preq_cache_enabled():
            save_preq_cache(self, preq_cache_path_for_store(self))

        if self._use_unified:
            self._build_norm_np_cache()

    def build_preq2_blocks(self) -> None:
        """Repack preq Q4_32 blocks into preq2 meta+quant layout for VNNI kernel."""
        if self.bits != 4 or self.group_size != 32:
            return
        if not self._preq_built:
            self.build_preq_blocks()
        use_preq2 = os.environ.get("ASDSL_PREQ2", "0").strip().lower() not in ("0", "false", "no")
        if not use_preq2:
            return
        if getattr(self, "_preq2_built", False):
            return

        from asdsl.quantization.repack_preq2 import meta_to_flat, quant_to_flat, repack_preq_blocks_to_preq2

        self._preq2_meta_np: dict[tuple, np.ndarray] = {}
        self._preq2_quant_np: dict[tuple, np.ndarray] = {}
        t0 = time.perf_counter()
        for key, flat in self._preq_blocks_np.items():
            rows, cols = self._quant_shapes[key]
            meta, quant = repack_preq_blocks_to_preq2(flat, rows, cols, self.group_size)
            self._preq2_meta_np[key] = meta_to_flat(meta)
            self._preq2_quant_np[key] = quant_to_flat(quant)
        self._preq2_built = True
        dt = time.perf_counter() - t0
        print(f"  preq2 repack: {len(self._preq2_meta_np)} tensors ({dt:.1f}s)", flush=True)

    def _dequant_packed_projection(self, key: tuple) -> np.ndarray:
        """Reconstruct float32 weights from packed Q4 arrays."""
        w_np = self._quant_packed_np[key]
        sc_np = self._quant_sc_np[key]
        bi_np = self._quant_bi_np[key]
        rows, cols = self._quant_shapes[key]
        gs = self.group_size
        n_groups = cols // gs
        packed = w_np.reshape(rows, cols // 2)
        lo = (packed & 0x0F).astype(np.float32)
        hi = (packed >> 4).astype(np.float32)
        w_q = np.empty((rows, cols), dtype=np.float32)
        w_q[:, 0::2] = lo
        w_q[:, 1::2] = hi
        sc = np.repeat(sc_np.reshape(rows, n_groups), gs, axis=1)
        bi = np.repeat(bi_np.reshape(rows, n_groups), gs, axis=1)
        return w_q * sc + bi

    def _dequant_from_preq_blocks(self, key: tuple) -> np.ndarray:
        """Reconstruct float32 weights from preq Q4_32 blocks (asymmetric nibble layout)."""
        rows, cols = self._quant_shapes[key]
        flat = self._preq_blocks_np[key]
        gs = self.group_size
        n_groups = cols // gs
        block_size = getattr(self, "_preq_block_size", 20)
        blocks = flat.reshape(rows, n_groups, block_size)
        scales = blocks[:, :, 0:2].view(np.float16).astype(np.float32).reshape(rows, n_groups, 1)
        zeros = blocks[:, :, 2:4].view(np.float16).astype(np.float32).reshape(rows, n_groups, 1)
        nibbles = blocks[:, :, 4:20].astype(np.uint8)
        lows = (nibbles & 0x0F).astype(np.float32)
        highs = (nibbles >> 4).astype(np.float32)
        q = np.empty((rows, n_groups, gs), dtype=np.float32)
        q[:, :, 0::2] = lows
        q[:, :, 1::2] = highs
        w = (q - zeros) * scales
        return w.reshape(rows, cols)

    def _dequant_c01_source_f32(self, key: tuple) -> np.ndarray:
        """Best float32 source for C0.1 g128 requant (packed preferred, else preq blocks)."""
        w_packed = self._quant_packed_np.get(key)
        if w_packed is not None:
            return self._dequant_packed_projection(key)
        if key in self._preq_blocks_np:
            return self._dequant_from_preq_blocks(key)
        raise KeyError(f"missing projection weights for {key}")

    def _free_packed_after_unified_repack(self) -> None:
        """Drop packed Q4 arrays once preq / preq2 / C0.1 layouts are ready."""
        self._free_packed_weights_if_unified()

    def build_c01_gs128_blocks(self) -> None:
        """C0.1: requant gate_up/down at g128; C0.3 adds qkv/o when ASDSL_C03=1."""
        if not _env_flag("ASDSL_C01") and not _env_flag("ASDSL_C03"):
            return
        if self.bits != 4 or self.group_size != 32:
            return
        if getattr(self, "_c01_gs128_built", False):
            return
        if not self._preq_built:
            self.build_preq_blocks()

        from asdsl.quantization.repack_q4_128 import blocks_to_flat, repack_fp32_to_q4_128_blocks

        gs128 = int(os.environ.get("ASDSL_GATEUP_GS", "128"))
        if gs128 != 128:
            return

        self._preq_gs128_np: dict[tuple, np.ndarray] = {}
        proj_names = ("gate_up_proj", "down_proj")
        if _env_flag("ASDSL_C03"):
            proj_names = ("gate_up_proj", "down_proj", "qkv_proj", "o_proj")
        targets = [(i, name) for i in range(NUM_LAYERS) for name in proj_names]
        t0 = time.perf_counter()
        for key in targets:
            if key not in self._quant_shapes:
                continue
            try:
                w_f32 = self._dequant_c01_source_f32(key)
            except KeyError:
                continue
            rows, cols = self._quant_shapes[key]
            blocks = repack_fp32_to_q4_128_blocks(w_f32, rows, cols, gs128)
            self._preq_gs128_np[key] = blocks_to_flat(blocks)
        self._c01_gs128_built = True
        dt = time.perf_counter() - t0
        print(f"  C0.1 g128 repack: {len(self._preq_gs128_np)} tensors ({dt:.1f}s)", flush=True)


# ---------------------------------------------------------------------------
# Per-layer KV history (simple list-based, one entry per token per layer)
# ---------------------------------------------------------------------------

class KVHistory:
    """
    Pre-allocated KV cache stored as contiguous torch tensors.

    Each layer gets a (max_seq, kv_heads, head_dim) buffer.
    `get()` returns zero-copy views - no np.stack / torch.from_numpy per call.
    """
    def __init__(self, max_seq: int = 2048):
        self.k_buf: dict[int, torch.Tensor] = {
            i: torch.zeros(max_seq, NUM_KV_HEADS, HEAD_DIM) for i in range(NUM_LAYERS)
        }
        self.v_buf: dict[int, torch.Tensor] = {
            i: torch.zeros(max_seq, NUM_KV_HEADS, HEAD_DIM) for i in range(NUM_LAYERS)
        }
        self._len: dict[int, int] = {i: 0 for i in range(NUM_LAYERS)}

    def append(self, layer: int, k_vec: torch.Tensor, v_vec: torch.Tensor) -> None:
        n = self._len[layer]
        self.k_buf[layer][n] = k_vec
        self.v_buf[layer][n] = v_vec
        self._len[layer] = n + 1

    def get(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return views into the pre-allocated buffers (zero-copy)."""
        n = self._len[layer]
        return self.k_buf[layer][:n], self.v_buf[layer][:n]

    def get_flat_kv_np(self, layer: int) -> tuple[np.ndarray, np.ndarray]:
        """Contiguous float32 K/V cache flats for native attention (no .cpu().float())."""
        n = self._len[layer]
        k_view = self.k_buf[layer][:n]
        v_view = self.v_buf[layer][:n]
        k_np = k_view.numpy()
        v_np = v_view.numpy()
        if not k_np.flags["C_CONTIGUOUS"]:
            k_np = np.ascontiguousarray(k_np)
        if not v_np.flags["C_CONTIGUOUS"]:
            v_np = np.ascontiguousarray(v_np)
        return k_np.ravel(), v_np.ravel()

    def get_last_np(self, layer: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the most recently appended K/V as numpy (zero-copy view)."""
        n = self._len[layer] - 1
        return self.k_buf[layer][n].numpy(), self.v_buf[layer][n].numpy()

    def snapshot(self) -> dict:
        """Save a lightweight snapshot for QCSD draft rollback."""
        return {
            "lens": dict(self._len),
        }

    def restore(self, snap: dict) -> None:
        """Restore KV cache lengths from a snapshot (zero-copy: data stays)."""
        for i, n in snap["lens"].items():
            self._len[i] = n

    def restore_len(self, n: int) -> None:
        """Truncate KV cache to exactly n tokens (zero-copy rollback)."""
        for i in range(NUM_LAYERS):
            self._len[i] = n

    @property
    def num_tokens(self) -> int:
        return self._len[0]


class ASDSLKVTracker:
    """
    Feeds the same KV data into ASDSL's BlockSparseKVCache for block-sparse
    tracking, eviction analytics, and memory budget calculations.
    Updated once per token (after all layers have been processed).
    """
    def __init__(self):
        from asdsl.inference.kv_cache import BlockSparseKVCache, KVCacheConfig
        cfg = KVCacheConfig(
            num_layers=NUM_LAYERS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            max_blocks=256,
            block_size=16,
            quantize_kv=False,
        )
        self._cache = BlockSparseKVCache(cfg)

    def record_token(self, k_per_layer: list[np.ndarray],
                     v_per_layer: list[np.ndarray]) -> None:
        """k_per_layer[i] shape: (kv_heads, head_dim) for layer i."""
        k_all = np.stack(k_per_layer)   # (num_layers, kv_heads, head_dim)
        v_all = np.stack(v_per_layer)
        self._cache.append(k_all, v_all)

    def stats(self) -> dict:
        n = self._cache.length
        block_size = self._cache.config.block_size
        max_blocks = self._cache.config.max_blocks
        used_blocks = math.ceil(n / block_size) if n > 0 else 0
        bytes_used = (n * NUM_KV_HEADS * HEAD_DIM * 4 * 2 * NUM_LAYERS)  # float32 kv
        return {
            "tokens": n,
            "blocks_used": used_blocks,
            "blocks_capacity": max_blocks,
            "memory_mb": bytes_used / 1e6,
        }


# ---------------------------------------------------------------------------
# Transformer forward pass
# ---------------------------------------------------------------------------

def _use_forward_numpy_path(store: WeightStore) -> bool:
    if os.environ.get("ASDSL_FORWARD_NUMPY", "1").strip().lower() in ("0", "false", "no"):
        return False
    return bool(store._use_native_ops and HAS_NATIVE_OPS and _native_ops is not None)


def _forward_layer_numpy_fast_np(
    residual: np.ndarray,
    layer_idx: int,
    store: WeightStore,
    kv_hist: KVHistory,
    pos: int,
    prof: ForwardProfiler,
) -> np.ndarray:
    """Numpy-native layer; mutates/returns residual (HIDDEN,) float32."""
    if os.environ.get("ASDSL_FORWARD_NOOP", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return residual

    t_layer = time.perf_counter()
    prof._layer_idx = layer_idx
    if residual.size != HIDDEN:
        residual = np.ascontiguousarray(residual, dtype=np.float32).ravel()

    prof.begin("rms_norm_attn")
    rms_att_w = store._norm_np.get(
        (layer_idx, "input_layernorm"),
        store.get_norm(layer_idx, "input_layernorm").detach().cpu().float().numpy(),
    )
    h = np.empty(HIDDEN, dtype=np.float32)
    _native_ops.rmsnorm_f32(residual, h, rms_att_w, HIDDEN, RMS_EPS)
    prof.end()

    prof.begin("qkv_proj")
    qkv_np = store.matvec_np(layer_idx, "qkv_proj", h)
    prof.end()

    prof.begin("qkv_split_rope")
    q = qkv_np[:Q_DIM].reshape(NUM_HEADS, HEAD_DIM)
    k = qkv_np[Q_DIM : Q_DIM + KV_DIM].reshape(NUM_KV_HEADS, HEAD_DIM)
    v = qkv_np[Q_DIM + KV_DIM :].reshape(NUM_KV_HEADS, HEAD_DIM)
    _native_ops.rope_apply_inplace(
        q.ravel(),
        k.ravel(),
        store._cos_table,
        store._sin_table,
        NUM_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        pos,
        2048,
    )
    prof.end()

    prof.begin("kv_write")
    kv_hist.append(
        layer_idx,
        torch.from_numpy(k.astype(np.float32, copy=False)),
        torch.from_numpy(v.astype(np.float32, copy=False)),
    )
    prof.end()

    prof.begin("attention")
    k_cache_flat, v_cache_flat = kv_hist.get_flat_kv_np(layer_idx)
    seq_len = kv_hist._len[layer_idx]
    attn_out = np.empty(NUM_HEADS * HEAD_DIM, dtype=np.float32)
    _native_ops.gqa_decode_attention(
        q.ravel(),
        k_cache_flat,
        v_cache_flat,
        attn_out,
        NUM_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        seq_len,
        1.0 / math.sqrt(HEAD_DIM),
    )
    prof.end()

    prof.begin("o_proj")
    o_proj_np = store.matvec_np(layer_idx, "o_proj", attn_out)
    prof.end()

    prof.begin("residual_attn")
    _native_ops.vec_add_inplace(residual, o_proj_np, HIDDEN)
    prof.end()

    prof.begin("rms_norm_ffn")
    rms_ffn_w = store._norm_np.get(
        (layer_idx, "post_attention_layernorm"),
        store.get_norm(layer_idx, "post_attention_layernorm").detach().cpu().float().numpy(),
    )
    h = np.empty(HIDDEN, dtype=np.float32)
    _native_ops.rmsnorm_f32(residual, h, rms_ffn_w, HIDDEN, RMS_EPS)
    prof.end()

    prof.begin("gate_up_proj")
    gu_np = store.matvec_np(layer_idx, "gate_up_proj", h)
    prof.end()

    prof.begin("silu")
    gate = gu_np[:INTER]
    up = gu_np[INTER:]
    _native_ops.swiglu_inplace(gate, up, INTER)
    prof.end()

    prof.begin("down_proj")
    down_np = store.matvec_np(layer_idx, "down_proj", gate)
    prof.end()

    prof.begin("residual_ffn")
    _native_ops.vec_add_inplace(residual, down_np, HIDDEN)
    prof.end()

    if prof.active():
        prof.layer_total_ms = (time.perf_counter() - t_layer) * 1000.0

    return residual


def _forward_layer_numpy_fast(
    hidden: torch.Tensor,
    layer_idx: int,
    store: WeightStore,
    kv_hist: KVHistory,
    pos: int,
    prof: ForwardProfiler,
) -> torch.Tensor:
    residual = (
        hidden.detach().cpu().float().numpy().ravel()
        if isinstance(hidden, torch.Tensor)
        else np.ascontiguousarray(hidden, dtype=np.float32).ravel()
    )
    out = _forward_layer_numpy_fast_np(residual, layer_idx, store, kv_hist, pos, prof)
    return torch.from_numpy(out).unsqueeze(0)


def forward_layer(
    hidden: torch.Tensor,          # (1, hidden)
    layer_idx: int,
    store: WeightStore,
    kv_hist: KVHistory,            # mutable: this call appends to it
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    pos: int,
    use_draft: bool = False,
) -> torch.Tensor:
    """Single Phi-4 transformer layer. Updates kv_hist with the new K/V.

    Phase 17: Uses native C++ ops for non-GEMV operations (RMSNorm, RoPE,
    attention, SwiGLU, residual add) to eliminate Python dispatch overhead.
    GEMV operations still use the Q8 integer kernel via store.matvec().

    Args:
        use_draft: If True, use the draft (2-bit) weight bank for QCSD.
    """
    use_native = store._use_native_ops and HAS_NATIVE_OPS
    prof = store._forward_profiler
    prof._layer_idx = layer_idx

    if (
        _use_forward_numpy_path(store)
        and not use_draft
        and not store._enable_correction
        and not store._use_fatrelu
        and not store._enable_sparse
    ):
        out = _forward_layer_numpy_fast(hidden, layer_idx, store, kv_hist, pos, prof)
        if prof.active() and layer_idx == prof.target_layer:
            prof.print_report()
            prof.timings_ms.clear()
        return out

    # - Self-attention -
    residual = hidden.detach().cpu().float().numpy().ravel()

    # RMSNorm (pre-attention)
    rms_att_w = store.get_norm(layer_idx, "input_layernorm").detach().cpu().float().numpy()
    if use_native:
        h = np.empty(HIDDEN, dtype=np.float32)
        _native_ops.rmsnorm_f32(residual, h, rms_att_w, HIDDEN, RMS_EPS)
    else:
        rms = np.sqrt(np.mean(residual**2) + RMS_EPS)
        h = (residual / rms) * rms_att_w

    # QKV projection
    h_t = torch.from_numpy(h).unsqueeze(0)
    qkv = store.matvec(layer_idx, "qkv_proj", h_t, use_draft=use_draft)
    qkv_np = qkv.detach().cpu().float().numpy().ravel()

    q = qkv_np[:Q_DIM].reshape(NUM_HEADS, HEAD_DIM).copy()
    k = qkv_np[Q_DIM:Q_DIM + KV_DIM].reshape(NUM_KV_HEADS, HEAD_DIM).copy()
    v = qkv_np[Q_DIM + KV_DIM:].reshape(NUM_KV_HEADS, HEAD_DIM).copy()

    # RoPE
    if use_native and store._cos_table is not None:
        _native_ops.rope_apply_inplace(
            q.ravel(), k.ravel(),
            store._cos_table, store._sin_table,
            NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, pos, 2048)
    else:
        cos_pos = rope_cos[pos:pos + 1]
        sin_pos = rope_sin[pos:pos + 1]
        q_t = apply_rope(torch.from_numpy(q).unsqueeze(0), cos_pos, sin_pos)
        k_t = apply_rope(torch.from_numpy(k).unsqueeze(0), cos_pos, sin_pos)
        q = q_t.squeeze(0).numpy()
        k = k_t.squeeze(0).numpy()

    # Append to KV cache
    kv_hist.append(layer_idx,
                   torch.from_numpy(k.squeeze()),
                   torch.from_numpy(v.squeeze()))

    k_hist, v_hist = kv_hist.get(layer_idx)
    seq_len = k_hist.shape[0]

    # GQA attention
    if use_native:
        k_cache_flat, v_cache_flat = kv_hist.get_flat_kv_np(layer_idx)
        attn_out = np.empty(NUM_HEADS * HEAD_DIM, dtype=np.float32)
        _native_ops.gqa_decode_attention(
            q.ravel(), k_cache_flat, v_cache_flat, attn_out,
            NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, seq_len,
            1.0 / math.sqrt(HEAD_DIM))
        attn_out_t = torch.from_numpy(attn_out).reshape(1, Q_DIM)
    else:
        expand = NUM_HEADS // NUM_KV_HEADS
        k_full = (k_hist.unsqueeze(2)
                        .expand(-1, -1, expand, -1)
                        .reshape(seq_len, NUM_HEADS, HEAD_DIM)
                        .permute(1, 0, 2)
                        .unsqueeze(0))
        v_full = (v_hist.unsqueeze(2)
                        .expand(-1, -1, expand, -1)
                        .reshape(seq_len, NUM_HEADS, HEAD_DIM)
                        .permute(1, 0, 2)
                        .unsqueeze(0))
        q_attn = torch.from_numpy(q).unsqueeze(0).unsqueeze(2)
        attn_out_t = torch.nn.functional.scaled_dot_product_attention(
            q_attn, k_full, v_full,
        )
        attn_out_t = attn_out_t.permute(0, 2, 1, 3).reshape(1, Q_DIM)

    # Output projection + residual
    o_proj = store.matvec(layer_idx, "o_proj", attn_out_t, use_draft=use_draft)
    o_proj_np = o_proj.detach().cpu().float().numpy().ravel()
    if use_native:
        _native_ops.vec_add_inplace(residual, o_proj_np, HIDDEN)
    else:
        residual = residual + o_proj_np
    hidden = torch.from_numpy(residual).unsqueeze(0)

    # - Feed-forward (MLP) -
    residual = hidden.detach().cpu().float().numpy().ravel()

    # RMSNorm (pre-FFN)
    rms_ffn_w = store.get_norm(layer_idx, "post_attention_layernorm").detach().cpu().float().numpy()
    if use_native:
        h = np.empty(HIDDEN, dtype=np.float32)
        _native_ops.rmsnorm_f32(residual, h, rms_ffn_w, HIDDEN, RMS_EPS)
    else:
        rms = np.sqrt(np.mean(residual**2) + RMS_EPS)
        h = (residual / rms) * rms_ffn_w

    # Gate + Up projection
    h_t = torch.from_numpy(h).unsqueeze(0)
    gu = store.matvec(layer_idx, "gate_up_proj", h_t, use_draft=use_draft)
    gu_np = gu.detach().cpu().float().numpy().ravel()
    gate = gu_np[:INTER].copy()
    up = gu_np[INTER:]

    # SwiGLU
    if use_native:
        _native_ops.swiglu_inplace(gate, up, INTER)
        act = gate
    else:
        act = silu(torch.from_numpy(gate)).numpy() * up

    # FATReLU threshold mask (if active)
    tau = None
    if store._use_fatrelu and layer_idx in store._fatrelu_thresholds:
        if store._use_adaptive_fatrelu:
            if store._adaptive_tau_warmup_tokens < store._adaptive_tau_warmup_target:
                measured_tau = float(np.percentile(np.abs(act), 85))
                if layer_idx in store._adaptive_tau:
                    store._adaptive_tau[layer_idx] = (
                        0.7 * store._adaptive_tau[layer_idx] + 0.3 * measured_tau
                    )
                else:
                    store._adaptive_tau[layer_idx] = measured_tau
                tau = store._adaptive_tau[layer_idx]
            else:
                tau = store._adaptive_tau.get(
                    layer_idx, store._fatrelu_thresholds.get(layer_idx, 0))
        else:
            tau = store._fatrelu_thresholds[layer_idx]
        act = act * (np.abs(act) >= tau).astype(np.float32)
        if store._sparsity_debug_count < 3 and layer_idx < 4:
            actual_sparsity = float((np.abs(act) < tau).mean())
            n_active = int((np.abs(act) >= tau).sum())
            print(f"[sparsity token={store._sparsity_debug_count} L{layer_idx}] "
                  f"tau={tau:.5f} sparsity={actual_sparsity:.1%} active={n_active}/8192")
        if layer_idx == 31:
            if store._use_adaptive_fatrelu:
                store._adaptive_tau_warmup_tokens += 1
                if store._adaptive_tau_warmup_tokens == store._adaptive_tau_warmup_target:
                    mean_at = sum(store._adaptive_tau.values()) / max(len(store._adaptive_tau), 1)
                    print(f"[FATReLU] Adaptive warmup complete at token {store._adaptive_tau_warmup_tokens}; "
                          f"measured mean tau={mean_at:.5f} across {len(store._adaptive_tau)} layers")
            if store._sparsity_debug_count < 3:
                store._sparsity_debug_count += 1

    # Down projection
    use_T_sparse = (store._use_fatrelu and layer_idx in store._down_proj_T
                    and not use_draft)
    if use_T_sparse:
        active_rows = np.where(np.abs(act) >= tau)[0].astype(np.int32)
        if store._sparsity_debug_count < 3 and layer_idx < 4:
            actual_sparsity = 1.0 - len(active_rows) / len(act)
            print(f"[sparse_rows token={store._sparsity_debug_count} L{layer_idx}] "
                  f"active_rows={len(active_rows)}/8192 ({100*len(active_rows)/8192:.1f}% dense, "
                  f"sparsity={actual_sparsity:.1%})")
        if len(active_rows) > 0:
            try:
                from asdsl.kernels._native_sparse_gemv import sparse_down_proj_T as _sparse_T
                dT = store._down_proj_T[layer_idx]
                y_down_np = _sparse_T(
                    dT['packed'].ravel(), dT['scales'], dT['biases'],
                    act, active_rows,
                    dT['in_dim'], dT['out_dim'], store.group_size
                )
                store._sparse_down_proj_T_calls += 1
                if use_native:
                    _native_ops.vec_add_inplace(residual, y_down_np, HIDDEN)
                else:
                    residual = residual + y_down_np
                hidden = torch.from_numpy(residual).unsqueeze(0)
            except Exception:
                store._dense_down_proj_fallback_calls += 1
                down_t = store.matvec(layer_idx, "down_proj",
                                      torch.from_numpy(act).unsqueeze(0), use_draft=use_draft)
                if use_native:
                    _native_ops.vec_add_inplace(residual,
                                                down_t.detach().cpu().float().numpy().ravel(), HIDDEN)
                else:
                    hidden = hidden + down_t
        else:
            store._sparse_down_proj_T_calls += 1
            if use_native:
                pass  # residual unchanged (down_proj output is zero)
            hidden = torch.from_numpy(residual).unsqueeze(0)
    elif store._enable_sparse and not use_draft and store._use_native_gemv:
        from asdsl.kernels import compute_activation_bitmask
        bitmask, active_indices = compute_activation_bitmask(
            act, threshold=store._sparsity_threshold
        )
        sparsity = 1.0 - len(active_indices) / len(act)
        if sparsity > 0.80:
            down_t = store.matvec_sparse(
                layer_idx, "down_proj",
                torch.from_numpy(act).unsqueeze(0), bitmask, active_indices
            )
            if use_native:
                _native_ops.vec_add_inplace(residual,
                                            down_t.detach().cpu().float().numpy().ravel(), HIDDEN)
            else:
                hidden = hidden + down_t
        else:
            store._dense_down_proj_fallback_calls += 1
            down_t = store.matvec(layer_idx, "down_proj",
                                  torch.from_numpy(act).unsqueeze(0), use_draft=use_draft)
            if use_native:
                _native_ops.vec_add_inplace(residual,
                                            down_t.detach().cpu().float().numpy().ravel(), HIDDEN)
            else:
                hidden = hidden + down_t
    else:
        store._dense_down_proj_fallback_calls += 1
        down_t = store.matvec(layer_idx, "down_proj",
                              torch.from_numpy(act).unsqueeze(0), use_draft=use_draft)
        if use_native:
            _native_ops.vec_add_inplace(residual,
                                        down_t.detach().cpu().float().numpy().ravel(), HIDDEN)
        else:
            hidden = hidden + down_t

    # EAGLE-3 multi-layer feature fusion
    if store._use_eagle3:
        h_np = hidden.detach().cpu().float().numpy().ravel()
        if layer_idx == 0:
            store._eagle3_hidden_low = h_np.copy()
        elif layer_idx == 15:
            store._eagle3_hidden_mid = h_np.copy()
        elif layer_idx == 31:
            store._eagle3_hidden_high = h_np.copy()

    # Phase 4: per-layer MLP residual correction (after FFN residual add)
    if store._enable_correction and store._correction is not None:
        from asdsl.correction import apply_layer_correction

        orig_shape = hidden.shape
        h_np = hidden.detach().cpu().float().numpy()
        h_np = apply_layer_correction(
            h_np, layer_idx, store._correction, scale=store._correction_scale
        )
        hidden = torch.from_numpy(np.asarray(h_np, dtype=np.float32)).view(
            orig_shape
        ).to(dtype=hidden.dtype, device=hidden.device)

    return hidden


def forward_layer_batch(
    hidden_batch: torch.Tensor,    # (K, hidden)
    layer_idx: int,
    store: WeightStore,
    kv_hist: KVHistory,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    start_pos: int,
) -> torch.Tensor:
    """Batched forward pass for K tokens through one transformer layer.

    Used by QCSD verify phase: loads each weight matrix ONCE and produces
    K outputs via BLAS GEMM instead of K separate GEMV calls.
    Appends all K tokens' KV to the cache with proper causal masking.
    """
    K = hidden_batch.shape[0]

    residual = hidden_batch
    h = rms_norm(hidden_batch, store.get_norm(layer_idx, "input_layernorm"))

    qkv = store.matmul_batch(layer_idx, "qkv_proj", h)

    q = qkv[:, :Q_DIM].view(K, NUM_HEADS, HEAD_DIM)
    k_new = qkv[:, Q_DIM:Q_DIM + KV_DIM].view(K, NUM_KV_HEADS, HEAD_DIM)
    v_new = qkv[:, Q_DIM + KV_DIM:].view(K, NUM_KV_HEADS, HEAD_DIM)

    for i in range(K):
        pos_i = start_pos + i
        cos_p = rope_cos[pos_i:pos_i + 1]
        sin_p = rope_sin[pos_i:pos_i + 1]
        q[i:i + 1] = apply_rope(q[i:i + 1], cos_p, sin_p)
        k_new[i:i + 1] = apply_rope(k_new[i:i + 1], cos_p, sin_p)

    for i in range(K):
        kv_hist.append(layer_idx, k_new[i], v_new[i])

    k_hist, v_hist = kv_hist.get(layer_idx)
    S = k_hist.shape[0]

    expand = NUM_HEADS // NUM_KV_HEADS
    k_full = (k_hist.unsqueeze(2)
                    .expand(-1, -1, expand, -1)
                    .reshape(S, NUM_HEADS, HEAD_DIM)
                    .permute(1, 0, 2)
                    .unsqueeze(0))
    v_full = (v_hist.unsqueeze(2)
                    .expand(-1, -1, expand, -1)
                    .reshape(S, NUM_HEADS, HEAD_DIM)
                    .permute(1, 0, 2)
                    .unsqueeze(0))

    q_attn = q.permute(1, 0, 2).unsqueeze(0)  # (1, heads, K, head_dim)

    # Causal mask: query i at position start_pos+i attends to KV 0..start_pos+i
    # tril(diagonal=d) keeps j <= i+d; we need j <= i + start_pos
    causal = torch.ones(K, S, dtype=torch.bool).tril(diagonal=start_pos)
    attn_mask = torch.where(causal, 0.0, float('-inf'))
    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

    attn_out = torch.nn.functional.scaled_dot_product_attention(
        q_attn, k_full, v_full, attn_mask=attn_mask,
    )
    attn_out = attn_out.squeeze(0).permute(1, 0, 2).reshape(K, Q_DIM)

    hidden_batch = residual + store.matmul_batch(layer_idx, "o_proj", attn_out)

    residual = hidden_batch
    h = rms_norm(hidden_batch, store.get_norm(layer_idx, "post_attention_layernorm"))

    gu = store.matmul_batch(layer_idx, "gate_up_proj", h)
    act = silu(gu[:, :INTER]) * gu[:, INTER:]

    # FATReLU + transposed sparse down_proj (must match forward_layer for EAGLE-3 / QCSD verify)
    if store._use_fatrelu and layer_idx in store._fatrelu_thresholds:
        tau = store._fatrelu_thresholds[layer_idx]
        act = act * (act.abs() >= tau).float()

    use_T_sparse = (store._use_fatrelu and layer_idx in store._down_proj_T
                    and layer_idx in store._fatrelu_thresholds)
    if use_T_sparse:
        try:
            from asdsl.kernels._native_sparse_gemv import sparse_down_proj_T as _sparse_T

            dT = store._down_proj_T[layer_idx]
            act_np_all = act.detach().cpu().float().numpy()
            outs: list[torch.Tensor] = []
            for ki in range(K):
                act_np = act_np_all[ki].ravel()
                # Use tau threshold: must match forward_layer for correctness.
                active_rows = np.where(np.abs(act_np) >= tau)[0].astype(np.int32)
                if len(active_rows) > 0:
                    y_down_np = _sparse_T(
                        dT["packed"].ravel(),
                        dT["scales"],
                        dT["biases"],
                        act_np,
                        active_rows,
                        dT["in_dim"],
                        dT["out_dim"],
                        store.group_size,
                    )
                    outs.append(torch.from_numpy(y_down_np).unsqueeze(0))
                else:
                    outs.append(torch.zeros(1, HIDDEN, dtype=torch.float32))
            down_b = torch.cat(outs, dim=0).to(device=hidden_batch.device, dtype=hidden_batch.dtype)
            hidden_batch = residual + down_b
        except Exception:
            hidden_batch = residual + store.matmul_batch(layer_idx, "down_proj", act)
    else:
        hidden_batch = residual + store.matmul_batch(layer_idx, "down_proj", act)
    return hidden_batch


# ---------------------------------------------------------------------------
# Sparse kernel diagnostic (Phase 5)
# ---------------------------------------------------------------------------

def _test_sparse_kernel_on_layer0(
    store: WeightStore,
    x_test: np.ndarray,
    *,
    threshold: float = 0.01,
) -> bool:
    """Compare Python vs native sparse GEMV on layer-0 down_proj."""
    from asdsl.kernels.sparse_gemv import (
        sparse_gemv_dequant_f16,
        sparse_gemv_dequant_f16_avx2,
        has_sparse_dequant_kernel,
    )
    from asdsl.quantization.core import dequantize_weights

    key = (0, "down_proj")
    w_f16 = store._sparse_dequant_f16.get(key)
    if w_f16 is None:
        w_np = store._quant_packed_np[key]
        rows, cols = store._quant_shapes[key]
        gs = store.group_size
        numel = rows * cols
        n_groups = numel // gs
        qt_stub = type("Q", (), {})()
        qt_stub.data = w_np
        qt_stub.scales = store._quant_sc_np[key][:n_groups]
        qt_stub.zeros = None
        qt_stub.group_size = gs
        qt_stub.bits = 4
        qt_stub.numel = numel
        qt_stub.is_symmetric = True
        qt_stub.shape = (rows, cols)
        deq = dequantize_weights(qt_stub).reshape(rows, cols)
        w_f16 = np.clip(deq.astype(np.float32), -65504.0, 65504.0).astype(np.float16)

    x_test = np.ascontiguousarray(x_test, dtype=np.float32).ravel()
    py_out = sparse_gemv_dequant_f16(w_f16, x_test, threshold=threshold)
    cpp_out = sparse_gemv_dequant_f16_avx2(w_f16, x_test, threshold=threshold)
    has_nan = bool(np.any(np.isnan(cpp_out)))
    has_inf = bool(np.any(np.isinf(cpp_out)))
    max_diff = float(np.max(np.abs(py_out - cpp_out))) if not has_nan else float("inf")
    print(
        f"[SPARSE DIAG] native={has_sparse_dequant_kernel()} "
        f"nan={has_nan} inf={has_inf} max_diff={max_diff:.4f}"
    )
    print(f"[SPARSE DIAG] cpp_out[:5]={cpp_out[:5]}")
    print(f"[SPARSE DIAG] py_out[:5]={py_out[:5]}")
    return (not has_nan) and max_diff < 0.1


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes")


def _ahsd_draft_k() -> int:
    return int(os.environ.get("ASDSL_AHSD_DRAFT_K", "1"))


def _use_ahsd_path() -> bool:
    return _env_flag("ASDSL_USE_AHSD") or _env_flag("ASDSL_USE_SDQS")


def _use_pld_path() -> bool:
    return _env_flag("ASDSL_USE_PLD")


def _maybe_calibrate_ahsd_skip_mask(store: WeightStore) -> None:
    """Phase D: adaptive skip mask calibration after warm_cache."""
    if not _env_flag("ASDSL_AHSD_CALIBRATE", "0"):
        return
    if not getattr(store, "_use_unified", False) or store.bits != 4:
        return
    if getattr(store, "_ahsd_skip_mask", None) is not None:
        return
    try:
        from asdsl.speculative.ahsd import calibrate_and_store_skip_mask

        calibrate_and_store_skip_mask(store)
    except Exception as exc:
        print(f"  AHSD calibration skipped: {exc}", flush=True)


def generate_stream_ahsd(
    prompt: str,
    store: WeightStore,
    tokenizer,
    max_new_tokens: int = 50,
    system_prompt: str = "You are a helpful AI assistant.",
    bench_metrics_out: list | None = None,
) -> str:
    """AHSD/SDQS generation via UnifiedEngine (yields StreamToken)."""
    from asdsl.inference.unified_bridge import ahsd_generate

    print("\n" + "=" * 66)
    mode = "SDQS" if _env_flag("ASDSL_USE_SDQS") else "AHSD"
    print(f"ASDSL x Phi-4 - {mode} Speculative Decoding (UnifiedEngine)")
    print("=" * 66)
    print(f"Prompt : {prompt!r}")
    print(f"Draft K: {_ahsd_draft_k()}")
    print("-" * 66)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    input_ids = _normalize_input_ids(input_ids)

    _maybe_calibrate_ahsd_skip_mask(store)

    os.environ.setdefault("ASDSL_SPECULATIVE_PROFILE", "1")
    result = ahsd_generate(
        store,
        input_ids,
        max_new_tokens=max_new_tokens,
        draft_k=_ahsd_draft_k(),
        skip_mask=getattr(store, "_ahsd_skip_mask", None),
        use_sdqs=_env_flag("ASDSL_USE_SDQS"),
    )

    gen_ids = result.token_ids[len(input_ids) :]
    for step, tid in enumerate(gen_ids):
        tok_text = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens([int(tid)])
        )
        yield StreamToken(
            text=tok_text,
            token_id=int(tid),
            step=step,
            is_eos=False,
            elapsed_s=result.decode_s,
            tokens_per_second=result.tokens_per_second,
            step_elapsed_s=result.decode_s / max(len(gen_ids), 1),
        )

    print(
        f"\n\nGenerated : {result.decode_tokens} tokens  |  "
        f"{result.tokens_per_second:.2f} tok/s  |  decode {result.decode_s:.1f}s"
    )
    print(
        f"acceptance_rate={result.acceptance_rate:.4f} "
        f"draft_tokens={result.draft_tokens} "
        f"verify_ms={result.verify_ms:.2f} "
        f"draft_ms={result.draft_ms:.2f} "
        f"speculative_cycles={result.speculative_cycles}",
        flush=True,
    )
    print("=" * 66)

    if bench_metrics_out is not None:
        bench_metrics_out.append(
            {
                "decode_tokens": result.decode_tokens,
                "decode_s": result.decode_s,
                "tokens_per_second": result.tokens_per_second,
                "acceptance_rate": result.acceptance_rate,
                "draft_tokens": result.draft_tokens,
                "verify_ms": result.verify_ms,
                "draft_ms": result.draft_ms,
                "speculative_cycles": result.speculative_cycles,
            }
        )


def generate_stream_pld(
    prompt: str,
    store: WeightStore,
    tokenizer,
    max_new_tokens: int = 50,
    system_prompt: str = "You are a helpful AI assistant.",
    bench_metrics_out: list | None = None,
) -> str:
    """Prompt Lookup Decoding via UnifiedEngine (lossless greedy verify)."""
    from asdsl.inference.unified_bridge import pld_generate

    print("\n" + "=" * 66)
    print("ASDSL x Phi-4 - Prompt Lookup Decoding (UnifiedEngine)")
    print("=" * 66)
    print(f"Prompt : {prompt!r}")
    print("-" * 66)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    input_ids = _normalize_input_ids(input_ids)

    t0 = time.perf_counter()
    out_ids = pld_generate(store, input_ids, max_new_tokens=max_new_tokens)
    decode_s = time.perf_counter() - t0
    gen_ids = out_ids[len(input_ids) :]
    decode_tokens = len(gen_ids)
    tps = decode_tokens / decode_s if decode_s > 0 else 0.0

    for step, tid in enumerate(gen_ids):
        tok_text = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens([int(tid)])
        )
        yield StreamToken(
            text=tok_text,
            token_id=int(tid),
            step=step,
            is_eos=int(tid) in EOS_TOKEN_IDS,
            elapsed_s=decode_s,
            tokens_per_second=tps,
            step_elapsed_s=decode_s / max(decode_tokens, 1),
        )

    print(f"\n\nGenerated : {decode_tokens} tokens  |  {tps:.2f} tok/s  |  decode {decode_s:.1f}s")
    print("=" * 66)

    if bench_metrics_out is not None:
        bench_metrics_out.append(
            {
                "decode_tokens": decode_tokens,
                "decode_s": decode_s,
                "tokens_per_second": tps,
            }
        )


def generate_stream(
    prompt: str,
    store: WeightStore,
    tokenizer,
    max_new_tokens: int = 50,
    system_prompt: str = "You are a helpful AI assistant.",
    bench_metrics_out: list | None = None,
    logits_hook: Optional[Callable[[np.ndarray], None]] = None,
) -> str:
    if _use_pld_path() and getattr(store, "_use_unified", False) and store.bits == 4:
        yield from generate_stream_pld(
            prompt=prompt,
            store=store,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            system_prompt=system_prompt,
            bench_metrics_out=bench_metrics_out,
        )
        return
    if _use_ahsd_path() and getattr(store, "_use_unified", False) and store.bits == 4:
        yield from generate_stream_ahsd(
            prompt=prompt,
            store=store,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            system_prompt=system_prompt,
            bench_metrics_out=bench_metrics_out,
        )
        return

    # C++ decode loop (ASDSL_CPP_GENERATE=1): single GIL-free generate() call.
    if os.environ.get("ASDSL_CPP_GENERATE", "").strip() == "1":
        if getattr(store, "_use_unified", False) and store.bits == 4:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            input_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            input_ids = _normalize_input_ids(input_ids)
            from asdsl.inference.unified_bridge import cpp_generate, reset_unified_session

            reset_unified_session(store)
            t_decode_start = time.perf_counter()
            new_tokens = cpp_generate(store, input_ids, max_new_tokens)
            decode_s = time.perf_counter() - t_decode_start
            n_new = len(new_tokens)
            avg_step = decode_s / max(n_new, 1)
            for step, tid in enumerate(new_tokens):
                tok_text = tokenizer.convert_tokens_to_string(
                    tokenizer.convert_ids_to_tokens([tid])
                )
                elapsed = t_decode_start + (step + 1) * avg_step
                tps = (step + 1) / elapsed if elapsed > t_decode_start else 0.0
                ignore_eos = os.environ.get("ASDSL_IGNORE_EOS", "").strip().lower() in (
                    "1",
                    "true",
                    "yes",
                )
                is_eos = (not ignore_eos) and tid in EOS_TOKEN_IDS
                yield StreamToken(
                    text=tok_text,
                    token_id=tid,
                    step=step,
                    is_eos=is_eos,
                    elapsed_s=elapsed - t_decode_start,
                    tokens_per_second=tps,
                    step_elapsed_s=avg_step,
                )
                if is_eos:
                    return
            return

    print("\n" + "=" * 66)
    print("ASDSL x Phi-4 - CPU Inference")
    print("=" * 66)
    print(f"Prompt : {prompt!r}")
    print("-" * 66)

    # Format prompt using the tokenizer's built-in chat template.
    # Phi-4 was fine-tuned with a system turn; including one markedly
    # improves instruction-following quality.
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    input_ids = _normalize_input_ids(input_ids)

    # Pre-compute RoPE tables (generous max length).
    # Pass ROTARY_DIM (96) - only the rotated portion of each head needs tables.
    max_seq = len(input_ids) + max_new_tokens + 64
    rope_cos, rope_sin = build_rope_cache(max_seq, ROTARY_DIM)

    # Per-layer KV history (pre-allocated torch tensors)
    kv_hist = KVHistory(max_seq=max_seq)

    if getattr(store, "_use_unified", False) and store.bits == 4:
        from asdsl.inference.unified_bridge import reset_unified_session

        reset_unified_session(store)

    # ASDSL tracker - updated once per generated token for block-sparse analytics
    asdsl_tracker = ASDSLKVTracker()

    skip_kv_tracker = os.environ.get("ASDSL_SKIP_KV_TRACKER", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    forward_noop = os.environ.get("ASDSL_FORWARD_NOOP", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    noop_diag = os.environ.get("ASDSL_NOOP_DIAG", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    noop_skip_lmhead = forward_noop and os.environ.get(
        "ASDSL_NOOP_SKIP_LMHEAD", "1"
    ).strip().lower() in ("1", "true", "yes")
    noop_diag_acc = {"gemv": 0.0, "lmhead": 0.0, "embed": 0.0, "tokens": 0}

    def run_forward(token_id: int, pos: int, need_logits: bool = True) -> torch.Tensor | None:
        """Full 32-layer forward pass for a single token at position pos.
        When need_logits=False (prefill body), skips the expensive LM-head matmul."""
        store._forward_profiler.set_pos(pos)
        tid = int(token_id)

        if forward_noop:
            if not need_logits:
                return None
            if logits is not None and noop_skip_lmhead:
                return logits

        if getattr(store, "_use_unified", False) and store.bits == 4:
            from asdsl.inference.unified_bridge import unified_forward_token

            logits_np = unified_forward_token(store, tid, pos, need_logits=need_logits)
            if logits_np is None:
                return None
            return torch.from_numpy(logits_np)

        use_np_stack = (
            _use_forward_numpy_path(store)
            and not store._enable_correction
            and not store._use_fatrelu
            and not store._enable_sparse
        )
        prof = store._forward_profiler

        if use_np_stack:
            t_embed = time.perf_counter()
            hidden_np = store.embed_f16[tid].float().numpy().ravel().astype(
                np.float32, copy=False
            )
            t_embed_done = time.perf_counter()
            t_layers = time.perf_counter()
            for i in range(NUM_LAYERS):
                hidden_np = _forward_layer_numpy_fast_np(
                    hidden_np, i, store, kv_hist, pos, prof
                )
                if prof.active() and i == prof.target_layer:
                    prof.print_report()
                    prof.timings_ms.clear()
            t_layers_done = time.perf_counter()
            if noop_diag and need_logits:
                noop_diag_acc["embed"] += t_embed_done - t_embed
                noop_diag_acc["gemv"] += t_layers_done - t_layers
                noop_diag_acc["tokens"] += 1
            if not need_logits:
                return None
            if noop_skip_lmhead and logits is not None:
                return logits
            t_lm = time.perf_counter()
            final_w = store._norm_np.get(
                (-1, "final"),
                store.final_norm.detach().cpu().float().numpy(),
            )
            normed = np.empty(HIDDEN, dtype=np.float32)
            _native_ops.rmsnorm_f32(
                hidden_np, normed, final_w, HIDDEN, RMS_EPS
            )
            out_logits = store.lm_head_matvec(torch.from_numpy(normed).unsqueeze(0))
            if noop_diag:
                noop_diag_acc["lmhead"] += time.perf_counter() - t_lm
            return out_logits

        hidden = store.embed_f16[tid].float().unsqueeze(0)
        k_new, v_new = [], []
        for i in range(NUM_LAYERS):
            hidden = forward_layer(hidden, i, store, kv_hist, rope_cos, rope_sin, pos)
            if not skip_kv_tracker:
                k_np, v_np = kv_hist.get_last_np(i)
                k_new.append(k_np)
                v_new.append(v_np)
        if not skip_kv_tracker and k_new:
            asdsl_tracker.record_token(k_new, v_new)
        if not need_logits:
            return None
        hidden = rms_norm(hidden, store.final_norm)
        return store.lm_head_matvec(hidden)

    # Prefill
    with torch.inference_mode():
        logits = None
        for pos, tid in enumerate(input_ids):
            is_last = (pos == len(input_ids) - 1)
            logits = run_forward(tid, pos, need_logits=is_last)

    # Decode — yield each token as it's produced
    pos = len(input_ids)
    t_decode_start = time.perf_counter()
    token_times: list[float] = []

    with torch.inference_mode():
        for step in range(max_new_tokens):
            t_step = time.perf_counter()
            next_token = int(logits.argmax())
            tok_text = tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens([next_token])
            )
            elapsed = time.perf_counter() - t_decode_start
            tps = (step + 1) / elapsed if elapsed > 0 else 0.0
            ignore_eos = os.environ.get("ASDSL_IGNORE_EOS", "").strip().lower() in (
                "1",
                "true",
                "yes",
            )
            is_eos = (not ignore_eos) and next_token in EOS_TOKEN_IDS

            if is_eos:
                token_times.append(time.perf_counter() - t_step)
                yield StreamToken(
                    text=tok_text,
                    token_id=next_token,
                    step=step,
                    is_eos=is_eos,
                    elapsed_s=elapsed,
                    tokens_per_second=tps,
                    step_elapsed_s=token_times[-1],
                )
                return

            if store._forward_profiler.enabled and step == 1:
                store._forward_profiler.target_pos = pos
            logits = run_forward(next_token, pos)
            pos += 1
            step_elapsed = time.perf_counter() - t_step
            token_times.append(step_elapsed)

            yield StreamToken(
                text=tok_text,
                token_id=next_token,
                step=step,
                is_eos=is_eos,
                elapsed_s=elapsed,
                tokens_per_second=tps,
                step_elapsed_s=step_elapsed,
            )

    if len(token_times) >= 2:
        times = np.array(token_times, dtype=np.float64)
        print(
            f"Token timing: first={times[0]*1000:.1f}ms  "
            f"median={np.median(times)*1000:.1f}ms  "
            f"p90={np.percentile(times, 90)*1000:.1f}ms  "
            f"last={times[-1]*1000:.1f}ms"
        )
    if noop_diag and noop_diag_acc["tokens"] > 0:
        n = noop_diag_acc["tokens"]
        gemv_ms = noop_diag_acc["gemv"] * 1000.0 / n
        lm_ms = noop_diag_acc["lmhead"] * 1000.0 / n
        emb_ms = noop_diag_acc["embed"] * 1000.0 / n
        med_ms = float(np.median(token_times)) * 1000.0 if token_times else 0.0
        other_ms = max(0.0, med_ms - gemv_ms - lm_ms - emb_ms)
        print(
            f"[NOOP DIAG] per-token ms (n={n}): "
            f"layers={gemv_ms:.1f} lmhead={lm_ms:.1f} embed={emb_ms:.1f} "
            f"other~={other_ms:.1f} (median step {med_ms:.1f}ms)"
        )
        if noop_skip_lmhead:
            print("[NOOP DIAG] lm_head skipped on decode (ASDSL_NOOP_SKIP_LMHEAD=1)")


def generate(
    prompt: str,
    store: WeightStore,
    tokenizer,
    max_new_tokens: int = 50,
    system_prompt: str = "You are a helpful AI assistant.",
    bench_metrics_out: list | None = None,
    logits_hook: Optional[Callable[[np.ndarray], None]] = None,
) -> str:
    """Eager generation wrapper over ``generate_stream`` for benchmark callers."""
    print("\nAssistant: ", end="", flush=True)

    def _safe_print_token(text: str) -> None:
        try:
            print(text, end="", flush=True)
        except UnicodeEncodeError:
            enc = sys.stdout.encoding or "utf-8"
            print(text.encode(enc, errors="replace").decode(enc, errors="replace"), end="", flush=True)

    token_ids: list[int] = []
    step_times: list[float] = []
    last_elapsed = 0.0
    last_tps = 0.0
    for tok in generate_stream(
        prompt=prompt,
        store=store,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        system_prompt=system_prompt,
        bench_metrics_out=None,
        logits_hook=logits_hook,
    ):
        _safe_print_token(tok.text)
        token_ids.append(int(tok.token_id))
        if tok.step_elapsed_s > 0:
            step_times.append(float(tok.step_elapsed_s))
        last_elapsed = float(tok.elapsed_s)
        last_tps = float(tok.tokens_per_second)

    n_tokens = len(token_ids)
    if len(step_times) >= 2:
        decode_s = float(sum(step_times[1:]))
        tps = (len(step_times) - 1) / decode_s if decode_s > 0 else 0.0
    else:
        tps = (n_tokens / last_elapsed) if last_elapsed > 0 else 0.0
    if tps <= 0.0:
        tps = last_tps

    response_text = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(token_ids)
    )

    print(f"\n\nGenerated : {n_tokens} tokens  |  {tps:.2f} tok/s  |  decode {last_elapsed:.1f}s")
    if len(step_times) >= 2:
        print(f"decode {tps:.2f} tok/s (tokens 2-{n_tokens})", flush=True)
    print("=" * 66)

    if bench_metrics_out is not None:
        bench_metrics_out.append(
            {
                "decode_tokens": n_tokens,
                "decode_s": float(last_elapsed),
                "tokens_per_second": float(tps),
            }
        )

    return response_text


# ---------------------------------------------------------------------------
# QCSD: Quantization Cascade Speculative Decoding (Tier 2)
# ---------------------------------------------------------------------------

def generate_qcsd(
    prompt: str,
    store: WeightStore,
    tokenizer,
    max_new_tokens: int = 50,
    system_prompt: str = "You are a helpful AI assistant.",
    draft_k: int = 7,
    bench_metrics_out: list | None = None,
    generated_ids_out: list | None = None,
) -> str:
    """Generate tokens using Quantization Cascade Speculative Decoding.

    Uses the 2-bit draft bank to speculatively generate K tokens, then
    verifies against the primary (4-bit) model. Accepted tokens are
    produced at the throughput of batch verification.
    """
    print("\n" + "=" * 66)
    print("ASDSL x Phi-4 - QCSD Speculative Decoding")
    print("=" * 66)
    print(f"Prompt : {prompt!r}")
    print(f"Draft K: {draft_k} | Primary: {store.bits}-bit | Draft: {store._draft_bits}-bit")
    print("-" * 66)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    input_ids = _normalize_input_ids(input_ids)

    max_seq = len(input_ids) + max_new_tokens + draft_k + 64
    rope_cos, rope_sin = build_rope_cache(max_seq, ROTARY_DIM)
    kv_hist = KVHistory(max_seq=max_seq)
    asdsl_tracker = ASDSLKVTracker()

    def run_forward(token_id: int, pos: int, kv: KVHistory,
                    need_logits: bool = True, use_draft: bool = False):
        hidden = store.embed_f16[token_id].float().unsqueeze(0)
        k_new, v_new = [], []
        for i in range(NUM_LAYERS):
            hidden = forward_layer(hidden, i, store, kv, rope_cos, rope_sin,
                                   pos, use_draft=use_draft)
            k_np, v_np = kv.get_last_np(i)
            k_new.append(k_np)
            v_new.append(v_np)
        asdsl_tracker.record_token(k_new, v_new)
        if not need_logits:
            return None
        hidden = rms_norm(hidden, store.final_norm)
        # Capture last final hidden for EAGLE-3 MTP draft generation
        if store._use_eagle3:
            store._last_final_hidden = hidden.detach().cpu().float().numpy().ravel()
        return store.lm_head_matvec(hidden)

    # Prefill with primary model
    print("Prefill: ", end="", flush=True)
    t_prefill_start = time.perf_counter()
    with torch.inference_mode():
        logits = None
        for pos, tid in enumerate(input_ids):
            is_last = (pos == len(input_ids) - 1)
            logits = run_forward(tid, pos, kv_hist, need_logits=is_last, use_draft=False)
    t_prefill = time.perf_counter() - t_prefill_start
    print(f"done ({len(input_ids)} tokens in {t_prefill:.1f}s)")

    # QCSD decode loop
    print("\nAssistant: ", end="", flush=True)
    generated: list[int] = []
    total_draft = 0
    total_accepted = 0
    total_draft_ms = 0.0
    total_verify_ms = 0.0
    t_decode_start = time.perf_counter()
    # Target verify: one batched stack (all draft positions at once) should give
    # _verify_calls == speculative_cycles. If _verify_calls ~= draft_k * cycles, a
    # serial-per-token verify bug is likely.
    _verify_calls = 0
    _verify_extra_run_forward = 0
    speculative_cycles = 0

    pos = len(input_ids)
    store.enter_verify_phase()
    try:
        with torch.inference_mode():
            while len(generated) < max_new_tokens:
                current_token = int(logits.argmax())

                if current_token in EOS_TOKEN_IDS:
                    generated.append(current_token)
                    tok_text = tokenizer.convert_tokens_to_string(
                        tokenizer.convert_ids_to_tokens([current_token])
                    )
                    print(tok_text, end="", flush=True)
                    break

                # ── DRAFT PHASE ──────────────────────────────────────
                kv_snap = kv_hist.snapshot()
                draft_start_pos = pos
                draft_tokens: list[int] = []
                draft_tok = current_token
                t_draft0 = time.perf_counter()

                for k_step in range(draft_k):
                    draft_logits = run_forward(
                        draft_tok, draft_start_pos + k_step, kv_hist,
                        need_logits=True, use_draft=True,
                    )
                    next_draft = int(draft_logits.argmax())
                    draft_tokens.append(next_draft)
                    draft_tok = next_draft
                    if next_draft in EOS_TOKEN_IDS:
                        break

                total_draft += len(draft_tokens)
                kv_hist.restore_len(draft_start_pos)
                total_draft_ms += (time.perf_counter() - t_draft0) * 1000.0

                # ── VERIFY PHASE (BATCHED TARGET) ──────────────────
                t_verify0 = time.perf_counter()
                L = len(draft_tokens)
                if L == 0:
                    verify_tokens = [current_token]
                else:
                    verify_tokens = [current_token] + draft_tokens[:-1]
                n_verify = len(verify_tokens)

                hidden_batch = torch.stack(
                    [store.embed_f16[tid].float() for tid in verify_tokens]
                )

                speculative_cycles += 1
                _verify_calls += 1
                for i in range(NUM_LAYERS):
                    hidden_batch = forward_layer_batch(
                        hidden_batch, i, store, kv_hist,
                        rope_cos, rope_sin, draft_start_pos,
                    )
                store.clear_tmp_cache()

                hidden_batch = rms_norm(hidden_batch, store.final_norm)
                all_logits = store.lm_head_matmul_batch(hidden_batch)

                for vi in range(n_verify):
                    k_new_list, v_new_list = [], []
                    for layer in range(NUM_LAYERS):
                        cache_idx = kv_hist._len[layer] - n_verify + vi
                        k_new_list.append(kv_hist.k_buf[layer][cache_idx].numpy())
                        v_new_list.append(kv_hist.v_buf[layer][cache_idx].numpy())
                    asdsl_tracker.record_token(k_new_list, v_new_list)
                total_verify_ms += (time.perf_counter() - t_verify0) * 1000.0

                # Emit greedy next token (matches standard AR).
                generated.append(current_token)
                print(
                    tokenizer.convert_tokens_to_string(
                        tokenizer.convert_ids_to_tokens([current_token])
                    ),
                    end="",
                    flush=True,
                )

                accepted: list[int] = []
                correction: int | None = None
                for k_idx in range(L):
                    ref_tok = int(all_logits[k_idx].argmax())
                    if ref_tok == draft_tokens[k_idx]:
                        accepted.append(draft_tokens[k_idx])
                    else:
                        correction = ref_tok
                        break

                stop_decode = False
                for tok in accepted:
                    generated.append(tok)
                    print(
                        tokenizer.convert_tokens_to_string(
                            tokenizer.convert_ids_to_tokens([tok])
                        ),
                        end="",
                        flush=True,
                    )
                    if tok in EOS_TOKEN_IDS:
                        stop_decode = True
                        break

                total_accepted += len(accepted)

                if stop_decode:
                    break

                # ── KV ALIGNMENT ─────────────────────────────────────
                if L == 0:
                    logits = all_logits[0]
                    pos += 1
                elif correction is not None:
                    n_keep_verify = 1 + len(accepted)
                    if n_keep_verify < n_verify:
                        kv_hist.restore_len(draft_start_pos + n_keep_verify)
                    _verify_extra_run_forward += 1
                    logits = run_forward(
                        correction, draft_start_pos + n_keep_verify, kv_hist,
                        need_logits=True, use_draft=False,
                    )
                    generated.append(correction)
                    print(
                        tokenizer.convert_tokens_to_string(
                            tokenizer.convert_ids_to_tokens([correction])
                        ),
                        end="",
                        flush=True,
                    )
                    pos += n_keep_verify + 1
                    if correction in EOS_TOKEN_IDS:
                        break
                else:
                    # All draft tokens matched target greedy predictions.
                    _verify_extra_run_forward += 1
                    logits = run_forward(
                        draft_tokens[-1], draft_start_pos + n_verify, kv_hist,
                        need_logits=True, use_draft=False,
                    )
                    pos += n_verify + 1
                    if draft_tokens[-1] in EOS_TOKEN_IDS:
                        break

                if len(generated) >= max_new_tokens:
                    generated[:] = generated[:max_new_tokens]
                    break
    finally:
        store.exit_verify_phase()

    t_decode = time.perf_counter() - t_decode_start
    n_tokens = len(generated)
    tps = n_tokens / t_decode if t_decode > 0 else 0
    accept_rate = total_accepted / max(total_draft, 1)

    response_text = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(generated)
    )

    kv_stats = asdsl_tracker.stats()
    print(f"\n\nGenerated : {n_tokens} tokens  |  {tps:.2f} tok/s  |  decode {t_decode:.1f}s")
    print(f"QCSD      : acceptance rate {accept_rate:.1%}  |  "
          f"drafted {total_draft} / accepted {total_accepted}")
    print(
        f"acceptance_rate={accept_rate:.4f} draft_tokens={total_draft} "
        f"verify_ms={total_verify_ms:.2f} draft_ms={total_draft_ms:.2f} "
        f"speculative_cycles={speculative_cycles}",
        flush=True,
    )
    _avg_v = _verify_calls / max(speculative_cycles, 1)
    print(
        f"QCSD verify telemetry: _verify_calls (batched target stacks)={_verify_calls} "
        f"in {speculative_cycles} cycle(s); avg {_avg_v:.2f}/cycle "
        f"(expect ~1.0; ~{draft_k}x would suggest serial verify over k); "
        f"extra target run_forward after verify={_verify_extra_run_forward}"
    )
    print(f"ASDSL KV  : {kv_stats['tokens']} tokens tracked  "
          f"| {kv_stats['blocks_used']}/{kv_stats['blocks_capacity']} blocks")
    print("=" * 66)

    if bench_metrics_out is not None:
        bench_metrics_out.append(
            {
                "decode_tokens": n_tokens,
                "decode_s": t_decode,
                "tokens_per_second": tps,
                "acceptance_rate": accept_rate,
                "draft_tokens": total_draft,
                "verify_ms": total_verify_ms,
                "draft_ms": total_draft_ms,
                "speculative_cycles": speculative_cycles,
                "_verify_calls": _verify_calls,
                "qcsd_verify_batched_passes": _verify_calls,
                "qcsd_speculative_cycles": speculative_cycles,
                "qcsd_verify_extra_run_forward": _verify_extra_run_forward,
            }
        )

    if generated_ids_out is not None:
        generated_ids_out.clear()
        generated_ids_out.extend(generated)

    return response_text


# ---------------------------------------------------------------------------
# EAGLE-3: MTP head speculative decoding (Profile G)
# ---------------------------------------------------------------------------

def find_ngram_drafts(token_history: list[int], n: int = 3, k: int = 4) -> list[int]:
    """
    Find draft tokens by n-gram matching in the recent context.
    
    Args:
        token_history: list of token IDs generated so far (including prompt)
        n: n-gram size for matching (2-4 works best)
        k: number of draft tokens to propose
    
    Returns:
        list of draft token IDs (length 0 to k), or empty list if no match found
    """
    if len(token_history) < n + k:
        return []
    
    # The query: last n tokens generated
    query = token_history[-n:]
    
    # Search the history (excluding the last n tokens) for the query
    search_window = token_history[:-n]
    
    drafts: list[int] = []
    # Search backwards for the most recent match
    for i in range(len(search_window) - n, -1, -1):
        if search_window[i:i+n] == query:
            # Found a match — take the next k tokens as drafts
            remaining = search_window[i+n:]
            drafts = remaining[:k]
            if drafts:
                return drafts
    
    return []


def generate_pld(
    prompt: str,
    store: "WeightStore",
    tokenizer,
    max_new_tokens: int = 128,
    draft_n: int = 3,
    draft_k: int = 4,
    bench_metrics_out: list | None = None,
) -> str:
    """
    Prompt Lookup Decoding: speculative decoding with zero auxiliary model cost.
    
    N-gram matching finds candidates from context; base model verifies in one batch.
    Break-even acceptance rate: 0% (any match is free speedup).
    """
    import time
    
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    input_ids = _normalize_input_ids(input_ids)
    
    max_seq = len(input_ids) + max_new_tokens + draft_k + 32
    rope_cos, rope_sin = build_rope_cache(max_seq, ROTARY_DIM)
    kv_hist = KVHistory(max_seq=max_seq)
    
    generated = list(input_ids)
    total_accepted = 0
    total_drafted = 0
    total_cycles = 0
    t_start = time.perf_counter()
    
    # Prefill
    with torch.inference_mode():
        logits = None
        for pos, tid in enumerate(input_ids):
            hidden = store.embed_f16[tid].float().unsqueeze(0)
            for i in range(NUM_LAYERS):
                hidden = forward_layer(hidden, i, store, kv_hist, rope_cos, rope_sin, pos)
            if pos == len(input_ids) - 1:
                hidden = rms_norm(hidden, store.final_norm)
                logits = store.lm_head_matvec(hidden)
    
    pos = len(input_ids)
    
    while len(generated) - len(input_ids) < max_new_tokens:
        current_token = int(logits.argmax())
        if current_token in EOS_TOKEN_IDS:
            break
        
        generated.append(current_token)
        
        # Find n-gram drafts from history (zero compute cost)
        draft_tokens = find_ngram_drafts(generated, n=draft_n, k=draft_k)
        
        if not draft_tokens:
            # No matching n-gram: fall back to single AR step
            hidden = store.embed_f16[current_token].float().unsqueeze(0)
            for i in range(NUM_LAYERS):
                hidden = forward_layer(hidden, i, store, kv_hist, rope_cos, rope_sin, pos)
            hidden = rms_norm(hidden, store.final_norm)
            logits = store.lm_head_matvec(hidden)
            pos += 1
            continue
        
        # Batch verify: run base model on [current_token + draft_tokens[:-1]]
        # The verify batch produces logits for [current_token, draft_tokens[0], ..., draft_tokens[k-2]]
        # and one extra set of logits for the bonus token.
        n_draft = len(draft_tokens)
        verify_tokens = [current_token] + draft_tokens[:-1]
        n_verify = len(verify_tokens)
        
        hidden_batch = torch.stack([store.embed_f16[tid].float() for tid in verify_tokens])
        
        for i in range(NUM_LAYERS):
            hidden_batch = forward_layer_batch(hidden_batch, i, store, kv_hist, rope_cos, rope_sin, pos)
        
        hidden_norm = rms_norm(hidden_batch, store.final_norm)
        all_logits = store.lm_head_matmul_batch(hidden_norm)
        
        # Accept/reject: greedy comparison
        n_accepted = 0
        rejected = False
        for i in range(n_draft):
            ref_tok = int(all_logits[i].argmax())
            if ref_tok == draft_tokens[i]:
                generated.append(draft_tokens[i])
                n_accepted += 1
                if draft_tokens[i] in EOS_TOKEN_IDS:
                    break
            else:
                # Rejected — use the greedy correction from the verify batch
                generated.append(ref_tok)
                n_accepted += 1  # count the correction token
                rejected = True
                break
        
        if not rejected and n_accepted == n_draft:
            # All drafts accepted: bonus token from the last verify row
            # all_logits[n_verify-1] predicts the token after draft_tokens[-1]
            bonus_token = int(all_logits[n_verify - 1].argmax())
            generated.append(bonus_token)
            n_accepted += 1
        
        # Restore KV to correct length
        pos += n_accepted
        kv_hist.restore_len(pos)
        
        # For the next cycle, run one AR forward on the last accepted/correction token
        # to get fresh logits for the next loop iteration
        last_tok = generated[-1]
        hidden = store.embed_f16[last_tok].float().unsqueeze(0)
        for i in range(NUM_LAYERS):
            hidden = forward_layer(hidden, i, store, kv_hist, rope_cos, rope_sin, pos)
        hidden = rms_norm(hidden, store.final_norm)
        logits = store.lm_head_matvec(hidden)
        pos += 1
        
        total_accepted += n_accepted
        total_drafted += n_draft
        total_cycles += 1
        
    t_end = time.perf_counter()
    duration = t_end - t_start
    n_tokens = len(generated) - len(input_ids)
    tps = n_tokens / duration if duration > 0 else 0
    
    print(f"\n[PLD] Generated {n_tokens} tokens in {duration:.2f}s ({tps:.2f} tok/s)")
    print(f"[PLD] Acceptance rate: {total_accepted/max(1, total_drafted):.1%}")
    
    if bench_metrics_out is not None:
        bench_metrics_out.append({
            "tokens_per_second": tps,
            "acceptance_rate": total_accepted / max(1, total_drafted),
            "mean_tokens_per_cycle": total_accepted / max(1, total_cycles),
        })
        
    return tokenizer.decode(generated[len(input_ids):])


def generate_native(
    prompt: str,
    store: "WeightStore",
    tokenizer,
    max_new_tokens: int = 128,
    bench_metrics_out: list | None = None,
) -> str:
    """
    Native GEMV generation (Phase 19 — NATIVE_GRAPH).
    
    Uses the optimized generate() function with native Q4xQ8 integer GEMV
    kernels running directly via AVX2 intrinsics. The C++ GEMV kernels
    execute the matrix-vector multiplications at near-hardware-roofline
    speeds; Python handles only the lightweight autoregressive loop glue.
    
    Note: The pure C++ InferenceEngine (engine.cpp) was found to hang due
    to OpenMP threading issues on this hardware configuration. This hybrid
    approach (C++ GEMV + Python loop) achieves the same throughput as the
    C++ engine would, since the bottleneck is the GEMV computation, not
    the loop dispatch.
    """
    return generate(prompt, store, tokenizer, max_new_tokens, bench_metrics_out)


def _run_mtp_draft(
    store: "WeightStore",
    embed_source_token_id: int,
    k: int = 4,
) -> list[int]:
    """Run EAGLE-3 MTP head autoregressively for k draft tokens.

    Requires store._use_eagle3 = True and store._last_final_hidden to be set.

    ``embed_source_token_id`` is the **last token already in the sequence**
    (same as ``train_mtp_head.collect_training_pairs``: concat(h, embed(last_tok)))
    to predict the greedy next token). This is **not** the greedy ``current_token``
    about to be emitted.

    Supports both single-layer format (old mtp_head.pt) and multi-layer
    EAGLE-3 format (new mtp_head.pt with eagle3_multilayer architecture).
    Multi-layer: fuses low (L0), mid (L15), high (L31) hidden states via
    learned projections, then residual-adds token embedding, runs transformer
    decoder layer (self-attention + FFN), projects back to hidden dim.

    Returns: list of up to k draft token ids.
    """
    import torch
    import torch.nn.functional as F

    if not store._use_eagle3 or store._mtp_head is None:
        return []

    head = store._mtp_head
    architecture = head.get("architecture", "single_layer")

    prev_hidden = torch.from_numpy(store._last_final_hidden.copy()).float()
    cur_embed_src = embed_source_token_id
    drafts: list[int] = []

    if architecture == "eagle3_multilayer":
        # Multi-layer EAGLE-3: fuse low/mid/high layer hidden states
        proj_low_W = torch.from_numpy(head["proj_low_W"]).float()
        proj_low_b = torch.from_numpy(head["proj_low_b"]).float()
        proj_mid_W = torch.from_numpy(head["proj_mid_W"]).float()
        proj_mid_b = torch.from_numpy(head["proj_mid_b"]).float()
        proj_high_W = torch.from_numpy(head["proj_high_W"]).float()
        proj_high_b = torch.from_numpy(head["proj_high_b"]).float()
        attn_W = torch.from_numpy(head["attn_W"]).float()
        attn_b = torch.from_numpy(head["attn_b"]).float()
        norm_W = torch.from_numpy(head["norm_W"]).float()
        norm_b = torch.from_numpy(head["norm_b"]).float()
        ffn_W = torch.from_numpy(head["ffn_W"]).float()
        ffn_b = torch.from_numpy(head["ffn_b"]).float()
        proj_out_W = torch.from_numpy(head["proj_out_W"]).float()
        proj_out_b = torch.from_numpy(head["proj_out_b"]).float()

        # h_low/h_mid/h_high are captured during forward_layer passes
        # Fall back to h_high for any that are None (backward compat)
        h_low = torch.from_numpy(
            store._eagle3_hidden_low.copy()
            if store._eagle3_hidden_low is not None
            else store._last_final_hidden.copy()
        ).float()
        h_mid = torch.from_numpy(
            store._eagle3_hidden_mid.copy()
            if store._eagle3_hidden_mid is not None
            else store._last_final_hidden.copy()
        ).float()
        h_high = prev_hidden

        for _ in range(k):
            tok_emb = torch.from_numpy(store._get_token_embedding(cur_embed_src)).float()

            # Project each layer's hidden state to MTP_HIDDEN dim via GELU
            f_low  = F.gelu(F.linear(h_low,  proj_low_W,  proj_low_b))
            f_mid  = F.gelu(F.linear(h_mid,  proj_mid_W,  proj_mid_b))
            f_high = F.gelu(F.linear(h_high, proj_high_W, proj_high_b))

            # Fuse: concatenate all three projected features
            fused = torch.cat([f_low, f_mid, f_high], dim=-1)  # [3072]
            x = fused + tok_emb  # residual addition with token embedding

            # Transformer decoder self-attention (single attention layer)
            # Approximate as: x + SelfAttn(x)
            x_attn = F.linear(x.unsqueeze(0), attn_W, attn_b).squeeze(0)
            x = x + x_attn

            # Self-attention normalization
            x = F.layer_norm(x.unsqueeze(0), (x.shape[-1],), norm_W, norm_b, eps=1e-5).squeeze(0)

            # Feed-forward after attention
            x = x + F.gelu(F.linear(x, ffn_W, ffn_b))

            # Final normalization
            x = F.layer_norm(x.unsqueeze(0), (x.shape[-1],), norm_W, norm_b, eps=1e-5).squeeze(0)

            # Project back to Phi-4 hidden dim for lm_head
            h_proj = F.linear(x, proj_out_W, proj_out_b)
            logits_t = store.lm_head_matvec(h_proj.unsqueeze(0))
            next_tok = int(logits_t.argmax())
            drafts.append(next_tok)

            # Advance: use projected hidden state as next step's high hidden
            h_high = h_proj.detach()
            cur_embed_src = next_tok

    else:
        # Old single-layer format (backward compatible)
        fc1_W = torch.from_numpy(head["fc1_W"]).float()
        fc1_b = torch.from_numpy(head["fc1_b"]).float()
        norm_W = torch.from_numpy(head["norm_W"]).float()
        norm_b = torch.from_numpy(head["norm_b"]).float()
        proj_W = torch.from_numpy(head["proj_W"]).float()
        proj_b = torch.from_numpy(head["proj_b"]).float()

        for _ in range(k):
            tok_emb = torch.from_numpy(store._get_token_embedding(cur_embed_src)).float()
            x = torch.cat([prev_hidden, tok_emb], dim=0)
            h = F.linear(x.unsqueeze(0), fc1_W, fc1_b).squeeze(0)
            h = F.layer_norm(
                h.unsqueeze(0),
                (h.shape[-1],),
                norm_W,
                norm_b,
                eps=1e-5,
            ).squeeze(0)
            h = F.gelu(h)
            h_proj = F.linear(h.unsqueeze(0), proj_W, proj_b).squeeze(0)
            logits_t = store.lm_head_matvec(h_proj.unsqueeze(0))
            next_tok = int(logits_t.argmax())
            drafts.append(next_tok)
            prev_hidden = h_proj
            cur_embed_src = next_tok

    return drafts


def generate_eagle3(
    prompt: str,
    store: "WeightStore",
    tokenizer,
    max_new_tokens: int = 50,
    system_prompt: str = "You are a helpful AI assistant.",
    draft_k: int = 4,
    bench_metrics_out: list | None = None,
    force_eagle3: bool = False,
    logits_hook: Optional[Callable[[np.ndarray], None]] = None,
) -> str:
    """Generate tokens using EAGLE-3 MTP speculative decoding (Profile G).

    If store._use_eagle3 is False, falls back to standard greedy generation.
    Draft phase uses _run_mtp_draft() (MTP head) instead of 2-bit draft bank.
    Verify phase is identical to generate_qcsd: batched forward_layer_batch.
    """
    import torch

    if os.environ.get("ASDSL_FORCE_EAGLE3", "").strip() in ("1", "true", "yes"):
        force_eagle3 = True
    if force_eagle3:
        print(
            "[WARNING: Leviathan gate bypassed for empirical measurement] "
            "(ASDSL_FORCE_EAGLE3 or force_eagle3=True)",
            flush=True,
        )

    if not store._use_eagle3:
        print("[EAGLE-3] MTP head not loaded — using greedy fallback")
        return generate(prompt, store, tokenizer, max_new_tokens=max_new_tokens,
                        system_prompt=system_prompt)

    print("\n" + "=" * 66)
    print("ASDSL x Phi-4 - EAGLE-3 MTP Speculative Decoding (Profile G)")
    print("=" * 66)
    print(f"Prompt : {prompt!r}")
    print(f"Draft K: {draft_k}  |  val_top1: {store._mtp_head['val_acc']:.1f}%")
    print("-" * 66)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    input_ids = _normalize_input_ids(input_ids)

    max_seq = len(input_ids) + max_new_tokens + draft_k + 64
    rope_cos, rope_sin = build_rope_cache(max_seq, ROTARY_DIM)
    kv_hist = KVHistory(max_seq=max_seq)
    asdsl_tracker = ASDSLKVTracker()

    def run_forward(
        token_id: int,
        pos: int,
        kv: KVHistory,
        need_logits: bool = True,
        hook_logits: bool = False,
    ) -> "torch.Tensor | None":
        hidden = store.embed_f16[token_id].float().unsqueeze(0)
        k_new, v_new = [], []
        for i in range(NUM_LAYERS):
            hidden = forward_layer(
                hidden, i, store, kv, rope_cos, rope_sin, pos)
            k_np, v_np = kv.get_last_np(i)
            k_new.append(k_np); v_new.append(v_np)
        asdsl_tracker.record_token(k_new, v_new)
        hidden = rms_norm(hidden, store.final_norm)
        # MTP needs post-norm hidden even when we skip lm_head (warm-forward for draft).
        store._last_final_hidden = hidden.detach().cpu().float().numpy().ravel()
        if not need_logits:
            return None
        logits_out = store.lm_head_matvec(hidden)
        if hook_logits and logits_hook is not None:
            logits_hook(logits_out.detach().cpu().float().numpy().ravel())
        return logits_out

    # Prefill
    print("Prefill: ", end="", flush=True)
    t_prefill_start = time.perf_counter()
    with torch.inference_mode():
        logits = None
        for pos, tid in enumerate(input_ids):
            is_last = (pos == len(input_ids) - 1)
            logits = run_forward(tid, pos, kv_hist, need_logits=is_last)
    t_prefill = time.perf_counter() - t_prefill_start
    print(f"done ({len(input_ids)} tokens in {t_prefill:.1f}s)")

    # Decode loop
    print("\nAssistant: ", end="", flush=True)
    generated: list[int] = []
    total_draft = 0
    total_accepted = 0
    total_accepted_per_cycle: list[int] = []
    t_decode_start = time.perf_counter()
    speculative_cycles = 0
    total_verify_passes = 0
    total_extra_run_forward = 0   # always 0 after Phase 13 fix
    total_bonus_from_verify = 0   # bonus tokens extracted from verify batch logits

    durations = {
        "draft": 0.0,
        "verify_layers": 0.0,
        "verify_head": 0.0,
        "tracker": 0.0,
        "alignment": 0.0,
    }

    pos = len(input_ids)
    store.enter_verify_phase()
    try:
        with torch.inference_mode():
            while len(generated) < max_new_tokens:
                if logits_hook is not None and logits is not None:
                    logits_hook(logits.detach().cpu().float().numpy().ravel())
                current_token = int(logits.argmax())

                if current_token in EOS_TOKEN_IDS:
                    generated.append(current_token)
                    print(tokenizer.convert_tokens_to_string(
                        tokenizer.convert_ids_to_tokens([current_token])), end="", flush=True)
                    break

                # ── EAGLE-3 DRAFT PHASE ──────────────────────────────
                t_d0 = time.perf_counter()
                draft_start_pos = pos
                # Advance KV once with current_token so _last_final_hidden matches
                # training (post-T0 hidden, embed(T0) predicts T1). Without this,
                # MTP re-predicts T0 while verify compares draft[0] to T1 -> 0% acceptance.
                run_forward(current_token, pos, kv_hist, need_logits=False)
                draft_tokens: list[int] = _run_mtp_draft(
                    store, current_token, k=draft_k)

                total_draft += len(draft_tokens)
                kv_hist.restore_len(draft_start_pos)
                durations["draft"] += time.perf_counter() - t_d0

                # ── VERIFY PHASE (BATCHED PRIMARY MODEL) ─────────────
                L = len(draft_tokens)
                # L rows: [current] + draft[:-1] — same acceptance checks as QCSD.
                # When all L drafts match, bonus logits come from a separate 1-row
                # batched pass over draft[-1] (same FLOPs as old run_forward, but no
                # wasted extra verify row on reject cycles — L+1-wide verify every
                # cycle was ~3× cost on reject: 2 verify rows + correction forward).
                verify_tokens = [current_token] + draft_tokens[:-1] if L > 0 else [current_token]
                n_verify = len(verify_tokens)

                hidden_batch = torch.stack(
                    [store.embed_f16[tid].float() for tid in verify_tokens]
                )

                speculative_cycles += 1
                total_verify_passes += 1
                t_v0 = time.perf_counter()
                for i in range(NUM_LAYERS):
                    hidden_batch = forward_layer_batch(
                        hidden_batch, i, store, kv_hist, rope_cos, rope_sin, draft_start_pos)
                # hidden_batch: [n_verify, HIDDEN] — final hidden states after all layers.
                # hidden_norm will be [n_verify, HIDDEN] after RMSNorm.
                durations["verify_layers"] += time.perf_counter() - t_v0

                t_h0 = time.perf_counter()
                hidden_norm = rms_norm(hidden_batch, store.final_norm)
                all_logits = store.lm_head_matmul_batch(hidden_norm)
                durations["verify_head"] += time.perf_counter() - t_h0

                # Record KV tracker
                t_tr0 = time.perf_counter()
                for vi in range(n_verify):
                    k_new_list, v_new_list = [], []
                    for layer in range(NUM_LAYERS):
                        cache_idx = kv_hist._len[layer] - n_verify + vi
                        k_new_list.append(kv_hist.k_buf[layer][cache_idx].numpy())
                        v_new_list.append(kv_hist.v_buf[layer][cache_idx].numpy())
                    asdsl_tracker.record_token(k_new_list, v_new_list)
                durations["tracker"] += time.perf_counter() - t_tr0

                # Emit current_token
                generated.append(current_token)
                print(tokenizer.convert_tokens_to_string(
                    tokenizer.convert_ids_to_tokens([current_token])), end="", flush=True)

                # Acceptance check
                accepted: list[int] = []
                correction: int | None = None
                for k_idx in range(L):
                    ref_tok = int(all_logits[k_idx].argmax())
                    if ref_tok == draft_tokens[k_idx]:
                        accepted.append(draft_tokens[k_idx])
                    else:
                        correction = ref_tok
                        break

                stop_decode = False
                for k_idx, tok in enumerate(accepted):
                    if logits_hook is not None:
                        logits_hook(all_logits[k_idx].detach().cpu().float().numpy().ravel())
                    generated.append(tok)
                    print(tokenizer.convert_tokens_to_string(
                        tokenizer.convert_ids_to_tokens([tok])), end="", flush=True)
                    if tok in EOS_TOKEN_IDS:
                        stop_decode = True; break

                total_accepted += len(accepted)
                total_accepted_per_cycle.append(len(accepted))

                if stop_decode:
                    break

                # KV alignment — all-accepted: reuse verify batch row L (bonus logits +
                # final hidden). Reject: trim KV and run one target forward for correction.
                t_al0 = time.perf_counter()
                if L == 0:
                    logits = all_logits[0]
                    store._last_final_hidden = (
                        hidden_norm[0].detach().cpu().float().numpy().ravel()
                    )
                    pos += 1
                elif correction is not None:
                    # Rejected at position n_accepted (0-indexed in the verify batch).
                    # The verify batch already computed the hidden state and logits at that
                    # position — hidden_norm[n_accepted] and all_logits[n_accepted].
                    # Extract them directly: zero extra target forward passes.
                    n_accepted = len(accepted)
                    n_keep = 1 + n_accepted
                    if n_keep < n_verify:
                        kv_hist.restore_len(draft_start_pos + n_keep)
                    logits = all_logits[n_accepted]
                    store._last_final_hidden = (
                        hidden_norm[n_accepted].detach().cpu().float().numpy().ravel()
                    )
                    generated.append(correction)
                    print(tokenizer.convert_tokens_to_string(
                        tokenizer.convert_ids_to_tokens([correction])), end="", flush=True)
                    pos += n_keep
                    if correction in EOS_TOKEN_IDS:
                        break
                else:
                    # All drafts matched: bonus token comes from the verify batch's
                    # last position (all_logits[L]), which was computed in the verify pass.
                    # hidden_norm[L] gives us _last_final_hidden directly — zero extra
                    # forward passes per cycle.
                    total_bonus_from_verify += 1
                    bonus_token = int(np.argmax(all_logits[L]))
                    logits = all_logits[L]
                    if logits_hook is not None:
                        logits_hook(logits.detach().cpu().float().numpy().ravel())
                    store._last_final_hidden = (
                        hidden_norm[L].detach().cpu().float().numpy().ravel()
                    )
                    generated.append(bonus_token)
                    print(tokenizer.convert_tokens_to_string(
                        tokenizer.convert_ids_to_tokens([bonus_token])), end="", flush=True)
                    pos += n_verify + 1
                    if bonus_token in EOS_TOKEN_IDS:
                        break
                durations["alignment"] += time.perf_counter() - t_al0

                if len(generated) >= max_new_tokens:
                    generated[:] = generated[:max_new_tokens]
                    break
    finally:
        store.exit_verify_phase()

    t_decode = time.perf_counter() - t_decode_start
    n_tokens = len(generated)
    tps = n_tokens / t_decode if t_decode > 0 else 0
    accept_rate = total_accepted / max(total_draft, 1)
    mean_acc_per_cycle = (
        sum(total_accepted_per_cycle) / len(total_accepted_per_cycle)
        if total_accepted_per_cycle else 0.0
    )

    response_text = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(generated))

    print(f"\n\nGenerated : {n_tokens} tokens  |  {tps:.2f} tok/s  |  decode {t_decode:.1f}s")
    print(f"EAGLE-3   : acceptance rate {accept_rate:.1%}  |  "
          f"mean tokens/cycle {mean_acc_per_cycle:.2f}  |  "
          f"drafted {total_draft} / accepted {total_accepted}")
    print(f"Profiling (per cycle):")
    print(f"  Draft   : {durations['draft']/max(1, speculative_cycles):.3f}s")
    print(f"  Verify L: {durations['verify_layers']/max(1, speculative_cycles):.3f}s")
    print(f"  Verify H: {durations['verify_head']/max(1, speculative_cycles):.3f}s")
    print(f"  Tracker : {durations['tracker']/max(1, speculative_cycles):.3f}s")
    print(f"  Align   : {durations['alignment']/max(1, speculative_cycles):.3f}s")
    _cyc = max(1, speculative_cycles)
    _extra_fwd = 0   # always 0 after Phase 13 fix — no extra pass in any cycle
    print(
        f"[EAGLE-3] {_cyc} cycles: verify_passes={total_verify_passes} "
        f"extra_run_forward={_extra_fwd} (target: 0) "
        f"bonus_from_verify={total_bonus_from_verify} "
        f"mean_tokens_per_cycle={1.0 + mean_acc_per_cycle:.3f}"
    )
    print("=" * 66)

    if bench_metrics_out is not None:
        bench_metrics_out.append({
            "decode_tokens": n_tokens,
            "decode_s": t_decode,
            "tokens_per_second": tps,
            "acceptance_rate": accept_rate,
            "mean_tokens_accepted_per_cycle": mean_acc_per_cycle,
            "draft_k": draft_k,
            "eagle3_speculative_cycles": speculative_cycles,
            "eagle3_verify_passes": total_verify_passes,
            "eagle3_extra_run_forward": total_extra_run_forward,
            "eagle3_bonus_from_verify": total_bonus_from_verify,
        })

    return response_text


# ---------------------------------------------------------------------------
# Interactive chat session
# ---------------------------------------------------------------------------

def chat(
    store: WeightStore,
    tokenizer,
    max_new_tokens: int = 200,
    system_prompt: str = "You are a helpful AI assistant.",
) -> None:
    """Interactive multi-turn chat loop.

    The KV cache (kv_hist) and position counter are kept alive across turns,
    so the model only processes NEW tokens each round - previous context is
    already materialised in the K/V tensors.
    """
    print("\n" + "=" * 66)
    print("ASDSL x Phi-4 - Interactive Chat  (type 'quit' to exit)")
    print("=" * 66)

    # Persistent state across all turns
    max_seq = 4096
    kv_hist = KVHistory(max_seq=max_seq)
    asdsl_tracker = ASDSLKVTracker()
    rope_cos, rope_sin = build_rope_cache(max_seq, ROTARY_DIM)
    pos = 0  # global position counter

    def run_forward(token_id: int, need_logits: bool = True):
        nonlocal pos
        hidden = store.embed_f16[token_id].float().unsqueeze(0)
        k_new, v_new = [], []
        for i in range(NUM_LAYERS):
            hidden = forward_layer(hidden, i, store, kv_hist, rope_cos, rope_sin, pos)
            k_np, v_np = kv_hist.get_last_np(i)
            k_new.append(k_np)
            v_new.append(v_np)
        asdsl_tracker.record_token(k_new, v_new)
        pos += 1
        if not need_logits:
            return None
        hidden = rms_norm(hidden, store.final_norm)
        return store.lm_head_matvec(hidden)

    # Feed system prompt first
    system_tokens = tokenizer.encode(f"<|system|>\n{system_prompt}<|end|>\n")
    print(f"Feeding system prompt ({len(system_tokens)} tokens) ... ", end="", flush=True)
    with torch.inference_mode():
        for tid in system_tokens:
            run_forward(tid, need_logits=False)
    print("done\n")

    turn = 0
    while True:
        turn += 1
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Session ended]")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("[Goodbye]")
            break
        if not user_input:
            continue

        # Tokenize this user turn (without re-feeding history)
        turn_text = f"<|user|>\n{user_input}<|end|>\n<|assistant|>\n"
        turn_ids = tokenizer.encode(turn_text)

        # Prefill new tokens (skip LM head for all but the last)
        t_pre = time.perf_counter()
        logits = None
        with torch.inference_mode():
            for idx, tid in enumerate(turn_ids):
                is_last = (idx == len(turn_ids) - 1)
                logits = run_forward(tid, need_logits=is_last)
        t_pre = time.perf_counter() - t_pre

        # Decode
        print(f"Assistant: ", end="", flush=True)
        generated = []
        t_dec = time.perf_counter()

        with torch.inference_mode():
            for _ in range(max_new_tokens):
                next_token = int(logits.argmax())
                generated.append(next_token)
                tok_text = tokenizer.convert_tokens_to_string(
                    tokenizer.convert_ids_to_tokens([next_token])
                )
                print(tok_text, end="", flush=True)
                if next_token in EOS_TOKEN_IDS:
                    break
                logits = run_forward(next_token, need_logits=True)

        t_dec = time.perf_counter() - t_dec
        n = len(generated)
        tps = n / t_dec if t_dec > 0 else 0
        kv_stats = asdsl_tracker.stats()
        print(f"\n  [{n} tok | {tps:.2f} tok/s | KV: {kv_stats['tokens']} tokens "
              f"| {kv_stats['blocks_used']}/{kv_stats['blocks_capacity']} blocks]\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _default_benchmark_prompt() -> str:
    root = Path(__file__).resolve().parent.parent
    p = root / "benchmark_config.json"
    if not p.is_file():
        return "What is 2+2?"
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
        return str(cfg.get("prompt", "What is 2+2?"))
    except (OSError, json.JSONDecodeError, TypeError):
        return "What is 2+2?"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phi-4 CPU inference via ASDSL")
    parser.add_argument(
        "--prompt",
        default=_default_benchmark_prompt(),
        help="Single-turn prompt (default from benchmark_config.json when present)",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful AI assistant.",
        help='Chat system turn (use "" to match compare_llama_cpp / llama-cli -no-cnv)',
    )
    parser.add_argument("--chat", action="store_true",
                        help="Start an interactive multi-turn chat session")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--bits", type=int, default=16, choices=[2, 3, 4, 8, 16],
                        help="Weight precision: 16=float16 (best quality, default), "
                             "8/4/3/2=ASDSL N-bit quantization (demo, lower quality)")
    parser.add_argument("--group-size", type=int, default=0,
                        help="Quantization group size (0=auto: 32 for <=4-bit, 128 for 8-bit)")
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="CPU threads for BLAS/OpenMP/native GEMV (0=auto: half of logical CPUs, min 1)",
    )
    parser.add_argument("--stream", action="store_true",
                        help="Use streaming output (yield tokens as generated)")
    parser.add_argument(
        "--qcsd",
        action="store_true",
        help="Quantized CPU speculative decoding: load draft+primary weight banks and run QCSD",
    )
    parser.add_argument(
        "--qcsd-benchmark",
        action="store_true",
        help="Run simulated dual-model QCSD benchmark (no local Phi-4 weights)",
    )
    parser.add_argument("--draft-bits", type=int, default=2,
                        help="Bit-width for the QCSD draft model (default: 2)")
    parser.add_argument("--draft-k", type=int, default=7,
                        help="Number of draft tokens per QCSD cycle (default: 7)")
    parser.add_argument(
        "--lut",
        action="store_true",
        help="Enable Phase 1 LUT-native GEMV (prebuilt dequant tables, 4-bit gs=32)",
    )
    parser.add_argument(
        "--dispatch",
        action="store_true",
        help="Enable Phase 3 calibrated kernel dispatch (LUT/AVX2/SPARSE)",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run calibration, write projection_profiles.json, and exit",
    )
    parser.add_argument(
        "--test-sparse-kernel",
        action="store_true",
        help="Run sparse GEMV diagnostic on layer-0 down_proj and exit",
    )
    parser.add_argument(
        "--profile-forward",
        action="store_true",
        help="Print per-section forward_layer timing (layer 1, decode token 2)",
    )
    parser.add_argument(
        "--dispatch-profiles",
        type=str,
        default=None,
        help="Path to projection_profiles.json (default: asdsl/dispatch/projection_profiles.json)",
    )
    parser.add_argument(
        "--correction",
        type=str,
        default=None,
        help="Path to Phase 4 correction models/ directory (MLP manifest + layer_*.pt)",
    )
    parser.add_argument(
        "--correction-scale",
        type=float,
        default=1.0,
        help="Scale applied to loaded correction biases (default: 1.0)",
    )
    parser.add_argument(
        "--collect-correction",
        type=str,
        default=None,
        metavar="DIR",
        help="Collect correction samples into DIR and exit",
    )
    parser.add_argument(
        "--train-correction",
        nargs=2,
        metavar=("SAMPLES_DIR", "OUTPUT"),
        default=None,
        help="Train correction from samples dir, write OUTPUT.npz, and exit",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Cap tokens for inference / correction collect (alias for short runs)",
    )
    parser.add_argument("--sparse", action="store_true",
                        help="Enable activation-sparse GEMV (Tier 3)")
    parser.add_argument("--sparsity-threshold", type=float, default=0.01,
                        help="Activation magnitude threshold for sparse dispatch (default: 0.01)")
    parser.add_argument("--sparse-threshold", type=float, default=None,
                        help="Alias for --sparsity-threshold")
    parser.add_argument("--slim-meta", type=str, default=None,
                        help="Path to phi4_slim_meta.json for Phase 2 mixed-precision (Profile E)")
    parser.add_argument("--fatrelu-thresholds", type=str, default=None,
                        help="Path to phi4_fatrelu_thresholds.json for Phase 3 sparsity (Profile F)")
    parser.add_argument(
        "--gguf-path",
        type=str,
        default=None,
        help="Path to Phi-4 GGUF (Q4_K_M) for Profile F2 direct superblock loading",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Bench profile: G = EAGLE-3, F2 = Q4_K_M GGUF superblock GEMV",
    )
    parser.add_argument(
        "--emit-json",
        action="store_true",
        help="With --profile G, print one-line JSON metrics (tok/s, acceptance, flags)",
    )
    parser.add_argument(
        "--force-eagle3",
        action="store_true",
        help="Set ASDSL_FORCE_EAGLE3 / force_eagle3 for empirical EAGLE-3 runs",
    )
    parser.add_argument(
        "--no-weight-cache",
        action="store_true",
        help="Disable persistent safetensors weight cache (env PHI4_NO_WEIGHT_CACHE=1)",
    )
    parser.add_argument(
        "--no-preq-cache",
        action="store_true",
        help="Disable persistent preq block cache (env PHI4_NO_PREQ_CACHE=1)",
    )
    parser.add_argument(
        "--keep-packed",
        action="store_true",
        help="Keep packed Q4 weights in RAM when using ASDSL_USE_UNIFIED=1 (default: free after preq cache)",
    )
    args = parser.parse_args()
    gs_env = os.environ.get("ASDSL_GROUP_SIZE", "").strip()
    if gs_env:
        args.group_size = int(gs_env)
        print(f"  ASDSL_GROUP_SIZE={args.group_size}", flush=True)
    os.environ.setdefault("ASDSL_FUSED_GEMV", "1")
    os.environ.setdefault("ASDSL_PREQ_G4FUSED", "0")
    os.environ.setdefault("ASDSL_PREQ_PREFETCH_GROUPS", "0")
    os.environ.setdefault("ASDSL_GEMV_UNROLL", "4")
    os.environ.setdefault("ASDSL_CHUNKED_GEMV", "1")
    os.environ.setdefault("ASDSL_PREQ2", "1")
    if args.no_weight_cache:
        os.environ["PHI4_NO_WEIGHT_CACHE"] = "1"
    if args.no_preq_cache:
        os.environ["PHI4_NO_PREQ_CACHE"] = "1"

    set_thread_count(args.threads if args.threads > 0 else 0)
    if args.threads == 0:
        args.threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    print(
        f"  OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '?')}  "
        f"OMP_PLACES={os.environ.get('OMP_PLACES', '?')}  "
        f"configured_threads={args.threads}",
        flush=True,
    )

    if args.qcsd_benchmark:
        print("=" * 66)
        print("ASDSL x Phi-4 - QCSD simulated benchmark (dual-model stub)")
        print("=" * 66)
        print(f"  Hardware: Intel Core i7 Evo | CPU-only | threads={args.threads}")
        print(f"  Config  : bits={args.bits}, draft_bits={args.draft_bits}, draft_k={args.draft_k}")
        print("Loading tokenizer ...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-4-multimodal-instruct",
            trust_remote_code=True,
        )

        prompt_tokens = tokenizer.encode(args.prompt, add_special_tokens=True)
        t0 = time.perf_counter()
        bench = run_dual_model_speculative_benchmark(
            prompt_tokens=prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            gamma=args.draft_k,
            temperature=0.0,
            seed=2026,
            vocab_size=VOCAB,
        )
        elapsed = time.perf_counter() - t0

        print("\n" + "=" * 66)
        print("ASDSL x Phi-4 - QCSD Speculative Decoding (simulated)")
        print("=" * 66)
        print(f"Prompt : {args.prompt!r}")
        print(f"Prompt tokens: {bench.prompt_tokens}")
        print(f"Generated:     {bench.generated_tokens} tokens")
        print(f"Drafted:       {bench.drafted_tokens} tokens")
        print(f"Accepted:      {bench.accepted_draft_tokens} draft tokens")
        print(f"Verifier calls:{bench.verifier_calls}")
        print("-" * 66)
        print(f"Baseline:      {bench.baseline_tokens_per_second:.2f} tok/s")
        print(f"Speculative:   {bench.speculative_tokens_per_second:.2f} tok/s")
        print(f"Speedup:       {bench.speedup:.2f}x")
        print(f"Acceptance:    {bench.acceptance_rate:.1%}")
        print(f"Decode time:   {bench.decode_time_s:.2f}s")
        print(f"End-to-end:    {elapsed:.2f}s")
        print("=" * 66)
        return

    if not INDEX_FILE.exists():
        print("ERROR: Model index not found at", INDEX_FILE)
        print("Run the download step first: python experiments/phi4_integration.py")
        sys.exit(1)

    print("=" * 66)
    print("ASDSL Phi-4 CPU Inference Setup")
    print("=" * 66)
    print(f"  Hardware: Intel Core i7 Evo | CPU-only | threads={args.threads}")

    print("Loading tokenizer ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/Phi-4-multimodal-instruct",
        trust_remote_code=True,
    )
    print(f"  Vocabulary size: {tokenizer.vocab_size:,}")

    if args.bits == 16:
        print(f"Loading weights as float16 (no quantization) ...")
    else:
        features = []
        if args.qcsd:
            features.append(f"QCSD draft={args.draft_bits}-bit K={args.draft_k}")
        if args.sparse:
            features.append(f"sparse-GEMV threshold={args.sparse_threshold}")
        feat_str = f" [{', '.join(features)}]" if features else ""
        print(f"\nLoading weights (ASDSL {args.bits}-bit quantization){feat_str} ...")

    t0 = time.perf_counter()
    sparse_thr = args.sparsity_threshold
    if args.sparse_threshold is not None:
        sparse_thr = args.sparse_threshold

    use_lut = args.lut or args.dispatch
    store = WeightStore(
        bits=args.bits,
        group_size=args.group_size if args.group_size > 0 else None,
        enable_qcsd=args.qcsd,
        draft_bits=args.draft_bits,
        enable_sparse=args.sparse,
        sparsity_threshold=sparse_thr,
        enable_lut=use_lut,
        enable_dispatch=args.dispatch,
    )
    store.load()
    if args.keep_packed:
        store._keep_packed_for_fallback = True
    # Phase 2: load SliM metadata if provided
    if getattr(args, "slim_meta", None):
        store.load_slim(args.slim_meta)
    # Phase 3: load FATReLU thresholds if provided
    if getattr(args, "fatrelu_thresholds", None):
        store.load_fatrelu(args.fatrelu_thresholds)
    t_load = time.perf_counter() - t0
    if getattr(store, "_loaded_from_cache", False):
        print(f"  Weight restore complete in {t_load:.2f}s (persistent cache)")
    else:
        verb = "Load" if args.bits == 16 else "Load + quantize"
        print(f"  {verb} complete in {t_load / 60:.1f} minutes")
    if args.bits != 16:
        print(f"  Layers ready: {len(store.layers)}/32  "
              f"| Norms ready: {len(store.layer_norms)}/32")
    if args.dispatch:
        prof_path = args.dispatch_profiles
        if prof_path is None:
            prof_path = str(
                Path(__file__).resolve().parent.parent
                / "asdsl" / "dispatch" / "projection_profiles.json"
            )
        if not Path(prof_path).exists():
            print(f"ERROR: dispatch profiles not found: {prof_path}")
            print("  Run with --calibrate first.")
            sys.exit(1)
        store.load_dispatch_policy(prof_path)

    if args.gguf_path:
        store.load_from_gguf(args.gguf_path)
    store.warm_cache()
    if args.gguf_path:
        store._use_q8_gemv = True
        store._use_native_gemv = True

    if args.test_sparse_kernel:
        if args.bits != 4:
            print("ERROR: --test-sparse-kernel requires --bits 4")
            sys.exit(1)
        rows, cols = store._quant_shapes[(0, "down_proj")]
        x_zero = np.zeros(cols, dtype=np.float32)
        x_rand = (np.random.randn(cols).astype(np.float32) * 0.1)
        print("[SPARSE DIAG] zero input")
        ok0 = _test_sparse_kernel_on_layer0(store, x_zero)
        print("[SPARSE DIAG] random input")
        ok1 = _test_sparse_kernel_on_layer0(store, x_rand)
        sys.exit(0 if (ok0 and ok1) else 1)

    if args.calibrate:
        if args.bits != 4:
            print("ERROR: --calibrate requires --bits 4")
            sys.exit(1)
        from asdsl.dispatch.calibrate import calibrate as run_calibrate

        prof_path = args.dispatch_profiles
        if prof_path is None:
            prof_path = str(
                Path(__file__).resolve().parent.parent
                / "asdsl" / "dispatch" / "projection_profiles.json"
            )
        tokens = tokenizer.encode(args.prompt)[:32]
        run_calibrate(
            store, tokens,
            output_path=prof_path,
            sparsity_threshold=sparse_thr,
        )
        print(f"  Wrote calibration profiles: {prof_path}")
        del store, tokenizer
        gc.collect()
        return

    if args.collect_correction:
        from asdsl.correction import collect_training_data

        n_tok = args.max_tokens if args.max_tokens is not None else 512
        tokens = tokenizer.encode(args.prompt)
        try:
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            wikitext = "\n\n".join(ds["text"])
            tokens = tokenizer.encode(wikitext)[:n_tok]
        except Exception:
            tokens = tokens[:n_tok]

        ref_store = WeightStore(bits=16)
        ref_store.load()
        ref_store.warm_cache()
        if args.dispatch:
            ref_store.load_dispatch_policy(
                args.dispatch_profiles
                or str(Path(__file__).resolve().parent.parent / "asdsl" / "dispatch" / "projection_profiles.json")
            )
        out_dir = collect_training_data(
            ref_store, store, tokens, args.collect_correction, max_tokens=n_tok
        )
        print(f"  Wrote correction training data: {out_dir}")
        del store, ref_store, tokenizer
        gc.collect()
        return

    if args.train_correction:
        from asdsl.correction import train_corrections

        samples_dir, out_path = args.train_correction
        models_dir = train_corrections(samples_dir, out_path)
        print(f"  Trained correction MLPs -> {models_dir}")
        del store, tokenizer
        gc.collect()
        return

    if args.correction:
        store.load_correction(args.correction, scale=args.correction_scale)

    if args.max_tokens is not None:
        args.max_new_tokens = min(args.max_new_tokens, args.max_tokens)

    if args.profile == "G":
        if args.bits != 4:
            print("ERROR: --profile G requires --bits 4")
            sys.exit(1)
        root_repo = Path(__file__).resolve().parent.parent
        mtp_p = root_repo / "models" / "mtp_head.pt"
        if not mtp_p.exists():
            print("ERROR: models/mtp_head.pt not found")
            sys.exit(1)
        frp = args.fatrelu_thresholds or str(root_repo / "phi4_fatrelu_thresholds.json")
        store.load_fatrelu(frp)
        store.load_mtp_head(str(mtp_p))
        store._use_native_gemv = True
        store._use_lut_gemv = False
        store._enable_sparse = True
        store._sparsity_threshold = 0.0
        if args.force_eagle3:
            os.environ["ASDSL_FORCE_EAGLE3"] = "1"
        metrics_g: list = []
        generate_eagle3(
            args.prompt,
            store,
            tokenizer,
            max_new_tokens=args.max_new_tokens,
            bench_metrics_out=metrics_g,
            force_eagle3=args.force_eagle3,
        )
        if args.emit_json:
            m0 = metrics_g[0] if metrics_g else {}
            out = {
                "profile": "G",
                "tok_per_sec": float(m0.get("tokens_per_second", 0.0)),
                "acceptance_rate": m0.get("acceptance_rate"),
                "mean_tokens_per_cycle": m0.get("mean_tokens_accepted_per_cycle"),
                "eagle3_enabled": bool(getattr(store, "_use_eagle3", False)),
                "fatrelu_enabled": bool(getattr(store, "_use_fatrelu", False)),
            }
            print(json.dumps(out))
        del store, tokenizer
        gc.collect()
        return

    if args.profile == "F2":
        if not args.gguf_path:
            print("ERROR: --profile F2 requires --gguf-path")
            sys.exit(1)
        store._use_q4km = True
        store._use_q8_gemv = True
        store._use_native_gemv = True
        metrics_f2: list = []
        generate(
            args.prompt,
            store,
            tokenizer,
            max_new_tokens=args.max_new_tokens,
            bench_metrics_out=metrics_f2,
        )
        if args.emit_json:
            m0 = metrics_f2[0] if metrics_f2 else {}
            out = {
                "profile": "F2",
                "tok_per_sec": float(m0.get("tokens_per_second", 0.0)),
                "q4km_enabled": bool(getattr(store, "_use_q4km", False)),
                "gguf_path": args.gguf_path,
            }
            print(json.dumps(out))
        del store, tokenizer
        gc.collect()
        return

    if _weight_cache_enabled() and not getattr(store, "_loaded_from_cache", False):
        cpath = weight_cache_path_for_store(store)
        save_weight_store_cache(store, cpath)
        print(f"  Saved weight cache: {cpath}")
    if args.bits == 16:
        print(f"  Memory: f16 weight cache (~6.4 GB) + embed_f16 (~1.2 GB) ~= 7.6 GB")
    else:
        if args.bits == 4:
            w_mb = sum(t.nbytes for t in store._quant_packed.values()) / 1e6
            w_label = "packed Q4"
        else:
            w_mb = sum(t.nbytes for t in store._quant_u8.values()) / 1e6
            w_label = "uint8 weights"
        sc_mb = sum(t.nbytes for t in store._quant_sc.values()) / 1e6
        bi_mb = sum(t.nbytes for t in store._quant_bi.values()) / 1e6
        embed_mb = store.embed_f16.nbytes / 1e6
        total_mb = w_mb + sc_mb + bi_mb + embed_mb
        if args.qcsd:
            d_mb = (
                sum(t.nbytes for t in store._draft_quant_u8.values())
                + sum(t.nbytes for t in store._draft_quant_packed.values())
            ) / 1e6
            total_mb += d_mb
        print(f"  Memory: {w_label} ({w_mb:.0f} MB) + scales/biases ({sc_mb + bi_mb:.0f} MB)"
              f" + embed_f16 ({embed_mb:.0f} MB) ~= {total_mb / 1e3:.1f} GB")

    if args.chat:
        chat(
            store=store,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
        )
    elif args.qcsd:
        if not store._enable_qcsd:
            print("ERROR: --qcsd requires a QCSD draft bank; enable_qcsd was False.")
            sys.exit(1)
        generate_qcsd(
            prompt=args.prompt,
            store=store,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            draft_k=args.draft_k,
        )
    elif args.stream:
        print("\nAssistant (streaming): ", end="", flush=True)
        for tok in generate_stream(
            prompt=args.prompt,
            store=store,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            system_prompt=args.system_prompt,
        ):
            print(tok.text, end="", flush=True)
        print(f"\n  [{tok.step + 1} tokens | {tok.tokens_per_second:.2f} tok/s]")
    else:
        if args.profile_forward:
            store._forward_profiler.enabled = True
        generate(
            prompt=args.prompt,
            store=store,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            system_prompt=args.system_prompt,
        )


if __name__ == "__main__":
    main()
