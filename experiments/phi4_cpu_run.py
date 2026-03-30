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


def _configure_cpu_torch_runtime() -> None:
    """Flush FP32 subnormals to zero on CPU (avoids 100× slowdowns in matmul/mv)."""
    if hasattr(torch, "set_flush_denormal"):
        try:
            torch.set_flush_denormal(True)
        except Exception:
            pass


_configure_cpu_torch_runtime()


def set_thread_count(n: int) -> None:
    """Set CPU threads for NumPy/BLAS/PyTorch and ASDSL native OpenMP GEMV.

    ``n <= 0`` (auto): use half of ``os.cpu_count()`` (minimum 1). That tends to map
    closer to physical cores on typical SMT CPUs and reduces hyperthreading
    contention on bandwidth-bound native GEMV while still letting explicit
    ``--threads`` override for tuning.
    """
    if n <= 0:
        logical = os.cpu_count() or 4
        n = max(1, logical // 2)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n)
    torch.set_num_threads(n)
    _configure_cpu_torch_runtime()
    
    # Force process affinity to n cores (if they are 0..n-1, usually P-cores)
    try:
        import psutil
        p = psutil.Process()
        # On this 4P+8E machine, 0-7 are the hyperthreaded P-cores
        p.cpu_affinity(list(range(min(n, 16))))
    except ImportError:
        pass
        
    try:
        from asdsl.kernels import _native_gemv as _ng

        if bool(getattr(_ng, "has_openmp", False)) and hasattr(_ng, "set_num_threads"):
            _ng.set_num_threads(int(n))
    except ImportError:
        pass


ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models" / "phi4-multimodal-instruct"
INDEX_FILE = MODEL_DIR / "model.safetensors.index.json"

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
            
            store._quant_packed_np[(i, nm)] = store._quant_packed[(i, nm)].numpy().ravel() if (i, nm) in store._quant_packed else None
            store._quant_sc_np[(i, nm)] = store._quant_sc[(i, nm)].float().numpy().ravel()
            store._quant_bi_np[(i, nm)] = store._quant_bi[(i, nm)].float().numpy().ravel()

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
                
                store._draft_quant_packed_np[(i, nm)] = store._draft_quant_packed[(i, nm)].numpy().ravel() if (i, nm) in store._draft_quant_packed else None
                store._draft_quant_sc_np[(i, nm)] = store._draft_quant_sc[(i, nm)].float().numpy().ravel()
                store._draft_quant_bi_np[(i, nm)] = store._draft_quant_bi[(i, nm)].float().numpy().ravel()
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


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Root-mean-square layer normalisation."""
    rms = x.pow(2).mean(-1, keepdim=True).add(RMS_EPS).sqrt()
    return (x / rms) * weight


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
                 enable_sparse: bool = False, sparsity_threshold: float = 0.01):
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


        # Native LUT/GEMV fast path
        self._use_native_gemv = False
        # Phase 1: vpshufb LUT kernel (Profile D)
        self._use_lut_gemv = False
        # Phase 2: SliM mixed-precision dispatch (Profile E)
        self._use_slim = False
        self._slim_meta = None          # loaded from phi4_slim_meta.json
        self._slim_npz = None           # loaded from phi4_slim_meta.npz
        self._repacked_layers: dict = {}  # lazy per-layer repacked buffers
        # Phase 3: FATReLU sparsity (Profile F)
        self._use_fatrelu = False
        self._fatrelu_thresholds: dict[int, float] = {}  # per-layer tau

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

    def enter_verify_phase(self) -> None:
        """Enable temporary dequantization caching for the speculative verify phase."""
        self._in_verify_phase = True

    def exit_verify_phase(self) -> None:
        """Clear and disable the temporary dequantization cache."""
        self._dequant_cache.clear()
        self._in_verify_phase = False

    # ------------------------------------------------------------------
    def load(self) -> None:
        self._loaded_from_cache = False
        if _weight_cache_enabled():
            cpath = weight_cache_path_for_store(self)
            if try_restore_weight_cache(self, cpath):
                nproj = NUM_LAYERS * 4
                print(
                    f"  Restored {nproj} projection weights from cache "
                    f"({cpath.name}, mmap CPU) — skipping shard read/quantize"
                )
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
                        if self.bits <= 3:
                            from asdsl.quantization.core import quantize_weights_with_outliers
                            # 2-bit: 3.0σ with 0.5% cap balances PPL improvement vs
                            # outlier correction overhead; 3-bit uses milder 3.5σ
                            sigma = 3.0 if self.bits == 2 else 3.5
                            cap = 0.005
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
        x_np = x.detach().cpu().float().contiguous().numpy().ravel()
        
        out_np = gemv_q4_packed(
            w_np, x_np, sc_np, bi_np, rows, cols, gs, use_lut=self._use_lut_gemv
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

    def _matvec_native_gemv(self, layer_idx: int, name: str, x: torch.Tensor,
                            use_draft: bool = False) -> torch.Tensor:
        """AVX2 GEMV fast path for 4-bit packed, 8-bit, 3-bit, and 2-bit weights."""
        key = (layer_idx, name)

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

    def load_fatrelu(self, thresholds_path) -> None:
        """Load FATReLU thresholds from phi4_fatrelu_thresholds.json.

        Sets self._use_fatrelu = True and populates self._fatrelu_thresholds.
        Also triggers transposed down_proj loading (Prerequisite B).
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

        n = len(self._fatrelu_thresholds)
        mean_tau = sum(self._fatrelu_thresholds.values()) / max(n, 1)
        sparsity = data.get("target_sparsity", 0.85)
        print(f"[FATReLU] Loaded: {n} layers, mean tau={mean_tau:.4f}, "
              f"target sparsity={sparsity:.0%}")

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

    def matvec(self, layer_idx: int, name: str, x: torch.Tensor,
               use_draft: bool = False) -> torch.Tensor:
        """Bandwidth-efficient matrix-vector product: y = W @ x."""
        if self.bits == 16:
            return self._matvec_f16(layer_idx, name, x)
        # Phase 2: SliM mixed-precision path
        if self._use_slim and not use_draft and self.bits == 4:
            return self._matvec_slim(layer_idx, name, x)
        if self._use_native_gemv or use_draft:
            return self._matvec_native_gemv(layer_idx, name, x, use_draft=use_draft)
        return self._matvec_quant(layer_idx, name, x)

    def _get_token_embedding(self, token_id: int) -> np.ndarray:
        """Returns the float32 embedding vector for a token as a numpy array."""
        return self.embed_f16[token_id].float().cpu().numpy().ravel()

    def load_mtp_head(self, path: str) -> None:
        """Load trained MTP head for EAGLE-3 speculative decoding (Profile G)."""
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
            
            # Map checkpoint keys to names used in _run_mtp_draft
            self._mtp_head = {
                "fc1_W": ckpt["fc1_weight"].float().numpy(),
                "fc1_b": ckpt["fc1_bias"].float().numpy(),
                "norm_W": ckpt["norm_weight"].float().numpy(),
                "norm_b": ckpt["norm_bias"].float().numpy(),
                "proj_W": ckpt["proj_weight"].float().numpy(),
                "proj_b": ckpt["proj_bias"].float().numpy(),
                "val_acc": val_acc
            }
            self._use_eagle3 = True
            print(f"[EAGLE-3] Loaded MTP head from {path} (val_acc={val_acc:.1f}%)")
        except Exception as e:
            print(f"[EAGLE-3] Error loading MTP head from {path}: {e}")
            self._use_eagle3 = False

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
        out = self.lm_head_matmul_batch(hidden)
        if out.shape[0] == 1:
            return out.squeeze(0)
        return out

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

        if getattr(self, "_loaded_from_cache", False):
            total = NUM_LAYERS * 4
            print(
                f"  Warm-cache: skipped (restored {total} projections from safetensors cache)"
            )
            kernel_labels = {4: "Q4 packed (gemv_q4_packed)", 8: "Q8", 3: "Q3", 2: "Q2"}
            if self.bits == 16:
                print("  Inference: chunked f16 matvec")
            elif has_gemv:
                kl = kernel_labels.get(self.bits, f"Q{self.bits}")
                print(f"  Inference: native AVX2 GEMV {kl}")
            else:
                print(
                    "  Inference: chunked uint8 dequant+matvec (in-place, no AVX GEMV)"
                )
            if self._enable_qcsd and self.bits != 16:
                d_bytes = sum(t.nbytes for t in self._draft_quant_u8.values())
                d_bytes += sum(t.nbytes for t in self._draft_quant_packed.values())
                print(
                    f"  QCSD draft bank: {d_bytes / 1e6:.0f} MB ({self._draft_bits}-bit)"
                )
            n_outliers = sum(len(v) for v in self._outlier_values.values())
            if n_outliers > 0:
                print(f"  SpQR outliers: {n_outliers:,} values in FP16 sparse format")
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

                        # Populate NumPy views for fast native dispatch
                        self._quant_packed_np[key] = packed_np.ravel()
                        self._quant_sc_np[key] = sc.numpy().ravel()
                        self._quant_bi_np[key] = bi.numpy().ravel()

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

            kernel_labels = {4: "Q4 packed (gemv_q4_packed)", 8: "Q8", 3: "Q3", 2: "Q2"}
            if has_gemv:
                kl = kernel_labels.get(self.bits, f"Q{self.bits}")
                print(f"  Inference: native AVX2 GEMV {kl}")
            else:
                print(f"  Inference: chunked uint8 dequant+matvec (in-place, no AVX GEMV)")

        # (Initialization moved up above the early return to fix benchmarking)

    def get_norm(self, layer_idx: int, name: str) -> torch.Tensor:
        return self.layer_norms[layer_idx][name]


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

    Uses chunked matvec (streaming dequant for quantized, chunked f16 read
    for float16) to minimise DRAM bandwidth.  KV history is pre-allocated
    torch tensors - no per-token np.stack / allocations.

    Args:
        use_draft: If True, use the draft (2-bit) weight bank for QCSD.
    """

    # - Self-attention -
    residual = hidden
    h = rms_norm(hidden, store.get_norm(layer_idx, "input_layernorm"))

    qkv = store.matvec(layer_idx, "qkv_proj", h, use_draft=use_draft)

    q = qkv[:, :Q_DIM].view(1, NUM_HEADS, HEAD_DIM)
    k = qkv[:, Q_DIM:Q_DIM + KV_DIM].view(1, NUM_KV_HEADS, HEAD_DIM)
    v = qkv[:, Q_DIM + KV_DIM:].view(1, NUM_KV_HEADS, HEAD_DIM)

    cos_pos = rope_cos[pos:pos + 1]
    sin_pos = rope_sin[pos:pos + 1]
    q = apply_rope(q, cos_pos, sin_pos)
    k = apply_rope(k, cos_pos, sin_pos)

    kv_hist.append(layer_idx, k.squeeze(0), v.squeeze(0))

    k_hist, v_hist = kv_hist.get(layer_idx)
    seq_len = k_hist.shape[0]

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

    q_attn = q.unsqueeze(2)
    attn_out = torch.nn.functional.scaled_dot_product_attention(
        q_attn, k_full, v_full,
    )
    attn_out = attn_out.permute(0, 2, 1, 3).reshape(1, Q_DIM)

    hidden = residual + store.matvec(layer_idx, "o_proj", attn_out, use_draft=use_draft)

    # - Feed-forward (MLP) -
    residual = hidden
    h = rms_norm(hidden, store.get_norm(layer_idx, "post_attention_layernorm"))

    gu = store.matvec(layer_idx, "gate_up_proj", h, use_draft=use_draft)
    act = silu(gu[:, :INTER]) * gu[:, INTER:]

    # Phase 3: FATReLU threshold mask — zero out elements below tau
    # This creates 85% sparsity in the FFN intermediate, enabling sparse down_proj.
    if store._use_fatrelu and layer_idx in store._fatrelu_thresholds:
        tau = store._fatrelu_thresholds[layer_idx]
        act = act * (act.abs() >= tau).float()

    # Phase 4 Prerequisite B: use transposed down_proj for column-sparse access
    # When FATReLU is active and transposed weights are available, use
    # sparse_down_proj_T for 6.7x memory traffic reduction.
    use_T_sparse = (store._use_fatrelu and layer_idx in store._down_proj_T
                    and not use_draft)
    if use_T_sparse:
        act_np = act.detach().cpu().float().contiguous().numpy().ravel()
        active_rows = np.where(np.abs(act_np) > 1e-9)[0].astype(np.int32)
        if len(active_rows) > 0:
            try:
                from asdsl.kernels._native_sparse_gemv import sparse_down_proj_T as _sparse_T
                dT = store._down_proj_T[layer_idx]
                y_down_np = _sparse_T(
                    dT['packed'].ravel(), dT['scales'], dT['biases'],
                    act_np, active_rows,
                    dT['in_dim'], dT['out_dim'], store.group_size
                )
                hidden = residual + torch.from_numpy(y_down_np).unsqueeze(0)
            except Exception:
                hidden = residual + store.matvec(layer_idx, "down_proj", act, use_draft=use_draft)
        else:
            # All zeros: output is zero
            hidden = residual + torch.zeros(1, HIDDEN, dtype=torch.float32)
    elif store._enable_sparse and not use_draft and store._use_native_gemv:
        from asdsl.kernels import compute_activation_bitmask
        act_np = act.detach().cpu().float().contiguous().numpy().ravel()
        bitmask, active_indices = compute_activation_bitmask(
            act_np, threshold=store._sparsity_threshold
        )
        sparsity = 1.0 - len(active_indices) / len(act_np)
        if sparsity > 0.80:
            hidden = residual + store.matvec_sparse(
                layer_idx, "down_proj", act, bitmask, active_indices
            )
        else:
            hidden = residual + store.matvec(layer_idx, "down_proj", act, use_draft=use_draft)
    else:
        hidden = residual + store.matvec(layer_idx, "down_proj", act, use_draft=use_draft)

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

    use_T_sparse = store._use_fatrelu and layer_idx in store._down_proj_T
    if use_T_sparse:
        try:
            from asdsl.kernels._native_sparse_gemv import sparse_down_proj_T as _sparse_T

            dT = store._down_proj_T[layer_idx]
            act_np_all = act.detach().cpu().float().numpy()
            outs: list[torch.Tensor] = []
            for ki in range(K):
                act_np = act_np_all[ki].ravel()
                active_rows = np.where(np.abs(act_np) > 1e-9)[0].astype(np.int32)
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
# Generation
# ---------------------------------------------------------------------------

def generate(
    prompt: str,
    store: WeightStore,
    tokenizer,
    max_new_tokens: int = 50,
    system_prompt: str = "You are a helpful AI assistant.",
    bench_metrics_out: list | None = None,
    logits_hook: Optional[Callable[[np.ndarray], None]] = None,
) -> str:
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

    # Pre-compute RoPE tables (generous max length).
    # Pass ROTARY_DIM (96) - only the rotated portion of each head needs tables.
    max_seq = len(input_ids) + max_new_tokens + 64
    rope_cos, rope_sin = build_rope_cache(max_seq, ROTARY_DIM)

    # Per-layer KV history (pre-allocated torch tensors)
    kv_hist = KVHistory(max_seq=max_seq)

    # ASDSL tracker - updated once per generated token for block-sparse analytics
    asdsl_tracker = ASDSLKVTracker()

    def run_forward(token_id: int, pos: int, need_logits: bool = True) -> torch.Tensor | None:
        """Full 32-layer forward pass for a single token at position pos.
        When need_logits=False (prefill body), skips the expensive LM-head matmul."""
        hidden = store.embed_f16[token_id].float().unsqueeze(0)

        k_new: list[np.ndarray] = []
        v_new: list[np.ndarray] = []

        for i in range(NUM_LAYERS):
            hidden = forward_layer(hidden, i, store, kv_hist, rope_cos, rope_sin, pos)
            k_np, v_np = kv_hist.get_last_np(i)
            k_new.append(k_np)
            v_new.append(v_np)

        # Feed this token's (all-layer) K/V into the ASDSL block-sparse tracker
        asdsl_tracker.record_token(k_new, v_new)

        if not need_logits:
            return None

        # Final norm + LM head (embed weights tied)
        hidden = rms_norm(hidden, store.final_norm)
        logits = store.lm_head_matvec(hidden)
        return logits

    # ------------------------------------------------------------------
    # Prefill: run every prompt token through the model sequentially.
    # We only keep the logits from the LAST token.
    # ------------------------------------------------------------------
    print("Prefill: ", end="", flush=True)
    t_prefill_start = time.perf_counter()

    with torch.inference_mode():
        logits = None
        for pos, tid in enumerate(input_ids):
            is_last = (pos == len(input_ids) - 1)
            logits = run_forward(tid, pos, need_logits=is_last)

    t_prefill = time.perf_counter() - t_prefill_start
    print(f"done ({len(input_ids)} tokens in {t_prefill:.1f}s)")

    # ------------------------------------------------------------------
    # Decode loop
    # ------------------------------------------------------------------
    print("\nAssistant: ", end="", flush=True)
    generated: list[int] = []
    t_decode_start = time.perf_counter()

    pos = len(input_ids)
    with torch.inference_mode():
        for _step in range(max_new_tokens):
            if logits_hook is not None and logits is not None:
                logits_hook(logits.detach().cpu().float().numpy().ravel())

            next_token = int(logits.argmax())
            generated.append(next_token)

            tok_text = tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens([next_token])
            )
            print(tok_text, end="", flush=True)

            if next_token in EOS_TOKEN_IDS:
                break

            logits = run_forward(next_token, pos)
            pos += 1

    t_decode = time.perf_counter() - t_decode_start
    n_tokens = len(generated)
    tps = n_tokens / t_decode if t_decode > 0 else 0

    response_text = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(generated)
    )

    kv_stats = asdsl_tracker.stats()
    quant_label = "float16 (no quant)" if store.bits == 16 else f"{store.bits}-bit ASDSL"
    print(f"Generated : {n_tokens} tokens  |  {tps:.2f} tok/s  |  decode {t_decode:.1f}s")
    print(f"ASDSL KV  : {kv_stats['tokens']} tokens tracked  "
          f"| {kv_stats['blocks_used']}/{kv_stats['blocks_capacity']} blocks  "
          f"| {kv_stats['memory_mb']:.1f} MB")
    print(f"Weights   : {quant_label}")
    print("=" * 66)

    if bench_metrics_out is not None:
        bench_metrics_out.append(
            {
                "decode_tokens": n_tokens,
                "decode_s": t_decode,
                "tokens_per_second": tps,
            }
        )

    return response_text


# ---------------------------------------------------------------------------
# Streaming generation
# ---------------------------------------------------------------------------

from dataclasses import dataclass

@dataclass
class StreamToken:
    """A single streamed token with metadata.

    Attributes:
        text: The decoded text fragment for this token.
        token_id: The integer token ID.
        step: Zero-based step index within the decode phase.
        is_eos: True if this token is an end-of-sequence marker.
        elapsed_s: Seconds elapsed since decode started (for this token).
        tokens_per_second: Running average tok/s up to this point.
    """
    text: str
    token_id: int
    step: int
    is_eos: bool = False
    elapsed_s: float = 0.0
    tokens_per_second: float = 0.0


def generate_stream(
    prompt: str,
    store: WeightStore,
    tokenizer,
    max_new_tokens: int = 50,
    system_prompt: str = "You are a helpful AI assistant.",
):
    """Generator that yields StreamToken objects as they are decoded.

    Usage::

        for tok in generate_stream("Hello!", store, tokenizer):
            print(tok.text, end="", flush=True)
            if tok.is_eos:
                break

    The generator handles prefill internally before yielding any tokens.
    After the final token (EOS or max_new_tokens), it yields one last
    StreamToken with ``is_eos=True``.
    """
    # Format prompt
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )

    max_seq = len(input_ids) + max_new_tokens + 64
    rope_cos, rope_sin = build_rope_cache(max_seq, ROTARY_DIM)
    kv_hist = KVHistory(max_seq=max_seq)
    asdsl_tracker = ASDSLKVTracker()

    def run_forward(token_id: int, pos: int, need_logits: bool = True):
        hidden = store.embed_f16[token_id].float().unsqueeze(0)
        k_new, v_new = [], []
        for i in range(NUM_LAYERS):
            hidden = forward_layer(hidden, i, store, kv_hist, rope_cos, rope_sin, pos)
            k_np, v_np = kv_hist.get_last_np(i)
            k_new.append(k_np)
            v_new.append(v_np)
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

    with torch.inference_mode():
        for step in range(max_new_tokens):
            next_token = int(logits.argmax())
            tok_text = tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens([next_token])
            )
            elapsed = time.perf_counter() - t_decode_start
            tps = (step + 1) / elapsed if elapsed > 0 else 0.0
            is_eos = next_token in EOS_TOKEN_IDS

            yield StreamToken(
                text=tok_text,
                token_id=next_token,
                step=step,
                is_eos=is_eos,
                elapsed_s=elapsed,
                tokens_per_second=tps,
            )

            if is_eos:
                return

            logits = run_forward(next_token, pos)
            pos += 1


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

                # ── VERIFY PHASE (BATCHED TARGET) ──────────────────
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
                "_verify_calls": _verify_calls,
                "qcsd_verify_batched_passes": _verify_calls,
                "qcsd_speculative_cycles": speculative_cycles,
                "qcsd_verify_extra_run_forward": _verify_extra_run_forward,
            }
        )

    return response_text


# ---------------------------------------------------------------------------
# EAGLE-3: MTP head speculative decoding (Profile G)
# ---------------------------------------------------------------------------

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

    Uses ``torch.nn.functional.layer_norm`` + ``gelu`` to match ``train_mtp_head``.

    Returns: list of up to k draft token ids.
    """
    import torch
    import torch.nn.functional as F

    if not store._use_eagle3 or store._mtp_head is None:
        return []

    head = store._mtp_head
    fc1_W = torch.from_numpy(head["fc1_W"]).float()
    fc1_b = torch.from_numpy(head["fc1_b"]).float()
    norm_W = torch.from_numpy(head["norm_W"]).float()
    norm_b = torch.from_numpy(head["norm_b"]).float()
    proj_W = torch.from_numpy(head["proj_W"]).float()
    proj_b = torch.from_numpy(head["proj_b"]).float()

    prev_hidden = torch.from_numpy(store._last_final_hidden.copy()).float()
    cur_embed_src = embed_source_token_id
    drafts: list[int] = []

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
    total_alignment_run_forward = 0
    total_bonus_one_row_pass = 0

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
                    n_keep = 1 + len(accepted)
                    if n_keep < n_verify:
                        kv_hist.restore_len(draft_start_pos + n_keep)
                    # Rejection: need target forward from correction token (unavoidable).
                    total_alignment_run_forward += 1
                    logits = run_forward(
                        correction,
                        draft_start_pos + n_keep,
                        kv_hist,
                        need_logits=True,
                        hook_logits=True,
                    )
                    generated.append(correction)
                    print(tokenizer.convert_tokens_to_string(
                        tokenizer.convert_ids_to_tokens([correction])), end="", flush=True)
                    pos += n_keep + 1
                    if correction in EOS_TOKEN_IDS:
                        break
                else:
                    # All drafts matched: one batched row for draft[-1] (bonus logits +
                    # _last_final_hidden). Replaces run_forward(draft[-1]) so reject
                    # cycles are not charged an unnecessary serial forward on accept.
                    total_bonus_one_row_pass += 1
                    bonus_row = store.embed_f16[draft_tokens[-1]].float().unsqueeze(0)
                    for i in range(NUM_LAYERS):
                        bonus_row = forward_layer_batch(
                            bonus_row, i, store, kv_hist, rope_cos, rope_sin,
                            draft_start_pos + L,
                        )
                    bonus_h = rms_norm(bonus_row, store.final_norm)
                    bonus_logits = store.lm_head_matmul_batch(bonus_h)
                    logits = bonus_logits[0]
                    if logits_hook is not None:
                        logits_hook(logits.detach().cpu().float().numpy().ravel())
                    store._last_final_hidden = (
                        bonus_h[0].detach().cpu().float().numpy().ravel()
                    )
                    pos += n_verify + 1
                    if draft_tokens[-1] in EOS_TOKEN_IDS:
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
    print(
        f"[EAGLE-3] {_cyc} cycles: verify_passes={total_verify_passes} "
        f"reject_run_forward={total_alignment_run_forward} "
        f"bonus_1row_batch={total_bonus_one_row_pass} "
        f"(target: 0 serial run_forward on all-accept; reject still needs 1)"
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
            "eagle3_alignment_run_forward": total_alignment_run_forward,
            "eagle3_bonus_one_row_pass": total_bonus_one_row_pass,
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
    parser.add_argument("--sparse", action="store_true",
                        help="Enable activation-sparse GEMV (Tier 3)")
    parser.add_argument("--sparse-threshold", type=float, default=0.01,
                        help="Activation sparsity threshold (default: 0.01)")
    parser.add_argument("--slim-meta", type=str, default=None,
                        help="Path to phi4_slim_meta.json for Phase 2 mixed-precision (Profile E)")
    parser.add_argument("--fatrelu-thresholds", type=str, default=None,
                        help="Path to phi4_fatrelu_thresholds.json for Phase 3 sparsity (Profile F)")
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Bench profile: G = FATReLU + LUT + EAGLE-3 MTP (requires --bits 4, local mtp_head.pt)",
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
    args = parser.parse_args()
    if args.no_weight_cache:
        os.environ["PHI4_NO_WEIGHT_CACHE"] = "1"

    set_thread_count(args.threads if args.threads > 0 else 0)
    if args.threads == 0:
        args.threads = int(os.environ.get("OMP_NUM_THREADS", "1"))

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
    store = WeightStore(
        bits=args.bits,
        group_size=args.group_size if args.group_size > 0 else None,
        enable_qcsd=args.qcsd,
        draft_bits=args.draft_bits,
        enable_sparse=args.sparse,
        sparsity_threshold=args.sparse_threshold,
    )
    store.load()
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
    store.warm_cache()

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
        ):
            print(tok.text, end="", flush=True)
        print(f"\n  [{tok.step + 1} tokens | {tok.tokens_per_second:.2f} tok/s]")
    else:
        generate(
            prompt=args.prompt,
            store=store,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
        )


if __name__ == "__main__":
    main()
