"""Shared configuration, constants, and data types for the ASDSL framework."""

from __future__ import annotations

import os
import platform
import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal

USE_ACTIVATION_SPARSE_GEMV = os.environ.get("ASDSL_SPARSE_GEMV", "0") == "1"

class QuantBits(IntEnum):
    """Supported quantization bit-widths."""

    TERNARY = 2  # 1.58-bit encoded as 2-bit (values: -1, 0, 1)
    TWO = 2
    THREE = 3
    FOUR = 4
    EIGHT = 8
    SIXTEEN = 16


class CPUFeature(IntEnum):
    """CPU SIMD feature levels for kernel dispatch."""

    SCALAR = 0
    SSE42 = 1
    AVX2 = 2
    AVX512 = 3
    AVX512_VNNI = 4
    AMX = 5
    NEON = 10
    SVE = 11


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a target Small Language Model."""

    name: str
    num_layers: int
    hidden_dim: int
    num_attention_heads: int
    num_kv_heads: int
    intermediate_dim: int
    vocab_size: int
    max_context_length: int
    rope_type: Literal["standard", "yarn"] = "yarn"
    gqa_groups: int = 4


# Pre-defined model configurations
PHI3_MINI_CONFIG = ModelConfig(
    name="phi-3-mini",
    num_layers=32,
    hidden_dim=3072,
    num_attention_heads=32,
    num_kv_heads=8,
    intermediate_dim=8192,
    vocab_size=32064,
    max_context_length=4096,
    rope_type="yarn",
    gqa_groups=4,
)


@dataclass
class QuantizationConfig:
    """Configuration for salience-driven mixed-precision quantization."""

    default_bits: int = 2
    salience_bits: int = 8
    salience_threshold: float = 0.01
    calibration_samples: int = 128
    group_size: int = 128
    protect_pivot_tokens: bool = True
    protect_task_circuits: bool = True
    # Block-wise quantization (replaces SpQR)
    use_blockwise: bool = True
    block_size: int = 32  # Cache-line friendly


@dataclass
class InferenceConfig:
    """Runtime inference configuration."""

    num_compute_cores: int = 2
    num_prefetch_cores: int = 1
    max_batch_size: int = 1
    use_huge_pages: bool = True
    pin_memory: bool = True
    numa_aware: bool = True
    speculative_draft_tokens: int = 4
    speculative_skip_layers: list[int] = field(default_factory=list)
    kv_cache_block_sparse: bool = True
    # INT8 KV cache for 4x memory savings
    kv_cache_quantize: bool = True
    # Thread affinity: pin to physical cores (avoids HT overlap)
    thread_affinity_cores: list[int] = field(default_factory=list)
    # QCSD speculative decoding
    qcsd_enabled: bool = False
    qcsd_draft_bits: int = 2
    qcsd_verify_bits: int = 4
    qcsd_num_draft_tokens: int = 5


def detect_cpu_features() -> set[CPUFeature]:
    """Detect available CPU SIMD features on the current platform.

    When the native C++ inference engine is available, uses CPUID-based
    detection for AVX-512 and VNNI. Otherwise conservatively assumes AVX2.
    """
    features: set[CPUFeature] = {CPUFeature.SCALAR}
    arch = platform.machine().lower()

    if arch in ("x86_64", "amd64", "x86"):
        if struct.calcsize("P") == 8:
            features.add(CPUFeature.AVX2)

        # Try runtime detection via native engine
        try:
            from asdsl.kernels._native_gemv import check_avx2, check_fma
            if check_avx2():
                features.add(CPUFeature.AVX2)
            try:
                from asdsl.kernels._native_gemv import check_avx512, check_vnni
                if check_avx512():
                    features.add(CPUFeature.AVX512)
                if check_vnni():
                    features.add(CPUFeature.AVX512_VNNI)
            except (ImportError, AttributeError):
                pass
        except ImportError:
            pass

    elif arch in ("aarch64", "arm64"):
        features.add(CPUFeature.NEON)

    return features


def get_system_info() -> dict:
    """Gather system information relevant to inference configuration."""
    import psutil

    mem = psutil.virtual_memory()
    cpu_features = detect_cpu_features()

    info = {
        "platform": platform.system(),
        "architecture": platform.machine(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "total_ram_gb": round(mem.total / (1024**3), 2),
        "available_ram_gb": round(mem.available / (1024**3), 2),
        "cpu_features": [f.name for f in cpu_features],
        "has_avx512": CPUFeature.AVX512 in cpu_features,
        "has_vnni": CPUFeature.AVX512_VNNI in cpu_features,
    }

    # Check for native engine
    try:
        from asdsl.kernels import _native_inference
        info["native_engine"] = True
        info["native_openmp"] = _native_inference.has_openmp
    except ImportError:
        info["native_engine"] = False

    return info
