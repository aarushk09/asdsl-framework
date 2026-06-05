"""Calibration pass: record sparsity and LUT footprint per projection."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from asdsl.dispatch.policy import (
    PHI4_PROJECTIONS,
    DispatchPolicy,
    ProjectionProfile,
)
from asdsl.lut.lut_table_builder import LUTTableBuilder

if TYPE_CHECKING:
    from experiments.phi4_cpu_run import WeightStore

_DEFAULT_PROFILE_PATH = (
    Path(__file__).resolve().parent / "projection_profiles.json"
)


def measure_activation_sparsity(x: np.ndarray, threshold: float = 0.01) -> float:
    """Fraction of elements with |x| >= threshold."""
    flat = np.abs(np.asarray(x, dtype=np.float32).ravel())
    if flat.size == 0:
        return 0.0
    return float(np.mean(flat >= threshold))


def lut_footprint_for_shape(
    cols: int,
    group_size: int = 32,
    tile_groups: int = 64,
) -> int:
    """Per-row LUT working set for L2 budget gating (not full materialized table)."""
    return LUTTableBuilder.tile_working_set_bytes(cols, group_size, tile_groups)


def build_profiles_from_store(
    store: WeightStore,
    *,
    sparsity_samples: dict[tuple[int, str], float] | None = None,
    tile_groups: int = 64,
) -> dict[tuple[int, str], ProjectionProfile]:
    """Build profiles from loaded WeightStore shapes + optional sparsity stats."""
    profiles: dict[tuple[int, str], ProjectionProfile] = {}
    bits = store.bits
    gs = store.group_size

    keys = list(store._quant_shapes.keys())
    for layer_idx, name in keys:
        if name not in PHI4_PROJECTIONS:
            continue
        rows, cols = store._quant_shapes[(layer_idx, name)]
        sp = 0.0
        if sparsity_samples:
            sp = sparsity_samples.get((layer_idx, name), 0.0)
        profiles[(layer_idx, name)] = ProjectionProfile(
            layer_idx=layer_idx,
            proj_name=name,
            rows=rows,
            cols=cols,
            bits=bits,
            group_size=gs,
            mean_sparsity=sp,
            lut_footprint_bytes=lut_footprint_for_shape(cols, gs, tile_groups),
            tile_groups=tile_groups,
        )

    return profiles


def calibrate_forward_pass(
    store: WeightStore,
    tokens: list[int],
    *,
    sparsity_threshold: float = 0.01,
    max_tokens: int = 32,
) -> dict[tuple[int, str], float]:
    """Run a short forward and accumulate per-projection activation sparsity."""
    try:
        from experiments.phi4_cpu_run import (
            KVHistory,
            NUM_LAYERS,
            ROTARY_DIM,
            build_rope_cache,
            forward_layer,
            rms_norm,
        )
    except ModuleNotFoundError:
        from phi4_cpu_run import (
            KVHistory,
            NUM_LAYERS,
            ROTARY_DIM,
            build_rope_cache,
            forward_layer,
            rms_norm,
        )

    sparsity_acc: dict[tuple[int, str], list[float]] = {
        (i, n): [] for i in range(NUM_LAYERS) for n in PHI4_PROJECTIONS
    }
    window = tokens[:max_tokens]
    if len(window) < 1:
        return {}

    max_seq = len(window) + 4
    rope_cos, rope_sin = build_rope_cache(max_seq, ROTARY_DIM)
    kv = KVHistory()

    for pos, tid in enumerate(window):
        hidden = store.embed_f16[tid].float().unsqueeze(0)
        for layer in range(NUM_LAYERS):
            residual = hidden.detach().cpu().float().numpy().ravel()
            rms_att_w = store.get_norm(layer, "input_layernorm").detach().cpu().float().numpy()
            rms = float(np.sqrt(np.mean(residual**2) + 1e-6))
            h_qkv = (residual / rms) * rms_att_w
            sparsity_acc[(layer, "qkv_proj")].append(
                measure_activation_sparsity(h_qkv, sparsity_threshold)
            )

            hidden = forward_layer(
                hidden, layer, store, kv, rope_cos, rope_sin, pos=pos
            )
            # o_proj / gate_up: do not reuse qkv hidden (inflates sparsity → false SPARSE).
            sparsity_acc[(layer, "o_proj")].append(0.0)
            sparsity_acc[(layer, "gate_up_proj")].append(0.0)
            # down_proj input is post-SwiGLU (often sparse); proxy via qkv norm sparsity.
            sparsity_acc[(layer, "down_proj")].append(
                measure_activation_sparsity(h_qkv, sparsity_threshold)
            )
        _ = rms_norm(hidden, store.final_norm)

    return {
        k: float(np.mean(v)) if v else 0.0
        for k, v in sparsity_acc.items()
    }


def calibrate(
    store: WeightStore,
    tokens: list[int] | None = None,
    *,
    output_path: Path | str | None = None,
    sparsity_threshold: float = 0.01,
    max_tokens: int = 32,
    tile_groups: int = 64,
) -> DispatchPolicy:
    """Calibrate store and write projection_profiles.json."""
    sparsity: dict[tuple[int, str], float] = {}
    if tokens:
        sparsity = calibrate_forward_pass(
            store, tokens,
            sparsity_threshold=sparsity_threshold,
            max_tokens=max_tokens,
        )
    try:
        from experiments.phi4_cpu_run import NUM_LAYERS
    except ModuleNotFoundError:
        from phi4_cpu_run import NUM_LAYERS

    # Bias down_proj toward SPARSE (post-activation); keep qkv/o below SPARSE threshold.
    for layer in range(NUM_LAYERS):
        sparsity[(layer, "down_proj")] = max(
            sparsity.get((layer, "down_proj"), 0.0), 0.72
        )
        sparsity[(layer, "qkv_proj")] = min(sparsity.get((layer, "qkv_proj"), 0.0), 0.35)
        sparsity[(layer, "o_proj")] = min(sparsity.get((layer, "o_proj"), 0.0), 0.35)

    profiles = build_profiles_from_store(
        store, sparsity_samples=sparsity, tile_groups=tile_groups
    )
    policy = DispatchPolicy(profiles)
    out = Path(output_path) if output_path else _DEFAULT_PROFILE_PATH
    policy.save_json(out)
    return policy
