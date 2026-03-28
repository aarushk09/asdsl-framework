#!/usr/bin/env python3
"""
Phase 2 — SliM-LLM Salience-Driven Calibration Pipeline for ASDSL.

Produces phi4_slim_meta.json at the repo root with per-group bit-width
assignments and calibrated scale/zero_point values.

Algorithm:
  1. Memory guard: abort if < 5 GB RAM available
  2. Load Phi-4 Q4 weights via WeightStore (no FP16 required)
  3. Run calibration prompts through the model, collect hidden states per layer
  4. Compute salience = activation_variance × scale² per weight group
  5. Assign bit-widths (2/3/4) to hit ~2.2 avg bits/parameter
  6. Run SQC (scale calibration) for groups with changed bit-width
  7. Write phi4_slim_meta.json

Usage:
  python scripts/slim_calibrate.py           # full calibration (32 prompts, 32 layers)
  python scripts/slim_calibrate.py --quick   # quick test (4 prompts, 4 layers)
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil

# ── path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── constants ──────────────────────────────────────────────────────────────────
NUM_LAYERS = 32
HIDDEN_DIM = 3072
MATRIX_NAMES = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]

# Phi-4 matrix dimensions (rows × cols) for each projection
# qkv_proj: (Q_DIM + 2*KV_DIM) × HIDDEN = 5120 × 3072
# o_proj:   HIDDEN × Q_DIM = 3072 × 3072
# gate_up_proj: 2*INTER × HIDDEN = 16384 × 3072
# down_proj: HIDDEN × INTER = 3072 × 8192
MATRIX_DIMS = {
    "qkv_proj":    (5120, 3072),
    "o_proj":      (3072, 3072),
    "gate_up_proj": (16384, 3072),
    "down_proj":   (3072, 8192),
}

CALIBRATION_PROMPTS = [
    "The capital of France is",
    "In mathematics, the derivative of x squared is",
    "def fibonacci(n):\n    if n <= 1:\n        return n",
    "The French Revolution began in the year",
    "To convert Celsius to Fahrenheit, multiply by",
    "The mitochondria is known as",
    "import numpy as np\narr = np.array([1, 2, 3])\nresult =",
    "Once upon a time in a land far away,",
    "The speed of light in a vacuum is approximately",
    "SELECT * FROM users WHERE",
    "The three laws of thermodynamics state that",
    "To train a neural network, we use",
    "The Battle of Waterloo took place in",
    "In Python, a list comprehension looks like",
    "The chemical formula for water is",
    "def quicksort(arr):\n    if len(arr) <= 1:",
    "The largest planet in our solar system is",
    "HTTP status code 404 means",
    "The square root of 144 is",
    "To reverse a string in Python,",
    "The Pythagorean theorem states that",
    "Machine learning models overfit when",
    "The main difference between RAM and ROM is",
    "In quantum mechanics, Heisenberg's uncertainty principle",
    "The Treaty of Versailles was signed in",
    "Binary search has a time complexity of",
    "The boiling point of water at sea level is",
    "Object-oriented programming is based on the concept of",
    "The first element in the periodic table is",
    "To calculate compound interest, the formula is",
    "The Renaissance period began in",
    "In SQL, the JOIN operation combines",
]


# ── memory guard ───────────────────────────────────────────────────────────────

def check_memory_headroom(required_gb: float = 5.0) -> None:
    available = psutil.virtual_memory().available / 1e9
    if available < required_gb:
        print(f"ERROR: only {available:.1f} GB RAM available, need {required_gb:.1f} GB minimum")
        print("Close other applications and retry.")
        sys.exit(1)
    print(f"Memory check passed: {available:.1f} GB available")


def log_memory(label: str = "") -> float:
    available = psutil.virtual_memory().available / 1e9
    if label:
        print(f"  [mem] {label}: {available:.1f} GB available", flush=True)
    return available


# ── activation collection ──────────────────────────────────────────────────────

def collect_hidden_states(
    store,
    tokenizer,
    prompts: list[str],
    n_layers: int,
    quick: bool = False,
) -> dict[int, dict[str, list[np.ndarray]]]:
    """
    Run each prompt through the model and collect the hidden state (input to
    each layer's weight matrices) for each layer.

    Returns:
        hidden_states[layer_idx][matrix_name] = list of np.ndarray (hidden_dim,)
        One entry per prompt token (we use only the last token of each prompt).
    """
    import torch

    # Import the forward pass infrastructure
    from experiments.phi4_cpu_run import (
        KVHistory, forward_layer, rms_norm,
        NUM_LAYERS as _NL, HIDDEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
        Q_DIM, KV_DIM, INTER,
    )

    # Build RoPE tables (needed for forward_layer)
    from experiments.phi4_cpu_run import build_rope_cache, ROTARY_DIM
    rope_cos, rope_sin = build_rope_cache(128, ROTARY_DIM)

    # hidden_states[layer][matrix_name] = list of activation vectors
    # For each layer, the input hidden state is the same for all matrices
    # (they all receive the same normalized hidden state)
    hidden_states: dict[int, list[np.ndarray]] = {i: [] for i in range(n_layers)}

    print(f"  Collecting hidden states from {len(prompts)} prompts, {n_layers} layers...")

    for p_idx, prompt in enumerate(prompts):
        if p_idx % 8 == 0:
            print(f"    Prompt {p_idx+1}/{len(prompts)}...", flush=True)

        # Tokenize
        try:
            tokens = tokenizer.encode(prompt, return_tensors="pt")
            if tokens.shape[1] > 32:
                tokens = tokens[:, :32]  # cap at 32 tokens
        except Exception:
            continue

        # Run prefill through n_layers, collecting hidden states
        kv = KVHistory(max_seq=64)

        try:
            # Get initial embedding
            embed = store.embed_f16
            if embed is None:
                continue
            token_ids = tokens[0].tolist()
            h = embed[token_ids[-1]].float().unsqueeze(0)  # (1, hidden_dim)

            # Run through each layer, collecting the pre-norm hidden state
            for layer_idx in range(n_layers):
                # The input to each weight matrix is the RMSNorm of h
                norm_w = store.get_norm(layer_idx, "input_layernorm")
                h_norm = rms_norm(h, norm_w)  # (1, hidden_dim)

                # Record this as the activation for this layer
                hidden_states[layer_idx].append(
                    h_norm.detach().cpu().float().numpy().ravel()
                )

                # Run the full layer to get the next hidden state
                h = forward_layer(
                    h, layer_idx, store, kv, rope_cos, rope_sin,
                    pos=len(token_ids) - 1
                )

        except Exception as e:
            print(f"    Warning: prompt {p_idx} failed at layer: {e}", flush=True)
            continue

        # Free KV cache
        del kv
        gc.collect()

    return hidden_states


# ── salience computation ───────────────────────────────────────────────────────

def compute_salience(
    hidden_states: dict[int, list[np.ndarray]],
    store,
    n_layers: int,
    group_size: int,
) -> dict[int, dict[str, np.ndarray]]:
    """
    Compute salience scores for each weight group.

    salience[layer][matrix_name][group_idx] = variance(activations) * scale²

    For fused matrices (qkv_proj, gate_up_proj), we compute salience over
    the input dimension (cols) which is shared across all sub-matrices.
    """
    salience: dict[int, dict[str, np.ndarray]] = {}

    for layer_idx in range(n_layers):
        salience[layer_idx] = {}
        acts = hidden_states.get(layer_idx, [])

        if not acts:
            # No activations collected — use uniform salience
            for name in MATRIX_NAMES:
                key = (layer_idx, name)
                if key in store._quant_sc:
                    n_groups = store._quant_sc[key].shape[0]
                    salience[layer_idx][name] = np.ones(n_groups, dtype=np.float32)
            continue

        # Stack activations: (n_prompts, hidden_dim)
        act_matrix = np.stack(acts, axis=0).astype(np.float32)  # (N, hidden_dim)

        for name in MATRIX_NAMES:
            key = (layer_idx, name)
            if key not in store._quant_sc:
                continue

            sc = store._quant_sc[key].float().numpy()  # (n_groups,)
            rows, cols = store._quant_shapes[key]
            n_groups_per_row = cols // group_size
            n_groups = rows * n_groups_per_row

            # Compute variance of activations per group (over input dimension)
            # act_matrix shape: (N, hidden_dim) where hidden_dim = cols
            # For each group g covering cols [g*gs : (g+1)*gs]:
            #   var_g = mean variance of activations in that column range
            group_salience = np.zeros(n_groups, dtype=np.float32)

            # Input dimension is cols (the activation dimension)
            act_cols = min(act_matrix.shape[1], cols)
            act_sub = act_matrix[:, :act_cols]  # (N, cols)

            for g in range(n_groups_per_row):
                col_start = g * group_size
                col_end = min(col_start + group_size, act_cols)
                if col_start >= act_cols:
                    break
                act_group = act_sub[:, col_start:col_end]  # (N, group_size)
                var_g = float(np.var(act_group))

                # Apply to all rows (each row has the same input activation)
                for row in range(rows):
                    gidx = row * n_groups_per_row + g
                    if gidx < n_groups:
                        group_salience[gidx] = var_g * float(sc[gidx]) ** 2

            salience[layer_idx][name] = group_salience

    return salience


# ── bit-width assignment ───────────────────────────────────────────────────────

def assign_bitwidths(
    salience: dict[int, dict[str, np.ndarray]],
    n_layers: int,
    target_avg_bits: float = 2.2,
    min_bits: int = 2,
    max_bits: int = 4,
) -> dict[int, dict[str, np.ndarray]]:
    """
    Assign bit-widths to each group using salience-ranked greedy allocation.

    Hard constraints:
    - First 2 and last 2 transformer layers: minimum 3 bits
    - qkv_proj and o_proj: minimum 3 bits (attention is sensitive)
    - All other groups: 2-4 bits based on salience rank

    Returns: bitwidths[layer][matrix_name] = np.ndarray of int (per group)
    """
    # Collect all (salience, layer, name, group_idx) tuples
    all_groups: list[tuple[float, int, str, int]] = []
    for layer_idx in range(n_layers):
        for name, sal_arr in salience.get(layer_idx, {}).items():
            for g, s in enumerate(sal_arr):
                all_groups.append((float(s), layer_idx, name, g))

    if not all_groups:
        # Fallback: all 4-bit
        bw: dict[int, dict[str, np.ndarray]] = {}
        for layer_idx in range(n_layers):
            bw[layer_idx] = {}
            for name, sal_arr in salience.get(layer_idx, {}).items():
                bw[layer_idx][name] = np.full(len(sal_arr), 4, dtype=np.int32)
        return bw

    total_groups = len(all_groups)

    # Sort by salience descending
    all_groups.sort(key=lambda x: x[0], reverse=True)

    # Initial assignment: top 25% → 4-bit, next 15% → 3-bit, rest → 2-bit
    top4_frac = 0.25
    top3_frac = 0.15

    # Iteratively adjust fractions to hit target_avg_bits
    for _ in range(20):
        n4 = int(total_groups * top4_frac)
        n3 = int(total_groups * top3_frac)
        n2 = total_groups - n4 - n3
        avg = (n4 * 4 + n3 * 3 + n2 * 2) / total_groups
        if abs(avg - target_avg_bits) < 0.05:
            break
        if avg > target_avg_bits:
            top4_frac = max(0.05, top4_frac - 0.02)
            top3_frac = max(0.05, top3_frac - 0.01)
        else:
            top4_frac = min(0.50, top4_frac + 0.02)
            top3_frac = min(0.30, top3_frac + 0.01)

    n4 = int(total_groups * top4_frac)
    n3 = int(total_groups * top3_frac)

    # Build initial assignment
    raw_bits: dict[tuple[int, str, int], int] = {}
    for rank, (sal, layer_idx, name, g) in enumerate(all_groups):
        if rank < n4:
            bits = 4
        elif rank < n4 + n3:
            bits = 3
        else:
            bits = 2
        raw_bits[(layer_idx, name, g)] = bits

    # Apply hard constraints
    for layer_idx in range(n_layers):
        for name, sal_arr in salience.get(layer_idx, {}).items():
            for g in range(len(sal_arr)):
                key = (layer_idx, name, g)
                current = raw_bits.get(key, 2)

                # First/last 2 layers: minimum 3 bits
                if layer_idx < 2 or layer_idx >= n_layers - 2:
                    current = max(current, 3)

                # Attention projections: minimum 3 bits
                if name in ("qkv_proj", "o_proj"):
                    current = max(current, 3)

                raw_bits[key] = current

    # Recompute achieved avg bits
    total_bits = sum(raw_bits.values())
    achieved_avg = total_bits / max(len(raw_bits), 1)
    print(f"  Bit-width assignment: {achieved_avg:.3f} avg bits "
          f"(target {target_avg_bits:.1f})")

    # Count per bit-width
    counts = {2: 0, 3: 0, 4: 0}
    for b in raw_bits.values():
        counts[b] = counts.get(b, 0) + 1
    print(f"  Groups: 4-bit={counts[4]}, 3-bit={counts[3]}, 2-bit={counts[2]}")

    # Build output dict for ALL layers (not just calibrated ones)
    # For uncalibrated layers, use the median bit-width from calibrated layers
    # as a proxy (they get the same distribution as the "middle" layers)
    bw: dict[int, dict[str, np.ndarray]] = {}

    # Compute median bit-width per matrix name from calibrated layers
    median_bw: dict[str, float] = {}
    for name in MATRIX_NAMES:
        bits_list = []
        for layer_idx in range(n_layers):
            sal_arr = salience.get(layer_idx, {}).get(name)
            if sal_arr is not None:
                for g in range(len(sal_arr)):
                    bits_list.append(raw_bits.get((layer_idx, name, g), 2))
        if bits_list:
            median_bw[name] = float(np.median(bits_list))

    # For all layers in the store, assign bit-widths
    # We need to know the n_groups for each layer/matrix
    # Use the salience dict for calibrated layers, and a default for others
    all_layer_indices = set()
    for layer_idx in range(n_layers):
        for name in salience.get(layer_idx, {}).keys():
            all_layer_indices.add(layer_idx)

    # Determine total layers from raw_bits keys
    max_layer = max((k[0] for k in raw_bits.keys()), default=n_layers - 1)

    for layer_idx in range(max(n_layers, max_layer + 1)):
        bw[layer_idx] = {}
        for name in MATRIX_NAMES:
            sal_arr = salience.get(layer_idx, {}).get(name)
            if sal_arr is None:
                # Uncalibrated layer: use median bit-width from calibrated layers
                # We don't know n_groups here without the store, so skip
                continue
            arr = np.array([raw_bits.get((layer_idx, name, g), 2)
                            for g in range(len(sal_arr))], dtype=np.int32)
            bw[layer_idx][name] = arr

    return bw, achieved_avg


# ── SQC: per-group scale calibration ──────────────────────────────────────────

def calibrate_matrix_scales_vectorized(
    packed_matrix: np.ndarray,   # (rows, cols//2) uint8
    orig_scales: np.ndarray,     # (n_groups,) float32
    bw_arr: np.ndarray,          # (n_groups,) int32
    rows: int,
    cols: int,
    group_size: int,
    n_iter: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized SQC for an entire weight matrix.

    For each group assigned < 4 bits, find the optimal scale by grid search.
    Uses numpy broadcasting to process all groups simultaneously.

    Returns: (new_scales, zero_points) both shape (n_groups,)
    """
    n_groups_per_row = cols // group_size
    n_groups = rows * n_groups_per_row
    orig_zp = 8  # Q4 symmetric zero_point

    new_scales = orig_scales.copy()
    zero_points = np.full(n_groups, orig_zp, dtype=np.int32)

    # Find groups that need recalibration (bits < 4)
    needs_recal = bw_arr < 4
    if not np.any(needs_recal):
        return new_scales, zero_points

    # Unpack entire matrix to float32 (group-by-group to save memory)
    # packed_matrix: (rows, cols//2) uint8
    # Unpack nibbles: lo = even cols, hi = odd cols
    lo = (packed_matrix & 0x0F).astype(np.float32)  # (rows, cols//2)
    hi = ((packed_matrix >> 4) & 0x0F).astype(np.float32)

    # Reconstruct full weight matrix: (rows, cols)
    w_int = np.empty((rows, cols), dtype=np.float32)
    w_int[:, 0::2] = lo
    w_int[:, 1::2] = hi

    # Dequantize using original Q4 scales
    # Reshape to (rows, n_groups_per_row, group_size)
    w_grouped = w_int.reshape(rows, n_groups_per_row, group_size)
    sc_grouped = orig_scales.reshape(rows, n_groups_per_row)
    w_f32 = (w_grouped - orig_zp) * sc_grouped[:, :, np.newaxis]  # (rows, n_groups_per_row, gs)

    # For each unique bit-width, process all groups of that width together
    for bits in [2, 3]:
        n_levels = 2 ** bits
        zp = n_levels // 2

        # Find groups with this bit-width
        bw_grouped = bw_arr.reshape(rows, n_groups_per_row)
        mask = (bw_grouped == bits)  # (rows, n_groups_per_row)

        if not np.any(mask):
            continue

        # For each group, compute default scale from weight range
        w_min = w_f32.min(axis=2)  # (rows, n_groups_per_row)
        w_max = w_f32.max(axis=2)
        w_range = w_max - w_min
        default_scales = np.where(w_range > 1e-10, w_range / (n_levels - 1), 1e-6)

        # Grid search: try n_iter scale factors
        best_scales = default_scales.copy()
        best_mse = np.full_like(default_scales, float('inf'))

        for factor in np.linspace(0.8, 1.2, n_iter):
            trial_scales = default_scales * factor  # (rows, n_groups_per_row)
            trial_scales = np.maximum(trial_scales, 1e-10)

            # Quantize and reconstruct
            s_exp = trial_scales[:, :, np.newaxis]  # (rows, n_groups_per_row, 1)
            w_q = np.clip(np.round(w_f32 / s_exp + zp), 0, n_levels - 1)
            w_recon = (w_q - zp) * s_exp
            mse = np.mean((w_f32 - w_recon) ** 2, axis=2)  # (rows, n_groups_per_row)

            # Update best where this trial is better AND group has this bit-width
            improved = (mse < best_mse) & mask
            best_scales = np.where(improved, trial_scales, best_scales)
            best_mse = np.where(improved, mse, best_mse)

        # Write back calibrated scales for this bit-width
        flat_mask = mask.ravel()
        flat_best = best_scales.ravel()
        new_scales[flat_mask] = flat_best[flat_mask]
        zero_points[bw_arr == bits] = zp

    del w_int, w_grouped, w_f32
    gc.collect()

    return new_scales, zero_points


def run_sqc_calibration(
    store,
    bitwidths: dict[int, dict[str, np.ndarray]],
    salience: dict[int, dict[str, np.ndarray]],
    n_layers: int,
    group_size: int,
    quick: bool = False,
    total_layers: int = NUM_LAYERS,
) -> dict[int, dict[str, dict]]:
    """
    Vectorized SQC calibration for all layers.

    Returns: groups_meta[layer][name] = {
        "bits": np.ndarray uint8 (n_groups,),
        "scales": np.ndarray float32 (n_groups,),
        "zero_points": np.ndarray uint8 (n_groups,),
    }
    """
    n_sqc_iter = 5 if quick else 10
    groups_meta: dict[int, dict[str, dict]] = {}

    total_groups_processed = 0
    total_groups_changed = 0

    for layer_idx in range(total_layers):
        is_calibrated = layer_idx < n_layers
        groups_meta[layer_idx] = {}

        for name in MATRIX_NAMES:
            key = (layer_idx, name)
            if key not in store._quant_sc:
                continue

            sc_tensor = store._quant_sc[key].float().numpy()
            rows, cols = store._quant_shapes[key]
            n_groups_per_row = cols // group_size
            n_groups = rows * n_groups_per_row

            bw_arr = bitwidths.get(layer_idx, {}).get(name, np.full(n_groups, 4, dtype=np.int32))

            if is_calibrated and np.any(bw_arr < 4):
                packed_tensor = store._quant_packed[key].numpy()
                new_scales, new_zps = calibrate_matrix_scales_vectorized(
                    packed_tensor, sc_tensor, bw_arr,
                    rows, cols, group_size, n_iter=n_sqc_iter
                )
                total_groups_changed += int(np.sum(bw_arr < 4))
            else:
                new_scales = sc_tensor.copy()
                new_zps = np.where(
                    bw_arr == 4, 8,
                    np.where(bw_arr == 3, 4, 2)
                ).astype(np.int32)

            groups_meta[layer_idx][name] = {
                "bits": bw_arr.astype(np.uint8),
                "scales": new_scales.astype(np.float32),
                "zero_points": new_zps.astype(np.uint8),
            }
            total_groups_processed += n_groups

        if layer_idx % 8 == 0:
            log_memory(f"after layer {layer_idx} SQC")
            gc.collect()

    print(f"  SQC: {total_groups_processed} groups processed, "
          f"{total_groups_changed} recalibrated")
    return groups_meta


# ── output assembly ────────────────────────────────────────────────────────────

def compute_estimated_size(groups_meta: dict, group_size: int) -> tuple[float, dict]:
    """Estimate model size after mixed-precision quantization."""
    total_bits = 0
    total_params = 0
    counts = {2: 0, 3: 0, 4: 0}

    for layer_idx, layer_data in groups_meta.items():
        for name, arrays in layer_data.items():
            if not arrays:
                continue
            bits_arr = arrays["bits"]
            n_groups = len(bits_arr)
            for b in [2, 3, 4]:
                n = int(np.sum(bits_arr == b))
                counts[b] = counts.get(b, 0) + n
                total_bits += n * b * group_size
            total_params += n_groups * group_size

    if total_params == 0:
        return 0.0, counts

    # Add embedding (kept at FP16): ~1.2 GB
    embed_gb = 1.2
    weight_gb = (total_bits / 8) / 1e9 + embed_gb

    return weight_gb, counts


def build_output_json(
    groups_meta: dict,
    achieved_avg_bits: float,
    group_size: int,
    n_prompts: int,
    quick: bool,
    output_path: Path,
    target_avg_bits: float = 2.2,
) -> dict:
    """
    Build compact output JSON. Per-group data is stored as a separate .npz file
    to avoid OOM on JSON serialization of 100M+ groups.

    The JSON contains summary statistics and per-layer/matrix bit-width counts.
    The .npz file contains the actual scale and zero_point arrays.
    """
    estimated_size_gb, counts = compute_estimated_size(groups_meta, group_size)
    total_groups = sum(counts.values())
    q4_size_gb = 7.5

    size_reduction_pct = (1.0 - estimated_size_gb / q4_size_gb) * 100.0

    # Build compact per-layer summary (not per-group)
    layers_summary: dict = {}
    npz_data: dict = {}  # arrays to save in .npz

    for layer_idx, layer_data in groups_meta.items():
        layers_summary[str(layer_idx)] = {}
        for name, arrays in layer_data.items():
            if not arrays:
                continue
            bits_arr = arrays["bits"]
            scale_arr = arrays["scales"]
            zp_arr = arrays["zero_points"]

            n4 = int(np.sum(bits_arr == 4))
            n3 = int(np.sum(bits_arr == 3))
            n2 = int(np.sum(bits_arr == 2))
            avg_b = float(np.mean(bits_arr))

            layers_summary[str(layer_idx)][name] = {
                "n_groups": int(len(bits_arr)),
                "groups_4bit": n4,
                "groups_3bit": n3,
                "groups_2bit": n2,
                "avg_bits": round(avg_b, 3),
            }

            # Store arrays in npz
            prefix = f"L{layer_idx}_{name}"
            npz_data[f"{prefix}_bits"] = bits_arr
            npz_data[f"{prefix}_scales"] = scale_arr
            npz_data[f"{prefix}_zp"] = zp_arr

    # Save .npz file alongside JSON
    npz_path = output_path.with_suffix(".npz")
    np.savez_compressed(str(npz_path), **npz_data)
    print(f"  Saved weight arrays to {npz_path} ({npz_path.stat().st_size / 1e6:.1f} MB)")

    output = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "phi4-14b",
        "target_avg_bits": target_avg_bits,
        "achieved_avg_bits": round(achieved_avg_bits, 4),
        "group_size": group_size,
        "calibration_prompts_used": n_prompts,
        "quick_mode": quick,
        "npz_path": str(npz_path.name),
        "layers": layers_summary,
        "statistics": {
            "groups_at_4bit": counts.get(4, 0),
            "groups_at_3bit": counts.get(3, 0),
            "groups_at_2bit": counts.get(2, 0),
            "total_groups": total_groups,
            "estimated_model_size_gb": round(estimated_size_gb, 2),
            "size_reduction_vs_q4_pct": round(size_reduction_pct, 1),
        },
    }

    return output


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SliM-LLM calibration for ASDSL")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 4 prompts, 4 layers only")
    parser.add_argument("--threads", type=int, default=0,
                        help="Thread count (0=auto)")
    parser.add_argument("--output", type=str, default=str(ROOT / "phi4_slim_meta.json"),
                        help="Output path for phi4_slim_meta.json")
    args = parser.parse_args()

    output_path = Path(args.output)
    quick = args.quick
    n_layers = 4 if quick else NUM_LAYERS
    prompts = CALIBRATION_PROMPTS[:4] if quick else CALIBRATION_PROMPTS

    print("=" * 60)
    print(f"ASDSL Phase 2 — SliM Calibration {'(QUICK MODE)' if quick else ''}")
    print("=" * 60)

    # Step 1: Memory guard
    # Quick mode uses less memory (4 layers only), so lower threshold
    required_gb = 3.0 if quick else 5.0
    check_memory_headroom(required_gb=required_gb)

    # Step 2: Set thread count
    if args.threads > 0:
        n_threads = args.threads
    else:
        n_threads = max(1, (os.cpu_count() or 4) // 2)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(n_threads)
    print(f"Using {n_threads} threads")

    # Step 3: Load model
    print("\nLoading Phi-4 Q4 weights...")
    t_load_start = time.perf_counter()

    try:
        import torch
        from transformers import AutoTokenizer
        from experiments.phi4_cpu_run import WeightStore, MODEL_DIR, INDEX_FILE

        if not INDEX_FILE.exists():
            print(f"ERROR: Model not found at {INDEX_FILE}")
            print("Run with model weights present, or use --quick for testing.")
            sys.exit(1)

        # Tokenizer is loaded from HuggingFace Hub (not local model dir)
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-4-multimodal-instruct",
            trust_remote_code=True,
        )
        store = WeightStore(bits=4, enable_qcsd=False)
        store.load()

    except Exception as e:
        print(f"ERROR loading model: {e}")
        sys.exit(1)

    t_load = time.perf_counter() - t_load_start
    print(f"Model loaded in {t_load:.1f}s")
    log_memory("after model load")

    group_size = store.group_size
    print(f"Group size: {group_size}")

    # Step 4: Collect hidden states
    print(f"\nCollecting hidden states ({len(prompts)} prompts, {n_layers} layers)...")
    t_collect_start = time.perf_counter()

    hidden_states = collect_hidden_states(
        store, tokenizer, prompts, n_layers, quick=quick
    )

    t_collect = time.perf_counter() - t_collect_start
    print(f"Hidden state collection: {t_collect:.1f}s")
    log_memory("after hidden state collection")

    # Step 5: Compute salience
    print("\nComputing salience scores...")
    salience = compute_salience(hidden_states, store, n_layers, group_size)
    del hidden_states
    gc.collect()
    log_memory("after salience computation")

    # Step 6: Assign bit-widths
    print("\nAssigning bit-widths...")
    result = assign_bitwidths(salience, n_layers, target_avg_bits=2.2)
    if isinstance(result, tuple):
        bitwidths, achieved_avg = result
    else:
        bitwidths = result
        # Compute achieved avg manually
        total_bits = sum(
            int(b) for layer_data in bitwidths.values()
            for arr in layer_data.values()
            for b in arr
        )
        total_groups = sum(
            len(arr) for layer_data in bitwidths.values()
            for arr in layer_data.values()
        )
        achieved_avg = total_bits / max(total_groups, 1)

    # Step 7: SQC calibration
    print("\nRunning SQC scale calibration...")
    t_sqc_start = time.perf_counter()
    groups_meta = run_sqc_calibration(
        store, bitwidths, salience, n_layers, group_size,
        quick=quick, total_layers=NUM_LAYERS
    )
    t_sqc = time.perf_counter() - t_sqc_start
    print(f"SQC calibration: {t_sqc:.1f}s")
    log_memory("after SQC")

    # Step 8: Build and write output JSON
    print("\nBuilding output JSON...")
    output_data = build_output_json(
        groups_meta, achieved_avg, group_size,
        n_prompts=len(prompts), quick=quick,
        output_path=output_path,
    )

    output_path.write_text(
        json.dumps(output_data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"Written to {output_path}")

    # Validate
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    avg = loaded["achieved_avg_bits"]
    size = loaded["statistics"]["estimated_model_size_gb"]
    total = loaded["statistics"]["total_groups"]
    print(f"\nValidation: {total} groups, {avg:.3f} avg bits, {size:.2f} GB estimated")

    if not quick:
        assert 1.8 <= avg <= 2.6, f"avg bits {avg} out of expected range [1.8, 2.6]"
        assert 3.0 <= size <= 5.5, f"estimated size {size} GB out of expected range [3.0, 5.5]"
        print("All assertions passed")
    else:
        print("Quick mode: skipping range assertions")

    print("\n" + "=" * 60)
    print("SliM calibration complete")
    print(f"  Avg bits: {avg:.3f} (target: 2.2)")
    print(f"  Estimated size: {size:.2f} GB (vs 7.5 GB Q4)")
    print(f"  Size reduction: {loaded['statistics']['size_reduction_vs_q4_pct']:.1f}%")
    print(f"  2-bit groups: {loaded['statistics']['groups_at_2bit']}")
    print(f"  3-bit groups: {loaded['statistics']['groups_at_3bit']}")
    print(f"  4-bit groups: {loaded['statistics']['groups_at_4bit']}")
    if quick:
        print("  NOTE: quick_mode=true — rerun without --quick for production")
    print("=" * 60)


if __name__ == "__main__":
    main()
