"""HEP model export: converts existing Phi-4 weight store to HEP format.

Input:  phi4_slim_meta.npz  (full FP16 weights, 190 MB on disk)
Output: phi4_14b_hep_r{R}.bin  (coefficient store)
        phi4_14b_hep_r{R}_meta.json (offset/shape metadata for bin file)

Binary layout of .bin file
--------------------------
For each tensor (in metadata order):
  [seeds]  : out_dim × 8 bytes (uint64 LE)
  [alphas] : out_dim × n_groups × rank bytes (int8)

Non-quantized tensors (embed, norm, lm_head) are stored as FP16 directly —
they do not benefit from HEP synthesis because they are accessed
non-sequentially (embedding lookup) or are small (norms).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .codec import HEPTensor, hep_encode, reconstruction_snr


# Tensors that are HEP-encoded (large projection weight matrices)
_HEP_SUFFIX_WHITELIST = (
    "qkv_proj.weight",
    "o_proj.weight",
    "gate_up_proj.weight",
    "down_proj.weight",
)

# Tensors stored as-is (FP16) — small or non-sequential access
_PASSTHROUGH_SUFFIXES = (
    "embed_tokens.weight",
    "norm.weight",
    "lm_head.weight",
    "layernorm",
    "input_layernorm",
    "post_attention_layernorm",
)


def _is_hep_candidate(key: str) -> bool:
    return any(key.endswith(s) for s in _HEP_SUFFIX_WHITELIST)


def _is_passthrough(key: str) -> bool:
    return any(s in key for s in _PASSTHROUGH_SUFFIXES)


def export_hep_model(
    npz_path: str,
    output_dir: str,
    rank: int = 4,
    group_size: int = 128,
    energy_threshold: float = 0.98,
    adaptive_rank: bool = False,
    verbose: bool = True,
) -> dict:
    """Convert Phi-4 FP16 weight store to HEP coefficient format.

    Args:
        npz_path:         Path to phi4_slim_meta.npz.
        output_dir:       Directory to write output files.
        rank:             Fixed basis rank R (used when adaptive_rank=False).
        group_size:       Elements per quantization group.
        energy_threshold: Energy fraction target for adaptive rank.
        adaptive_rank:    If True, per-row adaptive rank (slow at export time).
        verbose:          Print progress.

    Returns:
        metadata dict (also saved to JSON).
    """
    npz_path = Path(npz_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tag = f"r{rank}" if not adaptive_rank else f"adaptive_e{int(energy_threshold*100)}"
    bin_path  = output_dir / f"phi4_14b_hep_{tag}.bin"
    meta_path = output_dir / f"phi4_14b_hep_{tag}_meta.json"

    if verbose:
        print(f"Loading weight store from {npz_path} ...")
    data = np.load(npz_path, allow_pickle=True)
    keys = sorted(data.files)

    if verbose:
        print(f"Found {len(keys)} tensors. Encoding with rank={rank}, group_size={group_size}.")

    metadata: dict[str, dict] = {}
    snr_log: list[dict] = []
    offset = 0

    with open(bin_path, "wb") as fout:
        for key in keys:
            w = data[key]
            if isinstance(w, np.ndarray) and w.dtype == object:
                # Skip non-array entries
                continue

            t0 = time.time()

            if _is_hep_candidate(key) and w.ndim == 2:
                # Convert FP16 → FP32 for encoding
                w_f32 = w.astype(np.float32)

                if adaptive_rank:
                    # (slow — per-row rank selection)
                    from .codec import adaptive_rank_selector
                    row_ranks = adaptive_rank_selector(w_f32, max_rank=rank, energy_threshold=energy_threshold, group_size=group_size)
                    effective_rank = int(np.median(row_ranks))
                else:
                    effective_rank = rank

                tensor = hep_encode(w_f32, rank=effective_rank, group_size=group_size)

                # Write seeds then alphas
                seed_bytes  = tensor.seeds.astype(np.uint64).tobytes()
                alpha_bytes = tensor.alphas.tobytes()

                fout.write(seed_bytes)
                fout.write(alpha_bytes)

                snr_info = reconstruction_snr(w_f32, tensor)
                snr_log.append({"key": key, **snr_info})

                metadata[key] = {
                    "type":       "hep",
                    "offset":     offset,
                    "seed_bytes": len(seed_bytes),
                    "alpha_offset": offset + len(seed_bytes),
                    "alpha_bytes": len(alpha_bytes),
                    "rank":       effective_rank,
                    "group_size": group_size,
                    "shape":      list(tensor.shape),
                    "n_groups":   tensor.n_groups_per_row,
                    "snr_db":     snr_info["snr_db"],
                    "effective_bits": snr_info["effective_bits"],
                }
                offset += len(seed_bytes) + len(alpha_bytes)

                if verbose:
                    print(
                        f"  {key:55s}  shape={str(w.shape):18s}  "
                        f"SNR={snr_info['snr_db']:6.1f} dB  "
                        f"{snr_info['effective_bits']:.3f} bits/param  "
                        f"[{time.time()-t0:.1f}s]"
                    )

            else:
                # Passthrough: store as FP16
                w_f16 = w.astype(np.float16)
                raw = w_f16.tobytes()
                fout.write(raw)
                metadata[key] = {
                    "type":   "fp16",
                    "offset": offset,
                    "size_bytes": len(raw),
                    "shape":  list(w.shape),
                    "dtype":  "float16",
                }
                offset += len(raw)

                if verbose:
                    print(f"  {key:55s}  shape={str(w.shape):18s}  [passthrough FP16]")

    # Summary statistics
    hep_entries = [v for v in metadata.values() if v["type"] == "hep"]
    avg_snr = float(np.mean([e["snr_db"] for e in hep_entries])) if hep_entries else 0.0
    avg_bits = float(np.mean([e["effective_bits"] for e in hep_entries])) if hep_entries else 0.0

    summary = {
        "format":           "hep",
        "rank":             rank,
        "group_size":       group_size,
        "adaptive_rank":    adaptive_rank,
        "energy_threshold": energy_threshold,
        "total_bytes":      offset,
        "total_mb":         offset / (1024 * 1024),
        "avg_snr_db":       avg_snr,
        "avg_effective_bits": avg_bits,
        "n_hep_tensors":    len(hep_entries),
        "tensors":          metadata,
    }

    with open(meta_path, "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"\n{'='*60}")
        print(f"HEP export complete:")
        print(f"  Output:        {bin_path}")
        print(f"  Metadata:      {meta_path}")
        print(f"  Total size:    {offset/(1024*1024):.1f} MB")
        print(f"  HEP tensors:   {len(hep_entries)}")
        print(f"  Avg SNR:       {avg_snr:.1f} dB")
        print(f"  Avg bits/param:{avg_bits:.3f}")
        print(f"{'='*60}")

    return summary
