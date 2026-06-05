"""HEP validation tools: SNR analysis and perplexity comparison."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .codec import HEPTensor, hep_encode, hep_decode, reconstruction_snr


def validate_layer_snr(
    npz_path: str,
    keys: list[str] | None = None,
    rank: int = 4,
    group_size: int = 128,
    verbose: bool = True,
) -> list[dict]:
    """Run SNR validation on weight matrices from the NPZ store.

    Args:
        npz_path:  Path to phi4_slim_meta.npz or similar weight store.
        keys:      Specific tensor keys to validate (None = all 2D tensors).
        rank:      HEP basis rank.
        group_size: Elements per group.
        verbose:   Print results.

    Returns:
        List of per-tensor result dicts.
    """
    data = np.load(npz_path, allow_pickle=True)
    all_keys = sorted(data.files)

    if keys is not None:
        all_keys = [k for k in all_keys if k in keys]
    else:
        # Filter to large 2D tensors
        all_keys = [k for k in all_keys if data[k].ndim == 2 and data[k].size > 1024]

    results = []
    if verbose:
        header = f"{'Tensor':55s} {'Shape':18s} {'SNR(dB)':>8s} {'MSE':>10s} {'bits/p':>8s}"
        print(header)
        print("-" * len(header))

    for key in all_keys:
        w = data[key].astype(np.float32)
        t0 = time.time()
        tensor = hep_encode(w, rank=rank, group_size=group_size)
        stats  = reconstruction_snr(w, tensor)
        elapsed = time.time() - t0

        row = {
            "key":            key,
            "shape":          w.shape,
            "rank":           rank,
            "group_size":     group_size,
            "snr_db":         stats["snr_db"],
            "mse":            stats["mse"],
            "mae":            stats["mae"],
            "effective_bits": stats["effective_bits"],
            "encode_s":       elapsed,
        }
        results.append(row)

        if verbose:
            print(
                f"  {key:53s} {str(w.shape):18s} "
                f"{stats['snr_db']:8.2f} {stats['mse']:10.2e} "
                f"{stats['effective_bits']:8.3f}  [{elapsed:.1f}s]"
            )

    if verbose and results:
        avg_snr  = np.mean([r["snr_db"] for r in results])
        avg_bits = np.mean([r["effective_bits"] for r in results])
        print(f"\n  Average SNR: {avg_snr:.2f} dB   Average bits/param: {avg_bits:.3f}")

    return results


def rank_sweep(
    weights: np.ndarray,
    ranks: list[int] | None = None,
    group_size: int = 128,
) -> list[dict]:
    """Sweep over multiple rank values and report SNR + bits/param trade-off.

    Useful for choosing the right R for a given accuracy budget.

    Args:
        weights:    float32 (out_dim, in_dim) weight matrix.
        ranks:      List of rank values to test (default: [1,2,4,8,16,32]).
        group_size: Elements per group.

    Returns:
        List of result dicts, one per rank.
    """
    if ranks is None:
        ranks = [1, 2, 4, 8, 16, 32]

    results = []
    print(f"{'Rank':>6s} {'SNR(dB)':>10s} {'MSE':>12s} {'bits/param':>12s}")
    print("-" * 44)
    for r in ranks:
        tensor = hep_encode(weights, rank=r, group_size=group_size)
        stats  = reconstruction_snr(weights, tensor)
        results.append({"rank": r, **stats})
        print(
            f"  {r:4d} {stats['snr_db']:10.2f} {stats['mse']:12.2e} "
            f"{stats['effective_bits']:12.4f}"
        )

    return results


def print_summary(results: list[dict]) -> None:
    """Pretty-print a table of validation results."""
    if not results:
        print("No results.")
        return

    # Sort by SNR
    results = sorted(results, key=lambda x: x.get("snr_db", 0))
    print(f"\n{'='*80}")
    print(f"{'Tensor':45s} {'SNR':>8s} {'bits/p':>8s}")
    print(f"{'-'*80}")
    for r in results:
        print(f"  {r['key']:43s} {r['snr_db']:8.2f} {r['effective_bits']:8.3f}")

    snr_vals = [r["snr_db"] for r in results]
    print(f"\n  Min SNR: {min(snr_vals):.2f} dB   Max SNR: {max(snr_vals):.2f} dB   "
          f"Mean SNR: {np.mean(snr_vals):.2f} dB")
    print(f"{'='*80}")
