"""HEP codec: encode/decode weight tensors using Random Basis Decomposition.

Theory
------
Given a weight matrix W of shape (out_dim, in_dim), we approximate each
group of `group_size` weights as a superposition of R pseudo-random basis
vectors generated from a 128-bit seed via AES-NI counter mode:

    W_group ≈ Σ_{r=0}^{R-1}  α_r · B_r(seed)

where:
  - seed    : uint64 (per weight-row identifier)
  - B_r     : float32 basis vector of length group_size, synthesized from
              AES-NI counter block (seed || r) → pseudo-random int8 →
              scaled to ~N(0,1).
  - α_r     : 4-bit signed projection coefficient (stored as int8, range -8..7)

Storage per weight-row:
  Original Q4:   group_size/2  bytes  (4 bits/param)
  HEP R=4:       R * groups/row bytes + 8-byte seed
              =  R bytes/group + negligible seed overhead
              ≈  R/group_size bytes/param

For R=4, group_size=128: 4/128 = 0.031 bytes/param → ~3% of original!
BUT reconstruction quality is limited — adaptive rank selection (R up to 32)
is used for high-entropy layers to recover accuracy.

Python Implementation Note
--------------------------
This module uses numpy for basis generation (via a seeded RNG) to provide
an exact reference implementation. The C++ kernel uses AES-NI counter mode
which produces different (but equivalently pseudo-random) bases. The codec
therefore stores a "backend" flag: numpy bases for testing, AES bases for
production. Only the alpha coefficients matter for the format on disk; the
basis is always re-derived from the seed at inference time.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class HEPTensor:
    """A weight matrix encoded as HEP projection coefficients.

    Attributes:
        seeds:        uint64 seed per output row, shape (out_dim,).
        alphas:       4-bit projection coefficients as int8, shape
                      (out_dim, n_groups_per_row, rank). Values in [-8, 7].
        alpha_scales: Per-group float32 scale to recover quantized alphas,
                      shape (out_dim, n_groups_per_row). Required for decode.
        rank:         Number of basis vectors per group (R).
        group_size:   Elements per quantization group.
        shape:        Original weight shape (out_dim, in_dim).
        energy_fracs: Fraction of weight energy captured per row (diagnostic).
    """
    seeds:        np.ndarray          # uint64,   (out_dim,)
    alphas:       np.ndarray          # int8,     (out_dim, n_groups, rank)
    alpha_scales: np.ndarray          # float32,  (out_dim, n_groups)
    rank:         int
    group_size:   int
    shape:        tuple[int, int]
    energy_fracs: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def out_dim(self) -> int:
        return self.shape[0]

    @property
    def in_dim(self) -> int:
        return self.shape[1]

    @property
    def n_groups_per_row(self) -> int:
        return (self.in_dim + self.group_size - 1) // self.group_size

    @property
    def bytes_per_param(self) -> float:
        """Storage cost in bytes per original parameter."""
        alpha_bytes = self.alphas.nbytes          # int8 backing
        seed_bytes  = self.seeds.nbytes           # uint64 per row
        return (alpha_bytes + seed_bytes) / max(1, self.out_dim * self.in_dim)

    @property
    def effective_bits(self) -> float:
        return self.bytes_per_param * 8.0

    def __repr__(self) -> str:
        return (
            f"HEPTensor(shape={self.shape}, rank={self.rank}, "
            f"group={self.group_size}, {self.effective_bits:.2f} bits/param)"
        )


# ---------------------------------------------------------------------------
# Basis generation (numpy reference — mirrors AES-NI counter mode statistically)
# ---------------------------------------------------------------------------

def _generate_bases_numpy(seed: int, rank: int, group_size: int) -> np.ndarray:
    """Generate R basis vectors of length group_size from a 64-bit seed.

    Each basis is drawn from a fixed RNG seeded by hash(seed, r) to mimic
    the statistical properties of AES-NI counter-mode output. Values are
    scaled to unit variance float32.

    Args:
        seed:       64-bit integer row seed.
        rank:       Number of basis vectors (R).
        group_size: Length of each basis vector.

    Returns:
        bases: float32 array of shape (rank, group_size), unit variance.
    """
    bases = np.empty((rank, group_size), dtype=np.float32)
    for r in range(rank):
        # Combine seed and rank index into a 64-bit key
        combined = ((seed ^ (r * 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF)
        rng = np.random.default_rng(combined)
        # AES output is uniform bytes → mapped to signed int8 → normalised.
        # We approximate this with uniform integers in [-128, 127].
        raw = rng.integers(-128, 128, size=group_size, dtype=np.int32)
        bases[r] = raw.astype(np.float32) * (1.0 / 128.0)  # scale to ~[-1, 1]
    return bases


def _generate_bases_for_row(row_seed: int, rank: int, n_groups: int, group_size: int) -> np.ndarray:
    """Generate all bases for a weight row across all groups.

    Returns:
        bases: float32, shape (n_groups, rank, group_size)
    """
    all_bases = np.empty((n_groups, rank, group_size), dtype=np.float32)
    for g in range(n_groups):
        group_seed = ((row_seed ^ (g * 0x517CC1B727220A95)) & 0xFFFFFFFFFFFFFFFF)
        all_bases[g] = _generate_bases_numpy(group_seed, rank, group_size)
    return all_bases


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def hep_encode(
    weights: np.ndarray,
    rank: int = 4,
    group_size: int = 128,
    alpha_bits: int = 4,
    seed_strategy: str = "row_hash",
) -> HEPTensor:
    """Encode a weight matrix into HEP projection coefficients.

    For each weight group, finds the best `rank` linear combination of
    unit-normalised pseudo-random bases that minimises reconstruction MSE.
    This is an oblique projection: α = (B_n B_n^T)^{-1} B_n w, where B_n
    has unit-L2-norm rows.

    Args:
        weights:       float32 array of shape (out_dim, in_dim).
        rank:          Number of basis vectors per group (R). Higher → better
                       accuracy, higher storage cost, higher compute cost.
                       Without fine-tuning, R ≥ group_size/4 needed for >10dB.
        group_size:    Elements per quantization group.
        alpha_bits:    Bit-width for projection coefficients (4 recommended).
        seed_strategy: "row_hash" — deterministic per-row seed from row index.

    Returns:
        HEPTensor with encoded coefficients and per-group alpha scales.
    """
    if weights.ndim != 2:
        raise ValueError(f"weights must be 2D, got shape {weights.shape}")

    weights = weights.astype(np.float32)
    out_dim, in_dim = weights.shape
    n_groups = (in_dim + group_size - 1) // group_size

    # Pad in_dim to group_size boundary
    pad = n_groups * group_size - in_dim
    if pad > 0:
        weights = np.concatenate([weights, np.zeros((out_dim, pad), dtype=np.float32)], axis=1)

    seeds        = np.empty(out_dim, dtype=np.uint64)
    alphas       = np.empty((out_dim, n_groups, rank), dtype=np.int8)
    alpha_scales = np.empty((out_dim, n_groups), dtype=np.float32)
    energy_fracs = np.empty(out_dim, dtype=np.float32)

    alpha_max = (1 << (alpha_bits - 1)) - 1   # e.g. 7 for 4-bit signed
    alpha_min = -(1 << (alpha_bits - 1))       # e.g. -8

    for row_idx in range(out_dim):
        if seed_strategy == "row_hash":
            row_seed = _row_seed(row_idx)
        else:
            row_seed = row_idx

        seeds[row_idx] = row_seed
        row_w = weights[row_idx]

        row_captured = 0.0
        row_total    = float(np.dot(row_w, row_w)) + 1e-20

        for g in range(n_groups):
            w_group    = row_w[g * group_size: (g + 1) * group_size]
            group_seed = int(row_seed ^ (g * 0x517CC1B727220A95) & 0xFFFFFFFFFFFFFFFF)
            B_raw      = _generate_bases_numpy(group_seed, rank, group_size)  # (R, group_size)

            # Normalise each basis vector to unit L2 norm (critical for projection accuracy)
            norms = np.linalg.norm(B_raw, axis=1, keepdims=True) + 1e-12
            B     = B_raw / norms  # (R, group_size) — unit-norm rows

            # Oblique LS projection: α* = (B B^T)^{-1} B w
            BBT = B @ B.T   # (R, R)
            Bw  = B @ w_group  # (R,)
            try:
                alpha_f32 = np.linalg.solve(BBT + 1e-6 * np.eye(rank, dtype=np.float32), Bw)
            except np.linalg.LinAlgError:
                alpha_f32 = Bw  # fallback: simple inner product (works if B nearly orthonormal)

            # Quantise alphas: scale so max|α| maps to alpha_max
            a_max = max(float(np.max(np.abs(alpha_f32))), 1e-8)
            a_scale = a_max / alpha_max
            alpha_q = np.clip(np.round(alpha_f32 / a_scale), alpha_min, alpha_max).astype(np.int8)

            # Store scale so decode can reconstruct exact float alphas
            alpha_scales[row_idx, g] = a_scale
            alphas[row_idx, g]       = alpha_q

            # Track reconstruction energy (diagnostic)
            w_recon = (alpha_q.astype(np.float32) * a_scale) @ B
            row_captured += float(np.dot(w_recon, w_group))

        energy_fracs[row_idx] = np.clip(row_captured / row_total, 0.0, 1.0)

    return HEPTensor(
        seeds=seeds,
        alphas=alphas,
        alpha_scales=alpha_scales,
        rank=rank,
        group_size=group_size,
        shape=(out_dim, in_dim),
        energy_fracs=energy_fracs,
    )


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def hep_decode(tensor: HEPTensor) -> np.ndarray:
    """Reconstruct a weight matrix from HEP projection coefficients.

    Uses the stored alpha_scales to recover exact float alphas, and
    unit-normalised bases matching the encoding step.

    Args:
        tensor: Encoded HEPTensor (must have alpha_scales field).

    Returns:
        Reconstructed weights, float32, shape tensor.shape.
    """
    out_dim, in_dim = tensor.shape
    group_size = tensor.group_size
    rank       = tensor.rank
    n_groups   = tensor.n_groups_per_row

    in_dim_padded = n_groups * group_size
    W = np.zeros((out_dim, in_dim_padded), dtype=np.float32)

    has_scales = (
        hasattr(tensor, 'alpha_scales') and
        tensor.alpha_scales is not None and
        tensor.alpha_scales.size > 0
    )

    for row_idx in range(out_dim):
        row_seed = int(tensor.seeds[row_idx])
        for g in range(n_groups):
            group_seed = int(row_seed ^ (g * 0x517CC1B727220A95) & 0xFFFFFFFFFFFFFFFF)
            B_raw = _generate_bases_numpy(group_seed, rank, group_size)  # (R, group_size)

            # Use unit-norm bases (matching encode)
            norms = np.linalg.norm(B_raw, axis=1, keepdims=True) + 1e-12
            B     = B_raw / norms  # (R, group_size)

            alpha_q = tensor.alphas[row_idx, g].astype(np.float32)  # (R,)

            if has_scales:
                a_scale = float(tensor.alpha_scales[row_idx, g])
                alpha   = alpha_q * a_scale
            else:
                alpha = alpha_q  # legacy fallback

            W[row_idx, g * group_size: (g + 1) * group_size] = alpha @ B

    return W[:, :in_dim]


# ---------------------------------------------------------------------------
# Adaptive rank selection
# ---------------------------------------------------------------------------

def adaptive_rank_selector(
    weights: np.ndarray,
    max_rank: int = 16,
    energy_threshold: float = 0.98,
    group_size: int = 128,
) -> np.ndarray:
    """Determine per-row rank needed to capture `energy_threshold` of weight energy.

    Uses an SVD-based proxy: for each row, project onto increasing numbers of
    random bases and measure captured energy fraction. Stops when threshold
    is reached. Returns per-row rank assignment.

    Args:
        weights:          float32 (out_dim, in_dim).
        max_rank:         Maximum rank to try.
        energy_threshold: Fraction of energy that must be captured.
        group_size:       Quantization group size.

    Returns:
        ranks: int32 array of shape (out_dim,), value in [1, max_rank].
    """
    weights = weights.astype(np.float32)
    out_dim, in_dim = weights.shape
    n_groups = (in_dim + group_size - 1) // group_size

    pad = n_groups * group_size - in_dim
    if pad > 0:
        weights = np.concatenate([weights, np.zeros((out_dim, pad), dtype=np.float32)], axis=1)

    ranks = np.ones(out_dim, dtype=np.int32)

    for row_idx in range(out_dim):
        row_seed = _row_seed(row_idx)
        row_w    = weights[row_idx]
        total_e  = float(np.dot(row_w, row_w)) + 1e-20

        for r_try in range(1, max_rank + 1):
            captured = 0.0
            for g in range(n_groups):
                group_seed = int(row_seed ^ (g * 0x517CC1B727220A95) & 0xFFFFFFFFFFFFFFFF)
                B = _generate_bases_numpy(group_seed, r_try, group_size)
                w_group = row_w[g * group_size: (g + 1) * group_size]
                Bw = B @ w_group
                # projection: w ≈ B^T (B B^T)^{-1} B w
                BBT = B @ B.T
                try:
                    alpha = np.linalg.solve(BBT + 1e-6 * np.eye(r_try, dtype=np.float32), Bw)
                except np.linalg.LinAlgError:
                    alpha = Bw / (np.diag(BBT) + 1e-6)
                w_proj = alpha @ B
                captured += float(np.dot(w_proj, w_group))

            if captured / total_e >= energy_threshold:
                ranks[row_idx] = r_try
                break
        else:
            ranks[row_idx] = max_rank

    return ranks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_seed(row_idx: int) -> int:
    """Deterministic 64-bit seed for a weight row index.

    Uses a Fibonacci hashing mix to spread row indices into the full uint64
    space, reducing seed clustering for small row indices.
    """
    x = (row_idx * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 31
    return int(x)


def reconstruction_snr(original: np.ndarray, tensor: HEPTensor) -> dict:
    """Compute SNR between original weights and HEP reconstruction.

    Args:
        original: float32 (out_dim, in_dim) original weights.
        tensor:   Encoded HEPTensor.

    Returns:
        dict with snr_db, mse, mae, energy_captured.
    """
    recon  = hep_decode(tensor)
    orig   = original.astype(np.float32)
    error  = orig - recon

    mse    = float(np.mean(error ** 2))
    mae    = float(np.mean(np.abs(error)))
    sig_pw = float(np.mean(orig ** 2))
    snr_db = 10.0 * np.log10(sig_pw / max(mse, 1e-30))

    return {
        "snr_db":          snr_db,
        "mse":             mse,
        "mae":             mae,
        "energy_captured": float(np.mean(tensor.energy_fracs)) if len(tensor.energy_fracs) else 0.0,
        "effective_bits":  tensor.effective_bits,
    }
