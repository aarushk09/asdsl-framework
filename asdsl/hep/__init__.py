"""Hyperdimensional Entropy Projection (HEP) package.

HEP shifts LLM weight delivery from DRAM streaming to local algorithmic
synthesis via AES-NI counter mode. Instead of reading ~7 GB of weight bytes
per token from DDR5, the engine keeps only small "projection coefficients"
(α_r, 4-bit) in L3 cache and reconstructs weight rows on-the-fly as a
superposition of pseudo-random AES-derived basis vectors.

Theoretical benefit on AVX2+AES-NI hardware:
  DDR5 ceiling (Q4):   ~11.4 tok/s  (48 GB/s / 3.5 GB/token weight stream)
  L3-synthesis ceiling: ~31  tok/s  (estimated for R=4 on i5-13500H)
  Expected gain:         2-3×

References:
  HEP paper (submitted): "Hyperdimensional Entropy Projection for CPU LLM Inference"
  SeedLM: https://arxiv.org/abs/2406.07610
"""

from .codec import HEPTensor, hep_encode, hep_decode, adaptive_rank_selector
from .engine import HEPEngine

__all__ = ["HEPTensor", "hep_encode", "hep_decode", "adaptive_rank_selector", "HEPEngine"]
