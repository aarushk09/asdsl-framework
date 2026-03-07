"""
ASDSL - Asynchronous Salience-Driven Speculative Lookup Framework

High-performance CPU inference architecture for Small Language Models.
Achieves 35-55 tok/s on 2-4 CPU cores with under 2GB RAM through:
  - Salience-driven mixed-precision quantization (2-16 bit)
  - LUT-based matrix multiplication (eliminates FMA overhead)
  - SWIFT self-speculative decoding (1.3-1.6x speedup)
  - Asynchronous L2 cache prefetching
  - OS-level memory pinning and Huge Pages
"""

__version__ = "0.1.0"
