"""Quantization-Cascade Speculative Decoding (QCSD).

The Phi-4 CPU implementation (draft weight bank + batched target verify, KV
rollback) lives in ``experiments/phi4_cpu_run.py``. This module re-exports the
symbols expected by ``from asdsl.inference.speculative import ...``.
"""

from experiments.phi4_cpu_run import forward_layer_batch, generate_qcsd

__all__ = ["forward_layer_batch", "generate_qcsd"]
