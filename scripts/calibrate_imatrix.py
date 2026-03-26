#!/usr/bin/env python3
"""Build an importance matrix from calibration activations (WikiText-style outline).

Example workflow::

    # Collect (n_samples, K) activations for a layer input, then:
    from asdsl.quantization.imatrix import importance_from_activation_columns
    im = importance_from_activation_columns(hidden, group_size=128)

For full WikiText + HF integration, plug your dataloader here and stack rows.
"""

from __future__ import annotations

import argparse

import numpy as np

from asdsl.quantization.imatrix import importance_from_activation_columns


def main() -> None:
    p = argparse.ArgumentParser(description="Calibration imatrix (stub / demo)")
    p.add_argument("--out", type=str, default="imatrix_k.npy", help="Output .npy path")
    p.add_argument("--k", type=int, default=256, help="Hidden dimension K")
    p.add_argument("--samples", type=int, default=512, help="Synthetic calibration rows")
    args = p.parse_args()

    rng = np.random.default_rng(0)
    x = rng.standard_normal((args.samples, args.k)).astype(np.float32)
    im = importance_from_activation_columns(x, group_size=128)
    np.save(args.out, im)
    print(f"wrote {args.out} shape={im.shape}")


if __name__ == "__main__":
    main()
