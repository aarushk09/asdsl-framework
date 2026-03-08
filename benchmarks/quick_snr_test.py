"""Quick SNR comparison across group sizes and quantization modes."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['USE_TF'] = '0'
os.environ['USE_JAX'] = '0'

import torch
import numpy as np
from asdsl.quantization.core import quantize_weights, compute_quantization_error

torch.manual_seed(42)
w = torch.randn(3072, 3072) * 0.02  # typical LLM weight scale

print("=== 3-bit Quantization Quality ===")
for gs in [128, 64, 32, 16]:
    for sym, opt in [(True, False), (False, True)]:
        qt = quantize_weights(w.numpy(), bits=3, group_size=gs, symmetric=sym, optimize_clips=opt)
        err = compute_quantization_error(w, qt)
        label = 'sym' if sym else 'asym+opt'
        print(f"  gs={gs:3d} {label:10s}: SNR={err['snr_db']:6.2f} dB  MSE={err['mse']:.6f}  comp={err['compression_ratio']:.1f}x")

print("\n=== 4-bit Quantization Quality ===")
for gs in [128, 64, 32, 16]:
    for sym, opt in [(True, False), (False, True)]:
        qt = quantize_weights(w.numpy(), bits=4, group_size=gs, symmetric=sym, optimize_clips=opt)
        err = compute_quantization_error(w, qt)
        label = 'sym' if sym else 'asym+opt'
        print(f"  gs={gs:3d} {label:10s}: SNR={err['snr_db']:6.2f} dB  MSE={err['mse']:.6f}  comp={err['compression_ratio']:.1f}x")

print("\n=== 8-bit Quantization Quality ===")
for gs in [128, 64, 32]:
    qt = quantize_weights(w.numpy(), bits=8, group_size=gs, symmetric=True)
    err = compute_quantization_error(w, qt)
    print(f"  gs={gs:3d} sym       : SNR={err['snr_db']:6.2f} dB  MSE={err['mse']:.6f}  comp={err['compression_ratio']:.1f}x")
