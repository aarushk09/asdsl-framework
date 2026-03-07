"""Example: System capabilities and CPU feature detection.

Shows how to detect CPU features and configure the framework for your hardware.
"""

from asdsl.config import (
    detect_cpu_features,
    get_system_info,
    PHI3_MINI_CONFIG,
    QuantizationConfig,
    InferenceConfig,
)
from asdsl.kernels.simd import select_backend
from asdsl.lut.engine import estimate_lut_memory


def main() -> None:
    # Detect CPU capabilities
    print("=== System Information ===")
    info = get_system_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    print()

    # CPU feature detection
    print("=== CPU Features ===")
    features = detect_cpu_features()
    for feature in features:
        print(f"  ✓ {feature.value}")
    print()

    # SIMD backend
    backend = select_backend()
    print(f"=== Selected SIMD Backend: {backend.name} ===")
    print()

    # Model configuration
    print("=== Phi-3-mini Configuration ===")
    cfg = PHI3_MINI_CONFIG
    print(f"  Layers:     {cfg.num_layers}")
    print(f"  Hidden dim: {cfg.hidden_dim}")
    print(f"  Attn heads: {cfg.num_attention_heads}")
    print(f"  KV heads:   {cfg.num_kv_heads}")
    print(f"  Vocab size: {cfg.vocab_size}")
    print()

    # Memory estimates for different quantization settings
    print("=== Memory Estimates (Phi-3-mini) ===")
    for avg_bits in [2, 3, 3.5, 4, 8]:
        # Rough estimate: total params * avg_bits / 8
        total_params = 3.8e9
        model_size_gb = (total_params * avg_bits) / (8 * 1e9)
        print(f"  {avg_bits}-bit avg: ~{model_size_gb:.2f} GB")
    print()

    # LUT memory per layer
    print("=== LUT Memory per Layer ===")
    for bits in [2, 3, 4]:
        mem = estimate_lut_memory(
            out_features=cfg.hidden_dim,
            in_features=cfg.hidden_dim,
            bits=bits,
            group_size=128,
        )
        print(f"  {bits}-bit ({cfg.hidden_dim}x{cfg.hidden_dim}): "
              f"{mem['total_bytes']/1024:.1f} KB "
              f"(L1: {'✓' if mem['fits_l1'] else '✗'}, "
              f"L2: {'✓' if mem['fits_l2'] else '✗'})")

    # FFN layer (typically 4x wider)
    ffn_dim = cfg.hidden_dim * 4  # Approximate
    for bits in [2, 3, 4]:
        mem = estimate_lut_memory(
            out_features=ffn_dim,
            in_features=cfg.hidden_dim,
            bits=bits,
            group_size=128,
        )
        print(f"  {bits}-bit ({ffn_dim}x{cfg.hidden_dim}): "
              f"{mem['total_bytes']/1024:.1f} KB "
              f"(L1: {'✓' if mem['fits_l1'] else '✗'}, "
              f"L2: {'✓' if mem['fits_l2'] else '✗'})")


if __name__ == "__main__":
    main()
