"""Mixed-precision quantization pipeline.

Orchestrates salience analysis and per-group bit allocation to produce
a fully quantized model with protected salient weights. This is the
main entry point for the quantization subsystem.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from asdsl.config import ModelConfig, QuantizationConfig
from asdsl.quantization.core import (
    QuantizedTensor,
    compute_quantization_error,
    quantize_weights,
)
from asdsl.quantization.salience import (
    BitAllocation,
    SalienceMap,
    allocate_bits_by_salience,
    compute_gradient_salience,
    compute_hessian_salience,
)

logger = logging.getLogger(__name__)


@dataclass
class QuantizedLayer:
    """A single transformer layer with mixed-precision quantized weights."""

    layer_idx: int
    weights: dict[str, QuantizedTensor] = field(default_factory=dict)
    salience_maps: dict[str, SalienceMap] = field(default_factory=dict)
    bit_allocations: dict[str, BitAllocation] = field(default_factory=dict)

    @property
    def total_bytes(self) -> int:
        return sum(w.memory_bytes for w in self.weights.values())

    @property
    def average_bits(self) -> float:
        if not self.bit_allocations:
            bits = [w.bits for w in self.weights.values()]
            return float(np.mean(bits)) if bits else 0.0
        avg = [ba.average_bits for ba in self.bit_allocations.values()]
        return float(np.mean(avg)) if avg else 0.0


@dataclass
class QuantizedModel:
    """A fully quantized model ready for LUT-based inference."""

    config: ModelConfig
    quant_config: QuantizationConfig
    layers: list[QuantizedLayer] = field(default_factory=list)
    embedding_weights: QuantizedTensor | None = None
    lm_head_weights: QuantizedTensor | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def total_bytes(self) -> int:
        total = sum(layer.total_bytes for layer in self.layers)
        if self.embedding_weights:
            total += self.embedding_weights.memory_bytes
        if self.lm_head_weights:
            total += self.lm_head_weights.memory_bytes
        return total

    @property
    def total_bytes_mb(self) -> float:
        return self.total_bytes / (1024 * 1024)

    @property
    def average_bits(self) -> float:
        if not self.layers:
            return 0.0
        return float(np.mean([layer.average_bits for layer in self.layers]))


def quantize_model_mixed_precision(
    model: torch.nn.Module,
    model_config: ModelConfig,
    quant_config: QuantizationConfig,
    calibration_data: list[torch.Tensor] | None = None,
) -> QuantizedModel:
    """Quantize a full model using salience-driven mixed-precision.

    Pipeline:
    1. Compute salience maps for all weight tensors using calibration data
    2. Allocate per-group bit-widths based on salience scores
    3. Quantize each weight group at its assigned precision
    4. Package into QuantizedModel for LUT-based inference

    Args:
        model: Full-precision PyTorch model.
        model_config: Model architecture configuration.
        quant_config: Quantization hyperparameters.
        calibration_data: Input tensors for salience calibration.

    Returns:
        QuantizedModel ready for the LUT inference engine.
    """
    logger.info(
        "Starting mixed-precision quantization: default=%d-bit, salience=%d-bit",
        quant_config.default_bits,
        quant_config.salience_bits,
    )

    quantized_model = QuantizedModel(
        config=model_config,
        quant_config=quant_config,
        metadata={"method": "salience_mixed_precision"},
    )

    # Step 1: Compute salience maps
    salience_maps: dict[str, SalienceMap] = {}
    if calibration_data and quant_config.salience_threshold > 0:
        logger.info("Computing gradient salience over %d calibration samples", len(calibration_data))
        salience_maps = compute_gradient_salience(
            model=model,
            calibration_data=calibration_data,
            group_size=quant_config.group_size,
        )
        logger.info("Salience computed for %d weight tensors", len(salience_maps))

    # Step 2: Iterate over all layers and quantize
    layer_params: dict[int, dict[str, torch.Tensor]] = {}
    embedding_params: dict[str, torch.Tensor] = {}
    head_params: dict[str, torch.Tensor] = {}

    for name, param in model.named_parameters():
        if param.ndim < 2:
            continue  # Skip biases and 1D params

        if "embed" in name.lower():
            embedding_params[name] = param
        elif "lm_head" in name.lower() or "output" in name.lower():
            head_params[name] = param
        else:
            # Parse layer index from name
            layer_idx = _extract_layer_index(name)
            if layer_idx is not None:
                if layer_idx not in layer_params:
                    layer_params[layer_idx] = {}
                layer_params[layer_idx][name] = param

    # Quantize embedding (always 8-bit for accuracy)
    for name, param in embedding_params.items():
        logger.info("Quantizing embedding %s at 8-bit", name)
        quantized_model.embedding_weights = quantize_weights(
            param, bits=8, group_size=quant_config.group_size
        )

    # Quantize each transformer layer
    for layer_idx in sorted(layer_params.keys()):
        qlayer = _quantize_layer(
            layer_idx=layer_idx,
            params=layer_params[layer_idx],
            salience_maps=salience_maps,
            quant_config=quant_config,
        )
        quantized_model.layers.append(qlayer)

    # Quantize LM head (8-bit for output quality)
    for name, param in head_params.items():
        logger.info("Quantizing LM head %s at 8-bit", name)
        quantized_model.lm_head_weights = quantize_weights(
            param, bits=8, group_size=quant_config.group_size
        )

    logger.info(
        "Quantization complete: %.1f MB, avg %.2f bits",
        quantized_model.total_bytes_mb,
        quantized_model.average_bits,
    )

    return quantized_model


def _quantize_layer(
    layer_idx: int,
    params: dict[str, torch.Tensor],
    salience_maps: dict[str, SalienceMap],
    quant_config: QuantizationConfig,
) -> QuantizedLayer:
    """Quantize a single transformer layer with mixed precision."""
    qlayer = QuantizedLayer(layer_idx=layer_idx)

    for name, param in params.items():
        smap = salience_maps.get(name)

        if smap is not None:
            # Use salience-driven bit allocation
            target_bits = _compute_target_bits(name, quant_config)
            allocation = allocate_bits_by_salience(
                salience_map=smap,
                target_avg_bits=target_bits,
                min_bits=quant_config.default_bits,
                max_bits=quant_config.salience_bits,
            )
            qlayer.salience_maps[name] = smap
            qlayer.bit_allocations[name] = allocation

            # Quantize with mixed precision per group
            qtensor = _quantize_mixed_precision(
                param, allocation, quant_config.group_size
            )
        else:
            # Uniform quantization at default bits
            qtensor = quantize_weights(
                param,
                bits=quant_config.default_bits,
                group_size=quant_config.group_size,
            )

        qlayer.weights[name] = qtensor
        logger.debug(
            "  Layer %d: %s -> %d bytes (%.2f avg bits)",
            layer_idx,
            name,
            qtensor.memory_bytes,
            allocation.average_bits if smap else quant_config.default_bits,
        )

    return qlayer


def _quantize_mixed_precision(
    weights: torch.Tensor,
    allocation: BitAllocation,
    group_size: int,
) -> QuantizedTensor:
    """Quantize a tensor with per-group mixed bit-widths.

    For groups assigned different bit-widths, we quantize each group
    independently and store them with the maximum bit-width's packing.
    The bit allocation map is stored alongside for the LUT engine to
    select the correct lookup table per group.
    """
    w_np = weights.detach().cpu().float().numpy()
    original_shape = w_np.shape
    flat = w_np.reshape(-1)

    # Pad
    remainder = len(flat) % group_size
    if remainder:
        flat = np.concatenate([flat, np.zeros(group_size - remainder, dtype=np.float32)])

    num_groups = len(flat) // group_size
    grouped = flat.reshape(num_groups, group_size)

    # Quantize each group at its assigned precision
    # For storage, we use the dominant bit-width
    dominant_bits = int(np.median(allocation.bits_per_group))

    all_scales = []
    all_quantized = []

    for g in range(min(num_groups, len(allocation.bits_per_group))):
        bits = int(allocation.bits_per_group[g])
        group_data = grouped[g : g + 1]

        qmax = (1 << bits) - 1
        abs_max = np.maximum(np.abs(group_data).max(), 1e-10)
        scale = abs_max / (qmax / 2)
        quantized = np.round(group_data / scale + qmax / 2)
        quantized = np.clip(quantized, 0, qmax).astype(np.uint8)

        # If bits differ from dominant, re-encode to dominant width
        if bits != dominant_bits:
            # Dequantize and requantize at dominant width
            dequant = (quantized.astype(np.float32) - qmax / 2) * scale
            dom_qmax = (1 << dominant_bits) - 1
            dom_scale = abs_max / (dom_qmax / 2)
            quantized = np.round(dequant / dom_scale + dom_qmax / 2)
            quantized = np.clip(quantized, 0, dom_qmax).astype(np.uint8)
            scale = dom_scale

        all_scales.append(np.float16(scale))
        all_quantized.append(quantized.reshape(-1))

    all_quantized_arr = np.concatenate(all_quantized)
    scales_arr = np.array(all_scales, dtype=np.float16)

    # Pack using dominant bits
    from asdsl.quantization.core import _pack_bits

    packed = _pack_bits(all_quantized_arr, dominant_bits)

    return QuantizedTensor(
        data=packed,
        scales=scales_arr,
        zeros=None,
        bits=dominant_bits,
        group_size=group_size,
        shape=original_shape,
        is_symmetric=True,
    )


def _compute_target_bits(param_name: str, config: QuantizationConfig) -> float:
    """Compute target average bits for a parameter based on its role.

    Down-projection and attention output layers get slightly more bits
    since they're empirically more sensitive to quantization error.
    """
    name_lower = param_name.lower()

    # FFN down-projection is highly sensitive
    if "down_proj" in name_lower or "fc2" in name_lower:
        return config.default_bits + 1.0

    # Attention output projection
    if "o_proj" in name_lower or "out_proj" in name_lower:
        return config.default_bits + 0.5

    # Q/K projections affect attention patterns
    if "q_proj" in name_lower or "k_proj" in name_lower:
        return config.default_bits + 0.5

    return float(config.default_bits)


def _extract_layer_index(param_name: str) -> int | None:
    """Extract transformer layer index from a parameter name."""
    import re

    match = re.search(r"layers?[._](\d+)", param_name, re.IGNORECASE)
    if match:
        return int(match.group(1))

    match = re.search(r"h[._](\d+)", param_name)
    if match:
        return int(match.group(1))

    return None


def save_quantized_model(model: QuantizedModel, path: str | Path) -> None:
    """Save a quantized model to disk in the ASDSL format.

    Format: numpy npz archive with packed weights, scales, and metadata.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    save_dict: dict[str, np.ndarray] = {}

    # Metadata
    meta = {
        "model_name": model.config.name,
        "num_layers": model.config.num_layers,
        "hidden_dim": model.config.hidden_dim,
        "default_bits": model.quant_config.default_bits,
        "salience_bits": model.quant_config.salience_bits,
        "group_size": model.quant_config.group_size,
        "total_bytes": model.total_bytes,
        "average_bits": model.average_bits,
    }
    save_dict["metadata"] = np.array(str(meta), dtype=np.bytes_)

    # Embedding
    if model.embedding_weights:
        save_dict["embedding.data"] = model.embedding_weights.data
        save_dict["embedding.scales"] = model.embedding_weights.scales
        save_dict["embedding.shape"] = np.array(model.embedding_weights.shape)

    # Layers
    for layer in model.layers:
        prefix = f"layer.{layer.layer_idx}"
        for wname, qtensor in layer.weights.items():
            safe_name = wname.replace(".", "_")
            save_dict[f"{prefix}.{safe_name}.data"] = qtensor.data
            save_dict[f"{prefix}.{safe_name}.scales"] = qtensor.scales
            save_dict[f"{prefix}.{safe_name}.shape"] = np.array(qtensor.shape)
            save_dict[f"{prefix}.{safe_name}.bits"] = np.array(qtensor.bits)
            if qtensor.zeros is not None:
                save_dict[f"{prefix}.{safe_name}.zeros"] = qtensor.zeros

        # Bit allocations
        for wname, alloc in layer.bit_allocations.items():
            safe_name = wname.replace(".", "_")
            save_dict[f"{prefix}.{safe_name}.bit_alloc"] = alloc.bits_per_group

    # LM head
    if model.lm_head_weights:
        save_dict["lm_head.data"] = model.lm_head_weights.data
        save_dict["lm_head.scales"] = model.lm_head_weights.scales
        save_dict["lm_head.shape"] = np.array(model.lm_head_weights.shape)

    np.savez_compressed(str(path), **save_dict)
    logger.info("Saved quantized model to %s (%.1f MB)", path, path.stat().st_size / 1e6)
