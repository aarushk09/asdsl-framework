"""Salience analysis for mixed-precision bit allocation.

Implements SliM-LLM style salience-driven bit allocation and TaCQ-inspired
task-circuit protection. Identifies critical weights (pivot weights, task
circuits) that require higher precision to maintain model accuracy under
aggressive quantization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class SalienceMap:
    """Per-group salience scores for a weight tensor.

    Attributes:
        scores: Salience score per quantization group, shape (num_groups,).
        group_size: Elements per group.
        tensor_shape: Original weight tensor shape.
        method: The salience method used ('gradient', 'hessian', 'activation').
    """

    scores: np.ndarray
    group_size: int
    tensor_shape: tuple[int, ...]
    method: str

    @property
    def num_groups(self) -> int:
        return len(self.scores)

    def get_top_k_groups(self, k: int) -> np.ndarray:
        """Return indices of the top-k most salient groups."""
        k = min(k, self.num_groups)
        return np.argsort(self.scores)[-k:]

    def get_threshold_groups(self, threshold: float) -> np.ndarray:
        """Return indices of groups with salience above threshold."""
        return np.where(self.scores >= threshold)[0]


@dataclass
class BitAllocation:
    """Per-group bit-width assignment for a weight tensor.

    Attributes:
        bits_per_group: Array of bit-widths, shape (num_groups,).
        group_size: Elements per group.
        tensor_shape: Original tensor shape.
        average_bits: Weighted average bits across the tensor.
    """

    bits_per_group: np.ndarray
    group_size: int
    tensor_shape: tuple[int, ...]

    @property
    def average_bits(self) -> float:
        return float(np.mean(self.bits_per_group))


def compute_gradient_salience(
    model: torch.nn.Module,
    calibration_data: list[torch.Tensor],
    group_size: int = 128,
    layer_names: list[str] | None = None,
) -> dict[str, SalienceMap]:
    """Compute gradient-based salience scores for model weight groups.

    Uses the magnitude of gradients w.r.t. weights as a proxy for
    how much each weight group contributes to the model output.
    Higher gradient magnitude = more salient = needs higher precision.

    Args:
        model: The full-precision model.
        calibration_data: List of input tensors for calibration.
        group_size: Quantization group size.
        layer_names: If specified, only analyze these layers.

    Returns:
        Dictionary mapping layer names to SalienceMap objects.
    """
    salience_maps: dict[str, SalienceMap] = {}

    # Accumulate gradient magnitudes over calibration data
    gradient_accum: dict[str, torch.Tensor] = {}

    model.eval()
    for param_name, param in model.named_parameters():
        if layer_names and not any(ln in param_name for ln in layer_names):
            continue
        if param.ndim < 2:
            continue
        param.requires_grad_(True)
        gradient_accum[param_name] = torch.zeros_like(param)

    for batch in calibration_data:
        model.zero_grad()
        if hasattr(model, "forward"):
            output = model(batch)
            if isinstance(output, tuple):
                logits = output[0]
            elif hasattr(output, "logits"):
                logits = output.logits
            else:
                logits = output

            # Use sum of logits as a simple loss for gradient computation
            loss = logits.sum()
            loss.backward()

        for param_name, param in model.named_parameters():
            if param_name in gradient_accum and param.grad is not None:
                gradient_accum[param_name] += param.grad.abs()

    # Convert gradient magnitudes to per-group salience scores
    num_samples = max(len(calibration_data), 1)
    for param_name, grad_sum in gradient_accum.items():
        param_shape = grad_sum.shape
        flat_grad = (grad_sum / num_samples).reshape(-1).cpu().numpy()

        # Pad to group boundary
        remainder = len(flat_grad) % group_size
        if remainder:
            flat_grad = np.concatenate(
                [flat_grad, np.zeros(group_size - remainder, dtype=np.float32)]
            )

        grouped = flat_grad.reshape(-1, group_size)
        # Group salience = mean gradient magnitude within group
        group_scores = grouped.mean(axis=1)

        # Normalize to [0, 1]
        score_max = group_scores.max()
        if score_max > 0:
            group_scores = group_scores / score_max

        salience_maps[param_name] = SalienceMap(
            scores=group_scores,
            group_size=group_size,
            tensor_shape=param_shape,
            method="gradient",
        )

    return salience_maps


def compute_hessian_salience(
    weights: torch.Tensor | np.ndarray,
    activations: torch.Tensor | np.ndarray,
    group_size: int = 128,
) -> SalienceMap:
    """Compute Hessian-diagonal-based salience for a single weight tensor.

    Uses the approximation: salience_i ≈ w_i^2 * H_ii, where H_ii is
    estimated from the squared activations (Fisher information approximation).

    Args:
        weights: Weight tensor, shape (out_features, in_features).
        activations: Activation samples, shape (num_samples, in_features).
        group_size: Quantization group size.

    Returns:
        SalienceMap for this tensor.
    """
    if isinstance(weights, torch.Tensor):
        w = weights.detach().cpu().float().numpy()
    else:
        w = weights.astype(np.float32)

    if isinstance(activations, torch.Tensor):
        a = activations.detach().cpu().float().numpy()
    else:
        a = activations.astype(np.float32)

    original_shape = w.shape

    # Hessian diagonal approximation: H_ii ≈ E[x_i^2]
    hessian_diag = np.mean(a**2, axis=0)  # shape (in_features,)

    # Salience: w^2 * H_ii (broadcast across output dimension)
    salience = w**2 * hessian_diag[np.newaxis, :]

    flat_salience = salience.reshape(-1)

    # Pad and group
    remainder = len(flat_salience) % group_size
    if remainder:
        flat_salience = np.concatenate(
            [flat_salience, np.zeros(group_size - remainder, dtype=np.float32)]
        )

    grouped = flat_salience.reshape(-1, group_size)
    group_scores = grouped.mean(axis=1)

    score_max = group_scores.max()
    if score_max > 0:
        group_scores = group_scores / score_max

    return SalienceMap(
        scores=group_scores,
        group_size=group_size,
        tensor_shape=original_shape,
        method="hessian",
    )


def allocate_bits_by_salience(
    salience_map: SalienceMap,
    target_avg_bits: float = 2.5,
    min_bits: int = 2,
    max_bits: int = 8,
) -> BitAllocation:
    """Allocate bit-widths per group based on salience scores.

    Groups with higher salience receive more bits. The allocation minimizes
    expected quantization error subject to an average bit-width budget.

    Uses a greedy approach: start all groups at min_bits, then iteratively
    promote the most salient groups until the average bit budget is met.

    Args:
        salience_map: Pre-computed salience scores.
        target_avg_bits: Target average bits across all groups.
        min_bits: Minimum bits for any group.
        max_bits: Maximum bits for any group (salient groups).

    Returns:
        BitAllocation with per-group bit assignments.
    """
    num_groups = salience_map.num_groups
    scores = salience_map.scores.copy()
    bits = np.full(num_groups, min_bits, dtype=np.int32)

    # Total bit budget
    total_budget = target_avg_bits * num_groups
    current_total = float(np.sum(bits))

    # Valid bit-widths to promote through
    valid_bits = sorted({2, 3, 4, 8, min_bits, max_bits})
    valid_bits = [b for b in valid_bits if min_bits <= b <= max_bits]

    # Greedy: promote highest-salience groups through the bit tiers
    while current_total < total_budget:
        # Find the group with highest salience that can still be promoted
        promotable = bits < max_bits
        if not np.any(promotable):
            break

        # Score weighted by remaining promotability
        effective_scores = scores * promotable
        best_group = int(np.argmax(effective_scores))

        if not promotable[best_group]:
            break

        # Find next valid bit-width above current
        current_bits = bits[best_group]
        next_bits = None
        for vb in valid_bits:
            if vb > current_bits:
                next_bits = vb
                break

        if next_bits is None:
            scores[best_group] = -1  # Exhausted
            continue

        cost = next_bits - current_bits
        if current_total + cost > total_budget + 0.5 * num_groups:
            # Would exceed budget too much
            break

        bits[best_group] = next_bits
        current_total += cost

        # Slightly decay this group's priority so others get a chance
        scores[best_group] *= 0.5

    return BitAllocation(
        bits_per_group=bits,
        group_size=salience_map.group_size,
        tensor_shape=salience_map.tensor_shape,
    )


def identify_pivot_token_weights(
    attention_weights: torch.Tensor | np.ndarray,
    threshold_ratio: float = 0.5,
) -> np.ndarray:
    """Identify attention heads/positions associated with pivot tokens.

    Pivot tokens are initial tokens that receive disproportionately high
    attention scores. Weights governing these tokens need protection.

    Args:
        attention_weights: Attention score matrix, shape (num_heads, seq_len, seq_len).
        threshold_ratio: Fraction of total attention a position must receive
                         to be considered a pivot.

    Returns:
        Boolean mask of pivot positions, shape (seq_len,).
    """
    if isinstance(attention_weights, torch.Tensor):
        attn = attention_weights.detach().cpu().float().numpy()
    else:
        attn = attention_weights.astype(np.float32)

    # Average attention received per position across all heads and queries
    # attn shape: (num_heads, seq_len, seq_len) -> mean over heads and queries
    avg_attention_received = attn.mean(axis=(0, 1))  # shape (seq_len,)

    # Normalize
    total = avg_attention_received.sum()
    if total > 0:
        attention_frac = avg_attention_received / total
    else:
        attention_frac = np.zeros_like(avg_attention_received)

    # Pivot tokens: positions that receive > threshold_ratio of average attention
    uniform_share = 1.0 / max(len(attention_frac), 1)
    pivot_mask = attention_frac > (uniform_share * threshold_ratio * len(attention_frac))

    return pivot_mask


def compute_kl_divergence(
    logits_original: torch.Tensor,
    logits_quantized: torch.Tensor,
) -> float:
    """Compute KL divergence between original and quantized model output distributions.

    Used to evaluate the impact of a particular quantization configuration
    on model output quality.

    Args:
        logits_original: Logits from the full-precision model.
        logits_quantized: Logits from the quantized model.

    Returns:
        KL divergence (non-negative float, lower is better).
    """
    p = F.softmax(logits_original.float(), dim=-1)
    q = F.log_softmax(logits_quantized.float(), dim=-1)
    kl = F.kl_div(q, p, reduction="batchmean")
    return float(kl.item())
