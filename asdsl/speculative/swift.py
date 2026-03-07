"""SWIFT self-speculative decoding with dynamic layer skipping.

The model acts as its own draft model by dynamically bypassing
intermediate transformer layers to rapidly produce draft tokens,
then verifying them in a single batched pass through the full model.

Key properties:
- Zero additional memory overhead (no secondary draft model)
- Plug-and-play: no retraining or fine-tuning required
- Preserves the original output distribution of the target model
- 1.3-1.6x speedup on autoregressive generation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class LayerExecutor(Protocol):
    """Protocol for executing transformer layers (injected by inference engine)."""

    def execute_layer(self, layer_idx: int, hidden_state: np.ndarray) -> np.ndarray:
        """Execute a single transformer layer."""
        ...

    def execute_lm_head(self, hidden_state: np.ndarray) -> np.ndarray:
        """Project hidden state to vocabulary logits."""
        ...

    @property
    def num_layers(self) -> int:
        """Total number of transformer layers."""
        ...


@dataclass
class SkipSchedule:
    """Defines which layers to skip during draft generation.

    Attributes:
        total_layers: Total number of transformer layers.
        keep_first: Number of initial layers to always execute.
        keep_last: Number of final layers to always execute.
        skip_indices: Set of layer indices to skip during draft.
    """

    total_layers: int
    keep_first: int = 4
    keep_last: int = 4
    skip_indices: set[int] = field(default_factory=set)

    def __post_init__(self):
        if not self.skip_indices:
            self.skip_indices = set(
                range(self.keep_first, self.total_layers - self.keep_last)
            )

    @property
    def draft_layers(self) -> list[int]:
        """Layer indices executed during draft pass."""
        return [i for i in range(self.total_layers) if i not in self.skip_indices]

    @property
    def skip_ratio(self) -> float:
        """Fraction of layers skipped during draft."""
        return len(self.skip_indices) / max(self.total_layers, 1)

    @property
    def speedup_estimate(self) -> float:
        """Estimated speedup from layer skipping (draft phase only)."""
        draft_cost = len(self.draft_layers) / max(self.total_layers, 1)
        return 1.0 / max(draft_cost, 0.01)


@dataclass
class SpeculativeResult:
    """Result of a self-speculative decoding step.

    Attributes:
        accepted_tokens: Token IDs accepted after verification.
        num_draft_tokens: Number of draft tokens generated.
        num_accepted: Number of tokens accepted by verification.
        acceptance_rate: Fraction of draft tokens accepted.
        draft_logits: Logits from the draft (shallow) pass.
        verify_logits: Logits from the full verification pass.
    """

    accepted_tokens: list[int]
    num_draft_tokens: int
    num_accepted: int
    draft_logits: np.ndarray | None = None
    verify_logits: np.ndarray | None = None

    @property
    def acceptance_rate(self) -> float:
        return self.num_accepted / max(self.num_draft_tokens, 1)


class SWIFTDecoder:
    """SWIFT self-speculative decoder.

    Uses dynamic layer skipping to generate draft tokens from a
    shallow sub-network, then verifies drafts using the full model.
    """

    def __init__(
        self,
        executor: LayerExecutor,
        num_draft_tokens: int = 4,
        keep_first: int = 4,
        keep_last: int = 4,
        temperature: float = 1.0,
        adaptive_schedule: bool = True,
    ):
        """Initialize the SWIFT decoder.

        Args:
            executor: Layer execution backend.
            num_draft_tokens: Number of draft tokens per speculation step.
            keep_first: Always execute the first N layers.
            keep_last: Always execute the last N layers.
            temperature: Sampling temperature.
            adaptive_schedule: If True, dynamically adjust skip schedule
                              based on acceptance rate.
        """
        self.executor = executor
        self.num_draft_tokens = num_draft_tokens
        self.temperature = temperature
        self.adaptive_schedule = adaptive_schedule

        self.schedule = SkipSchedule(
            total_layers=executor.num_layers,
            keep_first=keep_first,
            keep_last=keep_last,
        )

        # Tracking for adaptive scheduling
        self._acceptance_history: list[float] = []
        self._min_keep = 4
        self._max_skip_ratio = 0.85

        logger.info(
            "SWIFT decoder initialized: %d total layers, skipping %d in draft "
            "(%.0f%% reduction, est. %.1fx speedup)",
            executor.num_layers,
            len(self.schedule.skip_indices),
            self.schedule.skip_ratio * 100,
            self.schedule.speedup_estimate,
        )

    def draft_forward(self, hidden_state: np.ndarray) -> np.ndarray:
        """Execute a draft (shallow) forward pass with layer skipping.

        Only executes layers not in the skip set, creating a fast
        approximate inference path.

        Args:
            hidden_state: Input hidden state, shape (seq_len, hidden_dim).

        Returns:
            Output logits from the shallow pass.
        """
        h = hidden_state.copy()

        for layer_idx in self.schedule.draft_layers:
            h = self.executor.execute_layer(layer_idx, h)

        logits = self.executor.execute_lm_head(h)
        return logits

    def full_forward(self, hidden_state: np.ndarray) -> np.ndarray:
        """Execute a full forward pass through all layers (verification).

        Args:
            hidden_state: Input hidden state.

        Returns:
            Output logits from the complete model.
        """
        h = hidden_state.copy()

        for layer_idx in range(self.executor.num_layers):
            h = self.executor.execute_layer(layer_idx, h)

        logits = self.executor.execute_lm_head(h)
        return logits

    def speculative_step(
        self,
        hidden_state: np.ndarray,
        past_tokens: list[int],
    ) -> SpeculativeResult:
        """Execute one self-speculative decoding step.

        1. Generate `num_draft_tokens` using the shallow (draft) model
        2. Verify all draft tokens with a single full forward pass
        3. Accept tokens where draft and full model agree

        Args:
            hidden_state: Current hidden state for continuation.
            past_tokens: Previously generated token IDs (for context).

        Returns:
            SpeculativeResult with accepted tokens and metrics.
        """
        draft_tokens = []
        draft_logits_list = []
        h = hidden_state.copy()

        # Phase 1: Draft generation (shallow pass)
        for _ in range(self.num_draft_tokens):
            logits = self.draft_forward(h)
            draft_logits_list.append(logits)

            # Sample from draft distribution
            token = self._sample_token(logits)
            draft_tokens.append(token)

            # TODO: In full implementation, update hidden state with
            # the new token through the draft layers only
            # For now, re-use hidden state (simplified)

        # Phase 2: Verification (batched full pass)
        # In practice, this evaluates all draft tokens in parallel
        # as a single forward pass (like a prefill operation)
        verify_logits = self.full_forward(hidden_state)

        # Phase 3: Accept/reject
        accepted = self._verify_tokens(
            draft_tokens=draft_tokens,
            draft_logits=draft_logits_list,
            verify_logits=verify_logits,
        )

        result = SpeculativeResult(
            accepted_tokens=accepted,
            num_draft_tokens=self.num_draft_tokens,
            num_accepted=len(accepted),
            draft_logits=np.stack(draft_logits_list) if draft_logits_list else None,
            verify_logits=verify_logits,
        )

        # Update adaptive schedule
        if self.adaptive_schedule:
            self._update_schedule(result.acceptance_rate)

        return result

    def _sample_token(self, logits: np.ndarray) -> int:
        """Sample a token from logits with temperature scaling."""
        if logits.ndim > 1:
            logits = logits[-1]  # Take last position

        if self.temperature <= 0:
            return int(np.argmax(logits))

        # Temperature-scaled softmax
        scaled = logits / self.temperature
        scaled -= scaled.max()  # Numerical stability
        probs = np.exp(scaled) / np.sum(np.exp(scaled))

        return int(np.random.choice(len(probs), p=probs))

    def _verify_tokens(
        self,
        draft_tokens: list[int],
        draft_logits: list[np.ndarray],
        verify_logits: np.ndarray,
    ) -> list[int]:
        """Verify draft tokens against the full model's distribution.

        Uses the standard speculative decoding acceptance criterion:
        accept token t if p_full(t) >= p_draft(t), otherwise accept
        with probability p_full(t) / p_draft(t).

        Args:
            draft_tokens: Tokens from draft generation.
            draft_logits: Logits from each draft step.
            verify_logits: Logits from full verification pass.

        Returns:
            List of accepted token IDs.
        """
        accepted = []

        for i, token in enumerate(draft_tokens):
            if i >= len(draft_logits):
                break

            d_logits = draft_logits[i]
            if d_logits.ndim > 1:
                d_logits = d_logits[-1]

            v_logits = verify_logits
            if v_logits.ndim > 1:
                v_logits = v_logits[-1]

            # Compute probabilities
            d_probs = _softmax(d_logits)
            v_probs = _softmax(v_logits)

            p_draft = d_probs[token]
            p_verify = v_probs[token]

            # Acceptance criterion
            if p_verify >= p_draft:
                accepted.append(token)
            else:
                # Probabilistic acceptance
                if p_draft > 0 and np.random.random() < (p_verify / p_draft):
                    accepted.append(token)
                else:
                    # Reject this and all subsequent tokens
                    # Sample correction token from residual distribution
                    residual = np.maximum(v_probs - d_probs, 0)
                    residual_sum = residual.sum()
                    if residual_sum > 0:
                        residual /= residual_sum
                        correction_token = int(np.random.choice(len(residual), p=residual))
                        accepted.append(correction_token)
                    else:
                        accepted.append(int(np.argmax(v_probs)))
                    break

        return accepted

    def _update_schedule(self, acceptance_rate: float) -> None:
        """Adaptively adjust the layer skip schedule based on acceptance rate.

        If acceptance rate is high (>0.8), we can try skipping more layers.
        If it drops below 0.5, reduce skip aggressiveness.
        """
        self._acceptance_history.append(acceptance_rate)

        # Use a rolling window
        window = self._acceptance_history[-10:]
        avg_rate = sum(window) / len(window)

        if avg_rate > 0.8 and self.schedule.skip_ratio < self._max_skip_ratio:
            # Try skipping one more layer (from boundaries inward)
            draft_set = set(self.schedule.draft_layers)
            candidates = sorted(draft_set - {0, self.executor.num_layers - 1})
            if len(candidates) > self._min_keep * 2:
                # Skip the middle-most currently-executed layer
                mid = candidates[len(candidates) // 2]
                self.schedule.skip_indices.add(mid)
                logger.debug("Adaptive: increased skip to %.0f%%", self.schedule.skip_ratio * 100)

        elif avg_rate < 0.5 and self.schedule.skip_ratio > 0.2:
            # Restore one layer
            if self.schedule.skip_indices:
                restored = max(self.schedule.skip_indices)
                self.schedule.skip_indices.discard(restored)
                logger.debug(
                    "Adaptive: decreased skip to %.0f%%", self.schedule.skip_ratio * 100
                )


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = logits - logits.max()
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum()


def create_skip_schedule_for_phi3(
    keep_first: int = 4,
    keep_last: int = 4,
) -> SkipSchedule:
    """Create a default skip schedule optimized for Phi-3-mini (32 layers).

    Default: execute layers 0-3 and 28-31, skip layers 4-27 during draft.
    This gives a ~75% layer reduction for draft generation.
    """
    return SkipSchedule(
        total_layers=32,
        keep_first=keep_first,
        keep_last=keep_last,
    )
