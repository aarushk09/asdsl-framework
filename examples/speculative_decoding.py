"""Example: SWIFT self-speculative decoding.

Shows how to configure and run SWIFT speculative decoding with
adaptive skip schedule tuning.
"""

import numpy as np

from asdsl.speculative.swift import (
    SWIFTDecoder,
    create_skip_schedule_for_phi3,
)


class SimpleExecutor:
    """Minimal layer executor for demonstration purposes."""

    def __init__(self, num_layers: int = 32, hidden_dim: int = 128, vocab_size: int = 1000):
        self._num_layers = num_layers
        self._hidden_dim = hidden_dim
        self._vocab_size = vocab_size
        np.random.seed(42)
        self._layer_weights = [
            np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.02
            for _ in range(num_layers)
        ]

    @property
    def num_layers(self) -> int:
        return self._num_layers

    def execute_layer(self, layer_idx: int, hidden_state: np.ndarray) -> np.ndarray:
        return np.tanh(hidden_state @ self._layer_weights[layer_idx])

    def execute_lm_head(self, hidden_state: np.ndarray) -> np.ndarray:
        if hidden_state.ndim == 1:
            return np.random.randn(self._vocab_size).astype(np.float32)
        return np.random.randn(hidden_state.shape[0], self._vocab_size).astype(np.float32)


def main() -> None:
    # Configure skip schedule for Phi-3-mini
    schedule = create_skip_schedule_for_phi3()
    print("SWIFT Skip Schedule (Phi-3-mini):")
    print(f"  Total layers:     {schedule.total_layers}")
    print(f"  Draft layers:     {schedule.draft_layers}")
    print(f"  Skip layers:      {sorted(schedule.skip_indices)}")
    print(f"  Skip ratio:       {schedule.skip_ratio:.0%}")
    print(f"  Speedup estimate: {schedule.speedup_estimate:.1f}x")
    print()

    # Create decoder
    executor = SimpleExecutor(num_layers=32, hidden_dim=128, vocab_size=1000)
    decoder = SWIFTDecoder(
        executor=executor,
        num_draft_tokens=4,
        keep_first=4,
        keep_last=4,
        temperature=0.0,
        adaptive_schedule=True,
    )

    # Run speculative decoding for several steps
    hidden = np.random.randn(128).astype(np.float32)
    all_tokens = [0]  # BOS token
    total_accepted = 0
    total_drafted = 0

    print("Generating tokens with SWIFT speculative decoding...")
    for step in range(20):
        result = decoder.speculative_step(hidden, past_tokens=all_tokens)
        all_tokens.extend(result.accepted_tokens)
        total_accepted += len(result.accepted_tokens)
        total_drafted += result.num_draft_tokens

        if step < 5 or step == 19:
            print(f"  Step {step:2d}: drafted={result.num_draft_tokens}, "
                  f"accepted={len(result.accepted_tokens)}, "
                  f"tokens={result.accepted_tokens}")

    print(f"\nSummary:")
    print(f"  Total tokens generated: {total_accepted}")
    print(f"  Total draft attempts:   {total_drafted}")
    print(f"  Overall acceptance:     {total_accepted / max(total_drafted, 1):.1%}")
    print(f"  Effective tokens/step:  {total_accepted / 20:.1f}")
    print(f"  Sequence: {all_tokens[:20]}{'...' if len(all_tokens) > 20 else ''}")


if __name__ == "__main__":
    main()
