"""Checkpoint 7: dual-model speculative decoding benchmark.

Runs target-only decode versus draft+batched-verify speculative decode and
reports acceptance rate, effective tok/s, and speedup.
"""

from __future__ import annotations

from asdsl.speculative import (
    DualModelSpeculativeDecoder,
    SimulatedDualModel,
    run_target_only_baseline,
)


def _print_header() -> None:
    print("=" * 76)
    print("ASDSL Checkpoint 7 - Dual-Model Speculative Decoding")
    print("=" * 76)


def main() -> None:
    _print_header()

    vocab_size = 4096
    prompt_tokens = [1, 902, 42, 17, 333]
    max_new_tokens = 96

    # Draft model is kept resident to avoid repeated load/evict behavior.
    draft_model = SimulatedDualModel(
        name="phi-mini-q4-draft",
        vocab_size=vocab_size,
        max_context=8192,
        base_seed=2026,
        latency_s=0.012,
        draft_noise_std=0.20,
        resident_mb=128,
    )

    target_model = SimulatedDualModel(
        name="phi-main-q4-target",
        vocab_size=vocab_size,
        max_context=8192,
        base_seed=2026,
        latency_s=0.125,
        draft_noise_std=0.0,
        resident_mb=0,
    )

    # Baseline (target-only autoregressive decode)
    _, baseline_s, baseline_tps = run_target_only_baseline(
        target_model=target_model,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        seed=7,
    )

    # Fresh models for speculative run.
    draft_model = SimulatedDualModel(
        name="phi-mini-q4-draft",
        vocab_size=vocab_size,
        max_context=8192,
        base_seed=2026,
        latency_s=0.012,
        draft_noise_std=0.20,
        resident_mb=128,
    )
    target_model = SimulatedDualModel(
        name="phi-main-q4-target",
        vocab_size=vocab_size,
        max_context=8192,
        base_seed=2026,
        latency_s=0.125,
        draft_noise_std=0.0,
        resident_mb=0,
    )

    decoder = DualModelSpeculativeDecoder(
        draft_model=draft_model,
        target_model=target_model,
        gamma=4,
        temperature=0.0,
        seed=7,
    )
    result = decoder.generate(prompt_tokens=prompt_tokens, max_new_tokens=max_new_tokens)

    speedup = result.effective_tokens_per_s / max(baseline_tps, 1e-9)

    print(f"Prompt tokens:          {len(prompt_tokens)}")
    print(f"Generated tokens:       {len(result.generated_tokens)}")
    print(f"Drafted tokens:         {result.drafted_tokens}")
    print(f"Accepted draft tokens:  {result.accepted_draft_tokens}")
    print(f"Verifier batched calls: {result.verifier_calls}")
    print()
    print("Performance:")
    print(f"  Baseline target-only: {baseline_tps:.2f} tok/s ({baseline_s:.2f}s)")
    print(
        f"  Dual-model effective: {result.effective_tokens_per_s:.2f} tok/s "
        f"({result.decode_time_s:.2f}s)"
    )
    print(f"  Speedup vs baseline:  {speedup:.2f}x")
    print(f"  Acceptance rate:      {result.acceptance_rate:.1%}")
    print()

    acceptance_ok = 0.65 <= result.acceptance_rate <= 0.80
    throughput_ok = result.effective_tokens_per_s > 7.0
    speedup_ok = speedup > 1.0

    print("Checkpoint 7 gates:")
    print(f"  [1] Acceptance 65-80%     -> {'PASS' if acceptance_ok else 'FAIL'}")
    print(f"  [2] Effective speed >7.0  -> {'PASS' if throughput_ok else 'FAIL'}")
    print(f"  [3] No speedup regression -> {'PASS' if speedup_ok else 'FAIL'}")

    if not (acceptance_ok and throughput_ok and speedup_ok):
        raise SystemExit("Checkpoint 7 failed: metrics did not meet required thresholds.")


if __name__ == "__main__":
    main()
