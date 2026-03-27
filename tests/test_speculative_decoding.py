"""Phase 5: greedy draft–verify speculative decoding (QCSD) tests."""

from __future__ import annotations

import os
import sys

import pytest

from asdsl.speculative.dual_model import (
    GreedyDualModelSpeculativeDecoder,
    SimulatedDualModel,
    run_greedy_baseline_tokens,
)


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.mark.parametrize("seed", [7, 42])
@pytest.mark.parametrize("gamma", [4, 5])
def test_greedy_speculative_matches_autoregressive_token_ids(seed: int, gamma: int) -> None:
    """Strict greedy equivalence: identical draft+target (noise-free) == AR baseline."""
    if _repo_root() not in sys.path:
        sys.path.insert(0, _repo_root())

    vocab = 512
    prompt = list(range(3, 18))
    max_new = 40

    def make(name: str) -> SimulatedDualModel:
        return SimulatedDualModel(
            name=name,
            vocab_size=vocab,
            max_context=4096,
            base_seed=seed,
            latency_s=0.0,
            draft_noise_std=0.0,
            resident_mb=0,
            sim_acceptance_rate=1.0,
        )

    draft = make("draft")
    target_a = make("target-ar")
    target_b = make("target-spec")

    ar_ids, _, _ = run_greedy_baseline_tokens(target_a, prompt, max_new)

    dec = GreedyDualModelSpeculativeDecoder(draft, target_b, gamma=gamma)
    spec = dec.generate(prompt, max_new)

    assert spec.generated_tokens == ar_ids


def test_acceptance_rate_reported(capsys: pytest.CaptureFixture[str]) -> None:
    """Logging: aggregate Bernoulli draft acceptance (see sim_acceptance_rate)."""
    vocab = 256
    prompt = [1, 2, 3, 4, 5]
    max_new = 48
    seed = 123

    draft = SimulatedDualModel(
        name="draft",
        vocab_size=vocab,
        max_context=2048,
        base_seed=seed,
        latency_s=0.0,
        draft_noise_std=0.15,
        resident_mb=0,
    )
    target = SimulatedDualModel(
        name="target",
        vocab_size=vocab,
        max_context=2048,
        base_seed=seed,
        latency_s=0.0,
        draft_noise_std=0.0,
        resident_mb=0,
    )

    dec = GreedyDualModelSpeculativeDecoder(draft, target, gamma=5, draft_sim_seed=seed)
    result = dec.generate(prompt, max_new)

    rate = result.acceptance_rate
    print(
        f"[spec_decode] drafted={result.drafted_tokens} "
        f"accepted_matches={result.accepted_draft_matches} "
        f"avg_acceptance_rate={rate:.4f} rounds={result.speculative_rounds}",
        file=sys.stderr,
    )
    assert 0.0 <= rate <= 1.0
    assert result.drafted_tokens >= result.accepted_draft_matches
    captured = capsys.readouterr()
    assert "avg_acceptance_rate=" in captured.err


def test_effective_speedup_vs_baseline() -> None:
    """Simulated latencies: batched verify should beat per-token target decode."""
    vocab = 2000
    prompt = [11, 22, 33, 44]
    max_new = 50
    seed = 2026

    draft = SimulatedDualModel(
        name="draft",
        vocab_size=vocab,
        max_context=4096,
        base_seed=seed,
        latency_s=0.002,
        draft_noise_std=0.0,
        resident_mb=64,
        sim_acceptance_rate=1.0,
    )
    target_slow = SimulatedDualModel(
        name="target",
        vocab_size=vocab,
        max_context=4096,
        base_seed=seed,
        latency_s=0.10,
        draft_noise_std=0.0,
        resident_mb=0,
        sim_acceptance_rate=1.0,
    )

    _, t_ar, tps_ar = run_greedy_baseline_tokens(target_slow, list(prompt), max_new)

    target_spec = SimulatedDualModel(
        name="target-spec",
        vocab_size=vocab,
        max_context=4096,
        base_seed=seed,
        latency_s=0.10,
        draft_noise_std=0.0,
        resident_mb=0,
        sim_acceptance_rate=1.0,
    )
    dec = GreedyDualModelSpeculativeDecoder(draft, target_spec, gamma=5, draft_sim_seed=seed)
    spec = dec.generate(list(prompt), max_new)

    tps_spec = spec.effective_tokens_per_s
    speedup = tps_spec / max(tps_ar, 1e-9)

    print(
        f"[spec_decode] baseline {t_ar:.3f}s ({tps_ar:.2f} tok/s) vs "
        f"speculative {spec.decode_time_s:.3f}s ({tps_spec:.2f} tok/s) "
        f"speedup={speedup:.2f}x",
        file=sys.stderr,
    )

    assert speedup >= 1.5, (
        f"expected >=1.5x effective tok/s (draft cheap + amortized verify); "
        f"got {speedup:.2f}x"
    )


def test_bernoulli_draft_acceptance_rate_tracks_sim_parameter() -> None:
    """Aggregate acceptance_rate should cluster near sim_acceptance_rate (no wall-clock sleep)."""
    vocab = 256
    prompt = [1, 2, 3, 4, 5]
    max_new = 1200
    p = 0.80
    seed = 99
    draft = SimulatedDualModel(
        name="draft",
        vocab_size=vocab,
        max_context=8192,
        base_seed=seed,
        latency_s=0.0,
        draft_noise_std=0.0,
        resident_mb=0,
        sim_acceptance_rate=p,
    )
    target = SimulatedDualModel(
        name="target",
        vocab_size=vocab,
        max_context=8192,
        base_seed=seed,
        latency_s=0.0,
        draft_noise_std=0.0,
        resident_mb=0,
        sim_acceptance_rate=1.0,
    )
    dec = GreedyDualModelSpeculativeDecoder(draft, target, gamma=7, draft_sim_seed=2025)
    res = dec.generate(list(prompt), max_new)
    assert res.bernoulli_trials > 100
    assert 0.74 <= res.acceptance_rate <= 0.86, (
        f"expected ~{p:.2f} aggregate Bernoulli rate, got {res.acceptance_rate:.4f}"
    )
