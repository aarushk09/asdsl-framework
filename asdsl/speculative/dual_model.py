"""Dual-model speculative decoding with batched verification.

This module implements standard speculative sampling with:
- a small always-resident draft model
- a larger target model that verifies draft tokens in one batched pass
- KV cache snapshot/rollback so rejected drafts do not corrupt state
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class SpeculativeModel(Protocol):
    """Model interface required by the dual-model decoder."""

    @property
    def name(self) -> str:
        ...

    @property
    def vocab_size(self) -> int:
        ...

    @property
    def kv_length(self) -> int:
        ...

    def prefill(self, prompt_tokens: list[int]) -> None:
        ...

    def snapshot_kv(self) -> int:
        ...

    def restore_kv(self, snapshot: int) -> None:
        ...

    def next_logits(self, count_cost: bool = True) -> np.ndarray:
        ...

    def verify_logits_batch(self, draft_tokens: list[int]) -> tuple[list[np.ndarray], np.ndarray]:
        """Return logits for each drafted position and one extra next-position logit.

        The model should advance its internal KV state by len(draft_tokens).
        """
        ...

    def advance_tokens(self, tokens: list[int]) -> None:
        ...


@dataclass
class SpeculativeRunResult:
    generated_tokens: list[int]
    decode_time_s: float
    effective_tokens_per_s: float
    acceptance_rate: float
    drafted_tokens: int
    accepted_draft_tokens: int
    verifier_calls: int


class FixedKVCache:
    """Preallocated KV tracker with O(1) snapshot/restore.

    This avoids per-step allocations and guarantees rollback without leaks.
    """

    def __init__(self, max_context: int):
        self._tokens = np.empty(max_context, dtype=np.int32)
        self._length = 0

    @property
    def length(self) -> int:
        return self._length

    def snapshot(self) -> int:
        return self._length

    def restore(self, snapshot: int) -> None:
        if snapshot < 0 or snapshot > self._length:
            raise ValueError(f"invalid KV snapshot {snapshot}")
        self._length = snapshot

    def append_many(self, tokens: list[int]) -> None:
        n = len(tokens)
        if n == 0:
            return
        end = self._length + n
        if end > self._tokens.shape[0]:
            raise RuntimeError("KV cache overflow")
        self._tokens[self._length:end] = np.asarray(tokens, dtype=np.int32)
        self._length = end


class SimulatedDualModel:
    """Deterministic simulated model used for speculative benchmarking.

    The draft model is initialized with a resident buffer to mirror a small
    model that remains fully loaded in memory. The target model exposes a
    batched verifier that amortizes weight-load cost across drafted tokens.
    """

    def __init__(
        self,
        name: str,
        vocab_size: int,
        max_context: int,
        base_seed: int,
        latency_s: float,
        draft_noise_std: float,
        resident_mb: int = 0,
    ):
        self._name = name
        self._vocab_size = vocab_size
        self._seed = np.uint64(base_seed)
        self._latency_s = float(latency_s)
        self._draft_noise_std = float(draft_noise_std)
        self._cache = FixedKVCache(max_context=max_context)

        # Separate state hash from kv length so restore is exact.
        self._state_hash = np.uint64(base_seed ^ 0x9E3779B97F4A7C15)
        self._hash_history = np.empty(max_context + 1, dtype=np.uint64)
        self._hash_history[0] = self._state_hash

        self._resident_weights = None
        if resident_mb > 0:
            # Keep the draft model memory resident and separate from verifier paths.
            count = (resident_mb * 1024 * 1024) // 4
            self._resident_weights = np.empty(count, dtype=np.float32)
            self._resident_weights.fill(0.001)

    @property
    def name(self) -> str:
        return self._name

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def kv_length(self) -> int:
        return self._cache.length

    def prefill(self, prompt_tokens: list[int]) -> None:
        self.restore_kv(0)
        self.advance_tokens(prompt_tokens)

    def snapshot_kv(self) -> int:
        return self._cache.snapshot()

    def restore_kv(self, snapshot: int) -> None:
        self._cache.restore(snapshot)
        self._state_hash = self._hash_history[snapshot]

    def next_logits(self, count_cost: bool = True) -> np.ndarray:
        if count_cost and self._latency_s > 0:
            time.sleep(self._latency_s)
        return self._compute_logits(self._state_hash)

    def verify_logits_batch(self, draft_tokens: list[int]) -> tuple[list[np.ndarray], np.ndarray]:
        if self._latency_s > 0:
            # Batched verifier amortizes cost instead of paying full per token.
            batch_factor = 0.55 + 0.08 * max(len(draft_tokens) - 1, 0)
            time.sleep(self._latency_s * batch_factor)

        logits_seq: list[np.ndarray] = []
        h = self._state_hash
        for tok in draft_tokens:
            logits_seq.append(self._compute_logits(h))
            h = self._mix_hash(h, int(tok))
        next_logits = self._compute_logits(h)

        self.advance_tokens(draft_tokens)
        return logits_seq, next_logits

    def advance_tokens(self, tokens: list[int]) -> None:
        self._cache.append_many(tokens)
        for tok in tokens:
            self._state_hash = self._mix_hash(self._state_hash, int(tok))
            self._hash_history[self._cache.length] = self._state_hash

    def _compute_logits(self, state_hash: np.uint64) -> np.ndarray:
        # Shared base distribution by state hash.
        base_seed = int((state_hash ^ self._seed) & np.uint64(0xFFFFFFFF))
        base_rng = np.random.default_rng(base_seed)
        logits = base_rng.standard_normal(self._vocab_size, dtype=np.float32)

        if self._draft_noise_std > 0:
            noise_seed = int((state_hash ^ (self._seed << np.uint64(1))) & np.uint64(0xFFFFFFFF))
            noise_rng = np.random.default_rng(noise_seed)
            logits += self._draft_noise_std * noise_rng.standard_normal(
                self._vocab_size, dtype=np.float32
            )

        return logits

    @staticmethod
    def _mix_hash(h: np.uint64, token: int) -> np.uint64:
        # Use Python integer wraparound arithmetic to avoid uint64 runtime warnings.
        mask = 0xFFFFFFFFFFFFFFFF
        h_int = int(h) & mask
        x = (int(token) + 0x9E3779B9) & mask
        mixed = (x + 0x9E3779B97F4A7C15 + ((h_int << 6) & mask) + (h_int >> 2)) & mask
        return np.uint64((h_int ^ mixed) & mask)


class DualModelSpeculativeDecoder:
    """Standard draft/verify speculative decoder with rollback-safe KV updates."""

    def __init__(
        self,
        draft_model: SpeculativeModel,
        target_model: SpeculativeModel,
        gamma: int = 4,
        temperature: float = 0.0,
        seed: int = 1234,
    ):
        if gamma < 1:
            raise ValueError("gamma must be >= 1")
        self.draft_model = draft_model
        self.target_model = target_model
        self.gamma = gamma
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        prompt_tokens: list[int],
        max_new_tokens: int,
    ) -> SpeculativeRunResult:
        self.draft_model.prefill(prompt_tokens)
        self.target_model.prefill(prompt_tokens)

        generated: list[int] = []
        drafted = 0
        accepted_drafts = 0
        verifier_calls = 0

        t0 = time.perf_counter()
        while len(generated) < max_new_tokens:
            if self.draft_model.kv_length != self.target_model.kv_length:
                raise RuntimeError("KV misalignment before speculative step")

            draft_snap = self.draft_model.snapshot_kv()
            target_snap = self.target_model.snapshot_kv()

            q_logits_seq: list[np.ndarray] = []
            draft_tokens: list[int] = []

            # Draft phase on small resident model.
            for _ in range(self.gamma):
                q_logits = self.draft_model.next_logits(count_cost=True)
                q_logits_seq.append(q_logits)
                tok = self._sample(q_logits)
                draft_tokens.append(tok)
                self.draft_model.advance_tokens([tok])

            drafted += len(draft_tokens)

            # Verify all drafted tokens in one batched target call.
            p_logits_seq, p_next_logits = self.target_model.verify_logits_batch(draft_tokens)
            verifier_calls += 1

            emit, accepted_this_round = self._accept_reject(draft_tokens, q_logits_seq, p_logits_seq, p_next_logits)

            # Rollback both models and commit only emitted tokens.
            self.draft_model.restore_kv(draft_snap)
            self.target_model.restore_kv(target_snap)
            self.draft_model.advance_tokens(emit)
            self.target_model.advance_tokens(emit)

            if self.draft_model.kv_length != self.target_model.kv_length:
                raise RuntimeError("KV misalignment after rollback/commit")

            accepted_drafts += accepted_this_round
            remaining = max_new_tokens - len(generated)
            generated.extend(emit[:remaining])

        decode_time = time.perf_counter() - t0
        tps = len(generated) / max(decode_time, 1e-9)
        acceptance = accepted_drafts / max(drafted, 1)

        return SpeculativeRunResult(
            generated_tokens=generated,
            decode_time_s=decode_time,
            effective_tokens_per_s=tps,
            acceptance_rate=acceptance,
            drafted_tokens=drafted,
            accepted_draft_tokens=accepted_drafts,
            verifier_calls=verifier_calls,
        )

    def _accept_reject(
        self,
        draft_tokens: list[int],
        q_logits_seq: list[np.ndarray],
        p_logits_seq: list[np.ndarray],
        p_next_logits: np.ndarray,
    ) -> tuple[list[int], int]:
        accepted_prefix: list[int] = []

        for i, tok in enumerate(draft_tokens):
            q_probs = _softmax(q_logits_seq[i])
            p_probs = _softmax(p_logits_seq[i])
            q_t = float(max(q_probs[tok], 1e-9))
            p_t = float(max(p_probs[tok], 0.0))

            alpha = min(1.0, p_t / q_t)
            if self.rng.random() <= alpha:
                accepted_prefix.append(tok)
                continue

            residual = np.maximum(p_probs - q_probs, 0.0)
            rs = float(residual.sum())
            if rs > 0:
                residual /= rs
                correction = int(self.rng.choice(len(residual), p=residual))
            else:
                correction = self._sample_from_probs(p_probs)
            return accepted_prefix + [correction], len(accepted_prefix)

        # All drafts accepted: add one extra token from target next distribution.
        bonus = self._sample(p_next_logits)
        return accepted_prefix + [bonus], len(accepted_prefix)

    def _sample(self, logits: np.ndarray) -> int:
        if self.temperature <= 0:
            return int(np.argmax(logits))
        scaled = logits / self.temperature
        probs = _softmax(scaled)
        return self._sample_from_probs(probs)

    def _sample_from_probs(self, probs: np.ndarray) -> int:
        return int(self.rng.choice(len(probs), p=probs))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp_vals = np.exp(shifted)
    denom = np.sum(exp_vals)
    if denom <= 0:
        return np.full_like(logits, 1.0 / logits.size)
    return exp_vals / denom


def run_target_only_baseline(
    target_model: SpeculativeModel,
    prompt_tokens: list[int],
    max_new_tokens: int,
    temperature: float = 0.0,
    seed: int = 1234,
) -> tuple[list[int], float, float]:
    """Decode with only the target model for speedup comparison."""
    rng = np.random.default_rng(seed)
    target_model.prefill(prompt_tokens)

    generated: list[int] = []
    t0 = time.perf_counter()
    for _ in range(max_new_tokens):
        logits = target_model.next_logits(count_cost=True)
        if temperature <= 0:
            tok = int(np.argmax(logits))
        else:
            probs = _softmax(logits / temperature)
            tok = int(rng.choice(len(probs), p=probs))
        generated.append(tok)
        target_model.advance_tokens([tok])

    elapsed = time.perf_counter() - t0
    tok_s = len(generated) / max(elapsed, 1e-9)
    return generated, elapsed, tok_s
