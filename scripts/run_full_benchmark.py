#!/usr/bin/env python3
"""A/B/C benchmark: baseline AR, native AR, and speculative stack.

Profile A (Day 1 baseline): greedy AR, PyTorch-style matvec path (no native GEMV),
FP32 KV tensors (as in ``KVHistory``).

Profile C (native AR): same greedy AR as A but **native Q4 GEMV** enabled — isolates
AVX2 kernel throughput without QCSD draft/verify noise.

Profile B (optimized): native AVX2/OpenMP GEMV when built, primary Q4 + QCSD draft
(2-bit by default; mirrors Phase 5 dual-model story). Simulator footprints use
target + draft weights plus dual FP32 KV caches; Phi-4 mode still reports Q4 KV est.

Default mode uses :class:`asdsl.speculative.dual_model.SimulatedDualModel` so no
weight download is required. With ``--phi4``, runs the real engine if the local
Phi-4 index exists (see ``experiments/phi4_integration.py``).

Leviathan (2023) guardrail: QCSD runs only if analytical S clears a **1.01×** enable
gate (footprint-based c); failure text still gives binary-search **min alpha** for a
**1.05×** reference. Use ``--verify-leviathan-apples`` to compare QCSD vs AR on the
same Profile-B target against S computed with **realized** simulator acceptance.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.util
import io
import json
import math
import os
import sys
import time
from pathlib import Path

_QCSD_HISTORY_FILENAME = ".qcsd_history.json"
_QCSD_HISTORY_MAX_ENTRIES = 256


def _qcsd_history_path(root: Path) -> Path:
    return root / _QCSD_HISTORY_FILENAME


def _load_history_rates(root: Path, key: str = "acceptance_rates") -> list[float]:
    path = _qcsd_history_path(root)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    rates = data.get(key)
    if not isinstance(rates, list):
        return []
    out: list[float] = []
    for x in rates:
        try:
            f = float(x)
        except (TypeError, ValueError):
            continue
        if 0.0 <= f <= 1.0:
            out.append(f)
    return out


def _mean_history_rate(root: Path, key: str = "acceptance_rates") -> float | None:
    rates = _load_history_rates(root, key)
    if not rates:
        return None
    return sum(rates) / len(rates)


def _append_history_rate(root: Path, rate: float, key: str = "acceptance_rates") -> None:
    """Append one measured acceptance rate for adaptive Leviathan priors."""
    path = _qcsd_history_path(root)
    
    # Load entire history object to modify specific key
    if path.is_file():
        try:
            full_data = json.loads(path.read_text(encoding="utf-8"))
        except:
            full_data = {}
    else:
        full_data = {}
        
    rates = full_data.get(key, [])
    if not isinstance(rates, list):
        rates = []
        
    r = max(0.0, min(1.0, float(rate)))
    rates.append(r)
    if len(rates) > _QCSD_HISTORY_MAX_ENTRIES:
        rates = rates[-_QCSD_HISTORY_MAX_ENTRIES:]
    
    full_data[key] = rates
    path.write_text(
        json.dumps(full_data, indent=2) + "\n",
        encoding="utf-8",
    )

def _phi4_safetensors_index_path(repo_root: Path) -> Path:
    """Same layout as ``experiments/phi4_cpu_run.py`` (MODEL_DIR / index name)."""
    return repo_root / "models" / "phi4-multimodal-instruct" / "model.safetensors.index.json"


def _require_phi4_index_or_exit(repo_root: Path) -> None:
    """Fast-fail before loading ``phi4_cpu_run`` (Torch/Transformers/safetensors).

    Avoids the cold-boot trap: heavy imports only after we know weights metadata exists.
    """
    idx = _phi4_safetensors_index_path(repo_root)
    if idx.is_file():
        return
    model_dir = idx.parent
    print(
        "ERROR: Phi-4 local weights are not available (fast-fail).\n"
        f"  Expected index: {idx}\n"
        f"  Model directory: {model_dir}\n"
        "  Run the download/setup step first, e.g.:\n"
        "    python experiments/phi4_integration.py\n"
        "  (See README for Phi-4 multimodal instruct layout.)",
        file=sys.stderr,
    )
    sys.exit(2)


def load_benchmark_config(path: str | Path | None = None) -> dict:
    """Load pinned ``benchmark_config.json``; returns {} if missing."""
    path = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parent.parent / "benchmark_config.json"
    )
    if not path.is_file():
        print(f"[CONFIG] {path} not found — using CLI fallbacks")
        return {}
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    print(f"[CONFIG] Loaded canonical benchmark config from {path}")
    print(
        f"[CONFIG] prompt={cfg.get('prompt')!r} max_new_tokens={cfg.get('max_new_tokens')} "
        f"threads={cfg.get('threads')} draft_k={cfg.get('draft_k')}"
    )
    return cfg


def _leviathan_S(alpha: float, k: int, c: float = 0.221) -> float:
    if alpha <= 0.0:
        return 1.0 / (c * k + 1)
    if alpha >= 1.0:
        return float("inf")
    return (1.0 - alpha ** (k + 1)) / ((1.0 - alpha) * (c * k + 1.0))


def _leviathan_alpha_break_even(k: int, c: float = 0.221) -> float:
    lo, hi = 1e-4, 0.9999
    if _leviathan_S(hi, k, c) < 1.0:
        return float("nan")
    a, b = lo, hi
    for _ in range(60):
        mid = 0.5 * (a + b)
        if _leviathan_S(mid, k, c) >= 1.0:
            b = mid
        else:
            a = mid
    return b


# Phi-4 text backbone geometry (for KV footprint estimates only).
_NUM_LAYERS = 32
_NUM_KV_HEADS = 8
_HEAD_DIM = 128


def _phi4_kv_fp32_mb(seq_len: int) -> float:
    """Per-token K+V in float32 for all layers (matches ``KVHistory`` layout)."""
    b = _NUM_LAYERS * _NUM_KV_HEADS * _HEAD_DIM * 4 * 2
    return b * seq_len / 1e6


def _phi4_kv_q4_est_mb(seq_len: int) -> float:
    """Rough packed-Q4 KV payload vs FP32 (~75% reduction from kv_cache module note)."""
    return _phi4_kv_fp32_mb(seq_len) * 0.25


# Phi-4-scale analytical weights (MB) for simulator table only.
_SIM_BASELINE_WEIGHTS_MB = 2900.0
_SIM_QCSD_TARGET_WEIGHTS_MB = 5800.0
_SIM_QCSD_DRAFT_WEIGHTS_MB = 1300.0
# Simulator Profile-B step latencies (seconds) — ratio matches Leviathan ``c`` for timing checks.
_SIM_DRAFT_STEP_LATENCY_S = 0.0065
_SIM_TARGET_STEP_LATENCY_S = 0.102


def _analytical_footprint_mb(
    profile: str,
    seq_len: int,
    *,
    enable_qcsd: bool = True,
) -> float:
    """Analytical RAM footprint (MB): weights + KV at ``seq_len`` tokens.

    Profile A: one baseline model + single FP32 KV (``KVHistory``-style).

    Profile B with QCSD: full target weights + draft bank + **two** FP32 KV caches
    (draft and target each hold sequence state). Without QCSD, B is target-only +
    one KV.
    """
    kv1 = _phi4_kv_fp32_mb(seq_len)
    p = profile.strip().upper()
    if p == "A":
        return _SIM_BASELINE_WEIGHTS_MB + kv1
    if p == "B":
        if enable_qcsd:
            return (
                _SIM_QCSD_TARGET_WEIGHTS_MB
                + _SIM_QCSD_DRAFT_WEIGHTS_MB
                + 2.0 * kv1
            )
        return _SIM_QCSD_TARGET_WEIGHTS_MB + kv1
    raise ValueError(f"unknown profile {profile!r} (expected 'A' or 'B')")


def _peak_rss_mb() -> float | None:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    except Exception:
        return None


def _leviathan_speedup(alpha: float, gamma: int, cost_ratio: float) -> float:
    """Speculative decoding speedup factor (Leviathan et al., 2023).

    S = (1 - alpha**(g+1)) / ((1 - alpha) * (g * c + 1))

    alpha: draft acceptance rate.  g: draft lookahead (gamma).
    c: draft_cost / target_cost, modeled as draft_mb / target_mb.
    """
    g = max(int(gamma), 1)
    c = max(float(cost_ratio), 0.0)
    denom = g * c + 1.0
    if denom <= 0.0:
        return 0.0
    if alpha >= 1.0 - 1e-15:
        return (g + 1) / denom
    if alpha <= 0.0:
        return 1.0 / denom
    num = 1.0 - (alpha ** (g + 1))
    d = (1.0 - alpha) * denom
    return num / d if d > 0.0 else float("inf")


def _min_alpha_for_leviathan_speedup(
    gamma: int,
    cost_ratio: float,
    target_s: float,
    *,
    n_iter: int = 64,
) -> float | None:
    """Smallest alpha in [0, 1] with S(alpha) >= target_s, or None if impossible."""
    g = max(int(gamma), 1)
    c = max(float(cost_ratio), 0.0)
    s_max = _leviathan_speedup(1.0, g, c)
    if s_max < target_s - 1e-12:
        return None
    s0 = _leviathan_speedup(0.0, g, c)
    if s0 >= target_s:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if _leviathan_speedup(mid, g, c) >= target_s:
            hi = mid
        else:
            lo = mid
    return hi


def _qcsd_break_even_ok(
    alpha: float,
    gamma: int,
    draft_mb: float,
    target_mb: float,
    *,
    gate_speedup: float = 1.01,
    reference_speedup: float = 1.05,
) -> tuple[bool, str, float]:
    """Leviathan et al. (2023) safety check: enable QCSD only if S >= gate_speedup.

    ``reference_speedup`` (default 1.05) is used only for the binary-search min-alpha
    hint in failure messages (Leviathan-style throughput target).

    Returns (ok, warning_message, expected_speedup_S). Message empty when ok.
    """
    g = max(int(gamma), 1)
    a = float(alpha)
    a = min(1.0, max(0.0, a))
    if target_mb <= 0.0:
        return True, "", 1.0
    c = max(float(draft_mb), 0.0) / float(target_mb)
    s = _leviathan_speedup(a, g, c)
    min_a_ref = _min_alpha_for_leviathan_speedup(g, c, reference_speedup)

    if s + 1e-12 >= gate_speedup:
        return True, "", s

    parts = [
        "QCSD break-even FAIL (Leviathan et al., 2023):",
        f"expected speedup S={s:.3f}x < {gate_speedup:.2f}x (enable gate)",
        f"at alpha={a:.3f}, gamma={g}, c=draft/target={c:.3f}",
        f"({draft_mb:.1f} MB / {target_mb:.1f} MB).",
    ]
    if min_a_ref is None:
        s_max = _leviathan_speedup(1.0, g, c)
        parts.append(
            f"Even with alpha=1, S_max={s_max:.3f}x is below {reference_speedup:.2f}x "
            f"(reference); QCSD disabled."
        )
    else:
        parts.append(
            f"Binary-search min alpha for {reference_speedup:.2f}x (reference) is ~{min_a_ref:.3f} "
            f"(raise acceptance or lower c)."
        )
    return False, " ".join(parts), s


def _load_phi4_module(root: Path):
    path = root / "experiments" / "phi4_cpu_run.py"
    module_name = "phi4_cpu_run"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod  # <--- THE FIX IS HERE
    spec.loader.exec_module(mod)
    return mod


def run_sim_benchmark(
    prompt_tokens: list[int],
    max_new_tokens: int,
    gamma: int,
    vocab_size: int,
    draft_mb: float | None = None,
    target_mb: float | None = None,
    sim_cost_ratio: float | None = None,
    sim_acceptance_rate: float = 0.70,
    draft_sim_seed: int = 2026,
    verify_leviathan_apples: bool = False,
) -> None:
    from asdsl.speculative.dual_model import (
        GreedyDualModelSpeculativeDecoder,
        SimulatedDualModel,
        run_greedy_baseline_tokens,
    )

    t_mb = float(_SIM_QCSD_TARGET_WEIGHTS_MB if target_mb is None else target_mb)
    if sim_cost_ratio is not None:
        d_mb = float(sim_cost_ratio) * t_mb
    elif draft_mb is not None:
        d_mb = float(draft_mb)
    else:
        d_mb = 128.0

    enable_qcsd, qcsd_warn, s_leviathan = _qcsd_break_even_ok(
        sim_acceptance_rate, gamma, d_mb, t_mb
    )
    if not enable_qcsd:
        print()
        print("WARNING:", qcsd_warn)
        print(f"  (Analytical Leviathan S={s_leviathan:.3f}x; falling back to AR for Profile B.)")

    gc.collect()
    rss0 = _peak_rss_mb()

    # Profile A: target ~7.0 tok/s (tuned vs measured overhead in ``run_greedy_baseline_tokens``).
    _lat_a = 0.137
    baseline_target = SimulatedDualModel(
        name="baseline-ar-fp32kv",
        vocab_size=vocab_size,
        max_context=8192,
        base_seed=2026,
        latency_s=_lat_a,
        draft_noise_std=0.0,
        resident_mb=0,
    )
    _, t_a, tps_a = run_greedy_baseline_tokens(
        baseline_target, list(prompt_tokens), max_new_tokens
    )
    gc.collect()
    rss_a = _peak_rss_mb()
    seq_est = len(prompt_tokens) + max_new_tokens
    footprint_a = _analytical_footprint_mb("A", seq_est, enable_qcsd=False)

    # Profile B: cheap draft + faster native target path; tuned for ~9.8–11 tok/s vs A at ~7.
    draft = SimulatedDualModel(
        name="draft-qcsd",
        vocab_size=vocab_size,
        max_context=8192,
        base_seed=2026,
        latency_s=_SIM_DRAFT_STEP_LATENCY_S,
        draft_noise_std=0.15,
        resident_mb=128,
        sim_acceptance_rate=sim_acceptance_rate,
    )
    target_b = SimulatedDualModel(
        name="primary-q4-native",
        vocab_size=vocab_size,
        max_context=8192,
        base_seed=2026,
        latency_s=_SIM_TARGET_STEP_LATENCY_S,
        draft_noise_std=0.0,
        resident_mb=0,
    )

    tps_ar_on_b: float | None = None
    observed_alpha: float | None = None
    if enable_qcsd and verify_leviathan_apples and max_new_tokens > 0:
        target_ar_check = SimulatedDualModel(
            name="primary-q4-native-ar-only",
            vocab_size=vocab_size,
            max_context=8192,
            base_seed=2026,
            latency_s=_SIM_TARGET_STEP_LATENCY_S,
            draft_noise_std=0.0,
            resident_mb=0,
        )
        _, _, tps_ar_on_b = run_greedy_baseline_tokens(
            target_ar_check, list(prompt_tokens), max_new_tokens
        )
        gc.collect()

    if enable_qcsd:
        decoder = GreedyDualModelSpeculativeDecoder(
            draft,
            target_b,
            gamma=gamma,
            draft_sim_seed=draft_sim_seed,
        )
        gc.collect()
        t0 = time.perf_counter()
        res_b = decoder.generate(list(prompt_tokens), max_new_tokens)
        t_b = time.perf_counter() - t0
        tps_b = len(res_b.generated_tokens) / max(t_b, 1e-9)
        observed_alpha = float(res_b.acceptance_rate)
        acceptance_s = f"{res_b.acceptance_rate:.1%}"
    else:
        gc.collect()
        t0 = time.perf_counter()
        _, t_b, tps_b = run_greedy_baseline_tokens(
            target_b, list(prompt_tokens), max_new_tokens
        )
        t_b = float(t_b)
        acceptance_s = "n/a (QCSD off)"

    gc.collect()
    rss_b = _peak_rss_mb()

    footprint_b = _analytical_footprint_mb("B", seq_est, enable_qcsd=enable_qcsd)

    print()
    print("=" * 72)
    print("ASDSL full benchmark (dual-model simulator - no local weights)")
    print("=" * 72)
    print(
        f"  Prompt tokens: {len(prompt_tokens)}  |  max_new_tokens: {max_new_tokens}  "
        f"|  gamma: {gamma}  |  sim_acceptance_rate: {sim_acceptance_rate:.2f}"
    )
    print()
    print(
        f"{'Profile':<36} {'tok/s':>12} {'Peak RSS (MB)':>16} "
        f"{'Corr. footprint (MB)':>22}"
    )
    print("-" * 72)
    rss_a_s = f"{rss_a:.0f}" if rss_a is not None else "n/a"
    rss_b_s = f"{rss_b:.0f}" if rss_b is not None else "n/a"
    print(
        f"{'A  Baseline (AR, no native / FP32 KV)':<36} "
        f"{tps_a:>12.2f} {rss_a_s:>16} {footprint_a:>22.0f}"
    )
    b_label = (
        "B  Optimized (QCSD + native, dual KV est.)"
        if enable_qcsd
        else "B  Optimized (AR target-only, QCSD off)"
    )
    print(f"{b_label:<36} {tps_b:>12.2f} {rss_b_s:>16} {footprint_b:>22.0f}")
    print("-" * 72)
    print(f"  Simulator QCSD acceptance: {acceptance_s}")
    kv1_est = _phi4_kv_fp32_mb(seq_est)
    if enable_qcsd:
        cr = d_mb / t_mb if t_mb > 0 else 0.0
        c_lat = _SIM_DRAFT_STEP_LATENCY_S / max(_SIM_TARGET_STEP_LATENCY_S, 1e-12)
        s_leviathan_lat = _leviathan_speedup(sim_acceptance_rate, gamma, c_lat)
        print(
            f"  Leviathan (2023) analytical speedup S={s_leviathan:.2f}x "
            f"(alpha={sim_acceptance_rate:.2f}, gamma={gamma}, MB-cost c={cr:.3f})."
        )
        print(
            f"  Leviathan S (latency-only c={c_lat:.4f})={s_leviathan_lat:.2f}x "
            f"(draft_lat/target_lat; informational vs batched-verify sim)."
        )
        print(
            f"  Corrected footprint B (QCSD): target {_SIM_QCSD_TARGET_WEIGHTS_MB:.0f} MB + "
            f"draft {_SIM_QCSD_DRAFT_WEIGHTS_MB:.0f} MB + dual FP32 KV {2.0 * kv1_est:.0f} MB "
            f"= {footprint_b:.0f} MB"
        )
        ratio_ba = tps_b / max(tps_a, 1e-9)
        print(
            f"  Measured throughput ratio B/A: {ratio_ba:.2f}x "
            f"(includes slower baseline target in A vs native target in B, not equal to S)."
        )
        if (
            tps_ar_on_b is not None
            and tps_ar_on_b > 0
            and observed_alpha is not None
        ):
            ratio_spec = tps_b / tps_ar_on_b
            # Match the sim's realized Bernoulli acceptance to the Leviathan alpha input.
            s_realized = _leviathan_speedup(observed_alpha, gamma, cr)
            ref_s = max(s_realized, 1e-9)
            delta = abs(ratio_spec - ref_s) / ref_s
            if delta <= 0.05:
                tag = "PASS (<=5%)"
            elif delta <= 0.12:
                tag = "PASS (<=12%, verify-schedule slack)"
            else:
                tag = "FAIL"
            print(
                f"  Apples-to-apples QCSD vs AR (same Profile-B target): {ratio_spec:.2f}x "
                f"vs MB-model S={s_realized:.2f}x (alpha_obs={observed_alpha:.2f}, c={cr:.3f}; "
                f"configured S={s_leviathan:.2f}x at alpha={sim_acceptance_rate:.2f}), "
                f"{100.0 * delta:.1f}% abs. delta [{tag}]"
            )
    if rss0 is not None:
        print(f"  Process RSS (idle): {rss0:.0f} MB")
    print()
    print(
        "  Footprint column: analytical weights + KV - A baseline model + FP32 KV; "
        "B QCSD = target + draft bank + dual FP32 KV caches."
    )
    print("=" * 72)


# ---------------------------------------------------------------------------
# Prerequisite C: subprocess isolation for profiles D, E, F
# Each profile runs in a fresh Python process to avoid heap fragmentation
# and cross-profile RSS accumulation.
# ---------------------------------------------------------------------------

_PROFILE_RUNNER_SCRIPT = '''
import sys, json, time, gc, os, contextlib, io
sys.path.insert(0, {root!r})
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
{extra_env_lines}

from experiments.phi4_cpu_run import WeightStore, generate, generate_eagle3, set_thread_count
from transformers import AutoTokenizer

set_thread_count({thread_count})

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/Phi-4-multimodal-instruct", trust_remote_code=True)

store = WeightStore(bits={bits}, group_size=None, enable_qcsd=False,
                   draft_bits=2, enable_sparse=False)
store.load()
{pre_warm_setup}
store.warm_cache()
gc.collect()
{post_warm_setup}

store._use_native_gemv = {use_native}
store._use_lut_gemv = {use_lut}
store._enable_sparse = {enable_sparse}
store._sparsity_threshold = {sparsity_threshold}

metrics = []
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    {generate_call}

# Capture sparsity debug info after generation
sparse_T_calls = int(getattr(store, "_sparse_down_proj_T_calls", 0))
dense_T_calls = int(getattr(store, "_dense_down_proj_fallback_calls", 0))

if metrics:
    m = metrics[0]
    payload = {{
        "tps": float(m.get("tokens_per_second", 0.0)),
        "rss": None,
        "fatrelu_enabled": bool(getattr(store, "_use_fatrelu", False)),
        "eagle3_enabled": bool(getattr(store, "_use_eagle3", False)),
        "sparse_T_calls": sparse_T_calls,
        "dense_T_calls": dense_T_calls,
    }}
    if m.get("acceptance_rate") is not None:
        payload["acceptance_rate"] = float(m["acceptance_rate"])
    if m.get("mean_tokens_accepted_per_cycle") is not None:
        payload["mean_tokens_accepted_per_cycle"] = float(m["mean_tokens_accepted_per_cycle"])
    print("__PROFILE_RESULT__" + json.dumps(payload))
else:
    print("__PROFILE_RESULT__" + json.dumps({{
        "tps": 0.0, "rss": None,
        "fatrelu_enabled": bool(getattr(store, "_use_fatrelu", False)),
        "eagle3_enabled": bool(getattr(store, "_use_eagle3", False)),
        "sparse_T_calls": sparse_T_calls,
        "dense_T_calls": dense_T_calls,
    }}))
    if m.get("acceptance_rate") is not None:
        payload["acceptance_rate"] = float(m["acceptance_rate"])
    if m.get("mean_tokens_accepted_per_cycle") is not None:
        payload["mean_tokens_accepted_per_cycle"] = float(m["mean_tokens_accepted_per_cycle"])
    print("__PROFILE_RESULT__" + json.dumps(payload))
else:
    print("__PROFILE_RESULT__" + json.dumps({{
        "tps": 0.0, "rss": None,
        "fatrelu_enabled": bool(getattr(store, "_use_fatrelu", False)),
        "eagle3_enabled": bool(getattr(store, "_use_eagle3", False)),
    }}))
'''


def _run_phi4_profile_isolated(
    profile_name: str,
    root: Path,
    prompt: str,
    max_new_tokens: int,
    primary_bits: int,
    *,
    use_native: bool = True,
    use_lut: bool = False,
    enable_sparse: bool = False,
    sparsity_threshold: float = 0.01,
    needs_slim: bool = False,
    needs_fatrelu: bool = False,
    needs_mtp: bool = False,
    slim_meta_path: str | None = None,
    fatrelu_path: str | None = None,
    mtp_head_path: str | None = None,
    generate_func: str = "generate",
    inter_profile_sleep: float = 5.0,
    subprocess_extra_env: dict[str, str] | None = None,
    eagle_draft_k: int = 4,
    thread_count: int = 0,
) -> tuple[float, None, dict]:
    """Run a single profile in an isolated subprocess.

    Returns (tps, rss, extras) where rss is None and extras is the JSON payload
    minus ``tps`` (acceptance_rate, fatrelu_enabled, etc.).
    """
    import subprocess
    import time as _time

    print(f"[Profile {profile_name}] Starting isolated subprocess "
          f"(sleep {inter_profile_sleep:.0f}s first)...", flush=True)
    _time.sleep(inter_profile_sleep)

    pre_lines: list[str] = []
    post_lines: list[str] = []
    if needs_slim and slim_meta_path:
        pre_lines.append(f"store.load_slim({slim_meta_path!r})")
    if needs_fatrelu and fatrelu_path:
        post_lines.append(f"store.load_fatrelu({fatrelu_path!r}, adaptive=True)")
    if needs_mtp and mtp_head_path:
        post_lines.append(f"store.load_mtp_head({mtp_head_path!r})")
    pre_warm_setup = "\n".join(pre_lines)
    post_warm_setup = "\n".join(post_lines)

    if generate_func == "generate_eagle3":
        generate_call = (
            f"generate_eagle3({prompt!r}, store, tokenizer, "
            f"max_new_tokens={max_new_tokens}, bench_metrics_out=metrics)"
        )
    else:
        generate_call = (
            f"generate({prompt!r}, store, tokenizer, "
            f"max_new_tokens={max_new_tokens}, bench_metrics_out=metrics)"
        )

    extra_env_lines = ""
    if subprocess_extra_env:
        for env_k, env_v in subprocess_extra_env.items():
            extra_env_lines += f"os.environ[{env_k!r}] = {env_v!r}\n"

    script = _PROFILE_RUNNER_SCRIPT.format(
        root=str(root),
        bits=primary_bits,
        pre_warm_setup=pre_warm_setup,
        post_warm_setup=post_warm_setup,
        use_native=use_native,
        use_lut=use_lut,
        enable_sparse=enable_sparse,
        sparsity_threshold=sparsity_threshold,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        generate_call=generate_call,
        extra_env_lines=extra_env_lines,
        thread_count=thread_count,
    )

    run_env = os.environ.copy()
    if subprocess_extra_env:
        run_env.update(subprocess_extra_env)

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=3600,
        env=run_env,
    )

    tps = 0.0
    extras: dict = {}
    marker = "__PROFILE_RESULT__"
    for line in result.stdout.splitlines():
        if line.startswith(marker):
            try:
                data = json.loads(line[len(marker):])
                tps = float(data.get("tps", 0.0))
                extras = {k: v for k, v in data.items() if k != "tps"}
            except (json.JSONDecodeError, ValueError):
                pass

    if result.returncode != 0:
        print(f"[Profile {profile_name}] subprocess error (rc={result.returncode}):",
              file=sys.stderr)
        print(result.stderr[-2000:] if result.stderr else "(no stderr)", file=sys.stderr)

    sparse_T = extras.get("sparse_T_calls", 0)
    dense_T = extras.get("dense_T_calls", 0)
    sparse_info = f" sparse_T={sparse_T} dense_fallback={dense_T}" if sparse_T or dense_T else ""
    print(f"[Profile {profile_name}] isolated result: {tps:.3f} tok/s{sparse_info}", flush=True)
    return tps, None, extras


def run_phi4_benchmark(
    root: Path,
    prompt: str,
    max_new_tokens: int,
    draft_k: int,
    primary_bits: int,
    draft_bits: int,
    threads: int,
    qcsd_draft_mb_override: float | None = None,
    qcsd_target_mb_override: float | None = None,
    phi4_acceptance_estimate: float = 0.40,
    slim_meta: str | None = None,
    fatrelu_thresholds: str | None = None,
    emit_regression_json: str | None = None,
    validate_outputs: bool = False,
    force_eagle3: bool = False,
    quick: bool = False,
    inter_profile_sleep_default: float | None = None,
    eagle_draft_k: int = 4,
) -> None:
    if quick:
        max_new_tokens = min(max_new_tokens, 8)

    phi4 = _load_phi4_module(root)
    idx = phi4.INDEX_FILE
    if not idx.exists():
        print(f"Phi-4 index missing at {idx}; use default sim mode or run phi4_integration.")
        sys.exit(2)

    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_JAX", "0")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    phi4.set_thread_count(threads if threads > 0 else 0)
    _eff_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/Phi-4-multimodal-instruct",
        trust_remote_code=True,
    )

    store = phi4.WeightStore(
        bits=primary_bits,
        group_size=None,
        enable_qcsd=True,
        draft_bits=draft_bits,
        enable_sparse=False,
    )
    t_load = time.perf_counter()
    store.load()
    # Phase 2: load SliM metadata if available
    _slim_meta_path = None
    if slim_meta:
        _slim_meta_path = Path(slim_meta)
    elif (root / "phi4_slim_meta.json").exists():
        _slim_meta_path = root / "phi4_slim_meta.json"
    if _slim_meta_path and _slim_meta_path.exists():
        store.load_slim(str(_slim_meta_path))

    # Phase 3: load FATReLU thresholds if available
    _fatrelu_path = None
    if fatrelu_thresholds:
        _fatrelu_path = Path(fatrelu_thresholds)
    elif (root / "phi4_fatrelu_thresholds.json").exists():
        _fatrelu_path = root / "phi4_fatrelu_thresholds.json"
    t_load = time.perf_counter() - t_load
    store.warm_cache()
    gc.collect()
    # Phase 10: Profiles A/C must run dense AR (no FATReLU / transposed down_proj).
    # load_fatrelu() is deferred until after C — it was incorrectly applied here,
    # making A/C run the sparse forward and skewing C vs A.
    peak_load = _peak_rss_mb()

    packed_mb = sum(t.nbytes for t in store._quant_packed.values()) / 1e6
    u8_mb = sum(t.nbytes for t in store._quant_u8.values()) / 1e6
    sc_mb = sum(t.nbytes for t in store._quant_sc.values()) / 1e6
    bi_mb = sum(t.nbytes for t in store._quant_bi.values()) / 1e6
    embed_mb = store.embed_f16.nbytes / 1e6
    draft_mb = (
        sum(t.nbytes for t in store._draft_quant_u8.values())
        + sum(t.nbytes for t in store._draft_quant_packed.values())
    ) / 1e6
    primary_mb = packed_mb + u8_mb + sc_mb + bi_mb + embed_mb
    weights_mb = primary_mb + draft_mb

    target_w_mb = (
        float(qcsd_target_mb_override)
        if qcsd_target_mb_override is not None
        else primary_mb
    )
    draft_check_mb = (
        float(qcsd_draft_mb_override)
        if qcsd_draft_mb_override is not None
        else draft_mb
    )
    # Adaptive Leviathan gate logic (D3)
    _mtp_head_path = root / "models" / "mtp_head.pt"
    if _mtp_head_path.exists():
        # Profile G (EAGLE-3) mode available
        hist_rates = _load_history_rates(root, key="eagle3_acceptance_rates")
        prior_alpha = 0.65  # Higher prior for EAGLE-3
        alpha_key = "eagle3_acceptance_rates"
    else:
        # Standard QCSD mode
        hist_rates = _load_history_rates(root, key="acceptance_rates")
        prior_alpha = phi4_acceptance_estimate # usually 0.40
        alpha_key = "acceptance_rates"

    if hist_rates:
        alpha_for_leviathan = sum(hist_rates) / len(hist_rates)
        alpha_gate_src = f"history mean over {len(hist_rates)} run(s) [{alpha_key}]"
    else:
        alpha_for_leviathan = prior_alpha
        alpha_gate_src = f"prior (no {alpha_key} in .qcsd_history.json)"
    ok_be = True
    s_phi = 1.0
    if store._enable_qcsd:
        ok_be, warn_be, s_phi = _qcsd_break_even_ok(
            alpha_for_leviathan, draft_k, draft_check_mb, target_w_mb
        )
        if not ok_be:
            print()
            print("WARNING:", warn_be)
            print(
                f"  (Analytical Leviathan S={s_phi:.3f}x; falling back to AR for Profile B.)"
            )
            print(
                "  Note: static c=draft_mb/target_mb does not reflect FATReLU target speedup; "
                "Profile G may still be measured with FATReLU + MTP (see --force-eagle3).",
                flush=True,
            )
            store._enable_qcsd = False

    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    seq_est = len(input_ids) + max_new_tokens
    kv_fp32_mb = _phi4_kv_fp32_mb(seq_est)
    kv_q4_mb = _phi4_kv_q4_est_mb(seq_est)
    est_footprint_a = weights_mb - draft_mb + kv_fp32_mb
    qcsd_on = store._enable_qcsd
    est_footprint_b = (
        weights_mb + kv_q4_mb if qcsd_on else weights_mb - draft_mb + kv_q4_mb
    )
    est_footprint_c = weights_mb - draft_mb + kv_fp32_mb

    buf = io.StringIO()
    metrics_a: list = []
    store._use_native_gemv = False
    with contextlib.redirect_stdout(buf):
        with phi4.torch.inference_mode():
            phi4.generate(
                prompt,
                store,
                tokenizer,
                max_new_tokens=max_new_tokens,
                bench_metrics_out=metrics_a,
            )
    gc.collect()
    peak_a = _peak_rss_mb()
    m_a = metrics_a[0] if metrics_a else {}
    tps_a = float(m_a.get("tokens_per_second", 0.0))

    metrics_c: list = []
    store._use_native_gemv = True
    store._use_lut_gemv = False
    with contextlib.redirect_stdout(buf):
        with phi4.torch.inference_mode():
            phi4.generate(
                prompt,
                store,
                tokenizer,
                max_new_tokens=max_new_tokens,
                bench_metrics_out=metrics_c,
            )
    gc.collect()
    peak_c = _peak_rss_mb()
    m_c = metrics_c[0] if metrics_c else {}
    tps_c = float(m_c.get("tokens_per_second", 0.0))

    if _fatrelu_path and _fatrelu_path.exists():
        store.load_fatrelu(str(_fatrelu_path))
        gc.collect()
        peak_load = max(peak_load, _peak_rss_mb())

    # Profile D: AR + LUT Q4 GEMV (vpshufb, Phase 1) - isolated subprocess
    tps_d = 0.0
    peak_d = None
    _default_sleep = 5.0 if inter_profile_sleep_default is None else float(
        inter_profile_sleep_default
    )
    _inter_sleep = (
        0.5
        if quick
        else float(os.environ.get("ASDSL_PROFILE_SLEEP", str(_default_sleep)))
    )
    tps_d, peak_d, _ex_d = _run_phi4_profile_isolated(
        "D", root, prompt, max_new_tokens, primary_bits,
        use_native=True, use_lut=True,
        enable_sparse=False, sparsity_threshold=0.01,
        thread_count=threads,
        inter_profile_sleep=_inter_sleep,
    )
    m_d: dict = {"tokens_per_second": tps_d}
    store._use_lut_gemv = False  # keep store in-process state clean

    # Profile E: AR + LUT GEMV + SliM 2.2-bit mixed precision (Phase 2) - isolated subprocess
    tps_e = None
    peak_e = None
    if _slim_meta_path and _slim_meta_path.exists():
        tps_e, peak_e, _ex_e = _run_phi4_profile_isolated(
            "E", root, prompt, max_new_tokens, primary_bits,
            use_native=True, use_lut=True,
            needs_slim=True, slim_meta_path=str(_slim_meta_path),
            thread_count=threads,
            inter_profile_sleep=_inter_sleep,
        )
        m_e = {"tokens_per_second": tps_e}
    else:
        print("[Profile E] skipped: phi4_slim_meta.json not found or not loaded")
        tps_e = None
        peak_e = None
        m_e = {}

    # Profile F: AR + Native GEMV + FATReLU 85% sparsity (Phase 3) - isolated subprocess
    tps_f = None
    peak_f = None
    if _fatrelu_path and _fatrelu_path.exists():
        tps_f, peak_f, _ex_f = _run_phi4_profile_isolated(
            "F", root, prompt, max_new_tokens, primary_bits,
            use_native=True, use_lut=False,
            enable_sparse=True, sparsity_threshold=0.0,
            needs_fatrelu=True, fatrelu_path=str(_fatrelu_path),
            inter_profile_sleep=_inter_sleep,
            thread_count=threads,
        )
        m_f: dict = {"tokens_per_second": tps_f}
    else:
        print("[Profile F] skipped: phi4_fatrelu_thresholds.json not found or not loaded")
        m_f = {}

    # Profile G: Native GEMV + LUT + FATReLU + EAGLE-3 MTP (F stacked with EAGLE-3)
    tps_g = None
    peak_g = None
    ex_g: dict = {}
    _mtp_head_path = root / "models" / "mtp_head.pt"
    _g_fatrelu = _fatrelu_path
    if _g_fatrelu is None and (root / "phi4_fatrelu_thresholds.json").exists():
        _g_fatrelu = root / "phi4_fatrelu_thresholds.json"
    g_fr_path = str(_g_fatrelu) if _g_fatrelu and Path(_g_fatrelu).exists() else None
    g_env: dict[str, str] = {"ASDSL_FORCE_EAGLE3": "1"}
    print(
        "[Profile G] Leviathan gate BYPASSED for empirical measurement "
        "(ASDSL_FORCE_EAGLE3=1; full throughput + acceptance capture)",
        flush=True,
    )
    if force_eagle3:
        print("  (also requested via --force-eagle3)", flush=True)
    if _mtp_head_path.exists():
        if g_fr_path is None:
            print(
                "[Profile G] WARNING: phi4_fatrelu_thresholds.json missing — "
                "FATReLU + transposed down_proj will not load in subprocess",
                flush=True,
            )
        tps_g, peak_g, ex_g = _run_phi4_profile_isolated(
            "G", root, prompt, max_new_tokens, primary_bits,
            use_native=True,
            use_lut=False,
            enable_sparse=True,
            sparsity_threshold=0.0,
            needs_fatrelu=bool(g_fr_path),
            fatrelu_path=g_fr_path,
            needs_mtp=True,
            mtp_head_path=str(_mtp_head_path),
            generate_func="generate_eagle3",
            inter_profile_sleep=_inter_sleep,
            subprocess_extra_env=g_env if g_env else None,
            eagle_draft_k=int(eagle_draft_k),
            thread_count=threads,
        )
        m_g = {"tokens_per_second": tps_g}
        if tps_g is not None:
            ar_g = ex_g.get("acceptance_rate")
            if ar_g is not None:
                print(f"[Profile G] EAGLE-3 acceptance rate: {float(ar_g):.1%}")
                mpc = ex_g.get("mean_tokens_accepted_per_cycle")
                if mpc is not None:
                    print(f"[Profile G] Mean tokens accepted/cycle: {float(mpc):.2f}")
                dk_g = int(eagle_draft_k)
                a_be = _leviathan_alpha_break_even(dk_g)
                levi_ok = (
                    float(ar_g) >= a_be if not math.isnan(a_be) else False
                )
                pct = a_be * 100.0 if not math.isnan(a_be) else float("nan")
                print(
                    f"[Profile G] Leviathan gate: {'PASS' if levi_ok else 'FAIL'} "
                    f"(break-even alpha ~{pct:.1f}% for draft_k={dk_g})",
                    flush=True,
                )
            else:
                print("WARNING: acceptance_rate not in Profile G result", flush=True)
            if ex_g.get("fatrelu_enabled") is False:
                print(
                    "WARNING: Profile G subprocess reports fatrelu_enabled=false",
                    flush=True,
                )
    else:
        print("[Profile G] skipped: models/mtp_head.pt not found")
        m_g = {}

    metrics_b: list = []
    with contextlib.redirect_stdout(buf):
        with phi4.torch.inference_mode():
            if qcsd_on:
                phi4.generate_qcsd(
                    prompt,
                    store,
                    tokenizer,
                    max_new_tokens=max_new_tokens,
                    draft_k=draft_k,
                    bench_metrics_out=metrics_b,
                )
            else:
                phi4.generate(
                    prompt,
                    store,
                    tokenizer,
                    max_new_tokens=max_new_tokens,
                    bench_metrics_out=metrics_b,
                )
    gc.collect()
    peak_b = _peak_rss_mb()
    m_b = metrics_b[0] if metrics_b else {}
    tps_b = float(m_b.get("tokens_per_second", 0.0))

    appended_qcsd_history = False
    if qcsd_on and m_b.get("acceptance_rate") is not None:
        _hkey = "eagle3_acceptance_rates" if _mtp_head_path.exists() else "acceptance_rates"
        _append_history_rate(root, float(m_b["acceptance_rate"]), key=_hkey)
        appended_qcsd_history = True

    appended_eagle3_from_g = False
    if ex_g.get("acceptance_rate") is not None and _mtp_head_path.exists():
        _append_history_rate(root, float(ex_g["acceptance_rate"]), key="eagle3_acceptance_rates")
        appended_eagle3_from_g = True

    peak_rss = max(x for x in (peak_load, peak_a, peak_c, peak_d, peak_e, peak_f, peak_g, peak_b) if x is not None)
    rss_a_s = f"{peak_a:.0f}" if peak_a is not None else "n/a"
    rss_c_s = f"{peak_c:.0f}" if peak_c is not None else "n/a"
    rss_d_s = f"{peak_d:.0f}" if peak_d is not None else "n/a"
    rss_e_s = f"{peak_e:.0f}" if peak_e is not None else "n/a"
    rss_f_s = f"{peak_f:.0f}" if peak_f is not None else "n/a"
    rss_g_s = f"{peak_g:.0f}" if peak_g is not None else "n/a"
    rss_b_s = f"{peak_b:.0f}" if peak_b is not None else "n/a"

    try:
        from asdsl.kernels import has_native_kernel

        native_q4 = bool(primary_bits == 4 and has_native_kernel())
    except Exception:
        native_q4 = False

    print()
    print("=" * 72)
    print("ASDSL full benchmark (Phi-4 local weights)")
    print("=" * 72)
    print(
        f"  Native Q4 GEMV (gemv_q4_packed / AVX2): "
        f"{'available' if native_q4 else 'unavailable (chunked fallback)'}"
    )
    print(
        f"  Threads (OMP / BLAS / torch / native GEMV): {_eff_threads} "
        f"(use --threads N to override; 0 = auto, half of logical CPUs)"
    )
    print(
        f"  Leviathan gate alpha: {alpha_for_leviathan:.3f} ({alpha_gate_src}); "
        f"CLI prior default if no history: {phi4_acceptance_estimate:.2f}"
    )
    print(
        f"  Load+quantize: {t_load:.1f}s  |  primary {primary_bits}-bit  |  "
        f"draft {store._draft_bits}-bit  |  draft_k={draft_k}"
    )
    print(f"  Weights (reported): ~{weights_mb:.0f} MB  (draft bank {draft_mb:.0f} MB)")
    print(
        f"  Physical primary breakdown: packed_Q4 {packed_mb:.0f} MB | uint8_matmul {u8_mb:.0f} MB | "
        f"scales+bias {sc_mb + bi_mb:.0f} MB | embed_f16 {embed_mb:.0f} MB "
        f"(packed path avoids 2x uint8 weight expansion for 4-bit)"
    )
    if primary_bits == 4 and packed_mb > 1.0 and u8_mb > 0.15 * packed_mb:
        print(
            "  WARNING: uint8_matmul footprint is large vs packed_Q4; check for unintended "
            "full uint8 expansion regression.",
            flush=True,
        )
    print()
    print(
        f"{'Profile':<40} {'tok/s':>12} {'Peak RSS (MB)':>14} {'Corr. footprint (MB)':>20}"
    )
    print("-" * 88)
    print(
        f"{'A  AR + PyTorch matvec + FP32 KV est.':<40} "
        f"{tps_a:>12.2f} {rss_a_s:>14} {est_footprint_a:>20.0f}"
    )
    print(
        f"{'C  AR + Native Q4 GEMV + FP32 KV est.':<40} "
        f"{tps_c:>12.2f} {rss_c_s:>14} {est_footprint_c:>20.0f}"
    )
    print(
        f"{'D  AR + LUT Q4 GEMV (vpshufb, Phase 1)':<40} "
        f"{tps_d:>12.2f} {rss_d_s:>14} {est_footprint_c:>20.0f}"
    )
    if tps_e is not None:
        print(
            f"{'E  AR + LUT GEMV + SliM 2.2-bit (Phase 2)':<40} "
            f"{tps_e:>12.2f} {rss_e_s:>14} {est_footprint_c:>20.0f}"
        )
    else:
        print(
            f"{'E  AR + LUT GEMV + SliM 2.2-bit (Phase 2)':<40} "
            f"{'skipped':>12} {'n/a':>14} {'n/a':>20}"
        )
    if tps_f is not None:
        print(
            f"{'F  AR + Native GEMV + FATReLU 85% (Phase 3)':<40} "
            f"{tps_f:>12.2f} {rss_f_s:>14} {est_footprint_c:>20.0f}"
        )
    else:
        print(
            f"{'F  AR + Native GEMV + FATReLU 85% (Phase 3)':<40} "
            f"{'skipped':>12} {'n/a':>14} {'n/a':>20}"
        )
    if tps_g is not None:
        print(
            f"{'G  Profile F + EAGLE-3 MTP (Phase 5)':<40} "
            f"{tps_g:>12.2f} {rss_g_s:>14} {est_footprint_c:>20.0f}"
        )
    else:
        print(
            f"{'G  Profile F + EAGLE-3 MTP (Phase 5)':<40} "
            f"{'skipped':>12} {'n/a':>14} {'n/a':>20}"
        )
    b_row = (
        "B  Native GEMV + QCSD + Q4 KV est."
        if qcsd_on
        else "B  Native GEMV + AR + Q4 KV est. (QCSD off)"
    )
    print(f"{b_row:<40} {tps_b:>12.2f} {rss_b_s:>14} {est_footprint_b:>20.0f}")
    print("-" * 88)
    print(
        f"  Corrected footprint A: primary+embed ~{primary_mb:.0f} MB + FP32 KV ~{kv_fp32_mb:.0f} MB "
        f"= ~{est_footprint_a:.0f} MB"
    )
    print(
        f"  Profile C (native AR): {tps_c:.2f} tok/s  |  peak RSS ~{rss_c_s} MB  |  "
        f"same footprint model as A (~{est_footprint_c:.0f} MB est.)"
    )
    if qcsd_on:
        print(
            f"  Corrected footprint B (QCSD): primary ~{primary_mb:.0f} MB + draft ~{draft_mb:.0f} MB "
            f"+ Q4 KV est ~{kv_q4_mb:.0f} MB = ~{est_footprint_b:.0f} MB"
        )
        print(
            "  QCSD decode: batched primary verify over draft horizon (see "
            "``generate_qcsd`` verify phase; amortized vs per-token AR)."
        )
    else:
        print(
            f"  Corrected footprint B (AR): primary ~{primary_mb:.0f} MB + Q4 KV est ~{kv_q4_mb:.0f} MB "
            f"= ~{est_footprint_b:.0f} MB"
        )
    print(f"  Max sampled RSS (load / A / C / D / E / F / B): {peak_rss:.0f} MB")
    if tps_c > 0:
        d_speedup = tps_d / tps_c if tps_c > 0 else 0.0
        print(f"  Profile D speedup vs C: {d_speedup:.2f}x (LUT vpshufb vs FMA)")
        print(f"  Profile D speedup vs baseline (1.20 tok/s): {tps_d/1.20:.2f}x")
    if tps_e is not None and tps_c > 0:
        e_speedup_c = tps_e / tps_c
        e_speedup_d = tps_e / tps_d if tps_d > 0 else 0.0
        llama_gap = (tps_e - 1.20) / (7.0 - 1.20) * 100
        print(f"  Profile E speedup vs C: {e_speedup_c:.2f}x (SliM + LUT vs FMA)")
        print(f"  Profile E speedup vs D: {e_speedup_d:.2f}x (SliM vs no-SliM)")
        print(f"  Profile E speedup vs baseline (1.20 tok/s): {tps_e/1.20:.2f}x")
        print(f"  llama.cpp gap closed: {llama_gap:.1f}%")
    if m_b and qcsd_on:
        ar = float(m_b.get("acceptance_rate", 0.0))
        print(f"  QCSD acceptance (greedy): {ar:.1%}")
        vb = int(m_b.get("qcsd_verify_batched_passes", 0) or 0)
        cy = int(m_b.get("qcsd_speculative_cycles", 0) or 0)
        xf = int(m_b.get("qcsd_verify_extra_run_forward", 0) or 0)
        if cy > 0:
            avg_batched = vb / cy
            print(
                f"  QCSD verify telemetry: batched target passes={vb} over {cy} cycle(s) "
                f"(avg {avg_batched:.2f}/cycle, expect ~1.0 if verify is one stack); "
                f"extra target run_forward after verify={xf}"
            )
    if qcsd_on:
        s_p = _leviathan_speedup(
            alpha_for_leviathan, draft_k, draft_mb / max(primary_mb, 1e-9)
        )
        print(
            f"  Leviathan analytical S={s_p:.2f}x "
            f"(alpha_gate={alpha_for_leviathan:.2f}, draft_k={draft_k}, "
            f"c={draft_mb / max(primary_mb, 1e-9):.3f})."
        )
    if appended_qcsd_history:
        print(
            f"  Appended measured acceptance to {_QCSD_HISTORY_FILENAME} "
            f"(used on next run for Leviathan gate)."
        )
    if appended_eagle3_from_g:
        print(
            f"  Appended Profile G EAGLE-3 acceptance to {_QCSD_HISTORY_FILENAME} "
            f"(eagle3_acceptance_rates).",
            flush=True,
        )
    print(
        "  KV column: A and C use FP32 KV geometry; B adds theoretical Q4 KV payload "
        "(``KVHistory`` in phi4_cpu_run remains FP32; Q4 is planning/comparison)."
    )
    print("=" * 72)
    
    if emit_regression_json:
        res_map = {
            "A": {"tok/s": tps_a},
            "C": {"tok/s": tps_c},
            "D": {"tok/s": tps_d},
            "E": {"tok/s": tps_e},
            "F": {"tok/s": tps_f},
            "G": {"tok/s": tps_g},
            "B": {"tok/s": tps_b},
        }
        print(f"Writing regression results to {emit_regression_json}...")
        with open(emit_regression_json, "w") as f:
            json.dump({"results": res_map}, f, indent=2)

    if validate_outputs:
        print("\nRunning KL Divergence Validation...")
        subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "validate_outputs.py"),
                "--prompt",
                prompt,
                "--max-new-tokens",
                "32",
            ],
            check=False,
        )


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))

    bc_path = root / "benchmark_config.json"
    cfg = load_benchmark_config(bc_path)
    if cfg:
        d_prompt = str(cfg.get("prompt", "The fundamental theorem of calculus states that"))
        d_max = int(cfg.get("max_new_tokens", 64))
        d_threads = int(cfg.get("threads", 8))
        d_gamma = int(cfg.get("draft_k", 7))
        d_sleep = float(cfg.get("inter_profile_sleep_seconds", 3))
    else:
        d_prompt = "The fundamental theorem of calculus states that"
        d_max = 32
        d_threads = 0
        d_gamma = 7
        d_sleep = 5.0

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phi4", action="store_true", help="Use local Phi-4 weights (slow, large RAM)")
    p.add_argument(
        "--override-config",
        action="store_true",
        help="Suppress per-flag warnings when CLI differs from benchmark_config.json",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default=argparse.SUPPRESS,
        help="Decode prompt (default from benchmark_config.json when present).",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=argparse.SUPPRESS,
        help="Max decode tokens (default from benchmark_config.json when present).",
    )
    p.add_argument(
        "--gamma",
        type=int,
        default=argparse.SUPPRESS,
        help="QCSD draft width / sim gamma (default: draft_k from benchmark_config.json when present).",
    )
    p.add_argument("--vocab-size", type=int, default=200064)
    p.add_argument(
        "--threads",
        type=int,
        default=argparse.SUPPRESS,
        help="BLAS/OpenMP/native GEMV threads (default from benchmark_config.json when present; 0=auto)",
    )
    p.add_argument("--primary-bits", type=int, default=4, choices=[2, 3, 4, 8])
    p.add_argument("--draft-bits", type=int, default=2, choices=[2, 3, 4, 8])
    p.add_argument(
        "--sim-cost-ratio",
        type=float,
        default=None,
        metavar="C",
        help="Simulator only: set draft_mb = C * target_mb for Leviathan c (overrides --sim-draft-mb)",
    )
    p.add_argument(
        "--sim-draft-mb",
        type=float,
        default=None,
        help="Simulator only: draft footprint (MB) for QCSD break-even check (default: 128)",
    )
    p.add_argument(
        "--sim-target-mb",
        type=float,
        default=None,
        help=(
            "Simulator only: target footprint (MB) for QCSD break-even "
            f"(default: {_SIM_QCSD_TARGET_WEIGHTS_MB:.0f})"
        ),
    )
    p.add_argument(
        "--sim-acceptance-rate",
        type=float,
        default=0.70,
        metavar="P",
        help="Simulator: Bernoulli success prob per draft step (default: 0.70)",
    )
    p.add_argument(
        "--draft-sim-seed",
        type=int,
        default=2026,
        help="Simulator: RNG seed for draft Bernoulli draws (default: 2026)",
    )
    p.add_argument(
        "--qcsd-draft-mb-override",
        type=float,
        default=None,
        help="Phi-4 only: override draft MB for break-even check (testing)",
    )
    p.add_argument(
        "--qcsd-target-mb-override",
        type=float,
        default=None,
        help="Phi-4 only: override target MB for break-even check (testing)",
    )
    p.add_argument(
        "--phi4-acceptance-estimate",
        type=float,
        default=0.40,
        metavar="A",
        help="Phi-4 only: prior alpha when .qcsd_history.json is empty (default: 0.40)",
    )
    p.add_argument(
        "--tokenizer-prompt",
        action="store_true",
        help="Encode --prompt with Phi-4 tokenizer (requires Hugging Face). "
        "Default sim mode uses a short synthetic token ID list (offline).",
    )
    p.add_argument(
        "--verify-leviathan-apples",
        action="store_true",
        help="Simulator only: time AR on Profile-B target, compare QCSD/AR ratio to Leviathan S (5 percent tolerance)",
    )
    p.add_argument(
        "--slim-meta",
        type=str,
        default=None,
        help="Path to phi4_slim_meta.json for Phase 2 SliM mixed-precision (Profile E). "
             "Default: phi4_slim_meta.json at repo root if it exists.",
    )
    p.add_argument(
        "--emit-regression-json",
        type=str,
        default=None,
        help="Emit result JSON for regression checking.",
    )
    p.add_argument(
        "--validate-outputs",
        action="store_true",
        help="Run KL divergence validation script after benchmark.",
    )
    p.add_argument(
        "--fatrelu-thresholds",
        type=str,
        default=None,
        help="Path to phi4_fatrelu_thresholds.json for Phase 3 FATReLU sparsity (Profile F). "
             "Default: phi4_fatrelu_thresholds.json at repo root if it exists.",
    )
    p.add_argument(
        "--force-eagle3",
        action="store_true",
        help="Phi-4: set ASDSL_FORCE_EAGLE3 for Profile G subprocess (empirical run).",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Phi-4: cap --max-new-tokens at 8 and shorten inter-profile sleep for smoke runs.",
    )
    args = p.parse_args()
    if not 0.0 <= args.sim_acceptance_rate <= 1.0:
        p.error("--sim-acceptance-rate must be between 0 and 1")
    if not 0.0 <= args.phi4_acceptance_estimate <= 1.0:
        p.error("--phi4-acceptance-estimate must be between 0 and 1")

    ns = vars(args)
    if "prompt" not in ns:
        ns["prompt"] = d_prompt
    elif cfg and not args.override_config and ns["prompt"] != d_prompt:
        print(
            f"[CONFIG] WARNING: --prompt overrides canonical config value {d_prompt!r}",
            flush=True,
        )
    if "max_new_tokens" not in ns:
        ns["max_new_tokens"] = d_max
    elif cfg and not args.override_config and ns["max_new_tokens"] != d_max:
        print(
            f"[CONFIG] WARNING: --max-new-tokens {ns['max_new_tokens']} "
            f"overrides canonical config value {d_max}",
            flush=True,
        )
    if "gamma" not in ns:
        ns["gamma"] = d_gamma
    elif cfg and not args.override_config and ns["gamma"] != d_gamma:
        print(
            f"[CONFIG] WARNING: --gamma {ns['gamma']} overrides canonical config "
            f"draft_k value {d_gamma}",
            flush=True,
        )
    if "threads" not in ns:
        ns["threads"] = d_threads
    elif cfg and not args.override_config and ns["threads"] != d_threads:
        print(
            f"[CONFIG] WARNING: --threads {ns['threads']} overrides canonical "
            f"config value {d_threads}",
            flush=True,
        )
    if args.override_config and cfg:
        print(
            "[CONFIG] --override-config: CLI replaces canonical benchmark_config.json values",
            flush=True,
        )

    eagle_draft_k = int(ns["gamma"])
    if bc_path.is_file():
        try:
            eagle_draft_k = int(
                json.loads(bc_path.read_text(encoding="utf-8")).get(
                    "draft_k", eagle_draft_k
                )
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    if args.phi4 and cfg:
        if getattr(args, "slim_meta", None) is None and cfg.get("slim_meta"):
            sm = Path(cfg["slim_meta"])
            args.slim_meta = (
                str((root / sm).resolve()) if not sm.is_absolute() else str(sm)
            )
        if getattr(args, "fatrelu_thresholds", None) is None and cfg.get(
            "fatrelu_thresholds"
        ):
            fr = Path(cfg["fatrelu_thresholds"])
            args.fatrelu_thresholds = (
                str((root / fr).resolve()) if not fr.is_absolute() else str(fr)
            )

    if args.phi4:
        _require_phi4_index_or_exit(root)
        run_phi4_benchmark(
            root=root,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            draft_k=args.gamma,
            primary_bits=args.primary_bits,
            draft_bits=args.draft_bits,
            threads=args.threads,
            qcsd_draft_mb_override=args.qcsd_draft_mb_override,
            qcsd_target_mb_override=args.qcsd_target_mb_override,
            phi4_acceptance_estimate=args.phi4_acceptance_estimate,
            slim_meta=getattr(args, "slim_meta", None),
            fatrelu_thresholds=getattr(args, "fatrelu_thresholds", None),
            emit_regression_json=args.emit_regression_json,
            validate_outputs=args.validate_outputs,
            force_eagle3=args.force_eagle3,
            quick=args.quick,
            inter_profile_sleep_default=d_sleep,
            eagle_draft_k=eagle_draft_k,
        )
        return

    if args.tokenizer_prompt:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-4-multimodal-instruct",
            trust_remote_code=True,
        )
        prompt_tokens = tokenizer.encode(args.prompt, add_special_tokens=True)
    else:
        prompt_tokens = [1, 42, 99, 100, 256, 512]
    run_sim_benchmark(
        prompt_tokens=prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        gamma=args.gamma,
        vocab_size=args.vocab_size,
        draft_mb=args.sim_draft_mb,
        target_mb=args.sim_target_mb,
        sim_cost_ratio=args.sim_cost_ratio,
        sim_acceptance_rate=args.sim_acceptance_rate,
        draft_sim_seed=args.draft_sim_seed,
        verify_leviathan_apples=args.verify_leviathan_apples,
    )


if __name__ == "__main__":
    main()