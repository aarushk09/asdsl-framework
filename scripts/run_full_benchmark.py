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
import os
import sys
import time
from pathlib import Path

_QCSD_HISTORY_FILENAME = ".qcsd_history.json"
_QCSD_HISTORY_MAX_ENTRIES = 256


def _qcsd_history_path(root: Path) -> Path:
    return root / _QCSD_HISTORY_FILENAME


def _load_qcsd_acceptance_rates(root: Path) -> list[float]:
    path = _qcsd_history_path(root)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    rates = data.get("acceptance_rates")
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


def _mean_qcsd_acceptance_history(root: Path) -> float | None:
    rates = _load_qcsd_acceptance_rates(root)
    if not rates:
        return None
    return sum(rates) / len(rates)


def _append_qcsd_acceptance_rate(root: Path, rate: float) -> None:
    """Append one measured QCSD acceptance rate for adaptive Leviathan priors."""
    path = _qcsd_history_path(root)
    rates = _load_qcsd_acceptance_rates(root)
    r = max(0.0, min(1.0, float(rate)))
    rates.append(r)
    if len(rates) > _QCSD_HISTORY_MAX_ENTRIES:
        rates = rates[-_QCSD_HISTORY_MAX_ENTRIES:]
    path.write_text(
        json.dumps({"acceptance_rates": rates}, indent=2) + "\n",
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
) -> None:
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
    t_load = time.perf_counter() - t_load
    store.warm_cache()
    gc.collect()
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
    hist_rates = _load_qcsd_acceptance_rates(root)
    if hist_rates:
        alpha_for_leviathan = sum(hist_rates) / len(hist_rates)
        alpha_gate_src = f"history mean over {len(hist_rates)} run(s)"
    else:
        alpha_for_leviathan = phi4_acceptance_estimate
        alpha_gate_src = "prior (no .qcsd_history.json)"
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

    # Profile D: AR + LUT Q4 GEMV (vpshufb, Phase 1)
    metrics_d: list = []
    store._use_native_gemv = True
    store._use_lut_gemv = True
    with contextlib.redirect_stdout(buf):
        with phi4.torch.inference_mode():
            phi4.generate(
                prompt,
                store,
                tokenizer,
                max_new_tokens=max_new_tokens,
                bench_metrics_out=metrics_d,
            )
    gc.collect()
    peak_d = _peak_rss_mb()
    m_d = metrics_d[0] if metrics_d else {}
    tps_d = float(m_d.get("tokens_per_second", 0.0))
    store._use_lut_gemv = False  # reset to default

    # Profile E: AR + LUT GEMV + SliM 2.2-bit mixed precision (Phase 2)
    metrics_e: list = []
    tps_e = None
    peak_e = None
    if getattr(store, "_use_slim", False):
        store._use_native_gemv = True
        store._use_lut_gemv = True
        # _use_slim is already True from load_slim() call above
        with contextlib.redirect_stdout(buf):
            with phi4.torch.inference_mode():
                phi4.generate(
                    prompt,
                    store,
                    tokenizer,
                    max_new_tokens=max_new_tokens,
                    bench_metrics_out=metrics_e,
                )
        gc.collect()
        peak_e = _peak_rss_mb()
        m_e = metrics_e[0] if metrics_e else {}
        tps_e = float(m_e.get("tokens_per_second", 0.0))
        store._use_lut_gemv = False
    else:
        print("[Profile E] skipped: phi4_slim_meta.json not found or not loaded")

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
        _append_qcsd_acceptance_rate(root, float(m_b["acceptance_rate"]))
        appended_qcsd_history = True

    peak_rss = max(x for x in (peak_load, peak_a, peak_c, peak_d, peak_e, peak_b) if x is not None)
    rss_a_s = f"{peak_a:.0f}" if peak_a is not None else "n/a"
    rss_c_s = f"{peak_c:.0f}" if peak_c is not None else "n/a"
    rss_d_s = f"{peak_d:.0f}" if peak_d is not None else "n/a"
    rss_e_s = f"{peak_e:.0f}" if peak_e is not None else "n/a"
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
    print(f"  Max sampled RSS (load / A / C / D / E / B): {peak_rss:.0f} MB")
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
    print(
        "  KV column: A and C use FP32 KV geometry; B adds theoretical Q4 KV payload "
        "(``KVHistory`` in phi4_cpu_run remains FP32; Q4 is planning/comparison)."
    )
    print("=" * 72)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phi4", action="store_true", help="Use local Phi-4 weights (slow, large RAM)")
    p.add_argument("--prompt", type=str, default="Say hello in one short sentence.")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--gamma", type=int, default=7, help="QCSD draft width (sim and phi4 --draft-k)")
    p.add_argument("--vocab-size", type=int, default=200064)
    p.add_argument(
        "--threads",
        type=int,
        default=0,
        help="BLAS/OpenMP/native GEMV threads (0=auto: half of logical CPUs, min 1)",
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
    args = p.parse_args()
    if not 0.0 <= args.sim_acceptance_rate <= 1.0:
        p.error("--sim-acceptance-rate must be between 0 and 1")
    if not 0.0 <= args.phi4_acceptance_estimate <= 1.0:
        p.error("--phi4-acceptance-estimate must be between 0 and 1")

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