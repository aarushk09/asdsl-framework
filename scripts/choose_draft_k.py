#!/usr/bin/env python3
"""Leviathan speedup S and draft_k selection from measured MTP test_top1 (proxy alpha)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def leviathan_S(alpha: float, k: int, c: float = 0.221) -> float:
    """S = (1 - alpha^(k+1)) / ((1 - alpha) * (c * k + 1))."""
    if alpha <= 0.0:
        return 1.0 / (c * k + 1)
    if alpha >= 1.0:
        return float("inf")
    num = 1.0 - (alpha ** (k + 1))
    den = (1.0 - alpha) * (c * k + 1.0)
    return num / den


def alpha_break_even(k: int, c: float = 0.221, lo: float = 1e-4, hi: float = 0.9999) -> float:
    """Smallest alpha in (lo, hi) with S(alpha, k, c) >= 1.0 (bisection)."""
    if leviathan_S(hi, k, c) < 1.0:
        return float("nan")
    # Expand hi if needed
    a, b = lo, hi
    for _ in range(60):
        mid = 0.5 * (a + b)
        if leviathan_S(mid, k, c) >= 1.0:
            b = mid
        else:
            a = mid
    return b


def choose_draft_k(test_top1_pct: float, c: float = 0.221) -> tuple[int, float]:
    alpha = max(0.0, min(1.0, test_top1_pct / 100.0))
    best_k, best_s = 1, 0.0
    for k in range(1, 8):
        s = leviathan_S(alpha, k, c)
        print(f"  k={k}: S={s:.3f} (alpha={alpha:.3f}, c={c:.3f})")
        if s > best_s:
            best_s, best_k = s, k
    print(f"  -> Optimal draft_k={best_k} (S={best_s:.3f}x)")
    return best_k, best_s


def print_break_even_table(
    c: float = 0.221,
    current_acceptance_pct: float = 8.9,
) -> dict[str, float]:
    """Print break-even alpha table for k in 1,2,3,7."""
    ks = [1, 2, 3, 7]
    alphas: dict[str, float] = {}
    cur = current_acceptance_pct / 100.0
    print(f"\nBreak-even acceptance rates (c={c}, Raptor Lake measured):")
    for k in ks:
        a = alpha_break_even(k, c)
        key = f"k{k}"
        alphas[key] = a * 100.0
        need_pp = (a - cur) * 100.0
        print(
            f"draft_k={k}: alpha_min = {a*100:.1f}%  "
            f"(current acceptance {current_acceptance_pct:.1f}% - need {need_pp:+.1f}pp)"
        )
    return alphas


def main() -> None:
    ap = argparse.ArgumentParser(description="Choose EAGLE-3 draft_k from test_top1 proxy")
    ap.add_argument(
        "--test-top1",
        type=float,
        required=True,
        metavar="PCT",
        help="MTP test_top1 accuracy (percent), used as proxy for acceptance alpha",
    )
    ap.add_argument("--c", type=float, default=0.221, help="Leviathan draft cost ratio c")
    ap.add_argument(
        "--current-acceptance",
        type=float,
        default=8.9,
        help="Phase-10 style acceptance %% for gap lines in break-even table",
    )
    ap.add_argument(
        "--update-config",
        type=str,
        default=None,
        metavar="PATH",
        help="If set, write chosen draft_k into benchmark_config.json at PATH",
    )
    args = ap.parse_args()

    print(f"\nLeviathan S vs k (test_top1={args.test_top1:.2f}% as alpha proxy, c={args.c}):")
    best_k, best_s = choose_draft_k(args.test_top1, c=args.c)
    print_break_even_table(c=args.c, current_acceptance_pct=args.current_acceptance)

    alpha = max(0.0, min(1.0, args.test_top1 / 100.0))
    print(f"\nAt current MTP test_top1 = {args.test_top1:.1f}% (proxy for acceptance):")
    print(
        f"  Optimal draft_k = {best_k}  (speedup S = {best_s:.3f}x at alpha = {alpha:.2f})"
    )

    chosen_k = best_k
    if best_s < 1.0:
        print(
            "  Note: best S < 1.0 - no k yields net speedup at this alpha; "
            "using draft_k=2 minimum in config until acceptance improves."
        )
        chosen_k = 2

    if args.update_config:
        p = Path(args.update_config)
        if not p.is_file():
            print(f"[choose_draft_k] config not found: {p}", file=sys.stderr)
            sys.exit(1)
        cfg = json.loads(p.read_text(encoding="utf-8"))
        cfg["draft_k"] = int(chosen_k)
        p.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        print(f"[choose_draft_k] Updated {p} draft_k -> {chosen_k}")


if __name__ == "__main__":
    main()
