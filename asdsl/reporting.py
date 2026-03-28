"""Reporting utilities for cross-model benchmark outputs."""

from __future__ import annotations

from pathlib import Path


def write_cross_model_report(report_path: Path, rows: list[dict], profiler_summary: str) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# ASDSL Cross-Model QCSD Benchmark Report",
        "",
        f"Hardware profile: {profiler_summary}",
        "",
        "| Model | Target Size (GB) | Theoretical Limit (tok/s) | Baseline Phase-8 (tok/s) | QCSD (tok/s) | Speedup | QCSD / Theoretical | Acceptance |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in rows:
        lines.append(
            f"| {r['model']} | {r['target_model_size_gb']:.2f} | {r['theoretical_limit_tps']:.2f} | "
            f"{r['baseline_tps']:.2f} | {r['qcsd_tps']:.2f} | {r['speedup']:.2f}x | "
            f"{r['qcsd_vs_theoretical']:.2f}x | {r['acceptance_rate']:.1%} |"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_cross_model_chart(image_path: Path, rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    image_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [r["model"] for r in rows]
    theoretical = np.array([r["theoretical_limit_tps"] for r in rows], dtype=float)
    baseline = np.array([r["baseline_tps"] for r in rows], dtype=float)
    qcsd = np.array([r["qcsd_tps"] for r in rows], dtype=float)

    x = np.arange(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width, theoretical, width, label="Theoretical AR Limit", color="#607D8B")
    ax.bar(x, baseline, width, label="Baseline Phase-8", color="#FF9800")
    ax.bar(x + width, qcsd, width, label="ASDSL QCSD", color="#2E7D32")

    ax.set_title("Cross-Model Throughput: Theoretical vs Baseline vs QCSD")
    ax.set_ylabel("Tokens / second")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    for i, val in enumerate(qcsd):
        ax.text(x[i] + width, val + max(qcsd) * 0.01, f"{val:.1f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(image_path, dpi=160)
    plt.close(fig)
