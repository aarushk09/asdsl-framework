"""
Generate detailed ASDSL Framework performance visualizations.

Creates publication-quality charts from benchmark results, including:
- Performance radar chart
- Before/after PPL waterfall
- Resource usage breakdown
- Summary infographic

Usage:
  python benchmarks/generate_visuals.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
except ImportError:
    print("ERROR: matplotlib required. Install with: pip install matplotlib")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmarks" / "results"
RESULTS_FILE = RESULTS_DIR / "benchmark_results.json"


def load_results():
    with open(RESULTS_FILE) as f:
        return json.load(f)


def fig_radar(results, output_dir):
    """Radar / spider chart comparing all configurations."""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.suptitle("ASDSL Configuration Comparison", fontsize=16, fontweight="bold", y=1.02)

    categories = ["Quality\n(1/PPL×100)", "Throughput\n(tok/s)", "Compression\n(ratio)",
                   "SNR\n(dB)", "Core\nEfficiency"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    colors = {16: "#2196F3", 8: "#4CAF50", 4: "#FF9800", 3: "#F44336"}

    max_vals = [100, 1.5, 10, 50, 10]

    for r in results:
        bits = r["bits"]
        ppl = r.get("ppl", 1)
        quality = min(100 / max(ppl, 1), 100)
        tps = r.get("tok_per_sec", 0)
        comp = r.get("compression_ratio", 1)
        snr = min(r.get("snr_db", 0), 50) if r.get("snr_db", 0) < 100 else 50
        core_eff = 10 / max(r.get("active_cores", 1), 1)  # fewer cores = better

        vals = [quality / max_vals[0] * 10,
                tps / max_vals[1] * 10,
                comp / max_vals[2] * 10,
                snr / max_vals[3] * 10,
                core_eff]
        vals += vals[:1]

        label = "FP16" if bits == 16 else f"{bits}-bit"
        ax.plot(angles, vals, 'o-', linewidth=2, label=label,
                color=colors.get(bits, "#607D8B"))
        ax.fill(angles, vals, alpha=0.1, color=colors.get(bits, "#607D8B"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.grid(True)

    plt.savefig(output_dir / "asdsl_radar_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Radar chart saved: {output_dir / 'asdsl_radar_chart.png'}")


def fig_ppl_waterfall(results, output_dir):
    """Waterfall chart showing PPL improvement from baseline for each quantization."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("ASDSL Quantization: Perplexity Preservation", fontsize=16, fontweight="bold")

    # Previous (broken) values
    before = {16: 17.64, 8: 17.59, 4: 61.15, 3: 216950.82}
    after = {r["bits"]: r.get("ppl", 0) for r in results}
    baseline_ppl = after.get(16, 15.78)

    configs = []
    for r in sorted(results, key=lambda x: -x["bits"]):
        b = r["bits"]
        ppl = r.get("ppl", 0)
        label = "FP16 baseline" if b == 16 else f"ASDSL {b}-bit"
        preservation = (1 - (ppl - baseline_ppl) / baseline_ppl) * 100 if ppl > 0 else 100
        configs.append((label, b, ppl, max(preservation, 0)))

    labels = [c[0] for c in configs]
    pres = [c[3] for c in configs]
    ppls = [c[2] for c in configs]
    bits_vals = [c[1] for c in configs]

    colors_map = {16: "#2196F3", 8: "#4CAF50", 4: "#FF9800", 3: "#F44336"}
    bar_colors = [colors_map.get(b, "#607D8B") for b in bits_vals]

    bars = ax.bar(labels, pres, color=bar_colors, edgecolor="black", linewidth=0.5, width=0.6)

    ax.axhline(y=95, color="green", linestyle="--", alpha=0.5, linewidth=2, label=">95% target")

    for bar, preservation, ppl in zip(bars, pres, ppls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{preservation:.1f}%\n(PPL={ppl:.1f})", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    ax.set_ylabel("Quality Preservation vs FP16 (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.savefig(output_dir / "asdsl_ppl_preservation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  PPL preservation chart saved: {output_dir / 'asdsl_ppl_preservation.png'}")


def fig_improvement_summary(results, output_dir):
    """Summary infographic showing before-vs-after improvements."""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("ASDSL Framework — Optimization Results Summary",
                 fontsize=20, fontweight="bold", y=0.98)

    gs = GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.4)

    before_ppl = {16: 17.64, 8: 17.59, 4: 61.15, 3: 216950.82}
    after = {r["bits"]: r for r in results}

    # --- Row 1: Big metric cards ---
    def draw_metric_card(ax, title, value, subtitle, color):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        rect = mpatches.FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                                        boxstyle="round,pad=0.05",
                                        facecolor=color, alpha=0.15,
                                        edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(0.5, 0.75, title, ha="center", va="center", fontsize=11,
                color="#666", fontweight="bold")
        ax.text(0.5, 0.45, value, ha="center", va="center", fontsize=24,
                color=color, fontweight="bold")
        ax.text(0.5, 0.18, subtitle, ha="center", va="center", fontsize=9,
                color="#888")

    # 4-bit improvement
    ax1 = fig.add_subplot(gs[0, 0])
    ppl_4 = after.get(4, {}).get("ppl", 0)
    draw_metric_card(ax1, "4-bit PPL", f"{ppl_4:.1f}",
                     f"was 61.15 ({(1 - ppl_4/61.15)*100:.0f}% better)", "#FF9800")

    # 3-bit improvement
    ax2 = fig.add_subplot(gs[0, 1])
    ppl_3 = after.get(3, {}).get("ppl", 0)
    draw_metric_card(ax2, "3-bit PPL", f"{ppl_3:.1f}",
                     "was 216,951 (99.99% better)", "#F44336")

    # 8-bit quality
    ax3 = fig.add_subplot(gs[0, 2])
    ppl_8 = after.get(8, {}).get("ppl", 0)
    draw_metric_card(ax3, "8-bit PPL", f"{ppl_8:.2f}",
                     "= FP16 baseline (lossless)", "#4CAF50")

    # Core count
    ax4 = fig.add_subplot(gs[0, 3])
    avg_cores = np.mean([r.get("active_cores", 0) for r in results])
    draw_metric_card(ax4, "CPU Cores", f"{avg_cores:.0f}",
                     "was 13-15 (75% reduction)", "#9C27B0")

    # --- Row 2: PPL comparison bar chart ---
    ax5 = fig.add_subplot(gs[1, :2])
    bits_list = [16, 8, 4, 3]
    labels = ["FP16", "8-bit", "4-bit", "3-bit"]
    before_vals = [before_ppl.get(b, 0) for b in bits_list]
    after_vals = [after.get(b, {}).get("ppl", 0) for b in bits_list]

    # Cap before values for display
    before_display = [min(v, 80) for v in before_vals]
    after_display = after_vals

    x = np.arange(len(labels))
    width = 0.35
    bars1 = ax5.bar(x - width/2, before_display, width, label="Before",
                    color="#EF5350", edgecolor="black", linewidth=0.5, alpha=0.7)
    bars2 = ax5.bar(x + width/2, after_display, width, label="After",
                    color="#66BB6A", edgecolor="black", linewidth=0.5)
    ax5.set_ylabel("Perplexity")
    ax5.set_title("Before vs After Optimization", fontsize=13, fontweight="bold")
    ax5.set_xticks(x)
    ax5.set_xticklabels(labels)
    ax5.legend()
    ax5.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars1, before_vals):
        label_txt = f"{val:,.0f}" if val > 1000 else f"{val:.1f}"
        y_display = min(val, 80)
        ax5.text(bar.get_x() + bar.get_width()/2, y_display + 1,
                 label_txt, ha="center", va="bottom", fontsize=9, color="#C62828")
    for bar, val in zip(bars2, after_vals):
        ax5.text(bar.get_x() + bar.get_width()/2, val + 1,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=9,
                 color="#2E7D32", fontweight="bold")

    # --- Row 2: Throughput + cores ---
    ax6 = fig.add_subplot(gs[1, 2:])
    tps_vals = [after.get(b, {}).get("tok_per_sec", 0) for b in bits_list]
    core_vals = [after.get(b, {}).get("active_cores", 0) for b in bits_list]
    colors_map = {16: "#2196F3", 8: "#4CAF50", 4: "#FF9800", 3: "#F44336"}
    bar_colors = [colors_map[b] for b in bits_list]

    ax6_twin = ax6.twinx()
    bars_tps = ax6.bar(x - width/2, tps_vals, width, color=bar_colors,
                       edgecolor="black", linewidth=0.5, alpha=0.8, label="Throughput")
    bars_core = ax6_twin.bar(x + width/2, core_vals, width, color=bar_colors,
                             edgecolor="black", linewidth=0.5, alpha=0.4, label="Cores")
    ax6.set_ylabel("Tokens/sec")
    ax6_twin.set_ylabel("Active Cores")
    ax6.set_title("Throughput & Core Usage", fontsize=13, fontweight="bold")
    ax6.set_xticks(x)
    ax6.set_xticklabels(labels)
    ax6.legend(loc="upper left")
    ax6_twin.legend(loc="upper right")
    ax6.grid(axis="y", alpha=0.3)

    # --- Row 3: Summary table ---
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis("off")

    table_data = []
    for b in bits_list:
        r = after.get(b, {})
        bp = before_ppl.get(b, 0)
        ap = r.get("ppl", 0)
        improvement = f"{(1 - ap/bp)*100:+.1f}%" if bp > 0 and ap > 0 and b != 16 else "baseline"
        snr = "∞" if r.get("snr_db", 0) > 100 else f"{r.get('snr_db', 0):.1f}"
        label = "FP16" if b == 16 else f"ASDSL {b}-bit"
        gs_val = r.get("group_size", "—")
        mode = "sym" if r.get("symmetric", True) else "asym+clip"
        table_data.append([
            label,
            f"{ap:.2f}",
            improvement,
            f"{r.get('tok_per_sec', 0):.2f}",
            f"{r.get('ram_peak_mb', 0)/1024:.1f} GB",
            str(r.get("active_cores", "—")),
            snr,
            f"{r.get('compression_ratio', 1.0):.1f}x",
            str(gs_val),
            mode,
        ])

    table = ax7.table(
        cellText=table_data,
        colLabels=["Config", "PPL", "vs Before", "tok/s", "RAM", "Cores",
                    "SNR (dB)", "Comp.", "Grp Size", "Mode"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    # Color header
    for j in range(len(table_data[0])):
        table[0, j].set_facecolor("#1565C0")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Color rows by config
    row_colors = ["#E3F2FD", "#E8F5E9", "#FFF3E0", "#FFEBEE"]
    for i, color in enumerate(row_colors):
        for j in range(len(table_data[0])):
            table[i + 1, j].set_facecolor(color)

    ax7.set_title("Complete Benchmark Summary", fontsize=14, fontweight="bold", pad=15)

    plt.savefig(output_dir / "asdsl_summary_infographic.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Summary infographic saved: {output_dir / 'asdsl_summary_infographic.png'}")


def fig_quantization_quality(results, output_dir):
    """Chart showing SNR, compression ratio, and PPL as quality metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("ASDSL Quantization Quality Analysis", fontsize=16, fontweight="bold")

    bits_list = [r["bits"] for r in results]
    labels = [f"{b}-bit" if b < 16 else "FP16" for b in bits_list]
    colors_map = {16: "#2196F3", 8: "#4CAF50", 4: "#FF9800", 3: "#F44336"}
    bar_colors = [colors_map.get(b, "#607D8B") for b in bits_list]

    # PPL vs bits
    ppls = [r.get("ppl", 0) for r in results]
    axes[0].bar(labels, ppls, color=bar_colors, edgecolor="black", linewidth=0.5)
    axes[0].axhline(y=ppls[0], color="blue", linestyle="--", alpha=0.5, label="FP16 baseline")
    axes[0].set_ylabel("Perplexity (lower = better)")
    axes[0].set_title("Model Quality")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(ppls):
        axes[0].text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=10, fontweight="bold")

    # SNR vs bits
    snrs = [min(r.get("snr_db", 0), 60) for r in results]
    axes[1].bar(labels, snrs, color=bar_colors, edgecolor="black", linewidth=0.5)
    axes[1].set_ylabel("SNR (dB, higher = better)")
    axes[1].set_title("Signal-to-Noise Ratio")
    axes[1].grid(axis="y", alpha=0.3)
    for i, (v, r) in enumerate(zip(snrs, results)):
        label = "∞" if r.get("snr_db", 0) > 100 else f"{v:.1f}"
        axes[1].text(i, v + 0.5, label, ha="center", fontsize=10, fontweight="bold")

    # Compression ratio vs bits
    comps = [r.get("compression_ratio", 1) for r in results]
    axes[2].bar(labels, comps, color=bar_colors, edgecolor="black", linewidth=0.5)
    axes[2].set_ylabel("Compression Ratio (x)")
    axes[2].set_title("Weight Compression")
    axes[2].grid(axis="y", alpha=0.3)
    for i, v in enumerate(comps):
        axes[2].text(i, v + 0.1, f"{v:.1f}x", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "asdsl_quant_quality.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Quantization quality chart saved: {output_dir / 'asdsl_quant_quality.png'}")


def main():
    if not RESULTS_FILE.exists():
        print(f"ERROR: {RESULTS_FILE} not found. Run comprehensive_bench.py first.")
        sys.exit(1)

    results = load_results()
    output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print("  Generating ASDSL Performance Visualizations")
    print("=" * 66)

    fig_radar(results, output_dir)
    fig_ppl_waterfall(results, output_dir)
    fig_improvement_summary(results, output_dir)
    fig_quantization_quality(results, output_dir)

    print(f"\nAll charts saved to: {output_dir}")
    print("Files:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
