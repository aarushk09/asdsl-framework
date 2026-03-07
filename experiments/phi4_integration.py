"""
Phi-4 Multimodal Instruct Integration Test for ASDSL Framework
==============================================================
Downloads real Phi-4 Multimodal Instruct (microsoft/Phi-4-multimodal-instruct)
weights and runs the full ASDSL quantization + LUT pipeline on them,
reporting compression ratios, reconstruction errors, throughput, and test pass/fail.

Usage:
    python experiments/phi4_integration.py

The 11 GB model weights are cached in models/phi4-multimodal-instruct/ (git-ignored).
Results are written to experiments/results/.
"""

import sys
import os
import time
import json
import gc
import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download

from asdsl.quantization.core import quantize_weights, dequantize_weights, compute_quantization_error
from asdsl.quantization.salience import compute_gradient_salience, allocate_bits_by_salience
from asdsl.lut.engine import build_lut_tables_for_layer, lut_matvec, estimate_lut_memory
from asdsl.speculative.swift import create_skip_schedule_for_phi3
from asdsl.kernels.simd import select_backend

MODEL_ID = "microsoft/Phi-4-multimodal-instruct"
MODEL_DIR = ROOT / "models" / "phi4-multimodal-instruct"
RESULTS_DIR = ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GROUP_SIZE = 128
BITS_TO_TEST = [4, 3, 2]


# ──────────────────────────────────────────────────────────────────────────────
# Download helpers
# ──────────────────────────────────────────────────────────────────────────────

def download_model_shards() -> list[Path]:
    """Download the 3 language-model safetensors shards (≈11 GB total)."""
    print(f"Downloading {MODEL_ID} → {MODEL_DIR}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    shard_names = [
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    ]
    paths = []
    for shard in shard_names:
        dest = MODEL_DIR / shard
        if dest.exists():
            print(f"  {shard}: already cached ({dest.stat().st_size / 1e9:.2f} GB)")
            paths.append(dest)
        else:
            print(f"  Downloading {shard}…")
            t0 = time.perf_counter()
            hf_hub_download(
                repo_id=MODEL_ID,
                filename=shard,
                local_dir=str(MODEL_DIR),
            )
            elapsed = time.perf_counter() - t0
            size_gb = dest.stat().st_size / 1e9
            print(f"  {shard}: done ({size_gb:.2f} GB in {elapsed:.0f}s)")
            paths.append(dest)

    return paths


def load_index() -> dict[str, str]:
    """Return weight_key → shard_filename mapping."""
    idx_path = MODEL_DIR / "model.safetensors.index.json"
    if not idx_path.exists():
        hf_hub_download(
            repo_id=MODEL_ID,
            filename="model.safetensors.index.json",
            local_dir=str(MODEL_DIR),
        )
    return json.load(open(idx_path))["weight_map"]


# ──────────────────────────────────────────────────────────────────────────────
# Layer-by-layer processing
# ──────────────────────────────────────────────────────────────────────────────

BACKBONE_PROJ_TYPES = {
    "qkv_proj": "attention",
    "o_proj":   "attention",
    "gate_up_proj": "ffn",
    "down_proj":    "ffn",
}


def get_backbone_keys(weight_map: dict[str, str]) -> list[str]:
    """Return sorted list of LM backbone projection weight keys."""
    return sorted(
        k for k in weight_map
        if k.startswith("model.layers.")
        and "base_layer.weight" in k
        and any(p in k for p in BACKBONE_PROJ_TYPES)
    )


class PerLayerStats:
    def __init__(self, key: str, shape: tuple):
        self.key = key
        self.shape = shape
        self.fp16_size_mb = shape[0] * shape[1] * 2 / 1e6
        self.quant_results: dict[int, dict] = {}  # bits → metrics

    def add_quant(self, bits: int, metrics: dict, lut_metrics: dict):
        self.quant_results[bits] = {**metrics, **lut_metrics}


def process_tensor(
    key: str,
    shard_path: Path,
    stats_list: list[PerLayerStats],
    log: list[str],
) -> None:
    """Quantize one backbone weight tensor and append stats."""
    from safetensors import safe_open

    with safe_open(str(shard_path), framework="np") as f:
        w_f16 = f.get_tensor(key)   # float16

    # Convert to float32 for ASDSL processing
    w_f32 = w_f16.astype(np.float32)
    rows, cols = w_f32.shape
    stat = PerLayerStats(key, (rows, cols))

    # Random input vector for matvec comparison
    rng = np.random.default_rng(seed=42)
    x = rng.standard_normal(cols).astype(np.float32)
    # Reference output in fp16 precision
    y_ref = (w_f16.astype(np.float64) @ x.astype(np.float64)).astype(np.float32)

    for bits in BITS_TO_TEST:
        # ── Quantization ──────────────────────────────────────────────────────
        t0 = time.perf_counter()
        qt = quantize_weights(w_f32, bits=bits, group_size=GROUP_SIZE)
        quant_time = time.perf_counter() - t0

        errors = compute_quantization_error(w_f32, qt)

        # ── LUT build ─────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        tables = build_lut_tables_for_layer(qt)
        lut_build_time = time.perf_counter() - t0

        # ── LUT matvec ────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        y_lut = lut_matvec(tables, x, out_features=rows, group_size=GROUP_SIZE)
        lut_matvec_time = time.perf_counter() - t0

        # ── Float32 reference matvec ──────────────────────────────────────────
        t0 = time.perf_counter()
        y_fp32 = w_f32 @ x
        fp32_time = time.perf_counter() - t0

        # ── Error vs FP16 reference ────────────────────────────────────────────
        mse_vs_ref = float(np.mean((y_lut - y_ref) ** 2))
        cosine = float(
            np.dot(y_lut, y_ref) / (np.linalg.norm(y_lut) * np.linalg.norm(y_ref) + 1e-10)
        )
        mem_est = estimate_lut_memory(rows, cols, bits, GROUP_SIZE)

        stat.add_quant(bits, errors, {
            "quant_time_ms": quant_time * 1000,
            "lut_build_time_ms": lut_build_time * 1000,
            "lut_matvec_time_ms": lut_matvec_time * 1000,
            "fp32_matvec_time_ms": fp32_time * 1000,
            "mse_vs_fp16_ref": mse_vs_ref,
            "cosine_vs_fp16_ref": cosine,
            "lut_memory_kb": mem_est["total_bytes"] / 1024,
            "lut_fits_l2": mem_est["fits_l2"],
        })

        msg = (
            f"  [{bits}b] ratio={errors['compression_ratio']:.1f}x "
            f"SNR={errors['snr_db']:.1f}dB "
            f"cosine={cosine:.5f} "
            f"lut_build={lut_build_time*1000:.0f}ms "
            f"lut_mv={lut_matvec_time*1000:.0f}ms"
        )
        print(msg)
        log.append(msg)

        del qt, tables
        gc.collect()

    del w_f16, w_f32, y_ref, y_fp32, y_lut
    gc.collect()
    stats_list.append(stat)


# ──────────────────────────────────────────────────────────────────────────────
# Existing unit tests runner
# ──────────────────────────────────────────────────────────────────────────────

def run_unit_tests(log: list[str]) -> tuple[int, int]:
    """Run the ASDSL unit tests via pytest and return (passed, failed)."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "-q"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    log.append("\n=== Unit Test Output ===")
    log.append(output[-3000:] if len(output) > 3000 else output)

    # Parse passed/failed
    passed = failed = 0
    for line in output.splitlines():
        if " passed" in line:
            try:
                passed = int(line.strip().split()[0])
            except Exception:
                pass
        if " failed" in line:
            try:
                failed = int(line.strip().split()[0])
            except Exception:
                pass
    return passed, failed


# ──────────────────────────────────────────────────────────────────────────────
# Summary report
# ──────────────────────────────────────────────────────────────────────────────

def write_report(
    stats_list: list[PerLayerStats],
    unit_passed: int,
    unit_failed: int,
    log: list[str],
    elapsed_total: float,
) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = RESULTS_DIR / f"phi4_integration_{ts}.txt"

    lines = []
    lines.append("=" * 80)
    lines.append("  ASDSL Framework — Phi-4 Multimodal Instruct Integration Results")
    lines.append(f"  Model:  {MODEL_ID}")
    lines.append(f"  Date:   {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Runtime: {elapsed_total/60:.1f} minutes")
    lines.append("=" * 80)
    lines.append("")

    # Unit tests
    lines.append(f"Unit Tests: {unit_passed} passed, {unit_failed} failed")
    lines.append("")

    # Per-layer summary table
    for bits in BITS_TO_TEST:
        lines.append(f"── {bits}-bit quantization ──────────────────────────────────────────")
        hdr = f"{'Layer':<55s} {'Ratio':>6s} {'SNR(dB)':>8s} {'Cosine':>8s} {'LUT-L2':>7s} {'LUT-mv(ms)':>10s}"
        lines.append(hdr)
        lines.append("-" * len(hdr))

        snr_vals, cos_vals, ratio_vals = [], [], []
        for s in stats_list:
            if bits not in s.quant_results:
                continue
            r = s.quant_results[bits]
            # Shorten key for display
            short = s.key.replace("model.layers.", "L").replace(".base_layer.weight", "").replace(".self_attn.", ".").replace(".mlp.", ".")
            lines.append(
                f"{short:<55s} {r['compression_ratio']:>6.1f}x "
                f"{r['snr_db']:>8.1f} "
                f"{r['cosine_vs_fp16_ref']:>8.5f} "
                f"{'Yes' if r['lut_fits_l2'] else 'No ':>7s} "
                f"{r['lut_matvec_time_ms']:>10.2f}"
            )
            snr_vals.append(r["snr_db"])
            cos_vals.append(r["cosine_vs_fp16_ref"])
            ratio_vals.append(r["compression_ratio"])

        if snr_vals:
            lines.append(
                f"\n  {'AVERAGE':>55s} {np.mean(ratio_vals):>6.1f}x "
                f"{np.mean(snr_vals):>8.1f} "
                f"{np.mean(cos_vals):>8.5f}"
            )
        lines.append("")

    # Aggregate summary
    lines.append("=" * 80)
    lines.append("AGGREGATE SUMMARY")
    lines.append("=" * 80)
    total_fp16_mb = sum(s.fp16_size_mb for s in stats_list)
    lines.append(f"Language model projection layers analysed: {len(stats_list)}")
    lines.append(f"Total FP16 weight size: {total_fp16_mb/1024:.2f} GB")
    for bits in BITS_TO_TEST:
        layer_results = [s.quant_results[bits] for s in stats_list if bits in s.quant_results]
        if not layer_results:
            continue
        avg_ratio = np.mean([r["compression_ratio"] for r in layer_results])
        compressed_mb = total_fp16_mb / avg_ratio
        avg_snr = np.mean([r["snr_db"] for r in layer_results])
        avg_cos = np.mean([r["cosine_vs_fp16_ref"] for r in layer_results])
        total_lut_build = sum(r["lut_build_time_ms"] for r in layer_results)
        lines.append(
            f"\n  {bits}-bit mixed precision:"
            f"\n    Compression ratio  : {avg_ratio:.2f}x"
            f"\n    Compressed size    : {compressed_mb/1024:.2f} GB"
            f"\n    Average SNR        : {avg_snr:.1f} dB"
            f"\n    Average cosine sim : {avg_cos:.6f}"
            f"\n    Total LUT build    : {total_lut_build/1000:.1f}s"
        )

    lines.append("\n")
    lines.extend(log)

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.perf_counter()
    log: list[str] = []

    print("=" * 72)
    print(f"ASDSL × Phi-4 Multimodal Instruct Integration Test")
    print("=" * 72)
    print()

    # ── System info ──────────────────────────────────────────────────────────
    backend = select_backend()
    schedule = create_skip_schedule_for_phi3()
    print(f"SIMD backend   : {backend.name}")
    print(f"Python         : {sys.version.split()[0]}")
    print(f"NumPy          : {np.__version__}")
    print(f"SWIFT schedule : {len(schedule.draft_layers)} draft layers, "
          f"{schedule.skip_ratio:.0%} skip ratio")
    print()
    log.append(f"Backend: {backend.name}")

    # ── Download ──────────────────────────────────────────────────────────────
    print("Step 1: Downloading model shards…")
    shard_paths = download_model_shards()
    weight_map = load_index()
    print()

    # ── Identify backbone keys ─────────────────────────────────────────────────
    backbone_keys = get_backbone_keys(weight_map)
    print(f"Step 2: Found {len(backbone_keys)} backbone projection weight matrices")
    print(f"  Processing all 32 layers × 4 projection types")
    print()

    # ── Process tensors layer by layer ───────────────────────────────────────
    print("Step 3: Quantize, LUT-build, and validate each weight matrix…")
    stats_list: list[PerLayerStats] = []
    shard_path_map = {
        Path(v).name: (ROOT / "models" / "phi4-multimodal-instruct" / Path(v).name)
        for v in set(weight_map.values())
        if v.startswith("model-") and v.endswith(".safetensors")
    }

    for i, key in enumerate(backbone_keys):
        shard_name = Path(weight_map[key]).name
        shard_path = MODEL_DIR / shard_name

        # Infer shape from key for display
        proj_type = next(p for p in BACKBONE_PROJ_TYPES if p in key)
        layer_num = key.split(".")[2]
        print(f"  [{i+1:03d}/{len(backbone_keys)}] layer={layer_num} {proj_type}")
        log.append(f"\nLayer {layer_num} {proj_type}:")

        try:
            process_tensor(key, shard_path, stats_list, log)
        except Exception as e:
            msg = f"    ERROR: {e}"
            print(msg)
            log.append(msg)

    print()

    # ── Unit tests ───────────────────────────────────────────────────────────
    print("Step 4: Running unit test suite…")
    passed, failed = run_unit_tests(log)
    status = "PASS" if failed == 0 else "FAIL"
    print(f"  Unit tests: {passed} passed, {failed} failed  [{status}]")
    print()

    # ── Final report ─────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    print("Step 5: Writing results report…")
    report_path = write_report(stats_list, passed, failed, log, elapsed)
    print(f"  Report saved: {report_path.relative_to(ROOT)}")
    print()

    # ── Quick console summary ─────────────────────────────────────────────────
    print("=" * 72)
    print("QUICK SUMMARY")
    print("=" * 72)
    total_fp16_gb = sum(s.fp16_size_mb for s in stats_list) / 1024
    print(f"Layers processed  : {len(stats_list)}")
    print(f"FP16 weight total : {total_fp16_gb:.2f} GB")
    for bits in BITS_TO_TEST:
        results = [s.quant_results[bits] for s in stats_list if bits in s.quant_results]
        if not results:
            continue
        avg_ratio = np.mean([r["compression_ratio"] for r in results])
        avg_snr = np.mean([r["snr_db"] for r in results])
        print(f"{bits}-bit avg ratio   : {avg_ratio:.1f}x → {total_fp16_gb/avg_ratio:.2f} GB  SNR={avg_snr:.1f} dB")
    print(f"Unit tests        : {passed} passed, {failed} failed")
    print(f"Total runtime     : {elapsed/60:.1f} minutes")
    print()

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
