"""Run all ASDSL benchmarks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.bench_quantization import main as bench_quant
from benchmarks.bench_lut import main as bench_lut
from benchmarks.bench_inference import main as bench_inference


def main() -> None:
    print()
    print("*" * 72)
    print("  ASDSL Framework — Complete Benchmark Suite")
    print("*" * 72)
    print()

    print("\n>>> Quantization Benchmarks <<<\n")
    bench_quant()

    print("\n>>> LUT Engine Benchmarks <<<\n")
    bench_lut()

    print("\n>>> Inference Benchmarks <<<\n")
    bench_inference()

    print()
    print("*" * 72)
    print("  All benchmarks complete.")
    print("*" * 72)


if __name__ == "__main__":
    main()
