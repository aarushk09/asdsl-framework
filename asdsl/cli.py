"""ASDSL command-line interface."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the ASDSL CLI."""
    parser = argparse.ArgumentParser(
        prog="asdsl",
        description="ASDSL - Asynchronous Salience-Driven Speculative Lookup Framework",
    )
    parser.add_argument("--version", action="version", version="asdsl 0.1.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Quantize command
    quant_parser = subparsers.add_parser("quantize", help="Quantize a model with salience analysis")
    quant_parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    quant_parser.add_argument(
        "--bits", type=int, default=2, choices=[2, 3, 4, 8], help="Default quantization bits"
    )
    quant_parser.add_argument(
        "--salience",
        default="auto",
        choices=["auto", "none", "slim", "tacq"],
        help="Salience analysis method",
    )
    quant_parser.add_argument("--output", "-o", help="Output path for quantized model")
    quant_parser.add_argument(
        "--calibration-samples", type=int, default=128, help="Number of calibration samples"
    )
    quant_parser.add_argument("--group-size", type=int, default=128, help="Quantization group size")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Run inference on a quantized model")
    serve_parser.add_argument("--model", required=True, help="Path to .asdsl quantized model")
    serve_parser.add_argument("--cores", type=int, default=4, help="Number of CPU cores to use")
    serve_parser.add_argument("--prompt", help="Input prompt for generation")
    serve_parser.add_argument("--max-tokens", type=int, default=256, help="Maximum tokens to generate")
    serve_parser.add_argument(
        "--speculative-tokens", type=int, default=4, help="Draft tokens per speculation step"
    )
    serve_parser.add_argument("--no-huge-pages", action="store_true", help="Disable Huge Pages")
    serve_parser.add_argument("--no-pin-memory", action="store_true", help="Disable memory pinning")

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run performance benchmarks")
    bench_parser.add_argument("--model", required=True, help="Path to .asdsl quantized model")
    bench_parser.add_argument("--cores", type=int, default=4, help="Number of CPU cores")
    bench_parser.add_argument("--iterations", type=int, default=10, help="Benchmark iterations")

    # Info command
    subparsers.add_parser("info", help="Display system information and compatibility")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "info":
        return _cmd_info()
    elif args.command == "quantize":
        return _cmd_quantize(args)
    elif args.command == "serve":
        return _cmd_serve(args)
    elif args.command == "benchmark":
        return _cmd_benchmark(args)

    return 0


def _cmd_info() -> int:
    """Display system information."""
    from asdsl.config import get_system_info

    info = get_system_info()
    print("ASDSL Framework - System Information")
    print("=" * 40)
    for key, value in info.items():
        print(f"  {key}: {value}")
    return 0


def _cmd_quantize(args: argparse.Namespace) -> int:
    """Run model quantization."""
    print(f"Quantizing model: {args.model}")
    print(f"  Default bits: {args.bits}")
    print(f"  Salience method: {args.salience}")
    print(f"  Calibration samples: {args.calibration_samples}")
    print(f"  Group size: {args.group_size}")
    # TODO: Integrate with quantization engine
    print("Quantization engine not yet implemented.")
    return 1


def _cmd_serve(args: argparse.Namespace) -> int:
    """Run model inference."""
    print(f"Loading model: {args.model}")
    print(f"  Cores: {args.cores}")
    print(f"  Speculative tokens: {args.speculative_tokens}")
    # TODO: Integrate with inference engine
    print("Inference engine not yet implemented.")
    return 1


def _cmd_benchmark(args: argparse.Namespace) -> int:
    """Run benchmarks."""
    print(f"Benchmarking model: {args.model}")
    print(f"  Cores: {args.cores}")
    print(f"  Iterations: {args.iterations}")
    # TODO: Integrate with benchmark suite
    print("Benchmark suite not yet implemented.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
