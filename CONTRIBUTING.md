# Contributing to ASDSL Framework

Thank you for your interest in contributing to the ASDSL framework!

## Development Setup

```bash
git clone https://github.com/aarushk09/asdsl-framework.git
cd asdsl-framework
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

## Running Benchmarks

```bash
python -m benchmarks.run_all
```

## Code Style

- Format with [Ruff](https://docs.astral.sh/ruff/) (`ruff format .`)
- Lint with `ruff check .`
- Type check with `mypy asdsl/`
- Target Python 3.10+

## Project Structure

```
asdsl/
├── quantization/   # Mixed-precision quantization
│   ├── core.py     # Quantize/dequantize, bit packing
│   ├── salience.py # Salience scoring, bit allocation
│   └── pipeline.py # End-to-end quantization
├── lut/            # LUT-based multiplication
│   ├── engine.py   # Table construction, matvec
│   └── permutation.py # Weight layout optimization
├── speculative/    # SWIFT speculative decoding
│   └── swift.py    # Draft/verify, adaptive schedule
├── prefetch/       # Async L2 cache prefetching
│   └── orchestrator.py # Dual-thread prefetch
├── memory/         # OS memory management
│   └── manager.py  # mlock, huge pages, NUMA
├── kernels/        # SIMD kernel interfaces
│   └── simd.py     # AVX2, AVX-512, NEON emulation
├── inference/      # Inference engine
│   ├── engine.py   # Main orchestrator
│   └── kv_cache.py # Block-sparse KV cache
├── config.py       # Configuration dataclasses
└── cli.py          # Command-line interface
```

## Commit Messages

Use descriptive commit messages following this pattern:
```
<component>: <short description>

<detailed explanation if needed>
```

## Pull Requests

1. Fork the repository
2. Create a feature branch from `master`
3. Write tests for new functionality
4. Ensure all tests pass
5. Submit a PR with a clear description
