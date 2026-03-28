# ASDSL Cross-Model QCSD Benchmark Report

Hardware profile: 11.21 GB/s (microbench_copy_256MB), RAM 15.7 GB, cores 12/16

| Model | Target Size (GB) | Theoretical Limit (tok/s) | Baseline Phase-8 (tok/s) | QCSD (tok/s) | Speedup | QCSD / Theoretical | Acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|
| Phi4 | 8.80 | 1.27 | 5.22 | 16.84 | 3.23x | 13.22x | 65.5% |
| TinyLlama | 4.60 | 2.44 | 9.86 | 29.40 | 2.98x | 12.06x | 68.4% |
| Qwen1p5B | 5.80 | 1.93 | 7.49 | 14.01 | 1.87x | 7.25x | 47.9% |
