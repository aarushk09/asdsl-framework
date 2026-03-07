# Experiments

This directory contains integration tests and real-model validation experiments.

## Phi-4 Multimodal Integration

`phi4_integration.py` — Downloads `microsoft/Phi-4-multimodal-instruct` (~11 GB)
and runs the full ASDSL pipeline (quantization + LUT engine + validation) on all
128 language model backbone projection matrices. Results are written to `results/`.

```bash
python experiments/phi4_integration.py
```

The model weights are downloaded to `models/phi4-multimodal-instruct/` (git-ignored).
Result `.txt` files in `results/` ARE tracked by git.
