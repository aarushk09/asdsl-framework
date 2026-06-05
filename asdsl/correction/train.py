"""Train per-layer bias corrections from collected residuals."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from asdsl.correction.apply_correction import HIDDEN, NUM_LAYERS

# Legacy bias trainer — superseded by train_corrections.py (MLP).
try:
    import numpy as np

    class CorrectionModel:  # type: ignore[no-redef]
        def __init__(self, biases: np.ndarray, hidden_size: int = HIDDEN, num_layers: int = NUM_LAYERS):
            self.biases = biases
            self.hidden_size = hidden_size
            self.num_layers = num_layers

        def save(self, path):  # pragma: no cover
            np.savez_compressed(path, biases=self.biases)
except Exception:
    pass


def train_correction(
    samples_dir: Path | str,
    output_path: Path | str,
    *,
    gain: float = 1.0,
    clip: float = 0.5,
) -> CorrectionModel:
    """Fit per-layer biases as clipped mean residuals × gain."""
    samples_dir = Path(samples_dir)
    npz_path = samples_dir / "correction_samples.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"missing {npz_path}; run collect first")

    data = np.load(npz_path)
    residuals = data["residuals"].astype(np.float32)
    if residuals.shape != (NUM_LAYERS, HIDDEN):
        raise ValueError(f"expected residuals ({NUM_LAYERS}, {HIDDEN}), got {residuals.shape}")

    biases = np.clip(residuals * gain, -clip, clip).astype(np.float32)
    model = CorrectionModel(biases=biases)
    model.save(output_path)

    report = {
        "gain": gain,
        "clip": clip,
        "mean_abs_bias": float(np.mean(np.abs(biases))),
        "max_abs_bias": float(np.max(np.abs(biases))),
    }
    out = Path(output_path)
    out.with_suffix(".train_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return model
