"""Apply per-layer MLP residual corrections at transformer block output."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

HIDDEN = 3072
NUM_LAYERS = 32


if torch is not None:

    class CorrectionMLP(nn.Module):
        """3072 -> 64 -> 64 -> 3072 residual predictor."""

        def __init__(self, hidden_dim: int = 64) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(HIDDEN, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, HIDDEN),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


@dataclass
class CorrectionModel:
    """Eager-loaded per-layer MLP correctors (~10 KB per layer)."""

    models_dir: Path
    manifest: dict
    hidden_dim: int = 64
    _mlps: list | None = None

    @property
    def num_layers(self) -> int:
        return int(self.manifest.get("num_layers", NUM_LAYERS))

    def _ensure_loaded(self) -> None:
        if self._mlps is not None or torch is None:
            return
        layers: list = []
        for layer_idx in range(self.num_layers):
            path = self.models_dir / f"layer_{layer_idx}_correction.pt"
            if not path.is_file():
                raise FileNotFoundError(f"missing correction MLP: {path}")
            mlp = CorrectionMLP(hidden_dim=self.hidden_dim)
            mlp.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            mlp.eval()
            layers.append(mlp)
        self._mlps = layers

    def predict_delta(self, hidden_np: np.ndarray, layer_idx: int) -> np.ndarray:
        """Return predicted residual delta for one hidden vector."""
        self._ensure_loaded()
        assert self._mlps is not None
        h = np.asarray(hidden_np, dtype=np.float32).ravel()
        if h.size != HIDDEN:
            raise ValueError(f"hidden size {h.size} != {HIDDEN}")
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"layer_idx {layer_idx} out of range")
        with torch.no_grad():
            x = torch.from_numpy(h).unsqueeze(0)
            delta = self._mlps[layer_idx](x).squeeze(0).numpy()
        return delta.astype(np.float32)

    @classmethod
    def load(cls, models_dir: Path | str) -> CorrectionModel:
        models_dir = Path(models_dir)
        manifest_path = models_dir / "correction_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return cls(
            models_dir=models_dir,
            manifest=manifest,
            hidden_dim=int(manifest.get("hidden_dim", 64)),
        )


def load_correction(path: Path | str | None) -> CorrectionModel | None:
    if path is None:
        return None
    p = Path(path)
    if p.is_dir():
        return CorrectionModel.load(p)
    if p.suffix == ".json" and p.parent.is_dir():
        return CorrectionModel.load(p.parent)
    if p.with_suffix(".json").is_file():
        return CorrectionModel.load(p.parent)
    return None


def apply_layer_correction(
    hidden: np.ndarray,
    layer_idx: int,
    model: CorrectionModel | None,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    """Add scaled MLP delta: corrected = hidden + scale * delta."""
    if model is None or scale == 0.0:
        return hidden
    h = np.asarray(hidden, dtype=np.float32)
    flat = h.ravel()
    delta = model.predict_delta(flat, layer_idx)
    out = flat + scale * delta
    if h.ndim > 1:
        return out.reshape(h.shape)
    return out


def apply_correction(
    hidden: np.ndarray,
    layer_idx: int,
    model: CorrectionModel | None,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    """Backward-compatible alias for ``apply_layer_correction``."""
    return apply_layer_correction(hidden, layer_idx, model, scale=scale)
