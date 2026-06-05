"""Phase 4 per-layer MLP correction tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from asdsl.correction.apply_correction import (
    CorrectionMLP,
    CorrectionModel,
    apply_layer_correction,
    load_correction,
)
from asdsl.correction.train_corrections import train_corrections

HIDDEN = 3072
NUM_LAYERS = 32


def _write_dummy_manifest(models_dir: Path, hidden_dim: int = 64) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    layers = []
    for layer in range(NUM_LAYERS):
        mlp = CorrectionMLP(hidden_dim=hidden_dim)
        torch.save(mlp.state_dict(), models_dir / f"layer_{layer}_correction.pt")
        layers.append({"layer": layer, "val_loss": 1e-4, "train_samples": 10})
    manifest = {
        "num_layers": NUM_LAYERS,
        "hidden_size": HIDDEN,
        "hidden_dim": hidden_dim,
        "layers": layers,
    }
    (models_dir / "correction_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_load_correction_from_manifest_dir(tmp_path):
    _write_dummy_manifest(tmp_path / "models")
    model = load_correction(tmp_path / "models")
    assert model is not None
    assert model.num_layers == NUM_LAYERS


def test_apply_preserves_shape(tmp_path):
    _write_dummy_manifest(tmp_path / "models")
    model = CorrectionModel.load(tmp_path / "models")
    h2d = np.random.randn(1, HIDDEN).astype(np.float32)
    out = apply_layer_correction(h2d, 0, model, scale=1.0)
    assert out.shape == h2d.shape


def test_mse_improves_on_synthetic_layer0(tmp_path):
    rng = np.random.default_rng(0)
    n = 64
    x = rng.standard_normal((n, HIDDEN)).astype(np.float32)
    true_mlp = CorrectionMLP(hidden_dim=32)
    with torch.no_grad():
        delta = true_mlp(torch.from_numpy(x)).numpy()
    y = delta + 0.01 * rng.standard_normal((n, HIDDEN)).astype(np.float32)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    arrays = {}
    z = np.zeros((n, HIDDEN), np.float32)
    for i in range(NUM_LAYERS):
        arrays[f"layer_{i}_hidden_quant"] = x if i == 0 else z
        arrays[f"layer_{i}_residual"] = y if i == 0 else z
    np.savez_compressed(data_dir / "layer_residuals.npz", **arrays)

    models_dir = train_corrections(
        data_dir, tmp_path / "trained", hidden_dim=32, epochs=40, early_stop_patience=6
    )
    model = CorrectionModel.load(models_dir)
    pred = model.predict_delta(x[0], 0)
    mse_before = float(np.mean(y[0] ** 2))
    mse_after = float(np.mean((y[0] - pred) ** 2))
    assert mse_after < mse_before


def test_correction_disabled_by_default():
    class _Store:
        _correction = None
        _enable_correction = False
        _correction_scale = 1.0

    h = np.ones(HIDDEN, dtype=np.float32)
    out = apply_layer_correction(h, 0, None, scale=1.0)
    np.testing.assert_array_equal(out, h)


def test_alpha_scaling(tmp_path):
    _write_dummy_manifest(tmp_path / "models")
    model = CorrectionModel.load(tmp_path / "models")
    h = np.zeros(HIDDEN, dtype=np.float32)
    d_full = model.predict_delta(h, 1)
    d_half = apply_layer_correction(h, 1, model, scale=0.5) - h
    np.testing.assert_allclose(d_half, 0.5 * d_full, rtol=1e-5, atol=1e-5)
