"""Train 32 per-layer MLP correctors from collected hidden-state residuals."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from asdsl.correction.apply_correction import CorrectionMLP, HIDDEN, NUM_LAYERS

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _load_residuals(data_dir: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    npz_path = data_dir / "layer_residuals.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"missing {npz_path}; run collect_training_data first")
    data = np.load(npz_path)
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for layer in range(NUM_LAYERS):
        q_key = f"layer_{layer}_hidden_quant"
        r_key = f"layer_{layer}_residual"
        if q_key not in data.files or r_key not in data.files:
            raise KeyError(f"missing {q_key} or {r_key} in {npz_path}")
        out[layer] = (
            data[q_key].astype(np.float32),
            data[r_key].astype(np.float32),
        )
    return out


def train_corrections(
    data_dir: Path | str,
    models_dir: Path | str | None = None,
    *,
    hidden_dim: int = 64,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    early_stop_patience: int = 8,
    val_fraction: float = 0.2,
) -> Path:
    """Train one MLP per layer; write layer_{l}_correction.pt + correction_manifest.json."""
    if torch is None:
        raise RuntimeError("PyTorch required for train_corrections")

    data_dir = Path(data_dir)
    models_dir = Path(models_dir or Path(__file__).resolve().parent / "models")
    models_dir.mkdir(parents=True, exist_ok=True)

    per_layer = _load_residuals(data_dir)
    manifest_layers: list[dict] = []

    for layer_idx in range(NUM_LAYERS):
        x_all, y_all = per_layer[layer_idx]
        n = x_all.shape[0]
        if n < 2:
            raise ValueError(f"layer {layer_idx}: need >= 2 samples, got {n}")

        rng = np.random.default_rng(42 + layer_idx)
        perm = rng.permutation(n)
        n_val = max(1, int(n * val_fraction))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:] if n_val < n else perm[:1]

        x_train = torch.from_numpy(x_all[train_idx])
        y_train = torch.from_numpy(y_all[train_idx])
        x_val = torch.from_numpy(x_all[val_idx])
        y_val = torch.from_numpy(y_all[val_idx])

        train_loader = DataLoader(
            TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True
        )
        model = CorrectionMLP(hidden_dim=hidden_dim)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()

        best_val = float("inf")
        stale = 0
        best_state = None

        for _epoch in range(epochs):
            model.train()
            for xb, yb in train_loader:
                opt.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(x_val), y_val).item())

            if val_loss < best_val - 1e-8:
                best_val = val_loss
                stale = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= early_stop_patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        out_pt = models_dir / f"layer_{layer_idx}_correction.pt"
        torch.save(model.state_dict(), out_pt)
        manifest_layers.append(
            {"layer": layer_idx, "val_loss": best_val, "train_samples": len(train_idx)}
        )

    manifest = {
        "num_layers": NUM_LAYERS,
        "hidden_size": HIDDEN,
        "hidden_dim": hidden_dim,
        "epochs_max": epochs,
        "layers": manifest_layers,
    }
    manifest_path = models_dir / "correction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return models_dir
