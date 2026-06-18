"""Collect imatrix-lite importance vectors for C0.1 g128 quant (Phase 4).

Bootstrap calibration: per-channel |embed[token]| statistics over a short token
window. Replace with full activation capture when engine hooks land.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    os.environ.setdefault("ASDSL_USE_UNIFIED", "1")

    from experiments.phi4_cpu_run import HIDDEN, INTER, NUM_LAYERS, VOCAB, WeightStore
    from asdsl.quantization.imatrix_cache import imatrix_cache_path_for_store, save_imatrix_cache
    from asdsl.quantization.imatrix_lite import collect_channel_importance

    n_tokens = int(os.environ.get("ASDSL_IMATRIX_TOKENS", "512"))

    store = WeightStore(bits=4)
    store.load()

    embed = store.embed_f16.detach().cpu().float().numpy()
    vocab = min(int(embed.shape[0]), VOCAB)
    rng = np.random.default_rng(int(os.environ.get("ASDSL_IMATRIX_SEED", "42")))
    ids = rng.integers(0, vocab, size=n_tokens, endpoint=False)
    gu_mat = np.stack([np.abs(embed[int(tid)][:HIDDEN]) for tid in ids], axis=0).astype(np.float32)
    gu_imp = collect_channel_importance(gu_mat)
    dn_imp = np.ones(INTER, dtype=np.float32)

    importance: dict[tuple[int, str], np.ndarray] = {}
    for layer in range(NUM_LAYERS):
        importance[(layer, "gate_up_proj")] = gu_imp.copy()
        importance[(layer, "down_proj")] = dn_imp.copy()

    path = imatrix_cache_path_for_store(store)
    save_imatrix_cache(store, path, importance, n_samples=len(gu_mat))
    print(f"Wrote {path} ({len(gu_mat)} embed samples)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
