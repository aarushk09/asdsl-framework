#!/usr/bin/env python3
"""
EAGLE-3 MTP (Multi-Token Prediction) head training for ASDSL speculative decoding.

Architecture (regularized variant — Phase 5):
  Input:  concat(prev_final_hidden[3072], **previous** token embed[3072]) = [6144]
  fc1:    Linear(6144, 1024, bias=True)
  norm:   LayerNorm(1024)
  gelu:   GELU activation
  drop:   Dropout(0.3) — training only
  proj:   Linear(1024, 3072, bias=True)  — maps back to Phi-4 hidden dim
  head:   Phi-4 lm_head(3072, vocab_size) float16 (frozen, shared with model)

Trainable params: ~9.5M  (vs 18.9M in old architecture)
Draft FLOP cost:  9.7M per step  (vs 37.7M — 3.9× faster)

Training procedure:
  1. Collect 32 prompts × 24 decode steps = 768 (prev_hidden, token) pairs
  2. 80/20 train/val split, AdamW (lr=3e-4, wd=0.05), 30 epochs, early stopping
  3. Save best checkpoint (best val accuracy) to models/mtp_head.pt

Usage:
    python scripts/train_mtp_head.py --quick    # 32 samples, 10 epochs (CI)
    python scripts/train_mtp_head.py            # full training (~4-5 min)
    python scripts/train_mtp_head.py --sanity   # load and verify saved head
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def _configure_env() -> None:
    import os
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_JAX", "0")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


_configure_env()

INPUT_DIM = 6144      # concat(hidden[3072], embed[3072])
MTP_HIDDEN = 1024     # new smaller hidden dim (was 3072)
PHI4_HIDDEN = 3072    # Phi-4 hidden dimension (for proj + lm_head)


def _check_memory(min_gb: float = 2.0) -> None:
    try:
        import psutil
        avail = psutil.virtual_memory().available / 1e9
        if avail < min_gb:
            raise MemoryError(
                f"Only {avail:.1f} GB available, need {min_gb:.1f} GB. "
                "Free memory before running this script."
            )
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Shared prompt corpus
# ---------------------------------------------------------------------------

from asdsl.calibration_data import CALIBRATION_PROMPTS, QUICK_PROMPTS


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_training_pairs(
    store,
    tokenizer,
    prompts: list[str],
    max_steps: int = 24,
    verbose: bool = True,
) -> tuple[list, list, list]:
    """Collect triples (prev_final_hidden, last_token_id, next_token_id).

    prev_final_hidden: post-rms_norm hidden **before** emitting next_token_id
    last_token_id:     last token already in the sequence (embedding is fed to MTP)
    next_token_id:     greedy next token (MTP target) — same as ``generate_eagle3``
        which uses ``concat(_last_final_hidden, embed(current_token))``.
    """
    import gc
    import numpy as np
    import torch

    try:
        from experiments.phi4_cpu_run import (
            KVHistory, build_rope_cache, forward_layer, rms_norm,
            NUM_LAYERS, ROTARY_DIM,
        )
    except ImportError as e:
        print(f"ERROR importing phi4_cpu_run: {e}")
        raise

    hiddens: list[np.ndarray] = []
    last_tok_ids: list[int] = []
    next_tok_ids: list[int] = []

    print(f"  Collecting {len(prompts)} prompts × {max_steps} steps...")

    for pi, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]
        try:
            input_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
        except Exception as e:
            print(f"  Prompt {pi}: tokenizer error: {e}")
            continue

        max_seq = len(input_ids) + max_steps + 32
        rope_cos, rope_sin = build_rope_cache(max_seq, ROTARY_DIM)
        kv_hist = KVHistory(max_seq=max_seq)

        with torch.inference_mode():
            # Prefill
            hidden = None
            for pos, tid in enumerate(input_ids):
                hidden = store.embed_f16[tid].float().unsqueeze(0)
                for i in range(NUM_LAYERS):
                    hidden = forward_layer(
                        hidden, i, store, kv_hist, rope_cos, rope_sin, pos)
            prev_hidden = rms_norm(hidden, store.final_norm)
            # Last token in KV before first decode step (matches _run_mtp_draft input).
            last_tok = int(input_ids[-1])

            # Decode: collect (prev_hidden, last_tok, next_tok) triples
            pos = len(input_ids)
            for step in range(max_steps):
                logits = store.lm_head_matvec(prev_hidden)
                next_tok = int(logits.argmax())

                h_np = prev_hidden.detach().cpu().float().numpy().ravel()  # [3072]
                hiddens.append(h_np)
                last_tok_ids.append(last_tok)
                next_tok_ids.append(next_tok)

                # Advance: embed next_tok → run through layers → new hidden
                hidden = store.embed_f16[next_tok].float().unsqueeze(0)
                for i in range(NUM_LAYERS):
                    hidden = forward_layer(
                        hidden, i, store, kv_hist, rope_cos, rope_sin, pos)
                prev_hidden = rms_norm(hidden, store.final_norm)
                last_tok = next_tok
                pos += 1

        del kv_hist
        gc.collect()
        if verbose:
            print(f"  Prompt {pi+1}/{len(prompts)}: {len(hiddens)} pairs total")

    return hiddens, last_tok_ids, next_tok_ids


# ---------------------------------------------------------------------------
# MTP head model
# ---------------------------------------------------------------------------

def _build_model(lm_head_f16, dropout_rate: float = 0.3):
    """Build MTPHead nn.Module with new 1024 hidden-dim architecture."""
    import torch
    import torch.nn as nn

    VOCAB = lm_head_f16.shape[0]

    class MTPHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1  = nn.Linear(INPUT_DIM, MTP_HIDDEN)
            self.norm = nn.LayerNorm(MTP_HIDDEN)
            self.drop = nn.Dropout(dropout_rate)
            self.proj = nn.Linear(MTP_HIDDEN, PHI4_HIDDEN)
            self.register_buffer("lm_head", lm_head_f16.float())

        def forward(self, x):
            h = self.drop(torch.nn.functional.gelu(self.norm(self.fc1(x))))
            h_proj = self.proj(h)
            return h_proj @ self.lm_head.T  # [batch, vocab]

    return MTPHead()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_mtp_head(
    out_path: Path,
    prompts: list[str],
    max_steps: int,
    n_epochs: int = 30,
    lr: float = 3e-4,
    weight_decay: float = 0.05,
    batch_size: int = 32,
    dropout_rate: float = 0.3,
    val_split: float = 0.20,
    early_stopping_patience: int = 5,
    verbose: bool = True,
) -> dict:
    """Train MTP head with val split, early stopping, and best-ckpt saving."""
    import gc
    import numpy as np
    import torch
    import torch.nn as nn

    _check_memory(min_gb=3.5)

    print("\n[EAGLE-3] Loading Phi-4 model for data collection...")
    from experiments.phi4_cpu_run import WeightStore
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/Phi-4-multimodal-instruct", trust_remote_code=True
    )
    store = WeightStore(bits=4, enable_qcsd=False)
    store.load()
    store.warm_cache()
    # Profile G runs with FATReLU + transposed down_proj; training must collect
    # hidden states from the same forward path or MTP sees OOD activations → 0% acceptance.
    _fr = ROOT / "phi4_fatrelu_thresholds.json"
    if _fr.exists():
        store.load_fatrelu(str(_fr))
        print("[EAGLE-3] FATReLU thresholds loaded for collection (matches Profile G forward)")
    gc.collect()

    print("[EAGLE-3] Collecting training pairs...")
    t0 = time.perf_counter()
    hiddens, last_tok_ids, next_tok_ids = collect_training_pairs(
        store, tokenizer, prompts, max_steps=max_steps, verbose=verbose
    )
    t_collect = time.perf_counter() - t0
    n_total = len(hiddens)
    print(f"[EAGLE-3] Collected {n_total} samples in {t_collect:.1f}s")

    if n_total < 4:
        raise RuntimeError(f"Too few samples: {n_total}.")

    # Retrieve lm_head and embed table from store before deletion
    lm_head_f16 = store.lm_head          # [vocab, 3072] float16 Tensor
    embed_f16   = store.embed_f16        # [vocab, 3072] float16 Tensor

    # X = concat(prev_hidden, embed(**last**_token)); Y = next greedy token
    print("[EAGLE-3] Building tensors...")
    X_list, Y_list = [], []
    for i in range(n_total):
        prev_h = torch.from_numpy(hiddens[i]).float()           # [3072]
        tok_emb = embed_f16[last_tok_ids[i]].float()            # [3072]
        X_list.append(torch.cat([prev_h, tok_emb], dim=0))       # [6144]
        Y_list.append(next_tok_ids[i])

    X = torch.stack(X_list)                                       # [N, 6144]
    Y = torch.tensor(Y_list, dtype=torch.long)                    # [N]
    del hiddens, last_tok_ids, next_tok_ids, X_list, Y_list, store
    gc.collect()

    # Train/val split
    N = X.shape[0]
    n_val    = max(1, int(N * val_split))
    n_train  = N - n_val
    perm     = torch.randperm(N)
    X_tr, Y_tr = X[perm[:n_train]], Y[perm[:n_train]]
    X_va, Y_va = X[perm[n_train:]], Y[perm[n_train:]]

    print(f"[EAGLE-3] Train samples: {n_train}, Val samples: {n_val}")

    # Model + optimizer
    print(f"[EAGLE-3] Training MTP head (input={INPUT_DIM}, hidden={MTP_HIDDEN}, "
          f"proj to {PHI4_HIDDEN}, vocab={lm_head_f16.shape[0]})...")
    model = _build_model(lm_head_f16, dropout_rate=dropout_rate)
    # Only optimize non-frozen params (lm_head is register_buffer)
    trainable = [p for n, p in model.named_parameters() if "lm_head" not in n]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    best_val_acc1   = -1.0
    best_train_acc1 =  0.0
    best_epoch      =  0
    patience_ctr    =  0
    best_state      = None
    # Val set can be tiny (quick mode); tie-break with train acc so we do not
    # keep epoch 1 when val stays 0% but train improves.
    best_score: tuple[float, float] = (-1.0, -1.0)

    for epoch in range(n_epochs):
        # --- train ---
        model.train()
        perm_tr = torch.randperm(n_train)
        Xs, Ys  = X_tr[perm_tr], Y_tr[perm_tr]
        tot_loss, tot_c1, n_batches = 0.0, 0, 0
        for b in range(0, n_train, batch_size):
            xb = Xs[b:b+batch_size]
            yb = Ys[b:b+batch_size]
            logits = model(xb)
            loss   = nn.functional.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            with torch.no_grad():
                tot_c1     += (logits.argmax(1) == yb).sum().item()
                tot_loss   += loss.item(); n_batches += 1
        scheduler.step()
        train_acc1  = tot_c1 / n_train * 100
        avg_loss    = tot_loss / max(n_batches, 1)

        # --- val ---
        model.eval()
        with torch.no_grad():
            va_logits = model(X_va)
            val_acc1  = (va_logits.argmax(1) == Y_va).float().mean().item() * 100

        # Early stopping + best checkpoint (lexicographic val, then train)
        score = (val_acc1, train_acc1)
        if score > best_score:
            best_score = score
            best_val_acc1 = val_acc1
            best_train_acc1 = train_acc1
            best_epoch = epoch + 1
            patience_ctr = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()
                          if k != "lm_head"}
        else:
            patience_ctr += 1

        if verbose:
            print(f"  Epoch {epoch+1:3d}/{n_epochs}: "
                  f"loss={avg_loss:.4f}  train_top1={train_acc1:.1f}%  "
                  f"val_top1={val_acc1:.1f}%"
                  + ("  *" if patience_ctr == 0 else ""))

        if patience_ctr >= early_stopping_patience:
            print(f"[EAGLE-3] Early stopping at epoch {epoch+1} "
                  f"(no val improvement for {early_stopping_patience} epochs)")
            break

    print(f"\n[EAGLE-3] Best checkpoint: epoch {best_epoch}  "
          f"train_top1={best_train_acc1:.1f}%  val_top1={best_val_acc1:.1f}%")

    # Save best checkpoint
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict: dict = {}
    if best_state is not None:
        save_dict.update(best_state)
    else:
        # Fallback: save current state
        for k, v in model.state_dict().items():
            if k != "lm_head":
                save_dict[k] = v

    checkpoint = {
        # Named keys expected by load_mtp_head()
        "fc1_weight":          save_dict["fc1.weight"].half(),
        "fc1_bias":            save_dict["fc1.bias"].half(),
        "norm_weight":         save_dict["norm.weight"].half(),
        "norm_bias":           save_dict["norm.bias"].half(),
        "proj_weight":         save_dict["proj.weight"].half(),
        "proj_bias":           save_dict["proj.bias"].half(),
        # Metadata
        "hidden_dim_mtp":      MTP_HIDDEN,
        "input_dim":           INPUT_DIM,
        "phi4_hidden_dim":     PHI4_HIDDEN,
        "val_top1_accuracy":   float(best_val_acc1),
        "train_top1_accuracy": float(best_train_acc1),
        "n_training_samples":  n_train,
        "n_val_samples":       n_val,
        "best_epoch":          best_epoch,
    }
    torch.save(checkpoint, str(out_path))
    size_mb = out_path.stat().st_size / 1e6
    print(f"[EAGLE-3] Saved MTP head to {out_path} ({size_mb:.1f} MB)")

    return {
        "val_top1":   best_val_acc1,
        "train_top1": best_train_acc1,
        "best_epoch": best_epoch,
        "n_train":    n_train,
        "n_val":      n_val,
    }


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def sanity_check_head(head_path: Path) -> None:
    """Load saved head, verify keys, run synthetic forward passes."""
    import numpy as np
    import torch

    if not head_path.exists():
        print(f"[sanity] ERROR: {head_path} not found")
        return

    ck = torch.load(str(head_path), map_location="cpu")
    val_acc   = ck.get("val_top1_accuracy", float("nan"))
    train_acc = ck.get("train_top1_accuracy", float("nan"))
    n_tr  = ck.get("n_training_samples", "?")
    n_va  = ck.get("n_val_samples", "?")
    hdim  = ck.get("hidden_dim_mtp", MTP_HIDDEN)

    print(f"[sanity] {head_path.name}")
    print(f"  hidden_dim_mtp:    {hdim}")
    print(f"  train_top1:        {train_acc:.1f}%")
    print(f"  val_top1:          {val_acc:.1f}%")
    print(f"  train/val samples: {n_tr}/{n_va}")

    fc1_w   = ck["fc1_weight"].float().numpy()    # [1024, 6144]
    fc1_b   = ck["fc1_bias"].float().numpy()      # [1024]
    norm_w  = ck["norm_weight"].float().numpy()   # [1024]
    norm_b  = ck["norm_bias"].float().numpy()     # [1024]
    proj_w  = ck["proj_weight"].float().numpy()   # [3072, 1024]
    proj_b  = ck["proj_bias"].float().numpy()     # [3072]
    vocab   = 200064

    rng = np.random.default_rng(42)
    lm_head_np = rng.standard_normal((vocab, PHI4_HIDDEN)).astype(np.float16)

    for i in range(3):
        x = np.concatenate([
            rng.standard_normal(PHI4_HIDDEN).astype(np.float32),
            rng.standard_normal(PHI4_HIDDEN).astype(np.float32),
        ])  # [6144]
        h = x @ fc1_w.T + fc1_b
        mean, std = h.mean(), h.std() + 1e-5
        h = (h - mean) / std * norm_w + norm_b
        h = h * 0.5 * (1.0 + np.tanh(0.7978845608 * (h + 0.044715 * h**3)))
        h_proj = h @ proj_w.T + proj_b
        logits = h_proj @ lm_head_np.T.astype(np.float32)
        print(f"  Run {i+1}: pred={int(np.argmax(logits))}  "
              f"logits shape={logits.shape}  max={float(logits.max()):.3f}")

    print("[sanity] OK")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="EAGLE-3 MTP head training")
    ap.add_argument("--quick",  action="store_true",
                    help="Quick mode: 4 prompts × 8 steps, 10 epochs, no early stopping")
    ap.add_argument("--sanity", action="store_true",
                    help="Load and verify saved head (no training)")
    ap.add_argument("--out",    type=str,   default=str(ROOT / "models" / "mtp_head.pt"))
    ap.add_argument("--epochs", type=int,   default=None)
    ap.add_argument("--lr",     type=float, default=3e-4)
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()

    out_path = Path(args.out)

    if args.sanity:
        sanity_check_head(out_path)
        return

    if args.quick:
        prompts  = QUICK_PROMPTS
        steps    = args.max_steps or 8
        epochs   = args.epochs or 10
        patience = 999   # no early stopping in quick mode
        print(f"[EAGLE-3] Quick mode: {len(prompts)} prompts × {steps} steps, "
              f"{epochs} epochs")
    else:
        prompts  = CALIBRATION_PROMPTS
        steps    = args.max_steps or 24
        epochs   = args.epochs or 30
        patience = 5
        print(f"[EAGLE-3] Full mode: {len(prompts)} prompts × {steps} steps, "
              f"{epochs} epochs, early stopping (patience={patience})")

    _check_memory(min_gb=4.0 if not args.quick else 3.0)
    t0 = time.perf_counter()
    stats = train_mtp_head(
        out_path=out_path,
        prompts=prompts,
        max_steps=steps,
        n_epochs=epochs,
        lr=args.lr,
        early_stopping_patience=patience,
    )
    total_s = time.perf_counter() - t0
    print(f"\n[EAGLE-3] Done in {total_s:.0f}s")
    print(f"  Train top-1: {stats['train_top1']:.1f}%")
    print(f"  Val   top-1: {stats['val_top1']:.1f}%")
    gap = stats["train_top1"] - stats["val_top1"]
    print(f"  Train-val gap: {gap:.1f}%  (target: <30%)")

    # Write summary JSON
    stats_path = out_path.with_suffix(".json")
    summary = {
        "val_top1_accuracy":   stats["val_top1"],
        "train_top1_accuracy": stats["train_top1"],
        "train_val_gap":       gap,
        "best_epoch":          stats["best_epoch"],
        "n_training_samples":  stats["n_train"],
        "n_val_samples":       stats["n_val"],
        "total_s":             total_s,
        "hidden_dim_mtp":      MTP_HIDDEN,
        "input_dim":           INPUT_DIM,
        "phi4_hidden_dim":     PHI4_HIDDEN,
    }
    stats_path.write_text(json.dumps(summary, indent=2))
    print(f"[EAGLE-3] Stats -> {stats_path}")

    if stats["val_top1"] < 5.0:
        print("[EAGLE-3] WARNING: val top-1 < 5% — head may need more data")
    elif gap < 30.0:
        print("[EAGLE-3] ✓ Overfitting check passed (gap < 30%)")
    else:
        print(f"[EAGLE-3] WARNING: train-val gap {gap:.1f}% may indicate overfitting")


if __name__ == "__main__":
    main()
