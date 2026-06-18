"""Greedy trajectory corpus: 5 prompts × 128 new tokens (plan standing protocol).

Golden hashes live in ``benchmarks/results/greedy_trajectory_golden.json``.
Regenerate after intentional model/kernel changes:

  python benchmarks/test_greedy_trajectory.py --update-golden
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GOLDEN_PATH = ROOT / "benchmarks" / "results" / "greedy_trajectory_golden.json"
CACHE_DIR = ROOT / "models" / "phi4_weight_cache"

# Five fixed user prompts (parity-style short decode).
TRAJECTORY_PROMPTS = [
    "The",
    "Hello world",
    "In 2026,",
    "def fibonacci(n):",
    "The quick brown fox",
]

N_NEW = 128


def _c0_env() -> None:
    os.environ["ASDSL_USE_UNIFIED"] = "1"
    os.environ["ASDSL_PREQ2"] = "1"
    os.environ["ASDSL_FUSED_GEMV"] = "1"
    os.environ["ASDSL_CHUNKED_GEMV"] = "1"
    os.environ["ASDSL_AFFINITY"] = "physical"
    os.environ["ASDSL_GROUP_SIZE"] = "32"
    os.environ.setdefault("OMP_NUM_THREADS", "12")


def _token_hash(tokens: list[int]) -> str:
    payload = ",".join(str(t) for t in tokens).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _encode_prompt(text: str) -> list[int]:
    from experiments.phi4_cpu_run import load_tokenizer

    tok = load_tokenizer()
    ids = tok.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors=None,
    )
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    return [int(x) for x in ids]


def _run_trajectories(config_label: str = "C0") -> dict[str, dict]:
    pytest.importorskip("asdsl.kernels._native_unified")
    if not CACHE_DIR.is_dir() or not any(CACHE_DIR.glob("*.safetensors")):
        pytest.skip("phi4 weight cache missing")

    from experiments.phi4_cpu_run import WeightStore, set_thread_count
    from asdsl.inference.unified_bridge import greedy_generate

    _c0_env()
    set_thread_count(12)
    store = WeightStore(bits=4, group_size=32, enable_qcsd=False)
    store.load()
    store.warm_cache()

    out: dict[str, dict] = {"config": config_label, "prompts": {}}
    for text in TRAJECTORY_PROMPTS:
        prompt_ids = _encode_prompt(text)
        tokens = greedy_generate(store, prompt_ids, N_NEW)
        new_tokens = tokens[len(prompt_ids) :]
        out["prompts"][text] = {
            "prompt_ids": prompt_ids,
            "new_tokens": new_tokens,
            "full_tokens": tokens,
            "sha256": _token_hash(new_tokens),
            "n_new": len(new_tokens),
        }
    return out


def update_golden() -> Path:
    doc = {
        "version": 1,
        "n_new_tokens": N_NEW,
        "trajectories": _run_trajectories("C0"),
    }
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return GOLDEN_PATH


@pytest.mark.slow
@pytest.mark.parametrize("prompt_text", TRAJECTORY_PROMPTS)
def test_greedy_trajectory_matches_golden(prompt_text: str) -> None:
    if not GOLDEN_PATH.is_file():
        pytest.skip(f"missing golden {GOLDEN_PATH}; run with --update-golden")

    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    expected = golden["trajectories"]["prompts"][prompt_text]
    live = _run_trajectories("C0")["prompts"][prompt_text]

    assert live["sha256"] == expected["sha256"], (
        f"trajectory drift for {prompt_text!r}\n"
        f"expected n={len(expected['new_tokens'])} got n={len(live['new_tokens'])}"
    )
    assert live["new_tokens"] == expected["new_tokens"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-golden", action="store_true")
    args = ap.parse_args()
    if args.update_golden:
        path = update_golden()
        print(f"Wrote {path}")
    else:
        raise SystemExit(pytest.main([__file__, "-q"]))
