"""Pre-flight and fast-fail behavior for ``scripts/run_full_benchmark.py``."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


def _bench_mod():
    root = Path(__file__).resolve().parent.parent
    path = root / "scripts" / "run_full_benchmark.py"
    spec = importlib.util.spec_from_file_location("_rfbench_phi4", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, root


def test_require_phi4_index_sys_exit_2_when_missing(tmp_path: Path) -> None:
    m, _ = _bench_mod()
    with pytest.raises(SystemExit) as ei:
        m._require_phi4_index_or_exit(tmp_path)
    assert ei.value.code == 2


def test_phi4_cli_fast_fail_without_weights() -> None:
    """Subprocess: no local index -> exit 2 and error text before full Phi-4 load."""
    _, root = _bench_mod()
    idx = root / "models" / "phi4-multimodal-instruct" / "model.safetensors.index.json"
    if idx.is_file():
        pytest.skip("Local Phi-4 index exists; fast-fail test requires missing weights")

    r = subprocess.run(
        [sys.executable, str(root / "scripts" / "run_full_benchmark.py"), "--phi4"],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 2, (r.stdout, r.stderr)
    assert "Expected index:" in r.stderr or "Expected index:" in r.stdout
    assert "fast-fail" in r.stderr.lower() or "fast-fail" in r.stdout.lower()
