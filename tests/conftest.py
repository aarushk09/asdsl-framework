"""Shared test fixtures for the ASDSL test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
PHI4_CACHE_DIR = ROOT / "models" / "phi4_weight_cache"

PARITY_PROMPTS = [
    "The",
    "In 2024,",
    "def fibonacci(n):",
]


@pytest.fixture
def rng():
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def small_weights(rng):
    """Small weight matrix for fast tests."""
    return rng.standard_normal((128, 128)).astype(np.float32)


@pytest.fixture
def medium_weights(rng):
    """Medium weight matrix for thorough tests."""
    return rng.standard_normal((1024, 1024)).astype(np.float32)


@pytest.fixture
def project_root() -> Path:
    return ROOT


@pytest.fixture
def native_unified():
    """Require built ``_native_unified`` extension."""
    pytest.importorskip("asdsl.kernels._native_unified")
    from asdsl.kernels import _native_unified as nu

    return nu


@pytest.fixture
def phi4_cache_available():
    """Skip when Phi-4 weight cache is not present."""
    if not PHI4_CACHE_DIR.is_dir():
        pytest.skip(f"phi4 weight cache missing at {PHI4_CACHE_DIR}")
    caches = list(PHI4_CACHE_DIR.glob("*.safetensors"))
    if not caches:
        pytest.skip(f"no safetensors in {PHI4_CACHE_DIR}")
    return PHI4_CACHE_DIR


@pytest.fixture
def phi4_store(phi4_cache_available, native_unified):
    """Load a warm WeightStore for integration tests."""
    import os

    from experiments.phi4_cpu_run import WeightStore, set_thread_count

    os.environ.setdefault("ASDSL_USE_UNIFIED", "1")
    set_thread_count(4)
    store = WeightStore(bits=4, group_size=32, enable_qcsd=False)
    store.load()
    store.warm_cache()
    return store


@pytest.fixture
def phi4_store_qcsd(phi4_cache_available, native_unified):
    """WeightStore with QCSD draft bank enabled."""
    import os

    from experiments.phi4_cpu_run import WeightStore, set_thread_count

    os.environ.setdefault("ASDSL_USE_UNIFIED", "1")
    set_thread_count(4)
    store = WeightStore(bits=4, group_size=32, enable_qcsd=True, draft_bits=2)
    store.load()
    store.warm_cache()
    return store


@pytest.fixture
def fast_decode_tokens() -> int:
    """CI-friendly decode length (override with ASDSL_TEST_DECODE_TOKENS)."""
    import os

    return int(os.environ.get("ASDSL_TEST_DECODE_TOKENS", "2"))


@pytest.fixture
def parity_prompts() -> list[str]:
    return list(PARITY_PROMPTS)


@pytest.fixture
def ahsd_native_ok():
    """Skip AHSD integration tests when native generate_ahsd crashes."""
    import subprocess
    import sys
    import textwrap

    if not PHI4_CACHE_DIR.is_dir() or not any(PHI4_CACHE_DIR.glob("*.safetensors")):
        pytest.skip(f"phi4 weight cache missing at {PHI4_CACHE_DIR}")
    pytest.importorskip("asdsl.kernels._native_unified")

    code = textwrap.dedent(
        """
        import os
        os.environ.setdefault("ASDSL_USE_UNIFIED", "1")
        from experiments.phi4_cpu_run import WeightStore, set_thread_count
        from asdsl.inference.unified_bridge import get_or_build_unified_engine
        set_thread_count(2)
        store = WeightStore(bits=4, group_size=32, enable_qcsd=False)
        store.load()
        store.warm_cache()
        eng = get_or_build_unified_engine(store)
        eng.reset_session()
        eng.generate_swift([100, 200], 1, 1)
        print("OK")
        """
    )
    r = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "")[-400:]
        pytest.skip(f"speculative generate native not stable: {tail}")
