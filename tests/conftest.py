"""Shared test fixtures for the ASDSL test suite."""

import numpy as np
import pytest


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
