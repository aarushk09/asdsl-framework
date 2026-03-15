"""Tests for streaming text generation."""

import time

import numpy as np
import pytest

from asdsl.inference.engine import (
    ASDSLEngine,
    GenerationResult,
    StreamToken,
)


class TestStreamToken:
    """Tests for the StreamToken dataclass."""

    def test_stream_token_creation(self):
        """StreamToken should hold all expected fields."""
        tok = StreamToken(
            token_id=42,
            step=0,
            is_eos=False,
            elapsed_s=0.5,
            tokens_per_second=2.0,
        )
        assert tok.token_id == 42
        assert tok.step == 0
        assert tok.is_eos is False
        assert tok.elapsed_s == 0.5
        assert tok.tokens_per_second == 2.0

    def test_stream_token_defaults(self):
        """StreamToken should have sensible defaults."""
        tok = StreamToken(token_id=1, step=0)
        assert tok.is_eos is False
        assert tok.elapsed_s == 0.0
        assert tok.tokens_per_second == 0.0

    def test_stream_token_eos(self):
        """EOS token should be flaggable."""
        tok = StreamToken(token_id=199999, step=5, is_eos=True)
        assert tok.is_eos is True

    def test_stream_token_timing_increases(self):
        """Sequential tokens should have increasing elapsed time."""
        tokens = [
            StreamToken(token_id=i, step=i, elapsed_s=i * 0.1)
            for i in range(5)
        ]
        for i in range(1, len(tokens)):
            assert tokens[i].elapsed_s > tokens[i - 1].elapsed_s


class TestStreamTokenFromPhi4:
    """Tests for the StreamToken dataclass in phi4_cpu_run."""

    @pytest.fixture(autouse=True)
    def _add_root(self):
        import sys
        from pathlib import Path
        root = str(Path(__file__).resolve().parent.parent)
        if root not in sys.path:
            sys.path.insert(0, root)

    def test_import_stream_token(self):
        """StreamToken should be importable from phi4_cpu_run."""
        from experiments.phi4_cpu_run import StreamToken as ST
        assert ST is not None

    def test_import_generate_stream(self):
        """generate_stream should be importable from phi4_cpu_run."""
        from experiments.phi4_cpu_run import generate_stream
        assert callable(generate_stream)

    def test_phi4_stream_token_fields(self):
        """phi4_cpu_run StreamToken has additional text field."""
        from experiments.phi4_cpu_run import StreamToken as ST
        tok = ST(
            text="hello",
            token_id=42,
            step=0,
            is_eos=False,
            elapsed_s=0.1,
            tokens_per_second=10.0,
        )
        assert tok.text == "hello"
        assert tok.token_id == 42
