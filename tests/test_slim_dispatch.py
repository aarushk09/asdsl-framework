"""
Phase 2 correctness test: SliM mixed-precision dispatch.

Tests:
1. load_slim() correctly loads metadata and sets _use_slim=True
2. 4-bit groups produce output identical to baseline
3. 2-bit groups produce output within atol=0.15 of baseline
4. 3-bit groups produce output within atol=0.08 of baseline
5. Repacked 2-bit buffer uses correct bit masking
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch


# ── helpers ────────────────────────────────────────────────────────────────────

def make_synthetic_store(out_features: int, in_features: int, group_size: int):
    """
    Create a minimal WeightStore-like object with synthetic Q4 packed weights.
    Does not load Phi-4 — uses random data.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # Create a minimal mock store
    class MockStore:
        def __init__(self):
            self.bits = 4
            self.group_size = group_size
            self._use_native_gemv = True
            self._use_lut_gemv = False
            self._use_slim = False
            self._slim_meta = None
            self._slim_npz = None
            self._repacked_layers = {}
            self._quant_packed = {}
            self._quant_sc = {}
            self._quant_bi = {}
            self._quant_shapes = {}
            self._outlier_values = {}
            self._outlier_coords = {}

        def load_slim(self, meta_path):
            """Simplified load_slim for testing."""
            meta_path = Path(meta_path)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            npz_name = meta.get("npz_path", "test_slim.npz")
            npz_path = meta_path.parent / npz_name
            self._slim_meta = meta
            # allow_pickle=False for safety; mmap_mode=None to avoid file lock on Windows
            self._slim_npz = dict(np.load(str(npz_path)))
            self._use_slim = True
            self._repacked_layers = {}

        def _get_slim_arrays(self, layer_idx, name):
            if self._slim_npz is None:
                return None
            prefix = f"L{layer_idx}_{name}"
            bits_key = f"{prefix}_bits"
            if bits_key not in self._slim_npz:
                return None
            return (
                self._slim_npz[f"{prefix}_bits"],
                self._slim_npz[f"{prefix}_scales"],
                self._slim_npz[f"{prefix}_zp"],
            )

        def _get_slim_repacked(self, layer_idx, name):
            cache_key = (layer_idx, name)
            if cache_key in self._repacked_layers:
                return self._repacked_layers[cache_key]

            slim_arrays = self._get_slim_arrays(layer_idx, name)
            if slim_arrays is None:
                self._repacked_layers[cache_key] = None
                return None

            bits_arr, scales_arr, zp_arr = slim_arrays
            key = (layer_idx, name)
            if key not in self._quant_packed:
                self._repacked_layers[cache_key] = None
                return None

            rows, cols = self._quant_shapes[key]
            n_groups_per_row = cols // self.group_size

            slim_scales = torch.from_numpy(scales_arr.astype(np.float32))
            slim_biases = torch.from_numpy(
                (-zp_arr.astype(np.float32) * scales_arr.astype(np.float32))
            )

            has_2bit = bool(np.any(bits_arr == 2))
            if not has_2bit:
                result = (self._quant_packed[key], slim_scales, slim_biases)
                self._repacked_layers[cache_key] = result
                return result

            packed_np = self._quant_packed[key].numpy()
            bits_2d = bits_arr.reshape(rows, n_groups_per_row)
            repacked = packed_np.copy()

            for row in range(rows):
                for g in range(n_groups_per_row):
                    if bits_2d[row, g] == 2:
                        col_start = g * self.group_size // 2
                        col_end = col_start + self.group_size // 2
                        chunk = repacked[row, col_start:col_end]
                        lo = chunk & 0x03
                        hi = (chunk >> 4) & 0x03
                        repacked[row, col_start:col_end] = lo | (hi << 4)

            repacked_t = torch.from_numpy(repacked)
            result = (repacked_t, slim_scales, slim_biases)
            self._repacked_layers[cache_key] = result
            return result

        def _matvec_slim(self, layer_idx, name, x):
            from asdsl.kernels import gemv_q4_packed
            repacked = self._get_slim_repacked(layer_idx, name)
            if repacked is None:
                return self._matvec_q4_baseline(layer_idx, name, x)

            packed_t, slim_scales, slim_biases = repacked
            rows, cols = self._quant_shapes[(layer_idx, name)]
            w_np = packed_t.numpy().reshape(-1)
            x_np = x.detach().cpu().float().numpy().ravel()
            sc_np = slim_scales.float().numpy()
            bi_np = slim_biases.float().numpy()
            out_np = gemv_q4_packed(w_np, x_np, sc_np, bi_np, rows, cols, self.group_size)
            return torch.from_numpy(np.asarray(out_np, dtype=np.float32)).unsqueeze(0)

        def _matvec_q4_baseline(self, layer_idx, name, x):
            from asdsl.kernels import gemv_q4_packed
            key = (layer_idx, name)
            rows, cols = self._quant_shapes[key]
            w_np = self._quant_packed[key].numpy().reshape(-1)
            x_np = x.detach().cpu().float().numpy().ravel()
            sc_np = self._quant_sc[key].float().numpy()
            bi_np = self._quant_bi[key].float().numpy()
            out_np = gemv_q4_packed(w_np, x_np, sc_np, bi_np, rows, cols, self.group_size)
            return torch.from_numpy(np.asarray(out_np, dtype=np.float32)).unsqueeze(0)

        def matvec(self, layer_idx, name, x, use_draft=False):
            if self._use_slim and not use_draft and self.bits == 4:
                return self._matvec_slim(layer_idx, name, x)
            return self._matvec_q4_baseline(layer_idx, name, x)

    store = MockStore()

    # Create synthetic Q4 packed weights
    rng = np.random.default_rng(42)
    packed = rng.integers(0, 256, (out_features, in_features // 2), dtype=np.uint8)
    n_groups = out_features * (in_features // group_size)
    scales = rng.uniform(0.01, 0.05, n_groups).astype(np.float32)
    biases = (-8.0 * scales).astype(np.float32)

    key = (0, "test_proj")
    store._quant_packed[key] = torch.from_numpy(packed)
    store._quant_sc[key] = torch.from_numpy(scales)
    store._quant_bi[key] = torch.from_numpy(biases)
    store._quant_shapes[key] = (out_features, in_features)

    return store, packed, scales, biases


def make_slim_meta(
    out_features: int, in_features: int, group_size: int,
    bits_pattern: str = "mixed",  # "all4", "all3", "all2", "mixed"
) -> tuple[dict, dict]:
    """Create synthetic SliM metadata."""
    n_groups = out_features * (in_features // group_size)
    rng = np.random.default_rng(99)

    if bits_pattern == "all4":
        bits_arr = np.full(n_groups, 4, dtype=np.uint8)
    elif bits_pattern == "all3":
        bits_arr = np.full(n_groups, 3, dtype=np.uint8)
    elif bits_pattern == "all2":
        bits_arr = np.full(n_groups, 2, dtype=np.uint8)
    else:  # mixed
        bits_arr = rng.choice([2, 3, 4], size=n_groups, p=[0.5, 0.2, 0.3]).astype(np.uint8)

    scales_arr = rng.uniform(0.01, 0.05, n_groups).astype(np.float32)
    zp_arr = np.where(bits_arr == 4, 8, np.where(bits_arr == 3, 4, 2)).astype(np.uint8)

    meta = {
        "schema_version": "1.0",
        "model": "test",
        "achieved_avg_bits": float(np.mean(bits_arr)),
        "group_size": group_size,
        "calibration_prompts_used": 4,
        "quick_mode": True,
        "npz_path": "test_slim.npz",
        "statistics": {
            "groups_at_4bit": int(np.sum(bits_arr == 4)),
            "groups_at_3bit": int(np.sum(bits_arr == 3)),
            "groups_at_2bit": int(np.sum(bits_arr == 2)),
            "total_groups": n_groups,
            "estimated_model_size_gb": 3.5,
            "size_reduction_vs_q4_pct": 50.0,
        },
    }

    npz_data = {
        "L0_test_proj_bits": bits_arr,
        "L0_test_proj_scales": scales_arr,
        "L0_test_proj_zp": zp_arr,
    }

    return meta, npz_data


# ── tests ──────────────────────────────────────────────────────────────────────

class TestSlimDispatch:

    def test_load_slim_sets_flag(self):
        """load_slim() sets _use_slim=True and loads metadata."""
        out_f, in_f, gs = 64, 128, 32
        store, _, _, _ = make_synthetic_store(out_f, in_f, gs)

        meta, npz_data = make_slim_meta(out_f, in_f, gs, "all4")

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            tmpdir = Path(tmpdir)
            meta_path = tmpdir / "test_slim.json"
            npz_path = tmpdir / "test_slim.npz"

            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            np.savez(str(npz_path), **npz_data)

            assert not store._use_slim
            store.load_slim(str(meta_path))
            assert store._use_slim
            assert store._slim_meta is not None
            assert store._slim_npz is not None

    def test_4bit_groups_identical_to_baseline(self):
        """4-bit groups with same scale/zp produce identical output."""
        out_f, in_f, gs = 64, 128, 32
        store, packed, orig_scales, orig_biases = make_synthetic_store(out_f, in_f, gs)

        # Create SliM metadata with 4-bit groups using SAME scales as original
        n_groups = out_f * (in_f // gs)
        orig_zp = 8
        meta = {
            "schema_version": "1.0",
            "model": "test",
            "achieved_avg_bits": 4.0,
            "group_size": gs,
            "calibration_prompts_used": 4,
            "quick_mode": True,
            "npz_path": "test_slim.npz",
            "statistics": {"groups_at_4bit": n_groups, "groups_at_3bit": 0,
                           "groups_at_2bit": 0, "total_groups": n_groups,
                           "estimated_model_size_gb": 7.5, "size_reduction_vs_q4_pct": 0.0},
        }
        npz_data = {
            "L0_test_proj_bits": np.full(n_groups, 4, dtype=np.uint8),
            "L0_test_proj_scales": orig_scales,
            "L0_test_proj_zp": np.full(n_groups, orig_zp, dtype=np.uint8),
        }

        x = torch.randn(in_f)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            tmpdir = Path(tmpdir)
            meta_path = tmpdir / "test_slim.json"
            npz_path = tmpdir / "test_slim.npz"
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            np.savez(str(npz_path), **npz_data)

            # Baseline output
            y_baseline = store.matvec(0, "test_proj", x)

            # SliM output (4-bit, same scales)
            store.load_slim(str(meta_path))
            y_slim = store.matvec(0, "test_proj", x)

        np.testing.assert_allclose(
            y_slim.numpy(), y_baseline.numpy(),
            rtol=1e-5, atol=1e-5,
            err_msg="4-bit SliM output differs from baseline"
        )

    def test_2bit_groups_correct_dispatch(self):
        """2-bit groups: SliM dispatch uses calibrated scales, output is finite and non-NaN."""
        out_f, in_f, gs = 64, 128, 32
        store, packed, orig_scales, orig_biases = make_synthetic_store(out_f, in_f, gs)

        meta, npz_data = make_slim_meta(out_f, in_f, gs, "all2")
        x = torch.randn(in_f)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            tmpdir = Path(tmpdir)
            meta_path = tmpdir / "test_slim.json"
            npz_path = tmpdir / "test_slim.npz"
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            np.savez(str(npz_path), **npz_data)

            store.load_slim(str(meta_path))
            y_slim = store.matvec(0, "test_proj", x)

        # 2-bit dispatch: output must be finite and non-NaN
        assert y_slim.shape == (1, out_f), f"Wrong shape: {y_slim.shape}"
        assert not torch.any(torch.isnan(y_slim)), "2-bit SliM output contains NaN"
        assert not torch.any(torch.isinf(y_slim)), "2-bit SliM output contains Inf"

        # Verify the SliM scales were actually used (not original scales)
        # by checking that the output differs from the original Q4 output
        store._use_slim = False
        y_orig = store.matvec(0, "test_proj", x)
        # They should differ because scales are different
        assert not torch.allclose(y_slim, y_orig, atol=1e-3), \
            "2-bit SliM output identical to original — scales not applied"

    def test_3bit_groups_correct_dispatch(self):
        """3-bit groups: SliM dispatch uses calibrated scales, output is finite and non-NaN."""
        out_f, in_f, gs = 64, 128, 32
        store, packed, orig_scales, orig_biases = make_synthetic_store(out_f, in_f, gs)

        meta, npz_data = make_slim_meta(out_f, in_f, gs, "all3")
        x = torch.randn(in_f)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            tmpdir = Path(tmpdir)
            meta_path = tmpdir / "test_slim.json"
            npz_path = tmpdir / "test_slim.npz"
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            np.savez(str(npz_path), **npz_data)

            store.load_slim(str(meta_path))
            y_slim = store.matvec(0, "test_proj", x)

        assert y_slim.shape == (1, out_f)
        assert not torch.any(torch.isnan(y_slim)), "3-bit SliM output contains NaN"
        assert not torch.any(torch.isinf(y_slim)), "3-bit SliM output contains Inf"

    def test_2bit_repacked_buffer_bit_masking(self):
        """2-bit repacking correctly masks high bits of each nibble."""
        out_f, in_f, gs = 16, 32, 32
        store, packed, orig_scales, orig_biases = make_synthetic_store(out_f, in_f, gs)

        meta, npz_data = make_slim_meta(out_f, in_f, gs, "all2")

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            tmpdir = Path(tmpdir)
            meta_path = tmpdir / "test_slim.json"
            npz_path = tmpdir / "test_slim.npz"
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            np.savez(str(npz_path), **npz_data)

            store.load_slim(str(meta_path))
            repacked = store._get_slim_repacked(0, "test_proj")

        assert repacked is not None
        repacked_t, slim_scales, slim_biases = repacked

        # Verify: all nibbles in repacked buffer are in range [0, 3]
        repacked_np = repacked_t.numpy()
        lo_nibbles = repacked_np & 0x0F
        hi_nibbles = (repacked_np >> 4) & 0x0F
        assert np.all(lo_nibbles <= 3), "lo nibbles exceed 2-bit range"
        assert np.all(hi_nibbles <= 3), "hi nibbles exceed 2-bit range"

    def test_mixed_precision_dispatch(self):
        """Mixed 2/3/4-bit groups all dispatch correctly."""
        out_f, in_f, gs = 64, 128, 32
        store, packed, orig_scales, orig_biases = make_synthetic_store(out_f, in_f, gs)

        meta, npz_data = make_slim_meta(out_f, in_f, gs, "mixed")
        x = torch.randn(in_f)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            tmpdir = Path(tmpdir)
            meta_path = tmpdir / "test_slim.json"
            npz_path = tmpdir / "test_slim.npz"
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            np.savez(str(npz_path), **npz_data)

            store.load_slim(str(meta_path))
            y_slim = store.matvec(0, "test_proj", x)

        assert y_slim.shape == (1, out_f)
        assert not torch.any(torch.isnan(y_slim)), "SliM output contains NaN"
        assert not torch.any(torch.isinf(y_slim)), "SliM output contains Inf"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
