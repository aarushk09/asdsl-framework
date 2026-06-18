"""Adaptive skip mask respects first/last guard bands."""

from __future__ import annotations

import numpy as np

from asdsl.speculative.ahsd import compute_skip_mask


def test_calibration_respects_guard_bands(phi4_store) -> None:
    mask = compute_skip_mask(
        phi4_store,
        calibration_tokens=16,
        threshold=0.97,
        skip_first=8,
        skip_last=8,
    )
    assert mask.shape == (32,)
    assert not mask[:8].any()
    assert not mask[-8:].any()
    assert mask[8:24].any()


def test_cached_mask_on_store(phi4_store) -> None:
    from asdsl.speculative.ahsd import calibrate_and_store_skip_mask

    mask = calibrate_and_store_skip_mask(phi4_store)
    assert np.array_equal(phi4_store._ahsd_skip_mask, mask)
