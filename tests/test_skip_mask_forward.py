"""Skip-mask API and default mask shape."""

from __future__ import annotations

import numpy as np

from asdsl.inference.unified_bridge import (
    clear_skip_mask,
    default_ahsd_skip_mask,
    get_or_build_unified_engine,
    set_skip_mask,
)


def test_default_mask_skips_layers_10_to_21() -> None:
    mask = default_ahsd_skip_mask(32)
    assert mask.shape == (32,)
    assert not mask[:10].any()
    assert mask[10:22].all()
    assert not mask[22:].any()


def test_set_skip_mask_roundtrip(phi4_store) -> None:
    eng = get_or_build_unified_engine(phi4_store)
    eng.reset_session()
    mask = default_ahsd_skip_mask()
    set_skip_mask(eng, mask)
    clear_skip_mask(eng)
    # Full forward still runs after mask cleared.
    logits = eng.forward_token(42, 0)
    assert logits is not None
    assert int(np.argmax(logits)) >= 0


def test_no_skip_draft_matches_full(phi4_store) -> None:
    """With empty skip mask, draft and full single-token forwards should match."""
    eng = get_or_build_unified_engine(phi4_store)
    eng.reset_session()
    clear_skip_mask(eng)
    full = np.asarray(eng.forward_token(42, 0), dtype=np.float32)
    eng.reset_session()
    clear_skip_mask(eng)
    draft = np.asarray(eng.forward_token_draft(42, 0), dtype=np.float32)
    np.testing.assert_allclose(full, draft, rtol=0, atol=0)
