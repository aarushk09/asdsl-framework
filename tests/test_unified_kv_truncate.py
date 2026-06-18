"""UnifiedEngine truncate_kv commits partial acceptance length."""

from __future__ import annotations

from asdsl.inference.unified_bridge import (
    get_kv_seq_len,
    get_or_build_unified_engine,
    snapshot_kv,
    truncate_kv,
)


def test_truncate_after_partial_draft(phi4_store) -> None:
    eng = get_or_build_unified_engine(phi4_store)
    eng.reset_session()

    for pos, tid in enumerate(range(8)):
        eng.forward_token(int(tid), int(pos))

    committed = get_kv_seq_len(eng)
    snapshot_kv(eng)
    for k in range(4):
        eng.forward_token_draft(200 + k, committed + k)

    assert get_kv_seq_len(eng) == committed + 4
    truncate_kv(eng, committed + 2)
    assert get_kv_seq_len(eng) == committed + 2
