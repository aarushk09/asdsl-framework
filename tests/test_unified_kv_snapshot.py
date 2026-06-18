"""UnifiedEngine KV snapshot / restore preserves committed length."""

from __future__ import annotations

from asdsl.inference.unified_bridge import (
    get_kv_seq_len,
    get_or_build_unified_engine,
    restore_kv,
    snapshot_kv,
)


def test_snapshot_restore_length(phi4_store) -> None:
    eng = get_or_build_unified_engine(phi4_store)
    eng.reset_session()

    for pos, tid in enumerate(range(10)):
        eng.forward_token(int(tid), int(pos))

    base_len = get_kv_seq_len(eng)
    assert base_len == 10

    snapshot_kv(eng)
    for k in range(3):
        eng.forward_token_draft(100 + k, base_len + k)

    assert get_kv_seq_len(eng) == 13
    restore_kv(eng)
    assert get_kv_seq_len(eng) == base_len
