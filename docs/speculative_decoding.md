# Speculative Decoding (AHSD / QCSD / SDQS)

ASDSL implements lossless speculative decoding on the Phi-4 UnifiedEngine path.
All speculative modes preserve greedy Q4 output when verification uses the full
32-layer model.

## Modes

| Mode | Entry | Draft | Verify | Extra weights |
|------|-------|-------|--------|---------------|
| **C0 AR** | `phi4_cpu_run.py` | — | full Q4 | none |
| **QCSD** | `--qcsd` | Q2 Python GEMV | batched Q4 | Q2 draft bank |
| **AHSD** | `ASDSL_USE_AHSD=1` | layer-skipped Q4 draft | batched Q4 | none |
| **SDQS** | `ASDSL_USE_SDQS=1` | Q2 draft + skip | batched Q4 (overlapped) | Q2 draft bank |

## Architecture

```mermaid
sequenceDiagram
    participant Loop as AHSD_Loop
    participant Draft as forward_token_draft
    participant KV as KVCache
    participant Verify as forward_batch

    Loop->>KV: snapshot_kv
    loop K draft steps
        Loop->>Draft: skip layers 10-21
        Draft->>KV: tentative writes
    end
    Loop->>KV: restore_kv
    Loop->>Verify: forward_verify_serial (full Q4)
    Verify->>KV: commit accepted
    Loop->>KV: truncate_kv on partial accept
```

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ASDSL_USE_AHSD` | `0` | Route `generate_stream` to UnifiedEngine AHSD |
| `ASDSL_AHSD_DRAFT_K` | `1` | Draft chain length |
| `ASDSL_AHSD_THRESH` | `0.97` | Cosine threshold for adaptive skip mask |
| `ASDSL_AHSD_SKIP_FIRST` | `8` | Never skip first N layers |
| `ASDSL_AHSD_SKIP_LAST` | `8` | Never skip last N layers |
| `ASDSL_USE_SDQS` | `0` | Enable SDQS dual-stream Q2+Q4 |
| `ASDSL_AHSD_CALIBRATE` | `0` | Run adaptive skip-mask calibration (static layers 10–21 when off) |
| `ASDSL_SPECULATIVE_PROFILE` | `0` | Print parseable telemetry line |

C0 parity runs keep all speculative flags **off**.

C0 kernel parity sets `ASDSL_GEMV_UNROLL=4` (hard default in `parity_manifest.json`). Fused GEMV defaults to 4-row; 8-row runs only when `ASDSL_GEMV_UNROLL=8` is set explicitly.

### Phase G regression postmortem (2026-06-05)

Phase G enabled a 4-group fused inner loop on **all** projections by default and auto-selected 8-row unroll for `out_features ≥ 8192`. Kernel microbench looked faster, but C0 E2E regressed (~9.05 → ~8.15 tok/s cold; gate_up profile ~41 ms → ~49 ms).

**Root cause:** (1) `preq_tile_accumulate_g4` on gate_up (16384 rows) added OMP partial-tile overhead without E2E benefit—microbench “before” measured single-group fallback, not the already-fast 4-row path. (2) 8-row auto for large projections caused static-schedule imbalance (`16384 / (12×8)` leaves remainder rows per thread). (3) Even with g4 off, the Phase G xloaded single-group loop differed from the pre-G per-row path.

**Fix:** Restore pre-G g4 gating (`ASDSL_PREQ_G4FUSED=0` → no g4 on parity path); remove 8-row auto (default 4-row); use classic `preq_row_accumulate_one_group` when g4 is off. Optional `ASDSL_PREQ_G4GATE_UP_ONLY=1` enables g4 only on gate_up for experiments.

**After fix (warm, 12t, parity env):** gate_up ~44 ms, total ~109 ms, decode ~8.5–8.6 tok/s on best runs (thermal variance remains high on laptop).

**Phase G net-neutral (2026-06-05, post unified-tail fix):** g4 tile and 8-row unroll remain gated behind `ASDSL_PREQ_G4FUSED=1` and `ASDSL_GEMV_UNROLL=8` respectively. Parity C0 uses classic `preq_row_accumulate_one_group` only (diagnostic: 0 xloaded calls on gate_up). Engine profile token 3: gate_up **~40–43 ms/layer** (÷32 from cumulative), total **~95–103 ms/layer** — matches pre-Phase-G baseline. `phi4_cpu_run` now defaults `ASDSL_PREQ_PREFETCH_GROUPS=0` and `ASDSL_GEMV_UNROLL=4`.

**Next IPC target:** QKV block prefetch (`ASDSL_PREQ_PREFETCH_GROUPS`); probe on this hardware shows groups=2 **regresses** qkv (~13 ms vs ~18 ms/layer) — keep parity at 0. lm_head 4-bit microbench remains optional follow-up.

## Parseable telemetry

Speculative runs print a single regex-friendly line:

```
acceptance_rate=0.8500 draft_tokens=21 verify_ms=123.45 draft_ms=45.67 speculative_cycles=3
```

`parity_benchmark.py` and `benchmarks/speculative_baseline.py` parse this line.

## Benchmarks

```bash
# Unified baseline (C0, QCSD, SWIFT, AHSD)
python benchmarks/speculative_baseline.py --phase 0

# Gate A2 dual-stream probe
python benchmarks/measure_dual_stream.py

# Parity config C4 (AHSD)
python benchmarks/parity_benchmark.py --config C4 --runs 1 --asdsl-only
```

## Tests

```bash
pytest tests/test_ahsd_lossless.py tests/test_qcsd_lossless.py \
  tests/test_unified_kv_snapshot.py tests/test_skip_mask_forward.py -q

# Q2 kernel (synthetic, no weights)
pytest benchmarks/test_q2_correctness.py -q
```

## Gates (Phase A)

| Gate | Criterion |
|------|-----------|
| A1 | Q2 decode ≥ 1.4× Q4 |
| A2 | Dual-stream ≥ 35 GB/s combined, ≥ 80% retention per stream |
| A3 | α ≥ 0.72 at K=3 on parity prompt |
| A4 | Verify at K=3 ≤ 1.15× single-token verify |

Results: `benchmarks/results/speculative_phaseA_gates.json`

## Parity config C4

Defined in `benchmarks/results/parity_manifest.json`:

- HF preq gs=32 (same weights as C0)
- `ASDSL_USE_AHSD=1`, `ASDSL_AHSD_DRAFT_K=1`
- Static skip mask layers 10–21 during draft

C4 is **not** the canonical quality headline; C0 remains the primary parity config.

## Error compounding (why per-layer skip fails)

Single-layer skip probes can show high hidden-state cosine (e.g. 29/32 layers above 0.97 on Phi-4) while a **combined** static skip mask still breaks greedy decoding. Hidden states compose through 32 layers; small per-layer drift accumulates. A block of 12 skipped layers (10–21) can yield high logit cosine (~0.97) yet a different argmax token — lossless verify recovers correctness, but draft acceptance collapses (α≈15%). **High per-layer similarity does not imply block skip is safe** for greedy AR parity.

## Diagnostic results (Case 2)

Phi-4-multimodal-instruct Q4_32 preq, i7-1360P (12 threads), prompt `"The"`, 2026-06-05.
Sources: `benchmarks/results/ahsd_diagnostic.json`, `benchmarks/results/layer_similarity_profile.json`.

**Verdict:** AHSD with static skip mask layers 10–21 is **not viable** for throughput (Case 2: high per-layer similarity, low combined acceptance).

| Metric | Value |
|--------|-------|
| Acceptance rate α | 15.4% |
| AHSD decode | 6.03 tok/s |
| C0 AR decode | 8.41 tok/s |
| Draft ms/cycle | 78.4 |
| Verify ms/cycle | 110.4 |
| Break-even α (K=1) | >0.62 |

- **Skip mask honored:** Draft path ~55% of full forward (microbench 89 ms vs 160 ms full); per-cycle draft ~78 ms vs verify ~110 ms matches skipping 12/32 layers.
- **Layer geometry:** 29/32 layers exceed 0.97 cosine on single-layer skip probes, but combined mask 10–21 yields logit cosine 0.971 with greedy argmax mismatch (tokens 200020 vs 382).
- **Lossless but slower:** AHSD preserves greedy Q4 output on full verify; end-to-end it is slower than C0 AR at current α.
- **C4 parity:** Not recommended for headline speed until α>0.62.
- **Next step:** Pivot to Phase G C0 IPC (batched verify / GEMM amortization), not wider static skip masks.
