"""Phase 3 dispatch policy tests (synthetic profiles)."""



from __future__ import annotations



import json

import os

from pathlib import Path



import pytest



from asdsl.dispatch.policy import (

    DispatchPolicy,

    KernelTag,

    ProjectionProfile,

    l2_budget_bytes,

    sparse_min_size,

)

from asdsl.lut.lut_table_builder import LUTTableBuilder





def _profile(

    layer: int = 0,

    name: str = "qkv_proj",

    rows: int = 3072,

    cols: int = 4096,

    bits: int = 4,

    gs: int = 32,

    sparsity: float = 0.0,

    tile_groups: int = 64,

) -> ProjectionProfile:

    return ProjectionProfile(

        layer_idx=layer,

        proj_name=name,

        rows=rows,

        cols=cols,

        bits=bits,

        group_size=gs,

        mean_sparsity=sparsity,

        lut_footprint_bytes=LUTTableBuilder.tile_working_set_bytes(
            cols, gs, tile_groups
        ),

        tile_groups=tile_groups,

    )





def test_lut_when_fits_l2_budget(monkeypatch):

    monkeypatch.setenv("ASDSL_L2_BUDGET_KB", "2048")

    p = _profile(rows=256, cols=512, tile_groups=64)

    assert LUTTableBuilder.tile_working_set_bytes(512, 32, 64) <= l2_budget_bytes()

    assert DispatchPolicy().kernel_for_profile(p) == KernelTag.LUT





def test_avx2_when_lut_footprint_exceeds_budget(monkeypatch):

    monkeypatch.setenv("ASDSL_L2_BUDGET_KB", "512")

    p = _profile(rows=3072, cols=16384, tile_groups=64)

    assert LUTTableBuilder.tile_working_set_bytes(16384, 32, 64) > l2_budget_bytes()

    assert DispatchPolicy().kernel_for_profile(p) == KernelTag.AVX2





def test_sparse_when_high_sparsity_and_large(monkeypatch):

    monkeypatch.setenv("ASDSL_SPARSE_MIN_SIZE", "1000")

    base = _profile(
        rows=2000, cols=2000, name="down_proj", sparsity=0.75, tile_groups=128
    )

    p = ProjectionProfile(

        layer_idx=base.layer_idx,

        proj_name="down_proj",

        rows=base.rows,

        cols=base.cols,

        bits=base.bits,

        group_size=base.group_size,

        mean_sparsity=base.mean_sparsity,

        lut_footprint_bytes=l2_budget_bytes() + 1,

        tile_groups=base.tile_groups,

    )

    assert p.rows * p.cols >= sparse_min_size()

    assert DispatchPolicy().kernel_for_profile(p) == KernelTag.SPARSE





def test_lut_priority_over_sparse(monkeypatch):

    monkeypatch.setenv("ASDSL_L2_BUDGET_KB", "2048")

    p = _profile(rows=256, cols=512, sparsity=0.9, tile_groups=64)

    assert LUTTableBuilder.tile_working_set_bytes(512, 32, 64) <= l2_budget_bytes()

    assert DispatchPolicy().kernel_for_profile(p) == KernelTag.LUT





def test_policy_load_save_roundtrip(tmp_path):

    profiles = {

        (0, "qkv_proj"): _profile(),

        (1, "down_proj"): ProjectionProfile(

            layer_idx=1,

            proj_name="down_proj",

            rows=4096,

            cols=4096,

            bits=4,

            group_size=32,

            mean_sparsity=0.8,

            lut_footprint_bytes=l2_budget_bytes() + 1,

            tile_groups=64,

        ),

    }

    policy = DispatchPolicy(profiles)

    path = tmp_path / "profiles.json"

    policy.save_json(path)

    loaded = DispatchPolicy.load_json(path)

    # Phi-4-scale shapes exceed L2 tile budget; LUT is not selected.
    assert loaded.get_kernel(0, "qkv_proj") == KernelTag.AVX2

    assert loaded.get_kernel(1, "down_proj") == KernelTag.SPARSE





@pytest.mark.skipif(

    not os.environ.get("MODEL_DIR"),

    reason="MODEL_DIR not set; skip integration calibrate",

)

def test_calibrate_integration_smoke():

    pytest.importorskip("transformers")

    from experiments.phi4_cpu_run import WeightStore



    store = WeightStore(bits=4, enable_lut=True)

    store.load()

    store.warm_cache()

    from asdsl.dispatch.calibrate import calibrate



    policy = calibrate(store, tokens=[0, 1, 2, 3], max_tokens=4)

    assert len(policy.profiles) > 0


