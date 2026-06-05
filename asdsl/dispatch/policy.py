"""Kernel dispatch policy for Phase 3 LUT / AVX2 / SPARSE routing."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from asdsl.lut.lut_table_builder import LUTTableBuilder


class KernelTag(str, Enum):
    LUT = "LUT"
    AVX2 = "AVX2"
    SPARSE = "SPARSE"


PHI4_PROJECTIONS = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")


def l2_budget_bytes() -> int:
    kb = int(os.environ.get("ASDSL_L2_BUDGET_KB", "384"))
    return kb * 1024


def sparse_min_size() -> int:
    return int(os.environ.get("ASDSL_SPARSE_MIN_SIZE", "2000000"))


@dataclass(frozen=True)
class ProjectionProfile:
    layer_idx: int
    proj_name: str
    rows: int
    cols: int
    bits: int
    group_size: int
    mean_sparsity: float
    lut_footprint_bytes: int
    tile_groups: int = LUTTableBuilder.DEFAULT_TILE_GROUPS

    @property
    def key(self) -> tuple[int, str]:
        return (self.layer_idx, self.proj_name)

    def to_dict(self, policy: "DispatchPolicy | None" = None) -> dict[str, Any]:
        d: dict[str, Any] = {
            "layer_idx": self.layer_idx,
            "proj_name": self.proj_name,
            "rows": self.rows,
            "cols": self.cols,
            "bits": self.bits,
            "group_size": self.group_size,
            "mean_sparsity": self.mean_sparsity,
            "lut_footprint_bytes": self.lut_footprint_bytes,
            "tile_groups": self.tile_groups,
        }
        if policy is not None:
            d["kernel"] = policy.kernel_for_profile(self).value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProjectionProfile:
        return cls(
            layer_idx=int(d["layer_idx"]),
            proj_name=str(d["proj_name"]),
            rows=int(d["rows"]),
            cols=int(d["cols"]),
            bits=int(d.get("bits", 4)),
            group_size=int(d.get("group_size", 32)),
            mean_sparsity=float(d.get("mean_sparsity", 0.0)),
            lut_footprint_bytes=int(
                d.get(
                    "lut_footprint_bytes",
                    LUTTableBuilder.tile_working_set_bytes(
                        int(d["cols"]),
                        int(d.get("group_size", 32)),
                        int(d.get("tile_groups", 64)),
                    ),
                )
            ),
            tile_groups=int(d.get("tile_groups", LUTTableBuilder.DEFAULT_TILE_GROUPS)),
        )


class DispatchPolicy:
    """Assign KernelTag per projection from calibration profiles."""

    def __init__(
        self,
        profiles: dict[tuple[int, str], ProjectionProfile] | None = None,
        *,
        l2_budget: int | None = None,
        sparse_min: int | None = None,
    ):
        self.profiles: dict[tuple[int, str], ProjectionProfile] = profiles or {}
        self.l2_budget = l2_budget if l2_budget is not None else l2_budget_bytes()
        self.sparse_min = sparse_min if sparse_min is not None else sparse_min_size()

    def kernel_for_profile(self, profile: ProjectionProfile) -> KernelTag:
        tile_fp = LUTTableBuilder.tile_working_set_bytes(
            profile.cols, profile.group_size, profile.tile_groups
        )
        if (
            tile_fp <= self.l2_budget
            and profile.bits == 4
            and profile.group_size == 32
        ):
            return KernelTag.LUT
        if (
            profile.mean_sparsity >= 0.60
            and profile.rows * profile.cols >= self.sparse_min
        ):
            return KernelTag.SPARSE
        return KernelTag.AVX2

    def get_kernel(self, layer_idx: int, proj_name: str) -> KernelTag:
        key = (layer_idx, proj_name)
        if key not in self.profiles:
            return KernelTag.AVX2
        return self.kernel_for_profile(self.profiles[key])

    def assignment_table(self) -> dict[str, str]:
        return {
            f"L{layer}:{name}": self.get_kernel(layer, name).value
            for (layer, name) in sorted(self.profiles.keys())
        }

    @classmethod
    def load_json(cls, path: Path | str) -> DispatchPolicy:
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        profiles: dict[tuple[int, str], ProjectionProfile] = {}
        entries = raw.get("profiles", raw)
        if isinstance(entries, list):
            for item in entries:
                p = ProjectionProfile.from_dict(item)
                profiles[p.key] = p
        elif isinstance(entries, dict):
            for key, item in entries.items():
                if isinstance(key, str) and ":" in key:
                    layer_s, name = key.split(":", 1)
                    item = {**item, "layer_idx": int(layer_s), "proj_name": name}
                p = ProjectionProfile.from_dict(item)
                profiles[p.key] = p
        return cls(
            profiles,
            l2_budget=int(raw["l2_budget_bytes"]) if "l2_budget_bytes" in raw else None,
            sparse_min=int(raw["sparse_min_size"]) if "sparse_min_size" in raw else None,
        )

    def save_json(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "l2_budget_bytes": self.l2_budget,
            "sparse_min_size": self.sparse_min,
            "profiles": [p.to_dict(self) for p in sorted(
                self.profiles.values(),
                key=lambda x: (x.layer_idx, x.proj_name),
            )],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
