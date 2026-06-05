"""Compute llama.cpp effective weight bytes per decode token from GGUF."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GGUF = ROOT / "models" / "llama_cpp" / "phi4-mm-Q4_K_M.gguf"


def main() -> int:
    try:
        import gguf
    except ImportError:
        print("pip install gguf", file=sys.stderr)
        return 1

    reader = gguf.GGUFReader(str(GGUF))
    proj_names = ["attn_qkv", "attn_output", "ffn_up", "ffn_down"]
    norm_names = ["attn_norm", "ffn_norm"]
    num_layers = 32

    total_proj_bytes = 0
    total_norm_bytes = 0
    per_layer: dict[str, float] = {}

    for t in reader.tensors:
        for pname in proj_names:
            if t.name == f"blk.0.{pname}.weight":
                bpl = int(t.data.nbytes)
                total_proj_bytes += bpl * num_layers
                per_layer[pname] = bpl / 1e6
        for nname in norm_names:
            if t.name == f"blk.0.{nname}.weight":
                total_norm_bytes += int(t.data.nbytes) * num_layers

    for t in reader.tensors:
        if t.name in ("output.weight", "output_norm.weight"):
            nbytes = int(t.data.nbytes)
            total_proj_bytes += nbytes
            per_layer[t.name] = nbytes / 1e6

    # Phi-4 GGUF ties lm_head to token_embd (no output.weight); use block formula from phase21.
    lm_head_bytes = 200_019 * (3072 // 256) * 210
    total_matvec_bytes = total_proj_bytes + lm_head_bytes

    # Analytical cross-check (Q4_K block sizes from phase21_measure)
    def qb(rows: int, cols: int, block: int) -> int:
        return num_layers * rows * (cols // 256) * block

    analytical = (
        qb(16384, 3072, 144)
        + qb(3072, 3072, 144)
        + qb(5120, 3072, 176)
        + qb(3072, 8192, 210)
        + lm_head_bytes
    )

    gb_per_token = total_matvec_bytes / 1e9
    out = {
        "gguf": str(GGUF),
        "per_layer_mb": per_layer,
        "layer_proj_bytes_gguf": total_proj_bytes,
        "lm_head_bytes_analytical": lm_head_bytes,
        "total_matvec_bytes": total_matvec_bytes,
        "total_norm_bytes": total_norm_bytes,
        "analytical_matvec_bytes": analytical,
        "gb_per_token": gb_per_token,
        "at_15_tok_s_gb_s": gb_per_token * 15.0,
    }
    print(json.dumps(out, indent=2))
    path = ROOT / "benchmarks" / "results" / "phase27_llama_bytes.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
