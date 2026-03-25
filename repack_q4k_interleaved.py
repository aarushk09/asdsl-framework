import json
from pathlib import Path

import numpy as np

Q4K_BLOCK_BYTES = 140
BLOCK_K = 256
INTERLEAVE_ROWS = 4


def repack_q4k_interleaved(row_major: bytes, rows: int, cols: int) -> bytes:
    if cols % BLOCK_K != 0:
        raise ValueError("cols must be divisible by 256")
    blocks_per_row = cols // BLOCK_K
    total_blocks = rows * blocks_per_row
    expected = total_blocks * Q4K_BLOCK_BYTES
    if len(row_major) != expected:
        raise ValueError(f"invalid byte length: {len(row_major)} != {expected}")

    src = np.frombuffer(row_major, dtype=np.uint8).reshape(total_blocks, Q4K_BLOCK_BYTES)
    groups = (rows + INTERLEAVE_ROWS - 1) // INTERLEAVE_ROWS
    dst = np.zeros((groups * blocks_per_row * INTERLEAVE_ROWS, Q4K_BLOCK_BYTES), dtype=np.uint8)

    for g in range(groups):
        for b in range(blocks_per_row):
            base = (g * blocks_per_row + b) * INTERLEAVE_ROWS
            for lane in range(INTERLEAVE_ROWS):
                r = g * INTERLEAVE_ROWS + lane
                if r >= rows:
                    continue
                src_idx = r * blocks_per_row + b
                dst[base + lane] = src[src_idx]

    return dst.reshape(-1).tobytes()


def main() -> None:
    src_bin = Path("models/phi4_q4.bin")
    src_meta = Path("models/phi4_q4_metadata.json")
    out_bin = Path("models/phi4_q4_i4.bin")
    out_meta = Path("models/phi4_q4_i4_metadata.json")

    if not src_bin.exists() or not src_meta.exists():
        raise FileNotFoundError("Missing models/phi4_q4.bin or models/phi4_q4_metadata.json")

    with src_meta.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    raw = src_bin.read_bytes()
    new_meta = {}
    out = bytearray()

    for name, info in meta.items():
        off = int(info["offset"])
        size = int(info["size_bytes"])
        dtype = str(info.get("dtype", ""))
        shape = info.get("shape", [])

        chunk = raw[off : off + size]
        new_off = len(out)

        if dtype == "q4_k_m" and len(shape) == 2:
            rows, cols = int(shape[0]), int(shape[1])
            chunk = repack_q4k_interleaved(chunk, rows, cols)
            dtype = "q4_k_m_i4"

        out.extend(chunk)
        new_meta[name] = {
            "shape": shape,
            "dtype": dtype,
            "offset": new_off,
            "size_bytes": len(chunk),
        }

    out_bin.write_bytes(bytes(out))
    with out_meta.open("w", encoding="utf-8") as f:
        json.dump(new_meta, f, indent=2)

    print(f"Wrote {out_bin} ({len(out) / (1024**3):.3f} GB)")
    print(f"Wrote {out_meta}")


if __name__ == "__main__":
    main()
