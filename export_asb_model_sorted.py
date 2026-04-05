import sys
import struct
import numpy as np
import torch
import json
from pathlib import Path
from safetensors import safe_open
from tqdm import tqdm

"""
Step 4: Mixed Precision ASB serialization (Sorted Group Layout).
Because benchmark proved a 52% penalty for random bit-width branching, 
we sort the ASB groups per row by bit_width to ensure inner-loop predictability.
We output the sorted ASB blocks, and a matching uint16 gathering permutation vector 
that C++ will use to permute the input activation cache 'x'.
"""

def compute_group(tensor_slice, bit_width=4):
    gsize = tensor_slice.shape[0]
    tensor_np = tensor_slice.float().numpy()
    
    _min = float(tensor_np.min())
    _max = float(tensor_np.max())
    
    if _max - _min < 1e-8:
        scale, zero = 0.0, _min
    else:
        levels = (1 << bit_width) - 1
        scale = (_max - _min) / levels
        zero = _min
        
    q = np.clip(np.round((tensor_np - zero) / max(scale, 1e-12)), 0, levels).astype(np.uint8)
    
    header = struct.pack('<B B e e 2s', bit_width, gsize, scale, zero, b'\x00\x00')
    
    if bit_width == 4:
        packed = (q[1::2] << 4) | (q[0::2] & 0x0F)
        data = packed.tobytes()
    elif bit_width == 8:
        data = q.tobytes()
    elif bit_width == 3:
        assert False, "Q3 packing logic unverified"
    elif bit_width == 2:
        packed = (q[3::4] << 6) | (q[2::4] << 4) | (q[1::4] << 2) | (q[0::4] & 0x03)
        data = packed.tobytes()
    else:
        raise ValueError(f"Unsupported bit width: {bit_width}")
        
    return header, data

def convert_to_asb_sorted(in_dir, out_path_bin):
    import os
    index_file = Path(in_dir) / "model.safetensors.index.json"
    if not index_file.exists():
        print(f"Error: {index_file} not found.")
        return

    with open(index_file) as f:
        idx = json.load(f)["weight_map"]

    shard_files = sorted(list(set(idx.values())))
    out_json_path = out_path_bin.replace('.bin', '_metadata.json')

    metadata = {}
    current_offset = 0

    print(f"Exporting Sorted ASB to {out_path_bin}...")
    
    # We will use mock salience distribution target 2.5 avg bits (e.g. 10% 8-bit, 20% 4-bit, 70% 2-bit)
    np.random.seed(42)

    with open(out_path_bin, "wb") as f_out:
        for shard in shard_files:
            shard_path = Path(in_dir) / shard
            print(f"Processing {shard}...")

            with safe_open(shard_path, framework="pt", device="cpu") as f:
                keys = f.keys()
                for key in tqdm(sorted(keys)):
                    tensor = f.get_tensor(key)
                    shape = list(tensor.shape)

                    if len(shape) == 2 and "embed" not in key:
                        dtype = "asb_mixed_sorted"
                        rows, cols = tensor.shape
                        group_size = 32
                        
                        pad_cols = (group_size - (cols % group_size)) % group_size
                        if pad_cols > 0:
                            tensor = torch.nn.functional.pad(tensor, (0, pad_cols))
                            cols += pad_cols
                            
                        # Process row by row
                        byte_array = bytearray()

                        perm_array = bytearray()

                        row_offsets = bytearray()


                        num_groups_per_row = cols // group_size

                        current_row_offset = 0


                        for r in range(rows):

                            row_offsets.extend(struct.pack('<I', current_row_offset))

                            row_tensor = tensor[r].view(num_groups_per_row, group_size)


                            # 1. Assign widths (mocked salience)

                            group_infos = []

                            for g in range(num_groups_per_row):

                                rnd = np.random.rand()

                                bw = 8 if rnd < 0.1 else (4 if rnd < 0.3 else 2) # avg = 2.8 bits

                                group_infos.append( (bw, g, row_tensor[g]) )


                            # 2. Sort groups in this row by descending bit width

                            group_infos.sort(key=lambda x: x[0], reverse=True)


                            # 3. Compute payloads and store permutation map (uint16)

                            for bw, g_idx, slice_data in group_infos:

                                perm_array.extend(struct.pack('<H', g_idx))

                                header, data = compute_group(slice_data, bit_width=bw)

                                byte_array.extend(header)

                                byte_array.extend(data)

                                current_row_offset += len(header) + len(data)


                        # Payload arrangement: [RowOffsets: rows * 4 bytes] + [Permutations: rows * num_groups_per_row * 2 bytes] + [ASB Payload...]

                        final_bytes = bytes(row_offsets) + bytes(perm_array) + bytes(byte_array)
                    else:
                        dtype = "fp16"
                        final_bytes = tensor.half().numpy().view(np.uint8).flatten().tobytes()

                    f_out.write(final_bytes)
                    size_bytes = len(final_bytes)
                    metadata[key] = {
                        "shape": shape,
                        "dtype": dtype,
                        "offset": current_offset,
                        "size_bytes": size_bytes,
                        "num_groups_per_row": cols // 32 if dtype == "asb_mixed_sorted" else 0
                    }
                    current_offset += size_bytes

    with open(out_json_path, "w") as jf:
        json.dump(metadata, jf, indent=2)

    print(f"Done! {out_path_bin} ({current_offset} bytes)")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python export_asb_model_sorted.py <model_dir> <out_bin>")
        sys.exit(1)
    convert_to_asb_sorted(sys.argv[1], sys.argv[2])
