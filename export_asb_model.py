import sys
import struct
import numpy as np
import torch
import json
from pathlib import Path
from safetensors import safe_open
from tqdm import tqdm

"""
Step 4: Mixed Precision ASB serialization.
Writes files perfectly matching the C++ Adaptive Salience Block specification:
- 8-byte header: (bit_width : uint8, group_size : uint8, scale : fp16, zero : fp16, reserved : 2 bytes)
- Followed by data bytes.
"""

def compute_group(tensor_slice, bit_width=4):
    """
    Computes a single ASB group block and returns (header_bytes, data_bytes)
    """
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
    
    # packing logic depends on bit_width
    if bit_width == 4:
        assert gsize % 2 == 0
        packed = (q[1::2] << 4) | (q[0::2] & 0x0F)
        data = packed.tobytes()
    elif bit_width == 8:
        data = q.tobytes()
    elif bit_width == 2:
        assert gsize % 4 == 0
        packed = (q[3::4] << 6) | (q[2::4] << 4) | (q[1::4] << 2) | (q[0::4] & 0x03)
        data = packed.tobytes()
    else:
        raise ValueError(f"Unsupported bit width: {bit_width}")
        
    return header, data

def convert_to_asb(in_dir, out_path):
    import os
    index_file = Path(in_dir) / "model.safetensors.index.json"
    if not index_file.exists():
        print(f"Error: {index_file} not found.")
        return

    with open(index_file) as f:
        idx = json.load(f)["weight_map"]

    shard_files = sorted(list(set(idx.values())))
    out_json_path = out_path.replace('.bin', '_metadata.json')

    metadata = {}
    current_offset = 0

    print(f"Exporting ASB to {out_path}...")

    # Cycle through 8, 4, 2 bit widths to test branch prediction
    bit_cycle = [8, 4, 2, 4]

    with open(out_path, "wb") as f_out:
        for shard in shard_files:
            shard_path = Path(in_dir) / shard
            print(f"Processing {shard}...")

            with safe_open(shard_path, framework="pt", device="cpu") as f:
                keys = f.keys()
                for key in tqdm(sorted(keys)):
                    tensor = f.get_tensor(key)
                    shape = list(tensor.shape)

                    # Quantize all 2D layers except embeddings
                    if len(shape) == 2 and "embed" not in key:
                        dtype = "asb_mixed"
                        rows, cols = tensor.shape
                        group_size = 32
                        
                        pad_cols = (group_size - (cols % group_size)) % group_size
                        if pad_cols > 0:
                            tensor = torch.nn.functional.pad(tensor, (0, pad_cols))
                            cols += pad_cols
                            
                        # Process group by group
                        tensor_reshaped = tensor.view(-1, group_size)
                        num_groups = tensor_reshaped.shape[0]
                        
                        byte_array = bytearray()
                        
                        for i in range(num_groups):
                            bw = bit_cycle[i % len(bit_cycle)]
                            slice_data = tensor_reshaped[i]
                            header, data = compute_group(slice_data, bit_width=bw)
                            byte_array.extend(header)
                            byte_array.extend(data)
                            
                        final_bytes = bytes(byte_array)
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
                    }
                    current_offset += size_bytes

    with open(out_json_path, "w") as jf:
        json.dump(metadata, jf, indent=2)

    print(f"Done! {out_path} ({current_offset} bytes)")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python export_asb_model.py <model_dir> <out_bin>")
        sys.exit(1)
    convert_to_asb(sys.argv[1], sys.argv[2])
