import os
import json
import struct
import numpy as np
import torch
from pathlib import Path
from safetensors import safe_open
from tqdm import tqdm

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models" / "phi4-multimodal-instruct"
INDEX_FILE = MODEL_DIR / "model.safetensors.index.json"

GROUP_SIZE = 32

def quantize_and_pack_q4_32(tensor: torch.Tensor) -> np.ndarray:
    """Returns a flattened uint8 numpy array containing the 18-byte packed blocks."""
    tensor_np = tensor.float().numpy()
    rows, cols = tensor_np.shape
    
    # Pad columns if necessary (shouldn't be, but just in case)
    pad_cols = (GROUP_SIZE - (cols % GROUP_SIZE)) % GROUP_SIZE
    if pad_cols > 0:
        tensor_np = np.pad(tensor_np, ((0, 0), (0, pad_cols)))
        cols += pad_cols
        
    N = (rows * cols) // GROUP_SIZE
    reshaped = tensor_np.reshape(N, GROUP_SIZE)
    
    # Symmetric quantization
    abs_max = np.max(np.abs(reshaped), axis=1)
    scales = (abs_max / 7.0).astype(np.float16)  # float16 is 2 bytes
    scales[scales == 0] = 1e-5
    
    # Scale and clip
    scaled = np.round(reshaped / scales[:, None].astype(np.float32))
    quantized = np.clip(scaled, -7, 7).astype(np.int8) + 8  # 0 to 15 space
    quantized = quantized.astype(np.uint8)
    
    # Pack 4-bit pairs (2 values per byte)
    lower = quantized[:, 0::2] & 0x0F
    upper = (quantized[:, 1::2] & 0x0F) << 4
    packed = lower | upper  # Shape (N, 16)
    
    # Construct blocks [Scale(2 bytes)] + [Packed(16 bytes)]
    scales_bytes = scales.view(np.uint8).reshape(N, 2)
    blocks = np.concatenate([scales_bytes, packed], axis=1)  # Shape (N, 18)
    
    return blocks.flatten()

def main():
    if not INDEX_FILE.exists():
        print(f"Error: {INDEX_FILE} not found.")
        return
        
    with open(INDEX_FILE) as f:
        idx = json.load(f)["weight_map"]
        
    # Get unique shards
    shard_files = sorted(list(set(idx.values())))
    
    out_bin_path = ROOT / "models" / "phi4_q4_32.bin"
    out_json_path = ROOT / "models" / "phi4_q4_32_metadata.json"
    
    metadata = {}
    current_offset = 0
    
    print(f"Exporting to {out_bin_path}...")
    
    # Open binary for raw append
    with open(out_bin_path, "wb") as f_out:
        for shard in shard_files:
            shard_path = MODEL_DIR / shard
            print(f"Processing {shard}...")
            
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                keys = f.keys()
                # Process in a deterministic order for this shard
                for key in tqdm(sorted(keys)):
                    tensor = f.get_tensor(key)
                    shape = list(tensor.shape)
                    
                    # Quantize all 2D layers except embeddings
                    if len(shape) == 2 and "embed" not in key:
                        dtype = "q4_32"
                        byte_array = quantize_and_pack_q4_32(tensor)
                    else:
                        dtype = "fp16"
                        byte_array = tensor.half().numpy().view(np.uint8).flatten()
                        
                    # Write immediately to disk, discarding the PyTorch tensor
                    f_out.write(byte_array.tobytes())
                    
                    size_bytes = len(byte_array)
                    metadata[key] = {
                        "shape": shape,
                        "dtype": dtype,
                        "offset": current_offset,
                        "size_bytes": size_bytes
                    }
                    current_offset += size_bytes
                    
    # Save the lookup index so our C++ engine can pointer-arithmetic its way to the correct memory offset
    with open(out_json_path, "w") as f_meta:
        json.dump(metadata, f_meta, indent=2)
        
    print("\n==================================")
    print("Export Complete!")
    print(f"Final Total Binary Size: {current_offset / (1024**3):.4f} GB")
    print("==================================")

if __name__ == '__main__':
    main()
