import os
import json
import struct
import numpy as np
import torch
from pathlib import Path
from safetensors import safe_open
from tqdm import tqdm

ROOT = Path(__file__).parent
SNAPSHOT_ID = "932b33c0ec9ca189badeb22480721a8de9d0e006"
MODEL_DIR = Path(f"C:/Users/aarus/.cache/huggingface/hub/models--microsoft--phi-4/snapshots/{SNAPSHOT_ID}")
INDEX_FILE = MODEL_DIR / "model.safetensors.index.json"

GROUP_SIZE = 32
Q4K_BLOCK_BYTES = 140


def repack_q4k_interleaved(
    q4k_row_major_bytes: bytes,
    rows: int,
    cols: int,
    block_k: int = 256,
) -> bytes:
    """Repack Q4_K_M blocks into 4-row interleaved layout.

    Input layout (row-major):
      row0[b0..bn], row1[b0..bn], row2..., row3...

    Output layout (4-row interleaved):
      group0:b0(row0,row1,row2,row3), b1(row0,row1,row2,row3), ...
      group1:...

    Each row-block is Q4K_BLOCK_BYTES (140 bytes).
    """
    if cols % block_k != 0:
        raise ValueError("cols must be divisible by 256 for q4_k_m repack")

    blocks_per_row = cols // block_k
    total_blocks = rows * blocks_per_row
    expected = total_blocks * Q4K_BLOCK_BYTES
    if len(q4k_row_major_bytes) != expected:
        raise ValueError(f"byte length mismatch: got={len(q4k_row_major_bytes)} expected={expected}")

    src = np.frombuffer(q4k_row_major_bytes, dtype=np.uint8).reshape(total_blocks, Q4K_BLOCK_BYTES)
    groups = (rows + 3) // 4

    out_blocks = np.zeros((groups * blocks_per_row * 4, Q4K_BLOCK_BYTES), dtype=np.uint8)

    for g in range(groups):
        for b in range(blocks_per_row):
            dst_base = (g * blocks_per_row + b) * 4
            for lane in range(4):
                r = g * 4 + lane
                if r >= rows:
                    continue
                src_idx = r * blocks_per_row + b
                out_blocks[dst_base + lane, :] = src[src_idx, :]

    return out_blocks.reshape(-1).tobytes()

def permute(weights: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    # Llama style permutation: [Standard HF] -> [Split-half for optimized RoPE]
    # HF: [head_dim] where pairs (0,1), (2,3) etc are rotated
    # Llama: [head_dim] where pairs (i, i+head_dim/2) are rotated
    return (
        weights.view(n_heads, head_dim // 2, 2, -1)
        .transpose(1, 2)
        .reshape(n_heads * head_dim, -1)
    )

def quantize_and_pack_q4_32(tensor: torch.Tensor) -> np.ndarray:
    """Returns a flattened uint8 numpy array containing the 18-byte packed blocks."""
    # tensor.float() handles BF16 -> F32 conversion
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
    # Ensure scales are cast to float32 for computation
    scales_f32 = scales.astype(np.float32)
    scaled = np.round(reshaped / scales_f32[:, None])
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
    
    # OUTPUTS
    out_bin_path = ROOT / "models" / "phi4_14b_q4_32.bin"
    out_json_path = ROOT / "models" / "phi4_14b_q4_32_metadata.json"
    
    metadata = {}
    current_offset = 0
    
    # Phi-4 14B Config for Permutation
    n_heads = 40
    n_kv_heads = 10
    head_dim = 128
    
    print(f"Exporting 14B model to {out_bin_path}...")
    
    # Open binary for raw append
    with open(out_bin_path, "wb") as f_out:
        for shard in shard_files:
            shard_path = MODEL_DIR / shard
            print(f"Processing {shard}...")
            
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                keys = f.keys()
                for key in tqdm(sorted(keys)):
                    tensor = f.get_tensor(key)
                    shape = list(tensor.shape)
                    
                    # PERMUTE Q/K HEADS FOR HF-to-ENGINE COMPATIBILITY
                    if "qkv_proj" in key:
                        # QKV is 7680 total rows. Q(5120), K(1280), V(1280)
                        q = tensor[:5120, :]
                        k = tensor[5120:5120+1280, :]
                        v = tensor[5120+1280:, :]
                        
                        # Permute Q and K
                        q = permute(q, n_heads, head_dim)
                        k = permute(k, n_kv_heads, head_dim)
                        
                        # Concatenate back
                        tensor = torch.cat([q, k, v], dim=0)
                        shape = list(tensor.shape)
                    
                    # Quantize all 2D layers except embeddings and LM head
                    # LM Head is float* in C++ engine
                    if len(shape) == 2 and "embed" not in key and "lm_head" not in key:
                        dtype = "q4_32"
                        byte_array = quantize_and_pack_q4_32(tensor)
                    else:
                        dtype = "fp16"
                        byte_array = tensor.half().numpy().view(np.uint8).flatten()
                        
                    # Write immediately to disk
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
