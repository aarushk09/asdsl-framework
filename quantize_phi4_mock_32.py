import json
import os

dim = 5120
hidden_dim = 17920
vocab_size = 100352
num_layers = 40

def q4_size(rows, cols):
    return rows * (cols // 32) * 18

print("Generating mock layout for Phi-4 14B Q4_32 (new unified format)...")
tensors = {}
offset = 0

size = vocab_size * dim * 2
tensors["model.embed_tokens.weight"] = {"offset": offset, "size_bytes": size, "dtype": "fp16", "shape": [vocab_size, dim]}
offset += size

for l in range(num_layers):
    pfx = f"model.layers.{l}."
    
    size = dim * 2
    tensors[pfx+"input_layernorm.weight"] = {"offset": offset, "size_bytes": size, "dtype": "fp16", "shape": [dim]}
    offset += size
    tensors[pfx+"post_attention_layernorm.weight"] = {"offset": offset, "size_bytes": size, "dtype": "fp16", "shape": [dim]}
    offset += size

    sz = q4_size(5120 + 1280 + 1280, dim)
    tensors[pfx+"self_attn.qkv_proj.base_layer.weight"] = {"offset": offset, "size_bytes": sz, "dtype": "q4_32", "shape": [7680, dim]}
    offset += sz

    sz = q4_size(dim, dim)
    tensors[pfx+"self_attn.o_proj.base_layer.weight"] = {"offset": offset, "size_bytes": sz, "dtype": "q4_32", "shape": [dim, dim]}
    offset += sz

    sz = q4_size(hidden_dim * 2, dim)
    tensors[pfx+"mlp.gate_up_proj.base_layer.weight"] = {"offset": offset, "size_bytes": sz, "dtype": "q4_32", "shape": [hidden_dim * 2, dim]}
    offset += sz

    sz = q4_size(dim, hidden_dim)
    tensors[pfx+"mlp.down_proj.base_layer.weight"] = {"offset": offset, "size_bytes": sz, "dtype": "q4_32", "shape": [dim, hidden_dim]}
    offset += sz

size = dim * 2
tensors["model.norm.weight"] = {"offset": offset, "size_bytes": size, "dtype": "fp16", "shape": [dim]}
offset += size

bin_file = "models/phi4_q4_32.bin"
print(f"Total projected Mmap size: {offset / (1024**3):.2f} GB")

with open(bin_file, "wb") as f:
    f.seek(offset - 1)
    f.write(b'\0')

with open("models/phi4_q4_32_metadata.json", "w") as f:
    json.dump(tensors, f, indent=4)

print("Done. Metadata and Mock Bin sequence generated.")
