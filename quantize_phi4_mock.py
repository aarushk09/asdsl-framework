import json
import os

dim = 5120
hidden_dim = 17920
vocab_size = 100352
num_layers = 40

def q4_size(rows, cols):
    # each 32 cols -> 18 bytes
    return rows * (cols // 32) * 18

print("Generating mock layout for Phi-4 14B...")
tensors = {}
offset = 0

# embed: fp32
size = vocab_size * dim * 4
tensors["embed"] = {"offset": offset, "size_bytes": size, "dtype": "fp32", "shape": [vocab_size, dim]}
offset += size

for l in range(num_layers):
    pfx = f"l{l}_"
    
    # rms fp32
    size = dim * 4
    tensors[pfx+"rms1"] = {"offset": offset, "size_bytes": size, "dtype": "fp32", "shape": [dim]}
    offset += size
    tensors[pfx+"rms2"] = {"offset": offset, "size_bytes": size, "dtype": "fp32", "shape": [dim]}
    offset += size
    
    # Q4 layers
    sz = q4_size(5120 + 1280 + 1280, dim)
    tensors[pfx+"qkv"] = {"offset": offset, "size_bytes": sz, "dtype": "q4", "shape": [7680, dim]}
    offset += sz
    
    sz = q4_size(dim, dim)
    tensors[pfx+"o"] = {"offset": offset, "size_bytes": sz, "dtype": "q4", "shape": [dim, dim]}
    offset += sz
    
    sz = q4_size(hidden_dim, dim)
    tensors[pfx+"gate"] = {"offset": offset, "size_bytes": sz, "dtype": "q4", "shape": [hidden_dim, dim]}
    offset += sz
    tensors[pfx+"up"] = {"offset": offset, "size_bytes": sz, "dtype": "q4", "shape": [hidden_dim, dim]}
    offset += sz
    
    sz = q4_size(dim, hidden_dim)
    tensors[pfx+"down"] = {"offset": offset, "size_bytes": sz, "dtype": "q4", "shape": [dim, hidden_dim]}
    offset += sz

# final rms fp32
size = dim * 4
tensors["final_rms"] = {"offset": offset, "size_bytes": size, "dtype": "fp32", "shape": [dim]}
offset += size

# lm_head fp32
size = vocab_size * dim * 4
tensors["lm_head"] = {"offset": offset, "size_bytes": size, "dtype": "fp32", "shape": [vocab_size, dim]}
offset += size

bin_file = "models/phi4_q4.bin"
print(f"Total projected Mmap size: {offset / (1024**3):.2f} GB")

# Create sparse mock file instantly
with open(bin_file, "wb") as f:
    f.seek(offset - 1)
    f.write(b'\0')

with open("models/phi4_q4_metadata.json", "w") as f:
    json.dump(tensors, f, indent=4)

print("Done. Metadata and Mock Bin sequence generated.")
