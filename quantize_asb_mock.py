import json
import os

dim = 5120
hidden_dim = 17920
vocab_size = 100352
num_layers = 40
group_size = 32

def padding(size, alignment=64):
    return (size + alignment - 1) & ~(alignment - 1)

def asb_size(rows, cols, avg_bits=2.5):
    # ASB Format: each group has 8 byte header.
    # Data size: group_size * avg_bits / 8
    # Total for group: 8 + group_size * avg_bits / 8
    num_groups = (rows * cols) // group_size
    bytes_per_group = 8 + (group_size * avg_bits / 8.0)
    
    # Plus: 4 bytes per row for row_offsets and 2 bytes per group for Permutations
    return int(rows * 4 + num_groups * 2 + num_groups * bytes_per_group)


print("Generating mock ASB layout for Phi-4 14B...")
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

    # ASB layers
    sz = asb_size(7680, dim)
    tensors[pfx+"qkv"] = {"offset": offset, "size_bytes": sz, "dtype": "asb_mixed", "shape": [7680, dim]}
    offset += sz

    sz = asb_size(dim, dim)
    tensors[pfx+"o"] = {"offset": offset, "size_bytes": sz, "dtype": "asb_mixed", "shape": [dim, dim]}
    offset += sz
    
    sz = asb_size(hidden_dim, dim)
    tensors[pfx+"gate"] = {"offset": offset, "size_bytes": sz, "dtype": "asb_mixed", "shape": [hidden_dim, dim]}
    offset += sz
    tensors[pfx+"up"] = {"offset": offset, "size_bytes": sz, "dtype": "asb_mixed", "shape": [hidden_dim, dim]}
    offset += sz

    sz = asb_size(dim, hidden_dim)
    tensors[pfx+"down"] = {"offset": offset, "size_bytes": sz, "dtype": "asb_mixed", "shape": [dim, hidden_dim]}
    offset += sz

# final rms fp32
size = dim * 4
tensors["final_rms"] = {"offset": offset, "size_bytes": size, "dtype": "fp32", "shape": [dim]}
offset += size

# lm_head fp32
size = vocab_size * dim * 4
tensors["lm_head"] = {"offset": offset, "size_bytes": size, "dtype": "fp32", "shape": [vocab_size, dim]}
offset += size

os.makedirs("models", exist_ok=True)
bin_file = "models/phi4_asb.bin"
print(f"Total projected Mmap size: {offset / (1024**3):.2f} GB")

# Create sparse mock file instantly
with open(bin_file, "wb") as f:
    f.seek(offset - 1)
    f.write(b'\0')

with open("models/phi4_asb_metadata.json", "w") as f:
    json.dump(tensors, f, indent=4)

print("Done. Metadata and Mock Bin sequence generated.")

