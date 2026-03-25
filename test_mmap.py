import json
import asdsl.kernels._native_forward as native_forward

with open('models/phi4_q4_32_metadata.json') as f:
    metadata = json.load(f)

print("-----------------------------------------")
print("Testing C++ MmapWeights constructor...")
print("-----------------------------------------")
mmap = native_forward.MmapWeights('models/phi4_q4_32.bin', metadata)

offset = metadata['model.layers.0.self_attn.qkv_proj.base_layer.weight']['offset']
with open('models/phi4_q4_32.bin', 'rb') as f:
    f.seek(offset)
    py_bytes = f.read(5)

print("\n-----------------------------------------")
print("[Python DEBUG] First 5 bytes exactly from file:  " + " ".join([f"{b:02X}" for b in py_bytes]))
print("-----------------------------------------")
