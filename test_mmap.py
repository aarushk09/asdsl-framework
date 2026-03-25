import json
from pathlib import Path
import asdsl.kernels._native_forward as native_forward

meta_candidates = [
    Path('models/phi4_q4_i4_metadata.json'),
    Path('models/phi4_q4_metadata.json'),
    Path('models/phi4_q4_32_metadata.json'),
]
bin_candidates = [
    Path('models/phi4_q4_i4.bin'),
    Path('models/phi4_q4.bin'),
    Path('models/phi4_q4_32.bin'),
]

meta_path = next((p for p in meta_candidates if p.exists()), None)
bin_path = next((p for p in bin_candidates if p.exists()), None)
if meta_path is None or bin_path is None:
    raise FileNotFoundError('No compatible model metadata/bin found in models/')

with open(meta_path) as f:
    metadata = json.load(f)

print("-----------------------------------------")
print("Testing C++ MmapWeights constructor...")
print("-----------------------------------------")
mmap = native_forward.MmapWeights(str(bin_path), metadata)

first_key = next(iter(metadata.keys()))
offset = metadata[first_key]['offset']
with open(bin_path, 'rb') as f:
    f.seek(offset)
    py_bytes = f.read(5)

print("\n-----------------------------------------")
print("[Python DEBUG] First 5 bytes exactly from file:  " + " ".join([f"{b:02X}" for b in py_bytes]))
print("-----------------------------------------")
