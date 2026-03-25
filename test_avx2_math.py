import json
import torch
import numpy as np
import asdsl.kernels._native_forward as native_forward

GROUP_SIZE = 32

def test_avx2_math():
    with open('models/phi4_q4_32_metadata.json') as f:
        metadata = json.load(f)

    print("Initializing C++ MMAP Engine...")
    mmap = native_forward.MmapWeights('models/phi4_q4_32.bin', metadata)
    
    # We will test against the first layer's Q_proj
    test_key = 'model.layers.0.self_attn.qkv_proj.base_layer.weight'
    info = metadata[test_key]
    rows, cols = info['shape']
    
    print(f"Testing Matrix: {test_key} | Shape: {rows}x{cols}")
    
    # Create dummy activations: exactly 'cols' amount of 1.0s
    X = torch.ones(cols, dtype=torch.float32)
    X_np = X.numpy()
    
    # ---------------------------------------------------------
    # 1. Native C++ AVX2 Execution
    # ---------------------------------------------------------
    c_result = mmap.test_gemv_q4(test_key, X_np)
    
    # ---------------------------------------------------------
    # 2. Python / PyTorch Emulation Execution
    # ---------------------------------------------------------
    # Read the raw bytes out of the file for the very first row
    num_blocks = cols // GROUP_SIZE
    row_bytes_len = num_blocks * 18 # 18 bytes per group
    
    with open('models/phi4_q4_32.bin', 'rb') as f:
        f.seek(info['offset'])
        raw_row_bytes = f.read(row_bytes_len)
        
    py_result = 0.0
    for b in range(num_blocks):
        block_data = raw_row_bytes[b*18 : (b+1)*18]
        
        # Unpack scale
        scale = np.frombuffer(block_data[:2], dtype=np.float16)[0]
        
        # Unpack weights
        packed = np.frombuffer(block_data[2:], dtype=np.uint8)
        lower = packed & 0x0F
        upper = (packed >> 4) & 0x0F
        
        # Since we packed them as lower=0, upper=1, lower=2, upper=3...
        # Interleave lower and upper
        unpacked = np.empty(GROUP_SIZE, dtype=np.int8)
        unpacked[0::2] = lower
        unpacked[1::2] = upper
        
        # Shift back to -7..7
        unpacked = unpacked - 8
        
        # Multiply by scale and X
        py_result += np.sum(unpacked * scale * X_np[b*GROUP_SIZE : (b+1)*GROUP_SIZE])
        
    # ---------------------------------------------------------
    # 3. Compare Results
    # ---------------------------------------------------------
    print("\n==================================")
    print(f"C++ AVX2 FMA Result : {c_result:.6f}")
    print(f"Python Numpy Result : {py_result:.6f}")
    print(f"Delta               : {abs(c_result - py_result):.6f}")
    print("==================================")

    if np.isclose(c_result, py_result, atol=1e-3):
        print("SUCCESS: Math matches within 1e-3 tolerance! Silicon verified.")
    else:
        print("FAILURE: Math mismatch detected.")

if __name__ == '__main__':
    test_avx2_math()
