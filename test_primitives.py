import torch
import numpy as np
import asdsl.kernels._native_forward as native_forward

def test_rmsnorm():
    print("--- Testing RMSNorm ---")
    dim = 4096
    eps = 1e-5
    
    # 1. Random inputs
    x = torch.randn(dim, dtype=torch.float32)
    weight = torch.ones(dim, dtype=torch.float32) * 0.5 + torch.rand(dim) * 0.5
    
    # 2. PyTorch baseline computation
    pytorch_rms = x * torch.rsqrt(x.pow(2).mean() + eps) * weight
    
    # 3. Native C++ execution
    x_np = x.numpy().copy()
    weight_np = weight.numpy().copy()
    
    # Executes in-place
    native_forward.apply_rmsnorm(x_np, weight_np, eps)
    
    # 4. Compare
    delta = np.abs(pytorch_rms.numpy() - x_np).max()
    print(f"Max Delta: {delta:.8f}")
    
    if delta < 1e-4:
        print("SUCCESS: RMSNorm matches exactly!")
    else:
        print("FAILURE: RMSNorm mismatch!")

def test_rope():
    print("\n--- Testing RoPE ---")
    seq_pos = 15
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    theta = 10000.0
    
    # 1. Random inputs
    q = torch.randn(num_heads, head_dim, dtype=torch.float32)
    k = torch.randn(num_kv_heads, head_dim, dtype=torch.float32)
    
    # 2. PyTorch baseline computation (Block-halves implementation)
    half_dim = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    val = seq_pos * freqs
    cos = torch.cos(val)
    sin = torch.sin(val)
    
    def apply_rope_pt(x):
        x0 = x[:, :half_dim]
        x1 = x[:, half_dim:]
        out0 = x0 * cos - x1 * sin
        out1 = x0 * sin + x1 * cos
        return torch.cat([out0, out1], dim=-1)
        
    q_pt = apply_rope_pt(q)
    k_pt = apply_rope_pt(k)
    
    # 3. Native C++ execution
    q_np = q.flatten().numpy().copy()
    k_np = k.flatten().numpy().copy()
    
    # Executes in-place
    native_forward.apply_rope(q_np, k_np, seq_pos, head_dim, num_heads, num_kv_heads, theta)
    
    q_np_reshaped = q_np.reshape(num_heads, head_dim)
    k_np_reshaped = k_np.reshape(num_kv_heads, head_dim)
    
    # 4. Compare
    q_delta = np.abs(q_pt.numpy() - q_np_reshaped).max()
    k_delta = np.abs(k_pt.numpy() - k_np_reshaped).max()
    
    print(f"Q Max Delta: {q_delta:.8f}")
    print(f"K Max Delta: {k_delta:.8f}")
    
    if q_delta < 1e-4 and k_delta < 1e-4:
        print("SUCCESS: RoPE matches exactly!")
    else:
        print("FAILURE: RoPE mismatch!")

if __name__ == '__main__':
    test_rmsnorm()
    test_rope()
