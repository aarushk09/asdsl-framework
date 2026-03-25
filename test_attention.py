import torch
import torch.nn.functional as F
import numpy as np
import asdsl.kernels._native_forward as native_forward

def test_attention():
    print("--- Testing SDPA with GQA ---")
    num_layers = 1
    max_seq_len = 128
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    layer_id = 0
    seq_pos = 5
    
    # 1. Initialize caches
    cache = native_forward.KVCache(num_layers, max_seq_len, num_kv_heads, head_dim)
    
    history_k = []
    history_v = []
    
    for p in range(seq_pos):
        k_p = torch.randn(num_kv_heads, head_dim, dtype=torch.float32)
        v_p = torch.randn(num_kv_heads, head_dim, dtype=torch.float32)
        history_k.append(k_p)
        history_v.append(v_p)
        cache.set_history(layer_id, p, k_p.numpy(), v_p.numpy())
        
    q_curr = torch.randn(num_heads, head_dim, dtype=torch.float32)
    k_curr = torch.randn(num_kv_heads, head_dim, dtype=torch.float32)
    v_curr = torch.randn(num_kv_heads, head_dim, dtype=torch.float32)
    
    history_k.append(k_curr)
    history_v.append(v_curr)
    
    # [seq_pos + 1, num_kv_heads, head_dim] -> [1, num_kv_heads, seq, head_dim]
    keys = torch.stack(history_k, dim=0).transpose(0, 1).unsqueeze(0)
    values = torch.stack(history_v, dim=0).transpose(0, 1).unsqueeze(0)
    
    # Repeat for GQA
    groups = num_heads // num_kv_heads
    keys = keys.unsqueeze(2).expand(-1, -1, groups, -1, -1).reshape(1, num_heads, seq_pos + 1, head_dim)
    values = values.unsqueeze(2).expand(-1, -1, groups, -1, -1).reshape(1, num_heads, seq_pos + 1, head_dim)
    
    q_pt = q_curr.unsqueeze(0).unsqueeze(2) 
    
    out_pt = F.scaled_dot_product_attention(q_pt, keys, values)
    out_pt = out_pt.squeeze(2).squeeze(0) # [num_heads, head_dim]
    
    out_cpp = native_forward.compute_attention(
        q_curr.numpy(), k_curr.numpy(), v_curr.numpy(), layer_id, seq_pos, num_heads, cache
    ).reshape(num_heads, head_dim)
    
    delta = np.abs(out_pt.numpy() - out_cpp).max()
    print(f"SDPA Max Delta: {delta:.8f}")
    
    if delta < 1e-3:
        print("SUCCESS: SDPA and KV Cache match exactly!")
    else:
        print("FAILURE: SDPA mismatch!")

if __name__ == '__main__':
    test_attention()
