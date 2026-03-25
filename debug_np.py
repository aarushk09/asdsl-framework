import torch
import torch.nn.functional as F
import numpy as np
import math
import test_layer

def debug_np():
    # Numpy equivalent vs torch
    dim, hidden_dim, num_heads, num_kv_heads, head_dim = 1024, 2816, 16, 4, 64
    seq_pos = 0; layer_id = 0
    torch.manual_seed(42)
    x = torch.randn(dim, dtype=torch.float32)
    rms1_w = torch.randn(dim, dtype=torch.float32)
    q_dim = num_heads * head_dim; kv_dim = num_kv_heads * head_dim
    qkv_w = torch.randn(q_dim + 2 * kv_dim, dim, dtype=torch.float32) / math.sqrt(dim)
    o_w = torch.randn(dim, q_dim, dtype=torch.float32) / math.sqrt(q_dim)
    rms2_w = torch.randn(dim, dtype=torch.float32)
    gate_w = torch.randn(hidden_dim, dim, dtype=torch.float32) / math.sqrt(dim)
    up_w = torch.randn(hidden_dim, dim, dtype=torch.float32) / math.sqrt(dim)
    down_w = torch.randn(dim, hidden_dim, dtype=torch.float32) / math.sqrt(hidden_dim)

    # 1. PyTorch 
    eps = 1e-5
    pt_rms = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * rms1_w
    qkv_out_pt = F.linear(pt_rms, qkv_w)
    
    q_pt = qkv_out_pt[:q_dim].view(num_heads, head_dim)
    k_pt = qkv_out_pt[q_dim : q_dim + kv_dim].view(num_kv_heads, head_dim)
    v_pt = qkv_out_pt[q_dim + kv_dim:].view(num_kv_heads, head_dim)

    q_rot, k_rot = test_layer.apply_rope_pt(q_pt, k_pt, seq_pos, head_dim)

    q_sdpa = q_rot.unsqueeze(0).unsqueeze(2)
    groups = num_heads // num_kv_heads
    k_sdpa = k_rot.repeat_interleave(groups, dim=0).view(1, num_heads, 1, head_dim)
    v_sdpa = v_pt.repeat_interleave(groups, dim=0).view(1, num_heads, 1, head_dim)

    attn_out_pt = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa).reshape(q_dim)

    o_out_pt = F.linear(attn_out_pt, o_w)
    h_resid_1 = x + o_out_pt

    rms2_out = h_resid_1 * torch.rsqrt(h_resid_1.pow(2).mean(-1, keepdim=True) + eps) * rms2_w

    gate_out = F.linear(rms2_out, gate_w)
    up_out = F.linear(rms2_out, up_w)
    silu_out = F.silu(gate_out) * up_out
    down_out = F.linear(silu_out, down_w)
    final_out_pt = h_resid_1 + down_out

    # 2. Numpy Equivalent (emulating C++)
    x_np = x.numpy().copy()
    rms1_np = rms1_w.numpy()
    sum_sq = np.sum(x_np**2)
    rms = np.sqrt(sum_sq / dim + eps)
    h_np = x_np * (1.0 / rms) * rms1_np
    
    print("RMS Delta:", np.abs(pt_rms.numpy() - h_np).max())

    qkv_np = qkv_w.numpy() @ h_np
    print("QKV Delta:", np.abs(qkv_out_pt.numpy() - qkv_np).max())

    # After QKV delta?
    q_np = qkv_np[:q_dim]
    k_np = qkv_np[q_dim : q_dim + kv_dim]
    v_np = qkv_np[q_dim + kv_dim:]
    
    import asdsl.kernels._native_forward as native_forward
    native_forward.apply_rope(q_np, k_np, seq_pos, head_dim, num_heads, num_kv_heads, 10000.0)
    print("RoPE Q Delta:", np.abs(q_rot.numpy().flatten() - q_np).max())

    # Attn
    cache = native_forward.KVCache(1, 128, num_kv_heads, head_dim)
    attn_out_cpp = native_forward.compute_attention(q_np, k_np, v_np, layer_id, seq_pos, num_heads, cache)
    print("SDPA Delta:", np.abs(attn_out_pt.numpy().flatten() - attn_out_cpp).max())
    
    o_np_out = o_w.numpy() @ attn_out_cpp
    print("O Delta:", np.abs(o_out_pt.numpy() - o_np_out).max())

    res1_np = x_np + o_np_out
    
    h2_np = res1_np.copy()
    sum_sq2 = np.sum(h2_np**2)
    rms2 = np.sqrt(sum_sq2 / dim + eps)
    h2_np = h2_np * (1.0 / rms2) * rms2_w.numpy()
    print("RMS2 Delta:", np.abs(rms2_out.numpy() - h2_np).max())

    gate_np_out = gate_w.numpy() @ h2_np
    up_np_out = up_w.numpy() @ h2_np
    print("Gate Delta:", np.abs(gate_out.numpy() - gate_np_out).max())
    print("Up Delta:", np.abs(up_out.numpy() - up_np_out).max())

    silu_np_out = (gate_np_out / (1.0 + np.exp(-gate_np_out))) * up_np_out
    print("SiLU Delta:", np.abs(silu_out.numpy() - silu_np_out).max())

    down_np_out = down_w.numpy() @ silu_np_out
    print("Down Delta:", np.abs(down_out.numpy() - down_np_out).max())

    final_np = res1_np + down_np_out
    print("Final Delta:", np.abs(final_out_pt.numpy() - final_np).max())
    
if __name__ == '__main__':
    debug_np()
