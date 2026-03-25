import torch
import torch.nn.functional as F
import numpy as np
import math
import asdsl.kernels._native_forward as native_forward
import test_layer

def debug():
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

    eps = 1e-5
    h_resid_0 = x.clone()
    pt_rms = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * rms1_w

    qkv_out_pt = F.linear(pt_rms, qkv_w)
    q_pt = qkv_out_pt[:q_dim].view(num_heads, head_dim)
    k_pt = qkv_out_pt[q_dim : q_dim + kv_dim].view(num_kv_heads, head_dim)
    v_pt = qkv_out_pt[q_dim + kv_dim:].view(num_kv_heads, head_dim)

    q_rot, k_rot = test_layer.apply_rope_pt(q_pt, k_pt, seq_pos, head_dim)

    q_sdpa = q_rot.unsqueeze(0).unsqueeze(2)
    groups = num_heads // num_kv_heads
    k_sdpa = k_rot.unsqueeze(0).repeat_interleave(groups, dim=0).view(num_heads, head_dim).unsqueeze(0).unsqueeze(2)
    v_sdpa = v_pt.unsqueeze(0).repeat_interleave(groups, dim=0).view(num_heads, head_dim).unsqueeze(0).unsqueeze(2)

    attn_out_pt = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa).reshape(q_dim)

    o_out_pt = F.linear(attn_out_pt, o_w)
    h_resid_1 = h_resid_0 + o_out_pt

    rms2_out = h_resid_1 * torch.rsqrt(h_resid_1.pow(2).mean(-1, keepdim=True) + eps) * rms2_w

    gate_out = F.linear(rms2_out, gate_w)
    up_out = F.linear(rms2_out, up_w)
    silu_out = F.silu(gate_out) * up_out
    down_out = F.linear(silu_out, down_w)
    final_out_pt = h_resid_1 + down_out

    # --- C++ ---
    x_cpp = x.clone().numpy()
    cache = native_forward.KVCache(1, 128, num_kv_heads, head_dim)
    native_forward.forward_layer(
        x_cpp, rms1_w.numpy(), qkv_w.numpy().flatten(), o_w.numpy().flatten(), rms2_w.numpy(),
        gate_w.numpy().flatten(), up_w.numpy().flatten(), down_w.numpy().flatten(),
        dim, hidden_dim, num_heads, num_kv_heads, head_dim, layer_id, seq_pos, cache
    )
    
    print("Delta:", np.abs(final_out_pt.numpy() - x_cpp).max())
if __name__ == '__main__':
    debug()