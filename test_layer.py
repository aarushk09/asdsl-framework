import torch
import torch.nn.functional as F
import numpy as np
import math
import asdsl.kernels._native_forward as native_forward

def apply_rope_pt(q, k, seq_pos, head_dim, theta=10000.0):
    half_dim = head_dim // 2
    freqs = seq_pos * (1.0 / (theta ** (torch.arange(0, half_dim, dtype=torch.float32) * 2 / head_dim)))
    cos_f = torch.cos(freqs)
    sin_f = torch.sin(freqs)

    def rot(x):
        # x is [..., head_dim]
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        return torch.cat([x1 * cos_f - x2 * sin_f, x2 * cos_f + x1 * sin_f], dim=-1)

    return rot(q), rot(k)

def main():
    print("--- Testing Full Transformer Layer ---")
    torch.manual_seed(42)

    dim = 1024
    hidden_dim = 2816
    num_heads = 16
    num_kv_heads = 4
    head_dim = 64
    seq_pos = 0
    layer_id = 0

    # Initialize input x
    x = torch.randn(dim, dtype=torch.float32)

    # Initialize weights
    rms1_w = torch.randn(dim, dtype=torch.float32)
    
    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    qkv_dim = q_dim + 2 * kv_dim
    qkv_w = torch.randn(qkv_dim, dim, dtype=torch.float32) / math.sqrt(dim)
    
    o_w = torch.randn(dim, q_dim, dtype=torch.float32) / math.sqrt(q_dim)

    rms2_w = torch.randn(dim, dtype=torch.float32)
    
    gate_w = torch.randn(hidden_dim, dim, dtype=torch.float32) / math.sqrt(dim)
    up_w = torch.randn(hidden_dim, dim, dtype=torch.float32) / math.sqrt(dim)
    down_w = torch.randn(dim, hidden_dim, dtype=torch.float32) / math.sqrt(hidden_dim)

    # --- PyTorch Execution ---
    eps = 1e-5
    
    # 1. RMSNorm 1
    rms1_out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * rms1_w
    
    # 2. QKV projection
    qkv_out = F.linear(rms1_out, qkv_w)
    q_pt = qkv_out[:q_dim].view(num_heads, head_dim)
    k_pt = qkv_out[q_dim : q_dim + kv_dim].view(num_kv_heads, head_dim)
    v_pt = qkv_out[q_dim + kv_dim:].view(num_kv_heads, head_dim)

    # 3. RoPE
    q_rot, k_rot = apply_rope_pt(q_pt, k_pt, seq_pos, head_dim)

    # 4. SDPA (Mock since seq_pos=0)
    # [1, num_heads, 1, head_dim]
    q_sdpa = q_rot.unsqueeze(0).unsqueeze(2)
    # Duplicate K and V for GQA
    groups = num_heads // num_kv_heads
    k_sdpa = k_rot.repeat_interleave(groups, dim=0).view(1, num_heads, 1, head_dim)
    v_sdpa = v_pt.repeat_interleave(groups, dim=0).view(1, num_heads, 1, head_dim)

    attn_out = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa)
    attn_out = attn_out.reshape(q_dim)

    # 5. O Projection
    o_out_pt = F.linear(attn_out, o_w)

    # 6. Residual 1
    h_resid_1 = x + o_out_pt

    # 7. RMSNorm 2
    rms2_out = h_resid_1 * torch.rsqrt(h_resid_1.pow(2).mean(-1, keepdim=True) + eps) * rms2_w

    # 8. Gate / Up
    gate_out = F.linear(rms2_out, gate_w)
    up_out = F.linear(rms2_out, up_w)

    # 9. SiLU
    silu_out = F.silu(gate_out) * up_out

    # 10. Down
    down_out = F.linear(silu_out, down_w)

    # 11. Residual 2
    final_out_pt = h_resid_1 + down_out

    # --- C++ Execution ---
    x_cpp = x.clone().numpy()
    rms1_np = rms1_w.numpy()
    qkv_np = qkv_w.numpy().flatten()
    o_np = o_w.numpy().flatten()
    rms2_np = rms2_w.numpy()
    gate_np = gate_w.numpy().flatten()
    up_np = up_w.numpy().flatten()
    down_np = down_w.numpy().flatten()

    cache = native_forward.KVCache(1, 128, num_kv_heads, head_dim)

    native_forward.forward_layer(
        x_cpp, rms1_np, qkv_np, o_np, rms2_np, gate_np, up_np, down_np,
        dim, hidden_dim, num_heads, num_kv_heads, head_dim, layer_id, seq_pos, cache
    )

    delta = np.abs(final_out_pt.numpy() - x_cpp).max()
    print(f"Layer Forward Max Delta: {delta:.8f}")

    if delta < 1e-4:
        print("SUCCESS: C++ forward_layer matches PyTorch layer output perfectly!")
    else:
        print("FAILURE: C++ forward_layer mismatch!")

if __name__ == '__main__':
    main()
