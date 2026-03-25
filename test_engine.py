import torch
import torch.nn.functional as F
import numpy as np
import math
import asdsl.kernels._native_forward as native_forward
from test_layer import apply_rope_pt

def test_engine():
    print("--- Testing Full Transformer Engine Orchestrator ---")
    torch.manual_seed(42)

    # Mock dimensions
    dim, hidden_dim, num_heads, num_kv_heads, head_dim = 128, 256, 4, 2, 32
    num_layers = 4
    vocab_size = 32000
    seq_pos = 0
    token_id = 1234 # Mock token id

    # Weights
    token_emb = torch.randn(vocab_size, dim, dtype=torch.float32) / math.sqrt(dim)
    
    layers_w_pt = []
    layers_w_cpp = []
    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    
    for _ in range(num_layers):
        rms1_w = torch.randn(dim, dtype=torch.float32)
        qkv_w = torch.randn(q_dim + 2 * kv_dim, dim, dtype=torch.float32) / math.sqrt(dim)
        o_w = torch.randn(dim, q_dim, dtype=torch.float32) / math.sqrt(q_dim)
        rms2_w = torch.randn(dim, dtype=torch.float32)
        gate_w = torch.randn(hidden_dim, dim, dtype=torch.float32) / math.sqrt(dim)
        up_w = torch.randn(hidden_dim, dim, dtype=torch.float32) / math.sqrt(dim)
        down_w = torch.randn(dim, hidden_dim, dtype=torch.float32) / math.sqrt(hidden_dim)
        
        layers_w_pt.append((rms1_w, qkv_w, o_w, rms2_w, gate_w, up_w, down_w))
        layers_w_cpp.append((
            rms1_w.numpy(), qkv_w.numpy().flatten(), o_w.numpy().flatten(),
            rms2_w.numpy(), gate_w.numpy().flatten(), up_w.numpy().flatten(), down_w.numpy().flatten()
        ))

    final_rms_w = torch.randn(dim, dtype=torch.float32)
    lm_head_w = torch.randn(vocab_size, dim, dtype=torch.float32) / math.sqrt(dim)

    # --- PyTorch HF Model Execution ---
    x = token_emb[token_id].clone()
    eps = 1e-5
    
    for l in range(num_layers):
        rms1, qkv, o, rms2, gate, up, down = layers_w_pt[l]
        h_resid = x.clone()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * rms1

        qkv_out_pt = F.linear(x, qkv)
        q_pt = qkv_out_pt[:q_dim].view(num_heads, head_dim)
        k_pt = qkv_out_pt[q_dim : q_dim + kv_dim].view(num_kv_heads, head_dim)
        v_pt = qkv_out_pt[q_dim + kv_dim:].view(num_kv_heads, head_dim)

        q_rot, k_rot = apply_rope_pt(q_pt, k_pt, seq_pos, head_dim)
        q_sdpa = q_rot.unsqueeze(0).unsqueeze(2)
        groups = num_heads // num_kv_heads
        k_sdpa = k_rot.repeat_interleave(groups, dim=0).view(1, num_heads, 1, head_dim)
        v_sdpa = v_pt.repeat_interleave(groups, dim=0).view(1, num_heads, 1, head_dim)

        attn_out_pt = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa).reshape(q_dim)

        o_out_pt = F.linear(attn_out_pt, o)
        x = h_resid + o_out_pt

        h_resid2 = x.clone()
        rms2_out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * rms2
        
        gate_out = F.linear(rms2_out, gate)
        up_out = F.linear(rms2_out, up)
        silu_out = F.silu(gate_out) * up_out
        down_out = F.linear(silu_out, down)
        
        x = h_resid2 + down_out

    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * final_rms_w
    logits = F.linear(x, lm_head_w)
    pt_next_token = logits.argmax().item()

    # --- C++ Engine Execution ---
    cache = native_forward.KVCache(num_layers, 128, num_kv_heads, head_dim)
    cpp_next_token = native_forward.generate_token(
        token_id, seq_pos, token_emb.numpy().flatten(), layers_w_cpp,
        final_rms_w.numpy(), lm_head_w.numpy().flatten(),
        num_layers, dim, hidden_dim, num_heads, num_kv_heads, head_dim, vocab_size,
        cache
    )

    print(f"PT Next Token ID:  {pt_next_token} (Logit: {logits[pt_next_token].item():.4f})")
    
    # Check what logit value C++ thinks it had, if we want to add trace but this is enough:
    print(f"C++ Next Token ID: {cpp_next_token}")

    if pt_next_token == cpp_next_token:
        print("SUCCESS: Full C++ Engine prediction perfectly matches HuggingFace PyTorch!")
    else:
        print("FAILURE: Predictions differ!")

if __name__ == '__main__':
    test_engine()
