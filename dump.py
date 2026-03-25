import torch
import torch.nn.functional as F
import numpy as np
import math
import asdsl.kernels._native_forward as native_forward
import test_layer

def dump_deltas():
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

    # test silu via torch simulation!
    eps = 1e-5
    x_cpp = x.clone().numpy()
    
    # Let's run individual parts by exporting new temporary pybind wrappers or just checking torch norms:
    print("RMSNorm PT Norm:", (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * rms1_w).norm().item())
    
    # Let's just create a modified C++ test for components by comparing PT to NP implementation
    
if __name__ == '__main__':
    dump_deltas()