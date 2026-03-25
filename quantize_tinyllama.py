import os
os.environ["USE_TF"] = "0"
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM

def f32_to_f16_bytes(val):
    return np.array([val], dtype=np.float16).view(np.uint8).tobytes()

def quantize_block_q4_k_m(tensor: torch.Tensor):
    # tensor is [rows, cols]
    rows, cols = tensor.shape
    assert cols % 256 == 0
    blocks_per_row = cols // 256
    
    data = tensor.float().numpy()
    data = data.reshape(rows * blocks_per_row, 8, 32)
    
    # Calculate min and max per sub-block (8 sub-blocks of 32 elements per super-block)
    mins = np.min(data, axis=2) # (N, 8)
    maxs = np.max(data, axis=2) # (N, 8)
    
    # Calculate super-block scales
    dmin = np.min(mins, axis=1) # (N,)
    dmax = np.max(maxs, axis=1) # (N,)
    
    d = np.where((dmax - dmin) > 0, (dmax - dmin) / ((1 << 6) - 1), 1e-9).astype(np.float16) # 6-bit scales
    dmin_val = dmin.astype(np.float16)
    
    # Calculate 6-bit scales per sub-block
    inv_d = 1.0 / d.astype(np.float32)
    scales = np.round((maxs - mins) * inv_d[:, None]).astype(np.uint8)
    scales = np.clip(scales, 0, 63) # 6-bit
    
    # Pack 8 x 6-bit scales into 6 bytes
    # scales shape is (N, 8). Pack into packed_scales (N, 6)
    s0, s1, s2, s3, s4, s5, s6, s7 = [scales[:, i].astype(np.uint16) for i in range(8)]
    packed_scales = np.empty((rows * blocks_per_row, 6), dtype=np.uint8)
    packed_scales[:, 0] = s0 | ((s1 & 0x03) << 6)
    packed_scales[:, 1] = (s1 >> 2) | ((s2 & 0x0F) << 4)
    packed_scales[:, 2] = (s2 >> 4) | ((s3 & 0x3F) << 2)
    packed_scales[:, 3] = s4 | ((s5 & 0x03) << 6)
    packed_scales[:, 4] = (s5 >> 2) | ((s6 & 0x0F) << 4)
    packed_scales[:, 5] = (s6 >> 4) | ((s7 & 0x3F) << 2)
    
    # Quantize weights
    sub_scales = (scales.astype(np.float32) * d.astype(np.float32)[:, None]) / 15.0 # 4-bit weights
    sub_scales = np.where(sub_scales > 0, sub_scales, 1e-9)
    
    q_data = np.round((data - mins[:, :, None]) / sub_scales[:, :, None]).astype(np.uint8)
    q_data = np.clip(q_data, 0, 15)
    
    # Pack 2 x 4-bit weights into 1 byte (N, 8, 16) -> (N, 128)
    q_data_packed = (q_data[:, :, 1::2] << 4) | q_data[:, :, 0::2]
    q_data_packed = q_data_packed.reshape(rows * blocks_per_row, 128)
    
    # Combine header (12 bytes) + weights (128 bytes) = 140 bytes
    combined = np.empty((rows * blocks_per_row, 140), dtype=np.uint8)
    combined[:, 0:2] = d.view(np.uint8).reshape(-1, 2)
    combined[:, 2:4] = dmin_val.view(np.uint8).reshape(-1, 2)
    combined[:, 4:10] = packed_scales
    combined[:, 10:12] = 0 # 2 bytes padding/alignment
    combined[:, 12:140] = q_data_packed
    
    return combined.tobytes()

def main():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"Loading {model_id} for quantization...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    
    out_bin = "models/tinyllama_q4.bin"
    out_meta = "models/tinyllama_q4_metadata.json"
    
    os.makedirs("models", exist_ok=True)
    f_out = open(out_bin, "wb")
    
    metadata = {}
    current_offset = 0
    
    def add_tensor(name, t_bytes, shape, dtype):
        nonlocal current_offset
        f_out.write(t_bytes)
        metadata[name] = {
            "shape": list(shape),
            "dtype": dtype,
            "offset": current_offset,
            "size_bytes": len(t_bytes)
        }
        current_offset += len(t_bytes)

    # Embeddings (FP32)
    embed = model.model.embed_tokens.weight.detach().numpy().astype(np.float32).tobytes()
    add_tensor("embed", embed, model.model.embed_tokens.weight.shape, "fp32")
    
    num_layers = model.config.num_hidden_layers
    for l in range(num_layers):
        print(f"Quantizing layer {l}/{num_layers}...")
        layer = model.model.layers[l]
        
        # RMS1 (FP32)
        r1 = layer.input_layernorm.weight.detach().numpy().astype(np.float32).tobytes()
        add_tensor(f"l{l}_rms1", r1, layer.input_layernorm.weight.shape, "fp32")
        
        # QKV (Q4)
        q = layer.self_attn.q_proj.weight.detach()
        k = layer.self_attn.k_proj.weight.detach()
        v = layer.self_attn.v_proj.weight.detach()
        qkv = torch.cat([q, k, v], dim=0)
        qkv_bytes = quantize_block_q4_k_m(qkv)
        add_tensor(f"l{l}_qkv", qkv_bytes, qkv.shape, "q4_k_m")
        
        # O (Q4)
        o_bytes = quantize_block_q4_k_m(layer.self_attn.o_proj.weight.detach())
        add_tensor(f"l{l}_o", o_bytes, layer.self_attn.o_proj.weight.shape, "q4_k_m")
        
        # RMS2 (FP32)
        r2 = layer.post_attention_layernorm.weight.detach().numpy().astype(np.float32).tobytes()
        add_tensor(f"l{l}_rms2", r2, layer.post_attention_layernorm.weight.shape, "fp32")
        
        # Gate, Up, Down (Q4)
        gate_bytes = quantize_block_q4_k_m(layer.mlp.gate_proj.weight.detach())
        add_tensor(f"l{l}_gate", gate_bytes, layer.mlp.gate_proj.weight.shape, "q4_k_m")
        
        up_bytes = quantize_block_q4_k_m(layer.mlp.up_proj.weight.detach())
        add_tensor(f"l{l}_up", up_bytes, layer.mlp.up_proj.weight.shape, "q4_k_m")
        
        down_bytes = quantize_block_q4_k_m(layer.mlp.down_proj.weight.detach())
        add_tensor(f"l{l}_down", down_bytes, layer.mlp.down_proj.weight.shape, "q4_k_m")

    # Final RMS (FP32)
    frms = model.model.norm.weight.detach().numpy().astype(np.float32).tobytes()
    add_tensor("final_rms", frms, model.model.norm.weight.shape, "fp32")
    
    # LM Head (FP32) (Wait, I can keep LM head FP32 or Q4. I'll keep it FP32 for max accuracy)
    lm = model.lm_head.weight.detach().numpy().astype(np.float32).tobytes()
    add_tensor("lm_head", lm, model.lm_head.weight.shape, "fp32")
    
    f_out.close()
    
    with open(out_meta, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Quantization complete! Saved to {out_bin}, size: {current_offset/1024/1024:.2f} MB")

if __name__ == '__main__':
    main()