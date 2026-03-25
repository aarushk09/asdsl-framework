import os
os.environ["USE_TF"] = "0"
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM

def f32_to_f16_bytes(val):
    return np.array([val], dtype=np.float16).view(np.uint8).tobytes()

def quantize_block_q4_32(tensor: torch.Tensor):
    # tensor is [rows, cols]
    rows, cols = tensor.shape
    assert cols % 32 == 0
    blocks_per_row = cols // 32
    
    data = tensor.float().numpy()
    data = data.reshape(rows * blocks_per_row, 32)
    
    abs_max = np.max(np.abs(data), axis=1)
    scale = np.where(abs_max > 0, abs_max / 7.0, 1e-9).astype(np.float16)  # shape (N,)
    
    # quantized values and shift +8
    q_data = np.round(data / scale[:, None].astype(np.float32)).astype(np.int8)
    q_data = np.clip(q_data, -8, 7)
    q_data_u8 = (q_data + 8).astype(np.uint8)
    
    # Pack 2 values into 1 byte.
    low_nibbles = q_data_u8[:, 0::2]
    high_nibbles = q_data_u8[:, 1::2]
    packed = (high_nibbles << 4) | low_nibbles  # shape (N, 16)
    
    # We want: scale (2 bytes), then packed (16 bytes), for each row.
    # We can create a structured numpy array or just byte buffer.
    # Easiest way in numpy:
    combined = np.empty((rows * blocks_per_row, 18), dtype=np.uint8)
    combined[:, 0:2] = scale.view(np.uint8).reshape(-1, 2)
    combined[:, 2:18] = packed
    
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
        qkv_bytes = quantize_block_q4_32(qkv)
        add_tensor(f"l{l}_qkv", qkv_bytes, qkv.shape, "q4_32")
        
        # O (Q4)
        o_bytes = quantize_block_q4_32(layer.self_attn.o_proj.weight.detach())
        add_tensor(f"l{l}_o", o_bytes, layer.self_attn.o_proj.weight.shape, "q4_32")
        
        # RMS2 (FP32)
        r2 = layer.post_attention_layernorm.weight.detach().numpy().astype(np.float32).tobytes()
        add_tensor(f"l{l}_rms2", r2, layer.post_attention_layernorm.weight.shape, "fp32")
        
        # Gate, Up, Down (Q4)
        gate_bytes = quantize_block_q4_32(layer.mlp.gate_proj.weight.detach())
        add_tensor(f"l{l}_gate", gate_bytes, layer.mlp.gate_proj.weight.shape, "q4_32")
        
        up_bytes = quantize_block_q4_32(layer.mlp.up_proj.weight.detach())
        add_tensor(f"l{l}_up", up_bytes, layer.mlp.up_proj.weight.shape, "q4_32")
        
        down_bytes = quantize_block_q4_32(layer.mlp.down_proj.weight.detach())
        add_tensor(f"l{l}_down", down_bytes, layer.mlp.down_proj.weight.shape, "q4_32")

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