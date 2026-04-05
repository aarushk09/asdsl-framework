import time
import argparse
import numpy as np
import json
from asdsl.kernels._native_unified import EngineConfig, UnifiedEngine

class MmapBinStore:
    def __init__(self, bin_path, json_path):
        with open(json_path, 'r') as f:
            self.metadata = json.load(f)
        self.mmap = np.memmap(bin_path, dtype=np.uint8, mode='r')

    def get_fp16(self, key):
        info = self.metadata[key]
        off = info['offset']
        sz = info['size_bytes']
        # Load from disk and return contiguous fp32 copy
        return self.mmap[off:off+sz].view(np.float16).astype(np.float32).reshape(info['shape'])

    def get_q4(self, key):
        info = self.metadata[key]
        off = info['offset']
        sz = info['size_bytes']
        return self.mmap[off:off+sz]
def main():
    print("Loading mapped weights...")
    store = MmapBinStore("models/phi4_asb.bin", "models/phi4_asb_metadata.json")
    print("Prepared memory map.")

    config = EngineConfig()
    config.num_layers = 32
    config.hidden_size = 3072
    config.num_heads = 24
    config.num_kv_heads = 8
    config.head_dim = 128
    config.rotary_dim = 96
    config.intermediate_size = 8192
    config.vocab_size = 200064
    config.rms_norm_eps = 1e-5
    config.group_size = 32 # Set correctly for out_bin_path
    config.max_seq_len = 2048

    token_embd = store.get_fp16("model.embed_tokens.weight")
    output_norm = store.get_fp16("model.norm.weight")
    output_proj = token_embd  # Tied embedding!
    
    # RoPE tables - Generate manually
    pos = np.arange(config.max_seq_len, dtype=np.float32)
    dim = np.arange(config.rotary_dim//2, dtype=np.float32)
    inv_freq = 1.0 / (10000.0 ** (2.0 * dim / config.head_dim))
    t = pos[:, None] * inv_freq[None, :]
    cos_table = np.cos(t).astype(np.float32)
    sin_table = np.sin(t).astype(np.float32)

    layers_dict = {}
    for l in range(config.num_layers):
        qkv_p = store.get_q4(f"model.layers.{l}.self_attn.qkv_proj.base_layer.weight")
        o_p = store.get_q4(f"model.layers.{l}.self_attn.o_proj.base_layer.weight")
        gu_p = store.get_q4(f"model.layers.{l}.mlp.gate_up_proj.base_layer.weight")
        dw_p = store.get_q4(f"model.layers.{l}.mlp.down_proj.base_layer.weight")

        layers_dict[l] = {
            'rms_att': store.get_fp16(f"model.layers.{l}.input_layernorm.weight"),
            'rms_ffn': store.get_fp16(f"model.layers.{l}.post_attention_layernorm.weight"),
            'qkv_proj': qkv_p,
            'o_proj': o_p,
            'gate_up_proj': gu_p,
            'down_proj': dw_p,
        }

    engine = UnifiedEngine(
        config,
        token_embd,
        output_norm,
        output_proj,
        cos_table,
        sin_table,
        layers_dict
    )
    print("Done. Generating...")
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-multimodal-instruct", trust_remote_code=True)
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Explain gravity in one sentence."}
    ]
    prompt_tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    prompt_tokens = [int(x) for x in prompt_tokens]  # Same normalization as python script
    
    print("Tokens to generate from:", prompt_tokens)
    t0 = time.perf_counter()
    try:
        out = engine.generate(prompt_tokens, 20)
        print("Success generation!")
    except Exception as e:
        print("Error during generation", e)
    t1 = time.perf_counter()
    
    print(f"Generated {len(out)} total tokens in {t1-t0:.2f}s ({20/(t1-t0):.2f} tok/s)")
    print("Output:", tokenizer.decode(out))

if __name__ == '__main__':
    main()
