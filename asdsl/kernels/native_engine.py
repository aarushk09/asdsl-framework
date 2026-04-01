"""Python wrapper for the Pure C++ Inference Engine."""
import numpy as np
import torch
import time

try:
    from asdsl.kernels import _native_engine
    HAS_NATIVE_ENGINE = True
except ImportError:
    HAS_NATIVE_ENGINE = False
    _native_engine = None


class NativeEngineWrapper:
    """Wraps the C++ InferenceEngine for use with ASDSL's benchmark infrastructure.
    
    C++ handles transformer layers, Python handles LM head via optimized matmul.
    """
    
    def __init__(self, store, max_seq_len=2048):
        if not HAS_NATIVE_ENGINE:
            raise RuntimeError("_native_engine not available")
        
        self.store = store
        self.max_seq_len = max_seq_len
        self.group_size = store.group_size
        self.vocab_size = 200064  # Phi-4 vocab size
        
        # Create C++ engine
        self.engine = _native_engine.InferenceEngine(
            num_layers=32,
            max_seq_len=max_seq_len,
            group_size=self.group_size,
            vocab_size=self.vocab_size
        )
        
        # Register layers with zero-copy pointers
        for i in range(32):
            key = (i, "qkv_proj")
            qkv_np = store._quant_packed_np[key]
            qkv_scales_np = store._quant_sc_np[key]
            
            key = (i, "o_proj")
            o_np = store._quant_packed_np[key]
            o_scales_np = store._quant_sc_np[key]
            
            key = (i, "gate_up_proj")
            gu_np = store._quant_packed_np[key]
            gu_scales_np = store._quant_sc_np[key]
            
            key = (i, "down_proj")
            down_np = store._quant_packed_np[key]
            down_scales_np = store._quant_sc_np[key]
            
            rms_att = store.layer_norms[i]["input_layernorm"].detach().cpu().float().numpy()
            rms_ffn = store.layer_norms[i]["post_attention_layernorm"].detach().cpu().float().numpy()
            
            self.engine.register_layer(
                i,
                qkv_np.ctypes.data, qkv_scales_np.ctypes.data,
                o_np.ctypes.data, o_scales_np.ctypes.data,
                gu_np.ctypes.data, gu_scales_np.ctypes.data,
                down_np.ctypes.data, down_scales_np.ctypes.data,
                rms_att.ctypes.data, rms_ffn.ctypes.data
            )
        
        # Set embedding and final norm
        embed_np = store.embed_f16.detach().cpu().numpy()
        self.engine.set_embedding(embed_np.ctypes.data)
        
        final_norm_np = store.final_norm.detach().cpu().float().numpy()
        self.engine.set_final_norm(final_norm_np.ctypes.data)
        
        # LM head weight as FP16 torch tensor (half memory bandwidth vs FP32)
        self.lm_head_weight = store.embed_f16  # [vocab, hidden] FP16, zero-copy
    
    def forward_hidden(self, token_id, pos):
        """Forward one token through transformer layers, return hidden state."""
        return self.engine.forward_hidden(token_id, pos)
    
    def compute_logits(self, hidden_np):
        """Compute logits via PyTorch FP16 matmul (half memory bandwidth vs FP32)."""
        hidden_t = torch.from_numpy(hidden_np).unsqueeze(0).half()  # [1, hidden] FP16
        logits = torch.matmul(hidden_t, self.lm_head_weight.T)  # [1, vocab] FP16
        return logits.squeeze(0).float().numpy()
    
    def generate(self, prompt_tokens, max_new_tokens):
        """Run full autoregressive generation entirely in C++ (zero Python overhead)."""
        # Use the C++ engine's generate() method directly — no Python loop
        return self.engine.generate(list(prompt_tokens), max_new_tokens)
