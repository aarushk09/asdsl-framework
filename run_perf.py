import numpy as np
import time
from asdsl.kernels._native_unified import EngineConfig, UnifiedEngine

config = EngineConfig()
config.max_seq_len = 2048
config.vocab_size = 100352
config.hidden_size = 3072
config.num_layers = 40
config.num_heads = 32
config.num_kv_heads = 32
config.intermediate_size = 8192
config.rms_norm_eps = 1e-5
config.group_size = 32
config.head_dim = 96
config.rotary_dim = 96

engine = UnifiedEngine(config)

# Dummy allocations
W_embed = np.zeros((config.vocab_size, config.hidden_size), dtype=np.float16)
W_lm_head = np.zeros((config.vocab_size, config.hidden_size), dtype=np.float16)
W_norm = np.ones(config.hidden_size, dtype=np.float32)

engine.set_embedding(W_embed)
engine.set_lm_head(W_norm, W_lm_head)

q_w = np.zeros((config.hidden_size * 3, config.hidden_size), dtype=np.uint8)
o_w = np.zeros((config.hidden_size, config.hidden_size), dtype=np.float32)

q_scales = np.ones(config.hidden_size * 3 * config.hidden_size // 32, dtype=np.float32)
q_zeros = np.zeros(config.hidden_size * 3 * config.hidden_size // 32, dtype=np.float32)
q_bits = np.full(config.hidden_size * 3 * config.hidden_size // 32, 4, dtype=np.int32)
norm_w = np.ones(config.hidden_size, dtype=np.float32)

m_w = np.zeros((config.intermediate_size * 2, config.hidden_size), dtype=np.uint8)
m_o = np.zeros((config.hidden_size, config.intermediate_size), dtype=np.float32)
m_scales = np.ones(config.intermediate_size * 2 * config.hidden_size // 32, dtype=np.float32)
m_zeros = np.zeros(config.intermediate_size * 2 * config.hidden_size // 32, dtype=np.float32)
m_bits = np.full(config.intermediate_size * 2 * config.hidden_size // 32, 4, dtype=np.int32)
m_norm = np.ones(config.hidden_size, dtype=np.float32)

for i in range(config.num_layers):
    engine.add_layer_norm(i, True, norm_w)
    engine.add_layer_norm(i, False, m_norm)
    # The actual C++ signatures need to be known. 
    # Let's hope UnifiedEngine expects these arrays correctly. 
    # wait, python dir() gave us the methods...
