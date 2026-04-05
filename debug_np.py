import numpy as np
import json

meta = json.load(open('phi4_slim_meta.json'))
weights = np.load('phi4_slim_meta.npz')
emb = weights['token_embd'][100264]
print(f'[DEBUG] Py emb dt: {emb[:10].tolist()}')
norm_w = weights['layers.0.input_layernorm.weight']
rms = np.sqrt(np.mean(emb**2) + 1e-5)
norm_emb = (emb / rms) * norm_w
print(f'[DEBUG] Py norm1 dt: {norm_emb[:10].tolist()}')
qkv_w = np.vstack([weights['layers.0.self_attn.q_proj.weight'], weights['layers.0.self_attn.k_proj.weight'], weights['layers.0.self_attn.v_proj.weight']])

# qkv_w is (qkv_dim, hidden_size)
qkv = norm_emb @ qkv_w.T
print(f'[DEBUG] Py qkv dt: {qkv[:10].tolist()}')

