import re
with open('test_unified_batch.py', 'r') as f:
    text = f.read()
text = re.sub(r'cos_table =.*', 'cos_table = (np.random.randn(config.max_seq_len, config.head_dim//2) * 0.001).astype(np.float32)', text)
text = re.sub(r'sin_table =.*', 'sin_table = (np.random.randn(config.max_seq_len, config.head_dim//2) * 0.001).astype(np.float32)', text)
with open('test_unified_batch.py', 'w') as f:
    f.write(text)
