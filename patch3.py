import re
with open('test_unified_batch.py', 'r') as f:
    text = f.read()

# Replace all np.random.randn/randint
text = text.replace(
'''def rand_q_w(out_f, in_f):
    return np.random.randint(0, 255, size=(out_f, in_f//2), dtype=np.uint8)''',
'''def rand_q_w(out_f, in_f):
    return np.full((out_f, in_f//2), 127, dtype=np.uint8)''')

text = text.replace(
'''def rand_s(out_f, in_f):
    return np.ones((out_f, in_f//config.group_size), dtype=np.float32) * 0.01''',
'''def rand_s(out_f, in_f):
    return np.full((out_f, in_f//config.group_size), 0.001, dtype=np.float32)''')

with open('test_unified_batch.py', 'w') as f:
    f.write(text)
