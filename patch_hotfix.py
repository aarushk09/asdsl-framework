lines = open('experiments/phi4_cpu_run.py', 'r').readlines()
for i, l in enumerate(lines):
    if 'next_token = int(logits.argmax())' in l:
        lines[i] = '            if step == 0: print("[DEBUG] GT logits[:10] =", logits.reshape(-1)[:10].tolist())\n            next_token = int(logits.argmax())\n'
with open('experiments/phi4_cpu_run.py', 'w') as f:
    f.writelines(lines)
