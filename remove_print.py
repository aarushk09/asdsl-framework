import re
with open('experiments/phi4_cpu_run.py', 'r') as f:
    text = f.read()

pattern = r'            kernel_labels = \{4: "Q4", 8: "Q8", 3: "Q3", 2: "Q2"\}[\s\S]*?print\(f"  Inference: chunked uint8 dequant\+matvec \(in-place, no AVX GEMV\)"\)'
m = re.search(pattern, text)
if m:
    text = text.replace(m.group(0), '')
    with open('experiments/phi4_cpu_run.py', 'w') as f:
        f.write(text)
    print("Cleaned up old print")
else:
    print("Not found")

