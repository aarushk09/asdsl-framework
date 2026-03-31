#!/usr/bin/env python3
# Fix paren count in set_thread_count function

with open('experiments/phi4_cpu_run.py', 'r') as f:
    content = f.read()

# Fix: 5 closing parens -> 4
old = 'p.cpu_affinity(list(range(min(n, 16)))))'
new = 'p.cpu_affinity(list(range(min(n, 16))))'
count = content.count(old)
print(f"Found: {count}")
if count > 0:
    content = content.replace(old, new)
    print("Fixed!")
else:
    print("Pattern not found, checking for variant...")
    # Try to find the line
    for i, line in enumerate(content.split('\n')):
        if 'cpu_affinity' in line and 'min(n' in line:
            print(f"Line {i+1}: {line}")

try:
    compile(content, 'experiments/phi4_cpu_run.py', 'exec')
    print("Syntax check: PASSED")
    with open('experiments/phi4_cpu_run.py', 'w') as f:
        f.write(content)
except SyntaxError as e:
    print(f"Syntax error at line {e.lineno}: {e.msg}")
    print(f"  {e.text}")
    lines = content.split('\n')
    for i in range(max(0, e.lineno-2), min(len(lines), e.lineno+2)):
        print(f"  {i+1}: {lines[i]}")
