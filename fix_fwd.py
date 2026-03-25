
code = open('asdsl/kernels/forward_loop.cpp', 'r', encoding='utf-16le').read()
# find gemv_q4_avx2_row
import re
match = re.search(r'void gemv_q4_avx2_row(.*?)\}', code, re.DOTALL)
if match:
    print(match.group(0))
