import re
with open('asdsl/kernels/native/unified_engine.cpp', 'r') as f:
    code = f.read()

code = re.sub(r'(std::cout << "\s*\[DB\].*?;)', r'// \1', code)
code = re.sub(r'(std::cout << "Token ".*?<< "... ";)', r'// \1', code)
code = re.sub(r'(std::cout << "forward_token complete. ";)', r'// \1', code)
code = re.sub(r'(std::cout << std::endl;)', r'// \1', code)

with open('asdsl/kernels/native/unified_engine.cpp', 'w') as f:
    f.write(code)
