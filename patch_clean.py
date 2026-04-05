import sys
file_path = 'asdsl/kernels/native/unified_engine.cpp'
with open(file_path, 'r') as f:
    lines = f.readlines()
with open(file_path, 'w') as f:
    f.writelines([line for line in lines if 'std::vector<float> b_swiglu_out(num_tokens * config_.intermediate_size);' not in line or line.startswith('    std::vector<float>')])
