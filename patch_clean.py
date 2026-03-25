import codecs
code = codecs.open('asdsl/kernels/forward_loop.cpp', 'r', 'utf-16le').read()
code = code.replace('    float max_val = logits[0]; printf("DEBUG GENERATE seq_pos: %d \\n", seq_pos); fflush(stdout);\r\n', '    float max_val = logits[0];\r\n')
codecs.open('asdsl/kernels/forward_loop.cpp', 'w', 'utf-16le').write(code)
