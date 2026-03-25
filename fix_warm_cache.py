import re

with open('experiments/phi4_cpu_run.py', 'r') as f:
    text = f.read()

m = re.search(r'        try:\n            from asdsl\.kernels import \([\s\S]*?            else:\n                has_gemv = False\n        except ImportError:\n            has_gemv = False\n\n        self\._use_native_gemv = has_gemv', text)
if m:
    rep = '''        self._use_native_gemv = False  # primary model always uses PyTorch BLAS

        # Only enable native kernel for the draft (2-bit packed) path
        self._use_native_draft_gemv = False
        try:
            from asdsl.kernels import has_native_q2_kernel
            self._use_native_draft_gemv = has_native_q2_kernel() if self._enable_qcsd else False
        except ImportError:
            pass'''
    text = text.replace(m.group(0), rep)
    
    m2 = re.search(r'        if self\.bits == 16:\n[\s\S]*?            print\("  Inference: chunked f16 matvec"\)\n        else:\n            done = 0', text)

    if m2:
        rep2 = f'''        if self.bits == 16:
            print(f"  Float16 weight cache already populated ({{total}} tensors)")
            print("  Inference: chunked f16 matvec (PyTorch BLAS)")
        else:
            if self._use_native_draft_gemv:
                print(f"  Inference: PyTorch BLAS Q{{self.bits}} primary | native Q{{self._draft_bits}} draft")
            else:
                print(f"  Inference: PyTorch BLAS Q{{self.bits}}")
            done = 0'''
        text = text.replace(m2.group(0), rep2)

    with open('experiments/phi4_cpu_run.py', 'w') as f:
        f.write(text)
    print("Fixed has_gemv and prints")
else:
    print("Not found")
