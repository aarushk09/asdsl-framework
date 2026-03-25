with open('experiments/phi4_cpu_run.py', 'r', encoding='utf-8') as f:
    text = f.read()

import re

# in init
text = text.replace(
    'self._use_native_gemv = False',
    'self._use_native_gemv = False\n        self._use_native_draft_gemv = False'
)

# in warm_cache
text = text.replace(
    '''        try:
            from asdsl.kernels import (
                has_native_kernel, has_native_q8_kernel,
                has_native_q3_kernel, has_native_q2_kernel,
            )
            if self.bits == 4:
                has_gemv = has_native_kernel()
            elif self.bits == 8:
                has_gemv = has_native_q8_kernel()
            elif self.bits == 3:
                has_gemv = has_native_q3_kernel()
            elif self.bits == 2:
                has_gemv = has_native_q2_kernel()
            else:
                has_gemv = False
        except ImportError:
            has_gemv = False

        self._use_native_gemv = has_gemv''',
    '''        self._use_native_gemv = False  # primary model always uses PyTorch BLAS

        # Only enable native kernel for the draft (2-bit packed) path
        self._use_native_draft_gemv = False
        try:
            from asdsl.kernels import has_native_q2_kernel
            self._use_native_draft_gemv = has_native_q2_kernel() if self.enable_qcsd else False
        except ImportError:
            pass'''
)

text = re.sub(
    r'        total = NUM_LAYERS \* 4\n        if self\.bits == 16:\n            print\(f"  Float16 weight cache already populated \(\{total\} tensors\)"\)\n            print\("  Inference: chunked f16 matvec"\)\n        else:',
    '''        total = NUM_LAYERS * 4
        if self.bits == 16:
            print(f"  Float16 weight cache already populated ({total} tensors)")
            print("  Inference: chunked f16 matvec (PyTorch BLAS)")
        elif self._use_native_draft_gemv:
            print(f"  Inference: PyTorch BLAS Q{self.bits} primary | native Q2 draft")
        else:
            print(f"  Inference: PyTorch BLAS Q{self.bits}")

        if self.bits != 16:''',
    text
)

with open('experiments/phi4_cpu_run.py', 'w', encoding='utf-8') as f:
    f.write(text)
print("Updated flag")
