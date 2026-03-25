import re
with open('experiments/phi4_cpu_run.py', 'r', encoding='utf-8') as f:
    text = f.read()

m_init = re.search(r'        self\._quant_shapes: dict\[tuple, tuple\] = \{\}\n\n        # Native LUT/GEMV fast path\n        self\._use_native_gemv = False', text)
if m_init:
    rep_init = '''        self._quant_shapes: dict[tuple, tuple] = {}

        self._quant_sc_f32_np:  dict[tuple, np.ndarray] = {}
        self._quant_bi_f32_np:  dict[tuple, np.ndarray] = {}
        self._quant_u8_np:      dict[tuple, np.ndarray] = {}
        self._draft_sc_f32_np:  dict[tuple, np.ndarray] = {}
        self._draft_bi_f32_np:  dict[tuple, np.ndarray] = {}
        self._draft_u8_np:      dict[tuple, np.ndarray] = {}
        
        self._quant_sc_f32: dict[tuple, torch.Tensor] = {}
        self._quant_bi_f32: dict[tuple, torch.Tensor] = {}

        # Native LUT/GEMV fast path
        self._use_native_gemv = False
        self._use_native_draft_gemv = False'''
    text = text.replace(m_init.group(0), rep_init)

m_try = re.search(r'        try:\n            from asdsl\.kernels import \(\n.*?            has_gemv = False\n\n        self\._use_native_gemv = has_gemv', text, re.DOTALL)
if m_try:
    rep_try = '''        self._use_native_gemv = False  # primary model always uses PyTorch BLAS

        # Only enable native kernel for the draft (2-bit packed) path
        self._use_native_draft_gemv = False
        try:
            from asdsl.kernels import has_native_q2_kernel
            self._use_native_draft_gemv = has_native_q2_kernel() if self._enable_qcsd else False
        except ImportError:
            pass'''
    text = text.replace(m_try.group(0), rep_try)

m_print = re.search(r'        total = NUM_LAYERS \* 4\n        if self\.bits == 16:\n            print\(f"  Float16 weight cache already populated \(\{total\} tensors\)"\)\n            print\("  Inference: chunked f16 matvec"\)\n        else:\n', text, re.DOTALL)
if m_print:
    rep_print = '''        total = NUM_LAYERS * 4
        if self.bits == 16:
            print(f"  Float16 weight cache already populated ({total} tensors)")
            print("  Inference: chunked f16 matvec (PyTorch BLAS)")
        elif self._use_native_draft_gemv:
            print(f"  Inference: PyTorch BLAS Q{self.bits} primary | native Q{self._draft_bits} draft")
        else:
            print(f"  Inference: PyTorch BLAS Q{self.bits}")

        if self.bits != 16:\n'''
    text = text.replace(m_print.group(0), rep_print)

text = re.sub(r'            kernel_labels = \{4: "Q4", 8: "Q8", 3: "Q3", 2: "Q2"\}.*?print\(f"  Inference: chunked uint8 dequant\+matvec \(in-place, no AVX GEMV\)"\)\n', '', text, flags=re.DOTALL)

with open('experiments/phi4_cpu_run.py', 'w', encoding='utf-8') as f:
    f.write(text)
