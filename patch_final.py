import re

with open('experiments/phi4_cpu_run.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Add _quant_w_f32 map to init 
if 'self._quant_w_f32: Dict[Tuple[int, str], torch.Tensor] = {}' not in text:
    text = text.replace(
        'self._quant_bi_f32_np: Dict[Tuple[int, str], np.ndarray] = {}',
        'self._quant_bi_f32_np: Dict[Tuple[int, str], np.ndarray] = {}\n        self._quant_w_f32: Dict[Tuple[int, str], torch.Tensor] = {}'
    )

# Fix double insertion in warm_cache
text = re.sub(
    r'                    self\._quant_u8_np\[key\] = self\._quant_u8\[key\]\.numpy\(\).*?self\._quant_bi_f32\[key\] = bi_f16\.float\(\)',
    '''                    # PRE-CALCULATE FULL F32 TENSOR (Zero conversion overhead)
                    u8_vals = unpacked.astype(np.float32).reshape(rows, n_groups, qt.group_size)
                    sc_f32 = sc_f16.float().numpy().reshape(rows, n_groups, 1)
                    bi_f32 = bi_f16.float().numpy().reshape(rows, n_groups, 1)
                    w_vals = u8_vals * sc_f32 + bi_f32
                    self._quant_w_f32[key] = torch.from_numpy(w_vals.reshape(rows, cols))''',
    text,
    flags=re.DOTALL
)

# And now switch _matvec_quant to use it
text = re.sub(
    r'    def _matvec_quant\(self, layer_idx: int, name: str, x: torch\.Tensor, use_draft: bool = False\) -> torch\.Tensor:.*?return res\.unsqueeze\(0\)',
    '''    def _matvec_quant(self, layer_idx: int, name: str, x: torch.Tensor, use_draft: bool = False) -> torch.Tensor:
        key = (layer_idx, name)
        
        # Pull pre-computed f32 from warm_cache mappings
        # (Zero conversion overhead!)
        q_weights = self._quant_w_f32[key] # fully f32 unrolled torch tensor
        
        # Pytorch threading takes over automatically here via underlying MKL
        x_flat = x.view(-1).float()
        res = torch.mv(q_weights, x_flat)
        return res.unsqueeze(0)''',
    text,
    flags=re.DOTALL
)

# And _matmul_quant_batch
text = re.sub(
    r'    def _matmul_quant_batch\(self, layer_idx: int, name: str, x: torch\.Tensor, use_draft: bool = False\) -> torch\.Tensor:.*?return res\.view\(\*x\.shape\[:-1\], -1\)',
    '''    def _matmul_quant_batch(self, layer_idx: int, name: str, x: torch.Tensor, use_draft: bool = False) -> torch.Tensor:
        key = (layer_idx, name)
        q_weights = self._quant_w_f32[key]
        x_flat = x.view(-1, x.shape[-1]).float()
        res = torch.mm(x_flat, q_weights.t())
        return res.view(*x.shape[:-1], -1)''',
    text,
    flags=re.DOTALL
)


with open('experiments/phi4_cpu_run.py', 'w', encoding='utf-8') as f:
    f.write(text)
print("done")
