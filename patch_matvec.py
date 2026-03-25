import re

with open('experiments/phi4_cpu_run.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = re.sub(
    r'    def _matvec_quant\(self, layer_idx: int, name: str, x: torch\.Tensor\) -> torch\.Tensor:.*?return result\.unsqueeze\(0\)',
    '''    def _matvec_quant(self, layer_idx: int, name: str, x: torch.Tensor, use_draft: bool = False) -> torch.Tensor:
        key = (layer_idx, name)
        
        # Pull pre-computed f32 from warm_cache mappings
        # (Zero conversion overhead!)
        q_weights = self._quant_sc_f32_np[key] # fully f32 unrolled tensor backing! wait, no I stored numpy array
        
        # Pytorch threading takes over automatically here via underlying MKL
        x_flat = x.view(-1)
        res = torch.mv(torch.from_numpy(q_weights), x_flat)
        return res.unsqueeze(0)''',
    text,
    flags=re.DOTALL
)

text = re.sub(
    r'    def _matmul_quant_batch\(self, layer_idx: int, name: str, x: torch\.Tensor\) -> torch\.Tensor:.*?return result',
    '''    def _matmul_quant_batch(self, layer_idx: int, name: str, x: torch.Tensor, use_draft: bool = False) -> torch.Tensor:
        key = (layer_idx, name)
        q_weights = self._quant_sc_f32_np[key]
        x_flat = x.view(-1, x.shape[-1])
        res = torch.mm(x_flat, torch.from_numpy(q_weights).t())
        return res.view(*x.shape[:-1], -1)''',
    text,
    flags=re.DOTALL
)


with open('experiments/phi4_cpu_run.py', 'w', encoding='utf-8') as f:
    f.write(text)
print("Updated matvec")