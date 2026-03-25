import re

with open('experiments/phi4_cpu_run.py', 'r') as f:
    text = f.read()

def replace_func(func_name, new_code):
    global text
    pattern = r'    def ' + func_name + r'\(.*?(?=\n    def |\Z)'
    m = re.search(pattern, text, re.DOTALL)
    if m:
        text = text.replace(m.group(0), new_code + '\n')
        print(f"Replaced {func_name}")
    else:
        print(f"NOT FOUND: {func_name}")

replace_func('_matmul_quant_batch', '''    def _matmul_quant_batch(self, layer_idx: int, name: str,
                            X_batch: torch.Tensor) -> torch.Tensor:
        """Batched dequant+matmul: Y = W @ X^T, loading W once for K tokens."""
        key = (layer_idx, name)
        u8 = self._quant_u8[key]
        sc = self._quant_sc_f32[key]
        bi = self._quant_bi_f32[key]
        rows, cols = self._quant_shapes[key]
        groups_per_row = cols // self.group_size

        K_batch = X_batch.shape[0]
        X_flat = X_batch.reshape(K_batch, cols)

        chunk_rows = max(1, _TARGET_CHUNK_BYTES // (cols * 4))
        result = torch.empty(rows, K_batch, dtype=torch.float32)

        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            n = end - start
            flat_len = n * cols
            buf = self._pool[:flat_len]
            buf.copy_(u8[start * cols:end * cols])
            vals = buf.view(n, groups_per_row, self.group_size)
            gs = start * groups_per_row
            ge = end * groups_per_row
            vals.mul_(sc[gs:ge].view(n, groups_per_row, 1))
            vals.add_(bi[gs:ge].view(n, groups_per_row, 1))
            torch.mm(vals.view(n, cols), X_flat.T, out=result[start:end, :])        

        return result.T''')

with open('experiments/phi4_cpu_run.py', 'w') as f:
    f.write(text)
