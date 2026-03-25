import re

with open('experiments/phi4_cpu_run.py', 'r') as f:
    text = f.read()

def replace_func(func_name, new_code):
    global text
    # Match the function definition until the next method starting with `    def `
    pattern = r'    def ' + func_name + r'\(.*?(?=\n    def |\Z)'
    m = re.search(pattern, text, re.DOTALL)
    if m:
        text = text.replace(m.group(0), new_code + '\n')
        print(f"Replaced {func_name}")
    else:
        print(f"NOT FOUND: {func_name}")

replace_func('_matvec_quant', '''    def _matvec_quant(self, layer_idx: int, name: str, x: torch.Tensor) -> torch.Tensor:
        key = (layer_idx, name)
        u8 = self._quant_u8[key]
        sc = self._quant_sc_f32[key]
        bi = self._quant_bi_f32[key]
        rows, cols = self._quant_shapes[key]
        groups_per_row = cols // self.group_size
        x_flat = x.view(-1)                    # single view, no copy
        chunk_rows = max(1, _TARGET_CHUNK_BYTES // (cols * 4))
        result = torch.empty(rows, dtype=torch.float32)
        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            n   = end - start
            flat_len = n * cols
            buf = self._pool[:flat_len]
            buf.copy_(u8[start * cols:end * cols])
            vals = buf.view(n, groups_per_row, self.group_size)
            gs_s = start * groups_per_row
            gs_e = end   * groups_per_row
            vals.mul_(sc[gs_s:gs_e].view(n, groups_per_row, 1))
            vals.add_(bi[gs_s:gs_e].view(n, groups_per_row, 1))
            torch.mv(vals.view(n, cols), x_flat, out=result[start:end])
        return result.unsqueeze(0)''')

with open('experiments/phi4_cpu_run.py', 'w') as f:
    f.write(text)
