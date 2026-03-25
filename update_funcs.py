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

replace_func('matvec', '''    def matvec(self, layer_idx: int, name: str, x: torch.Tensor,
               use_draft: bool = False) -> torch.Tensor:
        if use_draft and self._use_native_draft_gemv:
            return self._matvec_native_gemv(layer_idx, name, x, use_draft=True)
        if self.bits == 16 and not use_draft:
            return self._matvec_f16(layer_idx, name, x)
        return self._matvec_quant(layer_idx, name, x)   # PyTorch multi-threaded''')

replace_func('_matvec_quant', '''    def _matvec_quant(self, layer_idx: int, name: str, x: torch.Tensor) -> torch.Tensor:
        """De-quantize chunk bypass to f32 on-the-fly and rely on multi-threaded mv."""
        key = (layer_idx, name)
        u8 = self._quant_u8[key]
        sc = self._quant_sc_f32[key]
        bi = self._quant_bi_f32[key]
        rows, cols = self._quant_shapes[key]
        groups_per_row = cols // self.group_size
        x_flat = x.view(-1)                    # single view, no copy
        
        # Determine chunk size. Must be multiple of cols.
        chunk_rows = max(1, 16384 // (cols * 4)) # Example chunk_rows handling
        try:
            # Get _TARGET_CHUNK_BYTES from the module
            from asdsl.inference.engine import _TARGET_CHUNK_BYTES
            chunk_rows = max(1, _TARGET_CHUNK_BYTES // (cols * 4))
        except ImportError:
            chunk_rows = 128
        
        result = torch.empty(rows, dtype=torch.float32)
        
        # Re-use python loop but with f32 bias/scale
        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            n   = end - start
            flat_len = n * cols
            
            # Temporary buffer
            buf = torch.empty(flat_len, dtype=torch.uint8) # Or use pool if it exists
            if hasattr(self, '_pool'):
                buf = self._pool[:flat_len]
            
            buf.copy_(u8[start * cols:end * cols])
            vals = buf.view(n, groups_per_row, self.group_size).to(torch.float32)
            gs_s = start * groups_per_row
            gs_e = end   * groups_per_row
            
            vals.mul_(sc[gs_s:gs_e].view(n, groups_per_row, 1))
            vals.add_(bi[gs_s:gs_e].view(n, groups_per_row, 1))
            
            torch.mv(vals.view(n, cols), x_flat, out=result[start:end])
        return result.unsqueeze(0)''')

replace_func('_matvec_native_gemv', '''    def _matvec_native_gemv(self, layer_idx: int, name: str, x: torch.Tensor,
                            use_draft: bool = False) -> torch.Tensor:
        """Call native ASDSL AVX2 C-extension GEMV."""
        key = (layer_idx, name)

        if use_draft and key in self._draft_u8_np:
            w_data = self._draft_u8_np[key]        # no .numpy() call
            sc     = self._draft_sc_f32_np[key]    # already f32 numpy
            bi     = self._draft_bi_f32_np[key]
            rows, cols = self._quant_shapes[key]
            bits = self._draft_bits
            gs   = self._draft_group_size
        else:
            w_data = self._quant_u8_np[key]
            sc     = self._quant_sc_f32_np[key]
            bi     = self._quant_bi_f32_np[key]
            rows, cols = self._quant_shapes[key]
            bits = self.bits
            gs   = self.group_size

        x_np = x.view(-1).float().contiguous().numpy()   # single conversion

        if bits == 4:
            from asdsl.kernels import gemv_q4
            out_np = gemv_q4(w_data, x_np, sc, bi, rows, cols, gs)
        elif bits == 8:
            from asdsl.kernels import gemv_q8
            out_np = gemv_q8(w_data, x_np, sc, bi, rows, cols, gs)
        elif bits == 3:
            from asdsl.kernels import gemv_q3
            out_np = gemv_q3(w_data, x_np, sc, bi, rows, cols, gs)
        elif bits == 2:
            from asdsl.kernels import gemv_q2
            out_np = gemv_q2(w_data, x_np, sc, bi, rows, cols, gs)
        else:
            raise ValueError(f"No native kernel for {bits}-bit")

        result = torch.from_numpy(out_np).unsqueeze(0)

        # Outlier
        ovals = self._draft_outlier_values if use_draft else self._outlier_values
        ocoords = self._draft_outlier_coords if use_draft else self._outlier_coords

        if key in ovals and len(ovals[key]) > 0:
            ov = ovals[key].astype(np.float32)
            oc = ocoords[key]
            x_sel = x_np[oc[:, 1]]
            contributions = ov * x_sel
            out_corr = np.zeros(rows, dtype=np.float32)
            np.add.at(out_corr, oc[:, 0], contributions)
            result = result + torch.from_numpy(out_corr).unsqueeze(0)

        return result''')

with open('experiments/phi4_cpu_run.py', 'w') as f:
    f.write(text)

