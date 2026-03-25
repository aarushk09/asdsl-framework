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

replace_func('matvec', '''    def matvec(self, layer_idx: int, name: str, x: torch.Tensor,
               use_draft: bool = False) -> torch.Tensor:
        if use_draft and self._use_native_draft_gemv:
            return self._matvec_native_gemv(layer_idx, name, x, use_draft=True)
        if self.bits == 16 and not use_draft:
            return self._matvec_f16(layer_idx, name, x)
        return self._matvec_quant(layer_idx, name, x)   # PyTorch multi-threaded''')

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
            is_packed = True
        else:
            w_data = self._quant_u8_np[key]
            sc     = self._quant_sc_f32_np[key]
            bi     = self._quant_bi_f32_np[key]
            rows, cols = self._quant_shapes[key]
            bits = self.bits
            gs   = self.group_size
            is_packed = False

        x_np = x.view(-1).float().contiguous().numpy()   # single conversion

        if is_packed and bits == 2:
            from asdsl.kernels import gemv_q2_packed
            out_np = gemv_q2_packed(w_data, x_np, sc, bi, rows, cols, gs)
        elif bits == 4:
            from asdsl.kernels import gemv_q4_unpacked
            out_np = gemv_q4_unpacked(w_data, x_np, sc, bi, rows, cols, gs)
        elif bits == 8:
            from asdsl.kernels import gemv_q8_unpacked
            out_np = gemv_q8_unpacked(w_data, x_np, sc, bi, rows, cols, gs)
        elif bits == 3:
            from asdsl.kernels import gemv_q3_unpacked
            out_np = gemv_q3_unpacked(w_data, x_np, sc, bi, rows, cols, gs)
        elif bits == 2:
            from asdsl.kernels import gemv_q2_unpacked
            out_np = gemv_q2_unpacked(w_data, x_np, sc, bi, rows, cols, gs)
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

