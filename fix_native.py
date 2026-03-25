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

with open('experiments/phi4_cpu_run.py', 'w') as f:
    f.write(text)
