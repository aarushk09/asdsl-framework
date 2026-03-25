import re
with open('experiments/phi4_cpu_run.py', 'r', encoding='utf-8') as f:
    text = f.read()

m2 = re.search(r'                    self\._quant_sc\[key\] = sc_f16\n                    self\._quant_bi\[key\] = bi_f16\n', text)
if m2:
    rep2 = '''                    self._quant_sc[key] = sc_f16
                    self._quant_bi[key] = bi_f16
                    self._quant_u8_np[key] = self._quant_u8[key].numpy()
                    self._quant_sc_f32_np[key] = sc_f16.float().numpy().copy()
                    self._quant_bi_f32_np[key] = bi_f16.float().numpy().copy()
                    self._quant_sc_f32[key] = sc_f16.float()
                    self._quant_bi_f32[key] = bi_f16.float()
'''
    text = text.replace(m2.group(0), rep2)

m3 = re.search(r'                        self\._draft_quant_sc\[key\] = d_sc\n                        self\._draft_quant_bi\[key\] = d_bi\n', text)
if m3:
    rep3 = '''                        self._draft_quant_sc[key] = d_sc
                        self._draft_quant_bi[key] = d_bi
                        self._draft_u8_np[key] = self._draft_quant_u8[key].numpy()
                        self._draft_sc_f32_np[key] = d_sc.float().numpy().copy()
                        self._draft_bi_f32_np[key] = d_bi.float().numpy().copy()
'''
    text = text.replace(m3.group(0), rep3)

with open('experiments/phi4_cpu_run.py', 'w', encoding='utf-8') as f:
    f.write(text)
