import sys
file_path = 'experiments/phi4_cpu_run.py'
with open(file_path, 'r') as f:
    content = f.read()
content = content.replace('qkv = store.matmul(layer_idx, "qkv_proj", h)', 'qkv = store.matmul(layer_idx, "qkv_proj", h)
    if layer_idx == 0 and pos == 0:
        print(f"[DEBUG] Py qkv dt: {qkv[:10].tolist()}")')
content = content.replace('hidden = residual + o_proj', 'hidden = residual + o_proj
    if layer_idx == 0 and pos == 0:
        print(f"[DEBUG] Py o_proj dt: {o_proj[:10].tolist()}")')
content = content.replace('att_out = torch.einsum("bnqd,bndk->bnqk", att_softmax, v_full)', 'att_out = torch.einsum("bnqd,bndk->bnqk", att_softmax, v_full)
    if layer_idx == 0 and pos == 0:
        out_h = att_out.transpose(1, 2).reshape(-1)[0:10]
        print(f"[DEBUG] Py att_out dt: {out_h.tolist()}")')
with open(file_path, 'w') as f:
    f.write(content)
