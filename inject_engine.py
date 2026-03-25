import re

with open('asdsl/kernels/forward_loop.cpp', 'r') as f:
    content = f.read()

engine_code = """
int generate_token(int token_id, int seq_pos, 
                   const float* token_emb,
                   const std::vector<const float*>& rms1_ws,
                   const std::vector<const float*>& qkv_ws,
                   const std::vector<const float*>& o_ws,
                   const std::vector<const float*>& rms2_ws,
                   const std::vector<const float*>& gate_ws,
                   const std::vector<const float*>& up_ws,
                   const std::vector<const float*>& down_ws,
                   const float* final_rms_w,
                   const float* lm_head_w,
                   int num_layers, int dim, int hidden_dim, int num_heads, int num_kv_heads, int head_dim, int vocab_size,
                   KVCache& cache) {
    
    std::vector<float> x(dim);
    memcpy(x.data(), token_emb + (size_t)token_id * dim, dim * sizeof(float));

    for (int l = 0; l < num_layers; ++l) {
        forward_layer(x.data(), rms1_ws[l], qkv_ws[l], o_ws[l], rms2_ws[l], gate_ws[l], up_ws[l], down_ws[l],
                      dim, hidden_dim, num_heads, num_kv_heads, head_dim, 
                      l, seq_pos, cache);
    }

    apply_rmsnorm(x.data(), final_rms_w, dim, 1e-5f);

    std::vector<float> logits(vocab_size, 0.0f);
    gemv_f32(lm_head_w, x.data(), logits.data(), vocab_size, dim);

    int max_id = 0;
    float max_val = logits[0];
    for (int i = 1; i < vocab_size; ++i) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_id = i;
        }
    }

    return max_id;
}

int py_generate_token(int token_id, int seq_pos, 
                      py::array_t<float> token_emb,
                      py::list layers_weights,
                      py::array_t<float> final_rms_w,
                      py::array_t<float> lm_head_w,
                      int num_layers, int dim, int hidden_dim, int num_heads, int num_kv_heads, int head_dim, int vocab_size,
                      KVCache& cache) {
    
    py::buffer_info emb_buf = token_emb.request();
    py::buffer_info frms_buf = final_rms_w.request();
    py::buffer_info lm_buf = lm_head_w.request();

    std::vector<const float*> rms1_ws, qkv_ws, o_ws, rms2_ws, gate_ws, up_ws, down_ws;
    std::vector<py::array_t<float>> refs;

    for (int l = 0; l < num_layers; ++l) {
        py::tuple layer = layers_weights[l].cast<py::tuple>();
        refs.push_back(layer[0].cast<py::array_t<float>>()); rms1_ws.push_back((const float*)refs.back().request().ptr);
        refs.push_back(layer[1].cast<py::array_t<float>>()); qkv_ws.push_back((const float*)refs.back().request().ptr);
        refs.push_back(layer[2].cast<py::array_t<float>>()); o_ws.push_back((const float*)refs.back().request().ptr);
        refs.push_back(layer[3].cast<py::array_t<float>>()); rms2_ws.push_back((const float*)refs.back().request().ptr);
        refs.push_back(layer[4].cast<py::array_t<float>>()); gate_ws.push_back((const float*)refs.back().request().ptr);
        refs.push_back(layer[5].cast<py::array_t<float>>()); up_ws.push_back((const float*)refs.back().request().ptr);
        refs.push_back(layer[6].cast<py::array_t<float>>()); down_ws.push_back((const float*)refs.back().request().ptr);
    }

    return generate_token(token_id, seq_pos, 
                          (const float*)emb_buf.ptr,
                          rms1_ws, qkv_ws, o_ws, rms2_ws, gate_ws, up_ws, down_ws,
                          (const float*)frms_buf.ptr, (const float*)lm_buf.ptr,
                          num_layers, dim, hidden_dim, num_heads, num_kv_heads, head_dim, vocab_size,
                          cache);
}

PYBIND11_MODULE(_native_forward, m) {"""

content = content.replace("PYBIND11_MODULE(_native_forward, m) {", engine_code)

target = 'm.def("forward_layer", &py_forward_layer, "Forward layer execution");'
replacement = target + '\n    m.def("generate_token", &py_generate_token, "Generate a single token");'
content = content.replace(target, replacement)

with open('asdsl/kernels/forward_loop.cpp', 'w') as f:
    f.write(content)

print("C++ Orchestrator logic seamlessly injected.")
