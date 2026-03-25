import re

with open('asdsl/kernels/forward_loop.cpp', 'r') as f:
    text = f.read()

replacement2 = \"\"\"
// Pybind wrappers for TDD
void py_kvc_set_history(KVCache& cache, int layer, int pos, py::array_t<float> k, py::array_t<float> v) {
    py::buffer_info k_buf = k.request();
    py::buffer_info v_buf = v.request();
    cache.set_history(layer, pos, (const float*)k_buf.ptr, (const float*)v_buf.ptr);
}

py::array_t<float> py_compute_attention(py::array_t<float> q, py::array_t<float> k, py::array_t<float> v, int layer_id, int seq_pos, int num_heads, KVCache& cache) {
    py::buffer_info q_buf = q.request();
    py::buffer_info k_buf = k.request();
    py::buffer_info v_buf = v.request();
    auto out = py::array_t<float>(q_buf.size);
    py::buffer_info out_buf = out.request();
    compute_attention((float*)out_buf.ptr, (const float*)q_buf.ptr, (const float*)k_buf.ptr, (const float*)v_buf.ptr, layer_id, seq_pos, num_heads, cache);
    return out;
}

void py_apply_rmsnorm(py::array_t<float> x, py::array_t<float> weight, float eps) {
\"\"\"
text = re.sub(r'// Pybind wrappers for TDD[\s\S]*?void py_apply_rmsnorm\(py::array_t<float> x, py::array_t<float> weight, float eps\) \{', replacement2, text, flags=re.DOTALL)

replacement3 = \"\"\"
PYBIND11_MODULE(_native_forward, m) {
    py::class_<MmapWeights>(m, \"MmapWeights\")
        .def(py::init<const std::string&, py::dict>())
        .def(\"test_gemv_q4\", &MmapWeights::test_gemv_q4);

    py::class_<KVCache>(m, \"KVCache\")
        .def(py::init<int, int, int, int>())
        .def(\"set_history\", &py_kvc_set_history);

    m.def(\"compute_attention\", &py_compute_attention, \"Compute SDPA\");
    m.def(\"set_thread_affinity\", &pin_thread_to_core);
    m.def(\"apply_rmsnorm\", &py_apply_rmsnorm, \"Apply RMSNorm in-place\",
          py::arg(\"x\"), py::arg(\"weight\"), py::arg(\"eps\") = 1e-5f);
    m.def(\"apply_rope\", &py_apply_rope, \"Apply RoPE in-place\",
          py::arg(\"q\"), py::arg(\"k\"), py::arg(\"seq_pos\"), py::arg(\"head_dim\"),
          py::arg(\"num_heads\"), py::arg(\"num_kv_heads\"), py::arg(\"theta\") = 10000.0f);
}
\"\"\"
text = re.sub(r'PYBIND11_MODULE\(_native_forward, m\) \{[\s\S]*', replacement3, text, flags=re.DOTALL)

with open('asdsl/kernels/forward_loop.cpp', 'w') as f:
    f.write(text)

\"\"\"
