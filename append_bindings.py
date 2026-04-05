import os

with open("asdsl/kernels/native/unified_engine.cpp", "a") as f:
    f.write("""
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace asdsl {

py::class_<EngineConfig> register_config(py::module_& m) {
    auto cls = py::class_<EngineConfig>(m, "EngineConfig")
        .def(py::init<>())
        .def_readwrite("num_layers", &asdsl::EngineConfig::num_layers)
        .def_readwrite("hidden_size", &asdsl::EngineConfig::hidden_size)
        .def_readwrite("num_heads", &asdsl::EngineConfig::num_heads)
        .def_readwrite("num_kv_heads", &asdsl::EngineConfig::num_kv_heads)
        .def_readwrite("head_dim", &asdsl::EngineConfig::head_dim)
        .def_readwrite("intermediate_size", &asdsl::EngineConfig::intermediate_size)
        .def_readwrite("vocab_size", &asdsl::EngineConfig::vocab_size)
        .def_readwrite("rms_norm_eps", &asdsl::EngineConfig::rms_norm_eps)
        .def_readwrite("group_size", &asdsl::EngineConfig::group_size)
        .def_readwrite("max_seq_len", &asdsl::EngineConfig::max_seq_len);
    return cls;
}

// A wrapper to safely initialize the C++ engine from numpy arrays:
class UnifiedEnginePy {
    std::unique_ptr<UnifiedEngine> engine_;
    EngineWeights weights_;
    
    // We hold references to py::array so memory isn't GC'd
    std::vector<py::array> keep_alive_;

    template<typename T>
    const T* get_ptr(py::array_t<T>& arr) {
        if (arr.size() == 0) return nullptr;
        keep_alive_.push_back(arr);
        return arr.data();
    }

public:
    UnifiedEnginePy(
        EngineConfig config,
        py::array_t<float> token_embd,
        py::array_t<float> output_norm,
        py::array_t<uint8_t> output_proj,
        py::array_t<float> output_scales,
        py::array_t<float> cos_table,
        py::array_t<float> sin_table,
        py::dict layers_dict
    ) {
        weights_.token_embd = get_ptr(token_embd);
        weights_.output_norm = get_ptr(output_norm);
        weights_.output_proj = get_ptr(output_proj);
        weights_.output_scales = get_ptr(output_scales);
        weights_.cos_table = get_ptr(cos_table);
        weights_.sin_table = get_ptr(sin_table);

        for (auto item : layers_dict) {
            int layer_idx = item.first.cast<int>();
            py::dict l_dict = item.second.cast<py::dict>();
            
            LayerWeights lw;
            
            auto rms_att = l_dict["rms_att"].cast<py::array_t<float>>();
            lw.rms_att = get_ptr(rms_att);
            
            auto qkv_proj = l_dict["qkv_proj"].cast<py::array_t<uint8_t>>();
            lw.qkv_proj = get_ptr(qkv_proj);
            
            auto qkv_scales = l_dict["qkv_scales"].cast<py::array_t<float>>();
            lw.qkv_scales = get_ptr(qkv_scales);
            
            auto o_proj = l_dict["o_proj"].cast<py::array_t<uint8_t>>();
            lw.o_proj = get_ptr(o_proj);
            
            auto o_scales = l_dict["o_scales"].cast<py::array_t<float>>();
            lw.o_scales = get_ptr(o_scales);
            
            auto rms_ffn = l_dict["rms_ffn"].cast<py::array_t<float>>();
            lw.rms_ffn = get_ptr(rms_ffn);
            
            auto gate_up_proj = l_dict["gate_up_proj"].cast<py::array_t<uint8_t>>();
            lw.gate_up_proj = get_ptr(gate_up_proj);
            
            auto gate_up_scales = l_dict["gate_up_scales"].cast<py::array_t<float>>();
            lw.gate_up_scales = get_ptr(gate_up_scales);
            
            auto down_proj = l_dict["down_proj"].cast<py::array_t<uint8_t>>();
            lw.down_proj = get_ptr(down_proj);
            
            auto down_scales = l_dict["down_scales"].cast<py::array_t<float>>();
            lw.down_scales = get_ptr(down_scales);
            
            weights_.layers[layer_idx] = lw;
        }

        engine_ = std::make_unique<UnifiedEngine>(config, weights_);
    }

    std::vector<int32_t> generate(std::vector<int32_t> prompt, int max_tokens) {
        // Release GIL during generation process
        py::gil_scoped_release release;
        return engine_->generate(prompt, max_tokens);
    }
};

}

PYBIND11_MODULE(_native_unified, m) {
    asdsl::register_config(m);
    
    py::class_<asdsl::UnifiedEnginePy>(m, "UnifiedEngine")
        .def(py::init<
            asdsl::EngineConfig,
            py::array_t<float>,
            py::array_t<float>,
            py::array_t<uint8_t>,
            py::array_t<float>,
            py::array_t<float>,
            py::array_t<float>,
            py::dict
        >(), 
        py::arg("config"),
        py::arg("token_embd"),
        py::arg("output_norm"),
        py::arg("output_proj"),
        py::arg("output_scales"),
        py::arg("cos_table"),
        py::arg("sin_table"),
        py::arg("layers_dict"))
        .def("generate", &asdsl::UnifiedEnginePy::generate);
}
""")
