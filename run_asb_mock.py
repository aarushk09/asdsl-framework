import time
import argparse
import numpy as np
import json
import ctypes
from ctypes import wintypes
from asdsl.kernels._native_unified import EngineConfig, UnifiedEngine

def enable_virtuallock(mmap_obj):
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    size_bytes = len(mmap_obj)
    current_process = kernel32.GetCurrentProcess()
    min_size = ctypes.c_size_t(size_bytes + int(1.5 * 1024**3))
    max_size = ctypes.c_size_t(size_bytes + int(3.0 * 1024**3))
    kernel32.SetProcessWorkingSetSizeEx.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, wintypes.DWORD]
    kernel32.SetProcessWorkingSetSizeEx(current_process, min_size, max_size, 1 | 4)
    kernel32.VirtualLock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    kernel32.VirtualLock(ctypes.c_void_p(mmap_obj.ctypes.data), ctypes.c_size_t(size_bytes))

class MmapBinStore:
    def __init__(self, bin_path, json_path):
        with open(json_path, 'r') as f:
            self.metadata = json.load(f)
        self.mmap = np.memmap(bin_path, dtype=np.uint8, mode='r')
        enable_virtuallock(self.mmap)

    def get_fp32(self, key):
        info = self.metadata[key]
        off = info['offset']
        sz = info['size_bytes']
        return self.mmap[off:off+sz].view(np.float32).reshape(info['shape'])

    def get_fp16(self, key):
        info = self.metadata[key]
        off = info['offset']
        sz = info['size_bytes']
        return self.mmap[off:off+sz].view(np.float16).astype(np.float32).reshape(info['shape'])

    def get_q4(self, key):
        info = self.metadata[key]
        off = info['offset']
        sz = info['size_bytes']
        return self.mmap[off:off+sz]

def create_mock_config():
    config = EngineConfig()
    config.max_seq_len = 2048
    config.vocab_size = 100352
    config.hidden_size = 5120
    config.num_layers = 40
    config.num_heads = 40
    config.num_kv_heads = 10
    config.intermediate_size = 14336
    config.rms_norm_eps = 1e-5
    config.group_size = 32
    config.head_dim = 128
    config.rotary_dim = 128
    return config

def main():
    store = MmapBinStore('models/phi4_asb.bin', 'models/phi4_asb_metadata.json')
    config = create_mock_config()
    
    token_embd = store.get_fp32('embed')
    output_norm = store.get_fp32('final_rms')
    output_proj = store.get_fp32('lm_head')

    # Dummy tables for rope
    cos_table = np.ones((config.max_seq_len, config.rotary_dim // 2), dtype=np.float32)
    sin_table = np.zeros((config.max_seq_len, config.rotary_dim // 2), dtype=np.float32)

    fatrelu = {}
    import json, os
    if os.path.exists("phi4_fatrelu_thresholds.json"):
        with open("phi4_fatrelu_thresholds.json", "r") as f:
            data = json.load(f)
            for k, v in data.get("thresholds", {}).items():
                fatrelu[int(k)] = float(v)

    layers_dict = {}
    for i in range(config.num_layers):
        layers_dict[i] = {
            'rms_att': store.get_fp32(f'l{i}_rms1'),
            'qkv_proj': store.get_q4(f'l{i}_qkv'),
            'o_proj': store.get_q4(f'l{i}_o'),
            'rms_ffn': store.get_fp32(f'l{i}_rms2'),
            'gate_up_proj': store.get_q4(f'l{i}_gate'), # using gate as dummy for gate_up
            'down_proj': store.get_q4(f'l{i}_down'),
            'fatrelu_threshold': fatrelu.get(i, 0.0)
        }

    engine = UnifiedEngine(
        config,
        token_embd,
        output_norm,
        output_proj,
        cos_table,
        sin_table,
        layers_dict
    )

    input_tokens = [200022, 3575, 553, 261, 10297, 20837, 29186, 13, 200020, 200021, 176289, 44967, 306, 1001, 21872, 13, 200020, 200019]
    inputs = np.array(input_tokens, dtype=np.int32)
    max_new_tokens = 20

    print(f'Starting generate with {len(input_tokens)} tokens.')
    t0 = time.time()
    outputs = engine.generate(inputs, 57)
    t1 = time.time()
    dt = t1 - t0
    toks = len(outputs) + len(inputs)
    print(f'Generated {toks} total tokens in {dt:.2f}s ({toks/dt:.2f} tok/s)')

if __name__ == '__main__':
    main()

