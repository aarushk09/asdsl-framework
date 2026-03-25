import os
import json
import struct
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any

class MmapWeightStore:
    """
    Memory-maps .safetensors weight files instead of loading them into RAM.
    """
    def __init__(self, model_path: Path):
        self._mmaps = {}  # layer_name -> np.memmap
        self._index = {}
        self._file_path = None
        self._load_index(model_path)

    def _load_index(self, model_path: Path):
        # We need to find the safetensors file
        files = list(model_path.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"No .safetensors files found in {model_path}")
        
        # Taking the first one or looping through all.
        self._file_path = files[0]
        
        with open(self._file_path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header_json = f.read(header_len).decode("utf-8")
            self._index = json.loads(header_json)
            self._data_offset = 8 + header_len

    def get_tensor(self, name: str) -> torch.Tensor:
        if name not in self._index:
            return None
            
        meta = self._index[name]
        dtype_str = meta["dtype"]
        shape = meta["shape"]
        offsets = meta["data_offsets"]
        
        dtype_map = {
            "F32": np.float32,
            "F16": np.float16,
            "BF16": np.float32, # Fallback, needs special handling in torch
            "I8": np.int8,
            "U8": np.uint8,
        }
        np_dtype = dtype_map.get(dtype_str, np.float32)

        if name not in self._mmaps:
            self._mmaps[name] = np.memmap(
                self._file_path,
                dtype=np_dtype,
                mode="r",
                offset=self._data_offset + offsets[0],
                shape=tuple(shape)
            )
            
        # create view
        tensor = torch.frombuffer(self._mmaps[name], dtype=torch.float32 if dtype_str == "F32" else torch.float16).view(*shape)
        return tensor
