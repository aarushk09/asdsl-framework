import os
import time
import tempfile
from pathlib import Path
from asdsl.io.weight_streamer import WeightStreamer

def test_weight_streamer_basic():
    """Test basic prefetch and get via thread fallback."""
    # Create a dummy weight file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        data = b"abcdefghijklmnopqrstuvwxyz" * 100
        tmp.write(data)
        tmp_path = tmp.name

    try:
        streamer = WeightStreamer(tmp_path, use_iouring=False)
        
        # Test offset 0, length 10
        streamer.submit_prefetch(layer_idx=0, byte_offset=0, byte_length=10)
        out0 = streamer.wait_and_get(layer_idx=0)
        assert out0 == data[0:10]
        
        # Test offset 50, length 20
        streamer.submit_prefetch(layer_idx=5, byte_offset=50, byte_length=20)
        out5 = streamer.wait_and_get(layer_idx=5)
        assert out5 == data[50:70]
        
        streamer.close()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_weight_streamer_idempotent():
    """Test that submitting the same layer twice doesn't crash or double-queue."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"0123456789")
        tmp_path = tmp.name

    try:
        streamer = WeightStreamer(tmp_path, use_iouring=False)
        streamer.submit_prefetch(layer_idx=1, byte_offset=0, byte_length=5)
        streamer.submit_prefetch(layer_idx=1, byte_offset=5, byte_length=5) # should be ignored
        
        out = streamer.wait_and_get(layer_idx=1)
        assert out == b"01234" # First submission wins
        streamer.close()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_weight_streamer_sync_fallback():
    """Test wait_and_get without submission returns None (as designed for optimization)."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"hello")
        tmp_path = tmp.name

    try:
        streamer = WeightStreamer(tmp_path, use_iouring=False)
        out = streamer.wait_and_get(layer_idx=99)
        assert out is None
        streamer.close()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    test_weight_streamer_basic()
    test_weight_streamer_idempotent()
    test_weight_streamer_sync_fallback()
    print("All weight_streamer tests passed!")
