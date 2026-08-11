import numpy as np
import pytest

from kestrel.gguf.quants import quantize_q1_0

_F16_1_0 = np.frombuffer(np.float16(1.0).tobytes(), dtype=np.uint8).tobytes()
_F16_0_0 = np.frombuffer(np.float16(0.0).tobytes(), dtype=np.uint8).tobytes()


def test_q1_0_all_positive_block():
    out = quantize_q1_0(np.ones((1, 128), dtype=np.float32))
    assert len(out) == 18
    assert out[:2] == _F16_1_0
    assert out[2:] == b"\xff" * 16


def test_q1_0_sign_bits_little_endian():
    vals = np.tile(np.array([1.0, -1.0]), 64).astype(np.float32)
    out = quantize_q1_0(vals.reshape(1, 128))
    assert out[2:] == b"\x55" * 16


def test_q1_0_zeros_are_nonnegative():
    out = quantize_q1_0(np.zeros((1, 128), dtype=np.float32))
    assert out[:2] == _F16_0_0
    assert out[2:] == b"\xff" * 16


def test_q1_0_multi_row_multi_block():
    mat = np.ones((2, 256), dtype=np.float32)
    out = quantize_q1_0(mat)
    assert len(out) == 2 * 2 * 18
    assert out[:2] == _F16_1_0
    assert out[54:56] == _F16_1_0  # last block scale


def test_q1_0_scale_is_mean_absolute_value():
    # Blocks of -0.5/+0.5 have mean abs 0.5; 128*0.5/128 = 0.5 fp16.
    mat = np.tile(np.array([-0.5, 0.5]), 64).astype(np.float32).reshape(1, 128)
    out = quantize_q1_0(mat)
    expected = np.frombuffer(np.float16(0.5).tobytes(), dtype=np.uint8).tobytes()
    assert out[:2] == expected


def test_q1_0_rejects_non_divisible_row():
    with pytest.raises(ValueError):
        quantize_q1_0(np.zeros((1, 100), dtype=np.float32))
