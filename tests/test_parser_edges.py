"""Edge-case tests that GGUF header/metadata parsing fails cleanly.

A malformed GGUF (bad magic, declared sizes beyond EOF, absurd counts, unknown
dtype ids) must surface as :class:`kestrel.errors.CorruptModelError` with a
clear message and hint -- never a raw ``struct.error``, ``EOFError``, or
``IndexError``.
"""

import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from kestrel.errors import CorruptModelError  # noqa: E402
from kestrel.gguf.metadata import read_planner_metadata  # noqa: E402


def _valid_gguf() -> bytes:
    """A minimal well-formed GGUF v3 header with architecture metadata."""
    body = bytearray(b"GGUF")
    body += struct.pack("<I", 3)  # version
    body += struct.pack("<Q", 0)  # tensor count
    kvs = []

    def add_string(key: str, value: str) -> None:
        key_b = key.encode("utf-8")
        value_b = value.encode("utf-8")
        kvs.append(
            struct.pack("<Q", len(key_b)) + key_b + struct.pack("<I", 8) + struct.pack("<Q", len(value_b)) + value_b
        )

    def add_uint32(key: str, value: int) -> None:
        key_b = key.encode("utf-8")
        kvs.append(struct.pack("<Q", len(key_b)) + key_b + struct.pack("<I", 4) + struct.pack("<I", value))

    add_string("general.architecture", "qwen35moe")
    add_uint32("qwen35moe.block_count", 48)
    add_uint32("qwen35moe.embedding_length", 3072)
    add_uint32("qwen35moe.feed_forward_length", 1024)
    body += struct.pack("<Q", len(kvs))
    for kv in kvs:
        body += kv
    return bytes(body)


def _write(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    # Bypass both the in-process and on-disk caches so each corruption is parsed.
    import kestrel.gguf.metadata as metadata

    metadata._planner_metadata_cache = None
    return path


def test_bad_magic_raises_corrupt_model(tmp_path):
    data = bytearray(_valid_gguf())
    data[:4] = b"XGUF"
    path = _write(tmp_path, "bad-magic.gguf", bytes(data))
    with pytest.raises(CorruptModelError) as excinfo:
        read_planner_metadata(path)
    assert "magic" in str(excinfo.value)
    assert excinfo.value.hint


def test_declared_size_beyond_eof_raises(tmp_path):
    # The string value claims 100000 bytes but only 9 are present.
    key = b"general.architecture"
    kv = struct.pack("<Q", len(key)) + key + struct.pack("<I", 8) + struct.pack("<Q", 100000) + b"qwen35moe"
    body = bytearray(b"GGUF")
    body += struct.pack("<I", 3)
    body += struct.pack("<Q", 0)
    body += struct.pack("<Q", 1)
    body += kv
    path = _write(tmp_path, "truncated.gguf", bytes(body))
    with pytest.raises(CorruptModelError) as excinfo:
        read_planner_metadata(path)
    assert "truncat" in str(excinfo.value).lower()
    assert excinfo.value.hint


def test_unknown_dtype_id_raises(tmp_path):
    key = b"arch.unknown_key"
    kv = struct.pack("<Q", len(key)) + key + struct.pack("<I", 99)
    body = bytearray(b"GGUF")
    body += struct.pack("<I", 3)
    body += struct.pack("<Q", 0)
    body += struct.pack("<Q", 1)
    body += kv
    path = _write(tmp_path, "unknown-dtype.gguf", bytes(body))
    with pytest.raises(CorruptModelError) as excinfo:
        read_planner_metadata(path)
    assert "99" in str(excinfo.value)
    assert excinfo.value.hint


def test_absurd_tensor_count_raises(tmp_path):
    body = bytearray(_valid_gguf())
    body[8:16] = struct.pack("<Q", 2**40)
    path = _write(tmp_path, "huge-tensor-count.gguf", bytes(body))
    with pytest.raises(CorruptModelError) as excinfo:
        read_planner_metadata(path)
    assert "tensor count" in str(excinfo.value)
    assert excinfo.value.hint


def test_malformed_key_entry_raises(tmp_path):
    # A key string whose declared length is implausibly large (> 256 MiB).
    body = bytearray(b"GGUF")
    body += struct.pack("<I", 3)
    body += struct.pack("<Q", 0)
    body += struct.pack("<Q", 1)
    body += struct.pack("<Q", 300 * 1024**2)  # bogus key length, no payload
    path = _write(tmp_path, "big-key.gguf", bytes(body))
    with pytest.raises(CorruptModelError) as excinfo:
        read_planner_metadata(path)
    assert "length" in str(excinfo.value)
    assert excinfo.value.hint


def test_truncated_header_field_raises(tmp_path):
    body = bytearray(_valid_gguf())
    # Declare far more key/value pairs than the file contains.
    body[16:24] = struct.pack("<Q", 10**6)
    path = _write(tmp_path, "short-kvs.gguf", bytes(body))
    with pytest.raises(CorruptModelError) as excinfo:
        read_planner_metadata(path)
    assert "truncat" in str(excinfo.value).lower()
    assert excinfo.value.hint
