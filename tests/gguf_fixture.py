"""Helpers to build minimal GGUF files for unit tests."""

from __future__ import annotations

import struct
from pathlib import Path

GGUF_MAGIC = b"GGUF"
VERSION = 3

# value types
V_STRING = 8
V_UINT32 = 4

_ENCODING = "<"


def _string(out: bytearray, value: str) -> None:
    data = value.encode("utf-8")
    out += struct.pack(_ENCODING + "Q", len(data)) + data


def _bounded_string_with_length(handle) -> str:
    (length,) = struct.unpack(_ENCODING + "Q", handle.read(8))
    return handle.read(length).decode("utf-8")


def write_gguf(
    path: Path,
    *,
    architecture: str = "qwen35moe",
    n_layer: int = 48,
    n_exp: int = 256,
    n_used: int = 8,
    hidden: int = 3072,
    n_ff: int = 1024,
    mtp_layers: int = 0,
    tokenizer_keys: bool = True,
    tensor_count: int = 0,
    n_heads: int = 0,
    n_kv_heads: int = 0,
    head_dim: int = 0,
    file_type: int = 0,
    tensors: list[tuple[str, tuple[int, ...]]] | None = None,
) -> Path:
    """Write a minimal valid GGUF header with the planner-relevant metadata.

    ``tensors`` is an optional list of ``(name, dims)`` tensor-info entries
    written after the KV section; ``tensor_count`` is ignored when it is set.
    """
    if tensors is not None:
        tensor_count = len(tensors)
    body = bytearray(GGUF_MAGIC)
    body += struct.pack(_ENCODING + "I", VERSION)
    body += struct.pack(_ENCODING + "Q", tensor_count)

    # Build KV pairs we want to emit.
    kvs = []  # list of (key, value_type, raw_bytes)
    kvs.append(("general.architecture", V_STRING, _string_to_bytes(architecture)))
    kvs.append((f"{architecture}.block_count", V_UINT32, struct.pack(_ENCODING + "I", n_layer)))
    kvs.append((f"{architecture}.embedding_length", V_UINT32, struct.pack(_ENCODING + "I", hidden)))
    kvs.append((f"{architecture}.feed_forward_length", V_UINT32, struct.pack(_ENCODING + "I", n_ff)))
    if n_exp:
        kvs.append((f"{architecture}.expert_count", V_UINT32, struct.pack(_ENCODING + "I", n_exp)))
    if n_used:
        kvs.append((f"{architecture}.expert_used_count", V_UINT32, struct.pack(_ENCODING + "I", n_used)))
    if mtp_layers:
        kvs.append((f"{architecture}.nextn_predict_layers", V_UINT32, struct.pack(_ENCODING + "I", mtp_layers)))
    if n_heads:
        kvs.append((f"{architecture}.attention.head_count", V_UINT32, struct.pack(_ENCODING + "I", n_heads)))
    if n_kv_heads:
        kvs.append((f"{architecture}.attention.head_count_kv", V_UINT32, struct.pack(_ENCODING + "I", n_kv_heads)))
    if head_dim:
        kvs.append((f"{architecture}.attention.key_length", V_UINT32, struct.pack(_ENCODING + "I", head_dim)))
    if tokenizer_keys:
        kvs.append(("tokenizer.ggml.pre", V_STRING, _string_to_bytes("default")))
    # Some producers (llama.cpp's quantizer) write general.file_type at the
    # very end of the metadata, after the tokenizer section.
    if file_type:
        kvs.append(("general.file_type", V_UINT32, struct.pack(_ENCODING + "I", file_type)))

    body += struct.pack(_ENCODING + "Q", len(kvs))
    for key, vtype, payload in kvs:
        _string(body, key)
        body += struct.pack(_ENCODING + "I", vtype)
        body += payload
    if tensors:
        for name, dims in tensors:
            _string(body, name)
            body += struct.pack(_ENCODING + "I", len(dims))
            for dim in dims:
                body += struct.pack(_ENCODING + "Q", dim)
            body += struct.pack(_ENCODING + "I", 1)  # GGML_TYPE_F16
            body += struct.pack(_ENCODING + "Q", 0)  # tensor data offset
    path.write_bytes(bytes(body))
    return path


def _string_to_bytes(value: str) -> bytes:
    data = value.encode("utf-8")
    return struct.pack(_ENCODING + "Q", len(data)) + data


def read_all(path: Path) -> bytes:
    return path.read_bytes()


def read_string_at(handle, offset: int) -> tuple[str, int]:
    handle.seek(offset)
    length = struct.unpack(_ENCODING + "Q", handle.read(8))[0]
    return handle.read(length).decode("utf-8"), offset + 8 + length
