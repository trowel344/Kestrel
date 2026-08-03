"""Bounded GGUF metadata reads without mapping model tensor payloads."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO


class GGUFMetadataError(ValueError):
    pass


_SCALAR_FORMATS = {
    0: "B",   # uint8
    1: "b",   # int8
    2: "H",   # uint16
    3: "h",   # int16
    4: "I",   # uint32
    5: "i",   # int32
    6: "f",   # float32
    7: "?",   # bool
    10: "Q",  # uint64
    11: "q",  # int64
    12: "d",  # float64
}


def _unpack(handle: BinaryIO, endian: str, fmt: str):
    size = struct.calcsize(fmt)
    payload = handle.read(size)
    if len(payload) != size:
        raise GGUFMetadataError("truncated GGUF metadata")
    return struct.unpack(endian + fmt, payload)[0]


def _string(handle: BinaryIO, endian: str) -> str:
    length = _unpack(handle, endian, "Q")
    if length > 256 * 1024**2:
        raise GGUFMetadataError("unreasonable GGUF string length")
    payload = handle.read(length)
    if len(payload) != length:
        raise GGUFMetadataError("truncated GGUF string")
    return payload.decode("utf-8", errors="replace")


def _value(handle: BinaryIO, endian: str, value_type: int):
    if value_type in _SCALAR_FORMATS:
        return _unpack(handle, endian, _SCALAR_FORMATS[value_type])
    if value_type == 8:
        return _string(handle, endian)
    if value_type == 9:
        element_type = _unpack(handle, endian, "I")
        count = _unpack(handle, endian, "Q")
        if count > 100_000_000:
            raise GGUFMetadataError("unreasonable GGUF array length")
        if element_type in _SCALAR_FORMATS:
            handle.seek(struct.calcsize(_SCALAR_FORMATS[element_type]) * count, 1)
            return None
        if element_type == 8:
            for _ in range(count):
                _string(handle, endian)
            return None
        raise GGUFMetadataError(f"unsupported GGUF array element type {element_type}")
    raise GGUFMetadataError(f"unsupported GGUF value type {value_type}")


def read_planner_metadata(path: str | Path) -> dict:
    """Read only architecture/planner fields and stop before tensor metadata."""

    wanted_suffixes = {
        "block_count": "n_layer",
        "expert_count": "n_exp",
        "expert_used_count": "n_used",
        "embedding_length": "hidden",
        "expert_feed_forward_length": "n_ff",
        "feed_forward_length": "n_ff_fallback",
        "nextn_predict_layers": "mtp_layers",
    }
    values: dict[str, object] = {}
    orphans: dict[str, object] = {}
    with Path(path).open("rb") as handle:
        if handle.read(4) != b"GGUF":
            raise GGUFMetadataError("GGUF magic invalid")
        version_bytes = handle.read(4)
        if len(version_bytes) != 4:
            raise GGUFMetadataError("truncated GGUF header")
        little_version = struct.unpack("<I", version_bytes)[0]
        big_version = struct.unpack(">I", version_bytes)[0]
        if little_version in (2, 3):
            endian, version = "<", little_version
        elif big_version in (2, 3):
            endian, version = ">", big_version
        else:
            raise GGUFMetadataError(f"unsupported GGUF version {little_version}")
        _tensor_count = _unpack(handle, endian, "Q")
        kv_count = _unpack(handle, endian, "Q")
        if kv_count > 10_000_000:
            raise GGUFMetadataError("unreasonable GGUF metadata count")
        for _ in range(kv_count):
            key = _string(handle, endian)
            value_type = _unpack(handle, endian, "I")
            required = {"architecture", "n_layer", "hidden"}
            if (
                key.startswith("tokenizer.")
                and required.issubset(values)
                and ("n_ff" in values or "n_ff_fallback" in values)
            ):
                break
            value = _value(handle, endian, value_type)
            if key == "general.architecture":
                values["architecture"] = value
                # Resolve keys that were emitted before general.architecture.
                for full_name, orphan in list(orphans.items()):
                    for suffix, output_key in wanted_suffixes.items():
                        if full_name == f"{value}.{suffix}" and output_key not in values:
                            values[output_key] = orphan
                continue
            normalized = key.rsplit(".", 1)
            if len(normalized) != 2 or normalized[1] not in wanted_suffixes:
                continue
            output_key = wanted_suffixes[normalized[1]]
            architecture = values.get("architecture")
            if architecture and key == f"{architecture}.{normalized[1]}":
                if output_key not in values:
                    values[output_key] = value
            elif not architecture:
                # Guessed value for a size-bearing suffix; adopt it later only
                # if it matches the declared model architecture. This is
                # blocked so a producer listing architecture late still works.
                orphans[key] = value
    return {
        "architecture": str(values.get("architecture") or "unknown"),
        "n_layer": int(values.get("n_layer") or 0),
        "n_exp": int(values.get("n_exp") or 0),
        "n_used": int(values.get("n_used") or 0),
        "hidden": int(values.get("hidden") or 0),
        "n_ff": int(values.get("n_ff") or values.get("n_ff_fallback") or 0),
        "mtp_layers": int(values.get("mtp_layers") or 0),
        "gguf_version": version,
    }
