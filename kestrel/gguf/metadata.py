"""Bounded GGUF metadata reads without mapping model tensor payloads."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path
from typing import BinaryIO

from kestrel.errors import CorruptModelError
from kestrel.util import write_atomic

from .quants import GGUF_SCALAR_FORMATS


class GGUFMetadataError(CorruptModelError):
    """Invalid or truncated GGUF header/metadata, surfaced with an action hint.

    A ``GGUFMetadataError`` *is* a :class:`kestrel.errors.CorruptModelError`, so
    callers can catch the generic class while preserving the specific type for
    the bounded planner read.
    """


def _unpack(handle: BinaryIO, endian: str, fmt: str):
    size = struct.calcsize(fmt)
    payload = handle.read(size)
    if len(payload) != size:
        raise GGUFMetadataError(
            f"truncated GGUF metadata: needed {size} bytes for {fmt}, got {len(payload)} (file ends early)",
            hint="the model file is truncated or corrupt; redownload it",
        )
    try:
        return struct.unpack(endian + fmt, payload)[0]
    except struct.error as exc:
        raise GGUFMetadataError(
            f"malformed GGUF metadata field ({fmt}): {exc}",
            hint="the model file header is not a valid GGUF encoding",
        ) from exc


def _string(handle: BinaryIO, endian: str) -> str:
    length = _unpack(handle, endian, "Q")
    if length > 256 * 1024**2:
        raise GGUFMetadataError(
            f"unreasonable GGUF string length {length} (>256 MiB)",
            hint="a corrupt header inflated a key/value string size",
        )
    payload = handle.read(length)
    if len(payload) != length:
        raise GGUFMetadataError(
            f"truncated GGUF string: key/value needs {length} bytes but only {len(payload)} remain",
            hint="the model file is truncated or corrupt; redownload it",
        )
    return payload.decode("utf-8", errors="replace")


def _value(handle: BinaryIO, endian: str, value_type: int):
    if value_type in GGUF_SCALAR_FORMATS:
        return _unpack(handle, endian, GGUF_SCALAR_FORMATS[value_type])
    if value_type == 8:
        return _string(handle, endian)
    if value_type == 9:
        element_type = _unpack(handle, endian, "I")
        count = _unpack(handle, endian, "Q")
        if count > 100_000_000:
            raise GGUFMetadataError(
                f"unreasonable GGUF array length {count}",
                hint="the model header declared an implausibly large array",
            )
        if element_type in GGUF_SCALAR_FORMATS:
            skip = struct.calcsize(GGUF_SCALAR_FORMATS[element_type]) * count
            _seek_checked(handle, skip)
            return None
        if element_type == 8:
            for _ in range(count):
                _string(handle, endian)
            return None
        raise GGUFMetadataError(
            f"unsupported GGML array element type {element_type}",
            hint="the model uses a tensor/value type this reader does not know",
        )
    raise GGUFMetadataError(
        f"unsupported/unknown GGML type id {value_type} in key-value metadata",
        hint="the model uses a tensor type this reader does not recognize",
    )


def _seek_checked(handle: BinaryIO, offset: int) -> None:
    """Seek forward, failing fast with a clear error when the file is truncated."""
    current = handle.tell()
    handle.seek(0, 2)
    end = handle.tell()
    if current + offset > end:
        raise GGUFMetadataError(
            f"truncated GGUF metadata: array needs {offset} bytes past offset {current} but file ends at {end}",
            hint="the model file is truncated or corrupt; redownload it",
        )
    handle.seek(current + offset)


# Single-entry cache: one process often reads the same model's planner metadata
# twice back-to-back (profile build + KV-cache sizing), and reading the full
# metadata section of a large-vocabulary GGUF is not free. Keying on the file
# identity (mtime + size) keeps stale reads out after the file changes.
_planner_metadata_cache: tuple[tuple[str, int, int], dict] | None = None


def _planner_cache_dir() -> Path:
    override = os.environ.get("KESTREL_CACHE_DIR")
    if override:
        return Path(override)
    base = Path(os.environ.get("XDG_CACHE_HOME") or os.path.join(Path.home(), ".cache"))
    return base / "kestrel"


def _planner_cache_path(path: Path) -> Path:
    digest = hashlib.sha256(os.fsencode(str(path))).hexdigest()[:16]
    return _planner_cache_dir() / f"planner-{digest}.json"


def _planner_cache_read(path: Path, size: int, mtime_ns: int) -> dict | None:
    try:
        cache_file = _planner_cache_path(path)
        data = json.loads(cache_file.read_text())
        if data.get("file_size") == size and data.get("file_mtime_ns") == mtime_ns:
            metadata = data.get("metadata")
            return metadata if isinstance(metadata, dict) else None
    except (OSError, ValueError, TypeError):
        pass
    return None


def _planner_cache_write(path: Path, size: int, mtime_ns: int, metadata: dict) -> None:
    try:
        directory = _planner_cache_dir()
        directory.mkdir(parents=True, exist_ok=True)
        write_atomic(
            _planner_cache_path(path),
            json.dumps(
                {
                    "file_size": size,
                    "file_mtime_ns": mtime_ns,
                    "metadata": metadata,
                }
            ),
            backup=False,
        )
    except OSError:
        pass


def read_planner_metadata(path: str | Path) -> dict:
    """Read only architecture/planner fields and stop before tensor metadata."""

    p = Path(path)
    try:
        stat = p.stat()
        key = (str(p), stat.st_mtime_ns, stat.st_size)
    except OSError:
        key = None
    global _planner_metadata_cache
    if key is not None and _planner_metadata_cache is not None:
        if _planner_metadata_cache[0] == key:
            return dict(_planner_metadata_cache[1])
    if key is not None:
        cached = _planner_cache_read(p, key[2], key[1])
        if cached is not None:
            _planner_metadata_cache = (key, cached)
            return dict(cached)
    result = _read_planner_metadata_unwrapped(p)
    if key is not None:
        _planner_metadata_cache = (key, result)
        _planner_cache_write(p, key[2], key[1], result)
    return result


def _read_planner_metadata_unwrapped(path: Path) -> dict:
    wanted_suffixes = {
        "block_count": "n_layer",
        "expert_count": "n_exp",
        "expert_used_count": "n_used",
        "embedding_length": "hidden",
        "expert_feed_forward_length": "n_ff",
        "feed_forward_length": "n_ff_fallback",
        "nextn_predict_layers": "mtp_layers",
    }
    attention_suffixes = {
        "head_count": "n_heads",
        "head_count_kv": "n_kv_heads",
        "key_length": "head_dim",
    }
    values: dict[str, object] = {}
    orphans: dict[str, object] = {}
    with Path(path).open("rb") as handle:
        magic = handle.read(4)
        if magic != b"GGUF":
            raise GGUFMetadataError(
                f"invalid GGUF magic bytes: got {magic!r}, expected b'GGUF'",
                hint="the file is not a GGUF model (or is corrupt)",
            )
        version_bytes = handle.read(4)
        if len(version_bytes) != 4:
            raise GGUFMetadataError(
                "truncated GGUF header: file ends before the version field",
                hint="the model file is truncated or corrupt; redownload it",
            )
        try:
            little_version = struct.unpack("<I", version_bytes)[0]
            big_version = struct.unpack(">I", version_bytes)[0]
        except struct.error as exc:
            raise GGUFMetadataError(
                f"malformed GGUF version field: {exc}",
                hint="the file header is not a valid GGUF encoding",
            ) from exc
        if little_version in (2, 3):
            endian, version = "<", little_version
        elif big_version in (2, 3):
            endian, version = ">", big_version
        else:
            raise GGUFMetadataError(
                f"unsupported GGUF version {little_version}",
                hint="this build only understands GGUF v2/v3 headers",
            )
        tensor_count = _unpack(handle, endian, "Q")
        if tensor_count > 10_000_000:
            raise GGUFMetadataError(
                f"absurd GGUF tensor count {tensor_count} (>10M)",
                hint="a corrupt header claimed an implausible number of tensors",
            )
        kv_count = _unpack(handle, endian, "Q")
        if kv_count > 10_000_000:
            raise GGUFMetadataError(
                f"unreasonable GGUF metadata count {kv_count} (>10M)",
                hint="a corrupt header claimed an implausible number of KVs",
            )
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
                parent, _, leaf = key.rpartition(".")
                middle = parent.rpartition(".")[2]
                if middle == "attention" and leaf in attention_suffixes:
                    architecture = values.get("architecture")
                    output_key = attention_suffixes[leaf]
                    if architecture and key == f"{architecture}.attention.{leaf}":
                        if output_key not in values:
                            values[output_key] = value
                    elif not architecture:
                        orphans[key] = value
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
        "n_heads": int(values.get("n_heads") or 0),
        "n_kv_heads": int(values.get("n_kv_heads") or 0),
        "head_dim": int(values.get("head_dim") or 0),
        "gguf_version": version,
    }
