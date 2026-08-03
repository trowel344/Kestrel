#!/usr/bin/env python3
"""Restore dense GGUF payloads from source safetensors in row-major order."""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import gguf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kestrel.gguf.converter import (
    GGML_TYPE_BF16,
    GGML_TYPE_F16,
    GGML_TYPE_F32,
    NVFP4Converter,
)


def _source_bytes(tensor: torch.Tensor, ggml_type: int) -> bytes:
    tensor = tensor.contiguous()
    if ggml_type == GGML_TYPE_F32:
        return tensor.to(torch.float32).numpy().tobytes()
    if ggml_type == GGML_TYPE_F16:
        return tensor.to(torch.float16).numpy().tobytes()
    if ggml_type == GGML_TYPE_BF16:
        return (
            tensor.to(torch.bfloat16)
            .contiguous()
            .view(torch.uint16)
            .numpy()
            .tobytes()
        )
    raise ValueError(f"unsupported dense GGML type {ggml_type}")


def repair(model: Path, source: Path) -> dict[str, int]:
    converter = NVFP4Converter(str(source), include_mtp=False)
    reader = gguf.GGUFReader(str(model))
    records: list[tuple[str, int, int, int, str]] = []
    for tensor in reader.tensors:
        tensor_type = int(tensor.tensor_type)
        if tensor_type not in (GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16):
            continue
        source_key = converter._bf16_hf_key(tensor.name)
        if source_key is None:
            raise KeyError(f"no source mapping for dense tensor {tensor.name}")
        records.append(
            (
                tensor.name,
                int(tensor.data_offset),
                int(tensor.n_bytes),
                tensor_type,
                source_key,
            )
        )
    del reader
    gc.collect()

    fd = os.open(model, os.O_RDWR)
    total_bytes = 0
    try:
        for index, (name, offset, expected_size, tensor_type, source_key) in enumerate(
            records, 1
        ):
            source_tensor = converter._read_torch(source_key)
            if source_tensor is None:
                raise KeyError(f"missing source tensor {source_key}")
            source_tensor = converter._transform_source_tensor(source_tensor, source_key)
            payload = _source_bytes(source_tensor, tensor_type)
            if len(payload) != expected_size:
                raise ValueError(
                    f"{name}: source payload is {len(payload)} bytes, "
                    f"GGUF declares {expected_size}"
                )
            written = os.pwrite(fd, payload, offset)
            if written != expected_size:
                raise OSError(
                    f"{name}: wrote {written} bytes, expected {expected_size}"
                )
            total_bytes += written
            if index % 64 == 0 or index == len(records):
                print(f"restored {index}/{len(records)} dense tensors", flush=True)
        os.fsync(fd)
    finally:
        os.close(fd)

    return {"restored_tensors": len(records), "restored_bytes": total_bytes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--in-place", action="store_true")
    args = parser.parse_args()
    if not args.in_place:
        raise SystemExit("refusing to modify the model without --in-place")
    model = args.model.expanduser().resolve()
    source = args.source.expanduser().resolve()
    if not model.is_file():
        raise SystemExit(f"model does not exist: {model}")
    if not (source / "model.safetensors.index.json").is_file():
        raise SystemExit(f"source snapshot is incomplete: {source}")
    print(repair(model, source))


if __name__ == "__main__":
    main()
