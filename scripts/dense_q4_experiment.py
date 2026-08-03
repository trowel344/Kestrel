#!/usr/bin/env python3
"""Reversibly quantize non-expert BF16 matrices to Q4_0 in-place.

This is an R&D tool for the Qwen3.5-122B-A10B memory-traffic experiment. GGUF
tensor offsets are explicit, so the smaller Q4_0 payload can occupy the front
of the original BF16 tensor region without moving later tensors. The file does
not shrink, but llama.cpp reads and allocates the smaller logical payload.

The source safetensors are the recovery authority. A journal is written before
the first mutation, and ``--mode restore`` reconstructs every experimental
matrix from that source even after an interrupted quantization.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import struct
import sys
from pathlib import Path

import gguf
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kestrel.gguf.converter import (  # noqa: E402
    GGML_TYPE_BF16,
    GGML_TYPE_Q4_0,
    NVFP4Converter,
    quantize_q4_0,
)


def _type_field_offset(tensor) -> int:
    return (
        tensor.field.offset
        + 8
        + len(tensor.name.encode("utf-8"))
        + 4
        + len(tensor.shape) * 8
    )


def _is_experimental_matrix(tensor) -> bool:
    return (
        tensor.name != "token_embd.weight"
        and "_exps." not in tensor.name
        and len(tensor.shape) == 2
        and int(tensor.shape[0]) % 32 == 0
    )


def _source_tensor(converter: NVFP4Converter, name: str) -> torch.Tensor:
    source_key = converter._bf16_hf_key(name)
    if source_key is None:
        raise KeyError(f"no source mapping for dense tensor {name}")
    tensor = converter._read_torch(source_key)
    if tensor is None:
        raise KeyError(f"missing source tensor {source_key}")
    tensor = converter._transform_source_tensor(tensor, source_key)
    if tensor.ndim != 2:
        raise ValueError(f"{name}: expected matrix source, got {tuple(tensor.shape)}")
    return tensor.contiguous()


def _write_q4_rows(fd: int, offset: int, tensor: torch.Tensor, rows: int) -> int:
    written = 0
    for start in range(0, tensor.shape[0], rows):
        chunk = tensor[start : start + rows].to(torch.float32).numpy()
        payload = quantize_q4_0(chunk)
        count = os.pwrite(fd, payload, offset + written)
        if count != len(payload):
            raise OSError(f"short Q4_0 write: {count} of {len(payload)}")
        written += count
    return written


def _write_bf16_rows(fd: int, offset: int, tensor: torch.Tensor, rows: int) -> int:
    written = 0
    for start in range(0, tensor.shape[0], rows):
        chunk = (
            tensor[start : start + rows]
            .to(torch.bfloat16)
            .contiguous()
            .view(torch.uint16)
            .numpy()
        )
        payload = memoryview(chunk).cast("B")
        count = os.pwrite(fd, payload, offset + written)
        if count != len(payload):
            raise OSError(f"short BF16 write: {count} of {len(payload)}")
        written += count
    return written


def _records(model: Path) -> list[dict]:
    reader = gguf.GGUFReader(str(model))
    result = []
    for tensor in reader.tensors:
        if not _is_experimental_matrix(tensor):
            continue
        tensor_type = int(tensor.tensor_type)
        if tensor_type not in (GGML_TYPE_BF16, GGML_TYPE_Q4_0):
            continue
        n_elements = int(np.prod(tensor.shape))
        result.append(
            {
                "name": tensor.name,
                "data_offset": int(tensor.data_offset),
                "type_offset": _type_field_offset(tensor),
                "current_type": tensor_type,
                "elements": n_elements,
                "bf16_bytes": n_elements * 2,
                "q4_bytes": n_elements // 32 * 18,
            }
        )
    del reader
    gc.collect()
    return result


def mutate(
    model: Path,
    source: Path,
    mode: str,
    journal: Path,
    rows: int,
    max_matrices: int | None,
) -> dict:
    converter = NVFP4Converter(str(source), include_mtp=False)
    records = _records(model)
    if mode == "quantize":
        targets = [item for item in records if item["current_type"] == GGML_TYPE_BF16]
        if max_matrices is not None:
            targets = targets[:max_matrices]
        journal.write_text(
            json.dumps(
                {
                    "model": str(model),
                    "source": str(source),
                    "targets": [item["name"] for item in targets],
                },
                indent=2,
            )
        )
    else:
        if not journal.is_file():
            raise FileNotFoundError(f"restore journal does not exist: {journal}")
        target_names = set(json.loads(journal.read_text())["targets"])
        targets = [item for item in records if item["name"] in target_names]

    fd = os.open(model, os.O_RDWR)
    total_before = total_after = 0
    try:
        for index, item in enumerate(targets, 1):
            tensor = _source_tensor(converter, item["name"])
            if mode == "quantize":
                size = _write_q4_rows(fd, item["data_offset"], tensor, rows)
                expected = item["q4_bytes"]
                new_type = GGML_TYPE_Q4_0
                total_before += item["bf16_bytes"]
                total_after += item["q4_bytes"]
            else:
                size = _write_bf16_rows(fd, item["data_offset"], tensor, rows)
                expected = item["bf16_bytes"]
                new_type = GGML_TYPE_BF16
                total_before += item["q4_bytes"]
                total_after += item["bf16_bytes"]
            if size != expected:
                raise ValueError(
                    f"{item['name']}: wrote {size} bytes, expected {expected}"
                )
            os.pwrite(fd, struct.pack("<I", new_type), item["type_offset"])
            del tensor
            gc.collect()
            if index % 16 == 0 or index == len(targets):
                print(f"{mode}: {index}/{len(targets)} matrices", flush=True)
        os.fsync(fd)
    finally:
        os.close(fd)

    return {
        "mode": mode,
        "matrices": len(targets),
        "logical_bytes_before": total_before,
        "logical_bytes_after": total_after,
        "logical_bytes_saved": total_before - total_after,
        "journal": str(journal),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--mode", choices=("quantize", "restore"), required=True)
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--max-matrices", type=int)
    parser.add_argument("--in-place", action="store_true")
    args = parser.parse_args()
    if not args.in_place:
        raise SystemExit("refusing to mutate the model without --in-place")
    model = args.model.expanduser().resolve()
    source = args.source.expanduser().resolve()
    journal = (
        args.journal.expanduser().resolve()
        if args.journal
        else model.with_suffix(model.suffix + ".dense-q4-journal.json")
    )
    if not model.is_file():
        raise SystemExit(f"model does not exist: {model}")
    if not (source / "model.safetensors.index.json").is_file():
        raise SystemExit(f"source snapshot is incomplete: {source}")
    if args.rows < 1:
        raise SystemExit("--rows must be positive")
    if args.max_matrices is not None and args.max_matrices < 1:
        raise SystemExit("--max-matrices must be positive")
    print(
        mutate(
            model,
            source,
            args.mode,
            journal,
            args.rows,
            args.max_matrices,
        )
    )


if __name__ == "__main__":
    main()
