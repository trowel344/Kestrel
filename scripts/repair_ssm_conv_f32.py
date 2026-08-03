#!/usr/bin/env python3
"""Expand Qwen3.5 SSM convolution kernels from BF16 to F32 in a GGUF.

This is an in-place migration for artifacts created before Kestrel emitted
``blk.*.ssm_conv1d.weight`` as F32. llama.cpp's CPU ``ggml_ssm_conv`` kernel
requires an F32 convolution kernel. The migration shifts the contiguous tensor
payload backwards once, expands only the affected tensors, then updates GGUF
tensor types and offsets.
"""

from __future__ import annotations

import argparse
import gc
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import gguf
import numpy as np

BF16 = 30
F32 = 0
CHUNK_BYTES = 8 * 1024**2


@dataclass(frozen=True)
class TensorMove:
    name: str
    old_offset: int
    new_offset: int
    old_size: int
    new_size: int
    type_field_offset: int
    offset_field_offset: int
    expand_bf16: bool


def _metadata_offsets(tensor) -> tuple[int, int]:
    name_size = len(tensor.name.encode("utf-8"))
    n_dims = len(tensor.shape)
    type_offset = tensor.field.offset + 8 + name_size + 4 + n_dims * 8
    return type_offset, type_offset + 4


def _copy_backwards(fd: int, source: int, destination: int, size: int) -> None:
    if source == destination or size == 0:
        return
    remaining = size
    while remaining:
        amount = min(CHUNK_BYTES, remaining)
        start = remaining - amount
        data = os.pread(fd, amount, source + start)
        if len(data) != amount:
            raise OSError(
                f"short read at {source + start}: expected {amount}, got {len(data)}"
            )
        written = os.pwrite(fd, data, destination + start)
        if written != amount:
            raise OSError(
                f"short write at {destination + start}: expected {amount}, got {written}"
            )
        remaining = start


def _expand_bf16(fd: int, source: int, destination: int, size: int) -> None:
    raw = os.pread(fd, size, source)
    if len(raw) != size:
        raise OSError(f"short BF16 read at {source}: expected {size}, got {len(raw)}")
    bf16 = np.frombuffer(raw, dtype="<u2")
    f32_bits = bf16.astype("<u4") << np.uint32(16)
    expanded = f32_bits.view("<f4").tobytes()
    written = os.pwrite(fd, expanded, destination)
    if written != len(expanded):
        raise OSError(
            f"short F32 write at {destination}: expected {len(expanded)}, got {written}"
        )


def repair(path: Path) -> dict[str, int]:
    reader = gguf.GGUFReader(str(path))
    tensors = sorted(reader.tensors, key=lambda item: item.data_offset)
    if not tensors:
        raise ValueError("GGUF contains no tensors")

    old_size = path.stat().st_size
    last_end = tensors[-1].data_offset + tensors[-1].n_bytes
    if last_end != old_size:
        raise ValueError(
            f"tensor payload does not end at EOF: tensor end {last_end}, file {old_size}"
        )
    for left, right in zip(tensors, tensors[1:]):  # noqa: B905 (pairwise slices differ)
        if left.data_offset + left.n_bytes != right.data_offset:
            raise ValueError(
                f"tensor payload is not contiguous between {left.name} and {right.name}"
            )

    selected = {
        tensor.name
        for tensor in tensors
        if tensor.name.endswith(".ssm_conv1d.weight")
        and int(tensor.tensor_type) == BF16
    }
    if not selected:
        raise ValueError("no BF16 SSM convolution tensors need migration")

    moves: list[TensorMove] = []
    growth = 0
    for tensor in tensors:
        expand = tensor.name in selected
        old_tensor_size = int(tensor.n_bytes)
        new_tensor_size = old_tensor_size * 2 if expand else old_tensor_size
        type_offset, offset_offset = _metadata_offsets(tensor)
        moves.append(
            TensorMove(
                name=tensor.name,
                old_offset=int(tensor.data_offset),
                new_offset=int(tensor.data_offset) + growth,
                old_size=old_tensor_size,
                new_size=new_tensor_size,
                type_field_offset=type_offset,
                offset_field_offset=offset_offset,
                expand_bf16=expand,
            )
        )
        growth += new_tensor_size - old_tensor_size

    data_offset = int(reader.data_offset)
    del tensors
    del reader
    gc.collect()

    fd = os.open(path, os.O_RDWR)
    try:
        os.ftruncate(fd, old_size + growth)
        total = len(moves)
        for index, move in enumerate(reversed(moves), 1):
            if move.expand_bf16:
                _expand_bf16(
                    fd,
                    move.old_offset,
                    move.new_offset,
                    move.old_size,
                )
            else:
                _copy_backwards(
                    fd,
                    move.old_offset,
                    move.new_offset,
                    move.old_size,
                )
            if index % 64 == 0 or index == total:
                print(f"moved {index}/{total} tensors", flush=True)

        for move in moves:
            relative_offset = move.new_offset - data_offset
            os.pwrite(fd, struct.pack("<Q", relative_offset), move.offset_field_offset)
            if move.expand_bf16:
                os.pwrite(fd, struct.pack("<I", F32), move.type_field_offset)
        os.fsync(fd)
    finally:
        os.close(fd)

    return {
        "converted_tensors": len(selected),
        "old_size": old_size,
        "new_size": old_size + growth,
        "growth": growth,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="required acknowledgement that the GGUF will be modified in place",
    )
    args = parser.parse_args()
    if not args.in_place:
        raise SystemExit("refusing to modify the model without --in-place")
    path = args.model.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"model does not exist: {path}")
    result = repair(path)
    print(result)


if __name__ == "__main__":
    main()
