#!/usr/bin/env python3
"""Requantize ModelOpt NVFP4 routed experts to llama.cpp Q4_0 in place."""

from __future__ import annotations

import argparse
import gc
import os
import struct
import sys
from pathlib import Path

import gguf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kestrel.gguf.converter import (  # noqa: E402
    GGML_TYPE_NVFP4,
    GGML_TYPE_Q4_0,
    NVFP4Converter,
)


def _type_field_offset(tensor) -> int:
    return (
        tensor.field.offset
        + 8
        + len(tensor.name.encode("utf-8"))
        + 4
        + len(tensor.shape) * 8
    )


class _PositionedWriter:
    def __init__(self, fd: int, offset: int):
        self.fd = fd
        self.offset = offset

    def write(self, payload: bytes) -> int:
        written = os.pwrite(self.fd, payload, self.offset)
        if written != len(payload):
            raise OSError(
                f"short expert write at {self.offset}: "
                f"expected {len(payload)}, got {written}"
            )
        self.offset += written
        return written


def repair(model: Path, source: Path) -> dict[str, int]:
    converter = NVFP4Converter(str(source), include_mtp=False)
    reader = gguf.GGUFReader(str(model))
    layers: list[tuple[int, int, int, tuple[int, int]]] = []

    for layer in range(converter.n_layer):
        gate_name = f"blk.{layer}.ffn_gate_up_exps.weight"
        down_name = f"blk.{layer}.ffn_down_exps.weight"
        gate = next((item for item in reader.tensors if item.name == gate_name), None)
        down = next((item for item in reader.tensors if item.name == down_name), None)
        if gate is None or down is None:
            raise KeyError(f"missing routed expert tensors for layer {layer}")
        if int(gate.tensor_type) != GGML_TYPE_NVFP4 or int(down.tensor_type) != GGML_TYPE_NVFP4:
            raise ValueError(
                f"layer {layer} experts are not both NVFP4: "
                f"{gate.tensor_type}, {down.tensor_type}"
            )
        start = int(gate.data_offset)
        end = int(down.data_offset + down.n_bytes)
        if gate.data_offset + gate.n_bytes != down.data_offset:
            raise ValueError(f"layer {layer} expert tensors are not contiguous")
        layers.append(
            (
                layer,
                start,
                end,
                (_type_field_offset(gate), _type_field_offset(down)),
            )
        )

    del reader
    gc.collect()

    fd = os.open(model, os.O_RDWR)
    total_bytes = 0
    try:
        for layer, start, end, _ in layers:
            writer = _PositionedWriter(fd, start)
            converter._write_data_nvfp4(
                writer,
                f"model.language_model.layers.{layer}",
                layer,
            )
            if writer.offset != end:
                raise ValueError(
                    f"layer {layer} emitted {writer.offset - start} expert bytes; "
                    f"GGUF region is {end - start}"
                )
            total_bytes += end - start
            print(f"\nrequantized layer {layer + 1}/{len(layers)}", flush=True)

        for _, _, _, type_offsets in layers:
            for offset in type_offsets:
                written = os.pwrite(fd, struct.pack("<I", GGML_TYPE_Q4_0), offset)
                if written != 4:
                    raise OSError(f"short tensor-type write at {offset}")
        os.fsync(fd)
    finally:
        os.close(fd)

    return {
        "requantized_layers": len(layers),
        "requantized_tensors": len(layers) * 2,
        "rewritten_bytes": total_bytes,
    }


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
