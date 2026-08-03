#!/usr/bin/env python3
"""Reversibly requantize routed experts from Q4_0 to Q2_K in place.

The smaller Q2_K payload is written at the front of each tensor's existing
Q4_0 region. Tensor offsets remain unchanged, so this requires the experimental
llama.cpp loader that permits aligned gaps between tensors.

The NVIDIA safetensors are the recovery authority. A journal is committed
before the first model mutation; restore reconstructs complete affected layers
as deterministic Q4_0 using the project's NVFP4 converter.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import gguf
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kestrel.gguf.converter import (  # noqa: E402
    GGML_TYPE_Q4_0,
    NVFP4Converter,
)

GGML_TYPE_Q2_K = 10
GGML_TYPE_Q1_0 = 41


def _type_field_offset(tensor) -> int:
    return (
        tensor.field.offset
        + 8
        + len(tensor.name.encode("utf-8"))
        + 4
        + len(tensor.shape) * 8
    )


def _layer_number(name: str) -> int:
    return int(name.split(".", 2)[1])


def _records(model: Path) -> list[dict]:
    reader = gguf.GGUFReader(str(model))
    records = []
    for tensor in reader.tensors:
        if "_exps." not in tensor.name:
            continue
        current_type = int(tensor.tensor_type)
        if current_type not in (GGML_TYPE_Q4_0, GGML_TYPE_Q2_K, GGML_TYPE_Q1_0):
            continue
        n_cols = int(tensor.shape[0])
        n_rows = int(np.prod(tensor.shape[1:]))
        if n_cols % 256:
            raise ValueError(f"{tensor.name}: {n_cols} columns is not Q2_K aligned")
        elements = n_rows * n_cols
        records.append(
            {
                "name": tensor.name,
                "layer": _layer_number(tensor.name),
                "data_offset": int(tensor.data_offset),
                "type_offset": _type_field_offset(tensor),
                "current_type": current_type,
                "n_rows": n_rows,
                "n_cols": n_cols,
                "q4_bytes": elements // 32 * 18,
                "q2_bytes": elements // 256 * 84,
                "q1_bytes": elements // 128 * 18,
            }
        )
    del reader
    gc.collect()
    return records


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


def _set_type(fd: int, offset: int, tensor_type: int) -> None:
    written = os.pwrite(fd, struct.pack("<I", tensor_type), offset)
    if written != 4:
        raise OSError(f"short tensor-type write at {offset}")


def quantize(
    model: Path,
    source: Path,
    helper: Path,
    journal: Path,
    chunk_rows: int,
    max_tensors: int | None,
    jobs: int,
    target: str,
    helper_threads: int,
) -> dict:
    records = _records(model)
    source_type = GGML_TYPE_Q4_0 if target == "q2_k" else GGML_TYPE_Q2_K
    destination_type = GGML_TYPE_Q2_K if target == "q2_k" else GGML_TYPE_Q1_0
    source_bytes_key = "q4_bytes" if target == "q2_k" else "q2_bytes"
    destination_bytes_key = "q2_bytes" if target == "q2_k" else "q1_bytes"
    targets = [r for r in records if r["current_type"] == source_type]
    if max_tensors is not None:
        targets = targets[:max_tensors]
    journal.write_text(
        json.dumps(
            {
                "model": str(model),
                "source": str(source),
                "target_type": target,
                "targets": targets,
            },
            indent=2,
        )
    )
    with journal.open("rb") as handle:
        os.fsync(handle.fileno())

    def run_helper(item: dict) -> dict:
        command = [
            str(helper),
            str(model),
            str(item["data_offset"]),
            str(item["n_rows"]),
            str(item["n_cols"]),
            str(chunk_rows),
        ]
        if target == "q1_0":
            command += ["q2_k", str(helper_threads)]
        subprocess.run(command, check=True)
        return item

    fd = os.open(model, os.O_RDWR)
    total_q4 = total_q2 = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(run_helper, item) for item in targets]
            for index, future in enumerate(
                concurrent.futures.as_completed(futures), 1
            ):
                item = future.result()
                # Commit metadata last. An interrupted helper leaves the tensor
                # declared Q4_0, so restore can safely rebuild it from source.
                _set_type(fd, item["type_offset"], destination_type)
                os.fsync(fd)
                total_q4 += item[source_bytes_key]
                total_q2 += item[destination_bytes_key]
                print(
                    f"quantize: {index}/{len(targets)} {item['name']}",
                    flush=True,
                )
    finally:
        os.close(fd)
    return {
        "mode": f"quantize-{target}",
        "tensors": len(targets),
        "logical_bytes_before": total_q4,
        "logical_bytes_after": total_q2,
        "logical_bytes_saved": total_q4 - total_q2,
        "journal": str(journal),
    }


def _parse_layers(value: str | None) -> set[int] | None:
    if value is None:
        return None
    layers: set[int] = set()
    for part in value.split(","):
        bounds = part.strip().split("-", 1)
        start = int(bounds[0])
        end = int(bounds[-1])
        if start < 0 or end < start:
            raise ValueError(f"invalid layer range: {part}")
        layers.update(range(start, end + 1))
    return layers


def restore(
    model: Path,
    source: Path,
    journal: Path,
    selected_layers: set[int] | None = None,
) -> dict:
    if not journal.is_file():
        raise FileNotFoundError(f"restore journal does not exist: {journal}")
    payload = json.loads(journal.read_text())
    target_layers = sorted({int(item["layer"]) for item in payload["targets"]})
    if selected_layers is not None:
        target_layers = [layer for layer in target_layers if layer in selected_layers]
        if not target_layers:
            raise ValueError("none of --layers are present in the restore journal")
    records = _records(model)
    by_name = {item["name"]: item for item in records}
    converter = NVFP4Converter(str(source), include_mtp=False)

    fd = os.open(model, os.O_RDWR)
    rewritten = 0
    try:
        for index, layer in enumerate(target_layers, 1):
            gate_name = f"blk.{layer}.ffn_gate_up_exps.weight"
            down_name = f"blk.{layer}.ffn_down_exps.weight"
            gate = by_name[gate_name]
            down = by_name[down_name]
            if gate["data_offset"] + gate["q4_bytes"] != down["data_offset"]:
                raise ValueError(f"layer {layer}: original expert regions are not contiguous")
            writer = _PositionedWriter(fd, gate["data_offset"])
            converter._write_data_nvfp4(
                writer,
                f"model.language_model.layers.{layer}",
                layer,
            )
            expected_end = down["data_offset"] + down["q4_bytes"]
            if writer.offset != expected_end:
                raise ValueError(
                    f"layer {layer}: rebuilt through {writer.offset}, "
                    f"expected {expected_end}"
                )
            _set_type(fd, gate["type_offset"], GGML_TYPE_Q4_0)
            _set_type(fd, down["type_offset"], GGML_TYPE_Q4_0)
            os.fsync(fd)
            rewritten += gate["q4_bytes"] + down["q4_bytes"]
            print(f"restore: {index}/{len(target_layers)} layer {layer}", flush=True)
    finally:
        os.close(fd)
    return {
        "mode": "restore",
        "layers": len(target_layers),
        "tensors": len(target_layers) * 2,
        "rewritten_bytes": rewritten,
        "journal": str(journal),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("quantize", "quantize-q1", "restore"),
        required=True,
    )
    parser.add_argument("--helper", type=Path)
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--helper-threads", type=int, default=1)
    parser.add_argument("--max-tensors", type=int)
    parser.add_argument(
        "--layers",
        help="restore only these layers, for example 0-3,44-47",
    )
    parser.add_argument("--in-place", action="store_true")
    args = parser.parse_args()
    if not args.in_place:
        raise SystemExit("refusing to mutate the model without --in-place")
    model = args.model.expanduser().resolve()
    source = args.source.expanduser().resolve()
    helper = (
        args.helper.expanduser().resolve()
        if args.helper
        else (
            ROOT / "tools/q4_to_q1_0"
            if args.mode == "quantize-q1"
            else ROOT / "tools/q4_to_q2_k"
        ).resolve()
    )
    journal = (
        args.journal.expanduser().resolve()
        if args.journal
        else model.with_suffix(model.suffix + ".expert-q2-journal.json")
    )
    if not model.is_file():
        raise SystemExit(f"model does not exist: {model}")
    if not (source / "model.safetensors.index.json").is_file():
        raise SystemExit(f"source snapshot is incomplete: {source}")
    if args.mode != "restore" and not os.access(helper, os.X_OK):
        raise SystemExit(f"helper is not executable: {helper}")
    if args.chunk_rows < 1:
        raise SystemExit("--chunk-rows must be positive")
    if args.jobs < 1:
        raise SystemExit("--jobs must be positive")
    if args.helper_threads < 1:
        raise SystemExit("--helper-threads must be positive")
    if args.max_tensors is not None and args.max_tensors < 1:
        raise SystemExit("--max-tensors must be positive")
    try:
        selected_layers = _parse_layers(args.layers)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if selected_layers is not None and args.mode != "restore":
        raise SystemExit("--layers currently applies only to restore")
    if args.mode != "restore":
        result = quantize(
            model,
            source,
            helper,
            journal,
            args.chunk_rows,
            args.max_tensors,
            args.jobs,
            "q1_0" if args.mode == "quantize-q1" else "q2_k",
            args.helper_threads,
        )
    else:
        result = restore(model, source, journal, selected_layers)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
