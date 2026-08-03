"""Measure the extra error introduced by NVFP4 -> Q4_0 -> Q1_0.

This audit samples routed experts across the complete model without writing a
second GGUF.  It compares the actual Q1_0 encodings produced from the source
NVFP4 weights and from an existing Q4_0 sidecar.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass

import numpy as np

from kestrel.gguf.converter import (
    GGML_TYPE_Q4_0,
    MODEL_DIR,
    NVFP4Converter,
    dequantize_nvfp4,
    dequantize_q4_0,
    quantize_q1_0,
)


def q1_0_cosine(first: bytes, second: bytes) -> float:
    """Return cosine similarity between two decoded Q1_0 byte streams.

    Q1 values are merely a per-block magnitude and a sign bit.  Computing the
    dot product from those components avoids materializing the decoded F32
    tensors and keeps the full-model audit's peak memory bounded.
    """

    a = np.frombuffer(first, dtype=np.uint8)
    b = np.frombuffer(second, dtype=np.uint8)
    if a.shape != b.shape or a.size % 18:
        raise ValueError("Q1_0 streams must have equal, 18-byte-block lengths")
    a = a.reshape(-1, 18)
    b = b.reshape(-1, 18)
    da = np.ascontiguousarray(a[:, :2]).view(np.float16).astype(np.float64).reshape(-1)
    db = np.ascontiguousarray(b[:, :2]).view(np.float16).astype(np.float64).reshape(-1)

    # XOR sign bytes contain one bit for each weight whose signs disagree.
    differing = np.unpackbits(
        np.bitwise_xor(a[:, 2:], b[:, 2:]), axis=1, bitorder="little"
    ).sum(axis=1)
    signed_agreement = 128.0 - 2.0 * differing.astype(np.float64)
    dot = np.sum(da * db * signed_agreement)
    norm_a = np.sqrt(np.sum(da * da * 128.0))
    norm_b = np.sqrt(np.sum(db * db * 128.0))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0 if norm_a == norm_b else 0.0
    return float(dot / (norm_a * norm_b))


@dataclass(frozen=True)
class CascadeSample:
    layer: int
    expert: int
    tensor: str
    cosine: float


def _source_q1(converter: NVFP4Converter, layer: int, expert: int, tensor: str) -> bytes:
    prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}"
    projections = ("gate_proj", "up_proj") if tensor == "gate_up" else ("down_proj",)
    chunks = []
    for projection in projections:
        values = converter._read_nvfp4(f"{prefix}.{projection}")
        if values is None:
            raise KeyError(f"missing source tensor {prefix}.{projection}")
        chunks.append(dequantize_nvfp4(*values))
    matrix = chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)
    return quantize_q1_0(matrix)


def audit_cascade(
    model_dir: str,
    q4_gguf: str,
    *,
    experts_per_layer: int = 1,
    seed: int = 20260801,
) -> list[CascadeSample]:
    if experts_per_layer <= 0:
        raise ValueError("experts_per_layer must be positive")

    import gguf

    converter = NVFP4Converter(model_dir)
    reader = gguf.GGUFReader(q4_gguf)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    rng = np.random.default_rng(seed)
    results = []

    for layer in range(converter.n_layer):
        experts = np.sort(
            rng.choice(converter.n_exp, size=min(experts_per_layer, converter.n_exp), replace=False)
        )
        for tensor_name, suffix in (
            ("gate_up", "ffn_gate_up_exps.weight"),
            ("down", "ffn_down_exps.weight"),
        ):
            name = f"blk.{layer}.{suffix}"
            tensor = tensors.get(name)
            if tensor is None:
                raise KeyError(f"Q4 sidecar is missing tensor {name}")
            if int(tensor.tensor_type) != GGML_TYPE_Q4_0:
                raise ValueError(f"{name} is type {tensor.tensor_type}, expected Q4_0")
            data = np.asarray(tensor.data)
            for expert in experts:
                direct = _source_q1(converter, layer, int(expert), tensor_name)
                cascade = quantize_q1_0(dequantize_q4_0(data[int(expert)]))
                results.append(CascadeSample(
                    layer=layer,
                    expert=int(expert),
                    tensor=tensor_name,
                    cosine=q1_0_cosine(direct, cascade),
                ))
        layer_scores = [item.cosine for item in results if item.layer == layer]
        print(
            f"layer {layer + 1:02d}/{converter.n_layer}: "
            f"min={min(layer_scores):.6f} mean={np.mean(layer_scores):.6f}",
            flush=True,
        )
    return results


def summarize(samples: list[CascadeSample]) -> dict:
    by_tensor = {}
    for tensor in ("gate_up", "down"):
        values = np.asarray([sample.cosine for sample in samples if sample.tensor == tensor])
        if values.size:
            by_tensor[tensor] = {
                "count": int(values.size),
                "min": float(values.min()),
                "mean": float(values.mean()),
                "p05": float(np.quantile(values, 0.05)),
            }
    return {"by_tensor": by_tensor, "samples": [asdict(sample) for sample in samples]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--q4-gguf", required=True)
    parser.add_argument("--experts-per-layer", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--output")
    args = parser.parse_args()

    samples = audit_cascade(
        os.path.abspath(os.path.expanduser(args.model_dir)),
        os.path.abspath(os.path.expanduser(args.q4_gguf)),
        experts_per_layer=args.experts_per_layer,
        seed=args.seed,
    )
    report = summarize(samples)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
        print(json.dumps({
            "by_tensor": report["by_tensor"],
            "sample_count": len(report["samples"]),
            "output": os.path.abspath(args.output),
        }, indent=2, sort_keys=True))
    else:
        print(encoded)


if __name__ == "__main__":
    main()
