"""Bounded, deterministic quantization candidate assessment.

This module deliberately operates on an in-memory sample matrix.  It never
opens or writes a GGUF, which makes it safe to use before committing to a
multi-dozen-gigabyte model conversion.  The CLI accepts ``.npy`` matrices and
prints a JSON report suitable for comparing Q2_K calibration choices.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from kestrel.gguf.quants import (
    dequantize_q2_k,
    dequantize_q3_k,
    quantize_q2_k,
    quantize_q3_k,
)


@dataclass(frozen=True)
class CandidateMetric:
    name: str
    sample_bytes: int
    estimated_matrix_bytes: int
    cosine: float
    nrmse: float


def _metric(
    name: str, source: np.ndarray, raw: bytes, decoded: np.ndarray, *, full_rows: int | None = None
) -> CandidateMetric:
    reference = source.astype(np.float64, copy=False).reshape(-1)
    actual = decoded.astype(np.float64, copy=False).reshape(-1)
    ref_norm = float(np.linalg.norm(reference))
    actual_norm = float(np.linalg.norm(actual))
    cosine = 1.0 if ref_norm == 0.0 and actual_norm == 0.0 else 0.0
    if ref_norm and actual_norm:
        cosine = float(np.dot(reference, actual) / (ref_norm * actual_norm))
    error = float(np.linalg.norm(actual - reference))
    nrmse = error / ref_norm if ref_norm else (0.0 if error == 0.0 else float("inf"))
    rows, cols = source.shape
    # K quantizers use one fixed-size block per 256 input columns.
    estimated = (full_rows or rows) * (cols // 256) * (len(raw) // (rows * (cols // 256)))
    return CandidateMetric(name, len(raw), estimated, cosine, nrmse)


def assess_quant_candidates(
    matrix: np.ndarray,
    importance: np.ndarray | None = None,
    *,
    max_rows: int = 8,
    seed: int = 20260809,
    include_q3: bool = True,
) -> dict:
    """Compare baseline/weighted Q2_K and optional Q3_K on a bounded sample.

    Rows are selected without replacement using a stable seed.  If no
    activation importance is supplied, a clearly-labelled column-energy proxy
    is used so the weighted candidate remains comparable without pretending it
    is calibration data.
    """

    source = matrix if hasattr(matrix, "shape") and hasattr(matrix, "ndim") else np.asarray(matrix)
    if source.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got {source.shape}")
    rows, cols = source.shape
    if rows == 0 or cols == 0 or cols % 256:
        raise ValueError(f"matrix must have positive dimensions and width divisible by 256, got {source.shape}")
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    count = min(rows, max_rows)
    selected = np.sort(np.random.default_rng(seed).choice(rows, size=count, replace=False))
    # Select before converting dtype or checking finiteness. This is what
    # keeps an mmap-backed, model-sized matrix bounded by ``max_rows`` instead
    # of faulting in and copying the entire input merely to assess a sample.
    sample = np.ascontiguousarray(source[selected], dtype=np.float32)
    if not np.isfinite(sample).all():
        raise ValueError("sampled matrix rows must contain only finite values")
    if importance is None:
        weighted = np.mean(np.square(sample.astype(np.float64)), axis=0).astype(np.float32)
        importance_source = "column_energy_proxy"
    else:
        weighted = np.ascontiguousarray(importance, dtype=np.float32)
        if weighted.shape != (cols,):
            raise ValueError(f"importance must have shape ({cols},), got {weighted.shape}")
        if not np.isfinite(weighted).all() or (weighted < 0).any():
            raise ValueError("importance must contain only finite, non-negative values")
        importance_source = "activation_imatrix"

    baseline_raw = quantize_q2_k(sample)
    weighted_raw = quantize_q2_k(sample, weighted)
    results = [
        _metric(
            "baseline_q2_k",
            sample,
            baseline_raw,
            dequantize_q2_k(baseline_raw, *sample.shape),
            full_rows=rows,
        ),
        _metric(
            "weighted_q2_k",
            sample,
            weighted_raw,
            dequantize_q2_k(weighted_raw, *sample.shape),
            full_rows=rows,
        ),
    ]
    if include_q3:
        q3 = quantize_q3_k(sample)
        weighted_q3 = quantize_q3_k(sample, weighted)
        results.extend(
            (
                _metric("baseline_q3_k", sample, q3, dequantize_q3_k(q3, *sample.shape), full_rows=rows),
                _metric(
                    "weighted_q3_k",
                    sample,
                    weighted_q3,
                    dequantize_q3_k(weighted_q3, *sample.shape),
                    full_rows=rows,
                ),
            )
        )
    return {
        "seed": seed,
        "source_rows": rows,
        "source_cols": cols,
        "sample_rows": [int(row) for row in selected],
        "importance_source": importance_source,
        "candidates": [asdict(result) for result in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="bounded F32 matrix saved as .npy")
    parser.add_argument("--importance", type=Path, help="optional activation-importance vector saved as .npy")
    parser.add_argument("--max-rows", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--no-q3", action="store_true", help="skip Q3_K when the native engine lacks support")
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    args = parser.parse_args()
    matrix = np.load(args.input, mmap_mode="r")
    importance = np.load(args.importance, mmap_mode="r") if args.importance else None
    report = assess_quant_candidates(
        matrix,
        importance,
        max_rows=args.max_rows,
        seed=args.seed,
        include_q3=not args.no_q3,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
