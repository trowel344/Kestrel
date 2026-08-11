import numpy as np
import pytest

from kestrel.analysis.quant_candidates import assess_quant_candidates
from kestrel.gguf.quants import quantize_q2_k


def test_q2_importance_is_strictly_validated():
    matrix = np.ones((1, 256), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        quantize_q2_k(matrix, np.ones(255, dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        quantize_q2_k(matrix, np.array([np.nan] + [1.0] * 255, dtype=np.float32))
    with pytest.raises(ValueError, match="non-negative"):
        quantize_q2_k(matrix, np.array([-1.0] + [1.0] * 255, dtype=np.float32))
    with pytest.raises(ValueError, match="positive"):
        quantize_q2_k(matrix, np.zeros(256, dtype=np.float32))


def test_assessor_is_bounded_and_deterministic():
    matrix = np.arange(12 * 256, dtype=np.float32).reshape(12, 256) / 1000.0
    first = assess_quant_candidates(matrix, max_rows=3, seed=7)
    second = assess_quant_candidates(matrix, max_rows=3, seed=7)
    assert first == second
    assert len(first["sample_rows"]) == 3
    assert [item["name"] for item in first["candidates"]] == [
        "baseline_q2_k",
        "weighted_q2_k",
        "baseline_q3_k",
        "weighted_q3_k",
    ]
    assert all(item["sample_bytes"] > 0 for item in first["candidates"])
    assert all(np.isfinite(item["cosine"]) and np.isfinite(item["nrmse"]) for item in first["candidates"])


def test_assessor_rejects_bad_inputs():
    with pytest.raises(ValueError, match="divisible"):
        assess_quant_candidates(np.ones((2, 255), dtype=np.float32))
    with pytest.raises(ValueError, match="importance"):
        assess_quant_candidates(np.ones((2, 256), dtype=np.float32), np.ones(255, dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        assess_quant_candidates(np.array([[np.nan] + [1.0] * 255], dtype=np.float32))


def test_assessor_only_materializes_selected_rows():
    class BoundedMatrix:
        shape = (1_000_000, 256)
        ndim = 2

        def __array__(self, *_args, **_kwargs):
            raise AssertionError("full matrix materialized")

        def __getitem__(self, rows):
            assert len(rows) == 2
            return np.ones((2, 256), dtype=np.float32)

    report = assess_quant_candidates(BoundedMatrix(), max_rows=2)
    assert len(report["sample_rows"]) == 2
