"""Tests for the NVFP4 -> Q4_0 -> Q1_0 cascade cosine metric."""

import struct

import pytest

from kestrel.analysis.q1_cascade_quality import (
    CascadeSample,
    q1_0_cosine,
    summarize,
)


def _block(magnitude: float, signs: int) -> bytes:
    """One Q1_0 block: fp16 magnitude + 16 sign bytes (128 bits)."""
    return struct.pack("<e", magnitude) + signs.to_bytes(16, "little")


def _stream(magnitudes: list[float], sign_masks: list[int]) -> bytes:
    return b"".join(_block(m, s) for m, s in zip(magnitudes, sign_masks, strict=True))


def test_cosine_identical_is_one():
    stream = _stream([1.0, 2.0, 3.0], [0x00, 0xFF, 0xAA])
    assert q1_0_cosine(stream, stream) == pytest.approx(1.0)


def test_cosine_all_signs_flipped_is_minus_one():
    flip = (1 << 128) - 1
    a = _stream([1.0, 2.0, 3.0], [0x00, 0x00, 0x00])
    b = _stream([1.0, 2.0, 3.0], [flip, flip, flip])
    assert q1_0_cosine(a, b) == pytest.approx(-1.0)


def test_cosine_half_signs_flipped():
    flip = (1 << 64) - 1
    a = _stream([1.0], [0x00])
    b = _stream([1.0], [flip])
    value = q1_0_cosine(a, b)
    assert -1.0 < value < 1.0


def test_cosine_rejects_length_mismatch():
    a = _stream([1.0], [0x00])
    b = _stream([1.0, 2.0], [0x00, 0x00])
    with pytest.raises(ValueError):
        q1_0_cosine(a, b)


def test_cosine_rejects_non_multiple_of_18():
    with pytest.raises(ValueError):
        q1_0_cosine(b"x" * 20, b"y" * 20)


def test_cosine_zero_norm_handled():
    a = _stream([0.0], [0x00])
    b = _stream([0.0], [0x00])
    assert q1_0_cosine(a, b) == 1.0


def test_cosine_is_symmetric():
    a = _stream([1.0, 2.0], [0x00, 0xFF])
    b = _stream([2.0, 1.0], [0xFF, 0x00])
    assert q1_0_cosine(a, b) == pytest.approx(q1_0_cosine(b, a))


def test_summarize_empty():
    report = summarize([])
    assert report["by_tensor"] == {}
    assert report["samples"] == []


def test_summarize_groups_by_tensor():
    samples = [
        CascadeSample(layer=0, expert=1, tensor="gate_up", cosine=0.9),
        CascadeSample(layer=0, expert=2, tensor="gate_up", cosine=0.8),
        CascadeSample(layer=0, expert=3, tensor="down", cosine=0.7),
    ]
    report = summarize(samples)
    assert report["by_tensor"]["gate_up"]["count"] == 2
    assert report["by_tensor"]["gate_up"]["mean"] == pytest.approx(0.85)
    assert report["by_tensor"]["down"]["count"] == 1
    assert len(report["samples"]) == 3
