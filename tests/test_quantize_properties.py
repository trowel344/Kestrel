"""Deterministic property tests that the Q4_0/Q1_0 quantizers are byte-exact.

The in-place buffer quantizers ``_quantize_q4_0_buffer`` / ``_quantize_q1_0_buffer``
must produce *exactly* the same bytes as their public counterparts
``quantize_q4_0`` / ``quantize_q1_0``. We exercise many seeded random float32
rows (``n_cols % 32 == 0`` for Q4_0, ``n_cols % 128 == 0`` for Q1_0) plus a
battery of edge regimes (all zeros, same value, negative-heavy, denormals, max
magnitude, mixed signs). Every row is derived from :class:`random.Random`
seeds, so the results are fully deterministic.

If the two paths ever diverge the assertion fails immediately with the failing
vector in the message; the divergence is never masked or swallowed.
"""

import random
import warnings
from contextlib import contextmanager

import numpy as np

from kestrel.gguf.converter import (
    _quantize_q1_0_buffer,
    _quantize_q4_0_buffer,
    quantize_q1_0,
    quantize_q4_0,
)

_Q4_COLS = (32, 64, 96, 128, 160, 192, 224, 256)
_Q1_COLS = (128, 256, 384)
_SEED = 0xC0FFEE


def _fmt_vector(mat: np.ndarray) -> str:
    return f"shape={mat.shape}, values={mat.tolist()!r}"


def _random_rows(rng: random.Random, n_rows: int, n_cols: int) -> np.ndarray:
    return np.array(
        [[rng.uniform(-6.0, 6.0) for _ in range(n_cols)] for _ in range(n_rows)],
        dtype=np.float32,
    )


def _wide_random_rows(rng: random.Random, n_rows: int, n_cols: int) -> np.ndarray:
    """Rows spanning ~1e-38..1e30 so rounding is stressed across every exponent."""
    rows = []
    for _ in range(n_rows):
        rows.append(
            [
                (-1.0 if rng.random() < 0.5 else 1.0)
                * (10.0 ** rng.uniform(-38.0, 30.0))
                for _ in range(n_cols)
            ]
        )
    return np.array(rows, dtype=np.float32)


def _subnormal() -> np.float32:
    return np.nextafter(np.float32(0.0), np.float32(1.0))


def _q4_regimes():
    rng = random.Random(_SEED)
    yield "random_a", _random_rows(rng, 3, 64)
    yield "random_b", _random_rows(rng, 2, 128)
    yield "wide_random", _wide_random_rows(rng, 2, 128)
    yield "all_zeros", np.zeros((2, 64), dtype=np.float32)
    yield "same_value", np.full((3, 32), 2.5, dtype=np.float32)
    yield "negative_heavy", np.full((2, 32), -3.0, dtype=np.float32)
    yield "denormals", np.tile(
        np.array([_subnormal(), -_subnormal()], dtype=np.float32), (2, 16)
    )
    yield "max_magnitude", np.full((1, 32), np.float32(3.4e38), dtype=np.float32)
    yield "mixed_signs", np.tile(
        np.array([1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0], dtype=np.float32),
        (2, 4),
    )


def _q1_regimes():
    rng = random.Random(0xBEA67)
    yield "random_a", _random_rows(rng, 2, 256)
    yield "random_b", _random_rows(rng, 4, 128)
    yield "wide_random", _wide_random_rows(rng, 2, 128)
    yield "all_zeros", np.zeros((2, 128), dtype=np.float32)
    yield "same_value", np.full((1, 128), 3.0, dtype=np.float32)
    yield "negative_heavy", np.full((2, 128), -4.0, dtype=np.float32)
    yield "denormals", np.tile(
        np.array([_subnormal(), -_subnormal()], dtype=np.float32), (2, 64)
    )
    yield "max_magnitude", np.full((1, 128), np.float32(3.4e38), dtype=np.float32)
    yield "mixed_signs", np.tile(
        np.array([1.0, -1.0, 2.0, -0.0], dtype=np.float32), (2, 32)
    )


def _assert_q4_byte_exact(mat: np.ndarray) -> None:
    mat = np.asarray(mat, dtype=np.float32)
    ref = quantize_q4_0(mat)
    buf = _quantize_q4_0_buffer(np.array(mat, dtype=np.float32, copy=True))
    assert buf == ref, (
        "Q4_0 buffer quantizer diverged from quantize_q4_0 on vector "
        + _fmt_vector(mat)
    )


def _assert_q1_byte_exact(mat: np.ndarray) -> None:
    mat = np.asarray(mat, dtype=np.float32)
    ref = quantize_q1_0(mat)
    buf = _quantize_q1_0_buffer(np.array(mat, dtype=np.float32, copy=True))
    assert buf == ref, (
        "Q1_0 buffer quantizer diverged from quantize_q1_0 on vector "
        + _fmt_vector(mat)
    )


def _assert_q4_blocks(mat: np.ndarray) -> None:
    mat = np.asarray(mat, dtype=np.float32)
    raw = np.frombuffer(quantize_q4_0(mat), dtype=np.uint8).reshape(-1, 18)
    packed = raw[:, 2:]
    assert packed.size == 0 or (packed & 0x0F).min() >= 0 and (packed >> 4).max() <= 15, (
        "Q4_0 stored a nibble outside 0..15 on vector " + _fmt_vector(mat)
    )

    # Each block's stored fp16 scale must be the fp16 of signed_max / -8,
    # where signed_max is the signed value with the largest |x| in the block.
    blocks = mat.reshape(-1, 32)
    idx = np.argmax(np.abs(blocks), axis=1)
    signed_max = blocks[np.arange(blocks.shape[0]), idx]
    stored = np.ascontiguousarray(raw[:, :2]).view(np.float16).reshape(-1)
    expected = (signed_max / np.float32(-8.0)).astype(np.float16)
    assert np.array_equal(stored, expected), (
        "Q4_0 stored scale is not fp16(signed_max/-8) on vector "
        + _fmt_vector(mat)
        + f"\nstored={stored!r}\nexpected={expected!r}"
    )


def _assert_q1_signs(mat: np.ndarray) -> None:
    mat = np.asarray(mat, dtype=np.float32)
    raw = np.frombuffer(quantize_q1_0(mat), dtype=np.uint8).reshape(-1, 18)
    got = np.unpackbits(raw[:, 2:], axis=1, bitorder="little").reshape(-1, 128)
    expected = (mat.reshape(-1, 128) >= 0).astype(np.uint8)
    assert np.array_equal(got, expected), (
        "Q1_0 sign bits do not match the input sign pattern on vector "
        + _fmt_vector(mat)
    )


def _exercise(mat_q4: np.ndarray, mat_q1: np.ndarray) -> None:
    _assert_q4_byte_exact(mat_q4)
    _assert_q4_blocks(mat_q4)
    _assert_q1_byte_exact(mat_q1)
    _assert_q1_signs(mat_q1)


@contextmanager
def _quiet_overflows():
    # max-magnitude rows overflow fp16 scale casts by design; silence the noise.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


def test_regimes_byte_exact():
    q1_by_name = dict(_q1_regimes())
    for name, mat in _q4_regimes():
        with _quiet_overflows():
            _exercise(mat, q1_by_name.get(name, q1_by_name["random_a"]))


def test_property_q4_random_byte_exact():
    rng = random.Random(_SEED)
    for _ in range(60):
        n_cols = rng.choice(_Q4_COLS)
        n_rows = rng.randint(1, 4)
        mat = _random_rows(rng, n_rows, n_cols)
        with _quiet_overflows():
            _assert_q4_byte_exact(mat)
            _assert_q4_blocks(mat)


def test_property_q1_random_byte_exact():
    rng = random.Random(_SEED)
    for _ in range(60):
        n_cols = rng.choice(_Q1_COLS)
        n_rows = rng.randint(1, 4)
        mat = _random_rows(rng, n_rows, n_cols)
        with _quiet_overflows():
            _assert_q1_byte_exact(mat)
            _assert_q1_signs(mat)
