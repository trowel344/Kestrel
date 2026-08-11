"""Low-level GGML type constants and quant/dequant primitives.

These are the only pieces of the GGUF conversion pipeline that know the byte
layout of quantized tensors, the ctypes bridge to llama.cpp's native K-quant
routines, and the GGUF scalar value-type encodings shared by the writer
(``converter.py``) and the bounded metadata reader (``metadata.py``).  The
NVFP4Converter and the hand-rolled header writer in ``converter.py`` import
from here.
"""

from __future__ import annotations

import ctypes
import os
from functools import lru_cache

import numpy as np

# llama.cpp stores a half-sized UE4M3 block scale and compensates with this
# doubled E2M1 lookup.  NVIDIA ModelOpt source tensors do not: their E2M1
# codebook is the ordinary [0, .5, 1, ..., 6] range.
KVALUES = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.float32)
SOURCE_E2M1_VALUES = KVALUES * np.float32(0.5)

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_IQ1_S = 19
GGML_TYPE_BF16 = 30
GGML_TYPE_Q1_0 = 41

GGML_TYPE_SIZE = {0: 4, 1: 2, 2: 18, 10: 84, 11: 110, 19: 50, 30: 2, 41: 18}
GGML_TYPE_BLOCK_SIZE = {0: 1, 1: 1, 2: 32, 10: 256, 11: 256, 19: 256, 30: 1, 41: 128}

# GGUF key/value value-type id -> struct format (byte order applied per side).
# A single table so the manual GGUF writer and the metadata reader cannot drift
# on a type id again (converter.py once coded type 12 as uint64 while
# metadata.py read it as float64; the spec says float64).
GGUF_SCALAR_FORMATS = {
    0: "B",  # uint8
    1: "b",  # int8
    2: "H",  # uint16
    3: "h",  # int16
    4: "I",  # uint32
    5: "i",  # int32
    6: "f",  # float32
    7: "?",  # bool
    10: "Q",  # uint64
    11: "q",  # int64
    12: "d",  # float64
}


def dequantize_nvfp4(packed: np.ndarray, scales: np.ndarray, scale_2: float) -> np.ndarray:
    n_rows, n_packed = packed.shape
    n_cols = n_packed * 2
    out = np.empty((n_rows, n_cols), dtype=np.float32)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    # NVIDIA ModelOpt stores ordinary E2M1 code points, not signed linear
    # INT4 and not llama.cpp's doubled internal lookup values.
    out[:, 0::2] = SOURCE_E2M1_VALUES[low]
    out[:, 1::2] = SOURCE_E2M1_VALUES[high]
    gs_actual = scales.shape[1]
    if gs_actual == 0 or n_cols % gs_actual:
        raise ValueError(f"NVFP4 scale width {gs_actual} must evenly divide {n_cols} columns")
    gs = n_cols // gs_actual
    # Apply every group scale in one vectorized broadcast. The former Python
    # loop ran 64-192 iterations for every projection of every expert, making
    # full-model conversion needlessly CPU-bound.
    out.reshape(n_rows, gs_actual, gs)[:] *= (np.asarray(scales, dtype=np.float32) * np.float32(scale_2))[
        :, :, np.newaxis
    ]
    return out


def round_up(x, m):
    return ((x + m - 1) // m) * m


def ggml_type_size(t):
    return GGML_TYPE_SIZE[t]


def ggml_type_block_size(t):
    return GGML_TYPE_BLOCK_SIZE[t]


def _quantize_q1_0_buffer(buf: np.ndarray) -> bytes:
    """Pack Q1_0 rows from a private float32 buffer, consuming it.

    ``buf`` must be float32, C-contiguous, shape ``(n_rows, n_cols)`` with
    ``n_cols % 128 == 0``. The buffer is mutated in place (sign bits are read
    first, then ``|x|`` is computed in place for the mean). Byte-identical to
    ``quantize_q1_0``; callers must not reuse ``buf`` afterwards.
    """
    n_rows, n_cols = buf.shape
    blocks = buf.reshape(n_rows, n_cols // 128, 128)
    # Sign bits must be captured before the in-place absolute value, and
    # packbits in the numpy LUT-free bit path is what ``quantize_q1_0`` emits.
    signs = np.packbits(blocks >= 0.0, axis=2, bitorder="little")
    np.absolute(blocks, out=blocks)
    d = (blocks.sum(axis=2) / np.float32(128.0)).astype(np.float16)
    output = np.empty((n_rows, n_cols // 128, 18), dtype=np.uint8)
    output[:, :, :2] = d.view(np.uint8).reshape(n_rows, n_cols // 128, 2)
    output[:, :, 2:] = signs
    return output.tobytes()


def quantize_q1_0(mat: np.ndarray) -> bytes:
    """Quantize rows to llama.cpp's deterministic Q1_0 layout.

    Each block covers 128 weights: an FP16 scale equal to the mean absolute
    value, followed by 128 sign bits (bit j of byte j//8 set when the value is
    non-negative). 18 bytes per 128 weights.
    """

    n_rows, n_cols = mat.shape
    if n_cols % 128:
        raise ValueError(f"Q1_0 row width must be divisible by 128, got {n_cols}")
    # Copy so the caller's input is never mutated.
    blocks = np.array(mat, dtype=np.float32, copy=True)
    return _quantize_q1_0_buffer(blocks)


def _quantize_q4_0_buffer(buf: np.ndarray) -> bytes:
    """Pack Q4_0 rows from a private float32 buffer, consuming it.

    ``buf`` must be float32, C-contiguous, shape ``(n_rows, n_cols)`` with
    ``n_cols % 32 == 0``. The buffer is mutated in place (normalization runs
    on the private storage). Byte-identical to ``quantize_q4_0``; callers must
    not reuse ``buf`` afterwards.
    """
    n_rows, n_cols = buf.shape
    blocks = buf.reshape(n_rows, n_cols // 32, 32)
    max_indices = np.argmax(np.abs(blocks), axis=2)
    signed_max = np.take_along_axis(
        blocks,
        max_indices[:, :, np.newaxis],
        axis=2,
    )[:, :, 0]
    scales = signed_max / np.float32(-8.0)
    inverse = np.zeros_like(scales)
    with np.errstate(over="ignore"):
        np.divide(1.0, scales, out=inverse, where=scales != 0)
    # In-place normalize/round/clip reuses the single private buffer instead of
    # allocating a ~full-size float32 temporary for every op.
    np.multiply(blocks, inverse[:, :, np.newaxis], out=blocks)
    np.add(blocks, np.float32(8.5), out=blocks)
    np.trunc(blocks, out=blocks)
    np.clip(blocks, 0, 15, out=blocks)
    quants = blocks.astype(np.uint8)

    output = np.empty((n_rows, n_cols // 32, 18), dtype=np.uint8)
    output[:, :, :2] = scales.astype(np.float16).view(np.uint8).reshape(n_rows, n_cols // 32, 2)
    # Pack two 16-weight nibble halves in place on the private buffer, leaving
    # one small temporary for the final OR.
    high = quants[:, :, 16:]
    np.left_shift(high, 4, out=high)
    output[:, :, 2:] = quants[:, :, :16] | high
    return output.tobytes()


def quantize_q4_0(mat: np.ndarray) -> bytes:
    """Quantize rows using llama.cpp's deterministic Q4_0 reference layout."""

    n_rows, n_cols = mat.shape
    if n_cols % 32:
        raise ValueError(f"Q4_0 row width must be divisible by 32, got {n_cols}")
    # Copy into a private buffer (even when ``mat`` is already float32 and
    # C-contiguous) so every normalization step can run in place without the
    # caller's input being mutated and without intermediate temporaries.
    blocks = np.array(mat, dtype=np.float32, copy=True)
    return _quantize_q4_0_buffer(blocks)


def dequantize_q4_0(raw: np.ndarray) -> np.ndarray:
    """Decode llama.cpp Q4_0 rows without materializing a whole tensor.

    ``raw`` may be shaped ``[rows, row_bytes]`` or ``[rows, blocks, 18]``. The
    returned F32 matrix has 32 columns per block.
    """

    data = np.asarray(raw, dtype=np.uint8)
    if data.ndim == 2:
        if data.shape[1] % 18:
            raise ValueError(f"Q4_0 row bytes must be divisible by 18, got {data.shape[1]}")
        data = data.reshape(data.shape[0], data.shape[1] // 18, 18)
    if data.ndim != 3 or data.shape[2] != 18:
        raise ValueError(f"Q4_0 data must have shape [rows, blocks, 18], got {data.shape}")

    scales = (
        np.ascontiguousarray(data[:, :, :2]).view(np.float16).reshape(data.shape[0], data.shape[1]).astype(np.float32)
    )
    packed = data[:, :, 2:]
    values = np.empty((data.shape[0], data.shape[1], 32), dtype=np.float32)
    values[:, :, :16] = (packed & 0x0F).astype(np.float32) - np.float32(8.0)
    values[:, :, 16:] = (packed >> 4).astype(np.float32) - np.float32(8.0)
    values *= scales[:, :, np.newaxis]
    return values.reshape(data.shape[0], data.shape[1] * 32)


@lru_cache(maxsize=1)
def _load_ggml_base():
    """Load the ggml base library used for native K-quant conversion."""

    candidates = []
    explicit = os.environ.get("KESTREL_GGML_BASE_LIB")
    if explicit:
        candidates.append(explicit)
    llama_dir = os.environ.get("KESTREL_LLAMA_CPP_DIR")
    if llama_dir:
        candidates.append(os.path.join(llama_dir, "build", "bin", "libggml-base.so"))
    candidates.extend(
        os.path.join(root, "build", "bin", "libggml-base.so")
        for root in (
            os.path.expanduser("~/llama.cpp-moe-cache"),
            os.path.expanduser("~/llama.cpp"),
        )
    )
    errors = []
    for candidate in candidates:
        if not os.path.isfile(candidate):
            continue
        try:
            return ctypes.CDLL(candidate)
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
    detail = f" ({'; '.join(errors)})" if errors else ""
    raise RuntimeError(
        f"Q2_K conversion requires a built llama.cpp libggml-base.so; set KESTREL_GGML_BASE_LIB to its path{detail}"
    )


def _validate_importance(importance: np.ndarray | None, n_cols: int, quant_name: str) -> np.ndarray | None:
    """Validate and make a native quantizer's column-importance vector safe."""

    if importance is None:
        return None
    weights = np.ascontiguousarray(importance, dtype=np.float32)
    if weights.shape != (n_cols,):
        raise ValueError(f"{quant_name} importance must have shape ({n_cols},), got {weights.shape}")
    if not np.isfinite(weights).all():
        raise ValueError(f"{quant_name} importance must contain only finite values")
    if (weights < 0).any():
        raise ValueError(f"{quant_name} importance must be non-negative")
    if not np.any(weights > 0):
        raise ValueError(f"{quant_name} importance must contain at least one positive value")
    return weights


def quantize_q2_k(mat: np.ndarray, importance: np.ndarray | None = None) -> bytes:
    """Quantize F32 rows with Q2_K, optionally weighted by column importance.

    ``importance`` is the activation-derived vector consumed by llama.cpp's
    native K-quantizer.  It must have one finite, non-negative value per input
    column; rejecting malformed vectors here avoids silently producing a model
    with undefined calibration behavior.
    """

    source = np.ascontiguousarray(mat, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError(f"Q2_K input must be a matrix, got {source.shape}")
    n_rows, n_cols = source.shape
    if n_cols % 256:
        raise ValueError(f"Q2_K row width must be divisible by 256, got {n_cols}")
    weights = _validate_importance(importance, n_cols, "Q2_K")
    output_size = n_rows * (n_cols // 256) * 84
    output = np.empty(output_size, dtype=np.uint8)
    library = _load_ggml_base()
    quantize = library.quantize_q2_K
    quantize.argtypes = (
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_float),
    )
    quantize.restype = ctypes.c_size_t
    written = quantize(
        source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output.ctypes.data_as(ctypes.c_void_p),
        n_rows,
        n_cols,
        None if weights is None else weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if written != output_size:
        raise RuntimeError(f"Q2_K quantizer wrote {written} bytes, expected {output_size}")
    return output.tobytes()


def quantize_q3_k(mat: np.ndarray, importance: np.ndarray | None = None) -> bytes:
    """Quantize F32 rows with llama.cpp's Q3_K reference implementation."""

    source = np.ascontiguousarray(mat, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError(f"Q3_K input must be a matrix, got {source.shape}")
    n_rows, n_cols = source.shape
    if n_cols % 256:
        raise ValueError(f"Q3_K row width must be divisible by 256, got {n_cols}")
    weights = _validate_importance(importance, n_cols, "Q3_K")
    output_size = n_rows * (n_cols // 256) * 110
    output = np.empty(output_size, dtype=np.uint8)
    library = _load_ggml_base()
    quantize = library.quantize_q3_K
    quantize.argtypes = (
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_float),
    )
    quantize.restype = ctypes.c_size_t
    written = quantize(
        source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output.ctypes.data_as(ctypes.c_void_p),
        n_rows,
        n_cols,
        None if weights is None else weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if written != output_size:
        raise RuntimeError(f"Q3_K quantizer wrote {written} bytes, expected {output_size}")
    return output.tobytes()


def _dequantize_k(raw: bytes | np.ndarray, n_rows: int, n_cols: int, quant_name: str, block_bytes: int) -> np.ndarray:
    """Decode K-quant rows through llama.cpp without writing a temporary GGUF."""

    if n_rows <= 0 or n_cols <= 0 or n_cols % 256:
        raise ValueError(f"{quant_name} dimensions must be positive and divisible by 256, got {(n_rows, n_cols)}")
    data = np.ascontiguousarray(np.frombuffer(raw, dtype=np.uint8) if isinstance(raw, bytes) else raw, dtype=np.uint8)
    expected = n_rows * (n_cols // 256) * block_bytes
    if data.ndim != 1 or data.size != expected:
        raise ValueError(f"{quant_name} data has {data.size} bytes, expected {expected}")
    library = _load_ggml_base()
    dequantize = getattr(library, f"dequantize_row_{quant_name}")
    dequantize.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64)
    dequantize.restype = None
    row_bytes = (n_cols // 256) * block_bytes
    output = np.empty((n_rows, n_cols), dtype=np.float32)
    for row in range(n_rows):
        offset = row * row_bytes
        dequantize(
            data[offset:].ctypes.data_as(ctypes.c_void_p),
            output[row].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n_cols,
        )
    return output


def dequantize_q2_k(raw: bytes | np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    """Decode a Q2_K byte stream to an F32 matrix for bounded analysis."""

    return _dequantize_k(raw, n_rows, n_cols, "q2_K", 84)


def dequantize_q3_k(raw: bytes | np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    """Decode a Q3_K byte stream to an F32 matrix for bounded analysis."""

    return _dequantize_k(raw, n_rows, n_cols, "q3_K", 110)


def quantize_iq1_s(mat: np.ndarray, importance: np.ndarray | None = None) -> bytes:
    """Quantize an F32 matrix with IQ1_S and explicit column importance."""

    source = np.ascontiguousarray(mat, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError(f"IQ1_S input must be a matrix, got {source.shape}")
    n_rows, n_cols = source.shape
    if n_cols % 256:
        raise ValueError(f"IQ1_S row width must be divisible by 256, got {n_cols}")
    output_size = n_rows * (n_cols // 256) * 50
    output = np.empty(output_size, dtype=np.uint8)
    library = _load_ggml_base()
    library.ggml_quantize_init.argtypes = (ctypes.c_int,)
    library.ggml_quantize_init.restype = None
    library.ggml_quantize_init(GGML_TYPE_IQ1_S)
    if importance is None:
        weights = np.ones(n_cols, dtype=np.float32)
    else:
        weights = _validate_importance(importance, n_cols, "IQ1_S")
    quantize = library.quantize_iq1_s
    quantize.argtypes = (
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_float),
    )
    quantize.restype = ctypes.c_size_t
    written = quantize(
        source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output.ctypes.data_as(ctypes.c_void_p),
        n_rows,
        n_cols,
        weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if written != output_size:
        raise RuntimeError(f"IQ1_S quantizer wrote {written} bytes, expected {output_size}")
    return output.tobytes()


class ImportanceMatrix:
    """Normalized per-expert activation weights from llama-imatrix GGUF."""

    # Short calibration runs leave many MoE expert inputs completely unseen.
    # Pure zero importance lets IQ1_S spend no error budget on those columns,
    # which can produce extreme weights once a different prompt activates them.
    # A small uniform prior retains the measured activation preference while
    # making the artifact robust outside the calibration prompt.
    UNIFORM_PRIOR = np.float32(0.10)

    def __init__(self, path: str):
        import gguf

        resolved = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"importance matrix does not exist: {resolved}")
        reader = gguf.GGUFReader(resolved)
        tensors = {tensor.name: tensor for tensor in reader.tensors}
        self.path = resolved
        self.entries: dict[str, np.ndarray] = {}
        for name, tensor in tensors.items():
            if not name.endswith(".in_sum2"):
                continue
            base = name.removesuffix(".in_sum2")
            counts_tensor = tensors.get(f"{base}.counts")
            if counts_tensor is None:
                raise ValueError(f"importance matrix is missing counts for {base}")
            sums = np.asarray(tensor.data, dtype=np.float32).reshape(-1)
            counts = np.asarray(counts_tensor.data, dtype=np.float32).reshape(-1)
            if not len(counts) or len(sums) % len(counts):
                raise ValueError(f"invalid importance dimensions for {base}: {len(sums)} values, {len(counts)} counts")
            width = len(sums) // len(counts)
            normalized = sums.reshape(len(counts), width).copy()
            populated = counts > 0
            normalized[populated] /= counts[populated, np.newaxis]
            normalized[~populated] = 1.0
            row_means = normalized.mean(axis=1, keepdims=True)
            normalized = (np.float32(1.0) - self.UNIFORM_PRIOR) * normalized + self.UNIFORM_PRIOR * row_means
            self.entries[base] = normalized
        if not self.entries:
            raise ValueError(f"importance matrix contains no activation entries: {resolved}")

    def expert(self, tensor_name: str, expert: int, width: int) -> np.ndarray:
        values = self.entries.get(tensor_name)
        if values is None:
            raise KeyError(f"importance matrix is missing tensor: {tensor_name}")
        row = expert if values.shape[0] > 1 else 0
        if row >= values.shape[0]:
            raise ValueError(
                f"importance matrix has {values.shape[0]} expert rows for {tensor_name}, requested {expert}"
            )
        result = np.ascontiguousarray(values[row], dtype=np.float32)
        if result.shape != (width,):
            raise ValueError(f"importance width for {tensor_name} is {result.shape}, expected ({width},)")
        return result
