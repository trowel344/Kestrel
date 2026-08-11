"""Small, model-free behavioral checks for GGUF conversion layout and catalogs."""

import io
import struct

import pytest

from kestrel.gguf.converter import NVFP4Converter, tensor_data_layout
from kestrel.gguf.quants import GGML_TYPE_BF16, GGML_TYPE_F32, GGML_TYPE_Q1_0, GGML_TYPE_Q2_K, GGML_TYPE_Q4_0


def _catalog_converter(*, cold_tier="off", dense_q4=False, q2_edge_layers=0):
    """Build the smallest object needed to exercise catalog construction.

    The source-reader stub deliberately exposes only a few tensors.  That
    makes the catalog test model the converter's real "missing source tensor
    means absent from catalog" behavior without requiring Torch or a shard.
    """
    converter = NVFP4Converter.__new__(NVFP4Converter)
    converter.tcfg = {
        "rope_parameters": {"partial_rotary_factor": 0.25, "mrope_section": [1, 1, 1]},
        "full_attention_interval": 2,
        "linear_value_head_dim": 4,
        "linear_num_value_heads": 2,
    }
    converter.n_layer = 2
    converter.mtp_layers = 0
    converter.hidden = 32
    converter.n_ff = 32
    converter.shared_ff = 16
    converter.n_head = 4
    converter.n_kv = 2
    converter.n_used = 1
    converter.n_exp = 2
    converter._emitted_exp = 2
    converter.head_dim = 16
    converter.vocab = 64
    converter.layer_types = ["full_attention", "linear_attention"]
    converter.dense_q4 = dense_q4
    converter.cold_tier = cold_tier
    converter.q2_edge_layers = q2_edge_layers
    converter.experts_only = False
    converter.compact_expert_ggml_type = GGML_TYPE_Q1_0  # Irrelevant when Q2 edge applies.
    converter._load_tokenizer = lambda: None

    available = {
        "model.language_model.embed_tokens.weight": ([32, 64], 32 * 64 * 2),
        "model.language_model.norm.weight": ([32], 32 * 2),
        "lm_head.weight": ([32, 64], 32 * 64 * 2),
        "model.language_model.layers.0.input_layernorm.weight": ([32], 32 * 2),
        "model.language_model.layers.0.self_attn.q_proj.weight": ([32, 32], 32 * 32 * 2),
        "model.language_model.layers.0.mlp.gate.weight": ([32, 2], 32 * 2 * 2),
        "model.language_model.layers.1.linear_attn.in_proj_z.weight": ([32, 32], 32 * 32 * 2),
    }
    converter._read_bf16_info = available.get
    return converter


def _read_gguf_string(handle):
    size = struct.unpack("<Q", handle.read(8))[0]
    return handle.read(size).decode("utf-8")


def test_tensor_data_layout_returns_contiguous_relative_offsets():
    infos = [
        ("token_embd.weight", 2, [64, 32], GGML_TYPE_BF16, 4096),
        ("blk.0.attn_norm.weight", 1, [32], GGML_TYPE_F32, 128),
        ("blk.0.ffn_gate_up_exps.weight", 3, [32, 64, 2], GGML_TYPE_Q4_0, 2304),
    ]

    layout, total = tensor_data_layout(infos)

    assert [entry[-1] for entry in layout] == [0, 4096, 4224]
    assert total == 6528


@pytest.mark.parametrize(
    "info, message",
    [
        (("bad-dims", 2, [32], GGML_TYPE_BF16, 64), "declares 2 dimensions"),
        (("bad-shape", 1, [-1], GGML_TYPE_BF16, 64), "negative dimension"),
        (("bad-size", 1, [32], GGML_TYPE_BF16, -1), "negative payload size"),
        (("bad-bytes", 2, [32, 32], GGML_TYPE_Q4_0, 575), "payload size 575, expected 576"),
        (("bad-block", 2, [31, 31], GGML_TYPE_Q4_0, 540), "row width 31"),
        (("bad-row", 2, [16, 2], GGML_TYPE_Q4_0, 18), "row width 16"),
        (("bad-type", 1, [32], 999, 128), "unsupported GGML type 999"),
    ],
)
def test_tensor_data_layout_rejects_corrupt_catalog_entries(info, message):
    with pytest.raises(ValueError, match=message):
        tensor_data_layout([info])


def test_write_ti_serializes_catalog_offsets_from_validated_layout():
    writer = NVFP4Converter.__new__(NVFP4Converter)
    writer._tensor_infos = [
        ("first", 1, [2], GGML_TYPE_F32, 8),
        ("second", 2, [4, 3], GGML_TYPE_BF16, 24),
    ]
    target = io.BytesIO(b"prefix")
    target.seek(len(b"prefix"))

    data_start = writer._write_ti(target)

    target.seek(len(b"prefix"))
    assert _read_gguf_string(target) == "first"
    assert struct.unpack("<I", target.read(4))[0] == 1
    assert struct.unpack("<Q", target.read(8))[0] == 2
    assert struct.unpack("<I", target.read(4))[0] == GGML_TYPE_F32
    assert struct.unpack("<Q", target.read(8))[0] == 0
    assert _read_gguf_string(target) == "second"
    assert struct.unpack("<I", target.read(4))[0] == 2
    assert struct.unpack("<QQ", target.read(16)) == (4, 3)
    assert struct.unpack("<I", target.read(4))[0] == GGML_TYPE_BF16
    assert struct.unpack("<Q", target.read(8))[0] == 32
    assert data_start % 32 == 0
    assert writer._data_end == data_start + 56


def test_payload_padding_matches_aligned_catalog_offsets():
    writer = NVFP4Converter.__new__(NVFP4Converter)
    writer._data_start = 64
    writer._tensor_offsets = {"first": 0, "second": 32}
    target = io.BytesIO(b"\x00" * 64)
    target.seek(64)
    target.write(b"12345678")

    writer._pad_to_tensor(target, "second")

    assert target.tell() == 96
    assert target.getvalue()[72:96] == b"\x00" * 24


def test_write_ti_rejects_a_catalog_that_cannot_be_serialized_safely():
    writer = NVFP4Converter.__new__(NVFP4Converter)
    writer._tensor_infos = [("wrong", 2, [32], GGML_TYPE_BF16, 64)]

    with pytest.raises(ValueError, match="declares 2 dimensions"):
        writer._write_ti(io.BytesIO())


def test_catalog_keeps_dense_vectors_f32_and_matrices_bf16_with_q4_experts():
    converter = _catalog_converter()

    converter._init_gguf(None)

    infos = {name: (dims, dtype, nbytes) for name, _nd, dims, dtype, nbytes in converter._tensor_infos}
    assert infos["token_embd.weight"][1] == GGML_TYPE_BF16
    assert infos["output_norm.weight"][1] == GGML_TYPE_F32
    assert infos["blk.0.attn_norm.weight"][1] == GGML_TYPE_F32
    assert infos["blk.0.attn_q.weight"][1] == GGML_TYPE_BF16
    assert infos["blk.0.ffn_gate_inp.weight"][0] == [32, 2]
    assert infos["blk.0.ffn_gate_up_exps.weight"] == ([32, 64, 2], GGML_TYPE_Q4_0, 2304)
    assert infos["blk.0.ffn_down_exps.weight"] == ([32, 32, 2], GGML_TYPE_Q4_0, 1152)
    assert not any(name.endswith("_lp.weight") for name in infos)


def test_catalog_dense_q4_uses_q4_only_for_block_aligned_matrices():
    converter = _catalog_converter(dense_q4=True)

    converter._init_gguf(None)

    infos = {name: (dims, dtype, nbytes) for name, _nd, dims, dtype, nbytes in converter._tensor_infos}
    assert infos["blk.0.attn_q.weight"] == ([32, 32], GGML_TYPE_Q4_0, 576)
    # Vectors remain F32 because llama.cpp consumes these via elementwise ops.
    assert infos["blk.0.attn_norm.weight"] == ([32], GGML_TYPE_F32, 128)


def test_pruned_router_info_rewrites_only_the_expert_dimension():
    class _Slice:
        def get_shape(self):
            return [2, 32]  # Hugging Face router layout: (experts, hidden)

    converter = NVFP4Converter.__new__(NVFP4Converter)
    converter._read_torch_slice = lambda _key: _Slice()
    converter._kept = [1]
    converter._emitted_exp = 1
    converter.n_exp = 2

    assert converter._read_bf16_info("model.language_model.layers.0.mlp.gate.weight") == ((32, 1), 64)


def test_catalog_q1_only_with_edge_tiers_omits_q4_and_marks_both_edges_q2():
    converter = _catalog_converter(cold_tier="q1_only", q2_edge_layers=1)

    converter._init_gguf(None)

    expert_infos = [info for info in converter._tensor_infos if "_exps" in info[0]]
    assert len(expert_infos) == 4
    assert all(name.endswith("_exps.weight") for name, *_rest in expert_infos)
    assert all(dtype == GGML_TYPE_Q2_K for _name, _nd, _dims, dtype, _nbytes in expert_infos)
    assert all(dims[-1] == 2 for _name, _nd, dims, _dtype, _nbytes in expert_infos)


def test_catalog_experts_only_sidecar_retains_lp_names():
    converter = _catalog_converter(cold_tier="q1_only")
    converter.experts_only = True

    converter._init_gguf(None)

    expert_infos = [info for info in converter._tensor_infos if "_exps" in info[0]]
    assert expert_infos
    assert all(name.endswith("_exps_lp.weight") for name, *_rest in expert_infos)


class _RecordingExecutor:
    def __init__(self):
        self.windows = []

    def map(self, fn, values):
        values = list(values)
        self.windows.append(values)
        return [fn(value) for value in values]


def test_windowed_expert_write_preserves_source_order_and_worker_bound():
    converter = NVFP4Converter.__new__(NVFP4Converter)
    converter._kept = [5, 1, 7, 2, 9]
    converter.n_exp = 10
    converter.conversion_workers = 2
    executor = _RecordingExecutor()
    target = io.BytesIO()

    converter._write_windowed_map(target, executor, lambda expert: bytes([expert]))

    assert target.getvalue() == bytes([5, 1, 7, 2, 9])
    assert executor.windows == [[5, 1], [7, 2], [9]]
