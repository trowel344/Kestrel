import struct

from kestrel.gguf.audit import (
    audit_cold_sidecar_snapshot,
    audit_gguf,
    audit_snapshot,
)


def _fields(**overrides):
    fields = {
        "general.architecture": "qwen35moe",
        "qwen35moe.block_count": 48,
        "qwen35moe.nextn_predict_layers": 0,
        "tokenizer.ggml.tokens": ["a", "b", "c", ""],
        "tokenizer.ggml.token_type": [1, 1, 1, 5],
        "tokenizer.ggml.pre": "default",
    }
    fields.update(overrides)
    return fields


def _tensors(names):
    return [(name, 1) for name in names]


def _errors(findings):
    return [f for f in findings if f.severity == "error"]


def test_valid_basic_audit():
    names = [f"blk.{i}.attn_norm.weight" for i in range(48)]
    findings = audit_snapshot(_fields(), _tensors(names))
    assert not _errors(findings)


def test_missing_token_types_is_error():
    fields = _fields()
    fields["tokenizer.ggml.token_type"] = None
    findings = audit_snapshot(fields, _tensors([f"blk.{i}.x.weight" for i in range(48)]))
    assert any(f.code == "TOKEN_TYPES_MISSING" for f in _errors(findings))


def test_missing_pretokenizer_is_error():
    fields = _fields()
    del fields["tokenizer.ggml.pre"]
    findings = audit_snapshot(fields, _tensors([f"blk.{i}.x.weight" for i in range(48)]))
    assert any(f.code == "PRETOKENIZER_MISSING" for f in _errors(findings))


def test_empty_normal_tokens_error():
    fields = _fields()
    # token type 1 = NORMAL, not 5 = UNUSED
    fields["tokenizer.ggml.token_type"] = [1, 1, 1, 1]
    findings = audit_snapshot(fields, _tensors([f"blk.{i}.x.weight" for i in range(48)]))
    assert any(f.code == "EMPTY_NORMAL_TOKENS" for f in _errors(findings))


def test_empty_unused_tokens_ok():
    fields = _fields()
    findings = audit_snapshot(fields, _tensors([f"blk.{i}.x.weight" for i in range(48)]))
    # token_type 5 = UNUSED, so empty entry is safe
    assert not any(f.code == "EMPTY_NORMAL_TOKENS" for f in findings)


def test_tensorless_model_no_errors():
    findings = audit_snapshot(_fields(), [])
    assert not any(f.code == "TARGET_BLOCKS_MISSING" for f in _errors(findings))


def test_source_missing_blocks():
    source = {"num_hidden_layers": 48}
    names = [f"blk.{i}.x.weight" for i in range(46)]  # only 46 of 48
    findings = audit_snapshot(_fields(), _tensors(names), source_config=source)
    assert any(f.code == "TARGET_BLOCKS_MISSING" for f in _errors(findings))


def test_block_count_mismatch_with_mtp():
    source = {"num_hidden_layers": 48, "mtp_num_hidden_layers": 2}
    fields = _fields(**{"qwen35moe.nextn_predict_layers": 2})
    names = [f"blk.{i}.x.weight" for i in range(48)] + [f"blk.{i}.y.weight" for i in range(48, 50)]
    findings = audit_snapshot(fields, _tensors(names), source_config=source)
    assert any(f.code == "BLOCK_COUNT_MISMATCH" for f in _errors(findings))


def test_mtp_metadata_missing():
    source = {"num_hidden_layers": 48, "mtp_num_hidden_layers": 2}
    fields = _fields()
    fields["qwen35moe.nextn_predict_layers"] = 0
    names = [f"blk.{i}.x.weight" for i in range(48)]
    names += [f"blk.{i}.y.nextn.weight" for i in range(48, 50)]
    findings = audit_snapshot(fields, _tensors(names), source_config=source)
    assert any(f.code == "MTP_METADATA_MISSING" for f in _errors(findings))


def test_f32_expansion_warning():
    names = []
    for i in range(48):
        names.append((f"blk.{i}.a.weight", 0))
    findings = audit_snapshot(_fields(), names)
    assert any(f.code == "F32_EXPANSION" and f.severity == "warning" for f in findings)


def test_cold_sidecar_arch_check():
    fields = _fields(**{"general.architecture": "llama"})
    findings = audit_cold_sidecar_snapshot(fields, [])
    assert any(f.code == "SIDECAR_ARCH" for f in findings)


def test_cold_sidecar_valid():
    fields = _fields()
    tensors = []
    for i in range(48):
        tensors.append((f"blk.{i}.ffn_gate_up_exps_lp.weight", 41))
        tensors.append((f"blk.{i}.ffn_down_exps_lp.weight", 41))
    findings = audit_cold_sidecar_snapshot(fields, tensors)
    assert not findings


def test_cold_sidecar_wrong_type():
    fields = _fields()
    tensors = [
        ("blk.0.ffn_gate_up_exps_lp.weight", 1),
        ("blk.0.ffn_down_exps_lp.weight", 41),
    ]
    findings = audit_cold_sidecar_snapshot(fields, tensors)
    assert any(f.code == "SIDECAR_TYPE" for f in findings)


def test_cold_sidecar_extra_tensor():
    fields = _fields()
    tensors = [
        ("blk.0.ffn_gate_up_exps_lp.weight", 41),
        ("blk.0.ffn_down_exps_lp.weight", 41),
        ("blk.0.attn_norm.weight", 1),
    ]
    findings = audit_cold_sidecar_snapshot(fields, tensors)
    assert any(f.code == "SIDECAR_EXTRA_TENSOR" for f in findings)

def test_unparseable_gguf_returns_report(tmp_path):
    p = tmp_path / "bad.gguf"
    p.write_bytes(b"GGUF" + struct.pack("<IQ", 3, 1) + struct.pack("<Q", 0))
    report = audit_gguf(str(p))
    assert report["valid"] is False
    assert report["errors"] == 1
    assert any(f["code"] == "GGUF_UNPARSEABLE" for f in report["findings"])
