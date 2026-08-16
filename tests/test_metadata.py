import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from gguf_fixture import write_gguf  # noqa: E402

from kestrel.gguf.metadata import GGUFMetadataError, read_planner_metadata  # noqa: E402


def test_read_moe_metadata(tmp_path):
    p = write_gguf(
        tmp_path / "m.gguf",
        n_layer=48,
        n_exp=256,
        n_used=8,
        hidden=3072,
        n_ff=1024,
        mtp_layers=2,
    )
    m = read_planner_metadata(p)
    assert m["architecture"] == "qwen35moe"
    assert m["n_layer"] == 48
    assert m["n_exp"] == 256
    assert m["n_used"] == 8
    assert m["hidden"] == 3072
    assert m["n_ff"] == 1024
    assert m["mtp_layers"] == 2
    assert m["gguf_version"] == 3


def test_read_dense_metadata(tmp_path):
    p = write_gguf(
        tmp_path / "d.gguf",
        architecture="llama",
        n_layer=32,
        n_exp=0,
        n_used=0,
        hidden=4096,
        n_ff=11008,
    )
    m = read_planner_metadata(p)
    assert m["architecture"] == "llama"
    assert m["n_exp"] == 0
    assert m["n_used"] == 0


def test_metadata_without_tokenizer_keys(tmp_path):
    p = write_gguf(
        tmp_path / "t.gguf",
        n_layer=48,
        n_exp=256,
        n_used=8,
        hidden=3072,
        n_ff=1024,
        tokenizer_keys=False,
    )
    m = read_planner_metadata(p)
    assert m["n_layer"] == 48
    assert m["n_exp"] == 256


def test_invalid_magic(tmp_path):
    p = tmp_path / "bad.gguf"
    p.write_bytes(b"NOTGGUF" + b"\x00" * 16)
    with pytest.raises(GGUFMetadataError):
        read_planner_metadata(p)


def test_truncated_file(tmp_path):
    p = tmp_path / "trunc.gguf"
    p.write_bytes(b"GGUF")
    with pytest.raises(GGUFMetadataError):
        read_planner_metadata(p)


def test_nonfile_missing(tmp_path):
    with pytest.raises(OSError):
        read_planner_metadata(tmp_path / "missing.gguf")


def test_truncated_scalar_array_raises(tmp_path):
    p = tmp_path / "arr.gguf"
    body = bytearray(b"GGUF")
    body += struct.pack("<IQ", 3, 0)  # version, tensor_count
    body += struct.pack("<Q", 1)  # kv_count
    key = b"arch.test_key"
    body += struct.pack("<Q", len(key)) + key
    body += struct.pack("<I", 9)  # value type: array
    body += struct.pack("<I", 4)  # element type: uint32
    body += struct.pack("<Q", 8)  # declares 8 elements (32 bytes) that never follow
    p.write_bytes(bytes(body))
    with pytest.raises(GGUFMetadataError):
        read_planner_metadata(p)


def test_present_scalar_array_parses(tmp_path):
    p = tmp_path / "arr2.gguf"
    body = bytearray(b"GGUF")
    body += struct.pack("<IQ", 3, 0)
    body += struct.pack("<Q", 2)  # kv_count
    key = b"general.architecture"
    body += struct.pack("<Q", len(key)) + key
    body += struct.pack("<I", 8)  # value type: string
    value = b"qwen35moe"
    body += struct.pack("<Q", len(value)) + value
    key = b"arch.test_key"
    body += struct.pack("<Q", len(key)) + key
    body += struct.pack("<I", 9)  # value type: array
    body += struct.pack("<I", 4)  # element type: uint32
    body += struct.pack("<Q", 2)
    body += struct.pack("<II", 1, 2)  # element bytes present
    p.write_bytes(bytes(body))
    m = read_planner_metadata(p)
    assert m["architecture"] == "qwen35moe"


def test_file_type_read_when_written_after_tokenizer_section(tmp_path):
    """general.file_type emitted at the end is still recovered via the resume scan."""
    p = write_gguf(
        tmp_path / "late.gguf",
        n_layer=53,
        n_exp=128,
        n_used=6,
        hidden=2688,
        n_ff=1856,
        tokenizer_keys=True,
        file_type=30,
    )
    m = read_planner_metadata(p)
    assert m["file_type"] == 30


def test_file_type_missing_defaults_zero(tmp_path):
    p = write_gguf(
        tmp_path / "none.gguf",
        n_layer=48,
        n_exp=256,
        n_used=8,
        hidden=3072,
        n_ff=1024,
    )
    m = read_planner_metadata(p)
    assert m["file_type"] == 0


def test_planner_unknown_arch_defaults(tmp_path):
    p = write_gguf(
        tmp_path / "u.gguf",
        architecture="mystery",
        n_layer=0,
        n_exp=0,
        n_used=0,
        hidden=0,
        n_ff=0,
        tokenizer_keys=False,
    )
    m = read_planner_metadata(p)
    assert m["n_layer"] == 0


def test_arch_suffix_mismatch_ignored(tmp_path):
    """Keys whose architecture does not match general.architecture are ignored."""
    p = write_gguf(
        tmp_path / "s.gguf",
        architecture="llama",
        n_layer=32,
        n_exp=0,
        n_used=0,
        hidden=4096,
        n_ff=11008,
    )
    m = read_planner_metadata(p)
    # n_exp stays 0 because no llama.expert_count was emitted
    assert m["n_exp"] == 0


def test_attention_dims_parsed(tmp_path):
    p = write_gguf(
        tmp_path / "attn.gguf",
        architecture="qwen35moe",
        n_layer=48,
        n_exp=128,
        n_used=8,
        hidden=2048,
        n_ff=1024,
        n_heads=16,
        n_kv_heads=8,
        head_dim=128,
    )
    m = read_planner_metadata(p)
    assert m["n_heads"] == 16
    assert m["n_kv_heads"] == 8
    assert m["head_dim"] == 128


def test_attention_dims_missing_default_to_zero(tmp_path):
    p = write_gguf(
        tmp_path / "noattn.gguf",
        architecture="llama",
        n_layer=32,
        n_exp=0,
        n_used=0,
        hidden=4096,
        n_ff=11008,
    )
    m = read_planner_metadata(p)
    assert m["n_kv_heads"] == 0
    assert m["head_dim"] == 0


def _hybrid_tensors() -> list[tuple[str, tuple[int, ...]]]:
    tensors: list[tuple[str, tuple[int, ...]]] = []
    for lay in (5, 12, 19, 26, 33, 42, 52):
        tensors.append((f"blk.{lay}.attn_norm.weight", (2688,)))
        tensors.append((f"blk.{lay}.attn_q.weight", (2688, 4096)))
        tensors.append((f"blk.{lay}.attn_k.weight", (2688, 256)))
        tensors.append((f"blk.{lay}.attn_v.weight", (2688, 256)))
        tensors.append((f"blk.{lay}.attn_output.weight", (4096, 2688)))
    for lay in (7, 8, 9):
        tensors.append((f"blk.{lay}.ssm_in.weight", (2688, 1536)))
        tensors.append((f"blk.{lay}.ssm_out.weight", (1536, 2688)))
    for lay in (0, 1, 2):
        tensors.append((f"blk.{lay}.ffn_up_exps.weight", (1024, 2688, 128)))
    tensors.append(("token_embd.weight", (3072, 2688)))
    tensors.append(("output.weight", (2688, 3072)))
    return tensors


def test_hybrid_attention_kv_layout_scanned(tmp_path):
    """Hybrid models store KV only for full-attention layers; the tensor scan
    must count those (7) and their KV dim (8 heads * 32 = 256), not n_layer."""
    p = write_gguf(
        tmp_path / "hybrid.gguf",
        architecture="nemotron_h_moe",
        n_layer=53,
        n_exp=128,
        n_used=6,
        hidden=2688,
        n_ff=1856,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        tensors=_hybrid_tensors(),
    )
    m = read_planner_metadata(p)
    assert m["kv_layers"] == 7
    assert m["kv_values_per_token"] == 256
    assert m["n_tensors"] == len(_hybrid_tensors())


def test_kv_layout_missing_defaults_zero(tmp_path):
    p = write_gguf(
        tmp_path / "layoutless.gguf",
        architecture="qwen35moe",
        n_layer=48,
        n_exp=256,
        n_used=8,
        hidden=3072,
        n_ff=1024,
    )
    m = read_planner_metadata(p)
    assert m["kv_layers"] == 0
    assert m["kv_values_per_token"] == 0
    assert m["n_tensors"] == 0


def test_planner_disk_cache_rejects_non_object_metadata(tmp_path, monkeypatch):
    import kestrel.gguf.metadata as metadata

    monkeypatch.setenv("KESTREL_CACHE_DIR", str(tmp_path / "cache"))
    model = tmp_path / "model.gguf"
    cache = metadata._planner_cache_path(model)
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({"file_size": 12, "file_mtime_ns": 34, "metadata": ["bad"]}))

    assert metadata._planner_cache_read(model, 12, 34) is None
