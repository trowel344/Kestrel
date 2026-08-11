from types import SimpleNamespace

import pytest

from kestrel.gguf import converter as converter_module
from kestrel.gguf.converter import NVFP4Converter


def _writer_stub():
    writer = NVFP4Converter.__new__(NVFP4Converter)
    writer.model_dir = "/synthetic/source"
    writer._kept = None
    writer.experts_only = False
    writer.q4_sidecar_source = None
    writer._tensor_infos = []
    writer._init_gguf = lambda _output: None
    writer._write_header = lambda handle: handle.write(b"header")
    writer._write_kv = lambda handle: handle.write(b"-metadata")
    writer._write_ti = lambda handle: handle.tell()
    return writer


def test_convert_replaces_output_only_after_complete_write(tmp_path):
    output = tmp_path / "model.gguf"
    output.write_bytes(b"old artifact")
    writer = _writer_stub()

    writer.convert(str(output))

    assert output.read_bytes() == b"header-metadata"
    assert not list(tmp_path.glob("*.partial"))


def test_convert_failure_preserves_old_output_and_removes_partial(tmp_path):
    output = tmp_path / "model.gguf"
    output.write_bytes(b"old artifact")
    writer = _writer_stub()

    def fail_after_header(_handle):
        raise RuntimeError("synthetic writer failure")

    writer._write_kv = fail_after_header

    with pytest.raises(RuntimeError, match="synthetic writer failure"):
        writer.convert(str(output))

    assert output.read_bytes() == b"old artifact"
    assert not list(tmp_path.glob("*.partial"))


def test_convert_interrupt_preserves_old_output_and_removes_partial(tmp_path):
    output = tmp_path / "model.gguf"
    output.write_bytes(b"old artifact")
    writer = _writer_stub()

    def interrupt_after_header(_handle):
        raise KeyboardInterrupt

    writer._write_kv = interrupt_after_header

    with pytest.raises(KeyboardInterrupt):
        writer.convert(str(output))

    assert output.read_bytes() == b"old artifact"
    assert not list(tmp_path.glob("*.partial"))


def test_convert_refuses_insufficient_disk_before_opening_partial(tmp_path, monkeypatch):
    output = tmp_path / "model.gguf"
    writer = _writer_stub()
    writer._tensor_infos = [("large.weight", 2, [1, 1], 0, 1024)]
    monkeypatch.setattr(
        converter_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=1, used=1, free=0),
    )

    with pytest.raises(OSError, match="Insufficient disk space"):
        writer.convert(str(output))

    assert not output.exists()
    assert not list(tmp_path.glob("*.partial"))


def test_convert_uses_unique_partial_files_and_syncs_directory(tmp_path, monkeypatch):
    output = tmp_path / "model.gguf"
    writer = _writer_stub()
    partial_paths = []
    real_mkstemp = converter_module.tempfile.mkstemp

    def record_mkstemp(**kwargs):
        fd, path = real_mkstemp(**kwargs)
        partial_paths.append(path)
        return fd, path

    synced = []
    monkeypatch.setattr(converter_module.tempfile, "mkstemp", record_mkstemp)
    monkeypatch.setattr(converter_module.util, "sync_directory", lambda path: synced.append(path))

    writer.convert(str(output))
    writer.convert(str(output))

    assert len(set(partial_paths)) == 2
    assert synced == [str(tmp_path), str(tmp_path)]
