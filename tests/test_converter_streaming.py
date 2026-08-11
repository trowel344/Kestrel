import subprocess
from pathlib import Path
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


def test_generic_convert_atomically_replaces_prior_output(tmp_path, monkeypatch):
    output = tmp_path / "model.gguf"
    output.write_bytes(b"old-good")
    script = tmp_path / "convert_hf_to_gguf.py"
    script.write_text("placeholder")
    staged = []
    synced = []

    def convert(command, **_kwargs):
        stage = command[command.index("--outfile") + 1]
        staged.append(stage)
        assert stage.endswith(".partial.gguf")
        assert str(tmp_path) in stage
        with open(stage, "wb") as handle:
            handle.write(b"GGUF-new-good")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(converter_module.subprocess, "run", convert)
    monkeypatch.setattr(converter_module.util, "sync_directory", lambda path: synced.append(path))

    converter_module.generic_convert_hf_to_gguf("/source", str(output), llama_cpp_dir=str(tmp_path))

    assert output.read_bytes() == b"GGUF-new-good"
    assert not list(tmp_path.glob("*.partial.gguf"))
    assert synced == [tmp_path]


@pytest.mark.parametrize("failure", ["exit", "timeout", "interrupt"])
def test_generic_convert_failure_preserves_prior_output_and_cleans_stage(tmp_path, monkeypatch, failure):
    output = tmp_path / "model.gguf"
    output.write_bytes(b"old-good")
    (tmp_path / "convert_hf_to_gguf.py").write_text("placeholder")

    def fail(command, **_kwargs):
        stage = command[command.index("--outfile") + 1]
        with open(stage, "wb") as handle:
            handle.write(b"partial-bad")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 1)
        if failure == "interrupt":
            raise KeyboardInterrupt
        return SimpleNamespace(returncode=9, stdout="", stderr="conversion failed")

    monkeypatch.setattr(converter_module.subprocess, "run", fail)

    expected = KeyboardInterrupt if failure == "interrupt" else RuntimeError
    with pytest.raises(expected):
        converter_module.generic_convert_hf_to_gguf("/source", str(output), llama_cpp_dir=str(tmp_path))

    assert output.read_bytes() == b"old-good"
    assert not list(tmp_path.glob("*.partial.gguf"))


def test_generic_convert_discards_unbounded_logs(tmp_path, monkeypatch):
    output = tmp_path / "model.gguf"
    (tmp_path / "convert_hf_to_gguf.py").write_text("placeholder")

    def fail(_command, **kwargs):
        assert kwargs["stdout"] is subprocess.DEVNULL
        assert kwargs["stderr"] is subprocess.DEVNULL
        return SimpleNamespace(returncode=9)

    monkeypatch.setattr(converter_module.subprocess, "run", fail)

    with pytest.raises(RuntimeError, match="exit 9"):
        converter_module.generic_convert_hf_to_gguf("/source", str(output), llama_cpp_dir=str(tmp_path))

    assert not output.exists()
    assert not list(tmp_path.glob("*.partial.gguf"))


def test_generic_convert_rejects_tool_that_derives_a_different_output_path(tmp_path, monkeypatch):
    output = tmp_path / "model.gguf"
    output.write_bytes(b"old-good")
    (tmp_path / "convert_hf_to_gguf.py").write_text("placeholder")

    def derive(command, **_kwargs):
        stage = command[command.index("--outfile") + 1]
        # Simulate a converter that ignores the exact outfile and derives its
        # own sibling. Kestrel must not guess and replace the good model.
        with open(stage + ".derived", "wb") as handle:
            handle.write(b"wrong-path")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(converter_module.subprocess, "run", derive)

    with pytest.raises(RuntimeError, match="non-empty regular"):
        converter_module.generic_convert_hf_to_gguf("/source", str(output), llama_cpp_dir=str(tmp_path))

    assert output.read_bytes() == b"old-good"
    assert not list(tmp_path.glob("*.partial.gguf"))


def test_generic_convert_does_not_propagate_output_placeholders_to_stage(tmp_path, monkeypatch):
    output = tmp_path / "model-{ftype}.gguf"
    output.write_bytes(b"old-good")
    (tmp_path / "convert_hf_to_gguf.py").write_text("placeholder")

    def convert(command, **_kwargs):
        stage = command[command.index("--outfile") + 1]
        assert "{" not in stage and "}" not in stage
        with open(stage, "wb") as handle:
            handle.write(b"GGUF-new-good")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(converter_module.subprocess, "run", convert)
    converter_module.generic_convert_hf_to_gguf("/source", str(output), llama_cpp_dir=str(tmp_path))

    assert output.read_bytes() == b"GGUF-new-good"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["convert_hf_to_gguf.py", "model-{ftype}.gguf"]


def test_generic_convert_rejects_success_without_gguf_magic(tmp_path, monkeypatch):
    output = tmp_path / "model.gguf"
    output.write_bytes(b"GGUF-old-good")
    (tmp_path / "convert_hf_to_gguf.py").write_text("placeholder")

    def invalid(command, **_kwargs):
        Path(command[command.index("--outfile") + 1]).write_bytes(b"not-a-gguf")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(converter_module.subprocess, "run", invalid)
    with pytest.raises(RuntimeError, match="GGUF magic"):
        converter_module.generic_convert_hf_to_gguf("/source", str(output), llama_cpp_dir=str(tmp_path))

    assert output.read_bytes() == b"GGUF-old-good"
    assert not list(tmp_path.glob("*.partial.gguf"))
