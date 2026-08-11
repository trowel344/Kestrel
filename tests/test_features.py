import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from gguf_fixture import write_gguf  # noqa: E402

from kestrel import cli  # noqa: E402
from kestrel.backends.llama_cpp import LlamaCppBackend, LlamaCppCapabilities  # noqa: E402
from kestrel.gguf import metadata  # noqa: E402

# --- Multi-GPU detection ------------------------------------------------------


def _fake_gpu_smi(stdout: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(stdout=stdout, returncode=0, stderr="")


def test_detect_gpus_parses_multiple_rows(monkeypatch):
    rows = "NVIDIA RTX 4090, 24564, 20000\nNVIDIA A100, 81920, 60000\n"
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: _fake_gpu_smi(rows))
    devices = cli.detect_gpus()
    assert len(devices) == 2
    assert devices[0]["vram_total_mb"] == 24564
    assert devices[1]["vram_total_mb"] == 81920


def test_detect_gpu_aggregates_multi_gpu():
    gpu = cli._aggregate_gpu(
        [
            {"name": "NVIDIA RTX 4090", "vram_total_mb": 24576, "vram_free_mb": 20000},
            {"name": "NVIDIA A100", "vram_total_mb": 24576, "vram_free_mb": 60000},
        ]
    )
    assert gpu["count"] == 2
    assert gpu["vram_total_mb"] == 24576 + 24576
    assert gpu["vram_free_mb"] == 80000
    assert gpu["name"].startswith("NVIDIA RTX 4090")
    assert len(gpu["devices"]) == 2


def test_detect_gpu_aggregate_single_device_keeps_shape():
    gpu = cli._aggregate_gpu([{"name": "NVIDIA RTX 4090", "vram_total_mb": 24576, "vram_free_mb": 20000}])
    assert gpu["count"] == 1
    assert gpu["devices"][0]["name"] == "NVIDIA RTX 4090"
    assert gpu["name"] == "NVIDIA RTX 4090"


def test_detect_gpu_aggregate_empty_is_none():
    assert cli._aggregate_gpu([]) is None


def test_detect_gpus_error_returns_empty(monkeypatch):
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *a, **k: _fake_gpu_smi(""),
    )
    assert cli.detect_gpus() == []


def test_detect_gpus_permission_error_returns_empty(monkeypatch):
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("denied")),
    )
    assert cli.detect_gpus() == []


# -----------------------------------------------------------------------------
# tensor-split + extra passthrough helpers ------------------------------------


def test_tensor_split_arg_explicit_wins():
    gpu = {
        "devices": [
            {"vram_free_mb": 20000},
            {"vram_free_mb": 60000},
        ]
    }
    assert cli._tensor_split_arg(gpu, "70,30") == "70,30"


def test_tensor_split_arg_auto_ratios():
    gpu = {
        "devices": [
            {"vram_free_mb": 20000},
            {"vram_free_mb": 60000},
        ]
    }
    assert cli._tensor_split_arg(gpu, None) == "25,75"


def test_tensor_split_arg_single_gpu_none():
    gpu = {"devices": [{"vram_free_mb": 40960}]}
    assert cli._tensor_split_arg(gpu, None) is None
    assert cli._tensor_split_arg(None, None) is None


def test_flatten_extra_splits_shell_words():
    assert cli._flatten_extra(["--temp 0.4 --top-p 0.9"]) == [
        "--temp",
        "0.4",
        "--top-p",
        "0.9",
    ]
    assert cli._flatten_extra(["--a 1", "--b 2"]) == ["--a", "1", "--b", "2"]
    assert cli._flatten_extra([]) == []
    assert cli._flatten_extra(None) == []


# -----------------------------------------------------------------------------
# backend: mlock / tensor-split / extra passthrough ---------------------------


def _backend_with_caps(caps: LlamaCppCapabilities, tmp_path: Path) -> LlamaCppBackend:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF" + b"\x00" * 8)
    binary = tmp_path / "build" / "bin" / "llama-cli"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    backend = LlamaCppBackend(str(model), llama_cpp_dir=str(tmp_path))
    backend._capabilities = caps
    return backend


def test_base_cmd_mlock_and_extra_and_split(tmp_path):
    caps = LlamaCppCapabilities(help_text="--mlock\n--tensor-split\n--mmap\n--no-mmap\n")
    backend = _backend_with_caps(caps, tmp_path)
    backend.use_mmap = True
    backend.use_mlock = True
    backend.tensor_split = "25,75"
    backend.extra_args = ["--temp", "0.4"]
    cmd = backend._base_cmd()
    assert "--mlock" in cmd
    assert "--tensor-split" in cmd and "25,75" in cmd
    assert "--temp" in cmd and "0.4" in cmd


def test_base_cmd_mlock_only_when_supported(tmp_path):
    caps = LlamaCppCapabilities(help_text="--mmap\n")
    backend = _backend_with_caps(caps, tmp_path)
    backend.use_mlock = True
    cmd = backend._base_cmd()
    assert "--mlock" not in cmd


# -----------------------------------------------------------------------------
# generic HF -> GGUF conversion ------------------------------------------------


def test_find_convert_script_build_root(tmp_path):
    from kestrel.gguf.converter import find_convert_script

    build = tmp_path / "llama.cpp" / "build"
    build.mkdir(parents=True)
    (tmp_path / "llama.cpp" / "convert_hf_to_gguf.py").write_text("#!/usr/bin/env python\n")
    assert find_convert_script(str(build)) == str(tmp_path / "llama.cpp" / "convert_hf_to_gguf.py")


def test_find_convert_script_direct_dir(tmp_path, monkeypatch):
    from kestrel.gguf.converter import find_convert_script

    (tmp_path / "convert_hf_to_gguf.py").write_text("x")
    monkeypatch.delenv("KESTREL_LLAMA_CPP_DIR", raising=False)
    assert find_convert_script(str(tmp_path)) == str(tmp_path / "convert_hf_to_gguf.py")


def test_find_convert_script_missing(tmp_path, monkeypatch):
    from kestrel.gguf.converter import find_convert_script

    monkeypatch.setattr(
        "kestrel.gguf.converter.os.path.expanduser",
        lambda p: str(tmp_path / "missing-home"),
    )
    assert find_convert_script(str(tmp_path / "nope")) is None


def test_generic_convert_hf_to_gguf_runs_script(tmp_path, monkeypatch):
    from kestrel.gguf.converter import generic_convert_hf_to_gguf

    (tmp_path / "convert_hf_to_gguf.py").write_text("x")
    calls = {}

    def fake_run(cmd, **kwargs):
        calls["cmd"] = cmd
        Path(cmd[cmd.index("--outfile") + 1]).write_bytes(b"GGUF")
        return types.SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("kestrel.gguf.converter.subprocess.run", fake_run)
    command, code = generic_convert_hf_to_gguf(
        str(tmp_path / "model_hf"),
        str(tmp_path / "out.gguf"),
        outtype="q8_0",
        llama_cpp_dir=str(tmp_path),
    )
    assert code == 0
    assert "--outtype" in calls["cmd"]
    assert calls["cmd"][-1] == "q8_0"
    staged = calls["cmd"][calls["cmd"].index("--outfile") + 1]
    assert staged.endswith(".partial.gguf")
    assert (tmp_path / "out.gguf").read_bytes() == b"GGUF"
    assert str(tmp_path / "out.gguf") in command
    assert ".partial.gguf" not in command


def test_generic_convert_hf_to_gguf_propagates_failure(tmp_path, monkeypatch):
    from kestrel.gguf.converter import generic_convert_hf_to_gguf

    (tmp_path / "convert_hf_to_gguf.py").write_text("x")
    monkeypatch.setattr(
        "kestrel.gguf.converter.subprocess.run",
        lambda *a, **k: types.SimpleNamespace(returncode=1, stdout="", stderr="boom"),
    )
    with pytest.raises(RuntimeError, match="exit 1"):
        generic_convert_hf_to_gguf(str(tmp_path / "src"), str(tmp_path / "m.gguf"), llama_cpp_dir=str(tmp_path))


def test_generic_convert_timeout_is_typed(tmp_path, monkeypatch):
    import subprocess

    from kestrel.gguf.converter import generic_convert_hf_to_gguf

    (tmp_path / "convert_hf_to_gguf.py").write_text("x")
    monkeypatch.setattr(
        "kestrel.gguf.converter.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired(args[0], 1)),
    )

    with pytest.raises(RuntimeError, match="24-hour"):
        generic_convert_hf_to_gguf(str(tmp_path / "src"), str(tmp_path / "m.gguf"), llama_cpp_dir=str(tmp_path))


def test_generic_convert_missing_script_raises(tmp_path, monkeypatch):
    from kestrel.gguf.converter import generic_convert_hf_to_gguf

    monkeypatch.setattr(
        "kestrel.gguf.converter.os.path.expanduser",
        lambda p: str(tmp_path / "missing-home"),
    )
    with pytest.raises(FileNotFoundError, match="convert_hf_to_gguf.py"):
        generic_convert_hf_to_gguf(str(tmp_path / "m"), "/out/m.gguf", llama_cpp_dir=str(tmp_path / "empty"))


def test_cmd_convert_generic_branch(monkeypatch, tmp_path, capsys):
    args = types.SimpleNamespace(model="owner/repo", output="/out/m.gguf", generic=True, outtype="q8_0")
    # detect_model says it is a downloaded safetensors input at a fake path.
    monkeypatch.setattr(
        cli.model_source, "detect_model", lambda m: {"type": "safetensors", "path": "/fake/hf", "gguf_name": None}
    )
    seen = {}

    def fake_converter(src, out, *, outtype, llama_cpp_dir):
        seen.update(src=src, out=out, outtype=outtype)
        return "cmd", 0

    monkeypatch.setattr("kestrel.gguf.converter.generic_convert_hf_to_gguf", fake_converter)
    monkeypatch.setattr(cli.state, "LLAMA_CPP_DIR", "/nonexistent")
    cli.cmd_convert(args)
    assert seen["src"] == "/fake/hf"
    assert seen["out"] == "/out/m.gguf"
    assert seen["outtype"] == "q8_0"


# -----------------------------------------------------------------------------
# persistent metadata cache ---------------------------------------------------


def test_persistent_metadata_cache_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_CACHE_DIR", str(tmp_path / "cache"))
    gguf = write_gguf(tmp_path / "m.gguf", n_layer=48, n_exp=256, n_used=8)
    m1 = metadata.read_planner_metadata(gguf)
    assert m1["n_layer"] == 48
    cache_files = list((tmp_path / "cache").glob("planner-*.json"))
    assert len(cache_files) == 1

    # Bypass the in-process single-entry cache so we exercise disk read-back.
    metadata._planner_metadata_cache = None
    m2 = metadata.read_planner_metadata(gguf)
    assert m2 == m1


def test_persistent_metadata_cache_invalidates_on_change(tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_CACHE_DIR", str(tmp_path / "cache"))
    gguf = write_gguf(tmp_path / "m.gguf", n_layer=48)
    metadata._planner_metadata_cache = None
    assert metadata.read_planner_metadata(gguf)["n_layer"] == 48
    # Rewrite with a different size (stale cache key must not be reused).
    write_gguf(tmp_path / "m.gguf", n_layer=12)
    metadata._planner_metadata_cache = None
    assert metadata.read_planner_metadata(gguf)["n_layer"] == 12
