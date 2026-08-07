import os
import stat
from pathlib import Path

import pytest

from kestrel.backends.llama_cpp import (
    LlamaCppBackend,
    LlamaCppCapabilities,
    RunMetrics,
    _candidate_dirs,
    _find_binary,
    default_llama_cpp_dir,
    resolve_llama_binary,
)


def _make_bin(directory: str, name: str) -> str:
    path = os.path.join(directory, name)
    os.makedirs(directory, exist_ok=True)
    with open(path, "w") as handle:
        handle.write("#!/bin/sh\n")
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _make_dir() -> str:
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    tmp.joinpath("build", "bin").mkdir(parents=True, exist_ok=True)
    return str(tmp)


def test_find_binary_prefers_build_bin(tmp_path):
    root = str(tmp_path)
    build_bin = os.path.join(root, "build", "bin")
    os.makedirs(build_bin, exist_ok=True)
    _make_bin(build_bin, "llama-cli")
    root_file = _make_bin(root, "llama-cli")
    assert _find_binary(root, "llama-cli") == os.path.join(build_bin, "llama-cli")
    assert os.path.dirname(_find_binary(root, "llama-cli")) != os.path.dirname(root_file)


def test_find_binary_nonexecutable_ignored(tmp_path):
    root = str(tmp_path)
    path = os.path.join(root, "main")
    os.makedirs(root, exist_ok=True)
    with open(path, "w") as handle:
        handle.write("#!/bin/sh\n")
    assert _find_binary(root, "main") is None


def test_find_binary_missing_returns_none(tmp_path):
    assert _find_binary(str(tmp_path), "nope") is None


def test_candidate_dirs_override_first_then_dedup():
    os.environ["KESTREL_LLAMA_CPP_DIR"] = "/custom"
    try:
        dirs = _candidate_dirs(("/a", "/b"))
        assert dirs[0] == "/custom"
        assert dirs[1:] == ["/a", "/b"]
    finally:
        os.environ.pop("KESTREL_LLAMA_CPP_DIR", None)


def test_candidate_dirs_no_duplicates():
    os.environ["KESTREL_LLAMA_CPP_DIR"] = "/a"
    try:
        assert _candidate_dirs(("/a", "/b")) == ["/a", "/b"]
    finally:
        os.environ.pop("KESTREL_LLAMA_CPP_DIR", None)


def test_default_dir_wins_over_fallback():
    dirs = (_make_dir(), _make_dir(), _make_dir())
    _make_bin(os.path.join(dirs[1], "build", "bin"), "llama-server")
    assert default_llama_cpp_dir(dirs) == dirs[1]


def test_default_dir_missing_binary_returns_fallback():
    dirs = (_make_dir(),)
    assert default_llama_cpp_dir(dirs) == os.path.expanduser("~/llama.cpp")


def test_default_dir_override_wins():
    os.environ["KESTREL_LLAMA_CPP_DIR"] = "/override"
    try:
        assert default_llama_cpp_dir((_make_dir(),)) == "/override"
    finally:
        os.environ.pop("KESTREL_LLAMA_CPP_DIR", None)


def test_resolve_binary_returns_matching_executable():
    dirs = (_make_dir(), _make_dir())
    expected = _make_bin(os.path.join(dirs[0], "build"), "llama-cli")
    assert resolve_llama_binary("llama-cli", dirs) == expected


def test_resolve_binary_missing_returns_none():
    assert resolve_llama_binary("llama-cli", (_make_dir(),)) is None


def test_capabilities_spec_types():
    caps = LlamaCppCapabilities(help_text="--spec-type none,q1_0,q4_0,auto\n--fit on\n")
    assert caps.supports("--fit on")
    assert caps.spec_types == {"none", "q1_0", "q4_0", "auto"}


def test_capabilities_no_spec_type_key():
    caps = LlamaCppCapabilities(help_text="--fit on\n")
    assert caps.spec_types == set()
    assert caps.supports("--fit on")
    assert not caps.supports("--cpu-moe")


def test_resolved_spec_type_none_maps_to_none():
    backend = LlamaCppBackend("/model.gguf")
    caps = LlamaCppCapabilities(help_text="--spec-type none,q1_0")
    for value in ("", "none", None):
        backend.spec_type = value
        assert backend._resolved_spec_type(caps) is None


def test_resolved_spec_type_mtp_maps_to_draft_mtp():
    backend = LlamaCppBackend("/model.gguf")
    backend.spec_type = "mtp"
    caps = LlamaCppCapabilities(help_text="--spec-type none,draft-mtp")
    assert backend._resolved_spec_type(caps) == "draft-mtp"


def test_resolved_spec_type_unsupported_returns_none():
    backend = LlamaCppBackend("/model.gguf")
    backend.spec_type = "q1_0"
    caps = LlamaCppCapabilities(help_text="--spec-type none")
    assert backend._resolved_spec_type(caps) is None


def test_parse_metrics_populates_fields():
    stderr = """
ggml_sync: llama_get_logits done
llama_perf_context_print:        load time =   647.63 ms
llama_perf_context_print: prompt eval time =    1234.5 ms /     34 tokens (  27.52 tokens per second)
llama_perf_context_print:        eval time =    4567.8 ms /    128 runs   (  28.03 tokens per second)
"""
    metrics = LlamaCppBackend._parse_metrics(stderr, elapsed=9.0, returncode=0)
    assert metrics.elapsed_seconds == 9.0
    assert metrics.prompt_tokens == 34
    assert metrics.output_tokens == 128
    assert metrics.prompt_tokens_per_second == pytest.approx(27.52)
    assert metrics.output_tokens_per_second == pytest.approx(28.03)
    assert metrics.returncode == 0


def test_parse_metrics_no_match_defaults():
    metrics = LlamaCppBackend._parse_metrics("nothing useful here", elapsed=1.0, returncode=1)
    assert metrics.prompt_tokens is None
    assert metrics.output_tokens is None
    assert metrics.returncode == 1


def test_run_metrics_as_dict():
    metrics = RunMetrics(elapsed_seconds=2.5, prompt_tokens=10)
    data = metrics.as_dict()
    assert data["elapsed_seconds"] == 2.5
    assert data["prompt_tokens"] == 10


def test_backend_binary_not_found_raises(monkeypatch):
    backend = LlamaCppBackend("/model.gguf")
    monkeypatch.setattr(LlamaCppBackend, "_find_bin", lambda self, name: None)
    with pytest.raises(RuntimeError):
        _ = backend.binary


def _backend_with_caps(caps: LlamaCppCapabilities, tmp_path: Path) -> LlamaCppBackend:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF" + b"\x00" * 8)
    backend = LlamaCppBackend(str(model))
    backend._capabilities = caps
    return backend


def test_base_cmd_direct_io_overrides_mmap(tmp_path):
    backend = _backend_with_caps(
        LlamaCppCapabilities(help_text="--direct-io\n--no-mmap\n--mmap\n"), tmp_path
    )
    backend.direct_io = True
    cmd = backend._base_cmd()
    assert "--direct-io" in cmd
    assert "--no-mmap" in cmd
    assert "--mmap" not in cmd


def test_base_cmd_mmap_default_without_direct_io(tmp_path):
    backend = _backend_with_caps(
        LlamaCppCapabilities(help_text="--direct-io\n--no-mmap\n--mmap\n"), tmp_path
    )
    assert not backend.direct_io
    cmd = backend._base_cmd()
    assert "--mmap" in cmd
    assert "--direct-io" not in cmd


def test_base_cmd_direct_io_unsupported_falls_back_to_mmap(tmp_path):
    backend = _backend_with_caps(
        LlamaCppCapabilities(help_text="--no-mmap\n--mmap\n"), tmp_path
    )
    backend.direct_io = True
    cmd = backend._base_cmd()
    assert "--mmap" in cmd
    assert "--direct-io" not in cmd


def _backend_with_engine_bins(caps, tmp_path, server_caps=None):
    """Backend with fake llama-cli/llama-server binaries and given caps."""
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF" + b"\x00" * 8)
    build_bin = tmp_path / "build" / "bin"
    build_bin.mkdir(parents=True, exist_ok=True)
    _make_bin(str(build_bin), "llama-cli")
    _make_bin(str(build_bin), "llama-server")
    backend = LlamaCppBackend(str(model), llama_cpp_dir=str(tmp_path))
    backend._capabilities = caps
    backend._server_capabilities = server_caps or caps
    return backend


_ENGINE_HELP = (
    "--fit\n--fit-target\n--mlock\n--tensor-split\n--mmap\n--no-mmap\n"
    "--cpu-moe\n--moe-cache\n--cache-type-k\n--cache-type-v\n--flash-attn\n"
    "--threads\n--threads-batch\n--spec-type none,draft-mtp\n--spec-draft-n-max\n"
    "--direct-io\n--embeddings\n"
)


def test_build_server_cmd_emits_fit_and_engine_flags(tmp_path):
    backend = _backend_with_engine_bins(
        LlamaCppCapabilities(help_text=_ENGINE_HELP), tmp_path
    )
    backend.fit = True
    backend.fit_target_mib = 1024
    backend.use_mlock = True
    backend.tensor_split = "25,75"
    backend.use_mmap = True
    backend.extra_args = ["--temp", "0.4"]
    cmd = backend.build_server_cmd(host="0.0.0.0", port=9000, alias="m", embeddings=True)
    for flag in (
        "--fit", "on", "--fit-target", "1024", "--mlock",
        "--tensor-split", "25,75", "--embeddings",
        "--host", "0.0.0.0", "--port", "9000", "--alias", "m",
        "--temp", "0.4",
    ):
        assert flag in cmd


def test_build_server_cmd_embeddings_capability_gated(tmp_path):
    caps = LlamaCppCapabilities(help_text="--mmap\n")
    backend = _backend_with_engine_bins(caps, tmp_path)
    cmd = backend.build_server_cmd(embeddings=True)
    assert "--embeddings" not in cmd


def test_server_and_interactive_share_engine_flags(tmp_path):
    """The server path must emit the same engine block as interactive runs
    (regression: --fit was previously dropped, letting serve OOM)."""
    backend = _backend_with_engine_bins(
        LlamaCppCapabilities(help_text=_ENGINE_HELP), tmp_path
    )
    backend.fit = True
    backend.fit_target_mib = 1024
    backend.use_mlock = True
    backend.tensor_split = "25,75"
    backend.use_mmap = True
    base = backend._base_cmd()
    server = backend.build_server_cmd()
    for flag in (
        "--fit", "on", "--fit-target", "1024", "--mlock",
        "--tensor-split", "25,75", "--flash-attn", "auto",
    ):
        assert flag in base
        assert flag in server


def test_build_server_cmd_uses_server_binary_and_address(tmp_path):
    backend = _backend_with_engine_bins(
        LlamaCppCapabilities(help_text=_ENGINE_HELP), tmp_path
    )
    cmd = backend.build_server_cmd(host="0.0.0.0", port=9999)
    assert os.path.basename(cmd[0]) == "llama-server"
    assert cmd[1:3] == ["-m", backend.model_path]
    assert "--host" in cmd and "0.0.0.0" in cmd
    assert "--port" in cmd and "9999" in cmd
