import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from kestrel.backends.llama_cpp import (
    NATIVE_LLAMA_CPP_DIRS,
    LlamaCppBackend,
    LlamaCppCapabilities,
    default_llama_cpp_dir,
    resolve_llama_binary,
)


HELP = """
--mmap, --no-mmap
--fit [on|off]
--fit-target MiB
--cpu-moe
--cache-type-k TYPE
--cache-type-v TYPE
--flash-attn [on|off|auto]
--spec-draft-n-max N
--spec-type none,draft-mtp,ngram-cache
--single-turn
--simple-io
--perf
--threads N
--threads-batch N
"""


class LlamaCppBackendTests(unittest.TestCase):
    def make_backend(self):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        binary = root / "build" / "bin" / "llama-cli"
        binary.parent.mkdir(parents=True)
        binary.write_text("#!/bin/sh\nexit 0\n")
        binary.chmod(0o755)
        model = root / "model.gguf"
        model.write_bytes(b"GGUF")
        backend = LlamaCppBackend(
            str(model),
            n_gpu_layers="auto",
            spec_type="mtp",
            cpu_moe=True,
            fit_target_mib=1400,
            n_threads=14,
            llama_cpp_dir=str(root),
        )
        backend._capabilities = LlamaCppCapabilities(HELP, "test")
        return temporary, backend

    def test_mtp_alias_and_supported_memory_flags(self):
        temporary, backend = self.make_backend()
        self.addCleanup(temporary.cleanup)
        cmd = backend._build_cmd("hello", 10)
        self.assertIn("draft-mtp", cmd)
        self.assertIn("--cpu-moe", cmd)
        self.assertIn("--mmap", cmd)
        self.assertIn("--fit-target", cmd)
        self.assertEqual(cmd[cmd.index("--threads") + 1], "14")
        self.assertEqual(cmd[cmd.index("--threads-batch") + 1], "14")
        self.assertNotIn("--moe-hot-expert-k", cmd)

    def test_interactive_command_has_no_zero_token_limit(self):
        temporary, backend = self.make_backend()
        self.addCleanup(temporary.cleanup)
        cmd = backend._build_interactive_cmd()
        self.assertNotIn("-n", cmd)
        self.assertNotIn("-p", cmd)
        self.assertNotIn("--single-turn", cmd)

    def test_one_shot_command_cannot_enter_interactive_loop(self):
        temporary, backend = self.make_backend()
        self.addCleanup(temporary.cleanup)
        cmd = backend._build_cmd("hello", 10)
        self.assertIn("--single-turn", cmd)
        self.assertIn("--simple-io", cmd)
        self.assertIn("--perf", cmd)

    def test_unsupported_flags_are_omitted(self):
        temporary, backend = self.make_backend()
        self.addCleanup(temporary.cleanup)
        backend._capabilities = LlamaCppCapabilities("--mmap", "old")
        cmd = backend._build_cmd("hello", 10)
        self.assertNotIn("--spec-type", cmd)
        self.assertNotIn("--cpu-moe", cmd)
        self.assertNotIn("--cache-type-k", cmd)

    def test_missing_model_fails_before_process_launch(self):
        backend = LlamaCppBackend("/does/not/exist.gguf")
        backend._capabilities = LlamaCppCapabilities(HELP, "test")
        with self.assertRaises(FileNotFoundError):
            backend._build_cmd("", 0)

    def test_metrics_distinguish_prompt_and_decode_lines(self):
        stderr = """
llama_perf_context_print: prompt eval time = 10 ms / 5 tokens ( 500.0 tokens per second)
llama_perf_context_print: eval time = 100 ms / 10 runs ( 100.0 tokens per second)
"""
        metrics = LlamaCppBackend._parse_metrics(stderr, 0.2, 0)
        self.assertEqual(metrics.prompt_tokens, 5)
        self.assertEqual(metrics.output_tokens, 10)
        self.assertEqual(metrics.prompt_tokens_per_second, 500.0)
        self.assertEqual(metrics.output_tokens_per_second, 100.0)


def make_executable(path: Path):
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)


def make_build(root: Path, name: str, binary: str) -> Path:
    base = root / name
    (base / "build" / "bin").mkdir(parents=True)
    executable = base / "build" / "bin" / binary
    make_executable(executable)
    return executable


class BuildResolutionTests(unittest.TestCase):
    def test_default_dir_prefers_moe_cache_build_over_stock(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            make_build(tmp_path, "moe", "llama-server")
            make_build(tmp_path, "stock", "llama-server")
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertEqual(
                    default_llama_cpp_dir(
                        dirs=(str(tmp_path / "moe"), str(tmp_path / "stock"))
                    ),
                    str(tmp_path / "moe"),
                )

    def test_default_dir_falls_back_to_stock_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            make_build(tmp_path, "stock", "llama-cli")
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertEqual(
                    default_llama_cpp_dir(dirs=(str(tmp_path / "stock"),)),
                    str(tmp_path / "stock"),
                )

    def test_resolve_prefers_moe_cache_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            moe_server = make_build(tmp_path, "moe", "llama-server")
            make_build(tmp_path, "stock", "llama-server")
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertEqual(
                    resolve_llama_binary(
                        "llama-server",
                        dirs=(str(tmp_path / "moe"), str(tmp_path / "stock")),
                    ),
                    str(moe_server),
                )

    def test_backend_falls_back_to_stock_cli_when_native_build_lacks_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            moe = tmp_path / "moe"
            (moe / "build" / "bin").mkdir(parents=True)
            make_executable(moe / "build" / "bin" / "llama-server")
            stock_cli = make_build(tmp_path, "stock", "llama-cli")
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch(
                "kestrel.backends.llama_cpp.NATIVE_LLAMA_CPP_DIRS",
                (str(moe), str(tmp_path / "stock")),
            ):
                backend = LlamaCppBackend("/nope.gguf", llama_cpp_dir=str(moe))
                self.assertEqual(backend.binary, str(stock_cli))

    def test_server_binary_resolves_in_native_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            moe_server = make_build(tmp_path, "moe", "llama-server")
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch(
                "kestrel.backends.llama_cpp.NATIVE_LLAMA_CPP_DIRS",
                (str(tmp_path / "moe"),),
            ):
                backend = LlamaCppBackend("/nope.gguf", llama_cpp_dir=str(tmp_path / "moe"))
                self.assertEqual(backend.server_binary, str(moe_server))

    def test_server_binary_raises_when_no_native_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch(
                "kestrel.backends.llama_cpp.NATIVE_LLAMA_CPP_DIRS",
                (str(tmp_path / "empty"),),
            ):
                backend = LlamaCppBackend("/nope.gguf", llama_cpp_dir=str(tmp_path / "empty"))
                with self.assertRaises(RuntimeError):
                    backend.server_binary


class CapabilityCacheTests(unittest.TestCase):
    def test_cache_roundtrip_for_unchanged_binary(self):
        from kestrel.backends.llama_cpp import (
            _capability_cache_read,
            _capability_cache_write,
        )

        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "llama-cli"
            binary.write_bytes(b"#!/bin/sh\nexit 0\n")
            binary.chmod(0o755)
            cache_file = str(Path(tmp) / "cache.json")
            with mock.patch(
                "kestrel.backends.llama_cpp._capability_cache_path",
                return_value=cache_file,
            ):
                _capability_cache_write(str(binary), "--mmap", "b1234")
                self.assertEqual(
                    _capability_cache_read(str(binary)),
                    ("--mmap", "b1234"),
                )

    def test_cache_miss_after_binary_changes(self):
        from kestrel.backends.llama_cpp import (
            _capability_cache_read,
            _capability_cache_write,
        )

        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "llama-cli"
            binary.write_bytes(b"#!/bin/sh\nexit 0\n")
            binary.chmod(0o755)
            cache_file = str(Path(tmp) / "cache.json")
            with mock.patch(
                "kestrel.backends.llama_cpp._capability_cache_path",
                return_value=cache_file,
            ):
                _capability_cache_write(str(binary), "--mmap", "b1234")
                binary.write_bytes(b"#!/bin/sh\nexit 0\nnew build\n")
                self.assertIsNone(_capability_cache_read(str(binary)))

    def test_capabilities_skips_probe_when_cache_is_warm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "build" / "bin" / "llama-cli"
            binary.parent.mkdir(parents=True)
            binary.write_text("#!/bin/sh\nexit 0\n")
            binary.chmod(0o755)
            cache_file = str(root / "cache.json")
            with mock.patch(
                "kestrel.backends.llama_cpp._capability_cache_path",
                return_value=cache_file,
            ):
                backend = LlamaCppBackend(
                    "/nope.gguf",
                    llama_cpp_dir=str(root),
                )
                from kestrel.backends.llama_cpp import _capability_cache_write

                _capability_cache_write(str(binary), "--cpu-moe", "cached-build")
                with mock.patch(
                    "kestrel.backends.llama_cpp.subprocess.run",
                    side_effect=AssertionError("probe must be skipped"),
                ):
                    caps = backend.capabilities()
                self.assertEqual(caps.version, "cached-build")
                self.assertIn("--cpu-moe", caps.help_text)


if __name__ == "__main__":
    unittest.main()
