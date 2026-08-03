import argparse
import os
import re
import sys
import tempfile
import unittest
from importlib import metadata as _metadata
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

from kestrel.cli import (
    _context_size_arg,
    _cpu_moe_thread_sweep,
    _kestrel_version,
    _memory_snapshot,
    _run_with_oom_retries,
    _safetensors_size,
    _select_context_size,
    _summarize_benchmark_rows,
    detect_model,
)


class OomRetryTests(unittest.TestCase):
    def test_custom_environment_survives_oom_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "fake-llama"
            log = Path(directory) / "sidecar"
            script.write_text(
                "#!/bin/sh\n"
                f"echo \"$LLAMA_MOE_HOT_GGUF\" >> '{log}'\n"
                "exit 0\n"
            )
            script.chmod(0o755)
            env = os.environ.copy()
            env["LLAMA_MOE_HOT_GGUF"] = "/models/hot.gguf"

            result = _run_with_oom_retries(
                [str(script)], max_retries=2, env=env
            )

            self.assertEqual(result, 0)
            self.assertEqual(log.read_text().splitlines(), ["/models/hot.gguf"])

    def test_startup_oom_reduces_micro_batch_until_launch_succeeds(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "fake-llama"
            log = Path(directory) / "attempts"
            script.write_text(
                "#!/bin/sh\n"
                "while [ \"$1\" != \"\" ]; do\n"
                "  if [ \"$1\" = \"-ub\" ]; then shift; ub=\"$1\"; fi\n"
                "  shift\n"
                "done\n"
                f"echo \"$ub\" >> '{log}'\n"
                "if [ \"$ub\" -gt 16 ]; then\n"
                "  echo 'CUDA error: out of memory' >&2\n"
                "  exit 1\n"
                "fi\n"
                "exit 0\n"
            )
            script.chmod(0o755)
            result = _run_with_oom_retries(
                [str(script), "-ub", "64", "--fit-target", "1000"],
                max_retries=2,
            )
            self.assertEqual(result, 0)
            self.assertEqual(log.read_text().splitlines(), ["64", "32", "16"])

    def test_oom_retry_tolerates_command_without_ub_or_fit_target(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "fake-llama"
            script.write_text(
                "#!/bin/sh\n"
                "echo 'CUDA error: out of memory' >&2\n"
                "exit 1\n"
            )
            script.chmod(0o755)
            result = _run_with_oom_retries(
                [str(script)],
                max_retries=2,
            )
            self.assertEqual(result, 1)

    def test_oom_retry_without_fit_target_still_halves_micro_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "fake-llama"
            log = Path(directory) / "attempts"
            script.write_text(
                "#!/bin/sh\n"
                "while [ \"$1\" != \"\" ]; do\n"
                "  if [ \"$1\" = \"-ub\" ]; then shift; ub=\"$1\"; fi\n"
                "  shift\n"
                "done\n"
                f"echo \"$ub\" >> '{log}'\n"
                "if [ \"$ub\" -gt 16 ]; then\n"
                "  echo 'CUDA error: out of memory' >&2\n"
                "  exit 1\n"
                "fi\n"
                "exit 0\n"
            )
            script.chmod(0o755)
            result = _run_with_oom_retries(
                [str(script), "-ub", "64"],
                max_retries=2,
            )
            self.assertEqual(result, 0)
            self.assertEqual(log.read_text().splitlines(), ["64", "32", "16"])


class SafetensorsSizeTests(unittest.TestCase):
    def test_sums_unique_shards_referenced_by_weight_map(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "model-00001-of-00002.safetensors"
            second = root / "model-00002-of-00002.safetensors"
            first.write_bytes(b"\x00" * 2048)
            second.write_bytes(b"\x00" * 1024)
            converter = SimpleNamespace(
                model_dir=directory,
                wm={
                    "a.weight": "model-00001-of-00002.safetensors",
                    "b.weight": "model-00001-of-00002.safetensors",
                    "c.weight": "model-00002-of-00002.safetensors",
                    "missing.weight": "does-not-exist.safetensors",
                },
            )
            self.assertEqual(_safetensors_size(converter), 3072)


class ModelAliasTests(unittest.TestCase):
    def test_qwen_alias_uses_configured_gguf(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "qwen.gguf"
            model.touch()
            with mock.patch.dict(
                "os.environ", {"KESTREL_QWEN35_122B_GGUF": str(model)}
            ):
                detected = detect_model("Qwen3.5:122B-10A")
            self.assertEqual(detected["type"], "gguf")
            self.assertEqual(detected["path"], str(model.resolve()))

    def test_unknown_short_name_is_not_treated_as_hugging_face_id(self):
        self.assertIsNone(detect_model("not-a-real-alias"))

    def test_extensionless_ollama_blob_is_detected_by_magic(self):
        with tempfile.TemporaryDirectory() as directory:
            blob = Path(directory) / "sha256-deadbeef"
            blob.write_bytes(b"GGUF" + b"\x00" * 32)
            detected = detect_model(str(blob))
            self.assertEqual(detected["type"], "gguf")
            self.assertEqual(detected["path"], str(blob.resolve()))


class ContextSelectionTests(unittest.TestCase):
    def test_parser_accepts_auto_and_rejects_tiny_context(self):
        self.assertEqual(_context_size_arg("auto"), "auto")
        self.assertEqual(_context_size_arg("8192"), 8192)
        with self.assertRaises(argparse.ArgumentTypeError):
            _context_size_arg("128")

    def test_paging_model_gets_conservative_context(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "oversized.gguf"
            model.write_bytes(b"x" * 1024)
            with mock.patch("kestrel.cli.model_file_size", return_value=100 * 1024**3), mock.patch(
                "kestrel.cli._available_ram_mib", return_value=8192
            ):
                context, reason = _select_context_size(
                    {"path": str(model)},
                    {"vram_total_mb": 8192},
                )
        self.assertEqual(context, 2048)
        self.assertIn("paging", reason)

    def test_memory_snapshot_reports_nonnegative_swap(self):
        snapshot = _memory_snapshot()
        self.assertGreaterEqual(snapshot["ram_available_mib"], 0)
        self.assertGreaterEqual(snapshot["swap_used_mib"], 0)
        self.assertLessEqual(snapshot["swap_used_mib"], snapshot["swap_total_mib"])


class VersionResolutionTests(unittest.TestCase):
    def test_uses_package_version_when_available(self):
        from kestrel.cli import _kestrel_version

        self.assertNotEqual(_kestrel_version(), "unknown")

    def test_falls_back_to_installed_metadata_when_parent_is_namespace(self):
        namespace = ModuleType("kestrel")
        with mock.patch.dict(
            "sys.modules", {"kestrel": namespace}
        ), mock.patch.object(_metadata, "version", return_value="9.9.9"):
            self.assertEqual(_kestrel_version(), "9.9.9")

    def test_falls_back_to_pyproject_when_metadata_missing(self):
        namespace = ModuleType("kestrel")
        root = Path(__file__).resolve().parents[1]
        pyproject_version = None
        for line in (root / "pyproject.toml").read_text().splitlines():
            match = re.match(r'^version\s*=\s*["\']([^"\']+)["\']', line.strip())
            if match:
                pyproject_version = match.group(1)
                break
        self.assertIsNotNone(pyproject_version)
        with mock.patch.dict(
            "sys.modules", {"kestrel": namespace}
        ), mock.patch.object(
            _metadata, "version", side_effect=_metadata.PackageNotFoundError("kestrel")
        ):
            self.assertEqual(_kestrel_version(), pyproject_version)

    def test_falls_back_to_unknown_without_metadata_or_pyproject(self):
        namespace = ModuleType("kestrel")
        with mock.patch.dict(
            "sys.modules", {"kestrel": namespace}
        ), mock.patch.object(
            _metadata, "version", side_effect=_metadata.PackageNotFoundError("kestrel")
        ), mock.patch.object(
            Path, "read_text", side_effect=OSError("unreadable")
        ):
            self.assertEqual(_kestrel_version(), "unknown")


class BenchmarkSweepTests(unittest.TestCase):
    def test_auto_cpu_moe_sweep_covers_laptop_thread_range(self):
        self.assertEqual(_cpu_moe_thread_sweep(16), "8,10,12,14,16")

    def test_selects_prompt_row_matching_fastest_decode_threads(self):
        rows = [
            {"n_threads": 8, "n_prompt": 128, "n_gen": 0, "avg_ts": 100},
            {"n_threads": 8, "n_prompt": 0, "n_gen": 64, "avg_ts": 9},
            {"n_threads": 12, "n_prompt": 128, "n_gen": 0, "avg_ts": 90},
            {"n_threads": 12, "n_prompt": 0, "n_gen": 64, "avg_ts": 11},
        ]
        prompt, decode, sweep = _summarize_benchmark_rows(rows)
        self.assertEqual(decode["n_threads"], 12)
        self.assertEqual(prompt["n_threads"], 12)
        self.assertEqual(sweep[-1]["decode_tokens_per_second"], 11)


if __name__ == "__main__":
    unittest.main()
