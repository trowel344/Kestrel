import subprocess
import tempfile
import unittest
import json
from pathlib import Path
from unittest import mock


class ModelStoreTests(unittest.TestCase):
    def setUp(self):
        # The module-level ollama-list TTL cache must not leak between tests.
        import kestrel.model_store as ms

        ms._ollama_list_cache = None

    def test_resolves_local_ollama_blob(self):
        from kestrel.model_store import _ollama_model_roots, resolve_ollama_blob

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blob_dir = root / ".ollama" / "models" / "blobs"
            blob_dir.mkdir(parents=True)
            blob = blob_dir / "sha256-model"
            blob.write_bytes(b"GGUF")
            result = subprocess.CompletedProcess(
                [], 0, stdout=f"FROM {blob}\nPARAMETER temperature 1\n", stderr=""
            )
            with mock.patch(
                "kestrel.model_store._ollama_model_roots",
                return_value=[str(root / ".ollama" / "models")],
            ), mock.patch("kestrel.model_store._run", return_value=result):
                self.assertEqual(resolve_ollama_blob("qwen:latest"), blob.resolve())

    def test_rejects_ollama_blob_outside_managed_roots(self):
        from kestrel.model_store import resolve_ollama_blob

        with tempfile.TemporaryDirectory() as tmp:
            blob = Path(tmp) / "sha256-model"
            blob.write_bytes(b"GGUF")
            result = subprocess.CompletedProcess(
                [], 0, stdout=f"FROM {blob}\n", stderr=""
            )
            with mock.patch("kestrel.model_store._run", return_value=result), mock.patch(
                "kestrel.model_store._ollama_model_roots", return_value=[]
            ):
                self.assertIsNone(resolve_ollama_blob("qwen:latest"))

    def test_cloud_ollama_model_has_no_local_blob(self):
        from kestrel.model_store import resolve_ollama_blob

        result = subprocess.CompletedProcess(
            [], 0, stdout="FROM kimi-k3:cloud\n", stderr=""
        )
        with mock.patch("kestrel.model_store._run", return_value=result):
            self.assertIsNone(resolve_ollama_blob("kimi-k3:cloud"))

    def test_parses_ollama_list_without_local_resolution(self):
        from kestrel.model_store import list_ollama_models

        output = (
            "NAME        ID            SIZE      MODIFIED\n"
            "qwen:4b     abcdef123456  3.4 GB    2 days ago\n"
        )
        result = subprocess.CompletedProcess([], 0, stdout=output, stderr="")
        with mock.patch("kestrel.model_store._run", return_value=result):
            models = list_ollama_models()
        self.assertEqual(models[0].name, "qwen:4b")
        self.assertEqual(models[0].size, "3.4 GB")

    def test_resolved_ollama_models_keep_list_order(self):
        from kestrel.model_store import list_ollama_models

        output = (
            "NAME        ID            SIZE      MODIFIED\n"
            "first:1     a             1 GB      today\n"
            "second:1    b             2 GB      yesterday\n"
        )
        result = subprocess.CompletedProcess([], 0, stdout=output, stderr="")
        with mock.patch("kestrel.model_store._run", return_value=result), mock.patch(
            "kestrel.model_store.resolve_ollama_blob",
            side_effect=lambda name: Path("/models") / name,
        ):
            models = list_ollama_models(resolve_paths=True)

        self.assertEqual([item.name for item in models], ["first:1", "second:1"])
        self.assertEqual(models[1].local_path, Path("/models/second:1"))

    def test_huggingface_download_reports_deterministic_directory(self):
        from kestrel.model_store import pull_huggingface

        process = subprocess.CompletedProcess([], 0, stdout="[]", stderr="")
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "kestrel.model_store.shutil.which", return_value="/usr/bin/hf"
        ), mock.patch("kestrel.model_store._run", return_value=process) as run:
            result = pull_huggingface(
                "hf://owner/model",
                destination=Path(tmp) / "download",
                dry_run=True,
            )

        self.assertEqual(result.repository, "owner/model")
        self.assertEqual(result.stdout, "[]")
        self.assertIn("--dry-run", run.call_args.args[0])
        self.assertEqual(result.directory.name, "download")

    def test_huggingface_download_supports_split_model_glob(self):
        from kestrel.model_store import pull_huggingface

        process = subprocess.CompletedProcess([], 0, stdout="[]", stderr="")
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "kestrel.model_store.shutil.which", return_value="/usr/bin/hf"
        ), mock.patch("kestrel.model_store._run", return_value=process) as run:
            pull_huggingface(
                "owner/model",
                include="model-Q4_K_M-*.gguf",
                destination=Path(tmp),
                dry_run=True,
            )
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--include") + 1], "model-Q4_K_M-*.gguf")

    def test_huggingface_download_records_source_manifest(self):
        from kestrel.model_store import pull_huggingface

        process = subprocess.CompletedProcess([], 0, stdout="[]", stderr="")
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "kestrel.model_store.shutil.which", return_value="/usr/bin/hf"
        ), mock.patch("kestrel.model_store._run", return_value=process):
            target = Path(tmp) / "model"
            pull_huggingface(
                "owner/model",
                filename="model.gguf",
                revision="abc123",
                destination=target,
            )
            manifest = json.loads((target / ".kestrel-source.json").read_text())

        self.assertEqual(manifest["repository"], "owner/model")
        self.assertEqual(manifest["revision"], "abc123")
        self.assertEqual(manifest["file"], "model.gguf")

    def test_complete_split_gguf_selects_first_shard(self):
        from kestrel.model_store import choose_default_gguf

        paths = [
            Path("model-Q4_K_M-00002-of-00002.gguf"),
            Path("mmproj-F16.gguf"),
            Path("model-Q4_K_M-00001-of-00002.gguf"),
        ]
        self.assertEqual(
            choose_default_gguf(paths),
            Path("model-Q4_K_M-00001-of-00002.gguf"),
        )

    def test_incomplete_split_gguf_is_rejected(self):
        from kestrel.model_store import ModelStoreError, choose_default_gguf

        with self.assertRaises(ModelStoreError):
            choose_default_gguf([Path("model-00001-of-00002.gguf")])

    def test_huggingface_search_is_gguf_filtered_and_extracts_license(self):
        from kestrel.model_store import search_huggingface

        result = subprocess.CompletedProcess(
            [],
            0,
            stdout=(
                '[{"id":"owner/model-GGUF","downloads":12,"likes":3,'
                '"last_modified":"2026-01-01","tags":["gguf","license:apache-2.0"]}]'
            ),
            stderr="",
        )
        with mock.patch("kestrel.model_store.shutil.which", return_value="/usr/bin/hf"), mock.patch(
            "kestrel.model_store._run", return_value=result
        ) as run:
            rows = search_huggingface("model", limit=5)

        command = run.call_args.args[0]
        self.assertIn("gguf", command)
        self.assertEqual(rows[0]["license"], "apache-2.0")
        self.assertEqual(rows[0]["downloads"], 12)

    def test_huggingface_file_listing_keeps_integrity_metadata(self):
        from kestrel.model_store import list_huggingface_ggufs

        result = subprocess.CompletedProcess(
            [],
            0,
            stdout=(
                '[{"path":"README.md","size":1},'
                '{"path":"model-Q4_K_M.gguf","size":2048,'
                '"lfs":{"sha256":"abc"},"security":{"status":"safe"}}]'
            ),
            stderr="",
        )
        with mock.patch("kestrel.model_store.shutil.which", return_value="/usr/bin/hf"), mock.patch(
            "kestrel.model_store._run", return_value=result
        ):
            files = list_huggingface_ggufs("hf://owner/model")

        self.assertEqual(len(files), 1)
        self.assertEqual(files[0]["sha256"], "abc")
        self.assertEqual(files[0]["security_status"], "safe")


if __name__ == "__main__":
    unittest.main()
