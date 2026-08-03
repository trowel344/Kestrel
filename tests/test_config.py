import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class ConfigTests(unittest.TestCase):
    def test_round_trip_keeps_defaults_without_secrets(self):
        from kestrel.config import KestrelConfig, load_config, save_config

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            original = KestrelConfig(
                default_model="/models/qwen.gguf",
                llama_cpp_dir="/opt/llama.cpp",
                kimi_base_url="https://api.moonshot.ai/v1",
                kimi_model="kimi-k3",
            )
            save_config(original, path)
            self.assertEqual(load_config(path), original)
            self.assertNotIn("api_key", path.read_text().lower())

    def test_xdg_config_location_is_respected(self):
        from kestrel.config import config_path

        with patch.dict(os.environ, {"XDG_CONFIG_HOME": "/tmp/kestrel-config-test"}, clear=False):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("KESTREL_CONFIG", None)
                self.assertEqual(
                    config_path(), Path("/tmp/kestrel-config-test/kestrel/config.toml")
                )

    def test_invalid_toml_has_actionable_error(self):
        from kestrel.config import load_config

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.toml"
            path.write_text("[local\n")
            with self.assertRaisesRegex(ValueError, "invalid Kestrel config"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
