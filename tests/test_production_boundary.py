import tempfile
import unittest
import zipfile
from pathlib import Path

import kestrel
from kestrel.core.pipeline import InferencePipeline
from scripts.check_wheel import REQUIRED, inspect_wheel


class ProductionBoundaryTests(unittest.TestCase):
    def test_alpha_components_are_not_public_exports(self):
        for name in (
            "AdaptiveController",
            "MultiTierCache",
            "NVFP4Loader",
            "PredictiveExpertCache",
            "RouterScanner",
            "SpecMode",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(kestrel, name))

    def test_pipeline_does_not_enable_mtp_implicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.gguf"
            model.write_bytes(b"GGUF")
            pipeline = InferencePipeline(gguf_path=str(model))
            self.addCleanup(pipeline.close)
            self.assertEqual(pipeline.backend.spec_type, "none")

    def test_retired_cache_options_fail_instead_of_being_ignored(self):
        with self.assertRaises(TypeError):
            InferencePipeline(gguf_path="/model.gguf", l1_cache_gb=4)

    def test_wheel_check_rejects_model_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "kestrel.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                for name in REQUIRED:
                    archive.writestr(name, "")
                archive.writestr("kestrel/model.gguf", "weights")
            findings = inspect_wheel(wheel)
        self.assertTrue(any("model.gguf" in finding for finding in findings))


if __name__ == "__main__":
    unittest.main()
