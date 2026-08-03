import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


class OllamaProviderTests(unittest.TestCase):
    def test_generation_exposes_prompt_and_decode_rates(self):
        from kestrel.providers.ollama import OllamaClient

        response = FakeResponse(
            {
                "response": "four",
                "thinking": "calculate",
                "prompt_eval_count": 20,
                "prompt_eval_duration": 100_000_000,
                "eval_count": 10,
                "eval_duration": 500_000_000,
                "total_duration": 700_000_000,
            }
        )
        client = OllamaClient(opener=lambda *_args, **_kwargs: response)
        result = client.generate("qwen", "2+2")
        self.assertEqual(result.prompt_tps, 200)
        self.assertEqual(result.decode_tps, 20)
        self.assertEqual(result.response, "four")
        self.assertEqual(result.thinking, "calculate")

    def test_cli_benchmark_returns_report_and_uses_thinking_output(self):
        from kestrel.cli import cmd_benchmark
        from kestrel.providers.ollama import OllamaGeneration

        generation = OllamaGeneration(
            response="",
            thinking="reasoned answer",
            prompt_tokens=20,
            prompt_duration_ns=100_000_000,
            generated_tokens=10,
            generation_duration_ns=500_000_000,
            total_duration_ns=700_000_000,
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "kestrel.providers.ollama.OllamaClient.generate",
            return_value=generation,
        ):
            output = Path(directory) / "benchmark.json"
            report = cmd_benchmark(
                SimpleNamespace(
                    model="ollama://qwen",
                    prompt_tokens=16,
                    generate_tokens=8,
                    repetitions=2,
                    ctx_size=2048,
                    output=str(output),
                    quiet=True,
                )
            )

        self.assertEqual(report["prompt_tokens_per_second"], 200)
        self.assertEqual(report["decode_tokens_per_second"], 20)
        self.assertEqual(report["sample_output"], "reasoned answer")
        self.assertTrue(report["release_speed_floor_passed"])


if __name__ == "__main__":
    unittest.main()
