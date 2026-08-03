import io
import unittest
from contextlib import redirect_stderr
from types import SimpleNamespace
from unittest import mock

from scripts import full_model_suite


def make_args(**overrides):
    base = dict(
        server="/llama-server",
        model="/model.gguf",
        gpu_layers="12",
        fit="on",
        fit_target=1473,
        ctx_size=2048,
        parallel=1,
        batch_size=256,
        ubatch_size=64,
        threads=14,
        kv_cache_type="q8_0",
        moe_cache="auto",
        expert_used_count=None,
        port=8091,
        mtp_tokens=3,
        mtp_gpu_layers="0",
        draft_model=None,
        draft_hf=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class FullModelSuiteScoringTests(unittest.TestCase):
    def test_exact_match_ignores_empty_thinking_envelope(self):
        case = {
            "patterns": [r"^\s*6\s*$"],
        }

        self.assertTrue(
            full_model_suite.score_case(
                case,
                "<think>\n\n</think>\n\n6",
            )
        )

    def test_answer_text_does_not_strip_nonleading_content(self):
        content = "prefix <think>hidden</think> answer"

        self.assertEqual(full_model_suite.answer_text(content), content)

    def test_server_command_gives_one_slot_full_context_and_tuned_threads(self):
        command = full_model_suite.server_command(make_args(), "baseline")

        self.assertEqual(command[command.index("-np") + 1], "1")
        self.assertEqual(command[command.index("-t") + 1], "14")
        self.assertEqual(command[command.index("-tb") + 1], "14")

    def test_stability_assessment_accepts_varied_factual_answer(self):
        content = (
            "Blue light has a shorter wavelength than red light. In Earth's atmosphere, "
            "Rayleigh scatter is stronger for those shorter wavelengths, so sunlight is "
            "redirected across the sky. Our eyes perceive much of that scattered visible "
            "radiation as blue. The effect depends on the path through the atmosphere, "
            "which is why sunrise and sunset can instead look orange or red when light "
            "travels through more air and much of the blue component is scattered away."
        )
        result = full_model_suite.assess_stability(content)
        self.assertTrue(result["heuristic_passed"])
        self.assertTrue(result["manual_coherence_review_required"])

    def test_stability_assessment_rejects_repetition_collapse(self):
        result = full_model_suite.assess_stability("blue sky repeats " * 60)
        self.assertFalse(result["no_repetition_collapse"])
        self.assertFalse(result["heuristic_passed"])

    def test_release_summary_requires_64_tokens_speed_and_quality(self):
        result = {
            "performance": [
                {
                    "id": "decode_64",
                    "usage": {"completion_tokens": 64},
                    "timings": {
                        "prompt_per_second": 100,
                        "predicted_per_second": 12.5,
                    },
                }
            ],
            "accuracy_summary": {"passed": 7, "total": 8},
            "stability": {"assessment": {"heuristic_passed": True}},
        }
        summary = full_model_suite.evaluate_release_gates(result)
        self.assertTrue(summary["automated_gates_passed"])
        self.assertTrue(summary["manual_coherence_review_required"])

    def test_release_summary_rejects_short_fast_sample(self):
        result = {
            "performance": [
                {
                    "id": "decode_64",
                    "usage": {"completion_tokens": 32},
                    "timings": {"predicted_per_second": 100},
                }
            ],
            "accuracy_summary": {"passed": 8},
            "stability": {"assessment": {"heuristic_passed": True}},
        }
        self.assertFalse(
            full_model_suite.evaluate_release_gates(result)["speed_floor_passed"]
        )

    def test_resource_summary_records_thermal_peaks(self):
        monitor = full_model_suite.ResourceMonitor(SimpleNamespace(pid=1))
        monitor.samples = [
            full_model_suite.ResourceSample(1, 100, 2, 500, 70, 60.5),
            full_model_suite.ResourceSample(2, 120, 3, 600, 75, 65.0),
        ]
        summary = monitor.summary()
        self.assertEqual(summary["peak_gpu_temperature_c"], 75)
        self.assertEqual(summary["peak_gpu_power_w"], 65.0)


class ServerCapabilitiesTests(unittest.TestCase):
    def test_default_server_uses_native_build_when_found(self):
        with mock.patch(
            "scripts.full_model_suite.resolve_llama_binary",
            return_value="/opt/native/bin/llama-server",
        ):
            self.assertEqual(
                full_model_suite.default_server_binary(),
                "/opt/native/bin/llama-server",
            )

    def test_default_server_falls_back_to_stock_path(self):
        with mock.patch(
            "scripts.full_model_suite.resolve_llama_binary",
            return_value=None,
        ):
            self.assertTrue(
                full_model_suite.default_server_binary().endswith("llama-server")
            )

    def test_parse_server_help_extracts_flags_and_spec_types(self):
        help_text = (
            "--mmap\n"
            "--fit [on|off]\n"
            "--fit-target MiB\n"
            "--cpu-moe\n"
            "--cache-type-k TYPE\n"
            "--flash-attn [on|off|auto]\n"
            "--spec-type none,draft-mtp,draft-simple\n"
        )
        caps = full_model_suite.parse_server_help(help_text)
        self.assertTrue(caps.supports("--fit"))
        self.assertTrue(caps.supports("--fit-target"))
        self.assertFalse(caps.supports("--no-warmup"))
        self.assertEqual(
            caps.spec_types, {"none", "draft-mtp", "draft-simple"}
        )

    def test_unsupported_flags_are_omitted_and_reported(self):
        caps = full_model_suite.ServerCapabilities(
            flags=frozenset({"--threads", "--spec-type"}),
            spec_types=frozenset({"none"}),
        )
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            command = full_model_suite.server_command(make_args(), "baseline", caps)
        for flag in (
            "--fit",
            "--fit-target",
            "--cpu-moe",
            "--moe-cache",
            "--mmap",
            "--no-warmup",
            "--cache-type-k",
            "--metrics",
            "--reasoning",
        ):
            self.assertNotIn(flag, command)
        self.assertNotIn("-tb", command)
        self.assertIn("-t", command)
        self.assertIn("WARNING", stderr.getvalue())

    def test_mtp_mode_requires_spec_type_support(self):
        caps = full_model_suite.ServerCapabilities(
            flags=frozenset(),
            spec_types=frozenset(),
        )
        with self.assertRaises(ValueError):
            full_model_suite.server_command(make_args(), "mtp", caps)

    def test_mtp_mode_requires_advertised_spec_type(self):
        caps = full_model_suite.ServerCapabilities(
            flags=frozenset({"--spec-type"}),
            spec_types=frozenset({"none"}),
        )
        with self.assertRaises(ValueError):
            full_model_suite.server_command(make_args(), "mtp", caps)

    def test_draft_mode_requires_draft_model_or_hf(self):
        with self.assertRaises(ValueError):
            full_model_suite.server_command(make_args(), "draft")


if __name__ == "__main__":
    unittest.main()
