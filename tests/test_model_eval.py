"""Model-evaluation tests use a fake transport and never start llama.cpp."""

from __future__ import annotations

import json

from kestrel.cli import evaluate, parser
from kestrel.evals.model_eval import (
    DEFAULT_MANIFEST,
    EvalCase,
    EvalManifest,
    evaluate_case_output,
    run_model_evaluation,
)


def test_default_manifest_covers_qwen_modes_and_stability_cases():
    ids = {case.case_id for case in DEFAULT_MANIFEST.cases}
    assert {case.mode for case in DEFAULT_MANIFEST.cases} == {"off", "on"}
    assert {"arithmetic_vat", "strict_json", "python_odd_squares", "thinking_on_reasoning"} <= ids
    assert any(case.category == "stability" for case in DEFAULT_MANIFEST.cases)


def test_case_requests_are_deterministic_and_disable_thinking_explicitly():
    case = next(case for case in DEFAULT_MANIFEST.cases if case.case_id == "thinking_on_reasoning")
    request = case.request("qwen", 99)
    assert request["seed"] == 99
    assert request["temperature"] == 0
    assert request["chat_template_kwargs"] == {"enable_thinking": True}
    assert request == case.request("qwen", 99)


def test_output_validators_reject_malformed_json_and_repetition():
    json_case = EvalCase("json", "structured", "return json", validator="strict_json", expected={"ok": True})
    assert evaluate_case_output(json_case, '{"ok":true}')["passed"]
    assert not evaluate_case_output(json_case, "```json\n{bad}\n```")["passed"]
    exact = EvalCase("exact", "instruction", "answer", validator="exact_token", expected="blue")
    repeated = "blue " * 30
    assert not evaluate_case_output(exact, repeated)["passed"]


def test_semantic_validators_reject_keyword_and_arbitrary_list_false_positives():
    code = EvalCase("code", "code", "expression", validator="code_result", expected=84)
    assert evaluate_case_output(code, "sum(x*x for x in range(8) if x % 2 == 1)")["passed"]
    assert evaluate_case_output(code, "sum(x**2 for x in range(1, 8, 2))")["passed"]
    assert not evaluate_case_output(code, "sum(range(8)) # odd %")["passed"]
    colors = EvalCase(
        "colors",
        "stability",
        "colors",
        validator="three_unique_items",
        expected=("red", "green", "blue"),
    )
    assert evaluate_case_output(colors, "blue,red,green")["passed"]
    assert not evaluate_case_output(colors, "cat,dog,bird")["passed"]
    unknown = EvalCase("unknown", "test", "test", validator="typo")
    assert evaluate_case_output(unknown, "anything")["reason"] == "unknown_validator:typo"


def test_run_model_evaluation_with_fake_openai_transport(tmp_path):
    calls = []
    outputs = iter(("240", "blue|green|red"))
    manifest = EvalManifest(
        "fixture",
        "1",
        (
            EvalCase("a", "arithmetic", "a", expected="240", validator="exact_token"),
            EvalCase("b", "instruction", "b", expected="blue|green|red", validator="exact_token", mode="off"),
        ),
    )

    def fake_request(url, payload, timeout):
        calls.append((url, payload, timeout))
        if url.endswith("/health"):
            return {"status": "ok"}
        if url.endswith("/v1/models"):
            return {"data": [{"id": "fixture-model"}]}
        return {
            "choices": [{"message": {"role": "assistant", "content": next(outputs)}}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 2},
        }

    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"fixture")
    report = run_model_evaluation(
        "http://fake:8080/",
        manifest=manifest,
        artifact=artifact,
        request_json=fake_request,
    )
    assert report["status"] == "passed"
    assert report["capacity_gate"]["passed"] is True
    assert report["quality_gate"]["passed"] is True
    assert report["engine"]["model"] == "fixture-model"
    assert report["artifact"]["size_bytes"] == 7
    assert len([call for call in calls if call[1] is not None]) == 2
    assert all(call[1]["seed"] == 42 for call in calls if call[1] is not None)


def test_run_model_evaluation_returns_machine_report_on_server_failure():
    def broken(_url, _payload, _timeout):
        raise RuntimeError("offline")

    report = run_model_evaluation(request_json=broken)
    assert report["status"] == "failed"
    assert report["capacity_gate"]["passed"] is False
    assert report["cases"] == []
    assert "offline" in report["error"]


def test_partial_manifest_and_truncation_cannot_pass_full_quality_gate():
    manifest = EvalManifest(
        "fixture",
        "1",
        (
            EvalCase("a", "arithmetic", "a", expected="1", validator="exact_token"),
            EvalCase("b", "arithmetic", "b", expected="2", validator="exact_token"),
        ),
    )

    def fake_request(url, payload, _timeout):
        if url.endswith("/health"):
            return {"status": "ok"}
        if url.endswith("/v1/models"):
            return {"data": [{"id": "fixture"}]}
        return {"choices": [{"finish_reason": "stop", "message": {"content": "1"}}]}

    partial = run_model_evaluation(model="fixture", manifest=manifest, max_cases=1, request_json=fake_request)
    assert partial["capacity_gate"]["passed"] is True
    assert partial["quality_gate"]["selected_cases_passed"] is True
    assert partial["quality_gate"]["passed"] is False
    assert partial["selection"]["complete"] is False

    def truncated_request(url, payload, _timeout):
        if url.endswith("/health"):
            return {"status": "ok"}
        if url.endswith("/v1/models"):
            return {"data": [{"id": "fixture"}]}
        return {"choices": [{"finish_reason": "length", "message": {"content": "1"}}]}

    truncated = run_model_evaluation(
        model="fixture",
        manifest=EvalManifest("one", "1", (manifest.cases[0],)),
        request_json=truncated_request,
    )
    assert truncated["cases"][0]["evaluation"]["reason"] == "truncated_at_token_limit"
    assert truncated["quality_gate"]["passed"] is False


def test_health_and_requested_model_must_match_capacity_contract():
    def fake_request(url, payload, _timeout):
        if url.endswith("/health"):
            return {"status": "loading"}
        if url.endswith("/v1/models"):
            return {"data": [{"id": "actual"}]}
        return {"choices": [{"finish_reason": "stop", "message": {"content": "1"}}]}

    manifest = EvalManifest("one", "1", (EvalCase("a", "test", "a", expected="1", validator="exact_token"),))
    report = run_model_evaluation(model="wrong", manifest=manifest, request_json=fake_request)
    checks = {item["name"]: item["passed"] for item in report["capacity_gate"]["checks"]}
    assert checks["health"] is False
    assert checks["requested_model"] is False
    assert report["status"] == "failed"


def test_evaluate_cli_parser_and_json_contract(monkeypatch, capsys):
    seen = {}

    def fake_run(**kwargs):
        seen.update(kwargs)
        return {"schema_version": 1, "status": "passed", "cases": []}

    monkeypatch.setattr(evaluate, "run_model_evaluation", fake_run)
    args = parser.build_parser().parse_args(
        ["model-test", "qwen", "--endpoint", "http://fake", "--seed", "7", "--json"]
    )
    assert evaluate.cmd_evaluate(args) == 0
    assert seen["model"] == "qwen"
    assert seen["seed"] == 7
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out)["status"] == "passed"


def test_evaluate_cli_returns_failure_exit_code(monkeypatch, capsys):
    monkeypatch.setattr(evaluate, "run_model_evaluation", lambda **_kwargs: {"status": "failed"})
    args = parser.build_parser().parse_args(["evaluate", "--json"])
    assert evaluate.cmd_evaluate(args) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
