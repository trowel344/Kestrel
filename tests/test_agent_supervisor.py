from __future__ import annotations

import hashlib
import json

import pytest

pytest.importorskip("sunmap")

from sunmap import TaskCheckpoint

from kestrel.agent_supervisor import extract_tool_calls, protected_paths, reduce_request, required_commands


def test_chat_edit_updates_verified_workspace_digest(tmp_path):
    source = tmp_path / "module.py"
    source.write_text("value = 2\n", encoding="utf-8")
    payload = {
        "messages": [
            {"role": "user", "content": "Fix module.py"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-edit",
                        "function": {
                            "name": "edit",
                            "arguments": json.dumps({"path": "module.py", "edits": []}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-edit",
                "content": "Successfully replaced 1 block(s) in module.py.",
            },
        ]
    }
    checkpoint = TaskCheckpoint(objective="Fix module.py", status="active")

    assert reduce_request("/v1/chat/completions", payload, checkpoint, workspace=tmp_path)
    expected = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    assert checkpoint.changed_files == {"module.py": expected}
    assert checkpoint.progress_revision == 1


def test_unittest_result_becomes_structured_verification(tmp_path):
    payload = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-tests",
                        "function": {
                            "name": "bash",
                            "arguments": json.dumps({"command": "python -m unittest test_event_ledger -v"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-tests",
                "content": "Ran 10 tests in 0.005s\n\nOK\n",
            },
        ]
    }
    checkpoint = TaskCheckpoint(objective="Fix ledger", status="active")

    reduce_request("/v1/chat/completions", payload, checkpoint, workspace=tmp_path)

    assert checkpoint.verification.status == "passed"
    assert checkpoint.verification.passed == 10
    assert checkpoint.verification.failed == 0
    assert "10 passed" in checkpoint.verification.summary


def test_failed_pytest_is_not_mistaken_for_success(tmp_path):
    payload = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-tests",
                        "function": {
                            "name": "bash",
                            "arguments": {"command": "pytest -q"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-tests",
                "content": "8 passed, 2 failed in 0.20s",
            },
        ]
    }
    checkpoint = TaskCheckpoint(objective="Fix ledger", status="active")

    reduce_request("/v1/chat/completions", payload, checkpoint, workspace=tmp_path)

    assert checkpoint.verification.status == "failed"
    assert checkpoint.verification.passed == 8
    assert checkpoint.verification.failed == 2
    assert checkpoint.current_failures


def test_replayed_tool_result_is_idempotent(tmp_path):
    payload = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-read",
                        "function": {"name": "bash", "arguments": {"command": "cat module.py"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-read", "content": "source"},
        ]
    }
    checkpoint = TaskCheckpoint(objective="Inspect", status="active")

    assert reduce_request("/v1/chat/completions", payload, checkpoint, workspace=tmp_path)
    snapshot = checkpoint.to_dict()
    assert not reduce_request("/v1/chat/completions", payload, checkpoint, workspace=tmp_path)
    assert checkpoint.to_dict() == snapshot


def test_changed_file_is_marked_stale_on_later_request(tmp_path):
    source = tmp_path / "module.py"
    source.write_text("old\n", encoding="utf-8")
    checkpoint = TaskCheckpoint(objective="Fix", status="active")
    edit_payload = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-edit",
                        "function": {
                            "name": "edit",
                            "arguments": {"path": "module.py"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-edit",
                "content": "Successfully updated module.py",
            },
        ]
    }
    reduce_request("/v1/chat/completions", edit_payload, checkpoint, workspace=tmp_path)
    source.write_text("external change\n", encoding="utf-8")

    assert reduce_request("/v1/chat/completions", {"messages": []}, checkpoint, workspace=tmp_path)
    assert checkpoint.stale_files == ["module.py"]


def test_responses_and_anthropic_shapes_are_reduced():
    responses = {
        "input": [
            {
                "type": "function_call",
                "call_id": "response-call",
                "name": "bash",
                "arguments": json.dumps({"command": "pytest -q"}),
            },
            {
                "type": "function_call_output",
                "call_id": "response-call",
                "output": "12 passed in 0.1s",
            },
        ]
    }
    anthropic = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "anthropic-call",
                        "name": "bash",
                        "input": {"command": "pytest -q"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "anthropic-call",
                        "content": "13 passed in 0.1s",
                    }
                ],
            },
        ]
    }

    assert extract_tool_calls("/v1/responses", responses)[0].name == "bash"
    assert extract_tool_calls("/v1/messages", anthropic)[0].name == "bash"


def test_outside_workspace_path_is_not_hashed(tmp_path):
    outside = tmp_path.parent / "outside.py"
    outside.write_text("secret\n", encoding="utf-8")
    payload = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-edit",
                        "function": {
                            "name": "edit",
                            "arguments": {"path": str(outside)},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-edit",
                "content": "Successfully updated file",
            },
        ]
    }
    checkpoint = TaskCheckpoint(objective="Fix", status="active")

    reduce_request("/v1/chat/completions", payload, checkpoint, workspace=tmp_path)

    assert list(checkpoint.changed_files.values()) == [""]


def test_protected_test_mutation_is_flagged(tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    target = tests / "test_service.py"
    target.write_text("def test_service(): pass\n", encoding="utf-8")
    checkpoint = TaskCheckpoint(
        objective="Fix the service without changing tests",
        status="active",
        protected_files=protected_paths("Do not modify tests."),
    )
    payload = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "forbidden-edit",
                        "function": {
                            "name": "edit",
                            "arguments": {"path": "tests/test_service.py"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "forbidden-edit",
                "content": "Successfully updated tests/test_service.py",
            },
        ]
    }

    reduce_request("/v1/chat/completions", payload, checkpoint, workspace=tmp_path)

    assert "tests/**" in checkpoint.protected_files
    assert any("protected path" in warning for warning in checkpoint.warnings)


def test_protected_paths_accept_plain_filename_lists_and_storage_categories():
    result = protected_paths("Do not modify test_ledger.py, README.md, or any database/trace files.")

    assert "test_ledger.py" in result
    assert "README.md" in result
    assert "*.sqlite3" in result
    assert "*.jsonl" in result


def test_required_commands_extracts_only_explicit_exact_quoted_commands():
    result = required_commands(
        "Run the tests, then run exactly 'python -m pytest -q'. Execute exactly `git diff --check` before reporting."
    )

    assert result == ["python -m pytest -q", "git diff --check"]


def test_failed_test_command_without_counts_does_not_claim_zero_failures(tmp_path):
    checkpoint = TaskCheckpoint(objective="Repair", status="active")
    payload = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "bad-tests",
                        "function": {
                            "name": "bash",
                            "arguments": json.dumps({"command": "python -m pytest missing.py -q"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "bad-tests",
                "content": "ERROR: file or directory not found: missing.py",
            },
        ]
    }

    reduce_request("/v1/chat/completions", payload, checkpoint, workspace=tmp_path)

    assert checkpoint.verification.status == "failed"
    assert checkpoint.verification.failed is None
    assert "0 failed" not in checkpoint.verification.summary
