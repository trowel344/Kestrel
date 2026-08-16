from __future__ import annotations

import json

import pytest

from kestrel.agent_trajectory import AgentTrajectoryRecorder


def _records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_recorder_is_private_redacted_bounded_and_idempotent(tmp_path):
    target = tmp_path / "trace.jsonl"
    recorder = AgentTrajectoryRecorder(target, workspace=str(tmp_path))

    assert recorder.user("/v1/chat/completions", "Use api_key=super-secret-value")
    assert not recorder.user("/v1/chat/completions", "Use api_key=super-secret-value")
    recorder.observation(
        {
            "event_id": "tool-1",
            "signature": "bash:pytest",
            "kind": "verification",
            "tool": "bash",
            "outcome": "failure",
            "summary": "1 failed; password=hunter2",
            "command": "pytest -q",
            "raw_output": "MUST NEVER APPEAR",
        }
    )

    records = _records(target)
    rendered = json.dumps(records)
    assert target.stat().st_mode & 0o777 == 0o600
    assert "super-secret-value" not in rendered
    assert "hunter2" not in rendered
    assert "MUST NEVER APPEAR" not in rendered
    assert [record["record"] for record in records] == ["header", "user", "observation"]


def test_recorder_rejects_symlink_target(tmp_path):
    real = tmp_path / "real.jsonl"
    real.write_text("", encoding="utf-8")
    link = tmp_path / "trace.jsonl"
    link.symlink_to(real)

    with pytest.raises(ValueError, match="non-symlink"):
        AgentTrajectoryRecorder(link)


def test_recorder_restart_deduplicates_existing_ids(tmp_path):
    target = tmp_path / "trace.jsonl"
    first = AgentTrajectoryRecorder(target)
    first.user("/v1/chat/completions", "A durable requirement")

    restarted = AgentTrajectoryRecorder(target)

    assert not restarted.user("/v1/chat/completions", "A durable requirement")
