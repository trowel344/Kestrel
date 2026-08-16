"""Deterministic long-horizon state reduction for coding-agent tool loops.

The supervisor never persists raw tool output.  It pairs tool calls with their
reported results, extracts bounded operational facts, verifies changed file
digests inside the configured workspace, and updates Sun Map's task checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from sunmap import TaskCheckpoint, TaskObservation

_TEST_COMMAND = re.compile(
    r"(?:^|[;&|\s])(?:python(?:3)?\s+-m\s+(?:pytest|unittest)|pytest|cargo\s+test|"
    r"go\s+test|npm\s+(?:run\s+)?test|pnpm\s+(?:run\s+)?test|ctest)(?:\s|$)",
    re.IGNORECASE,
)
_PYTEST_PASSED = re.compile(r"\b(\d+)\s+passed\b", re.IGNORECASE)
_PYTEST_FAILED = re.compile(r"\b(\d+)\s+failed\b", re.IGNORECASE)
_UNITTEST_RAN = re.compile(r"\bRan\s+(\d+)\s+tests?\b", re.IGNORECASE)
_FAILURE_MARKERS = (
    "could not find",
    "traceback (most recent call last)",
    "command exited with code",
    "command failed",
)
_PROTECTED_PATH = re.compile(
    r"(?:do\s+not|don't|never)\s+(?:modify|change|edit)[^\n.]{0,100}`([^`]+)`",
    re.IGNORECASE,
)
_PROTECTED_CLAUSE = re.compile(
    r"(?:do\s+not|don't|never)\s+(?:modify|change|edit)\s+([^\n]{1,240}?)(?:\.(?=\s|$)|$)",
    re.IGNORECASE,
)
_PATH_SHAPED_TOKEN = re.compile(r"(?<![\w/.-])(?:[\w*.-]+/)*[\w*-]+\.(?:[A-Za-z0-9*_-]+)(?![\w/.-])")
_REQUIRED_COMMAND = re.compile(
    r"\b(?:run|use|execute)\s+exactly\s+(['\"`])([^\n'\"`]{1,500})\1",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]
    result: str
    result_is_error: bool = False


def _bounded_line(value: str, limit: int = 300) -> str:
    value = " ".join(value.replace("\x00", "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        text = content.get("text") or content.get("content")
        return _content_text(text)
    if isinstance(content, list):
        return "".join(_content_text(item) for item in content)
    return ""


def _arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"raw": _bounded_line(value, 500)}
        return parsed if isinstance(parsed, dict) else {"raw": _bounded_line(value, 500)}
    return {}


def _chat_tool_calls(payload: dict[str, Any]) -> list[_ToolCall]:
    pending: dict[str, tuple[str, dict[str, Any]]] = {}
    completed: list[_ToolCall] = []
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return completed
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            calls = message.get("tool_calls")
            if isinstance(calls, list):
                for call in calls:
                    if not isinstance(call, dict):
                        continue
                    call_id = str(call.get("id") or "")
                    function = call.get("function")
                    function = function if isinstance(function, dict) else call
                    name = str(function.get("name") or "")
                    if call_id and name:
                        pending[call_id] = (name, _arguments(function.get("arguments")))
        elif role == "tool":
            call_id = str(message.get("tool_call_id") or message.get("call_id") or "")
            if call_id in pending:
                name, arguments = pending[call_id]
                completed.append(
                    _ToolCall(
                        call_id,
                        name,
                        arguments,
                        _content_text(message.get("content")),
                        bool(message.get("is_error") or message.get("isError")),
                    )
                )
    return completed


def _responses_tool_calls(payload: dict[str, Any]) -> list[_ToolCall]:
    pending: dict[str, tuple[str, dict[str, Any]]] = {}
    outputs: dict[str, tuple[str, bool]] = {}
    items = payload.get("input")
    if not isinstance(items, list):
        return []
    for item in items:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "function_call":
            call_id = str(item.get("call_id") or item.get("id") or "")
            name = str(item.get("name") or "")
            if call_id and name:
                pending[call_id] = (name, _arguments(item.get("arguments")))
        elif item_type == "function_call_output":
            call_id = str(item.get("call_id") or "")
            outputs[call_id] = (
                _content_text(item.get("output")),
                bool(item.get("is_error") or item.get("isError")),
            )
    return [
        _ToolCall(call_id, name, arguments, outputs[call_id][0], outputs[call_id][1])
        for call_id, (name, arguments) in pending.items()
        if call_id in outputs
    ]


def _anthropic_tool_calls(payload: dict[str, Any]) -> list[_ToolCall]:
    pending: dict[str, tuple[str, dict[str, Any]]] = {}
    outputs: dict[str, tuple[str, bool]] = {}
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                call_id = str(block.get("id") or "")
                name = str(block.get("name") or "")
                if call_id and name:
                    pending[call_id] = (name, _arguments(block.get("input")))
            elif block.get("type") == "tool_result":
                call_id = str(block.get("tool_use_id") or "")
                outputs[call_id] = (
                    _content_text(block.get("content")),
                    bool(block.get("is_error")),
                )
    return [
        _ToolCall(call_id, name, arguments, outputs[call_id][0], outputs[call_id][1])
        for call_id, (name, arguments) in pending.items()
        if call_id in outputs
    ]


def extract_tool_calls(path: str, payload: dict[str, Any]) -> list[_ToolCall]:
    if path == "/v1/chat/completions":
        return _chat_tool_calls(payload)
    if path == "/v1/responses":
        return _responses_tool_calls(payload)
    if path == "/v1/messages":
        return _anthropic_tool_calls(payload)
    return []


def protected_paths(text: str) -> list[str]:
    """Extract conservative protected-path patterns from the user objective."""

    result = [match.strip() for match in _PROTECTED_PATH.findall(text) if match.strip()]
    for clause in _PROTECTED_CLAUSE.findall(text):
        result.extend(_PATH_SHAPED_TOKEN.findall(clause))
        if re.search(r"\bdatabase(?:s)?\b", clause, re.IGNORECASE):
            result.extend(("*.sqlite", "*.sqlite3", "*.db"))
        if re.search(r"\btrace(?:s)?\b", clause, re.IGNORECASE):
            result.append("*.jsonl")
    if re.search(
        r"(?:do\s+not|don't|never)\s+(?:modify|change|edit)\s+(?:the\s+)?tests?\b",
        text,
        re.IGNORECASE,
    ):
        result.extend(("tests/**", "test_*.py", "*_test.*"))
    return list(dict.fromkeys(result))[:64]


def required_commands(text: str) -> list[str]:
    """Extract commands the user explicitly required to be run exactly."""

    commands = [" ".join(match[1].split()) for match in _REQUIRED_COMMAND.findall(text)]
    return list(dict.fromkeys(command for command in commands if command))[:16]


def _workspace_path(workspace: Path | None, value: Any) -> tuple[str, Path | None]:
    if not isinstance(value, str) or not value.strip():
        return "", None
    candidate = Path(value).expanduser()
    if workspace is None:
        return str(candidate), None
    resolved = (workspace / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        relative = resolved.relative_to(workspace)
    except ValueError:
        return _bounded_line(str(candidate), 240), None
    return str(relative), resolved


def _file_digest(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()[:16]


def refresh_workspace_state(checkpoint: TaskCheckpoint, workspace: Path | None) -> bool:
    if workspace is None:
        return False
    changed = False
    for relative, recorded in list(checkpoint.changed_files.items()):
        current = _file_digest((workspace / relative).resolve())
        if current != recorded and relative not in checkpoint.stale_files:
            checkpoint.mark_stale(relative)
            changed = True
    return changed


def _verification(command: str, output: str) -> tuple[str, int | None, int | None, str]:
    passed_match = _PYTEST_PASSED.search(output)
    failed_match = _PYTEST_FAILED.search(output)
    unittest_match = _UNITTEST_RAN.search(output)
    passed = int(passed_match.group(1)) if passed_match else None
    failed = int(failed_match.group(1)) if failed_match else None
    if unittest_match:
        total = int(unittest_match.group(1))
        if re.search(r"(?:^|\n)OK(?:\s|$)", output):
            passed, failed = total, 0
        elif "FAILED" in output:
            failed_counts = [int(value) for value in re.findall(r"(?:failures|errors)=(\d+)", output)]
            failed = sum(failed_counts) or 1
            passed = max(0, total - failed)
    success = (
        (failed or 0) == 0
        and passed is not None
        and ("passed" in output.lower() or re.search(r"(?:^|\n)OK(?:\s|$)", output))
    )
    outcome = "success" if success else "failure"
    if success and failed is None:
        failed = 0
    counts = []
    if passed is not None:
        counts.append(f"{passed} passed")
    if failed is not None:
        counts.append(f"{failed} failed")
    summary = f"Verification {outcome}: " + (", ".join(counts) or "test command did not report a passing total")
    return outcome, passed, failed, _bounded_line(summary)


def _observation(call: _ToolCall, workspace: Path | None) -> TaskObservation:
    canonical_arguments = json.dumps(call.arguments, sort_keys=True, separators=(",", ":"))
    signature = hashlib.sha256(f"{call.name}\0{canonical_arguments}".encode()).hexdigest()[:20]
    event_id = hashlib.sha256(f"{call.call_id}\0{signature}\0{call.result}".encode()).hexdigest()[:24]
    name = call.name.lower()
    command = str(call.arguments.get("command") or "") if name in {"bash", "shell", "exec"} else ""
    raw_path = call.arguments.get("path") or call.arguments.get("file_path")
    relative, resolved = _workspace_path(workspace, raw_path)
    mutates = name in {"edit", "write", "apply_patch", "applypatch"}

    if command and _TEST_COMMAND.search(command):
        outcome, passed, failed, summary = _verification(command, call.result)
        return TaskObservation(
            event_id=event_id,
            signature=signature,
            kind="verification",
            tool=call.name,
            outcome=outcome,
            summary=summary,
            command=_bounded_line(command, 500),
            source="client_tool_result",
            tests_passed=passed,
            tests_failed=failed,
        )

    lower = call.result.lower()
    failed = call.result_is_error or any(marker in lower for marker in _FAILURE_MARKERS)
    if mutates:
        succeeded = not failed and any(
            marker in lower for marker in ("successfully", "updated", "written", "applied", "replaced")
        )
        outcome = "success" if succeeded else ("failure" if failed else "unknown")
        summary = f"{call.name} {outcome}"
        if relative:
            summary += f" for {relative}"
    else:
        outcome = "failure" if failed else "success"
        action = _bounded_line(command, 180) if command else call.name
        summary = f"{call.name} {outcome}: {action}"
    return TaskObservation(
        event_id=event_id,
        signature=signature,
        kind="mutation" if mutates else "inspection",
        tool=call.name,
        outcome=outcome,
        summary=_bounded_line(summary),
        source="client_tool_result",
        command=_bounded_line(command, 500),
        path=relative,
        file_digest=_file_digest(resolved) if outcome == "success" and mutates else "",
        mutates_workspace=mutates,
    )


def reduce_request(
    path: str,
    payload: dict[str, Any],
    checkpoint: TaskCheckpoint,
    *,
    workspace: str | Path | None = None,
) -> bool:
    """Reduce all previously unseen tool results in one request into state."""

    root = Path(workspace).expanduser().resolve() if workspace else None
    changed = refresh_workspace_state(checkpoint, root)
    for call in extract_tool_calls(path, payload):
        observation = _observation(call, root)
        changed = checkpoint.apply(observation) or changed
        if (
            observation.mutates_workspace
            and observation.path
            and any(
                fnmatch(observation.path, pattern) or observation.path == pattern
                for pattern in checkpoint.protected_files
            )
        ):
            warning = f"Tool attempted to modify protected path: {observation.path}"
            if warning not in checkpoint.warnings:
                checkpoint.warnings.append(warning)
                del checkpoint.warnings[:-16]
                changed = True
    return changed


__all__ = [
    "extract_tool_calls",
    "protected_paths",
    "required_commands",
    "reduce_request",
    "refresh_workspace_state",
]
