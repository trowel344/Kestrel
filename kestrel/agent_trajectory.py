"""Privacy-bounded JSONL capture for reproducible coding-agent trajectories."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

TRACE_SCHEMA = 1
MAX_TEXT = 2_000
MAX_RECORD_BYTES = 16 * 1024

_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [^-]+ PRIVATE KEY-----.*?-----END [^-]+ PRIVATE KEY-----", re.I | re.S),
    re.compile(r"\b(?:sk|ghp|github_pat|xox[baprs])[-_A-Za-z0-9]{12,}\b", re.I),
    re.compile(r"(?i)\b(password|passwd|api[_-]?key|access[_-]?token|secret)\s*[:=]\s*([^\s,;]+)"),
)


def _redact(text: str) -> str:
    value = text.strip()
    for pattern in _SECRET_PATTERNS:
        if pattern.groups:
            value = pattern.sub(lambda match: f"{match.group(1)}=[REDACTED]", value)
        else:
            value = pattern.sub("[REDACTED]", value)
    if len(value) > MAX_TEXT:
        value = value[: MAX_TEXT - 14].rstrip() + " [TRUNCATED]"
    return value


def _digest(kind: str, value: str) -> str:
    return hashlib.sha256(f"{kind}\0{value}".encode()).hexdigest()[:24]


class AgentTrajectoryRecorder:
    """Append-only recorder containing no raw tool payloads or source bodies."""

    def __init__(self, path: str | Path, *, workspace: str | None = None) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._seen: set[str] = set()
        self._initialize(workspace)

    def _open_append(self) -> int:
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        return os.open(self.path, flags, 0o600)

    def _initialize(self, workspace: str | None) -> None:
        if self.path.exists() and (self.path.is_symlink() or not self.path.is_file()):
            raise ValueError("trajectory path must be a regular non-symlink file")
        if self.path.exists() and self.path.stat().st_size:
            self._load_seen()
            return
        self._append(
            {
                "record": "header",
                "schema": TRACE_SCHEMA,
                "created_unix": int(time.time()),
                "workspace_digest": _digest("workspace", workspace or ""),
                "privacy": "redacted user text; structured tool summaries; no raw tool output or source",
            },
            dedupe=False,
        )

    def _load_seen(self) -> None:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        for line in lines:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                self._seen.add(item["id"])

    def _append(self, payload: dict[str, Any], *, dedupe: bool = True) -> bool:
        identifier = payload.get("id")
        if dedupe and isinstance(identifier, str) and identifier in self._seen:
            return False
        encoded = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        if len(encoded) > MAX_RECORD_BYTES:
            raise ValueError("trajectory record exceeds bounded record size")
        with self._lock:
            descriptor = self._open_append()
            try:
                os.write(descriptor, encoded)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            if isinstance(identifier, str):
                self._seen.add(identifier)
        return True

    def user(self, route: str, text: str) -> bool:
        redacted = _redact(text)
        return self._append(
            {
                "record": "user",
                "id": _digest("user", redacted),
                "route": route,
                "text": redacted,
            }
        )

    def task_start(
        self,
        objective: str,
        protected_files: list[str],
        required_commands: list[str] | None = None,
    ) -> bool:
        redacted = _redact(objective)
        commands = [_redact(command) for command in (required_commands or [])[-16:]]
        value = json.dumps([redacted, protected_files, commands], separators=(",", ":"))
        return self._append(
            {
                "record": "task_start",
                "id": _digest("task_start", value),
                "objective": redacted,
                "protected_files": [str(path)[:512] for path in protected_files[-32:]],
                "required_commands": commands,
            }
        )

    def observation(self, event: dict[str, Any]) -> bool:
        allowed = {
            key: event.get(key)
            for key in (
                "event_id",
                "signature",
                "kind",
                "tool",
                "outcome",
                "summary",
                "source",
                "command",
                "path",
                "file_digest",
                "mutates_workspace",
                "tests_passed",
                "tests_failed",
                "progress_revision",
            )
            if event.get(key) not in (None, "")
        }
        allowed["summary"] = _redact(str(allowed.get("summary", "")))
        allowed["command"] = _redact(str(allowed.get("command", "")))
        identifier = str(allowed.get("event_id") or _digest("observation", json.dumps(allowed, sort_keys=True)))
        return self._append({"record": "observation", "id": identifier, "observation": allowed})

    def usage(self, request_id: str, usage: dict[str, int]) -> bool:
        bounded = {key: max(0, int(value)) for key, value in usage.items()}
        return self._append(
            {
                "record": "usage",
                "id": _digest("usage", request_id),
                "request_id": request_id,
                "tokens_estimate": bounded,
            }
        )

    def terminal(self, text: str, *, status: str, verification: str) -> bool:
        value = _redact(text)
        return self._append(
            {
                "record": "terminal",
                "id": _digest("terminal", value),
                "assistant_digest": _digest("assistant", value),
                "assistant_chars": len(value),
                "task_status": status,
                "verification": verification,
            }
        )


__all__ = ["AgentTrajectoryRecorder", "TRACE_SCHEMA"]
