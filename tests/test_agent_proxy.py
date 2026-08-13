from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen

import pytest

from kestrel.agent_proxy import (
    AgentProxyServer,
    _terminate_child,
    build_backend_command,
    merge_responses_instructions,
)


class _BackendHandler(BaseHTTPRequestHandler):
    seen: list[tuple[str, dict[str, str], bytes]] = []

    def log_message(self, _format, *args):
        return None

    def do_GET(self):  # noqa: N802
        self.__class__.seen.append((self.path, dict(self.headers), b""))
        if self.path == "/delayed":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b"data: first\n\n")
            self.wfile.flush()
            time.sleep(0.3)
            self.wfile.write(b"data: second\n\n")
            self.wfile.flush()
            return
        body = b"event: ready\ndata: ok\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def do_POST(self):  # noqa: N802
        length = int(self.headers["Content-Length"])
        body = self.rfile.read(length)
        self.__class__.seen.append((self.path, dict(self.headers), body))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def proxy_pair():
    _BackendHandler.seen = []
    backend = ThreadingHTTPServer(("127.0.0.1", 0), _BackendHandler)
    backend_thread = threading.Thread(target=backend.serve_forever, daemon=True)
    backend_thread.start()
    proxy = AgentProxyServer(
        ("127.0.0.1", 0),
        backend.server_address[1],
        max_body_bytes=1024,
        request_timeout=2.0,
    )
    proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()
    try:
        yield f"http://127.0.0.1:{proxy.server_address[1]}"
    finally:
        proxy.shutdown()
        proxy.server_close()
        proxy_thread.join(timeout=2)
        backend.shutdown()
        backend.server_close()
        backend_thread.join(timeout=2)


def test_responses_transform_merges_only_leading_system_messages():
    payload = {
        "instructions": "Existing policy",
        "input": [
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "Dev"}]},
            {"type": "message", "role": "system", "content": "System"},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hi"}]},
            {"type": "function_call", "call_id": "1", "name": "read", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "1", "output": "ok"},
        ],
    }

    transformed = merge_responses_instructions(payload)

    assert transformed["instructions"] == "Existing policy\n\nDev\n\nSystem"
    assert [item["type"] for item in transformed["input"]] == ["message", "function_call", "function_call_output"]
    assert transformed["input"][0]["role"] == "user"
    assert payload["input"][0]["role"] == "developer"


def test_responses_transform_leaves_tool_first_input_unchanged():
    payload = {"input": [{"type": "function_call", "call_id": "1", "name": "read", "arguments": "{}"}]}
    assert merge_responses_instructions(payload) == payload


def test_responses_transform_matches_codex_first_and_followup_shapes():
    instructions = "I" * 20751
    developer = "environment context\n" * 100
    tools = [{"type": "function", "name": f"tool_{index}"} for index in range(7)]
    first = {
        "instructions": instructions,
        "input": [
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": developer}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "context"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "read proof"}]},
        ],
        "tools": tools,
    }

    normalized = merge_responses_instructions(first)
    assert normalized["instructions"] == instructions + "\n\n" + developer
    assert [item["role"] for item in normalized["input"]] == ["user", "user"]
    assert normalized["tools"] is tools

    followup = {
        **first,
        "input": [
            first["input"][0],
            {"type": "function_call", "call_id": "call-1", "name": "exec_command", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call-1", "output": "proof"},
        ],
    }
    normalized_followup = merge_responses_instructions(followup)
    assert [item["type"] for item in normalized_followup["input"]] == ["function_call", "function_call_output"]


def test_shell_free_child_command_contains_managed_options():
    command = build_backend_command(
        "model.gguf",
        19999,
        alias="local",
        context="32768",
        reasoning="high",
        api_key_file="/tmp/api-key",
        managed_token="opaque-owner-token",
        timeout=30.0,
    )

    assert command[:4] == [sys.executable, "-m", "kestrel", "serve"]
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "19999"
    assert command[command.index("--api-key-file") + 1] == "/tmp/api-key"
    assert command[command.index("--managed-token") + 1] == "opaque-owner-token"


def test_proxy_forwards_auth_and_streams_response(proxy_pair):
    response = urlopen(
        Request(
            f"{proxy_pair}/health",
            headers={"Authorization": "Bearer secret", "X-Api-Key": "local-key"},
        ),
        timeout=2,
    )
    assert response.read() == b"event: ready\ndata: ok\n\n"
    assert response.headers["Content-Type"] == "text/event-stream"
    path, headers, _body = _BackendHandler.seen[-1]
    assert path == "/health"
    assert headers["Authorization"] == "Bearer secret"
    assert headers["X-Api-Key"] == "local-key"


def test_proxy_does_not_buffer_delayed_sse_event(proxy_pair):
    started = time.monotonic()
    response = urlopen(f"{proxy_pair}/delayed", timeout=2)

    assert response.readline() == b"data: first\n"
    assert time.monotonic() - started < 0.25
    assert response.read() == b"\ndata: second\n\n"


def test_proxy_transforms_responses_and_preserves_tools(proxy_pair):
    payload = {
        "input": [
            {"type": "message", "role": "developer", "content": "Use tools"},
            {"type": "function_call", "call_id": "x", "name": "read", "arguments": "{}"},
        ]
    }
    response = urlopen(
        Request(
            f"{proxy_pair}/v1/responses",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer secret"},
            method="POST",
        ),
        timeout=2,
    )
    transformed = json.loads(response.read())
    assert transformed["instructions"] == "Use tools"
    assert transformed["input"] == [payload["input"][1]]


def test_proxy_rejects_body_over_limit(proxy_pair):
    request = Request(
        f"{proxy_pair}/v1/responses",
        data=b"x" * 1025,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(Exception) as error:
        urlopen(request, timeout=2)
    assert getattr(error.value, "code", None) == 413
    assert not _BackendHandler.seen


def test_terminate_child_reaps_process_group():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    _terminate_child(child, timeout=2)
    assert child.poll() is not None
