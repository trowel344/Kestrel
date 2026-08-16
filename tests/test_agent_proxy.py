from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.error import URLError
from urllib.request import Request, urlopen

import pytest

from kestrel.agent_proxy import (
    AgentProxyServer,
    _llama_token_counter,
    _ProxyHandler,
    _terminate_child,
    _warm_ollama_model,
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
    assert command[command.index("--parallel") + 1] == "1"


def test_backend_command_forwards_split_overrides():
    command = build_backend_command(
        "model.gguf",
        19999,
        alias="local",
        context="8192",
        reasoning="medium",
        api_key_file=None,
        managed_token=None,
        timeout=30.0,
        gpu_layers="24",
        cpu_moe="on",
    )

    assert command[command.index("--gpu-layers") + 1] == "24"
    assert command[command.index("--cpu-moe") + 1] == "on"


def test_backend_command_omits_auto_split_overrides():
    command = build_backend_command(
        "model.gguf",
        19999,
        alias="local",
        context="8192",
        reasoning="medium",
        api_key_file=None,
        managed_token=None,
        timeout=30.0,
    )

    assert "--gpu-layers" not in command
    assert "--cpu-moe" not in command


def test_backend_command_rejects_invalid_split_overrides():
    with pytest.raises(ValueError, match="gpu_layers"):
        build_backend_command(
            "model.gguf",
            19999,
            alias="local",
            context="8192",
            reasoning="medium",
            api_key_file=None,
            managed_token=None,
            timeout=30.0,
            gpu_layers="banana",
        )
    with pytest.raises(ValueError, match="cpu_moe"):
        build_backend_command(
            "model.gguf",
            19999,
            alias="local",
            context="8192",
            reasoning="medium",
            api_key_file=None,
            managed_token=None,
            timeout=30.0,
            cpu_moe="sometimes",
        )


def test_ollama_warmup_loads_selected_model_without_generating(monkeypatch):
    seen = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"done":true,"done_reason":"load"}'

    def fake_urlopen(request, *, timeout):
        seen["url"] = request.full_url
        seen["payload"] = json.loads(request.data)
        seen["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("kestrel.agent_proxy.urllib.request.urlopen", fake_urlopen)

    _warm_ollama_model("qwen3.5:4b", timeout=42.0)

    assert seen == {
        "url": "http://127.0.0.1:11434/api/generate",
        "payload": {
            "model": "qwen3.5:4b",
            "prompt": "",
            "stream": False,
            "keep_alive": "5m",
        },
        "timeout": 42.0,
    }


def test_ollama_warmup_fails_startup_when_model_cannot_load(monkeypatch):
    monkeypatch.setattr(
        "kestrel.agent_proxy.urllib.request.urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(URLError("model load failed")),
    )

    with pytest.raises(RuntimeError, match="Ollama could not load model 'too-large:latest'"):
        _warm_ollama_model("too-large:latest", timeout=1.0)


def test_llama_token_counter_authenticates_caches_and_counts_exact_tokens(monkeypatch):
    seen = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"tokens":[1,2,3]}'

    def fake_urlopen(request, *, timeout):
        seen.append((request, timeout))
        return FakeResponse()

    monkeypatch.setattr("kestrel.agent_proxy.urllib.request.urlopen", fake_urlopen)
    counter = _llama_token_counter("http://127.0.0.1:9999", "private-token")

    assert counter("exact text") == 3
    assert counter("exact text") == 3
    assert len(seen) == 1
    request, timeout = seen[0]
    assert request.full_url == "http://127.0.0.1:9999/tokenize"
    assert request.headers["Authorization"] == "Bearer private-token"
    assert timeout == 5.0


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


def test_proxy_treats_downstream_disconnect_as_normal(monkeypatch):
    class FakeResponse:
        headers = {}
        status = 200
        reason = "OK"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    handler = SimpleNamespace(
        path="/health",
        command="GET",
        headers={},
        close_connection=False,
        proxy_server=SimpleNamespace(
            backend_model=None,
            backend_base_url="http://127.0.0.1:1",
            request_timeout=1.0,
            memory=None,
        ),
    )
    monkeypatch.setattr("kestrel.agent_proxy.urllib.request.urlopen", lambda *_args, **_kwargs: FakeResponse())
    monkeypatch.setattr(
        "kestrel.agent_proxy._forward_response",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(BrokenPipeError()),
    )

    _ProxyHandler._proxy(handler, None)

    assert handler.close_connection is True


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


def test_proxy_enriches_request_and_observes_streamed_response():
    class FakeMemory:
        def __init__(self):
            self.observed = []
            self.observed_event = threading.Event()

        def enrich_request(self, path, payload):
            assert path == "/v1/responses"
            return {**payload, "instructions": "remembered state"}

        def observe_response(self, path, content_type, body):
            self.observed.append((path, content_type, body))
            self.observed_event.set()

    memory = FakeMemory()
    _BackendHandler.seen = []
    backend = ThreadingHTTPServer(("127.0.0.1", 0), _BackendHandler)
    backend_thread = threading.Thread(target=backend.serve_forever, daemon=True)
    backend_thread.start()
    proxy = AgentProxyServer(
        ("127.0.0.1", 0),
        backend.server_address[1],
        max_body_bytes=1024,
        request_timeout=2.0,
        memory=memory,
    )
    proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()
    try:
        response = urlopen(
            Request(
                f"http://127.0.0.1:{proxy.server_address[1]}/v1/responses",
                data=json.dumps({"input": "remember this"}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            ),
            timeout=2,
        )
        returned = json.loads(response.read())
        assert returned["instructions"] == "remembered state"
        assert memory.observed_event.wait(timeout=2)
        assert memory.observed
        assert memory.observed[0][0] == "/v1/responses"
        assert b"remembered state" in memory.observed[0][2]
        assert proxy.memory_errors == 0
    finally:
        proxy.shutdown()
        proxy.server_close()
        proxy_thread.join(timeout=2)
        backend.shutdown()
        backend.server_close()
        backend_thread.join(timeout=2)


def test_proxy_live_sunmap_checkpoint_and_code_evidence_round_trip(tmp_path):
    from kestrel.sunmap_memory import SunMapMemory

    source = tmp_path / "service.py"
    source.write_text("def repair_service():\n    return 'current-proof'\n", encoding="utf-8")
    database = tmp_path / "memory.sqlite3"
    memory = SunMapMemory(database, token_budget=2048, workspace=tmp_path)
    _BackendHandler.seen = []
    backend = ThreadingHTTPServer(("127.0.0.1", 0), _BackendHandler)
    backend_thread = threading.Thread(target=backend.serve_forever, daemon=True)
    backend_thread.start()
    proxy = AgentProxyServer(
        ("127.0.0.1", 0),
        backend.server_address[1],
        max_body_bytes=32 * 1024,
        request_timeout=2.0,
        memory=memory,
        api_token="private-token",
    )
    proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()
    base = f"http://127.0.0.1:{proxy.server_address[1]}"
    try:
        first_payload = {
            "messages": [
                {
                    "role": "user",
                    "content": "Repair repair_service. Do not modify tests.",
                }
            ]
        }
        first = urlopen(
            Request(
                f"{base}/v1/chat/completions",
                data=json.dumps(first_payload).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer private-token",
                },
                method="POST",
            ),
            timeout=2,
        )
        enriched = json.loads(first.read())
        suffix = enriched["messages"][-1]["content"]
        assert "<CURRENT_CODE_EVIDENCE>" in suffix
        assert "current-proof" in suffix
        assert "<SUNMAP_TASK_STATE>" in suffix
        assert "tests/**" in suffix

        followup_payload = {
            "messages": [
                *first_payload["messages"],
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "tests",
                            "function": {
                                "name": "bash",
                                "arguments": json.dumps({"command": "pytest -q"}),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "tests",
                    "content": "3 passed in 0.01s",
                },
            ]
        }
        followup = urlopen(
            Request(
                f"{base}/v1/chat/completions",
                data=json.dumps(followup_payload).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer private-token",
                },
                method="POST",
            ),
            timeout=2,
        )
        checkpoint = json.loads(followup.read())["messages"][-1]["content"]
        assert "verify=passed 3/0 @client" in checkpoint

        restarted = SunMapMemory(database, token_budget=2048, workspace=tmp_path)
        assert restarted.status()["task"]["verification"] == "passed"
    finally:
        proxy.shutdown()
        proxy.server_close()
        proxy_thread.join(timeout=2)
        backend.shutdown()
        backend.server_close()
        backend_thread.join(timeout=2)


def test_ollama_backend_enforces_proxy_auth_and_rewrites_model():
    _BackendHandler.seen = []
    backend = ThreadingHTTPServer(("127.0.0.1", 0), _BackendHandler)
    backend_thread = threading.Thread(target=backend.serve_forever, daemon=True)
    backend_thread.start()
    proxy = AgentProxyServer(
        ("127.0.0.1", 0),
        backend.server_address[1],
        max_body_bytes=1024,
        request_timeout=2.0,
        api_token="private-token",
        backend_base_url=f"http://127.0.0.1:{backend.server_address[1]}",
        backend_model="qwen3.5:4b",
        backend_reasoning="off",
    )
    proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()
    base = f"http://127.0.0.1:{proxy.server_address[1]}"
    try:
        with pytest.raises(Exception) as error:
            urlopen(f"{base}/health", timeout=2)
        assert getattr(error.value, "code", None) == 401
        assert not _BackendHandler.seen

        health = urlopen(Request(f"{base}/health", headers={"Authorization": "Bearer private-token"}), timeout=2)
        assert health.status == 200
        assert _BackendHandler.seen[-1][0] == "/api/tags"

        response = urlopen(
            Request(
                f"{base}/v1/chat/completions",
                data=json.dumps({"model": "kestrel-local", "messages": []}).encode(),
                headers={"Content-Type": "application/json", "X-Api-Key": "private-token"},
                method="POST",
            ),
            timeout=2,
        )
        forwarded = json.loads(response.read())
        assert forwarded["model"] == "qwen3.5:4b"
        assert forwarded["reasoning_effort"] == "none"
        _path, forwarded_headers, _body = _BackendHandler.seen[-1]
        assert "Authorization" not in forwarded_headers
        assert "X-Api-Key" not in forwarded_headers

        unsupported = Request(
            f"{base}/v1/responses",
            data=b"{}",
            headers={"Authorization": "Bearer private-token"},
            method="POST",
        )
        with pytest.raises(Exception) as error:
            urlopen(unsupported, timeout=2)
        assert getattr(error.value, "code", None) == 404
    finally:
        proxy.shutdown()
        proxy.server_close()
        proxy_thread.join(timeout=2)
        backend.shutdown()
        backend.server_close()
        backend_thread.join(timeout=2)


def test_llama_backend_keeps_validated_credential_for_upstream():
    _BackendHandler.seen = []
    backend = ThreadingHTTPServer(("127.0.0.1", 0), _BackendHandler)
    backend_thread = threading.Thread(target=backend.serve_forever, daemon=True)
    backend_thread.start()
    proxy = AgentProxyServer(
        ("127.0.0.1", 0),
        backend.server_address[1],
        max_body_bytes=1024,
        request_timeout=2.0,
        api_token="shared-token",
    )
    proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()
    try:
        response = urlopen(
            Request(
                f"http://127.0.0.1:{proxy.server_address[1]}/health",
                headers={"Authorization": "Bearer shared-token"},
            ),
            timeout=2,
        )
        assert response.status == 200
        assert _BackendHandler.seen[-1][1]["Authorization"] == "Bearer shared-token"
    finally:
        proxy.shutdown()
        proxy.server_close()
        proxy_thread.join(timeout=2)
        backend.shutdown()
        backend.server_close()
        backend_thread.join(timeout=2)


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
