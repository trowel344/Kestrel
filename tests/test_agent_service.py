import json
import os

import pytest

from kestrel import agent_service
from kestrel.errors import IntegrationError


@pytest.fixture
def isolated_agent_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_AGENT_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("KESTREL_AGENT_CONFIG_DIR", str(tmp_path / "config"))
    return tmp_path


def test_credential_is_private_and_stable(isolated_agent_dirs):
    first = agent_service.read_credential()
    second = agent_service.read_credential()
    path = agent_service.credential_path()

    assert first == second
    assert len(first) >= 32
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700


def test_credential_rejects_broad_permissions(isolated_agent_dirs):
    path = agent_service.credential_path()
    path.parent.mkdir(parents=True)
    path.write_text("secret\n", encoding="utf-8")
    path.chmod(0o644)

    with pytest.raises(IntegrationError, match="permissions"):
        agent_service.read_credential()


@pytest.mark.parametrize(
    "value",
    ["", "# comment only\n", "first\nsecond\n", " leading-space\n", "valid-key-value-\x7f\n", "valid-key-value-é\n"],
)
def test_invalid_credential_is_rejected_before_spawn(isolated_agent_dirs, monkeypatch, value):
    monkeypatch.setattr(agent_service.sys, "platform", "linux")
    path = agent_service.credential_path()
    path.parent.mkdir(parents=True)
    path.write_text(value, encoding="utf-8")
    path.chmod(0o600)
    monkeypatch.setattr(agent_service, "_port_busy", lambda _port: False)
    spawned = []
    monkeypatch.setattr(agent_service.subprocess, "Popen", lambda *_args, **_kwargs: spawned.append(True))

    with pytest.raises(IntegrationError, match="credential"):
        agent_service.start_server("model.gguf")

    assert spawned == []
    assert not agent_service.state_path().exists()


def test_managed_server_fails_before_spawn_off_linux(isolated_agent_dirs, monkeypatch):
    monkeypatch.setattr(agent_service.sys, "platform", "darwin")
    monkeypatch.setattr(
        agent_service.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("spawned")),
    )

    with pytest.raises(IntegrationError, match="require Linux"):
        agent_service.start_server("model.gguf")


def test_start_builds_authenticated_loopback_owned_process(isolated_agent_dirs, monkeypatch):
    monkeypatch.setattr(agent_service.sys, "platform", "linux")
    monkeypatch.setattr(agent_service, "_require_sunmap", lambda: None)
    captured = {}

    class FakeProcess:
        pid = 43210

        def poll(self):
            return None

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(agent_service.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(agent_service, "_port_busy", lambda _port: False)
    monkeypatch.setattr(agent_service, "_ready", lambda _port, _token: True)
    monkeypatch.setattr(
        agent_service,
        "server_status",
        lambda: {"status": "ready", "running": True, "healthy": True, "url": "http://127.0.0.1:8111"},
    )

    result = agent_service.start_server(
        "/models/model.gguf",
        alias="work-model",
        port=8111,
        context="4096",
        reasoning="high",
        sunmap_db=str(isolated_agent_dirs / "memory.sqlite3"),
        sunmap_tokens=2048,
    )

    command = captured["command"]
    assert command[:3] == [os.sys.executable, "-m", "kestrel.agent_proxy"]
    assert command[command.index("--model") + 1] == "/models/model.gguf"
    assert "--api-key-file" in command
    assert "--managed-token" in command
    assert command[command.index("--sunmap-db") + 1].endswith("memory.sqlite3")
    assert command[command.index("--sunmap-tokens") + 1] == "2048"
    assert captured["kwargs"]["start_new_session"] is True
    state = json.loads(agent_service.state_path().read_text(encoding="utf-8"))
    assert state["pid"] == 43210
    assert state["model"] == "/models/model.gguf"
    assert state["sunmap"]["token_budget"] == 2048
    assert result["reused"] is False


def test_start_forwards_gpu_split_overrides(isolated_agent_dirs, monkeypatch):
    monkeypatch.setattr(agent_service.sys, "platform", "linux")
    captured = {}

    class FakeProcess:
        pid = 43211

        def poll(self):
            return None

    def fake_popen(command, **kwargs):
        captured["command"] = command
        return FakeProcess()

    monkeypatch.setattr(agent_service.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(agent_service, "_port_busy", lambda _port: False)
    monkeypatch.setattr(agent_service, "_ready", lambda _port, _token: True)
    monkeypatch.setattr(
        agent_service,
        "server_status",
        lambda: {"status": "ready", "running": True, "healthy": True, "url": "http://127.0.0.1:8111"},
    )

    result = agent_service.start_server(
        "/models/model.gguf",
        context="8192",
        gpu_layers="24",
        cpu_moe="on",
    )

    command = captured["command"]
    assert command[command.index("--gpu-layers") + 1] == "24"
    assert command[command.index("--cpu-moe") + 1] == "on"
    state = json.loads(agent_service.state_path().read_text(encoding="utf-8"))
    assert state["gpu_layers"] == "24"
    assert state["cpu_moe"] == "on"
    assert result["reused"] is False


def test_ollama_agent_start_refuses_context_smaller_than_advertised(isolated_agent_dirs, monkeypatch):
    monkeypatch.setattr(agent_service.sys, "platform", "linux")
    monkeypatch.setattr(agent_service, "_ollama_context", lambda _model: 4096)

    with pytest.raises(IntegrationError, match="configured for 4096 tokens") as error:
        agent_service.start_server("ollama://qwen:small", context="16384")

    assert "PARAMETER num_ctx 16384" in str(error.value.hint)


def test_stop_refuses_stale_pid_without_signaling(isolated_agent_dirs, monkeypatch):
    monkeypatch.setattr(agent_service.sys, "platform", "linux")
    state = {
        "schema": 1,
        "pid": 1234,
        "token": "owner",
        "model": "model",
        "alias": "alias",
        "port": 8080,
        "url": "http://127.0.0.1:8080",
        "context": "auto",
        "reasoning": "auto",
        "log": "log",
    }
    path = agent_service.state_path()
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(agent_service, "_owned_alive", lambda _state: False)
    called = []
    monkeypatch.setattr(agent_service.os, "killpg", lambda *_args: called.append(True))

    with pytest.raises(IntegrationError, match="stale"):
        agent_service.stop_server()

    assert called == []


def test_owned_alive_recovers_backend_after_proxy_leader_death(monkeypatch):
    state = {"pid": 43210, "token": "owner-token"}
    monkeypatch.setattr(agent_service, "_cmdline", lambda _pid: [])
    monkeypatch.setattr(
        agent_service, "_owned_group_member", lambda pgid, token: (pgid, token) == (43210, "owner-token")
    )

    assert agent_service._owned_alive(state)


def test_protocol_probe_requires_all_client_routes(monkeypatch):
    monkeypatch.setattr(
        agent_service,
        "server_status",
        lambda: {"healthy": True, "url": "http://127.0.0.1:8080"},
    )
    monkeypatch.setattr(agent_service, "read_credential", lambda: "token")
    codes = {
        "/v1/chat/completions": 400,
        "/v1/responses": 422,
        "/v1/messages": 400,
        "/v1/messages/count_tokens": 404,
    }
    monkeypatch.setattr(agent_service, "_request", lambda path, **_kwargs: codes[path])

    assert agent_service.probe_protocols() == {
        "chat_completions": True,
        "responses": True,
        "anthropic_messages": False,
    }


def test_ready_requires_authenticated_route(monkeypatch):
    monkeypatch.setattr(agent_service, "_request", lambda *_args, **_kwargs: 401)

    assert agent_service._ready(8080, "wrong-token") is False


def test_ready_rejects_server_that_does_not_enforce_auth(monkeypatch):
    monkeypatch.setattr(agent_service, "_request", lambda *_args, **_kwargs: 400)

    assert agent_service._ready(8080, "token") is False


def test_protocol_probe_rejects_auth_failures(monkeypatch):
    monkeypatch.setattr(
        agent_service,
        "server_status",
        lambda: {"healthy": True, "url": "http://127.0.0.1:8080"},
    )
    monkeypatch.setattr(agent_service, "read_credential", lambda: "wrong-token")
    monkeypatch.setattr(agent_service, "_request", lambda *_args, **_kwargs: 401)

    assert not any(agent_service.probe_protocols().values())


def test_log_tail_is_bounded(isolated_agent_dirs):
    path = agent_service.log_path()
    path.parent.mkdir(parents=True)
    path.write_text("\n".join(str(index) for index in range(20)), encoding="utf-8")

    assert agent_service.tail_logs(3).splitlines() == ["17", "18", "19"]
