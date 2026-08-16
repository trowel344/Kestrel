import json
from pathlib import Path

from kestrel.cli import agents, state
from kestrel.cli.main import _run_dispatched
from kestrel.cli.parser import build_parser
from kestrel.integrations import LaunchMetadata


def _metadata(client="pi"):
    return LaunchMetadata(
        client=client,
        command=(client, "--model", "kestrel-local"),
        environment={},
        config_path=Path(f"/tmp/{client}-config"),
        endpoint="http://127.0.0.1:8080",
        model="kestrel-local",
        context_size=4096,
        reasoning="medium",
        max_tokens=8192,
    )


def test_agents_setup_json_dry_run_is_single_document(monkeypatch, capsys):
    monkeypatch.setattr(agents.integrations, "launch_metadata", lambda client, **_kwargs: _metadata(client))
    monkeypatch.setattr(agents.agent_service, "ensure_credential", lambda: (_ for _ in ()).throw(AssertionError()))
    parser = build_parser()

    rc = _run_dispatched(
        parser,
        parser.parse_args(["agents", "setup", "pi", "--dry-run", "--json"]),
    )

    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["dry_run"] is True
    assert payload["integrations"][0]["client"] == "pi"
    assert captured.err == ""


def test_agents_reject_context_too_small_for_tool_clients(capsys):
    parser = build_parser()

    rc = _run_dispatched(
        parser,
        parser.parse_args(["agents", "setup", "pi", "--context", "4096", "--dry-run", "--json"]),
    )

    assert rc == 1
    assert "at least 8192" in json.loads(capsys.readouterr().out)["error"]["message"]


def test_agents_status_reports_server_and_integrations(monkeypatch, capsys):
    monkeypatch.setattr(
        agents.agent_service,
        "server_status",
        lambda: {
            "status": "ready",
            "running": True,
            "healthy": True,
            "url": "http://127.0.0.1:8080",
            "alias": "kestrel-local",
            "context": "32768",
        },
    )
    monkeypatch.setattr(
        agents.agent_service,
        "probe_protocols",
        lambda: {"chat_completions": True, "responses": True, "anthropic_messages": True},
    )
    monkeypatch.setattr(agents, "_integration_rows", lambda: [])
    parser = build_parser()

    rc = _run_dispatched(parser, parser.parse_args(["agents", "status", "--json"]))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["server"]["status"] == "ready"
    assert payload["protocols"]["responses"] is True


def test_agents_doctor_does_not_claim_unrun_client_smoke(monkeypatch, capsys):
    monkeypatch.setattr(
        agents.agent_service,
        "server_status",
        lambda: {
            "status": "ready",
            "running": True,
            "healthy": True,
            "url": "http://127.0.0.1:8080",
            "alias": "kestrel-local",
            "context": "32768",
        },
    )
    monkeypatch.setattr(
        agents.agent_service,
        "probe_protocols",
        lambda: {"chat_completions": True, "responses": True, "anthropic_messages": True},
    )
    monkeypatch.setattr(
        agents,
        "_integration_rows",
        lambda: [
            {
                "client": "pi",
                "installed": True,
                "configured": True,
                "owned": True,
                "path": "/tmp/models.json",
                "details": {
                    "endpoint": "http://127.0.0.1:8080",
                    "model": "kestrel-local",
                    "context": "32768",
                },
            }
        ],
    )
    parser = build_parser()

    rc = _run_dispatched(parser, parser.parse_args(["agents", "doctor", "--json"]))

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["status"] == "route_ready"
    assert payload["checks"][0]["client_smoke_test"] == "not_run"
    assert "usable" not in payload["checks"][0]


def test_agents_launch_rejects_live_json(capsys):
    parser = build_parser()
    rc = _run_dispatched(parser, parser.parse_args(["agents", "launch", "pi", "--json"]))

    assert rc == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "invalid_input"


def test_agents_launch_dry_run_does_not_write_configuration(monkeypatch, capsys):
    monkeypatch.setattr(agents.integrations, "launch_metadata", lambda client, **_kwargs: _metadata(client))
    monkeypatch.setattr(
        agents.integrations,
        "setup_agent_integration",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("dry-run wrote configuration")),
    )
    parser = build_parser()

    rc = _run_dispatched(
        parser,
        parser.parse_args(["agents", "launch", "pi", "--model", "model.gguf", "--dry-run", "--json"]),
    )

    assert rc == 0
    assert json.loads(capsys.readouterr().out)["dry_run"] is True


def test_agents_start_passes_sunmap_settings(monkeypatch, tmp_path, capsys):
    captured = {}

    def start_server(model, **kwargs):
        captured["model"] = model
        captured.update(kwargs)
        return {"status": "ready", "url": "http://127.0.0.1:8080", "alias": "kestrel-local"}

    monkeypatch.setattr(agents.agent_service, "start_server", start_server)
    monkeypatch.setattr(
        agents.agent_service,
        "probe_protocols",
        lambda: {"chat_completions": True, "responses": True, "anthropic_messages": False},
    )
    database = tmp_path / "agent-memory.sqlite3"
    parser = build_parser()

    rc = _run_dispatched(
        parser,
        parser.parse_args(
            [
                "agents",
                "start",
                "model.gguf",
                "--sunmap-db",
                str(database),
                "--sunmap-tokens",
                "2048",
                "--json",
            ]
        ),
    )

    assert rc == 0
    assert captured["model"] == "model.gguf"
    assert captured["sunmap_db"] == str(database)
    assert captured["sunmap_tokens"] == 2048
    assert json.loads(capsys.readouterr().out)["status"] == "ready"


def test_agents_start_passes_split_overrides(monkeypatch, capsys):
    captured = {}

    def start_server(model, **kwargs):
        captured["model"] = model
        captured.update(kwargs)
        return {"status": "ready", "url": "http://127.0.0.1:8080", "alias": "kestrel-local"}

    monkeypatch.setattr(agents.agent_service, "start_server", start_server)
    monkeypatch.setattr(
        agents.agent_service,
        "probe_protocols",
        lambda: {"chat_completions": True, "responses": True, "anthropic_messages": False},
    )
    parser = build_parser()

    rc = _run_dispatched(
        parser,
        parser.parse_args(
            ["agents", "start", "model.gguf", "--gpu-layers", "24", "--cpu-moe", "on", "--json"]
        ),
    )

    assert rc == 0
    assert captured["model"] == "model.gguf"
    assert captured["gpu_layers"] == "24"
    assert captured["cpu_moe"] == "on"


def test_agents_launch_dry_run_accepts_split_overrides(monkeypatch, capsys):
    monkeypatch.setattr(agents.integrations, "launch_metadata", lambda client, **_kwargs: _metadata(client))
    parser = build_parser()

    rc = _run_dispatched(
        parser,
        parser.parse_args(
            [
                "agents",
                "launch",
                "pi",
                "--model",
                "model.gguf",
                "--gpu-layers",
                "all",
                "--cpu-moe",
                "off",
                "--dry-run",
                "--json",
            ]
        ),
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["server_model"] == "model.gguf"


def test_agents_launch_checks_executable_before_config_or_server(monkeypatch, capsys):
    monkeypatch.setattr(agents.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        agents.integrations,
        "setup_agent_integration",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("wrote configuration")),
    )
    monkeypatch.setattr(
        agents.agent_service,
        "start_server",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("started server")),
    )
    parser = build_parser()

    rc = _run_dispatched(
        parser,
        parser.parse_args(["agents", "launch", "claude", "--model", "model.gguf"]),
    )

    assert rc == 1
    assert "not installed" in capsys.readouterr().err


def test_agents_token_prints_only_credential(monkeypatch, capsys):
    monkeypatch.setattr(agents.agent_service, "read_credential", lambda: "private-token")
    parser = build_parser()

    rc = _run_dispatched(parser, parser.parse_args(["agents", "token"]))

    assert rc == 0
    assert capsys.readouterr().out == "private-token\n"


def test_agents_usage_reports_process_and_throughput(monkeypatch, capsys):
    monkeypatch.setattr(
        agents.agent_service,
        "server_status",
        lambda: {
            "status": "ready",
            "running": True,
            "healthy": True,
            "pid": 1234,
            "model": "m.gguf",
            "alias": "kestrel-local",
            "url": "http://127.0.0.1:8080",
            "context": "32768",
            "reasoning": "high",
        },
    )
    monkeypatch.setattr(
        agents,
        "_process_usage",
        lambda pid: {
            "available": True,
            "pid": 1234,
            "rss_mib": 4096,
            "cpu_percent": 12.5,
            "vram_mib": 5120,
            "elapsed_seconds": 900,
        },
    )
    monkeypatch.setattr(agents.telemetry, "_server_tps", lambda *_a, **_k: 17.25)
    monkeypatch.setattr(agents.probes, "_memory_snapshot", lambda: {"ram_available_mib": 8192})
    monkeypatch.setattr(agents.probes, "detect_gpu", lambda: None)
    parser = build_parser()

    rc = _run_dispatched(parser, parser.parse_args(["agents", "usage", "--json"]))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["action"] == "usage"
    assert payload["process"]["rss_mib"] == 4096
    assert payload["process"]["vram_mib"] == 5120
    assert payload["process"]["cpu_percent"] == 12.5
    assert payload["tokens_per_second"] == 17.25
    assert payload["ram"]["ram_available_mib"] == 8192


def test_agents_usage_degrades_when_server_stopped(monkeypatch, capsys):
    monkeypatch.setattr(agents.agent_service, "server_status", lambda: {"status": "stopped", "running": False})
    monkeypatch.setattr(agents.agent_service, "probe_protocols", lambda: {})
    parser = build_parser()

    rc = _run_dispatched(parser, parser.parse_args(["agents", "usage", "--json"]))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["server"]["running"] is False
    assert payload["process"]["available"] is False
    assert payload["tokens_per_second"] is None


def test_agents_usage_tolerates_missing_proc_and_gpu(monkeypatch, capsys):
    monkeypatch.setattr(
        agents.agent_service,
        "server_status",
        lambda: {
            "status": "ready",
            "running": True,
            "healthy": True,
            "pid": 4321,
            "alias": "a",
            "url": "http://127.0.0.1:8081",
        },
    )
    monkeypatch.setattr(agents, "_proc_field", lambda *_a, **_k: None)
    monkeypatch.setattr(agents, "_process_vram_mib", lambda *_a: None)

    class _Ps:
        def __init__(self, *_a, **_k):
            self.returncode = 1
            self.stdout = ""

    monkeypatch.setattr(agents.subprocess, "run", lambda *_a, **_k: _Ps())
    monkeypatch.setattr(agents.telemetry, "_server_tps", lambda *_a, **_k: None)
    parser = build_parser()

    rc = _run_dispatched(parser, parser.parse_args(["agents", "usage", "--json"]))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["process"]["available"] is True
    assert payload["process"]["rss_mib"] is None
    assert payload["process"]["cpu_percent"] is None
    assert payload["process"]["vram_mib"] is None


def test_agents_usage_ignores_main_config_error(monkeypatch, capsys):
    monkeypatch.setattr(state, "CONFIG_ERROR", "broken main config")
    monkeypatch.setattr(agents.agent_service, "server_status", lambda: {"status": "stopped", "running": False})
    parser = build_parser()

    rc = _run_dispatched(parser, parser.parse_args(["agents", "usage", "--json"]))

    assert rc == 0


def test_agents_token_ignores_unrelated_main_config_error(monkeypatch, capsys):
    monkeypatch.setattr(state, "CONFIG_ERROR", "broken main config")
    monkeypatch.setattr(agents.agent_service, "read_credential", lambda: "private-token")
    parser = build_parser()

    rc = _run_dispatched(parser, parser.parse_args(["agents", "token"]))

    assert rc == 0
    assert capsys.readouterr().out == "private-token\n"


def test_remote_serve_rejects_empty_api_key_file(tmp_path, capsys):
    key = tmp_path / "keys"
    key.write_text("# no keys\n\n", encoding="utf-8")
    key.chmod(0o600)
    parser = build_parser()

    rc = _run_dispatched(
        parser,
        parser.parse_args(
            [
                "serve",
                "missing.gguf",
                "--host",
                "0.0.0.0",
                "--api-key-file",
                str(key),
                "--dry-run",
                "--json",
            ]
        ),
    )

    assert rc == 2
    payload = json.loads(capsys.readouterr().out)
    assert "no usable API key" in payload["error"]["message"]
