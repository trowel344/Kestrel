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
