"""Parser-to-dispatch contracts for the public experimental node commands."""

from __future__ import annotations

import importlib
import json

from kestrel import nodes
from kestrel.cli import nodes as node_commands
from kestrel.cli.parser import build_parser

cli_main = importlib.import_module("kestrel.cli.main")


def _dispatch(argv):
    parser = build_parser()
    return cli_main._run_dispatched(parser, parser.parse_args(argv))


def test_nodes_add_list_remove_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KESTREL_CONFIG", str(tmp_path / "config.toml"))
    assert (
        _dispatch(
            [
                "nodes",
                "add",
                "worker",
                "--endpoint",
                "127.0.0.1:50052",
                "--memory-mib",
                "8192",
                "--engine-commit",
                "a" * 40,
                "--json",
            ]
        )
        == 0
    )
    added = json.loads(capsys.readouterr().out)
    assert added["node"]["name"] == "worker"
    assert _dispatch(["nodes", "list", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert [entry["name"] for entry in listed["nodes"]] == ["worker"]
    assert _dispatch(["nodes", "remove", "worker", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "removed"


def test_nodes_rejects_direct_lan_by_default(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KESTREL_CONFIG", str(tmp_path / "config.toml"))
    status = _dispatch(
        [
            "nodes",
            "add",
            "lan",
            "--endpoint",
            "10.0.0.2:50052",
            "--memory-mib",
            "8192",
            "--engine-commit",
            "a" * 40,
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert status != 0
    assert payload["error"]["code"] == "node_security_error"


def test_nodes_plan_empty_inventory_is_typed_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KESTREL_CONFIG", str(tmp_path / "config.toml"))
    monkeypatch.setattr(node_commands, "_local_provenance", lambda: ([1024], "a" * 40))
    status = _dispatch(["nodes", "plan", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert status != 0
    assert payload["error"]["code"] == "node_planning_error"


def test_nodes_doctor_protocol_failure_is_nonzero_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KESTREL_CONFIG", str(tmp_path / "config.toml"))
    nodes.NodeStore().save([nodes.Node("worker", "127.0.0.1:50052", 8192, engine_commit="a" * 40)])
    monkeypatch.setattr(
        nodes,
        "probe_rpc",
        lambda *_args, **_kwargs: nodes.RpcProbeResult(True, False, error="not an RPC worker"),
    )
    status = _dispatch(["nodes", "doctor", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert status == 1
    assert payload["status"] == "failed"
    assert payload["checks"][0]["rpc_protocol"] is False


def test_nodes_plan_reports_device_order_and_coarse_fit(tmp_path, monkeypatch, capsys):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF" + b"x" * 1024)
    monkeypatch.setattr(node_commands, "_local_provenance", lambda: ([1024], "a" * 40))
    monkeypatch.setattr(
        nodes,
        "resolve_node_plan",
        lambda **_kwargs: {
            "status": "planned",
            "nodes": [{"name": "worker", "endpoint": "127.0.0.1:50052"}],
            "rpc_endpoints": ["127.0.0.1:50052"],
            "tensor_split": "0.25,0.75",
            "device_order": ["local:0", "worker:0"],
            "capacities_mib": [1024, 3072],
            "total_capacity_mib": 4096,
            "total_devices": 2,
            "probe_evidence": {"worker": {"rpc_protocol": True}},
        },
    )
    status = _dispatch(["nodes", "plan", str(model), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert status == 0
    assert payload["device_order"] == ["local:0", "worker:0"]
    assert payload["coarse_accelerator_fit"] is True
    assert "weights-only" in payload["fit_scope"]


def test_nodes_plan_single_node_does_not_implicitly_select_all(monkeypatch, capsys):
    captured = {}
    monkeypatch.setattr(node_commands, "_local_provenance", lambda: ([1024], "a" * 40))

    def resolve(**kwargs):
        captured.update(kwargs)
        return {
            "status": "planned",
            "nodes": [{"name": "only", "endpoint": "127.0.0.1:50052"}],
            "rpc_endpoints": ["127.0.0.1:50052"],
            "tensor_split": "0.5,0.5",
            "device_order": ["local:0", "only:0"],
            "capacities_mib": [1024, 1024],
            "total_capacity_mib": 2048,
            "total_devices": 2,
            "probe_evidence": {},
        }

    monkeypatch.setattr(nodes, "resolve_node_plan", resolve)
    assert _dispatch(["nodes", "plan", "--node", "only", "--json"]) == 0
    capsys.readouterr()
    assert captured["names"] == ["only"]
    assert captured["selector"] is None


def test_nodes_parser_rejects_bad_name_as_one_json_document(capsys):
    parser = build_parser()
    try:
        parser.parse_args(["nodes", "remove", "../bad", "--json"])
    except SystemExit as exc:
        assert exc.code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "invalid_input"
