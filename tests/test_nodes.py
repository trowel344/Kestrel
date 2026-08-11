from __future__ import annotations

import json
import socket
import struct

import pytest

from kestrel.nodes import (
    EngineProvenance,
    Node,
    NodePlanningError,
    NodeSecurityError,
    NodeStateError,
    NodeStore,
    NodeValidationError,
    parse_endpoint,
    plan_placement,
    probe_reachability,
    probe_rpc,
    resolve_node_plan,
)

HASH = "a" * 64


def node(name="n", endpoint="127.0.0.1:50052", memory=100, **kwargs):
    return Node(name, endpoint, memory, **kwargs)


def test_endpoint_forms_and_canonical_ipv6():
    assert parse_endpoint("127.0.0.1:1").address == "127.0.0.1:1"
    assert parse_endpoint("worker.local:50052").address == "worker.local:50052"
    assert parse_endpoint("[::1]:50052").address == "[::1]:50052"
    with pytest.raises(NodeValidationError):
        parse_endpoint("2001:db8::1:50052")
    with pytest.raises(NodeValidationError):
        parse_endpoint("localhost")


def test_store_roundtrip_and_atomic_writer(monkeypatch, tmp_path):
    store = NodeStore(tmp_path / "nodes.json")
    item = node("z", model_cache_hashes=(HASH,), ram_mib=512)
    calls = []
    monkeypatch.setattr("kestrel.nodes.write_atomic", lambda *args, **kwargs: calls.append((args, kwargs)) or args[0])
    store.save([item])
    assert calls
    monkeypatch.undo()
    store.save([item])
    assert store.load() == (item,)
    assert store.get("z") == item
    assert store.list() == (item,)


def test_store_rejects_corrupt_and_unknown_state_without_replacing_it(tmp_path):
    path = tmp_path / "nodes.json"
    path.write_text(json.dumps({"schema_version": 1, "nodes": [{"name": "x"}]}))
    original = path.read_bytes()
    with pytest.raises(NodeStateError):
        NodeStore(path).load()
    assert path.read_bytes() == original

    path.write_text(json.dumps({"schema_version": 1, "nodes": [], "future": True}))
    with pytest.raises(NodeStateError):
        NodeStore(path).load()


def test_store_duplicate_names_and_endpoints_are_rejected(tmp_path):
    store = NodeStore(tmp_path / "nodes.json")
    with pytest.raises(NodeValidationError, match="duplicate names"):
        store.save([node("a"), node("a", "127.0.0.1:50053")])
    with pytest.raises(NodeValidationError, match="duplicate endpoints"):
        store.save([node("a"), node("b")])


def test_store_follows_symlink_referent_atomically(tmp_path):
    target = tmp_path / "real.json"
    link = tmp_path / "nodes.json"
    target.write_text(json.dumps({"schema_version": 1, "nodes": []}))
    link.symlink_to(target)
    NodeStore(link).save([node()])
    assert link.is_symlink()
    assert NodeStore(link).load() == (node(),)


def test_security_defaults_to_loopback_and_explicit_opt_in(tmp_path):
    remote = node(endpoint="worker.example:50052")
    with pytest.raises(NodeSecurityError):
        NodeStore(tmp_path / "nodes.json").save([remote])
    NodeStore(tmp_path / "nodes.json", allow_insecure_direct_rpc=True).save([remote])
    assert NodeStore(tmp_path / "nodes.json", allow_insecure_direct_rpc=True).load() == (remote,)


def test_reachability_is_tcp_only_and_bounded(monkeypatch):
    calls = []

    class Socket:
        def close(self):
            calls.append("close")

    monkeypatch.setattr(
        socket, "create_connection", lambda address, timeout: calls.append((address, timeout)) or Socket()
    )
    assert probe_reachability("localhost:50052", timeout=999) is True
    assert calls[0][1] <= 10
    assert calls[-1] == "close"


def test_rpc_probe_distinguishes_protocol_and_reports_version_and_devices(monkeypatch):
    response = struct.pack("<Q", 28) + bytes((4, 0, 1, 0)) + bytes(24)
    response += struct.pack("<Q", 4) + struct.pack("<I", 2)
    response += struct.pack("<Q", 16) + struct.pack("<QQ", 100, 200)
    response += struct.pack("<Q", 16) + struct.pack("<QQ", 300, 400)

    class Socket:
        def __init__(self):
            self.sent = []

        def sendall(self, value):
            self.sent.append(value)

        def recv(self, size):
            value, self.data = self.data[:size], self.data[size:]
            return value

        def close(self):
            pass

    sock = Socket()
    sock.data = response
    monkeypatch.setattr(socket, "create_connection", lambda *args, **kwargs: sock)
    result = probe_rpc("localhost:50052")
    assert result.tcp_reachable and result.rpc_protocol
    assert result.version == (4, 0, 1)
    assert result.device_count == 2
    assert result.device_memory == ((100, 200), (300, 400))
    assert result.protocol_compatible
    assert result.usable


def test_rpc_probe_marks_tcp_service_without_protocol(monkeypatch):
    class Socket:
        def sendall(self, value):
            pass

        def recv(self, size):
            return b"http"

        def close(self):
            pass

    monkeypatch.setattr(socket, "create_connection", lambda *args, **kwargs: Socket())
    result = probe_rpc("localhost:50052")
    assert result.tcp_reachable and not result.rpc_protocol


@pytest.mark.parametrize("version", [(3, 0, 0), (4, 1, 0)])
def test_rpc_probe_marks_incompatible_protocol_version(monkeypatch, version):
    response = struct.pack("<Q", 28) + bytes((*version, 0)) + bytes(24)
    response += struct.pack("<Q", 4) + struct.pack("<I", 1)
    response += struct.pack("<Q", 16) + struct.pack("<QQ", 100, 200)

    class Socket:
        def __init__(self):
            self.data = response

        def sendall(self, value):
            pass

        def recv(self, size):
            value, self.data = self.data[:size], self.data[size:]
            return value

        def close(self):
            pass

    monkeypatch.setattr(socket, "create_connection", lambda *args, **kwargs: Socket())
    result = probe_rpc("localhost:50052")
    assert result.rpc_protocol and not result.protocol_compatible and not result.usable


def test_rpc_probe_zero_devices_is_not_usable(monkeypatch):
    response = struct.pack("<Q", 28) + bytes((4, 0, 0, 0)) + bytes(24)
    response += struct.pack("<Q", 4) + struct.pack("<I", 0)

    class Socket:
        def __init__(self):
            self.data = response

        def sendall(self, value):
            pass

        def recv(self, size):
            value, self.data = self.data[:size], self.data[size:]
            return value

        def close(self):
            pass

    monkeypatch.setattr(socket, "create_connection", lambda *args, **kwargs: Socket())
    result = probe_rpc("localhost:50052")
    assert result.protocol_compatible and result.device_count == 0 and not result.usable


def test_rpc_probe_rejects_unbounded_device_count(monkeypatch):
    response = struct.pack("<Q", 28) + bytes((4, 0, 1, 0)) + bytes(24)
    response += struct.pack("<Q", 4) + struct.pack("<I", 2**32 - 1)

    class Socket:
        def __init__(self):
            self.data = response

        def sendall(self, value):
            pass

        def recv(self, size):
            value, self.data = self.data[:size], self.data[size:]
            return value

        def close(self):
            pass

    monkeypatch.setattr(socket, "create_connection", lambda *args, **kwargs: Socket())
    result = probe_rpc("localhost:50052")
    assert result.tcp_reachable and not result.usable
    assert "too many devices" in result.error


def test_planner_is_deterministic_and_excludes_disabled_unreachable_and_ram():
    nodes = [
        node("z", "127.0.0.1:50053", 300, ram_mib=9999),
        node("disabled", "127.0.0.1:50054", 900, enabled=False),
        node("unreachable", "127.0.0.1:50055", 900),
    ]
    result = plan_placement(100, nodes, reachable={"z": True, "unreachable": False})
    assert result.rpc_endpoints == ("127.0.0.1:50053",)
    assert result.capacities_mib == (100, 300)
    assert result.tensor_split == (0.25, 0.75)
    assert result.total_capacity_mib == 400
    assert result.tensor_split_arg == "0.25,0.75"


def test_planner_rejects_zero_capacity_and_provenance_mismatch():
    with pytest.raises(NodePlanningError, match="zero"):
        plan_placement(0, [node(memory=0)], reachable={"n": True})
    same = node("same", engine_version="1", engine_commit="abcdef1")
    with pytest.raises(NodePlanningError, match="provenance"):
        plan_placement(
            10,
            [same],
            reachable={"same": True},
            expected_engine=EngineProvenance("1", "def"),
            local_engine_version="1",
            local_engine_commit="defaced",
        )


def test_planner_rejects_duplicate_endpoints_and_unknown_provenance():
    with pytest.raises(NodeValidationError, match="duplicate endpoints"):
        plan_placement(10, [node("a"), node("b")], reachable={"a": True, "b": True})
    with pytest.raises(NodePlanningError, match="local engine"):
        plan_placement(10, [], expected_engine_commit="abcdef1", local_engine_commit=None)


def test_resolver_uses_rpc_protocol_and_per_device_live_memory(monkeypatch, tmp_path):
    remote = node("remote", "127.0.0.1:50060", 9999)
    NodeStore(tmp_path / "nodes.json").save([remote])
    from kestrel.nodes import RpcProbeResult

    monkeypatch.setattr(
        "kestrel.nodes.probe_rpc",
        lambda item, **kwargs: RpcProbeResult(
            True,
            True,
            (4, 0, 0),
            2,
            device_memory=((100 * 2**20, 200 * 2**20), (50 * 2**20, 100 * 2**20)),
        ),
    )
    result = resolve_node_plan(
        names=["remote"],
        store=NodeStore(tmp_path / "nodes.json"),
        local_free_vram_mib=[25, 25],
    )
    assert result["rpc_endpoints"] == ["127.0.0.1:50060"]
    assert result["total_devices"] == 4
    assert result["tensor_split_ratios"] == (0.125, 0.125, 0.5, 0.25)


def test_resolver_rejects_empty_inventory(tmp_path):
    with pytest.raises(NodePlanningError, match="no RPC nodes"):
        resolve_node_plan(store=NodeStore(tmp_path / "nodes.json"), selector="all", local_free_vram_mib=[100])


def test_planner_ignores_unspecified_expected_version():
    remote = node("remote", engine_version="v1", engine_commit="abcdef1")
    result = plan_placement(
        [100],
        [remote],
        reachable={"remote": True},
        expected_engine={"commit": "abcdef1", "version": None},
        local_engine_commit="abcdef1",
    )
    assert result.reachable_nodes == ("remote",)
