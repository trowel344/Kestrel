from __future__ import annotations

import io

import pytest

from kestrel.nodes import (
    Node,
    NodeSecurityError,
    NodeStore,
    NodeValidationError,
    RpcProbeResult,
    SshTunnel,
    SshTunnelError,
)

HOST_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="


def _node(identity: str, **kwargs) -> Node:
    name = kwargs.pop("name", "worker")
    fields = {
        "ssh_host": "worker.example",
        "ssh_user": "kestrel",
        "ssh_identity_file": identity,
        "ssh_host_key": HOST_KEY,
        "remote_rpc_port": 50052,
    }
    fields.update(kwargs)
    return Node(
        name,
        "127.0.0.1:50052",
        4096,
        **fields,
    )


class FakeSocket:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeProcess:
    pid = 999999

    def __init__(self, return_code=None, stderr="") -> None:
        self.return_code = return_code
        self.stderr = io.StringIO(stderr)
        self.stdout = io.StringIO()
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.return_code

    def terminate(self):
        self.terminated = True
        self.return_code = -15

    def kill(self):
        self.killed = True
        self.return_code = -9

    def wait(self, timeout=None):
        return self.return_code


def _identity(tmp_path, mode=0o600):
    path = tmp_path / "id_ed25519"
    path.write_text("private key placeholder")
    path.chmod(mode)
    return path


def _probe(*_args, **_kwargs):
    return RpcProbeResult(True, True, (4, 0, 1), 1, device_memory=((1, 1),))


def test_tunnel_uses_pinned_host_key_and_injection_safe_argv(tmp_path):
    identity = _identity(tmp_path)
    process = FakeProcess()
    captured = {}

    def spawn(argv, **kwargs):
        captured.update(argv=argv, kwargs=kwargs)
        return process

    tunnel = SshTunnel(
        _node(str(identity)),
        ssh_binary="/usr/bin/ssh",
        process_factory=spawn,
        connect=lambda *_a, **_k: FakeSocket(),
        rpc_probe=_probe,
        listener_owner=lambda *_a: True,
    )
    with tunnel:
        assert tunnel.endpoint.startswith("127.0.0.1:")
        assert "shell" not in captured["kwargs"]
        argv = captured["argv"]
        assert "-F" in argv and argv[argv.index("-F") + 1] == "/dev/null"
        assert "-o" in argv and "StrictHostKeyChecking=yes" in argv
        assert "BatchMode=yes" in argv and "IdentitiesOnly=yes" in argv
        assert all(";" not in item and "&&" not in item for item in argv)
        known_hosts = tunnel.known_hosts_path
        assert known_hosts is not None and known_hosts.exists()
        line = known_hosts.read_text()
        assert line.startswith("kestrel-node-worker ssh-ed25519 ")
        assert "HostKeyAlias=kestrel-node-worker" in argv
    assert process.terminated
    assert known_hosts is not None and not known_hosts.exists()


def test_tunnel_rejects_weak_identity_permissions_before_spawn(tmp_path):
    identity = _identity(tmp_path, 0o644)
    spawned = []
    with pytest.raises(NodeSecurityError, match="identity file"):
        SshTunnel(
            _node(str(identity)),
            ssh_binary="/usr/bin/ssh",
            process_factory=lambda *args, **kwargs: spawned.append(args) or FakeProcess(),
            connect=lambda *_a, **_k: FakeSocket(),
            rpc_probe=_probe,
            listener_owner=lambda *_a: True,
        ).start()
    assert not spawned


def test_tunnel_host_key_mismatch_is_typed_and_cleans_files(tmp_path):
    identity = _identity(tmp_path)
    process = FakeProcess(255, "Warning: remote host identification has changed. Host key verification failed.")
    tunnel = SshTunnel(
        _node(str(identity)),
        ssh_binary="/usr/bin/ssh",
        process_factory=lambda *args, **kwargs: process,
        connect=lambda *_a, **_k: FakeSocket(),
        rpc_probe=_probe,
        listener_owner=lambda *_a: True,
    )
    with pytest.raises(NodeSecurityError, match="host-key"):
        tunnel.start()
    assert tunnel.process is None
    assert tunnel.known_hosts_path is None


def test_tunnel_early_death_and_timeout_are_cleaned(tmp_path):
    identity = _identity(tmp_path)
    dead = FakeProcess(7, "connection refused")
    tunnel = SshTunnel(
        _node(str(identity)),
        ssh_binary="/usr/bin/ssh",
        process_factory=lambda *args, **kwargs: dead,
        connect=lambda *_a, **_k: FakeSocket(),
        rpc_probe=_probe,
        listener_owner=lambda *_a: True,
    )
    with pytest.raises(SshTunnelError, match="exited"):
        tunnel.start()
    assert tunnel.process is None

    live = FakeProcess()
    timeout_tunnel = SshTunnel(
        _node(str(identity)),
        ssh_binary="/usr/bin/ssh",
        timeout=0.01,
        poll_interval=0.01,
        process_factory=lambda *args, **kwargs: live,
        connect=lambda *_a, **_k: (_ for _ in ()).throw(OSError("not ready")),
        rpc_probe=_probe,
        listener_owner=lambda *_a: True,
    )
    with pytest.raises(SshTunnelError, match="ready"):
        timeout_tunnel.start()
    assert live.terminated


def test_tunnel_errors_redact_identity_and_temp_known_hosts_paths(tmp_path):
    identity = _identity(tmp_path)
    process = FakeProcess(7, f'Load key "{identity}": bad permissions')
    tunnel = SshTunnel(
        _node(str(identity)),
        ssh_binary="/usr/bin/ssh",
        process_factory=lambda *args, **kwargs: process,
        connect=lambda *_a, **_k: FakeSocket(),
        rpc_probe=_probe,
        listener_owner=lambda *_a: True,
    )
    with pytest.raises(SshTunnelError) as raised:
        tunnel.start()
    assert str(identity) not in str(raised.value)


def test_managed_node_rejects_controls_and_ambiguous_key_material(tmp_path):
    identity = _identity(tmp_path)
    with pytest.raises(NodeValidationError):
        _node(str(identity), ssh_user="bad;user")
    with pytest.raises(NodeValidationError):
        _node(str(identity), ssh_host_key="*.example ssh-ed25519 AAAA")


def test_managed_node_public_summary_never_reveals_private_key_material(tmp_path):
    identity = _identity(tmp_path)
    item = _node(str(identity))
    summary = item.as_dict()
    assert summary["ssh_managed"] is True
    assert "ssh_identity_file" not in summary
    assert "ssh_host_key" not in summary


def test_managed_nodes_may_share_persisted_loopback_placeholder(tmp_path):
    identity = _identity(tmp_path)
    first = _node(str(identity), name="one")
    second = _node(str(identity), name="two", ssh_host="other.example")
    store = NodeStore(tmp_path / "nodes.json")
    store.save([first, second])
    assert {item.name for item in store.load()} == {"one", "two"}


def test_managed_inventory_rejects_world_readable_existing_state(tmp_path):
    identity = _identity(tmp_path)
    path = tmp_path / "nodes.json"
    path.write_text('{"schema_version": 2, "nodes": []}')
    path.chmod(0o644)
    store = NodeStore(path)
    with pytest.raises(NodeSecurityError, match="readable"):
        store.save([_node(str(identity))])
