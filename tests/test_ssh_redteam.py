"""Adversarial regressions for managed SSH node transport."""

from __future__ import annotations

import io

import pytest

from kestrel.nodes import Node, NodeValidationError, RpcProbeResult, SshTunnel, SshTunnelError

HOST_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="


class _Socket:
    def close(self) -> None:
        pass


class _Process:
    pid = 999_999

    def __init__(self, stderr: str = "") -> None:
        self.stderr = io.StringIO(stderr)
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


def _node(identity: str, **overrides) -> Node:
    fields = {
        "ssh_host": "worker.example",
        "ssh_user": "kestrel",
        "ssh_identity_file": identity,
        "ssh_host_key": HOST_KEY,
        "remote_rpc_port": 50052,
    }
    fields.update(overrides)
    return Node("worker", "127.0.0.1:50052", 1024, **fields)


def _identity(tmp_path):
    path = tmp_path / "identity"
    path.write_text("placeholder")
    path.chmod(0o600)
    return path


def _usable_probe(*_args, **_kwargs):
    return RpcProbeResult(True, True, (4, 0, 1), 1, device_memory=((1, 1),))


@pytest.mark.parametrize(
    ("field", "payload"),
    [
        ("ssh_host", "-oProxyCommand=touch${IFS}/tmp/pwned"),
        ("ssh_host", "worker\nProxyCommand=evil"),
        ("ssh_user", "-oForwardAgent=yes"),
        ("ssh_user", "user;touch-pwned"),
        ("ssh_identity_file", "/tmp/key\n-oProxyCommand=evil"),
        ("ssh_host_key", "ssh-ed25519 AAAA\nattacker ssh-rsa AAAA"),
    ],
)
def test_managed_node_rejects_ssh_argument_and_line_injection(tmp_path, field, payload):
    with pytest.raises(NodeValidationError):
        _node(str(_identity(tmp_path)), **{field: payload})


def test_fake_rpc_listener_cannot_win_local_port_race(tmp_path):
    """TCP plus a spoofed RPC handshake is insufficient without SSH FD ownership."""

    process = _Process()
    tunnel = SshTunnel(
        _node(str(_identity(tmp_path))),
        ssh_binary="/usr/bin/ssh",
        timeout=0.01,
        poll_interval=0.005,
        process_factory=lambda *_a, **_k: process,
        connect=lambda *_a, **_k: _Socket(),
        rpc_probe=_usable_probe,
        listener_owner=lambda *_a: False,
    )
    with pytest.raises(SshTunnelError, match="ready"):
        tunnel.start()
    assert process.returncode == -15
    assert tunnel.known_hosts_path is None


def test_ssh_failure_redacts_identity_and_pin_paths(tmp_path):
    identity = _identity(tmp_path)
    process = _Process()

    def spawn(*_args, **_kwargs):
        process.returncode = 255
        # This is representative of OpenSSH key-load errors and terminal
        # control injection from an untrusted peer/banner.
        process.stderr = io.StringIO(f'Load key "{identity}": invalid format \x1b[31m')
        return process

    tunnel = SshTunnel(
        _node(str(identity)),
        ssh_binary="/usr/bin/ssh",
        process_factory=spawn,
        connect=lambda *_a, **_k: _Socket(),
        rpc_probe=_usable_probe,
        listener_owner=lambda *_a: True,
    )
    with pytest.raises(SshTunnelError) as caught:
        tunnel.start()
    message = str(caught.value)
    assert str(identity) not in message
    assert "\x1b" not in message
    assert "<redacted>" in message


def test_effective_ssh_argv_disables_ambient_credentials_and_config(tmp_path):
    identity = _identity(tmp_path)
    process = _Process()
    captured = {}

    def spawn(argv, **kwargs):
        captured["argv"] = argv
        return process

    tunnel = SshTunnel(
        _node(str(identity)),
        ssh_binary="/usr/bin/ssh",
        process_factory=spawn,
        connect=lambda *_a, **_k: _Socket(),
        rpc_probe=_usable_probe,
        listener_owner=lambda *_a: True,
    )
    with tunnel:
        argv = captured["argv"]
        required = {
            "BatchMode=yes",
            "IdentitiesOnly=yes",
            "IdentityAgent=none",
            "AddKeysToAgent=no",
            "StrictHostKeyChecking=yes",
            "GlobalKnownHostsFile=/dev/null",
            "ExitOnForwardFailure=yes",
            "ForwardAgent=no",
            "ForwardX11=no",
            "PermitLocalCommand=no",
            "ProxyCommand=none",
            "PasswordAuthentication=no",
            "KbdInteractiveAuthentication=no",
        }
        assert required.issubset(argv)
        assert argv[argv.index("-F") + 1] == "/dev/null"
        assert sum(value.startswith("127.0.0.1:") for value in argv) == 1
