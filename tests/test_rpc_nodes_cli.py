"""Focused contracts for the experimental llama.cpp RPC node surface."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kestrel import engine
from kestrel.backends.llama_cpp import LlamaCppBackend, LlamaCppCapabilities
from kestrel.cli import parser, run, runtime
from kestrel.errors import BackendError, InputError


def test_parser_accepts_repeatable_nodes_and_all_selector():
    args = parser.build_parser().parse_args(
        ["run", "model.gguf", "--node", "desktop", "--node", "laptop", "--nodes", "all", "--allow-insecure-rpc"]
    )
    assert args.node == ["desktop", "laptop"]
    assert args.nodes == "all"
    assert args.allow_insecure_rpc is True


def test_backend_emits_rpc_as_single_endpoint_list(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    backend = LlamaCppBackend(str(model), rpc_endpoints=["127.0.0.1:50052", "localhost:50053"])
    backend._capabilities = LlamaCppCapabilities(help_text="--rpc\n--mmap\n")
    command = backend._base_cmd()
    assert command[command.index("--rpc") + 1] == "127.0.0.1:50052,localhost:50053"


def test_backend_rejects_rpc_when_engine_lacks_capability(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    backend = LlamaCppBackend(str(model), rpc_endpoints=["127.0.0.1:50052"])
    backend._capabilities = LlamaCppCapabilities(help_text="--mmap\n")
    with pytest.raises(BackendError, match="does not support RPC"):
        backend._base_cmd()


def test_node_plan_requires_loopback_or_explicit_override(monkeypatch):
    import kestrel

    class FakeNodes:
        class NodeStore:
            def __init__(self, **_kwargs):
                pass

            def load(self):
                return ()

        SshTunnel = object

        @staticmethod
        def resolve_node_plan(**_kwargs):
            return {"nodes": [{"name": "lan", "rpc_endpoint": "10.0.0.2:50052"}]}

    monkeypatch.setattr(kestrel, "nodes", FakeNodes, raising=False)
    args = SimpleNamespace(node=["lan"], nodes=None, allow_insecure_rpc=False)
    with pytest.raises(InputError, match="not loopback"):
        runtime._resolve_node_plan(args)
    args.allow_insecure_rpc = True
    plan = runtime._resolve_node_plan(args)
    assert plan["rpc_endpoints"] == ["10.0.0.2:50052"]


def test_engine_selects_legacy_rpc_target_without_substring_match(tmp_path):
    source = tmp_path / "engine"
    rpc = source / "tools" / "rpc"
    rpc.mkdir(parents=True)
    (rpc / "CMakeLists.txt").write_text("set(TARGET ggml-rpc-server)\n")
    assert engine._rpc_target_for_source(str(source)) == "ggml-rpc-server"


@pytest.mark.parametrize(("handler", "inner"), [(run.cmd_run, "_cmd_run_live"), (run.cmd_serve, "_cmd_serve_live")])
def test_managed_tunnels_close_when_run_or_serve_is_interrupted(monkeypatch, handler, inner):
    args = SimpleNamespace()
    closed = []
    monkeypatch.setattr(run, inner, lambda _args: (_ for _ in ()).throw(KeyboardInterrupt()))
    monkeypatch.setattr(runtime, "_close_node_tunnels", lambda value: closed.append(value))
    with pytest.raises(KeyboardInterrupt):
        handler(args)
    assert closed == [args]
