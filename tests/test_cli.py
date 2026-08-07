import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import kestrel.cli as cli

sys.path.insert(0, str(Path(__file__).parent))

from gguf_fixture import write_gguf  # noqa: E402


def _sized_gguf(tmp_path, name, gib, **kwargs):
    p = write_gguf(tmp_path / name, **kwargs)
    if gib:
        with open(p, "r+b") as f:
            f.seek(int(gib * 1024**3) - 1)
            f.write(b"\x00")
    return p


def _no_swap(monkeypatch):
    monkeypatch.setattr(cli, "_available_ram_mib", lambda: 64 * 1024)
    monkeypatch.setattr(
        cli,
        "_memory_snapshot",
        lambda: {"swap_total_mib": 0, "swap_used_mib": 0},
    )


def test_safetensors_info_reads_config(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3MoeForCausalLM"],
                "hidden_size": 3072,
                "num_hidden_layers": 48,
                "intermediate_size": 1024,
            }
        )
    )
    (tmp_path / "model.safetensors").write_bytes(b"\x00" * 10)
    info = cli._safetensors_info(str(tmp_path))
    assert info["type"] == "safetensors"
    assert info["size_bytes"] == 10
    assert info["config"]["architectures"] == ["Qwen3MoeForCausalLM"]
    assert info["estimated_params_b"] > 0


def test_safetensors_info_missing_config(tmp_path):
    info = cli._safetensors_info(str(tmp_path))
    assert "error" in info["config"]


def test_safetensors_info_bad_config(tmp_path):
    (tmp_path / "config.json").write_text("not json")
    info = cli._safetensors_info(str(tmp_path))
    assert "error" in info["config"]


def test_safetensors_info_missing_arch_fields():
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    (tmp / "config.json").write_text(json.dumps({"model_type": "bert"}))
    info = cli._safetensors_info(str(tmp))
    assert "estimated_params_b" not in info


def test_cached_gguf_path_suffix():
    assert cli._cached_gguf_path("/tmp/model.gguf") == "/tmp/model.gguf.gguf"
    assert cli._cached_gguf_path("/tmp/moe/") == "/tmp/moe.gguf"


def test_kv_cache_bytes_uses_gqa_value_dim(tmp_path):
    """GQA models store n_kv_heads*head_dim values per layer, not hidden."""
    p = _sized_gguf(
        tmp_path,
        "m.gguf",
        0,
        architecture="qwen35moe",
        n_layer=48,
        hidden=2048,
        n_heads=16,
        n_kv_heads=8,
        head_dim=128,
    )
    bytes_per_token = cli._kv_cache_bytes_per_token({"type": "gguf", "path": str(p)})
    # 2 * layers * (8*128) * 1.1; the hidden-size fallback would be 2x this.
    assert bytes_per_token == pytest.approx(2 * 48 * (8 * 128) * 1.1)


def test_select_context_gpu_resident_accounts_for_kv_cache(tmp_path, monkeypatch):
    """A 17 GiB MoE on a 24 GiB card fits by weight, but a 32K context would
    push weights+KV past free VRAM; the planner must cap the context."""
    _no_swap(monkeypatch)
    p = _sized_gguf(
        tmp_path,
        "moe.gguf",
        17,
        architecture="qwen35moe",
        n_layer=48,
        n_exp=128,
        n_used=8,
        hidden=2048,
        n_ff=1024,
        n_heads=16,
        n_kv_heads=8,
        head_dim=128,
    )
    ctx, reason, overcommitted = cli._select_context_size(
        {"type": "gguf", "path": str(p)}, {"vram_free_mb": 22000}
    )
    assert not overcommitted
    assert ctx < 32768
    kv_per_token = cli._kv_cache_bytes_per_token({"type": "gguf", "path": str(p)})
    weights_budget = (22000 - 1024 - 1536) * 1024**2  # free - fit margin - overhead
    assert 17 * 1024**3 + ctx * kv_per_token <= weights_budget


def test_select_context_big_gpu_allows_large_context(tmp_path, monkeypatch):
    """A small model on a large card still gets a large context."""
    _no_swap(monkeypatch)
    p = _sized_gguf(
        tmp_path,
        "small.gguf",
        4.5,
        architecture="llama",
        n_layer=32,
        n_exp=0,
        n_used=0,
        hidden=4096,
        n_ff=14336,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
    )
    ctx, reason, overcommitted = cli._select_context_size(
        {"type": "gguf", "path": str(p)}, {"vram_free_mb": 22000}
    )
    assert not overcommitted
    assert ctx >= 16384


def test_select_context_oversized_model_is_overcommit(tmp_path, monkeypatch):
    """A model larger than RAM+swap picks the minimum context and flags it."""
    _no_swap(monkeypatch)
    p = _sized_gguf(
        tmp_path,
        "big.gguf",
        90,
        architecture="qwen35moe",
        n_layer=48,
        n_exp=256,
        n_used=8,
        hidden=3072,
        n_ff=1024,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
    )
    ctx, reason, overcommitted = cli._select_context_size(
        {"type": "gguf", "path": str(p)}, {"vram_free_mb": 6000}
    )
    assert overcommitted
    assert ctx == 512


def test_configure_backend_maps_config_and_args(monkeypatch, tmp_path):
    model = write_gguf(tmp_path / "m.gguf", n_layer=48)
    monkeypatch.setattr(cli, "detect_gpu", lambda: None)
    args = SimpleNamespace(
        batch_size=None,
        ubatch_size=None,
        mtp_tokens=4,
        fit_target=None,
        kv_cache_type="q4_1",
        no_mmap=False,
        mlock=True,
        direct_io=True,
        tensor_split="60,40",
        extra=["--temp 0.4"],
        threads=None,
        no_mtp=False,
        _gpu=None,
    )
    config = {
        "use_mtp": True,
        "gpu_layers": "auto",
        "context_size": 4096,
        "batch_size": 128,
        "ubatch_size": 64,
        "cpu_moe": True,
        "fit": True,
        "fit_target_mib": 512,
        "cache_type_k": "q8_0",
        "cache_type_v": "q8_0",
        "threads": 8,
        "moe_cache": "on",
        "moe_cache_budget_mib": 2048,
    }
    backend = cli._configure_backend(
        {"path": str(model)}, config, args
    )
    assert backend.n_gpu_layers == "auto"
    assert backend.n_ctx == 4096
    assert backend.spec_type == "mtp"
    assert backend.spec_draft_n == 4
    assert backend.cpu_moe is True
    assert backend.fit is True
    assert backend.fit_target_mib == 512
    assert backend.use_mlock is True
    assert backend.direct_io is True
    assert backend.tensor_split == "60,40"
    assert backend.extra_args == ["--temp", "0.4"]
    assert backend.moe_cache == "2048"


class _StubBackend:
    def __init__(self):
        self.server_kwargs = None
        self.interactive_called = False

    def build_server_cmd(self, **kwargs):
        self.server_kwargs = kwargs
        return ["llama-server", "--sentinel"]

    def _build_interactive_cmd(self):
        self.interactive_called = True
        return ["llama-cli", "--sentinel"]

    def capabilities(self):
        return SimpleNamespace(version="9.9.9")


def test_build_server_cmd_passes_serve_args(monkeypatch, tmp_path):
    stub = _StubBackend()
    monkeypatch.setattr(cli, "_configure_backend", lambda *a, **k: stub)
    args = SimpleNamespace(host="0.0.0.0", port=9000, alias="m", embeddings=True)
    cmd = cli._build_server_cmd({}, {}, args)
    assert cmd == ["llama-server", "--sentinel"]
    assert stub.server_kwargs == {
        "host": "0.0.0.0",
        "port": 9000,
        "alias": "m",
        "embeddings": True,
    }


def test_build_server_cmd_default_serving_defaults(monkeypatch, tmp_path):
    stub = _StubBackend()
    monkeypatch.setattr(cli, "_configure_backend", lambda *a, **k: stub)
    cli._build_server_cmd({}, {}, None)
    assert stub.server_kwargs["host"] == "127.0.0.1"
    assert stub.server_kwargs["port"] == 8080
    assert stub.server_kwargs["embeddings"] is False


def test_human_stream_routes_by_json_flag():
    import sys

    assert cli._human_stream(SimpleNamespace(json=True)) is sys.stderr
    assert cli._human_stream(SimpleNamespace(json=False)) is sys.stdout
    assert cli._human_stream(SimpleNamespace()) is sys.stdout


def test_finish_json_emits_only_under_json_flag(capsys):
    plain = SimpleNamespace(json=False)
    assert cli._finish_json(plain, {"exit_code": 3, "tag": "x"}) == 3
    assert capsys.readouterr().out == ""

    js = SimpleNamespace(json=True)
    assert cli._finish_json(js, {"exit_code": 2, "model": "m"}) == 2
    assert json.loads(capsys.readouterr().out) == {"exit_code": 2, "model": "m"}


class _GenStub:
    def __init__(self, output="hello from kestrel", error=None):
        self.output = output
        self.error = error
        self.last_metrics = SimpleNamespace(
            returncode=0,
            elapsed_seconds=1.5,
            prompt_tokens=10,
            output_tokens=20,
            prompt_tokens_per_second=100.0,
            output_tokens_per_second=40.0,
        )
        self.prompt = None
        self.max_tokens = None

    def _build_cmd(self, prompt, max_tokens):
        self.prompt = prompt
        self.max_tokens = max_tokens
        return ["llama-cli", "-p", prompt]

    def generate(self, prompt, max_tokens):
        self.prompt = prompt
        self.max_tokens = max_tokens
        if self.error:
            raise RuntimeError(self.error)
        return self.output


def test_oneshot_run_json_emits_structured_result(capsys):
    stub = _GenStub()
    args = SimpleNamespace(
        json=True, model="m.gguf", prompt="hi", max_tokens=64, dry_run=False
    )
    rc = cli._oneshot_run(stub, ["llama-cli"], args)
    assert stub.prompt == "hi"
    assert stub.max_tokens == 64
    assert rc == 0
    out = capsys.readouterr()
    payload = json.loads(out.out.splitlines()[-1])
    assert payload["status"] == "ok"
    assert payload["model"] == "m.gguf"
    assert payload["output"] == "hello from kestrel"
    assert payload["output_tokens"] == 20
    assert payload["output_tokens_per_second"] == 40.0
    assert payload["command"] == ["llama-cli", "-p", "hi"]
    # Human mirror goes to stderr, keeping stdout a single JSON document.
    assert "hello from kestrel" in out.err
    assert out.out.count("\n") == 1


def test_oneshot_run_plain_shows_output_on_stdout(capsys):
    stub = _GenStub()
    args = SimpleNamespace(
        json=False, model="m.gguf", prompt="hi", max_tokens=8, dry_run=False
    )
    rc = cli._oneshot_run(stub, ["llama-cli"], args)
    assert rc == 0
    out = capsys.readouterr()
    assert out.out == "hello from kestrel\n"
    assert out.err == ""


def test_oneshot_run_generation_error_emits_error_json(capsys):
    stub = _GenStub(error="llama.cpp failed with exit 1:\nboom")
    args = SimpleNamespace(
        json=True, model="m.gguf", prompt="hi", max_tokens=8, dry_run=False
    )
    rc = cli._oneshot_run(stub, ["llama-cli"], args)
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert "boom" in payload["error"]


def test_wait_ready_returns_true_for_healthy_server():
    import http.server
    import socketserver
    import threading

    # Use a local HTTP server that is ready immediately.
    handler = type(
        "ReadyHandler",
        (http.server.BaseHTTPRequestHandler,),
        {"do_GET": lambda self: (self.send_response(200), self.end_headers(), self.wfile.write(b"ok"))},
    )

    with socketserver.TCPServer(("127.0.0.1", 0), handler) as srv:
        port = srv.server_address[1]
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        try:
            assert cli._wait_ready("127.0.0.1", port, timeout=5.0, interval=0.05) is True
        finally:
            srv.shutdown()
            thread.join()


def test_wait_ready_times_out_when_not_ready():
    import http.server
    import socketserver
    import threading
    import time

    handler = type(
        "SlowHandler",
        (http.server.BaseHTTPRequestHandler,),
        {
            "do_GET": lambda self: (
                time.sleep(2),
                self.send_response(200),
                self.end_headers(),
                self.wfile.write(b"ok"),
            )
        },
    )

    with socketserver.TCPServer(("127.0.0.1", 0), handler) as srv:
        port = srv.server_address[1]
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        try:
            started = time.perf_counter()
            assert cli._wait_ready("127.0.0.1", port, timeout=0.4, interval=0.05) is False
            assert time.perf_counter() - started < 2.0
        finally:
            srv.shutdown()
            thread.join()


def _run_args(model, **overrides):
    base = dict(
        model=model,
        dry_run=True,
        json=False,
        prompt=None,
        max_tokens=256,
        no_convert=True,
        moe_hot_model=None,
        moe_cold_model=None,
        fit_target=None,
        threads=None,
        batch_size=None,
        ubatch_size=None,
        kv_cache_type="q8_0",
        gpu_layers="auto",
        cpu_moe="auto",
        direct_io=False,
        warm_cache=False,
        no_oom_retry=False,
        no_mtp=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_cmd_run_dry_run_json_keeps_stdout_clean(capsys, monkeypatch, tmp_path):
    model = write_gguf(tmp_path / "m.gguf", n_layer=48)
    config = {
        "model_size_gib": 3.0,
        "gpu_layers": "all",
        "fit_target_mib": 512,
        "cpu_moe": False,
        "moe_cache": "off",
        "moe_cache_budget_mib": 0,
        "threads": 8,
        "context_size": 4096,
        "context_reason": "test",
        "batch_size": 128,
        "ubatch_size": 64,
        "predicted_decode_tps": 0,
        "prediction_confidence": "low",
        "memory_overcommit": False,
        "has_mtp": False,
        "use_mtp": False,
    }
    monkeypatch.setattr(cli, "detect_gpu", lambda: None)
    monkeypatch.setattr(
        cli,
        "estimate_config",
        lambda model_info, gpu_info, args: dict(config),
    )
    monkeypatch.setattr(cli, "_configure_backend", lambda *a, **k: _StubBackend())
    args = _run_args(str(model), json=True)
    rc = cli.cmd_run(args)
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip())
    assert payload["dry_run"] is True
    assert payload["model"] == str(model)
    assert payload["command"] == ["llama-cli", "--sentinel"]
    # Human plan/command must live on stderr; stdout is one JSON document.
    assert "Runtime plan" in captured.err
    assert captured.out.count("\n") == 1


def test_resolve_ollama_native_returns_local_blob(monkeypatch):
    from pathlib import Path

    blob = Path("/home/u/.ollama/models/blobs/sha256-abc")
    monkeypatch.setattr(
        "kestrel.model_store.resolve_ollama_blob", lambda name: blob
    )
    assert cli._resolve_ollama_native("qwen3.6:35b") == str(blob)


def test_resolve_ollama_native_none_for_cloud_model(monkeypatch):
    monkeypatch.setattr(
        "kestrel.model_store.resolve_ollama_blob", lambda name: None
    )
    assert cli._resolve_ollama_native("qwen3-cloud") is None


def test_resolve_ollama_native_swallows_resolution_failure(monkeypatch):
    def boom(name):
        raise RuntimeError("daemon down")

    monkeypatch.setattr("kestrel.model_store.resolve_ollama_blob", boom)
    assert cli._resolve_ollama_native("qwen3.6:35b") is None


def test_cmd_run_ollama_resolves_blob_into_native_plan(capsys, monkeypatch, tmp_path):
    """ollama:// models with a local blob run through the Kestrel runtime
    (fit/context guardrails), not the Ollama passthrough."""
    blob = write_gguf(tmp_path / "blob", n_layer=48)  # no .gguf suffix
    monkeypatch.setattr(cli, "_resolve_ollama_native", lambda name: str(blob))
    config = {
        "model_size_gib": 3.0,
        "gpu_layers": "all",
        "fit_target_mib": 512,
        "cpu_moe": False,
        "moe_cache": "off",
        "moe_cache_budget_mib": 0,
        "threads": 8,
        "context_size": 4096,
        "context_reason": "test",
        "batch_size": 128,
        "ubatch_size": 64,
        "predicted_decode_tps": 0,
        "prediction_confidence": "low",
        "memory_overcommit": False,
        "has_mtp": False,
        "use_mtp": False,
    }
    monkeypatch.setattr(cli, "detect_gpu", lambda: None)
    monkeypatch.setattr(
        cli, "estimate_config", lambda model_info, gpu_info, args: dict(config)
    )
    monkeypatch.setattr(cli, "_configure_backend", lambda *a, **k: _StubBackend())
    args = _run_args("ollama://qwen3.6:35b", json=True)
    rc = cli.cmd_run(args)
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    # The command is a native llama-cli launch, not `ollama run`.
    assert payload["command"] == ["llama-cli", "--sentinel"]
    assert payload["model"] == str(blob)


def test_cmd_run_ollama_cloud_model_keeps_passthrough(monkeypatch, capsys, tmp_path):
    """No local blob (cloud model) keeps the `ollama run` passthrough."""
    monkeypatch.setattr(cli, "_resolve_ollama_native", lambda name: None)
    calls = []

    def fake_run(cmd):
        calls.append(cmd)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    args = _run_args("ollama://qwen3-cloud:latest", dry_run=False)
    with pytest.raises(SystemExit) as exc:
        cli.cmd_run(args)
    assert exc.value.code == 7
    assert calls == [["ollama", "run", "qwen3-cloud:latest"]]


def test_self_update_wheel_sha256_mismatch(monkeypatch, tmp_path):
    import kestrel.cli as cli
    from kestrel.errors import IntegrityError

    wheel = tmp_path / "kestrel-1.5.0-py3-none-any.whl"
    wheel.write_bytes(b"not-the-real-wheel")
    args = SimpleNamespace(
        repo=None, wheel=str(wheel), sha256="0" * 64, dry_run=True, json=False, yes=False
    )
    with pytest.raises(IntegrityError) as exc:
        cli.cmd_self_update(args)
    assert exc.value.code == "integrity_error"


def test_self_update_wheel_sha256_ok_then_dry_run(monkeypatch, tmp_path, capsys):
    from hashlib import sha256

    import kestrel.cli as cli

    wheel = tmp_path / "kestrel-1.5.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    digest = sha256(b"wheel").hexdigest()
    args = SimpleNamespace(
        repo=None, wheel=str(wheel), sha256=digest, dry_run=True, json=False, yes=False
    )
    cli.cmd_self_update(args)
    out = capsys.readouterr().out
    assert "Would install Kestrel" in out


def test_self_update_rolls_back_on_failed_post_check(monkeypatch, tmp_path, capsys):
    import kestrel.cli as cli

    calls = {"restore": False}

    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda cmd, **kw: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(cli, "_post_install_check", lambda: (False, "1.0.0"))
    monkeypatch.setattr(cli, "_snapshot_installed", lambda: {"from": None, "to": None})
    monkeypatch.setattr(cli, "_restore_install", lambda snap: calls.update(restore=True))
    args = SimpleNamespace(
        repo=str(Path(__file__).resolve().parents[1]),
        wheel=None,
        sha256=None,
        dry_run=False,
        json=False,
        yes=False,
    )
    from kestrel.errors import IntegrityError

    with pytest.raises(IntegrityError):
        cli.cmd_self_update(args)
    assert calls["restore"] is True


def test_self_update_rolls_back_json_result(monkeypatch, tmp_path, capsys):
    import json as _json

    import kestrel.cli as cli

    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda cmd, **a: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    # After rollback the post-check passes again.
    state = {"ok": False}

    def post_check():
        if state["ok"]:
            return True, "1.5.0"
        state["ok"] = True
        return False, ""

    monkeypatch.setattr(cli, "_post_install_check", post_check)
    monkeypatch.setattr(cli, "_snapshot_installed", lambda: {"from": None, "to": None})
    monkeypatch.setattr(cli, "_restore_install", lambda snap: None)
    args = SimpleNamespace(
        repo=str(Path(__file__).resolve().parents[1]),
        wheel=None,
        sha256=None,
        dry_run=False,
        json=True,
        yes=False,
    )
    cli.cmd_self_update(args)
    payload = _json.loads(capsys.readouterr().out)
    assert payload["status"] == "rolled_back"
    assert payload["rolled_back"] is True
    assert payload["to_version"] == "1.5.0"
