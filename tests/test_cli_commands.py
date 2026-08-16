"""Behavioral coverage for the noninteractive command families.

These tests intentionally exercise command handlers with their provider and
binary boundaries replaced by small fakes.  That keeps the parser-facing
contracts testable without requiring a GPU, Ollama daemon, Hugging Face CLI,
or a llama.cpp checkout on the test host.
"""

from __future__ import annotations

import importlib
import json
import subprocess
from types import SimpleNamespace

import pytest

from kestrel import config
from kestrel.cli import bench, health, models, parser, run, state

main = importlib.import_module("kestrel.cli.main")
from kestrel.errors import BackendError, InputError, ModelError, ServiceError
from kestrel.model_store import ModelStoreError, OllamaModel
from kestrel.providers.ollama import OllamaError, OllamaGeneration


class _FakeBenchProcess:
    """Popen-shaped stand-in for llama-bench runs in cmd_benchmark."""

    def __init__(self, command, *, stdout="", stderr="", returncode=0, timeout_error=None, **kwargs):
        self.command = command
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self._timeout_error = timeout_error
        self.killed = False

    def poll(self):
        return self.returncode

    def communicate(self, timeout=None):
        self.communicate_timeout = timeout
        if self._timeout_error is not None and not self.killed:
            raise self._timeout_error
        return self._stdout, self._stderr

    def kill(self):
        self.killed = True

    @classmethod
    def for_success(cls, command, rows, **kwargs):
        return cls(command, stdout=json.dumps(rows), returncode=0, **kwargs)


def _dispatch(argv: list[str]) -> int:
    cli_parser = parser.build_parser()
    return main._run_dispatched(cli_parser, cli_parser.parse_args(argv))


def _bench_args(model: str, **overrides):
    values = dict(
        model=model,
        prompt_tokens=8,
        generate_tokens=4,
        repetitions=2,
        ctx_size=1024,
        gpu_layers="auto",
        cpu_moe="off",
        threads="4,8",
        batch_size=32,
        ubatch_size=16,
        kv_cache_type="q8_0",
        output=None,
        quiet=True,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_benchmark_ollama_json_report_and_file(monkeypatch, tmp_path, capsys):
    calls = []

    class FakeClient:
        def generate(self, name, prompt, **kwargs):
            calls.append((name, prompt, kwargs))
            return OllamaGeneration("answer", "", 100, 1_000_000_000, 20, 1_000_000_000, 0)

    monkeypatch.setattr(bench.parser, "_default_model", lambda args, error: "ollama://demo")
    monkeypatch.setattr("kestrel.providers.ollama.OllamaClient", FakeClient)
    output = tmp_path / "bench.json"
    report = bench.cmd_benchmark(_bench_args("ignored", output=str(output)))

    assert report["engine"] == "ollama"
    assert report["decode_tokens_per_second"] == pytest.approx(20)
    assert report["release_speed_floor_passed"] is True
    assert len(calls) == 3  # warmup plus two measured repetitions
    assert json.loads(output.read_text())["sample_output"] == "answer"
    assert "Wrote" in capsys.readouterr().err


def test_benchmark_ollama_failure_is_structured(monkeypatch):
    class BrokenClient:
        def generate(self, *args, **kwargs):
            raise OllamaError("daemon offline")

    monkeypatch.setattr(bench.parser, "_default_model", lambda args, error: "ollama://demo")
    monkeypatch.setattr("kestrel.providers.ollama.OllamaClient", BrokenClient)
    with pytest.raises(ServiceError, match="daemon offline"):
        bench.cmd_benchmark(_bench_args("ignored"))


def test_benchmark_json_is_one_compact_document(monkeypatch, capsys):
    class FakeClient:
        def generate(self, *_args, **_kwargs):
            return OllamaGeneration("answer", "", 10, 1_000_000_000, 4, 1_000_000_000, 0)

    monkeypatch.setattr("kestrel.providers.ollama.OllamaClient", FakeClient)
    assert _dispatch(["benchmark", "ollama://demo", "--repetitions", "1", "--json"]) == 0

    captured = capsys.readouterr()
    assert captured.out.count("\n") == 1
    assert json.loads(captured.out)["engine"] == "ollama"
    assert "Benchmark" not in captured.err


def test_benchmark_local_builds_exact_command_and_report(monkeypatch, tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    command_seen = []
    rows = [
        {"n_prompt": 8, "n_threads": 8, "avg_ts": 100.0},
        {"n_gen": 4, "n_threads": 8, "n_gpu_layers": 24, "avg_ts": 12.5},
        {"n_gen": 4, "n_threads": 4, "n_gpu_layers": 24, "avg_ts": 10.0},
    ]

    monkeypatch.setattr(bench.parser, "_default_model", lambda args, error: str(model))
    monkeypatch.setattr(bench.model_source, "detect_model", lambda value: {"type": "gguf", "path": str(model)})
    monkeypatch.setattr(bench, "resolve_llama_binary", lambda name, dirs=None: "/fake/llama-bench")
    monkeypatch.setattr(
        bench, "_load_capabilities", lambda binary, refresh=False: SimpleNamespace(supports=lambda flag: False)
    )
    monkeypatch.setattr(bench.probes, "detect_gpu", lambda: None)
    monkeypatch.setattr(
        bench.planning,
        "estimate_config",
        lambda info, gpu, args: {
            "gpu_layers": "auto",
            "cpu_moe": True,
            "fit_target_mib": 512,
            "n_layers": 48,
            "model_size_gib": 3.5,
            "threads": 4,
        },
    )

    processes = []

    def fake_popen(command, **kwargs):
        command_seen.append(command)
        proc = _FakeBenchProcess.for_success(command, rows)
        processes.append(proc)
        return proc

    monkeypatch.setattr(bench.subprocess, "Popen", fake_popen)
    report = bench.cmd_benchmark(_bench_args(str(model), output=str(tmp_path / "report.json")))

    command = command_seen[0]
    assert command[:4] == ["/fake/llama-bench", "-m", str(model), "-p"]
    assert "-fitt" in command and "-ncmoe" in command
    assert "--moe-cache" not in command
    assert "-tb" not in command
    assert processes[0].communicate_timeout == 30 * 60 * 2
    assert report["placement"]["threads"] == 8
    assert report["decode_tokens_per_second"] == 12.5


def test_benchmark_emits_threads_batch_when_llama_bench_supports_it(monkeypatch, tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    rows = [{"n_prompt": 8, "n_threads": 4, "avg_ts": 100.0}, {"n_gen": 4, "n_threads": 4, "avg_ts": 12.5}]
    command_seen = []
    monkeypatch.setattr(bench.parser, "_default_model", lambda args, error: str(model))
    monkeypatch.setattr(bench.model_source, "detect_model", lambda value: {"type": "gguf", "path": str(model)})
    monkeypatch.setattr(bench, "resolve_llama_binary", lambda name, dirs=None: "/fake/llama-bench")
    monkeypatch.setattr(
        bench, "_load_capabilities", lambda binary, refresh=False: SimpleNamespace(supports=lambda flag: True)
    )
    monkeypatch.setattr(bench.probes, "detect_gpu", lambda: None)
    monkeypatch.setattr(
        bench.planning,
        "estimate_config",
        lambda info, gpu, args: {
            "gpu_layers": "33",
            "cpu_moe": True,
            "n_layers": 48,
            "threads": 4,
            "threads_batch": 16,
            "model_size_gib": 3.5,
        },
    )

    def fake_popen(command, **kwargs):
        command_seen.append(command)
        return _FakeBenchProcess.for_success(command, rows)

    monkeypatch.setattr(bench.subprocess, "Popen", fake_popen)
    bench.cmd_benchmark(_bench_args(str(model)))
    command = command_seen[0]
    assert "-tb" in command and command[command.index("-tb") + 1] == "16"


@pytest.mark.parametrize(
    ("result", "exception"),
    [
        (subprocess.TimeoutExpired("llama-bench", 1), ServiceError),
        (subprocess.CompletedProcess([], 2, stdout="", stderr="bad"), BackendError),
    ],
)
def test_benchmark_local_process_failures(monkeypatch, tmp_path, result, exception):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    monkeypatch.setattr(bench.parser, "_default_model", lambda args, error: str(model))
    monkeypatch.setattr(bench.model_source, "detect_model", lambda value: {"type": "gguf", "path": str(model)})
    monkeypatch.setattr(bench, "resolve_llama_binary", lambda name, dirs=None: "/fake/llama-bench")
    monkeypatch.setattr(bench.probes, "detect_gpu", lambda: None)
    monkeypatch.setattr(
        bench.planning,
        "estimate_config",
        lambda *args: {"gpu_layers": "all", "cpu_moe": False, "threads": 4, "model_size_gib": 1},
    )
    monkeypatch.setattr(bench, "_load_capabilities", lambda binary, refresh=False: None)

    def fake_popen(command, **kwargs):
        if isinstance(result, Exception):
            return _FakeBenchProcess(command, timeout_error=result)
        return _FakeBenchProcess(command, returncode=result.returncode, stderr=result.stderr)

    monkeypatch.setattr(bench.subprocess, "Popen", fake_popen)
    with pytest.raises(exception):
        bench.cmd_benchmark(_bench_args(str(model)))


@pytest.mark.parametrize("payload", [{}, {"x": 1}, [], [{"n_prompt": 8, "n_threads": 4, "avg_ts": 10}]])
def test_benchmark_rejects_invalid_success_json(monkeypatch, tmp_path, payload):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    monkeypatch.setattr(bench.parser, "_default_model", lambda args, error: str(model))
    monkeypatch.setattr(bench.model_source, "detect_model", lambda value: {"type": "gguf", "path": str(model)})
    monkeypatch.setattr(bench, "resolve_llama_binary", lambda name, dirs=None: "/fake/llama-bench")
    monkeypatch.setattr(bench.probes, "detect_gpu", lambda: None)
    monkeypatch.setattr(
        bench.planning,
        "estimate_config",
        lambda *args: {"gpu_layers": "all", "cpu_moe": False, "threads": 4, "model_size_gib": 1},
    )
    monkeypatch.setattr(bench, "_load_capabilities", lambda binary, refresh=False: None)
    monkeypatch.setattr(
        bench.subprocess,
        "Popen",
        lambda command, **kwargs: _FakeBenchProcess.for_success(command, payload),
    )

    with pytest.raises(BackendError, match="schema|missing"):
        bench.cmd_benchmark(_bench_args(str(model)))


def test_benchmark_rejects_nonpositive_programmatic_values():
    with pytest.raises(InputError, match="repetitions"):
        bench.cmd_benchmark(_bench_args("model", repetitions=0))


def test_benchmark_parse_failure_is_one_json_document(capsys):
    argv = ["kestrel", "benchmark", "ollama://demo", "--repetitions", "nope", "--json"]

    with pytest.raises(SystemExit) as exc:
        parser.build_parser().parse_args(argv[1:])

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out)["error"]["code"] == "invalid_input"


@pytest.mark.parametrize(
    "arguments",
    [
        ["run", "model.gguf", "--max-tokens", "0", "--json"],
        ["run", "model.gguf", "--gpu-layers", "-1", "--json"],
        ["run", "model.gguf", "--moe-cache", "-4", "--json"],
        ["run", "model.gguf", "--tensor-split", "60,nan", "--json"],
        ["serve", "model.gguf", "--port", "65536", "--json"],
        ["serve", "model.gguf", "--wait", "-1", "--json"],
    ],
)
def test_runtime_numeric_parse_failures_are_json(arguments, capsys):
    with pytest.raises(SystemExit) as exc:
        parser.build_parser().parse_args(arguments)

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out)["error"]["code"] == "invalid_input"


def test_tensor_split_can_explicitly_exclude_one_gpu():
    args = parser.build_parser().parse_args(["run", "model.gguf", "--tensor-split", "100,0"])
    assert args.tensor_split == "100,0"


def test_serve_json_interrupt_during_readiness_reaps_child(monkeypatch, tmp_path, capsys):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    process = SimpleNamespace(returncode=None, terminated=False)
    process.poll = lambda: process.returncode

    def terminate():
        process.terminated = True
        process.returncode = -15

    process.terminate = terminate
    process.kill = lambda: None
    process.wait = lambda timeout=None: process.returncode

    monkeypatch.setattr(run.model_source, "_resolve_model_source", lambda args: {"type": "gguf", "path": str(model)})
    monkeypatch.setattr(run.model_source, "_ensure_local_gguf", lambda info, args: info)
    monkeypatch.setattr(run.probes, "detect_gpu", lambda: None)
    monkeypatch.setattr(
        run.planning,
        "estimate_config",
        lambda *args: {
            "context_size": 2048,
            "context_reason": "test",
            "gpu_layers": 0,
            "cpu_moe": False,
        },
    )
    monkeypatch.setattr(run.runtime, "_build_server_cmd", lambda *args: ["llama-server"])
    monkeypatch.setattr(run.runtime, "_wait_ready", lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt))
    monkeypatch.setattr(run.subprocess, "Popen", lambda *args, **kwargs: process)

    assert _dispatch(["serve", str(model), "--wait", "1", "--json"]) == 130
    assert process.terminated is True
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "interrupted"
    assert payload["exit_code"] == 130


def test_benchmark_missing_model_is_typed_error(monkeypatch):
    monkeypatch.setattr(
        bench.parser, "_default_model", lambda args, error: (_ for _ in ()).throw(ModelError("missing"))
    )
    with pytest.raises(ModelError, match="missing"):
        bench.cmd_benchmark(_bench_args("ignored"))


def test_summarize_benchmark_rows_prefers_fastest_decode_thread():
    prompt, decode, sweep = bench._summarize_benchmark_rows(
        [
            {"n_prompt": 1, "n_threads": 4, "avg_ts": 30},
            {"n_prompt": 1, "n_threads": 8, "avg_ts": 40},
            {"n_gen": 1, "n_threads": 4, "avg_ts": 5},
            {"n_gen": 1, "n_threads": 8, "avg_ts": 9},
        ]
    )
    assert prompt["n_threads"] == 8
    assert decode["avg_ts"] == 9
    assert [row["threads"] for row in sweep] == [4, 8]


def test_benchmark_rejects_nonfinite_measurements():
    with pytest.raises(BackendError, match="positive avg_ts"):
        bench._validate_benchmark_rows(
            [
                {"n_prompt": 8, "n_threads": 4, "avg_ts": float("nan")},
                {"n_gen": 4, "n_threads": 4, "avg_ts": 10.0},
            ]
        )


def test_models_search_and_files_json(monkeypatch, capsys):
    monkeypatch.setattr(
        "kestrel.model_store.search_huggingface",
        lambda query, limit: [{"id": "org/model", "downloads": 1, "likes": 2, "license": "MIT"}],
    )
    monkeypatch.setattr(
        "kestrel.model_store.list_huggingface_ggufs",
        lambda source: [{"path": "m.gguf", "size_bytes": 7, "security_status": "unknown"}],
    )
    assert _dispatch(["models", "search", "qwen", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["models"][0]["id"] == "org/model"
    assert _dispatch(["models", "files", "hf://org/repo", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["files"][0]["path"] == "m.gguf"


def test_missing_engine_subcommand_is_typed(capsys):
    assert _dispatch(["engine"]) == 2
    assert "choose an engine command" in capsys.readouterr().err


def test_models_search_provider_failure_is_json(monkeypatch, capsys):
    monkeypatch.setattr(
        "kestrel.model_store.search_huggingface",
        lambda query, limit: (_ for _ in ()).throw(ModelStoreError("hub unavailable")),
    )
    assert _dispatch(["models", "search", "qwen", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "model_error"
    assert "hub unavailable" in payload["error"]["message"]


def test_models_list_json_contains_local_and_ollama(monkeypatch, tmp_path, capsys):
    local = tmp_path / "model.gguf"
    local.write_bytes(b"GGUF")
    monkeypatch.setattr(state, "USER_CONFIG", SimpleNamespace(models_dir=str(tmp_path)))
    monkeypatch.setattr("kestrel.model_store.discover_local_models", lambda root: [local])
    monkeypatch.setattr("kestrel.model_store.complete_gguf_models", lambda paths: paths)
    monkeypatch.setattr("kestrel.model_store.model_total_size", lambda path: 42)
    monkeypatch.setattr(
        "kestrel.model_store.list_ollama_models",
        lambda resolve_paths: [OllamaModel("qwen", "id", "4 GB", "today", local if resolve_paths else None)],
    )
    assert _dispatch(["models", "list", "--resolve", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kestrel"][0]["size_bytes"] == 42
    assert payload["ollama"][0]["local_path"] == str(local)


def test_models_import_sets_default_for_local_gguf(monkeypatch, tmp_path, capsys):
    source = tmp_path / "model.gguf"
    source.write_bytes(b"GGUF")
    saved = []
    monkeypatch.setattr(models.model_source, "detect_model", lambda path: {"type": "gguf", "path": str(source)})
    monkeypatch.setattr(models.health, "_save_default_model", lambda value: saved.append(value))
    assert _dispatch(["models", "import", str(source), "--set-default"]) == 0
    assert saved == [source.resolve()]
    assert "Set as the default" in capsys.readouterr().out


def test_models_pull_ollama_dry_run_is_safe(capsys):
    assert _dispatch(["models", "pull", "ollama://qwen3", "--dry-run"]) == 0
    assert "ollama pull qwen3" in capsys.readouterr().out


def test_models_recommend_json_ranks_fit(monkeypatch, tmp_path, capsys):
    paths = [tmp_path / "small.gguf", tmp_path / "large.gguf"]
    for path in paths:
        path.write_bytes(b"GGUF")
    monkeypatch.setattr(state, "USER_CONFIG", SimpleNamespace(models_dir=str(tmp_path)))
    monkeypatch.setattr("kestrel.model_store.discover_local_models", lambda root: paths)
    monkeypatch.setattr("kestrel.model_store.complete_gguf_models", lambda values: values)
    monkeypatch.setattr("kestrel.model_store.model_total_size", lambda path: 100 if path == paths[0] else 10**12)
    monkeypatch.setattr("kestrel.model_store.list_ollama_models", lambda **_kwargs: [])
    monkeypatch.setattr(models.probes, "detect_gpu", lambda: {"name": "test", "vram_total_mb": 1000})
    monkeypatch.setattr(models.probes, "_available_ram_mib", lambda: 100)
    monkeypatch.setattr(models.model_source, "read_gguf_config", lambda path: {"architecture": "qwen", "n_layer": 2})
    assert _dispatch(["models", "recommend", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["models"][0]["fit"] == "excellent"
    assert payload["models"][1]["fit"] == "paging"


class _FakeCaps:
    help_text = "--fit --cpu-moe --mmap --direct-io --cache-type-k --spec-type draft-mtp"
    version = "fake-1"
    spec_types = {"draft-mtp"}

    def supports(self, flag):
        return flag in self.help_text


def _patch_health_environment(monkeypatch, tmp_path, *, default_model=None):
    monkeypatch.setattr(
        state, "USER_CONFIG", SimpleNamespace(models_dir=str(tmp_path / "models"), default_model=default_model)
    )
    monkeypatch.setattr(health, "config_path", lambda: tmp_path / "config.toml")
    monkeypatch.setattr(health, "available_disk_bytes", lambda path: 10 * 1024**3)
    monkeypatch.setattr(health, "_writable_probe", lambda path: (True, ""))
    monkeypatch.setattr(health.probes, "detect_gpu", lambda: {"name": "test-gpu"})
    monkeypatch.setattr(health.probes, "_memory_snapshot", lambda: {"swap_total_mib": 0, "swap_used_mib": 0})
    monkeypatch.setattr(health.probes, "_available_ram_mib", lambda: 4096)
    monkeypatch.setattr(
        health.probes,
        "_cpu_power_policy",
        lambda: {"governor": "performance", "energy_performance_preference": "performance", "turbo_enabled": True},
    )


def test_doctor_json_reports_fallback_capabilities(monkeypatch, tmp_path, capsys):
    _patch_health_environment(monkeypatch, tmp_path)

    class FakeBackend:
        def __init__(self, *args, **kwargs):
            pass

        llama_cpp_dir = str(tmp_path / "llama")
        binary = str(tmp_path / "llama" / "llama-cli")
        server_binary = str(tmp_path / "llama" / "llama-server")

        def capabilities(self):
            raise RuntimeError("cli unavailable")

        def server_capabilities(self):
            return _FakeCaps()

    monkeypatch.setattr(health, "LlamaCppBackend", FakeBackend)
    assert _dispatch(["doctor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    # A missing default model is an expected warning on a fresh host; engine
    # compatibility is nevertheless healthy and doctor exits successfully.
    assert payload["status"] == "warn"
    assert payload["llama_cli"]["available"] is False
    assert payload["llama_server"]["available"] is True
    assert any("via llama-server" in item["message"] for item in payload["checks"])


def test_doctor_json_returns_failure_without_any_engine(monkeypatch, tmp_path, capsys):
    _patch_health_environment(monkeypatch, tmp_path)

    class FakeBackend:
        def __init__(self, *args, **kwargs):
            pass

        llama_cpp_dir = str(tmp_path / "llama")

        def capabilities(self):
            raise RuntimeError("missing cli")

        def server_capabilities(self):
            raise RuntimeError("missing server")

    monkeypatch.setattr(health, "LlamaCppBackend", FakeBackend)
    assert _dispatch(["doctor", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "fail"
    assert payload["llama_cli"]["error"] == "missing cli"


def test_doctor_checks_detects_invalid_default_gguf(monkeypatch, tmp_path):
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"NOPE")
    _patch_health_environment(monkeypatch, tmp_path, default_model=str(bad))
    monkeypatch.setattr(health.model_source, "detect_model", lambda value: {"type": "gguf", "path": str(bad)})
    checks = health._doctor_checks(SimpleNamespace(llama_cpp_dir=str(tmp_path)), _FakeCaps(), None)
    integrity = next(item for item in checks if item["name"] == "model cache integrity")
    assert integrity["status"] == "fail"


def test_doctor_treats_ollama_default_as_provider_managed(monkeypatch, tmp_path):
    _patch_health_environment(monkeypatch, tmp_path, default_model="ollama://demo")

    checks = health._doctor_checks(SimpleNamespace(llama_cpp_dir=str(tmp_path)), _FakeCaps(), None)

    integrity = next(item for item in checks if item["name"] == "model cache integrity")
    assert integrity["status"] == "ok"
    assert "provider-managed" in integrity["message"]


def test_writable_probe_uses_unique_file_and_leaves_no_residue(tmp_path):
    ok, detail = health._writable_probe(tmp_path)

    assert ok is True
    assert detail == ""
    assert list(tmp_path.iterdir()) == []


def test_status_json_reads_profile_and_reports_model_match(monkeypatch, tmp_path, capsys):
    _patch_health_environment(monkeypatch, tmp_path, default_model="model.gguf")
    config_file = tmp_path / "config.toml"
    config_file.write_text("x")
    profile = config_file.with_name("hardware-profile.json")
    profile.write_text(
        json.dumps(
            {"model": {"source": "model.gguf"}, "plan": {"context_size": 4096}, "benchmark": {"status": "measured"}}
        )
    )
    monkeypatch.setattr(
        health, "load_config", lambda: SimpleNamespace(default_model="model.gguf", models_dir="/models")
    )
    assert _dispatch(["status", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["profile_matches_model"] is True
    assert payload["benchmark"]["status"] == "measured"


def test_setup_persists_resolved_model_and_refreshes_state(monkeypatch, tmp_path, capsys):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    monkeypatch.setattr(health, "load_config", lambda: config.KestrelConfig())
    monkeypatch.setattr(health.model_source, "detect_model", lambda value: {"type": "gguf", "path": str(model)})
    monkeypatch.setattr(health, "default_llama_cpp_dir", lambda: str(tmp_path / "llama"))
    saved = []
    monkeypatch.setattr(health, "save_config", lambda value: saved.append(value) or tmp_path / "config.toml")
    refreshed = []
    monkeypatch.setattr(health.state, "reload_state", lambda: refreshed.append(True))
    args = parser.build_parser().parse_args(
        [
            "setup",
            "--model",
            str(model),
            "--models-dir",
            str(tmp_path / "models"),
            "--context",
            "8192",
            "--reasoning",
            "high",
        ]
    )
    assert main._run_dispatched(parser.build_parser(), args) == 0
    assert saved[0].default_model == str(model.resolve())
    assert saved[0].models_dir == str(tmp_path / "models")
    assert saved[0].context_size == 8192
    assert saved[0].reasoning_level == "high"
    assert refreshed == [True]
    assert "configuration saved" in capsys.readouterr().out


def test_settings_show_and_update_persistent_model_defaults(monkeypatch, tmp_path, capsys):
    current = config.KestrelConfig(default_model="model.gguf", context_size="auto", reasoning_level="auto")
    monkeypatch.setattr(health, "load_config", lambda: current)
    saved = []
    monkeypatch.setattr(health, "save_config", lambda value: saved.append(value) or tmp_path / "config.toml")
    monkeypatch.setattr(health.state, "reload_state", lambda: None)

    assert _dispatch(["settings", "--context", "16384", "--reasoning", "medium", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["context_size"] == 16384
    assert payload["reasoning_level"] == "medium"
    assert payload["reasoning_budgets"]["medium"] == 2048
    assert saved[0].default_model == "model.gguf"


def test_settings_persist_kv_and_placement_defaults(monkeypatch, tmp_path, capsys):
    current = config.KestrelConfig(
        default_model="model.gguf",
        context_size="auto",
        reasoning_level="auto",
        kv_cache_type="auto",
        kv_cache_turbo=False,
        gpu_layers="auto",
        cpu_moe="auto",
    )
    monkeypatch.setattr(health, "load_config", lambda: current)
    saved = []
    monkeypatch.setattr(health, "save_config", lambda value: saved.append(value) or tmp_path / "config.toml")
    monkeypatch.setattr(health.state, "reload_state", lambda: None)

    rc = _dispatch(
        [
            "settings",
            "--kv-cache-type",
            "q4_0",
            "--turbo-kv",
            "--gpu-layers",
            "24",
            "--cpu-moe",
            "on",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kv_cache_type"] == "q4_0"
    assert payload["kv_cache_turbo"] is True
    assert payload["gpu_layers"] == "24"
    assert payload["cpu_moe"] == "on"
    assert saved[0].kv_cache_type == "q4_0"
    assert saved[0].kv_cache_turbo is True
    assert saved[0].gpu_layers == "24"
    assert saved[0].cpu_moe == "on"


def test_settings_turbo_kv_toggle_can_turn_off(monkeypatch, tmp_path, capsys):
    current = config.KestrelConfig(default_model="model.gguf", kv_cache_turbo=True)
    monkeypatch.setattr(health, "load_config", lambda: current)
    saved = []
    monkeypatch.setattr(health, "save_config", lambda value: saved.append(value) or tmp_path / "config.toml")
    monkeypatch.setattr(health.state, "reload_state", lambda: None)

    assert _dispatch(["settings", "--no-turbo-kv", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["kv_cache_turbo"] is False
    assert saved[0].kv_cache_turbo is False


def test_run_parser_uses_saved_settings_and_allows_one_launch_override(monkeypatch):
    monkeypatch.setattr(state, "USER_CONFIG", config.KestrelConfig(context_size=8192, reasoning_level="high"))
    saved = parser.build_parser().parse_args(["run", "model.gguf"])
    overridden = parser.build_parser().parse_args(["run", "model.gguf", "--ctx-size", "4096", "--reasoning", "off"])

    assert (saved.ctx_size, saved.reasoning) == (8192, "high")
    assert (overridden.ctx_size, overridden.reasoning) == (4096, "off")


def test_run_and_chat_accept_session_flag():
    parsed = parser.build_parser().parse_args(["run", "model.gguf", "--session", "work"])
    assert parsed.session == "work"
    chat = parser.build_parser().parse_args(["chat", "--session", "work"])
    assert chat.session == "work"
