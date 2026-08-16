import json
import sys
from pathlib import Path
from types import SimpleNamespace

from kestrel import tuning
from kestrel.cli import bench, planning
from kestrel.core.planner import ModelProfile
from kestrel.errors import KestrelError

sys.path.insert(0, str(Path(__file__).parent))

from gguf_fixture import write_gguf  # noqa: E402


def _gpu(free=8000):
    return {
        "name": "GPU",
        "vram_total_mb": 8192,
        "vram_free_mb": free,
        "devices": [{"name": "GPU", "vram_total_mb": 8192, "vram_free_mb": free}],
    }


def _write_profile(path, model_info, gpu, **tuning_overrides):
    selected = {
        "gpu_layers": "33",
        "cpu_moe": True,
        "n_cpu_moe_layers": 28,
        "batch_size": 512,
        "ubatch_size": 256,
        "threads": 14,
    }
    payload = {
        "schema_version": tuning.PROFILE_SCHEMA,
        "tuning": {
            "status": "measured",
            "model_identity": tuning.model_identity(model_info),
            "hardware_identity": tuning.hardware_identity(gpu),
            "engine_identity": tuning.engine_identity(),
            "context_size": 16384,
            "minimum_free_vram_mib": 6000,
            "selected_plan": selected,
            **tuning_overrides,
        },
    }
    path.write_text(json.dumps(payload))
    return selected


def test_matching_tuned_plan_requires_exact_identity_and_headroom(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    model_info = {"type": "gguf", "path": str(model)}
    profile = tmp_path / "profile.json"
    selected = _write_profile(profile, model_info, _gpu())

    plan, reason = tuning.matching_tuned_plan(model_info, _gpu(), context_size=16384, path=profile)
    assert plan == selected
    assert reason == "exact measured model/hardware profile"

    assert tuning.matching_tuned_plan(model_info, _gpu(5000), context_size=16384, path=profile)[0] is None
    larger_plan, larger_reason = tuning.matching_tuned_plan(model_info, _gpu(), context_size=32768, path=profile)
    assert larger_plan == selected
    assert larger_reason == tuning.LARGER_CONTEXT_REASON
    model.write_bytes(b"GGUF-changed")
    assert tuning.matching_tuned_plan(model_info, _gpu(), context_size=16384, path=profile)[0] is None


def test_old_or_malformed_profile_fails_closed(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    info = {"type": "gguf", "path": str(model)}
    profile = tmp_path / "profile.json"
    profile.write_text("not-json")
    assert tuning.matching_tuned_plan(info, _gpu(), context_size=2048, path=profile)[0] is None
    profile.write_text(json.dumps({"schema_version": 1, "plan": {"gpu_layers": "99"}}))
    assert tuning.matching_tuned_plan(info, _gpu(), context_size=2048, path=profile)[0] is None


def test_generic_tuning_candidates_include_safe_and_bounded_moe_profiles():
    candidates = bench._tuning_candidates(
        {
            "gpu_layers": "4",
            "n_cpu_moe_layers": None,
            "batch_size": 256,
            "ubatch_size": 64,
            "n_layers": 40,
            "n_experts": 64,
        }
    )
    assert candidates[0]["label"] == "planner_baseline"
    assert any(item["gpu_layers"] == "41" and item["cpu_moe_layers"] == 40 for item in candidates)
    assert any(item["cpu_moe_layers"] == 35 and item["ubatch_size"] == 256 for item in candidates)


def test_optimize_search_persists_fastest_measured_exact_profile(monkeypatch, tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    model_info = {"type": "gguf", "path": str(model)}
    gpu = _gpu()
    profile = {
        "schema_version": 1,
        "plan": {
            "gpu_layers": "4",
            "n_cpu_moe_layers": None,
            "batch_size": 256,
            "ubatch_size": 64,
            "n_layers": 40,
            "n_experts": 64,
            "model_size_gib": 4,
            "context_size": 4096,
            "kv_cache_type": "q8_0",
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "threads": 8,
        },
    }
    monkeypatch.setattr(bench.model_source, "detect_model", lambda value: model_info)
    monkeypatch.setattr(bench.probes, "detect_gpu", lambda: gpu)

    def fake_benchmark(args):
        prompt = 20.0 if args.ubatch_size == 256 else 8.0
        decode = 15.0 if args.cpu_moe_layers == 35 else 14.0
        return {"prompt_tokens_per_second": prompt, "decode_tokens_per_second": decode}

    monkeypatch.setattr(bench, "cmd_benchmark", fake_benchmark)
    args = SimpleNamespace(quality="balanced", no_save=True)
    bench._run_optimize_benchmark(profile, str(model), tmp_path / "profile.json", args)

    assert profile["schema_version"] == tuning.PROFILE_SCHEMA
    assert profile["tuning"]["status"] == "measured"
    assert profile["tuning"]["model_identity"] == tuning.model_identity(model_info)
    assert profile["tuning"]["hardware_identity"] == tuning.hardware_identity(gpu)
    assert profile["tuning"]["selected_plan"]["ubatch_size"] == 256
    assert profile["benchmark"]["status"] == "measured"


def test_estimate_config_applies_matching_profile_only_for_auto_plan(monkeypatch):
    model = ModelProfile(
        path="/model.gguf",
        n_layers=32,
        n_experts=64,
        n_experts_used=4,
        hidden_size=2048,
        expert_ff_size=1024,
        has_mtp=False,
        file_size_bytes=8 * 1024**3,
    )
    monkeypatch.setattr(planning.model_source, "_model_profile", lambda info: model)
    monkeypatch.setattr(planning.probes, "_available_ram_mib", lambda: 16000)
    tuned = {
        "gpu_layers": "33",
        "cpu_moe": True,
        "n_cpu_moe_layers": 28,
        "batch_size": 512,
        "ubatch_size": 256,
        "threads": 12,
    }
    monkeypatch.setattr(
        planning,
        "matching_tuned_plan",
        lambda *args, **kwargs: (tuned, "exact measured model/hardware profile"),
    )
    args = SimpleNamespace(
        gpu_layers="auto",
        ctx_size=4096,
        cpu_moe="auto",
        moe_cache="off",
        moe_cold_model=None,
        target="auto",
        reasoning="auto",
        use_tuning_profile=True,
    )
    config = planning.estimate_config({"type": "gguf", "path": "/model.gguf"}, _gpu(), args)
    assert config["gpu_layers"] == "33"
    assert config["n_cpu_moe_layers"] == 28
    assert config["batch_size"] == 512
    assert config["tuning_profile_applied"] is True


def test_estimate_config_honors_saved_placement_and_kv_settings(monkeypatch):
    model = ModelProfile(
        path="/model.gguf",
        n_layers=32,
        n_experts=64,
        n_experts_used=4,
        hidden_size=2048,
        expert_ff_size=1024,
        has_mtp=False,
        file_size_bytes=8 * 1024**3,
    )
    monkeypatch.setattr(planning.model_source, "_model_profile", lambda info: model)
    monkeypatch.setattr(planning.probes, "_available_ram_mib", lambda: 16000)
    monkeypatch.setattr(
        planning,
        "matching_tuned_plan",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("profile applied")),
    )
    from kestrel import config as kestrel_config
    from kestrel.cli import state

    monkeypatch.setattr(
        state,
        "USER_CONFIG",
        kestrel_config.KestrelConfig(kv_cache_type="q4_0", gpu_layers="24", cpu_moe="on"),
    )
    args = SimpleNamespace(
        gpu_layers="auto",
        ctx_size=4096,
        cpu_moe="auto",
        kv_cache_type="auto",
        moe_cache="off",
        moe_cold_model=None,
        target="auto",
        reasoning="auto",
        use_tuning_profile=True,
    )
    config = planning.estimate_config({"type": "gguf", "path": "/model.gguf"}, _gpu(), args)
    assert config["gpu_layers"] == "24"
    assert config["cpu_moe"] is True
    assert config["cache_type_k"] == "q4_0"
    assert config["cache_type_v"] == "q4_0"
    assert config["kv_cache_type"] == "q4_0"
    assert config["tuning_profile_applied"] is False


def test_estimate_config_cli_flag_beats_saved_setting(monkeypatch):
    model = ModelProfile(
        path="/model.gguf",
        n_layers=32,
        n_experts=64,
        n_experts_used=4,
        hidden_size=2048,
        expert_ff_size=1024,
        has_mtp=False,
        file_size_bytes=8 * 1024**3,
    )
    monkeypatch.setattr(planning.model_source, "_model_profile", lambda info: model)
    monkeypatch.setattr(planning.probes, "_available_ram_mib", lambda: 16000)
    monkeypatch.setattr(
        planning,
        "matching_tuned_plan",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("profile applied")),
    )
    from kestrel import config as kestrel_config
    from kestrel.cli import state

    monkeypatch.setattr(state, "USER_CONFIG", kestrel_config.KestrelConfig(gpu_layers="24"))
    args = SimpleNamespace(
        gpu_layers="all",
        ctx_size=4096,
        cpu_moe="auto",
        kv_cache_type="q4_1",
        moe_cache="off",
        moe_cold_model=None,
        target="auto",
        reasoning="auto",
        use_tuning_profile=True,
    )
    config = planning.estimate_config({"type": "gguf", "path": "/model.gguf"}, _gpu(), args)
    assert config["gpu_layers"] == "all"
    assert config["cache_type_k"] == "q4_1"


def _auto_config(**overrides):
    base = {
        "gpu_layers": "4",
        "cpu_moe": True,
        "n_cpu_moe_layers": None,
        "n_layers": 40,
        "n_experts": 64,
        "model_size_gib": 4.0,
        "context_size": 4096,
        "batch_size": 256,
        "ubatch_size": 64,
        "cache_type_k": "q8_0",
        "cache_type_v": "q8_0",
        "threads": 8,
        "predicted_decode_tps": 5.0,
        "prediction_confidence": "uncalibrated-model-estimate",
        "tuning_profile_applied": False,
        "tuning_profile_reason": "no measured profile",
        "has_mtp": False,
        "use_mtp": False,
    }
    base.update(overrides)
    return base


def _auto_args(**overrides):
    base = {"no_auto_tune": False, "dry_run": False}
    base.update(overrides)
    return SimpleNamespace(**base)


def test_estimate_config_applies_placement_at_larger_context_with_bounded_batches(monkeypatch):
    model = ModelProfile(
        path="/model.gguf",
        n_layers=32,
        n_experts=64,
        n_experts_used=4,
        hidden_size=2048,
        expert_ff_size=1024,
        has_mtp=False,
        file_size_bytes=8 * 1024**3,
    )
    monkeypatch.setattr(planning.model_source, "_model_profile", lambda info: model)
    monkeypatch.setattr(planning.probes, "_available_ram_mib", lambda: 16000)
    tuned = {
        "gpu_layers": "33",
        "cpu_moe": True,
        "n_cpu_moe_layers": 28,
        "batch_size": 512,
        "ubatch_size": 256,
        "threads": 12,
    }
    rates = {"prompt_tokens_per_second": 88.5, "decode_tokens_per_second": 12.3}
    monkeypatch.setattr(
        planning,
        "matching_tuned_plan",
        lambda *args, **kwargs: (tuned, tuning.LARGER_CONTEXT_REASON),
    )
    monkeypatch.setattr(planning, "profile_measured_rates", lambda *args, **kwargs: rates)
    args = SimpleNamespace(
        gpu_layers="auto",
        ctx_size=32768,
        cpu_moe="auto",
        moe_cache="off",
        moe_cold_model=None,
        target="auto",
        reasoning="auto",
        use_tuning_profile=True,
    )
    config = planning.estimate_config({"type": "gguf", "path": "/model.gguf"}, _gpu(), args)
    assert config["gpu_layers"] == "33"
    assert config["n_cpu_moe_layers"] == 28
    assert config["batch_size"] == 256
    assert config["ubatch_size"] == 64
    assert config["tuning_profile_applied"] is True
    assert config["context_scaled"] is True
    assert config["predicted_decode_tps"] == 12.3
    assert config["prediction_confidence"] == "measured"


def test_auto_tune_plan_skips_when_not_triggered(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    model_info = {"type": "gguf", "path": str(model)}
    args = _auto_args()
    for config in (
        _auto_config(tuning_profile_applied=True),
        _auto_config(tuning_profile_reason="free VRAM 4000 MiB is below tuned floor 6000 MiB"),
        _auto_config(tuning_profile_reason=None),
    ):
        assert bench.auto_tune_plan(model_info, _gpu(), config, args) is config
    for overrides in ({"dry_run": True}, {"no_auto_tune": True}):
        config = _auto_config()
        assert bench.auto_tune_plan(model_info, _gpu(), config, _auto_args(**overrides)) is config


def test_auto_tune_plan_skips_without_bench_binary(tmp_path, monkeypatch):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    model_info = {"type": "gguf", "path": str(model)}
    monkeypatch.setattr(bench, "resolve_llama_binary", lambda *a, **k: None)
    config = _auto_config()
    assert bench.auto_tune_plan(model_info, _gpu(), config, _auto_args()) is config


def test_auto_tune_plan_measures_and_persists_profile(monkeypatch, tmp_path):
    model = write_gguf(tmp_path / "model.gguf", n_layer=40, n_exp=64, n_used=8)
    model_info = {"type": "gguf", "path": str(model)}
    profile_target = tmp_path / "profiles" / "model.json"
    monkeypatch.setattr(bench, "resolve_llama_binary", lambda *a, **k: "/fake/llama-bench")
    monkeypatch.setattr(bench, "profile_path_for", lambda *a, **k: profile_target)

    def fake_benchmark(args):
        if isinstance(args.threads, str):
            return {
                "prompt_tokens_per_second": 30.0,
                "decode_tokens_per_second": 12.0,
                "placement": {"threads": 10},
                "thread_sweep": [{"threads": 10, "decode_tokens_per_second": 12.0}],
            }
        prompt = 25.0 if args.ubatch_size == 256 else 8.0
        decode = 16.0 if args.cpu_moe_layers == 35 else 14.0
        return {"prompt_tokens_per_second": prompt, "decode_tokens_per_second": decode}

    monkeypatch.setattr(bench, "cmd_benchmark", fake_benchmark)
    config = bench.auto_tune_plan(model_info, _gpu(), _auto_config(), _auto_args())

    assert config["gpu_layers"] == "41"
    assert config["cpu_moe"] is True
    assert config["n_cpu_moe_layers"] == 35
    assert config["batch_size"] == 512
    assert config["ubatch_size"] == 256
    assert config["threads"] == 10
    assert config["predicted_decode_tps"] == 16.0
    assert config["prediction_confidence"] == "measured"
    assert config["tuning_profile_applied"] is True
    assert config["measured_prompt_tps"] == 25.0
    assert profile_target.is_file()
    saved = json.loads(profile_target.read_text())
    assert saved["tuning"]["status"] == "measured"
    assert saved["tuning"]["selected_plan"]["n_cpu_moe_layers"] == 35
    assert saved["benchmark"]["status"] == "measured"
    assert saved["benchmark"]["decode_tokens_per_second"] == 16.0


def test_auto_tune_plan_keeps_conservative_config_when_all_candidates_fail(monkeypatch, tmp_path, capsys):
    model = write_gguf(tmp_path / "model.gguf", n_layer=40, n_exp=64, n_used=8)
    model_info = {"type": "gguf", "path": str(model)}
    profile_target = tmp_path / "profiles" / "model.json"
    monkeypatch.setattr(bench, "resolve_llama_binary", lambda *a, **k: "/fake/llama-bench")
    monkeypatch.setattr(bench, "profile_path_for", lambda *a, **k: profile_target)
    monkeypatch.setattr(bench, "cmd_benchmark", lambda args: (_ for _ in ()).throw(KestrelError("boom")))

    config = bench.auto_tune_plan(model_info, _gpu(), _auto_config(), _auto_args())

    assert config["gpu_layers"] == "4"
    assert config["prediction_confidence"] == "uncalibrated-model-estimate"
    assert config["tuning_profile_applied"] is False
    assert profile_target.is_file()
    saved = json.loads(profile_target.read_text())
    assert saved["tuning"]["status"] == "failed"
    assert saved["benchmark"]["status"] == "failed"
    assert "no usable placement" in capsys.readouterr().err


def test_profile_measured_rates_uses_benchmark_section(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    model_info = {"type": "gguf", "path": str(model)}
    profile = tmp_path / "profile.json"
    _write_profile(profile, model_info, _gpu())
    assert tuning.profile_measured_rates(model_info, _gpu(), context_size=16384, path=profile) is None
    payload = json.loads(profile.read_text())
    payload["benchmark"] = {
        "status": "measured",
        "prompt_tokens_per_second": 88.5,
        "decode_tokens_per_second": 12.3,
    }
    profile.write_text(json.dumps(payload))
    rates = tuning.profile_measured_rates(model_info, _gpu(), context_size=16384, path=profile)
    assert rates == {"prompt_tokens_per_second": 88.5, "decode_tokens_per_second": 12.3}


def test_estimate_config_uses_measured_rate_from_profile(monkeypatch):
    model = ModelProfile(
        path="/model.gguf",
        n_layers=32,
        n_experts=64,
        n_experts_used=4,
        hidden_size=2048,
        expert_ff_size=1024,
        has_mtp=False,
        file_size_bytes=8 * 1024**3,
    )
    monkeypatch.setattr(planning.model_source, "_model_profile", lambda info: model)
    monkeypatch.setattr(planning.probes, "_available_ram_mib", lambda: 16000)
    tuned = {
        "gpu_layers": "33",
        "cpu_moe": True,
        "n_cpu_moe_layers": 28,
        "batch_size": 512,
        "ubatch_size": 256,
        "threads": 12,
    }
    rates = {"prompt_tokens_per_second": 88.5, "decode_tokens_per_second": 12.3}
    monkeypatch.setattr(
        planning,
        "matching_tuned_plan",
        lambda *args, **kwargs: (tuned, "exact measured model/hardware profile"),
    )
    monkeypatch.setattr(planning, "profile_measured_rates", lambda *args, **kwargs: rates)
    args = SimpleNamespace(
        gpu_layers="auto",
        ctx_size=4096,
        cpu_moe="auto",
        moe_cache="off",
        moe_cold_model=None,
        target="auto",
        reasoning="auto",
        use_tuning_profile=True,
    )
    config = planning.estimate_config({"type": "gguf", "path": "/model.gguf"}, _gpu(), args)
    assert config["predicted_decode_tps"] == 12.3
    assert config["prediction_confidence"] == "measured"
    assert config["measured_prompt_tps"] == 88.5
    assert config["measured_decode_tps"] == 12.3
