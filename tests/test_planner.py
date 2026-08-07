import pytest

import kestrel
from kestrel.core.planner import (
    HardwareProfile,
    ModelProfile,
    effective_bytes_per_param,
    estimate_parameters,
    gpu_bandwidth_gb_s,
    plan_runtime,
    predict_decode_tokens_per_second,
)


def test_version_consistent():
    assert kestrel.__version__ == "1.5.0"


def test_estimate_parameters_dense():
    model = ModelProfile(
        path="/x", n_layers=32, n_experts=0, n_experts_used=0,
        hidden_size=2048, expert_ff_size=5632, has_mtp=False,
    )
    params = estimate_parameters(model)
    assert params["total_params"] > 0
    assert params["total_params"] == params["active_params"]
    # dense FFN present, not zero
    assert params["dense_per_layer"] > 0


def test_estimate_parameters_moe():
    model = ModelProfile(
        path="/x", n_layers=48, n_experts=256, n_experts_used=8,
        hidden_size=3072, expert_ff_size=1024, has_mtp=False,
    )
    params = estimate_parameters(model)
    assert params["expert_params"] > 0
    assert params["total_params"] > params["active_params"]
    # active uses only the 8 routed experts per layer (vs 256 total)
    assert params["active_params"] < params["total_params"] // 10


def test_effective_bytes_per_param_uses_file_density():
    model = ModelProfile(
        path="/x", n_layers=1, n_experts=0, n_experts_used=0,
        hidden_size=1000, expert_ff_size=2000, has_mtp=False,
        file_size_bytes=2_000_000,
    )
    params = estimate_parameters(model)
    bpp = effective_bytes_per_param(model, params, "q4_0")
    # file density 2M / params should be within [0.1, 2.0] and returned directly
    assert 0.1 <= bpp <= 2.0


def test_effective_bytes_per_param_fallback():
    model = ModelProfile(
        path="/x", n_layers=1, n_experts=0, n_experts_used=0,
        hidden_size=1000, expert_ff_size=2000, has_mtp=False,
    )
    params = estimate_parameters(model)
    assert effective_bytes_per_param(model, params, "q4_0") == pytest.approx(0.56)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("NVIDIA GeForce RTX 4060 Laptop GPU", 256.0),
        ("NVIDIA GeForce RTX 4090", 1008.0),
        ("NVIDIA A100 SXM4", 2039.0),
        ("some unknown gpu", 256.0),
        (None, 256.0),
        ("rtx 4060 ti", 288.0),
    ],
)
def test_gpu_bandwidth(name, expected):
    assert gpu_bandwidth_gb_s(name) == expected


def _dense_27b():
    return ModelProfile(
        path="/x", n_layers=46, n_experts=0, n_experts_used=0,
        hidden_size=3584, expert_ff_size=18944, has_mtp=False,
        file_size_bytes=17 * 1024**3,
    )


def _moe_122b():
    return ModelProfile(
        path="/x", n_layers=48, n_experts=256, n_experts_used=8,
        hidden_size=3072, expert_ff_size=1024, has_mtp=False,
        file_size_bytes=72 * 1024**3,
    )


def _small_gpu():
    return HardwareProfile(
        gpu_name="rtx 4060", vram_total_mib=8188, vram_free_mib=6000,
        ram_available_mib=24000, logical_cpu_count=16,
    )


def test_plan_dense_model_not_pinned_to_cpu():
    """Regression: a dense model larger than VRAM must not be pinned to 4 CPU
    layers while cpu_moe is off. It should keep auto GPU-layer placement."""
    plan = plan_runtime(_dense_27b(), _small_gpu(), context_size=2048)
    assert plan.cpu_moe is False
    assert plan.gpu_layers == "auto"


def test_plan_moe_known_placement():
    plan = plan_runtime(_moe_122b(), _small_gpu(), context_size=2048)
    assert plan.cpu_moe is True
    assert plan.gpu_layers == "12"


def test_plan_unknown_moe_conservative():
    unknown = ModelProfile(
        path="/x", n_layers=48, n_experts=128, n_experts_used=8,
        hidden_size=2048, expert_ff_size=1024, has_mtp=False,
        file_size_bytes=35 * 1024**3,
    )
    plan = plan_runtime(unknown, _small_gpu(), context_size=2048)
    assert plan.cpu_moe is True
    assert plan.gpu_layers == "4"


def test_plan_context_minimum():
    plan = plan_runtime(_moe_122b(), _small_gpu(), context_size=10)
    assert plan.context_size >= 512


def test_plan_explicit_cpu_moe_off_dense():
    plan = plan_runtime(
        _dense_27b(), _small_gpu(), context_size=2048, requested_cpu_moe=False
    )
    assert plan.cpu_moe is False


def test_predict_nonzero():
    tps = predict_decode_tokens_per_second(
        _moe_122b(), _small_gpu(), cpu_moe=True
    )
    assert tps > 0
    assert isinstance(tps, float)


def test_predict_smaller_gpu_layers_helps_cpumoe():
    baseline = predict_decode_tokens_per_second(
        _dense_27b(), _small_gpu(), cpu_moe=True, gpu_layers_offloaded=0
    )
    offloaded = predict_decode_tokens_per_second(
        _dense_27b(), _small_gpu(), cpu_moe=True, gpu_layers_offloaded=12
    )
    assert offloaded >= baseline


def test_plan_modes():
    base = plan_runtime(_moe_122b(), _small_gpu(), context_size=2048)
    quality = plan_runtime(_moe_122b(), _small_gpu(), context_size=2048, mode="quality")
    assert quality.batch_size <= base.batch_size
    assert quality.use_mtp is False
    # balanced is identity
    balanced = plan_runtime(_moe_122b(), _small_gpu(), context_size=2048, mode="balanced")
    assert balanced == base
