"""Planning: turning a resolved model + hardware into a runtime configuration.

Owns the explainable config estimate shared by ``run``/``serve``/``benchmark``
and the memory-safe context selection that keeps a launch from OOMing the host.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ..core.planner import (
    HardwareProfile,
    estimate_parameters,
    plan_runtime,
    predict_decode_tokens_per_second,
)
from ..model_store import model_total_size
from . import model_source, probes


def _plan_mode(model, requested: str) -> str:
    """Resolve the placement target.

    ``auto`` resolves to ``balanced``. The explicit ``speed`` and ``quality``
    profiles are never auto-selected: an experimental throughput bias must be
    requested deliberately (see ``parser --target`` and ``plan_runtime``).
    """
    if requested in ("balanced", "quality", "speed"):
        return requested
    return "balanced"


def _model_size_bytes(model_info: dict) -> int:
    """Total on-disk model bytes, summing split-GGUF shards when applicable.

    ``_select_context_size`` receives the raw ``model_info`` document (before
    a :class:`ModelProfile` exists), so it re-derives the footprint here. A
    split GGUF is several files; sizing from one shard undercounts the model.
    """
    path = model_info.get("path")
    if not path:
        return 0
    candidate = Path(path)
    if model_info.get("type") == "gguf":
        if not candidate.is_file():
            return 0
        try:
            return model_total_size(candidate)
        except (OSError, ValueError):
            return 0
    try:
        return sum(part.stat().st_size for part in candidate.glob("*.safetensors"))
    except OSError:
        return 0


def _kv_cache_bytes_per_token(model_info: dict) -> float:
    """Rough Q8 KV-cache bytes per token from the model architecture.

    GQA/MoE models only store the KV-projection values (n_kv_heads * head_dim)
    per layer per token, not the full hidden size; using the value dim keeps
    the estimate from over- or under-counting by the GQA ratio. Falls back to
    the hidden size when the KV dim is unreadable, and returns 0.0 when the
    architecture is unreadable so callers pick a conservative context instead
    of assuming a large one is safe.
    """
    values_per_token = None
    try:
        if model_info.get("type") == "gguf":
            cfg = model_source.read_gguf_config(model_info["path"])
            layers = cfg["n_layer"]
            hidden = cfg["hidden"]
            if cfg.get("n_kv_heads") and cfg.get("head_dim"):
                values_per_token = cfg["n_kv_heads"] * cfg["head_dim"]
        else:
            cfg_path = Path(model_info["path"]) / "config.json"
            cfg = json.loads(cfg_path.read_text())
            layers = cfg.get("num_hidden_layers") or cfg.get("n_layer") or 0
            hidden = cfg.get("hidden_size") or cfg.get("d_model") or 0
            heads = cfg.get("num_key_value_heads")
            head_dim = cfg.get("head_dim")
            if head_dim is None and hidden and cfg.get("num_attention_heads"):
                head_dim = hidden / cfg["num_attention_heads"]
            if heads and head_dim:
                values_per_token = heads * head_dim
    except (OSError, ValueError, KeyError, TypeError):
        return 0.0
    if not layers or not hidden:
        return 0.0
    if not values_per_token:
        values_per_token = hidden
    # K and V state per token, quantized KV (q8_0 ≈ 1.06 bytes/value) plus
    # bookkeeping overhead for the rope/scratch blocks.
    return 2.0 * int(layers) * int(values_per_token) * 1.1


def _select_context_size(
    model_info: dict,
    gpu_info: dict | None,
    model_size: int | None = None,
) -> tuple[int, str, bool]:
    """Choose a context tier that cannot push the host into an OOM crash.

    Returns ``(context, reason, overcommitted)``. ``overcommitted`` is True
    when the model file itself cannot fit alongside the launch overhead in
    free RAM + swap; in that case Kestrel picks the smallest context and the
    caller surfaces an explicit warning instead of silently paging the host.
    ``model_size`` may be passed in when the caller already built a
    :class:`ModelProfile`; otherwise it is derived from ``model_info``.
    """
    if model_size is None:
        model_size = _model_size_bytes(model_info)
    vram_free = (gpu_info or {}).get("vram_free_mb", 0) * 1024**2
    ram = probes._available_ram_mib() * 1024**2
    memory = probes._memory_snapshot()
    swap_free = (
        max(
            0,
            memory.get("swap_total_mib", 0) - memory.get("swap_used_mib", 0),
        )
        * 1024**2
    )
    kv_per_token = _kv_cache_bytes_per_token(model_info)
    overhead = 1536 * 1024**2  # CUDA runtime, compute scratch, tokenizer, OS
    fit_margin = 1024 * 1024**2  # reserve llama.cpp keeps via --fit-target

    # Model weights resident on the GPU: they and the KV cache must both fit in
    # free VRAM together with the CUDA runtime and the fit reserve. Selecting a
    # large context just because the weights fit (ignoring the KV cache that
    # also lives in VRAM) OOMs the GPU for every model that is barely small
    # enough to offload whole, e.g. a ~20 GiB MoE on a 24 GiB card.
    if vram_free and model_size:
        weights_budget = vram_free - fit_margin - overhead
        if weights_budget > 0 and model_size <= weights_budget:
            kv_budget = weights_budget - model_size
            if kv_per_token <= 0:
                return (
                    8192,
                    "GPU-resident; conservative default; KV cache cost unknown",
                    False,
                )
            chosen = 512
            for tier in (512, 1024, 2048, 4096, 8192, 16384, 32768):
                if tier * kv_per_token <= kv_budget:
                    chosen = tier
            if chosen >= 32768:
                reason = "GPU-resident; ample VRAM for weights and KV cache"
            elif chosen >= 4096:
                reason = "GPU-resident; KV cache fits alongside the weights"
            else:
                reason = "GPU-resident; KV cache constrains context on this GPU"
            return chosen, reason, False

    # Everything else shares free RAM/swap with the KV cache and compute
    # scratch. Beyond that point llama.cpp pages the host and the OS can OOM.
    headroom = ram + swap_free
    if model_size + overhead > headroom:
        return (
            512,
            "model exceeds available RAM+swap; smallest context to reduce OOM risk",
            True,
        )
    # Keep the KV cache mostly in RAM; only a sliver of swap credit is allowed
    # so a borderline model still gets a workable, conservative context.
    budget = max(0.0, (ram + swap_free * 0.5 - model_size - overhead) * 0.8)
    if kv_per_token <= 0:
        chosen, reason = 8192, "conservative default; KV cache cost unknown"
    else:
        chosen = 512
        for tier in (512, 1024, 2048, 4096, 8192, 16384, 32768):
            if tier * kv_per_token <= budget:
                chosen = tier
        if chosen >= 8192:
            reason = "ample RAM headroom for weights and KV cache"
        elif chosen >= 4096:
            reason = "moderate RAM headroom for weights and KV cache"
        elif chosen >= 2048:
            reason = "RAM-bound; moderate context to preserve stability"
        else:
            reason = "RAM-bound; conservative context to avoid memory pressure"
    return chosen, reason, False


def _context_size_arg(value: str) -> int | str:
    if value.lower() == "auto":
        return "auto"
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("context must be 'auto' or an integer") from exc
    if parsed < 512:
        raise argparse.ArgumentTypeError("context must be at least 512 tokens")
    return parsed


def _cpu_moe_thread_sweep(logical_cpu_count: int) -> str:
    maximum = max(1, logical_cpu_count)
    candidates = {
        max(1, maximum // 2),
        max(1, maximum - 6),
        max(1, maximum - 4),
        max(1, maximum - 2),
        maximum,
    }
    return ",".join(str(value) for value in sorted(candidates))


def estimate_config(model_info: dict, gpu_info: dict | None, args=None) -> dict:
    model = model_source._model_profile(model_info)
    hardware = HardwareProfile(
        gpu_name=gpu_info["name"] if gpu_info else None,
        vram_total_mib=gpu_info["vram_total_mb"] if gpu_info else 0,
        vram_free_mib=gpu_info["vram_free_mb"] if gpu_info else 0,
        ram_available_mib=probes._available_ram_mib(),
        logical_cpu_count=os.cpu_count() or 0,
    )
    requested_cpu_moe = None
    requested_gpu_layers = "auto"
    context_size = 2048
    context_reason = "default"
    if args:
        requested_gpu_layers = args.gpu_layers
        context_size = args.ctx_size
        if context_size == "auto":
            context_size, context_reason, overcommitted = _select_context_size(
                model_info,
                gpu_info,
                model_size=model.file_size_bytes,
            )
        else:
            context_reason = "explicit user setting"
            overcommitted = False
        requested_cpu_moe = {"on": True, "off": False, "auto": None}[args.cpu_moe]
    plan = plan_runtime(
        model,
        hardware,
        context_size=context_size,
        requested_gpu_layers=requested_gpu_layers,
        requested_cpu_moe=requested_cpu_moe,
        mode=_plan_mode(model, getattr(args, "target", "auto")),
    )
    moe_cache, moe_cache_budget_mib = plan.moe_cache, plan.moe_cache_budget_mib
    if args:
        if args.moe_cache == "on":
            moe_cache, moe_cache_budget_mib = "on", plan.moe_cache_budget_mib
        elif args.moe_cache == "off":
            moe_cache, moe_cache_budget_mib = "off", 0
        elif args.moe_cache != "auto":
            try:
                moe_cache_budget_mib = int(args.moe_cache)
                moe_cache = "on"
            except ValueError:
                moe_cache = "auto"
    result = plan.as_dict()
    try:
        gpu_layers_offloaded = int(plan.gpu_layers)
    except ValueError:
        gpu_layers_offloaded = 0
    result["n_gpu_layers"] = plan.gpu_layers
    result["has_mtp"] = model.has_mtp
    result["n_layers"] = model.n_layers
    result["n_experts"] = model.n_experts
    result["hidden_size"] = model.hidden_size
    result["n_ff"] = model.expert_ff_size
    result["model_size_gib"] = round(model.file_size_bytes / 1024**3, 2)
    result["moe_cache"] = moe_cache
    result["moe_cache_budget_mib"] = moe_cache_budget_mib
    result["predicted_decode_tps"] = predict_decode_tokens_per_second(
        model,
        hardware,
        cpu_moe=plan.cpu_moe,
        moe_cache_budget_mib=moe_cache_budget_mib,
        gpu_layers_offloaded=gpu_layers_offloaded,
        draft=plan.use_mtp,
        cpu_expert_quant=("q1_0" if args and getattr(args, "moe_cold_model", None) else None),
    )
    result["active_params_b"] = round(estimate_parameters(model)["active_params"] / 1e9, 1)
    result["prediction_confidence"] = (
        "measured-q1-fallback" if args and getattr(args, "moe_cold_model", None) else "uncalibrated-model-estimate"
    )
    result["context_reason"] = context_reason
    result["reasoning_level"] = getattr(args, "reasoning", "auto") if args else "auto"
    result["memory_overcommit"] = overcommitted
    return result
