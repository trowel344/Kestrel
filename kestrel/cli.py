from __future__ import annotations

import argparse
import functools
import json
import os
import platform
import re
import shlex
import shutil
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TextIO

from . import ui
from .backends.llama_cpp import (
    LlamaCppBackend,
    default_llama_cpp_dir,
    resolve_llama_binary,
)
from .config import KestrelConfig, config_path, load_config, save_config
from .core.planner import (
    HardwareProfile,
    ModelProfile,
    estimate_parameters,
    model_file_size,
    plan_runtime,
    predict_decode_tokens_per_second,
)
from .errors import KestrelError, MissingModelError, ServiceError

try:
    USER_CONFIG = load_config()
except ValueError as exc:
    USER_CONFIG = KestrelConfig()
    CONFIG_ERROR = str(exc)
else:
    CONFIG_ERROR = None
LLAMA_CPP_DIR = USER_CONFIG.llama_cpp_dir or default_llama_cpp_dir()

# Friendly names for models Kestrel has an explicit, tested runtime profile for.
# The environment override makes the alias durable when the GGUF is moved out
# of /tmp; the remaining paths cover Kestrel's managed-model location and the
# current research artifact.
MODEL_ALIASES = {
    "qwen3.5:122b-10a": (
        "KESTREL_QWEN35_122B_GGUF",
        "~/.local/share/kestrel/models/qwen3.5-122b-a10b-nvfp4.gguf",
        "/tmp/qwen3.5-122b-a10b-nvfp4.gguf",
    ),
}


def _ttl_cache(seconds: float):
    """Memoize a probe result briefly so repeated reads in one invocation
    (or rapid menu redraws) do not re-spawn ``nvidia-smi`` or re-read
    ``/proc/meminfo`` on every call. Results age out so live hardware stays
    reasonably fresh in a long-running interactive session."""

    def deco(fn):
        cached_at = None
        value = None

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            nonlocal cached_at, value
            now = time.monotonic()
            if cached_at is not None and now - cached_at < seconds:
                return value
            value = fn(*args, **kwargs)
            cached_at = now
            return value

        return wrapped

    return deco


@_ttl_cache(seconds=5)
def detect_gpu() -> dict | None:
    """Return the aggregated GPU profile across all devices.

    The returned dict keeps the historical single-GPU shape (``name``,
    ``vram_total_mb``, ``vram_free_mb``) for callers, plus ``count`` and
    ``devices`` so multi-GPU rigs can fit and tensor-split over combined VRAM.
    """
    return _aggregate_gpu(detect_gpus())


def _aggregate_gpu(devices: list[dict]) -> dict | None:
    if not devices:
        return None
    if len(devices) == 1:
        return {**devices[0], "count": 1, "devices": devices}
    return {
        "name": ", ".join(device["name"] for device in devices),
        "vram_total_mb": sum(device["vram_total_mb"] for device in devices),
        "vram_free_mb": sum(device["vram_free_mb"] for device in devices),
        "count": len(devices),
        "devices": devices,
    }


def detect_gpus() -> list[dict]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []
        devices = []
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            # Split the memory columns from the right so a vendor name containing a
            # comma ("Foo, Inc. ...") does not misalign the following columns.
            head, total, free = line.rsplit(",", 2)
            devices.append(
                {
                    "name": head.strip() or "unknown",
                    "vram_total_mb": int(total.strip()),
                    "vram_free_mb": int(free.strip()),
                }
            )
        return devices
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError):
        return []


@_ttl_cache(seconds=5)
def _available_ram_mib() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size / 1024**2)
    except (ValueError, OSError, AttributeError):
        return 0


@_ttl_cache(seconds=5)
def _memory_snapshot() -> dict:
    fields = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, raw = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                fields[key] = int(raw.split()[0]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    swap_total = fields.get("SwapTotal", 0)
    swap_free = fields.get("SwapFree", 0)
    return {
        "ram_total_mib": fields.get("MemTotal", 0),
        "ram_available_mib": fields.get("MemAvailable", _available_ram_mib()),
        "swap_total_mib": swap_total,
        "swap_used_mib": max(0, swap_total - swap_free),
    }


def _cpu_power_policy() -> dict:
    base = Path("/sys/devices/system/cpu/cpu0/cpufreq")

    def read(path: Path) -> str | None:
        try:
            return path.read_text().strip() or None
        except OSError:
            return None

    no_turbo = read(Path("/sys/devices/system/cpu/intel_pstate/no_turbo"))
    return {
        "governor": read(base / "scaling_governor"),
        "energy_performance_preference": read(base / "energy_performance_preference"),
        "turbo_enabled": None if no_turbo is None else no_turbo == "0",
    }


def _resolve_hf_model_dir(path: Path) -> Path | None:
    if (path / "config.json").is_file():
        return path
    refs_main = path / "refs" / "main"
    if refs_main.is_file():
        revision = refs_main.read_text().strip()
        snapshot = path / "snapshots" / revision
        if (snapshot / "config.json").is_file():
            return snapshot
    snapshots = path / "snapshots"
    if snapshots.is_dir():
        candidates = sorted(
            (item for item in snapshots.iterdir() if (item / "config.json").is_file()),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
    return None


def _resolve_model_alias(model_str: str) -> Path | None:
    candidates = MODEL_ALIASES.get(model_str.strip().lower())
    if candidates is None:
        return None
    env_name, *paths = candidates
    configured = os.environ.get(env_name)
    if configured:
        paths.insert(0, configured)
    for raw_path in paths:
        candidate = Path(raw_path).expanduser()
        if candidate.is_file():
            return candidate.resolve()
    return None


def detect_model(model_str: str) -> dict | None:
    alias_path = _resolve_model_alias(model_str)
    if alias_path is not None:
        path = str(alias_path)
        return {"type": "gguf", "path": path, "gguf_name": path}

    candidate = Path(model_str).expanduser()
    if candidate.is_file():
        try:
            if candidate.suffix.lower() == ".gguf":
                is_gguf = True
            else:
                with candidate.open("rb") as handle:
                    is_gguf = handle.read(4) == b"GGUF"
        except OSError:
            is_gguf = False
        if is_gguf:
            path = str(candidate.resolve())
            return {"type": "gguf", "path": path, "gguf_name": path}
    if candidate.is_dir():
        resolved = _resolve_hf_model_dir(candidate.resolve())
        if resolved:
            return {"type": "safetensors", "path": str(resolved), "hub_id": None}

    if model_str.startswith(("hf://", "huggingface://")):
        hub_id = model_str.split("://", 1)[1]
    else:
        hub_id = model_str
    if "/" not in hub_id:
        return None

    cache_root = Path(
        os.path.expanduser(
            f"~/.cache/huggingface/hub/models--{hub_id.replace('/', '--')}"
        )
    )
    resolved = _resolve_hf_model_dir(cache_root)
    if resolved:
        return {"type": "safetensors", "path": str(resolved), "hub_id": hub_id}
    return {"type": "safetensors", "path": None, "hub_id": hub_id}


def read_gguf_config(gguf_path: str) -> dict:
    from .gguf.metadata import read_planner_metadata

    return read_planner_metadata(gguf_path)


def _safetensors_info(directory: str) -> dict:
    """Report config.json summary + param estimate for a safetensors model dir."""
    base = Path(directory)
    info: dict = {
        "type": "safetensors",
        "path": str(base),
        "size_bytes": sum(p.stat().st_size for p in base.glob("*.safetensors")),
    }
    config_file = base / "config.json"
    if not config_file.is_file():
        info["config"] = {"error": "no config.json present"}
        return info
    try:
        config = json.loads(config_file.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        info["config"] = {"error": f"unreadable config.json: {exc}"}
        return info
    info["config"] = {
        key: config.get(key)
        for key in (
            "architectures",
            "model_type",
            "n_layer",
            "num_hidden_layers",
            "n_experts",
            "num_local_experts",
            "num_experts_per_tok",
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
            "max_position_embeddings",
        )
    }
    hidden = config.get("hidden_size") or config.get("d_model")
    layers = config.get("num_hidden_layers") or config.get("n_layer")
    if hidden and layers:
        intermediate = config.get("intermediate_size") or 4 * hidden
        params = layers * (4 * hidden * hidden + 3 * hidden * intermediate)
        info["estimated_params_b"] = round(params / 1e9, 2)
    return info


def _warm_page_cache(paths: list[str]) -> None:
    """Prime the OS page cache for model files before llama.cpp starts.

    A bounded pre-read (max 64 MiB from the start, which covers the GGUF
    header, metadata, and first tensors) plus a Linux ``WILLNEED`` hint for the
    rest. Bounded on purpose so a cold launch is never made meaningfully slower.
    """
    if not paths:
        return
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_WILLNEED"):
        for path in paths:
            try:
                fd = os.open(path, os.O_RDONLY)
            except OSError:
                continue
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
            except OSError:
                pass
            finally:
                os.close(fd)
    for path in paths:
        try:
            with open(path, "rb") as f:
                remaining = 64 * 1024 * 1024
                while remaining > 0:
                    chunk = f.read(min(8 * 1024 * 1024, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
        except OSError:
            continue


def _plan_mode(model: ModelProfile, requested: str) -> str:
    """Resolve the adaptive placement target.

    ``auto`` adapts from free RAM to the model working set: ``speed`` when
    there is real headroom, otherwise ``balanced``. The explicit targets are
    honored verbatim and are never auto-selected, so an unstable faster
    profile can't appear on its own.
    """
    if requested in ("balanced", "quality", "speed"):
        return requested
    size_mib = model.file_size_bytes / 1024**2 if model.file_size_bytes else 0
    ram_mib = _available_ram_mib()
    if ram_mib >= 4096 and ram_mib >= size_mib * 0.8:
        return "speed"
    return "balanced"


def _model_profile(model_info: dict) -> ModelProfile:
    if model_info["type"] == "gguf":
        cfg = read_gguf_config(model_info["path"])
        return ModelProfile(
            path=model_info["path"],
            n_layers=cfg["n_layer"],
            n_experts=cfg["n_exp"],
            n_experts_used=cfg["n_used"],
            hidden_size=cfg["hidden"],
            expert_ff_size=cfg["n_ff"],
            has_mtp=cfg["mtp_layers"] > 0,
            file_size_bytes=model_file_size(model_info["path"]),
        )

    from .gguf.converter import NVFP4Converter

    converter = NVFP4Converter(model_info["path"])
    return ModelProfile(
        path=model_info["path"],
        n_layers=converter.n_layer,
        n_experts=converter.n_exp,
        n_experts_used=converter.n_used,
        hidden_size=converter.hidden,
        expert_ff_size=converter.n_ff,
        has_mtp=converter.mtp_layers > 0,
        file_size_bytes=_safetensors_size(converter),
    )


def _safetensors_size(converter) -> int:
    """Total size of the shard files the source model's weight map references.

    This lets the planner apply its model-larger-than-VRAM heuristic to a
    safetensors source, matching what a converted GGUF would provide.
    """
    total = 0
    for shard in set(converter.wm.values()):
        try:
            total += os.path.getsize(os.path.join(converter.model_dir, shard))
        except OSError:
            continue
    return total


def estimate_config(model_info: dict, gpu_info: dict | None, args=None) -> dict:
    model = _model_profile(model_info)
    hardware = HardwareProfile(
        gpu_name=gpu_info["name"] if gpu_info else None,
        vram_total_mib=gpu_info["vram_total_mb"] if gpu_info else 0,
        vram_free_mib=gpu_info["vram_free_mb"] if gpu_info else 0,
        ram_available_mib=_available_ram_mib(),
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
                model_info, gpu_info
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
        cpu_expert_quant=(
            "q1_0" if args and getattr(args, "moe_cold_model", None) else None
        ),
    )
    result["active_params_b"] = round(
        estimate_parameters(model)["active_params"] / 1e9, 1
    )
    result["prediction_confidence"] = (
        "measured-q1-fallback"
        if args and getattr(args, "moe_cold_model", None)
        else "uncalibrated-model-estimate"
    )
    result["context_reason"] = context_reason
    result["memory_overcommit"] = overcommitted
    return result


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
            cfg = read_gguf_config(model_info["path"])
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


def _select_context_size(model_info: dict, gpu_info: dict | None) -> tuple[int, str, bool]:
    """Choose a context tier that cannot push the host into an OOM crash.

    Returns ``(context, reason, overcommitted)``. ``overcommitted`` is True
    when the model file itself cannot fit alongside the launch overhead in
    free RAM + swap; in that case Kestrel picks the smallest context and the
    caller surfaces an explicit warning instead of silently paging the host.
    """
    model_size = model_file_size(model_info.get("path") or "")
    vram_free = (gpu_info or {}).get("vram_free_mb", 0) * 1024**2
    ram = _available_ram_mib() * 1024**2
    memory = _memory_snapshot()
    swap_free = max(
        0,
        memory.get("swap_total_mib", 0) - memory.get("swap_used_mib", 0),
    ) * 1024**2
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


def _flatten_extra(values: list[str] | None) -> list[str]:
    """Flatten ``--extra`` strings (shell-whitespace separated) into tokens."""
    if not values:
        return []
    tokens: list[str] = []
    for value in values:
        tokens.extend(shlex.split(value))
    return tokens


def _tensor_split_arg(gpu: dict | None, explicit: str | None) -> str | None:
    """Return a llama.cpp ``--tensor-split`` ratio list.

    An explicit user value wins. Otherwise, when more than one GPU is present,
    ratios are derived from each device's free VRAM (a rough load-balance
    heuristic; llama.cpp re-uses them as fraction denominators). Single-GPU and
    unknown layouts return ``None`` so the backend keeps its default.
    """
    if explicit:
        return explicit
    devices = (gpu or {}).get("devices") or []
    if len(devices) < 2:
        return None
    free = [max(0, int(device.get("vram_free_mb") or 0)) for device in devices]
    total = sum(free)
    if total <= 0:
        return None
    ratios = [f"{value * 100 // total}" for value in free]
    return ",".join(ratios)


def _configure_backend(model_info: dict, config: dict, args=None) -> LlamaCppBackend:
    """Build the LlamaCppBackend from a planned runtime config.

    Shared by the interactive and server launch builders so backend
    construction and engine-tune arguments have a single source of truth.
    """
    use_mtp = config["use_mtp"] and not (args and args.no_mtp)
    return LlamaCppBackend(
        model_path=model_info["path"],
        n_gpu_layers=config["gpu_layers"],
        n_ctx=config["context_size"],
        n_batch=(args.batch_size if args and args.batch_size else config["batch_size"]),
        n_ubatch=(args.ubatch_size if args and args.ubatch_size else config["ubatch_size"]),
        spec_type="mtp" if use_mtp else "none",
        spec_draft_n=args.mtp_tokens if args else 3,
        cpu_moe=config["cpu_moe"],
        fit=config["fit"],
        fit_target_mib=(
            args.fit_target if args and args.fit_target else config["fit_target_mib"]
        ),
        cache_type_k=args.kv_cache_type if args else config["cache_type_k"],
        cache_type_v=args.kv_cache_type if args else config["cache_type_v"],
        use_mmap=not (args and args.no_mmap),
        use_mlock=bool(args and args.mlock),
        direct_io=bool(args and args.direct_io),
        tensor_split=_tensor_split_arg(
            (args and getattr(args, "_gpu", None)) or detect_gpu(),
            getattr(args, "tensor_split", None) or None,
        ),
        extra_args=_flatten_extra(args.extra if args else None),
        n_threads=(
            args.threads
            if args and args.threads is not None
            else config["threads"]
        ),
        llama_cpp_dir=LLAMA_CPP_DIR,
        moe_cache=(
            str(config["moe_cache_budget_mib"])
            if config["moe_cache"] == "on" and config["moe_cache_budget_mib"]
            else config["moe_cache"]
        ),
    )


def _run_with_oom_retries(
    cmd: list[str], max_retries: int = 2, env: dict[str, str] | None = None
) -> int:
    """Run an interactive llama.cpp process and retry startup CUDA OOMs.

    Output and input stay attached to the terminal. Only stderr is mirrored so
    Kestrel can identify a CUDA allocation failure.
    """
    current = list(cmd)
    for attempt in range(max_retries + 1):
        tail: list[str] = []
        started = time.monotonic()
        process = subprocess.Popen(
            current,
            stdin=None,
            stdout=None,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )

        def mirror_stderr(process=process, tail=tail):
            assert process.stderr is not None
            for line in process.stderr:
                sys.stderr.write(line)
                tail.append(line)
                if len(tail) > 300:
                    del tail[:100]

        reader = threading.Thread(target=mirror_stderr, daemon=True)
        reader.start()
        interrupted = False
        try:
            returncode = process.wait()
        except KeyboardInterrupt:
            interrupted = True
            process.terminate()
            try:
                returncode = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = process.wait()
        finally:
            reader.join(timeout=2)
            if process.stderr is not None:
                process.stderr.close()
        if interrupted:
            return returncode
        if returncode == 0:
            return 0

        error_text = "".join(tail).lower()
        startup_failure = time.monotonic() - started < 180
        is_oom = any(
            marker in error_text
            for marker in (
                "out of memory",
                "cuda error",
                "failed to allocate",
                "allocation failed",
            )
        )
        if not is_oom or not startup_failure or attempt >= max_retries:
            if returncode != 0:
                print(
                    f"\nKestrel: llama.cpp exited with status {returncode} "
                    "(stderr shown above).",
                    file=sys.stderr,
                )
                from .engine import matches_too_old_signature

                if matches_too_old_signature(error_text):
                    print(
                        "  Hint: this model may need a newer llama.cpp engine. "
                        "Run `kestrel engine status`, then `kestrel engine update`.",
                        file=sys.stderr,
                    )
            return returncode

        def flag_value(flag: str) -> str | None:
            try:
                index = current.index(flag)
            except ValueError:
                return None
            if index + 1 >= len(current):
                return None
            return current[index + 1]

        retried_ubatch = False
        ubatch_value = flag_value("-ub")
        if ubatch_value is not None:
            try:
                ubatch = max(16, int(ubatch_value) // 2)
            except ValueError:
                ubatch = None
            if ubatch is not None:
                current[current.index("-ub") + 1] = str(ubatch)
                retried_ubatch = True

        retried_fit = False
        fit_value = flag_value("--fit-target")
        if fit_value is not None:
            try:
                fit_target = int(fit_value) + 512
            except ValueError:
                fit_target = None
            if fit_target is not None:
                current[current.index("--fit-target") + 1] = str(fit_target)
                retried_fit = True

        changes = []
        if retried_ubatch:
            changes.append(f"micro-batch {ubatch}")
        if retried_fit:
            changes.append("a larger VRAM margin")
        print(
            "\nKestrel: CUDA OOM during startup; retrying with "
            + (" and ".join(changes) if changes else "the same settings") + ".",
            file=sys.stderr,
        )
        print("  " + shlex.join(current), file=sys.stderr)
    return 1


def _cached_gguf_path(source_path: str) -> str:
    return source_path.rstrip(os.sep) + ".gguf"


def _human_stream(args) -> TextIO:
    """Stream for human-readable output: stderr under ``--json`` so that
    stdout stays a single parseable JSON document (the agent contract)."""
    return sys.stderr if getattr(args, "json", False) else sys.stdout


def _finish_json(args, result: dict) -> int:
    """Emit ``result`` as JSON on stdout and return its exit code.

    Prints nothing when ``--json`` is not set, so non-JSON callers keep their
    existing human output and this is a no-op.
    """
    if getattr(args, "json", False):
        sys.stdout.write(json.dumps(result) + "\n")
    return int(result.get("exit_code", 0))


def _print_failure(exc, *, json_output: bool) -> int:
    """Render a :class:`KestrelError` consistently and return its exit code.

    Under ``--json`` the failure is a single stable document on **stdout**:
    ``{"error": {"code", "message", "hint"}}``. Otherwise a ``Error:`` prose
    line plus an optional hint go to stderr.
    """
    if json_output:
        print(json.dumps({"error": exc.as_dict()}, indent=2, default=str))
    else:
        print(f"Error: {exc.message}", file=sys.stderr)
        if getattr(exc, "hint", None):
            print(f"  Hint: {exc.hint}", file=sys.stderr)
    return getattr(exc, "exit_code", 1)


def _oneshot_run(backend, cmd: list[str], args) -> int:
    """Run a one-shot generation and return its exit code.

    The completion text is always shown on the human stream; under ``--json``
    a structured result (output + token counts + timings + command) is the
    only thing written to stdout.
    """
    oneshot = backend._build_cmd(args.prompt, int(args.max_tokens))
    try:
        output = backend.generate(args.prompt, int(args.max_tokens))
    except (RuntimeError, FileNotFoundError) as exc:
        return _finish_json(
            args,
            {
                "model": args.model,
                "status": "error",
                "error": str(exc),
                "exit_code": 1,
                "command": oneshot,
            },
        )
    metrics = backend.last_metrics
    print(output, file=_human_stream(args))
    return _finish_json(
        args,
        {
            "model": args.model,
            "status": "ok",
            "exit_code": metrics.returncode,
            "duration_s": round(metrics.elapsed_seconds, 3),
            "prompt_tokens": metrics.prompt_tokens,
            "output_tokens": metrics.output_tokens,
            "prompt_tokens_per_second": metrics.prompt_tokens_per_second,
            "output_tokens_per_second": metrics.output_tokens_per_second,
            "command": oneshot,
            "output": output,
        },
    )


def _wait_ready(host: str, port: int, *, timeout: float, interval: float = 0.25) -> bool:
    """Poll ``/health`` (then ``/v1/health``) until llama-server reports ready
    or ``timeout`` expires.

    Returns True only after a confirmed HTTP 200. Best-effort: connection
    refusals (server still booting) are retried, not treated as failures.
    """
    import urllib.error
    import urllib.request

    deadline = time.perf_counter() + timeout
    while True:
        for path in ("/health", "/v1/health"):
            url = f"http://{host}:{port}{path}"
            try:
                with urllib.request.urlopen(url, timeout=max(0.5, interval)) as resp:
                    if resp.status == 200:
                        return True
            except (urllib.error.URLError, TimeoutError, OSError):
                pass
        if time.perf_counter() >= deadline:
            return False
        time.sleep(interval)


def _resolve_ollama_native(name: str) -> str | None:
    """Resolve an Ollama model to its local GGUF blob so Kestrel can run it
    through its own planned llama.cpp runtime instead of delegating to the
    Ollama daemon.

    Returns ``None`` for cloud-hosted models (no reusable local GGUF), which
    then keep the ``ollama run`` passthrough.
    """
    try:
        from .model_store import resolve_ollama_blob

        blob = resolve_ollama_blob(name)
    except Exception:
        return None
    return str(blob) if blob is not None else None


def _prepare_run_model(args):
    """Resolve the requested model, converting/placing it for the Kestrel
    runtime and producing the fitting config.

    Falls back to passthrough for cloud-only Ollama models, raising
    ``SystemExit`` exactly where the old ``cmd_run`` did. Returns ``None``
    when the Ollama adapter has taken over in dry-run mode, otherwise a
    ``(model_info, config)`` pair."""
    if not args.model:
        args.model = _default_model(
            args,
            error="Error: no model selected. Pass a GGUF path or set KESTREL_MODEL.",
        )
    if args.model.startswith("ollama://"):
        name = args.model.removeprefix("ollama://")
        blob = _resolve_ollama_native(name)
        if blob is not None:
            out = _human_stream(args)
            print(
                ui.dim(
                    f"Ollama model {name} resolved to local GGUF: {blob}"
                ),
                file=out,
            )
            print(
                ui.dim(
                    "Running through the Kestrel runtime with fit and context "
                    "guardrails (Ollama runtime bypassed)."
                ),
                file=out,
            )
            args.model = blob
        else:
            if args.dry_run:
                out = _human_stream(args)
                print("Runtime: Ollama adapter", file=out)
                print("Command: " + shlex.join(["ollama", "run", name]), file=out)
                print(
                    "Note: context and placement are owned by the Ollama runtime for this adapter.",
                    file=out,
                )
                return None
            try:
                result = subprocess.run(["ollama", "run", name])
            except FileNotFoundError as exc:
                raise SystemExit("Error: Ollama is not installed") from exc
            raise SystemExit(result.returncode)
    model_info = detect_model(args.model)
    if model_info is None:
        raise MissingModelError(
            f"could not resolve model or GGUF path: {args.model}",
            hint="check the model name and run `kestrel models list`",
        )
    if model_info["type"] == "safetensors" and not model_info["path"]:
        raise MissingModelError(
            "model source exists but its safetensors are not downloaded",
            hint=f"huggingface-cli download {model_info['hub_id']}",
        )

    gpu_info = detect_gpu()
    args._gpu = gpu_info
    out = _human_stream(args)
    print(ui.kv("Model", args.model, value_color=ui.bold), file=out)
    if gpu_info:
        print(
            ui.kv(
                "GPU",
                f"{gpu_info['name']} ({gpu_info['vram_free_mb']}/{gpu_info['vram_total_mb']} MiB free)",
                value_color=ui.green,
            ),
            file=out,
        )
    else:
        print(ui.kv("GPU", "not detected; planning a CPU-compatible launch", value_color=ui.yellow), file=out)

    if model_info["type"] == "safetensors":
        output = _cached_gguf_path(model_info["path"])
        if os.path.isfile(output):
            print(ui.kv("Cached GGUF", output))
        elif args.no_convert:
            raise SystemExit(
                "Model is safetensors but no cached GGUF exists. "
                "Remove --no-convert or pass a GGUF file."
            )
        else:
            from .gguf.converter import NVFP4Converter

            converter = NVFP4Converter(model_info["path"], include_mtp=False)
            converter.convert(output)
        model_info = {"type": "gguf", "path": output, "gguf_name": output}

    config = estimate_config(model_info, gpu_info, args)
    return model_info, config


def _print_run_plan(args, config, cmd, llama_cli_version, *, hot_model_path=None, cold_model_path=None):
    """Print the explainable runtime plan, notes and command block before
    launching the interactive/runtime execution phase of ``run``."""
    out = _human_stream(args)
    mtp_enabled = "--spec-type" in cmd
    direct_io = bool(args.direct_io and "--direct-io" in cmd)
    plan_lines = [
        ui.kv("Model size", f"{config['model_size_gib']:.2f} GiB", value_color=ui.bold),
        ui.kv("GPU layers", f"{config['gpu_layers']} (llama.cpp fit enabled)", value_color=ui.cyan),
        ui.kv(
            "Load path",
            "direct I/O (uncached)" if direct_io else "mmap",
            value_color=ui.cyan if direct_io else None,
        ),
        ui.kv("VRAM safety margin", f"{args.fit_target or config['fit_target_mib']} MiB"),
        ui.kv("CPU MoE", "enabled" if config["cpu_moe"] else "disabled",
              value_color=ui.green if config["cpu_moe"] else ui.dim),
        ui.kv(
            "MoE cache",
            f"{config['moe_cache']} ({config['moe_cache_budget_mib']} MiB budget)",
        ),
    ]
    if hot_model_path:
        plan_lines.append(ui.kv("MoE Q4 hot sidecar", hot_model_path))
    if cold_model_path:
        plan_lines.append(ui.kv("MoE Q1 cold sidecar", cold_model_path))
    planned_threads = (
        args.threads
        if args.threads is not None
        else config["threads"] or "llama.cpp default"
    )
    plan_lines.extend(
        [
            ui.kv("Threads", str(planned_threads)),
            ui.kv("Context", f"{config['context_size']} ({config['context_reason']})"),
            ui.kv("KV cache", str(args.kv_cache_type)),
            ui.kv(
                "Batch / micro-batch",
                f"{args.batch_size or config['batch_size']} / "
                f"{args.ubatch_size or config['ubatch_size']}",
            ),
            ui.kv("MTP", "enabled" if mtp_enabled else "disabled",
                  value_color=ui.green if mtp_enabled else ui.dim),
        ]
    )
    predicted = config["predicted_decode_tps"]
    plan_lines.append(
        ui.kv(
            "Decode estimate",
            f"{predicted} tok/s ({config['prediction_confidence']}; benchmark required)",
            value_color=ui.green if (predicted and predicted >= 10) else ui.yellow,
        )
    )
    print(ui.box("Runtime plan", "\n".join(plan_lines)), file=out)
    notes = []
    if config.get("memory_overcommit"):
        notes.append(
            "This model file is larger than the free RAM + swap on this host. "
            "Context was reduced to the minimum to avoid an OOM crash, but "
            "expect heavy disk paging and low speed. Free memory or use a "
            "smaller quant/tier to run it comfortably."
        )
    if predicted and predicted < 10:
        if cold_model_path:
            notes.append(
                "Below the 10 tok/s release floor. This estimate uses the measured "
                "Q1 cold-fallback rate scaled to all model layers. Reaching the "
                "release floor still needs a native fused hybrid expert path or "
                "hardware that can hold the active working set."
            )
        else:
            notes.append(
                "Below the 10 tok/s release floor. Decode is bound by CPU expert "
                "work and expert traffic on this hardware. Reaching the release "
                "floor needs a native fused hybrid expert path or hardware that can "
                "hold the active working set."
            )
    if config["has_mtp"] and not mtp_enabled and not args.no_mtp:
        if not config["use_mtp"]:
            notes.append("MTP auto-disabled because this memory profile is slower or OOM-prone")
        else:
            notes.append(f"this llama.cpp build does not expose draft-mtp ({llama_cli_version})")
    for note in notes:
        print(f"  {ui.warn_mark()} {ui.yellow(note)}", file=out)
    command_lines = []
    if hot_model_path:
        command_lines.append(f"  LLAMA_MOE_HOT_GGUF={shlex.quote(hot_model_path)}")
    command_lines.append("  " + shlex.join(cmd))
    print(ui.box("Command", "\n".join(command_lines), title_color=ui.cyan), file=out)


def cmd_run(args):
    prepared = _prepare_run_model(args)
    if prepared is None:
        return
    model_info, config = prepared
    run_env = None
    hot_model_path = None
    cold_model_path = None
    if args.moe_hot_model and args.moe_cold_model:
        raise SystemExit("Error: --moe-hot-model and --moe-cold-model are mutually exclusive")
    if args.moe_hot_model:
        hot_model_path = str(Path(args.moe_hot_model).expanduser().resolve())
        if not Path(hot_model_path).is_file():
            raise SystemExit(f"Error: MoE hot sidecar does not exist: {hot_model_path}")
        if config["moe_cache"] != "on":
            raise SystemExit(
                "Error: --moe-hot-model requires --moe-cache on or an explicit MiB budget"
            )
        run_env = os.environ.copy()
        run_env.pop("LLAMA_MOE_COLD_GGUF", None)
        run_env["LLAMA_MOE_HOT_GGUF"] = hot_model_path
    if args.moe_cold_model:
        cold_model_path = str(Path(args.moe_cold_model).expanduser().resolve())
        if not Path(cold_model_path).is_file():
            raise SystemExit(f"Error: MoE cold sidecar does not exist: {cold_model_path}")
        if config["moe_cache"] != "on":
            raise SystemExit(
                "Error: --moe-cold-model requires --moe-cache on or an explicit MiB budget"
            )
        run_env = os.environ.copy()
        run_env.pop("LLAMA_MOE_HOT_GGUF", None)
        run_env["LLAMA_MOE_COLD_GGUF"] = cold_model_path
    backend = _configure_backend(model_info, config, args)
    cmd = backend._build_interactive_cmd()
    llama_cli_version = backend.capabilities().version

    _print_run_plan(
        args,
        config,
        cmd,
        llama_cli_version,
        hot_model_path=hot_model_path,
        cold_model_path=cold_model_path,
    )
    direct_io = bool(args.direct_io and "--direct-io" in cmd)
    if args.dry_run:
        return _finish_json(
            args,
            {
                "model": args.model,
                "dry_run": True,
                "model_path": model_info["path"],
                "command": cmd,
            },
        )
    if args.prompt:
        return _oneshot_run(backend, cmd, args)
    if args.warm_cache:
        if direct_io:
            print(
                ui.dim(
                    "--warm-cache has no effect with --direct-io "
                    "(direct I/O bypasses the page cache)."
                ),
                file=_human_stream(args),
            )
        else:
            warm_paths = [model_info["path"]]
            if cold_model_path:
                warm_paths.append(cold_model_path)
            if hot_model_path:
                warm_paths.append(hot_model_path)
            print(ui.dim("Warming the model page cache before launch..."), file=_human_stream(args))
            _warm_page_cache(warm_paths)
    retries = 0 if args.no_oom_retry else 2
    rc = _run_with_oom_retries(cmd, max_retries=retries, env=run_env)
    return _finish_json(args, {"model": args.model, "exit_code": rc})


def _build_server_cmd(model_info: dict, config: dict, args=None) -> list[str]:
    """Build a ``llama-server`` command from a planned runtime configuration.

    Delegates argument assembly to ``LlamaCppBackend.build_server_cmd`` so the
    server path shares the exact engine tuning flags as interactive runs
    (including ``--fit``, which was previously dropped and let serve OOM where
    run would not).
    """
    backend = _configure_backend(model_info, config, args)
    return backend.build_server_cmd(
        host=getattr(args, "host", "127.0.0.1"),
        port=getattr(args, "port", 8080),
        alias=getattr(args, "alias", None),
        embeddings=bool(args and args.embeddings),
    )


def cmd_serve(args):
    """Serve the planned model over an OpenAI-compatible llama-server endpoint."""
    if not args.model:
        args.model = _default_model(
            args,
            error="Error: no model selected. Pass a GGUF path or set KESTREL_MODEL.",
        )
    if args.model.startswith("ollama://"):
        name = args.model.removeprefix("ollama://")
        blob = _resolve_ollama_native(name)
        if blob is not None:
            print(
                ui.dim(
                    f"Ollama model {name} resolved to local GGUF: {blob} "
                    "(serving via Kestrel's llama-server runtime)"
                ),
                file=_human_stream(args),
            )
            args.model = blob
        else:
            raise SystemExit(
                "Error: kestrel serve hosts GGUF models through llama-server. "
                "For Ollama models use `ollama serve`/`ollama run`."
            )
    model_info = detect_model(args.model)
    if model_info is None:
        raise MissingModelError(
            f"could not resolve model or GGUF path: {args.model}",
            hint="check the model name and run `kestrel models list`",
        )
    if model_info["type"] == "safetensors" and not model_info["path"]:
        raise MissingModelError(
            "model source exists but its safetensors are not downloaded",
            hint=f"huggingface-cli download {model_info['hub_id']}",
        )
    if model_info["type"] == "safetensors":
        output = _cached_gguf_path(model_info["path"])
        if os.path.isfile(output):
            print(ui.kv("Cached GGUF", output))
        elif args.no_convert:
            raise SystemExit(
                "Model is safetensors but no cached GGUF exists. "
                "Remove --no-convert or pass a GGUF file."
            )
        else:
            from .gguf.converter import NVFP4Converter

            converter = NVFP4Converter(model_info["path"], include_mtp=False)
            converter.convert(output)
        model_info = {"type": "gguf", "path": output, "gguf_name": output}

    gpu_info = detect_gpu()
    args._gpu = gpu_info
    config = estimate_config(model_info, gpu_info, args)
    cmd = _build_server_cmd(model_info, config, args)
    host = args.host
    port = args.port
    out = _human_stream(args)
    print(ui.box(
        "Serving",
        "\n".join([
            ui.kv("Model", model_info["path"], value_color=ui.bold),
            ui.kv("Engine", "llama-server"),
            ui.kv("URL", f"http://{host}:{port}", value_color=ui.cyan),
            ui.kv("Context", f"{config['context_size']} ({config['context_reason']})"),
            ui.kv("GPU layers", str(config["gpu_layers"])),
            ui.kv("CPU MoE", "enabled" if config["cpu_moe"] else "disabled"),
            "",
            "OpenAI-compatible endpoints: /v1/chat/completions, /v1/completions, /v1/models",
            "Press Ctrl+C to stop the server.",
        ]),
        title_color=ui.green,
    ), file=out)
    if args.dry_run:
        print("  " + shlex.join(cmd), file=out)
        return _finish_json(
            args,
            {
                "model": args.model,
                "dry_run": True,
                "url": f"http://{host}:{port}",
                "command": cmd,
            },
        )
    if not args.wait and not getattr(args, "json", False):
        raise SystemExit(_run_with_oom_retries(list(cmd)))
    proc = subprocess.Popen(list(cmd))
    result = {
        "model": args.model,
        "url": f"http://{host}:{port}",
        "host": host,
        "port": port,
        "command": cmd,
    }
    timeout = float(args.wait) if args.wait else 30.0
    print(
        ui.dim(f"Waiting for the server on http://{host}:{port}/health ..."),
        file=out,
    )
    started = time.perf_counter()
    ready = _wait_ready(host, port, timeout=timeout)
    elapsed = round(time.perf_counter() - started, 3)
    if not ready:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise SystemExit(
            _print_failure(
                ServiceError(
                    f"llama-server did not become ready on "
                    f"http://{host}:{port}/health within {timeout:.0f}s.",
                    hint=(
                        "Check the llama-server output above for load errors "
                        "(CUDA OOM, missing GGUF, or an incompatible quant) and "
                        "free GPU memory before retrying."
                    ),
                ),
                json_output=bool(getattr(args, "json", False)),
            )
        )
    result["status"] = "ready"
    result["exit_code"] = 0
    result["ready_after_s"] = elapsed
    print(ui.green("Server is ready."), file=out)
    _finish_json(args, result)
    proc.wait()
    raise SystemExit(proc.returncode)


def cmd_build(_args):
    print(f"Building llama.cpp at {LLAMA_CPP_DIR}...")
    build_dir = os.path.join(LLAMA_CPP_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)
    subprocess.run(
        [
            "cmake",
            "..",
            "-DLLAMA_CUDA=ON",
            "-DLLAMA_CUDA_NVFP4=ON",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        cwd=build_dir,
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", ".", "-j", str(os.cpu_count() or 4)],
        cwd=build_dir,
        check=True,
    )


def _engine_dir(args) -> str:
    from .backends.llama_cpp import default_llama_cpp_dir

    directory = getattr(args, "dir", None)
    if directory:
        return str(Path(directory).expanduser().resolve())
    override = os.environ.get("KESTREL_LLAMA_CPP_DIR")
    if override:
        return override
    if USER_CONFIG.llama_cpp_dir:
        return USER_CONFIG.llama_cpp_dir
    return default_llama_cpp_dir()


def cmd_engine(args):
    from . import engine

    directory = _engine_dir(args)
    command = args.engine_command
    out = _human_stream(args)
    js = bool(getattr(args, "json", False))

    if command == "status":
        try:
            status = engine.engine_status(
                directory, check_remote=not getattr(args, "no_remote", False)
            )
        except engine.EngineError:
            raise
        if js:
            print(json.dumps(status, indent=2, default=str))
            return
        rows: list[tuple[str, str, object]] = [
            ("Directory", status["directory"], None),
            ("Git", "yes" if status["git"] else "no", None),
            ("Remote", status["remote"] or "—", None),
            ("Branch", status["branch"] or "detached", None),
            ("Commit", (status["commit"] or "—")[:12], None),
            ("Managed", "yes" if status["manifest"] else "no", None),
        ]
        if status["git"] and status["remote"]:
            if status["behind"] is None:
                rows.append(("Stale", "unknown (remote unreachable)", ui.dim))
            elif status["stale"]:
                rows.append(
                    (
                        "Stale",
                        f"{status['behind']} behind upstream — run `kestrel engine update`",
                        ui.red,
                    )
                )
            else:
                rows.append(("Up to date", "current", None))
        if status["dirty"]:
            rows.append(("Dirty", "uncommitted changes", ui.yellow))
        if status.get("artifacts"):
            present = [name for name, artifact in status["artifacts"].items() if artifact["exists"]]
            rows.append(("Built", ", ".join(present) or "none", None))
        body = "\n".join(ui.kv(key, value, value_color=color) for key, value, color in rows)
        print(ui.box("Engine", body, title_color=ui.cyan), file=out)
        return

    try:
        if command == "update":
            result = engine.update(
                directory,
                dry_run=getattr(args, "dry_run", False),
                force=getattr(args, "force", False),
                remote=getattr(args, "remote", None),
            )
        elif command == "rebuild":
            result = engine.rebuild(directory, dry_run=getattr(args, "dry_run", False))
        elif command == "set":
            result = {"status": "adopted", "directory": directory, **engine.adopt(directory, getattr(args, "remote", None)).as_dict()}
        else:
            raise SystemExit(f"unknown engine subcommand: {command}")
    except engine.EngineError:
        raise

    if js:
        print(json.dumps(result, indent=2, default=str))
        return
    if result.get("status") == "up_to_date":
        print(ui.kv("Engine", "already up to date", value_color=ui.green), file=out)
        return
    if result.get("status") == "dry_run":
        print(
            ui.kv("Engine", "rebuild planned (dry run)"),
            file=out,
        )
    for key, value in result.items():
        if key in ("directory", "cmake_flags", "targets", "artifacts", "status"):
            continue
        print(ui.kv(key.replace("_", " ").title(), value), file=out)
    print(ui.kv("Status", result.get("status", "")), file=out)


def cmd_self_update(args):
    """Update the running Kestrel package from its source tree or a wheel.

    ``--wheel`` may be a local path or an ``https://`` URL. When ``--sha256``
    is given the artifact is verified before install (trusted distribution).
    The currently-installed package is snapshotted first and restored on a
    failed post-install check, so an update can never leave Kestrel broken.
    """
    from hashlib import sha256

    repo = getattr(args, "repo", None) or os.environ.get("KESTREL_REPO")
    wheel = getattr(args, "wheel", None)
    if wheel:
        artifact = _materialize_wheel(wheel)
        source_sha = args.sha256.lower().replace("sha256:", "") if args.sha256 else None
        source_label = str(wheel)
    else:
        if repo is None:
            root = Path(__file__).resolve().parents[1]
            repo = str(root) if (root / ".git").is_dir() else None
        if repo is None:
            raise SystemExit(
                "Error: cannot locate the Kestrel repository (set --repo or --wheel)"
            )
        artifact, source_sha = repo, None
        source_label = repo

    from_version = _kestrel_version()
    out = _human_stream(args)
    js = bool(getattr(args, "json", False))
    plan = {
        "status": "planned",
        "from_version": from_version,
        "to_version": None,
        "rolled_back": False,
        "source": source_label,
        "dry_run": bool(args.dry_run),
    }

    if source_sha:
        actual = sha256(Path(artifact).read_bytes()).hexdigest()
        if actual != source_sha:
            from .errors import IntegrityError

            raise IntegrityError(
                f"SHA256 mismatch for {source_label}",
                hint="refusing to install an unverified artifact",
            )

    if args.dry_run:
        plan["status"] = "dry_run"
        plan["to_version"] = "pending"
        if js:
            print(json.dumps(plan, indent=2, default=str))
            return
        print(ui.kv("Source", source_label, value_color=ui.bold), file=out)
        print(f"  Would install Kestrel from {source_label} via pip.", file=out)
        return

    backup = _snapshot_installed()
    print(ui.kv("Source", source_label, value_color=ui.bold), file=out)
    install = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            artifact,
        ],
        capture_output=True,
        text=True,
    )
    if install.returncode != 0:
        raise SystemExit(f"Error: install failed:\n{install.stderr[-2000:]}")

    ok, version = _post_install_check()
    rolled_back = False
    if not ok and backup is not None:
        _restore_install(backup)
        ok, version = _post_install_check()
        rolled_back = True
    if not ok:
        from .errors import IntegrityError

        raise IntegrityError(
            f"post-install verification failed for {source_label}",
            hint="Kestrel may require a manual reinstall; the previous install was not restorable",
        )

    if js:
        plan["status"] = "rolled_back" if rolled_back else "updated"
        plan["to_version"] = version
        plan["rolled_back"] = rolled_back
        print(json.dumps(plan, indent=2, default=str))
        return
    if rolled_back:
        print(f"Installation rolled back; Kestrel still at {version}.", file=out)
    else:
        print(
            f"Updated Kestrel to {version}. Restart the shell for changes to take effect.",
            file=out,
        )


def _materialize_wheel(target: str) -> str:
    """Return a local path for ``target`` (downloading https wheels to temp)."""
    if not target.startswith(("http://", "https://")):
        return target
    import tempfile
    import urllib.request

    with urllib.request.urlopen(target, timeout=60) as response:
        suffix = Path(target).suffix or ".whl"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            shutil.copyfileobj(response, handle)
            return handle.name


def _snapshot_installed():
    """Copy the installed ``kestrel`` package for rollback, or ``None``.

    Editable (source-tree) installs are not snapshotted — the source tree is
    not modified by ``pip install`` so it needs no rollback.
    """
    import shutil
    import tempfile

    try:
        import kestrel
    except ImportError:
        return None
    src = Path(kestrel.__file__).resolve().parent
    repo_root = Path(__file__).resolve().parents[1]
    if not src.is_dir() or src.is_relative_to(repo_root):
        return None
    backup_root = Path(tempfile.mkdtemp(prefix="kestrel-snapshot-"))
    backup = backup_root / src.name
    shutil.copytree(src, backup)
    return {"from": src, "to": backup}


def _restore_install(snapshot) -> None:
    import shutil

    src, backup = snapshot["from"], snapshot["to"]
    shutil.rmtree(src, ignore_errors=True)
    shutil.copytree(backup, src)


def _post_install_check() -> tuple[bool, str]:
    """Return ``(ok, version)`` by loading kestrel in a fresh interpreter."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.metadata as m; print(m.version('kestrel'))",
        ],
        capture_output=True,
        text=True,
    )
    version = result.stdout.strip()
    return (result.returncode == 0 and bool(re.match(r"^\d+(\.\d+)+", version))), version


def cmd_convert(args):
    model_info = detect_model(args.model)
    if not model_info or model_info["type"] != "safetensors" or not model_info["path"]:
        raise SystemExit("Error: conversion input must be a downloaded safetensors model")
    output = args.output or _cached_gguf_path(model_info["path"])
    if args.generic:
        from .gguf.converter import generic_convert_hf_to_gguf

        try:
            _command, _code = generic_convert_hf_to_gguf(
                model_info["path"],
                output,
                outtype=args.outtype,
                llama_cpp_dir=LLAMA_CPP_DIR,
            )
        except FileNotFoundError as exc:
            raise SystemExit(f"Error: {exc}") from exc
        except RuntimeError as exc:
            raise SystemExit(f"Error: {exc}") from exc
        print(f"Converted {model_info['path']} → {output}")
        return
    from .gguf.converter import NVFP4Converter

    NVFP4Converter(
        model_info["path"],
        include_mtp=args.include_mtp,
        dense_q4=args.dense_q4,
        cold_tier=args.cold_tier,
        q4_sidecar_source=args.q4_sidecar_source,
        experts_only=args.experts_only,
        q2_edge_layers=args.q2_edge_layers,
        compact_expert_type=args.compact_expert_type,
        imatrix_path=args.imatrix,
        conversion_workers=args.conversion_workers,
        experts_keep=args.experts_keep,
        expert_importance=args.expert_importance,
    ).convert(output)


# byte thresholds for the disk-space sanity checks (fail below CRITICAL,
# warn below the softer bound; GGs are routinely many GiB).
_CHECK_DISK_CRITICAL = 1 << 30  # 1 GiB
_CHECK_DISK_WARN = 5 << 30      # 5 GiB


def _fmt_bytes(n: int) -> str:
    """Human-friendly byte count (binary units)."""
    f = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if f < 1024 or unit == "TiB":
            return f"{f:.1f} {unit}" if unit != "B" else f"{int(f)} B"
        f /= 1024
    return f"{f:.1f} TiB"


def _writable_probe(path: Path) -> tuple[bool, str]:
    """Return (True, "") if a real write succeeds in ``path``'s directory."""
    try:
        target = Path(path).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        probe = target / f".kestrel-write-probe-{os.getpid()}"
        with open(probe, "w") as fh:
            fh.write("ok")
        probe.unlink()
        return True, ""
    except OSError as exc:
        return False, str(exc)


def _doctor_checks(
    backend,
    cli_caps,
    cli_error: str | None,
    server_caps=None,
    server_error: str | None = None,
) -> list[dict]:
    """Extra `doctor` sanity checks shared by the human and JSON rendering.

    Each entry is ``{"name", "status" (ok|warn|fail), "message"}``. The
    llama.cpp compatibility probe falls back to ``llama-server`` when the
    ``llama-cli`` binary is unavailable.
    """
    from .model_store import default_models_dir
    from .util import available_disk_bytes

    models_dir = (
        Path(USER_CONFIG.models_dir).expanduser()
        if USER_CONFIG.models_dir
        else default_models_dir()
    )
    engine_dir = Path(backend.llama_cpp_dir)
    checks: list[dict] = []

    for label, path in (("models dir", models_dir), ("llama.cpp dir", engine_dir)):
        free = available_disk_bytes(path)
        name = f"disk free ({label})"
        if free is None:
            checks.append(
                {
                    "name": name,
                    "status": "warn",
                    "message": f"could not determine free space for {path}",
                }
            )
        elif free < _CHECK_DISK_CRITICAL:
            checks.append(
                {
                    "name": name,
                    "status": "fail",
                    "message": f"only {_fmt_bytes(free)} free under {path}",
                }
            )
        elif free < _CHECK_DISK_WARN:
            checks.append(
                {
                    "name": name,
                    "status": "warn",
                    "message": f"{_fmt_bytes(free)} free under {path}",
                }
            )
        else:
            checks.append(
                {
                    "name": name,
                    "status": "ok",
                    "message": f"{_fmt_bytes(free)} free under {path}",
                }
            )

    for label, path in (("config path", config_path().parent), ("models dir", models_dir)):
        ok, err = _writable_probe(path)
        checks.append(
            {
                "name": f"writable ({label})",
                "status": "ok" if ok else "fail",
                "message": f"{path} is writable" if ok else f"{path} not writable: {err}",
            }
        )

    if cli_caps is None and server_caps is None:
        checks.append(
            {
                "name": "llama.cpp compatibility",
                "status": "fail",
                "message": (
                    f"llama-cli/llama-server unavailable: "
                    f"{cli_error or server_error}"
                ),
            }
        )
    else:
        caps = cli_caps if cli_caps is not None else server_caps
        body = caps.help_text
        fallback = "" if cli_caps is not None else " (via llama-server)"
        checks.append(
            {
                "name": "llama.cpp compatibility",
                "status": "ok",
                "message": (
                    f"version {caps.version}{fallback}, "
                    f"nvfp4={'yes' if ('nvfp4' in body or 'nv-fp4' in body) else 'no'}, "
                    f"q1_0={'yes' if ('q1_0' in body or 'q1-moe' in body) else 'no'}"
                ),
            }
        )

    default = USER_CONFIG.default_model
    if not default:
        checks.append(
            {
                "name": "model cache integrity",
                "status": "warn",
                "message": "no default model configured",
            }
        )
    else:
        info = detect_model(default)
        path = (info or {}).get("path")
        if not path or not os.path.isfile(path):
            checks.append(
                {
                    "name": "model cache integrity",
                    "status": "warn",
                    "message": f"configured model {default} not found on disk",
                }
            )
        else:
            gguf = Path(path)
            try:
                size = gguf.stat().st_size
                if size <= 0:
                    checks.append(
                        {
                            "name": "model cache integrity",
                            "status": "fail",
                            "message": f"{gguf} reports zero size",
                        }
                    )
                else:
                    with gguf.open("rb") as handle:
                        magic = handle.read(4)
                    if magic == b"GGUF":
                        checks.append(
                            {
                                "name": "model cache integrity",
                                "status": "ok",
                                "message": (
                                    f"{gguf.name} {_fmt_bytes(size)}, GGUF header verified"
                                ),
                            }
                        )
                    else:
                        checks.append(
                            {
                                "name": "model cache integrity",
                                "status": "fail",
                                "message": (
                                    f"{gguf.name} header magic {magic!r} != GGUF"
                                ),
                            }
                        )
            except OSError as exc:
                checks.append(
                    {
                        "name": "model cache integrity",
                        "status": "fail",
                        "message": f"{gguf} unreadable: {exc}",
                    }
                )
    return checks


def _checks_status(checks: list[dict]) -> str:
    """Aggregate per-check statuses into one overall ok/warn/fail."""
    if any(c.get("status") == "fail" for c in checks):
        return "fail"
    if any(c.get("status") == "warn" for c in checks):
        return "warn"
    return "ok"


def cmd_doctor(args):
    gpu = detect_gpu()
    memory = _memory_snapshot()
    power = _cpu_power_policy()
    backend = LlamaCppBackend("", llama_cpp_dir=LLAMA_CPP_DIR)
    cli_caps = None
    cli_error = None
    try:
        cli_caps = backend.capabilities()
    except (RuntimeError, subprocess.SubprocessError) as exc:
        cli_error = str(exc)
    server_caps = None
    server_error = None
    try:
        server_caps = backend.server_capabilities()
    except (RuntimeError, subprocess.SubprocessError) as exc:
        server_error = str(exc)

    checks = _doctor_checks(
        backend, cli_caps, cli_error, server_caps, server_error
    )
    overall_status = _checks_status(checks)
    checks_failed = overall_status == "fail"

    if getattr(args, "json", False):
        payload = {
            "status": overall_status,
            "checks": checks,
            "host": {
                "python": sys.version.split()[0],
                "gpu": gpu,
                "memory": memory,
                "available_ram_mib": _available_ram_mib(),
                "cpu_power_policy": power,
            },
            "llama_cli": {
                "available": cli_caps is not None,
                "dir": backend.llama_cpp_dir,
                "binary": backend.binary if cli_caps is not None else None,
                "native": (
                    bool(cli_caps is not None and str(Path(backend.binary)).startswith(
                        str(Path(backend.llama_cpp_dir)) + os.sep
                    ))
                ),
                "version": cli_caps.version if cli_caps is not None else None,
                "fit": cli_caps.supports("--fit") if cli_caps is not None else None,
                "cpu_moe": cli_caps.supports("--cpu-moe") if cli_caps is not None else None,
                "mmap": cli_caps.supports("--mmap") if cli_caps is not None else None,
                "direct_io": cli_caps.supports("--direct-io") if cli_caps is not None else None,
                "quantized_kv": (
                    cli_caps.supports("--cache-type-k") if cli_caps is not None else None
                ),
                "mtp": "draft-mtp" in cli_caps.spec_types if cli_caps is not None else None,
                "error": cli_error,
            },
            "llama_server": {
                "available": server_caps is not None,
                "dir": backend.llama_cpp_dir,
                "binary": backend.server_binary if server_caps is not None else None,
                "native": (
                    bool(server_caps is not None and str(Path(backend.server_binary)).startswith(
                        str(Path(backend.llama_cpp_dir)) + os.sep
                    ))
                ),
                "version": server_caps.version if server_caps is not None else None,
                "moe_cache": (
                    server_caps.supports("--moe-cache") if server_caps is not None else None
                ),
                "spec_types": sorted(server_caps.spec_types) if server_caps is not None else [],
                "error": server_error,
            },
        }
        print(json.dumps(payload, indent=2))
        if not (cli_caps is not None or server_caps is not None):
            raise SystemExit(1)
        if checks_failed:
            raise SystemExit(1)
        return

    def mark(ok: bool) -> str:
        return ui.pass_mark() if ok else ui.fail_mark()

    swap_warning = memory["swap_total_mib"] and memory["swap_used_mib"] > memory["swap_total_mib"] // 2
    host_lines = [
        ui.kv("Python", sys.version.split()[0]),
        ui.kv(
            "GPU",
            gpu["name"] if gpu else "not detected",
            value_color=ui.green if gpu else ui.dim,
        ),
        ui.kv("Available RAM", f"{_available_ram_mib()} MiB", value_color=ui.cyan),
        ui.kv(
            "Swap",
            f"{memory['swap_used_mib']}/{memory['swap_total_mib']} MiB used",
            value_color=ui.yellow if swap_warning else None,
        ),
        ui.kv(
            "CPU policy",
            f"governor={power['governor'] or 'unknown'}, "
            f"EPP={power['energy_performance_preference'] or 'unknown'}, "
            f"turbo={power['turbo_enabled']}",
        ),
    ]
    if swap_warning:
        host_lines.append(
            f"  {ui.warn_mark()} swap is over 50% used; cold large-model benchmarks may be distorted"
        )
    print(ui.box("Host", "\n".join(host_lines)))

    cli_lines = [ui.kv("llama.cpp dir", backend.llama_cpp_dir)]
    cli_available = cli_caps is not None
    if cli_caps is not None:
        cli_native = str(Path(backend.binary)).startswith(
            str(Path(backend.llama_cpp_dir)) + os.sep
        )
        cli_label = "" if cli_native else " (fallback from another build)"
        cli_lines.append(ui.kv("llama-cli", f"{backend.binary}{cli_label}", value_color=ui.bold))
        cli_lines.append(ui.kv("version", cli_caps.version))
        cli_lines.append(f"  {mark(cli_caps.supports('--fit'))} automatic fitting")
        cli_lines.append(f"  {mark(cli_caps.supports('--cpu-moe'))} CPU MoE")
        cli_lines.append(f"  {mark(cli_caps.supports('--mmap'))} mmap")
        cli_lines.append(f"  {mark(cli_caps.supports('--direct-io'))} direct I/O load")
        cli_lines.append(f"  {mark(cli_caps.supports('--cache-type-k'))} quantized KV cache")
        cli_lines.append(f"  {mark('draft-mtp' in cli_caps.spec_types)} MTP")
    else:
        cli_lines.append(f"  {ui.fail_mark()} llama-cli: unavailable ({cli_error})")
    print(ui.box(
        "llama-cli",
        "\n".join(cli_lines),
        title_color=ui.green if cli_available else ui.yellow,
    ))

    server_available = server_caps is not None
    if server_caps is not None:
        server_native = str(Path(backend.server_binary)).startswith(
            str(Path(backend.llama_cpp_dir)) + os.sep
        )
        server_label = "" if server_native else " (fallback from another build)"
        server_lines = [
            ui.kv("llama-server", f"{backend.server_binary}{server_label}", value_color=ui.bold),
            ui.kv("version", server_caps.version),
            f"  {mark(server_caps.supports('--moe-cache'))} MoE cache",
            f"  {mark(bool(server_caps.spec_types))} server spec types: {sorted(server_caps.spec_types) or 'none'}",
        ]
    else:
        server_lines = [f"  {ui.fail_mark()} llama-server: unavailable ({server_error})"]
    print(ui.box(
        "llama-server",
        "\n".join(server_lines),
        title_color=ui.green if server_available else ui.yellow,
    ))

    check_marks = {
        "ok": ui.pass_mark(),
        "warn": ui.warn_mark(),
        "fail": ui.fail_mark(),
    }
    check_lines = [
        f"  {check_marks.get(check['status'], ui.info_mark())} "
        f"{check['name']}: {check['message']}"
        for check in checks
    ]
    print(ui.box(
        f"Checks ({overall_status})",
        "\n".join(check_lines),
        title_color=(
            ui.green
            if overall_status == "ok"
            else (ui.yellow if overall_status == "warn" else ui.red)
        ),
    ))
    if checks_failed:
        raise SystemExit(1)
    if not (cli_available or server_available):
        raise SystemExit(1)


def cmd_status(args):
    """Show the configured runtime and last measured optimization state."""

    from .model_store import default_models_dir

    configured = load_config()
    gpu = detect_gpu()
    profile_path = config_path().with_name("hardware-profile.json")
    profile = None
    profile_error = None
    if profile_path.is_file():
        try:
            profile = json.loads(profile_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            profile_error = str(exc)
    model = configured.default_model
    engine = "ollama" if model and model.startswith("ollama://") else "llama.cpp"
    profile_model_data = (profile.get("model") or {}) if profile else {}
    profile_model = profile_model_data.get("source") or profile_model_data.get("path")
    profile_matches_model = bool(model and profile_model == model)
    payload = {
        "model": model,
        "engine": engine if model else None,
        "models_dir": configured.models_dir or str(default_models_dir()),
        "hardware": {
            "gpu": gpu,
            **_memory_snapshot(),
            "cpu_power_policy": _cpu_power_policy(),
        },
        "optimization_profile": str(profile_path) if profile else None,
        "profile_model": profile_model,
        "profile_matches_model": profile_matches_model,
        "plan": profile.get("plan") if profile else None,
        "benchmark": profile.get("benchmark") if profile else None,
        "profile_error": profile_error,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return
    model_line = ui.kv(
        "Model",
        model or "not configured",
        value_color=ui.green if model else ui.dim,
    )
    engine_line = ui.kv(
        "Engine",
        payload["engine"] or "not selected",
        value_color=ui.cyan if payload["engine"] else ui.dim,
    )
    gpu_line = ui.kv(
        "GPU",
        (gpu or {}).get("name", "not detected"),
        value_color=ui.green if gpu else ui.dim,
    )
    swap_used = payload["hardware"]["swap_used_mib"]
    swap_total = payload["hardware"]["swap_total_mib"]
    swap_warning = swap_total and swap_used > swap_total // 2
    status_lines = [
        model_line,
        engine_line,
        gpu_line,
        ui.kv("Available RAM", f"{payload['hardware']['ram_available_mib']} MiB", value_color=ui.cyan),
        ui.kv(
            "Swap",
            f"{swap_used}/{swap_total} MiB used",
            value_color=ui.yellow if swap_warning else None,
        ),
    ]
    print(ui.box("Kestrel status", "\n".join(status_lines)))
    profile_lines = []
    if profile_error:
        profile_lines.append(f"  {ui.fail_mark()} optimization profile unreadable: {profile_error}")
    elif not profile:
        profile_lines.append(ui.dim("Optimization profile not created; run `kestrel optimize MODEL`"))
    else:
        plan = payload["plan"] or {}
        benchmark = payload["benchmark"] or {"status": "not_run"}
        profile_lines.append(
            f"  {ui.kv('Profile model', profile_model or 'unknown', value_color=ui.bold)}"
            + ("  " + ui.green("(active)") if profile_matches_model else "  " + ui.yellow("(not the configured model)"))
        )
        profile_lines.append(
            ui.kv(
                "Plan",
                f"mode={plan.get('quality_profile', 'auto')}, "
                f"context={plan.get('context_size', 'unknown')}, "
                f"GPU layers={plan.get('gpu_layers', 'unknown')}, "
                f"threads={plan.get('threads', 'unknown')}, "
                f"KV={plan.get('kv_cache_type', plan.get('cache_type_k', 'unknown'))}",
            )
        )
        status = benchmark.get("status", "unknown")
        profile_lines.append(
            ui.kv(
                "Benchmark",
                status,
                value_color=ui.green if status == "measured" else (ui.red if status == "failed" else ui.dim),
            )
        )
        if benchmark.get("status") == "measured":
            profile_lines.append(
                f"  {ui.bullet()} prompt={benchmark.get('prompt_tokens_per_second')} tok/s, "
                f"decode={benchmark.get('decode_tokens_per_second')} tok/s, "
                f"speed floor={'pass' if benchmark.get('release_speed_floor_passed') else 'fail'}, "
                f"quality={benchmark.get('quality_gate', 'not_run')}"
            )
        elif benchmark.get("status") == "failed":
            profile_lines.append(f"  {ui.fail_mark()} {benchmark.get('error', 'unknown benchmark error')}")
    if profile_lines:
        print(ui.box("Optimization profile", "\n".join(profile_lines)))


def cmd_setup(args):
    current = KestrelConfig() if args.reset else load_config()
    model = args.model or current.default_model
    if model:
        if model.startswith("ollama://"):
            from .model_store import ModelStoreError, resolve_ollama_blob

            try:
                # Local and cloud models are both valid through the Ollama
                # adapter; this call also proves the model exists.
                resolve_ollama_blob(model.removeprefix("ollama://"))
            except ModelStoreError as exc:
                raise SystemExit(f"Error: setup could not resolve model: {exc}") from exc
        else:
            detected = detect_model(model)
            if detected is None or (detected["type"] == "safetensors" and not detected["path"]):
                raise SystemExit(f"Error: setup could not resolve model: {model}")
            if detected["path"]:
                model = str(Path(detected["path"]).expanduser().resolve())
    elif _resolve_model_alias("qwen3.5:122b-10a"):
        model = "qwen3.5:122b-10a"

    llama_dir = args.llama_cpp_dir or current.llama_cpp_dir or default_llama_cpp_dir()
    llama_dir = str(Path(llama_dir).expanduser().resolve())
    configured = KestrelConfig(
        default_model=model,
        models_dir=args.models_dir or current.models_dir,
        llama_cpp_dir=llama_dir,
    )
    target = save_config(configured)
    print(ui.box("Kestrel configuration saved", "\n".join([
        ui.kv("File", str(target), value_color=ui.cyan),
        ui.kv(
            "Local model",
            configured.default_model or "not configured",
            value_color=ui.green if configured.default_model else ui.dim,
        ),
        ui.kv("Models directory", configured.models_dir or "platform default"),
        ui.kv("llama.cpp", configured.llama_cpp_dir),
    ])))


def _save_default_model(model: str | Path) -> None:
    current = load_config()
    value = str(model) if str(model).startswith("ollama://") else str(Path(model).resolve())
    save_config(
        KestrelConfig(
            default_model=value,
            models_dir=current.models_dir,
            llama_cpp_dir=current.llama_cpp_dir,
        )
    )


def _models_search(args):
    from .model_store import search_huggingface

    rows = search_huggingface(args.query, limit=args.limit)
    if args.json:
        print(json.dumps({"models": rows}, indent=2))
        return
    print(ui.box(
        f"Model market: {args.query}",
        "\n".join(
            f"  {ui.bold(item['id'])}\n"
            f"  {ui.dim('{} downloads, {} likes, {}'.format(item['downloads'], item['likes'], item['license'] or 'unspecified license'))}"
            for item in rows
        ) or ui.dim("  no results"),
    ))
    print(ui.dim("Inspect model cards and choose a file before downloading; popularity is not a quality score."))


def _models_files(args):
    from .model_store import list_huggingface_ggufs

    rows = list_huggingface_ggufs(args.source)
    if args.json:
        print(json.dumps({"repository": args.source, "files": rows}, indent=2))
        return
    print(ui.box(
        f"GGUF files: {args.source.removeprefix('hf://')}",
        "\n".join(
            f"  {ui.bold(item['path'])}\n"
            f"  {ui.dim('{:.2f} GiB'.format(item['size_bytes'] / 1024**3))}  "
            f"Hub scan: {ui.yellow(item['security_status'])}"
            for item in rows
        )
        if rows
        else ui.dim("  none"),
    ))
    print(ui.dim("Hub scan status is metadata from Hugging Face, not a Kestrel security guarantee."))


def _models_list(args, root):
    from .model_store import (
        complete_gguf_models,
        discover_local_models,
        list_ollama_models,
        model_total_size,
    )

    local = complete_gguf_models(discover_local_models(root))
    ollama = list_ollama_models(resolve_paths=args.resolve)
    payload = {
        "kestrel": [{"path": str(path), "size_bytes": model_total_size(path)} for path in local],
        "ollama": [
            {
                "name": item.name,
                "id": item.model_id,
                "size": item.size,
                "modified": item.modified,
                "local_path": str(item.local_path) if item.local_path else None,
            }
            for item in ollama
        ],
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return
    kestrel_body = "\n".join(
        f"  {ui.bold(item['path'])}  {ui.dim('{:.2f} GiB'.format(item['size_bytes'] / 1024**3))}"
        for item in payload["kestrel"]
    ) or ui.dim("  none")
    print(ui.box(f"Kestrel models ({root})", kestrel_body))
    ollama_rows = [
        [
            item.name,
            item.size,
            str(item.local_path) if item.local_path else "cloud or unresolved",
        ]
        for item in ollama
    ]
    if args.resolve:
        ollama_headers = ["name", "size", "location"]
    else:
        ollama_headers = ["name", "size"]
        ollama_rows = [row[:2] for row in ollama_rows]
    print(ui.box(
        "Ollama models",
        ui.table(ollama_headers, ollama_rows) if ollama_rows else ui.dim("  none"),
    ))


def _models_import(args):
    from .model_store import ModelStoreError, resolve_ollama_blob

    source = args.source
    default_value: str | Path
    if source.startswith("ollama://"):
        name = source.removeprefix("ollama://")
        path = resolve_ollama_blob(name)
        if path is None:
            print(f"Imported {name} through the Ollama provider (no local GGUF blob).")
            if args.set_default:
                _save_default_model(source)
                print("Set as the default Kestrel model.")
            return
        default_value = source
    else:
        path = Path(source).expanduser().resolve()
        default_value = path
    detected = detect_model(str(path))
    if not detected or detected["type"] != "gguf":
        raise ModelStoreError(f"source is not a readable GGUF model: {path}")
    print(f"Imported {path}")
    if args.set_default:
        _save_default_model(default_value)
        print("Set as the default Kestrel model.")


def _models_pull(args):
    from .model_store import (
        ModelStoreError,
        choose_default_gguf,
        discover_local_models,
        pull_huggingface,
        pull_ollama,
    )

    source = args.source
    if source.startswith("ollama://"):
        name = source.removeprefix("ollama://")
        if args.dry_run:
            print(f"Would run: ollama pull {shlex.quote(name)}")
            return
        item = pull_ollama(name)
        print(f"Pulled Ollama model {item.name} ({item.size})")
        if item.local_path:
            print(f"Ollama-managed local blob: {item.local_path}")
            print("Kestrel will use Ollama's compatible engine for this model.")
            if args.set_default:
                _save_default_model(f"ollama://{item.name}")
                print("Set as the default Kestrel model.")
        else:
            print("This model is served remotely by Ollama and has no local GGUF blob.")
        return
    result = pull_huggingface(
        source,
        filename=args.file,
        include=args.include,
        revision=args.revision,
        destination=Path(args.destination).expanduser() if args.destination else None,
        dry_run=args.dry_run,
    )
    print(result.stdout.strip())
    if args.set_default:
        if args.dry_run:
            raise ModelStoreError("cannot set a default model during a dry-run")
        try:
            selected = choose_default_gguf(discover_local_models(result.directory))
        except ModelStoreError as exc:
            raise ModelStoreError(
                f"{exc}. Choose one with "
                "`kestrel models import PATH --set-default`."
            ) from exc
        metadata = read_gguf_config(str(selected))
        if metadata["architecture"] == "unknown" or not metadata["n_layer"]:
            raise ModelStoreError(
                f"downloaded GGUF has unusable planner metadata: {selected}"
            )
        _save_default_model(selected)
        print(f"Set {selected} as the default Kestrel model.")


def _models_info(args):
    from .model_store import ModelStoreError, resolve_ollama_blob

    source = args.source
    if source.startswith("ollama://"):
        path = resolve_ollama_blob(source.removeprefix("ollama://"))
        if path is None:
            raise ModelStoreError("Ollama model has no reusable local GGUF blob")
    else:
        path = Path(source).expanduser()
    detected = detect_model(str(path))
    if not detected:
        raise ModelStoreError(f"could not resolve model: {source}")
    if detected["type"] == "safetensors" and detected["path"]:
        info = _safetensors_info(detected["path"])
        print(json.dumps(info, indent=2))
        return
    if detected["type"] != "gguf":
        raise ModelStoreError("model info currently requires a local GGUF or safetensors directory")
    cfg = read_gguf_config(detected["path"])
    cfg.update(
        path=detected["path"],
        size_bytes=Path(detected["path"]).stat().st_size,
    )
    source_manifest = Path(detected["path"]).parent / ".kestrel-source.json"
    if source_manifest.is_file():
        try:
            cfg["source"] = json.loads(source_manifest.read_text())
        except (OSError, json.JSONDecodeError):
            cfg["source"] = {"error": "unreadable source manifest"}
    print(json.dumps(cfg, indent=2))


def _models_recommend(args, root):
    from .model_store import (
        complete_gguf_models,
        discover_local_models,
        list_ollama_models,
        model_total_size,
    )

    gpu = detect_gpu()
    vram = (gpu or {}).get("vram_total_mb", 0) * 1024**2
    ram = _available_ram_mib() * 1024**2
    candidates: dict[str, Path] = {
        str(path): path for path in complete_gguf_models(discover_local_models(root))
    }
    for item in list_ollama_models(resolve_paths=True):
        if item.local_path:
            candidates[f"ollama://{item.name}"] = item.local_path
    ranked = []
    for label, path in candidates.items():
        size = model_total_size(path)
        if vram and size <= vram * 0.82:
            fit, detail, rank = "excellent", "fits in safe GPU weight budget", 0
        elif size <= ram + vram * 0.65:
            fit, detail, rank = "viable", "requires CPU/RAM offload", 1
        else:
            fit, detail, rank = "paging", "exceeds working memory; expect storage stalls", 2
        try:
            cfg = read_gguf_config(str(path))
            architecture = cfg["architecture"]
            layers = cfg["n_layer"]
        except (OSError, struct.error, ValueError, KeyError):
            architecture, layers = "unreadable", 0
            fit, detail, rank = "unsupported", "GGUF metadata could not be read", 3
        ranked.append(
            {
                "source": label,
                "engine": "ollama" if label.startswith("ollama://") else "llama.cpp",
                "path": str(path),
                "size_gib": round(size / 1024**3, 2),
                "architecture": architecture,
                "layers": layers,
                "fit": fit,
                "reason": detail,
                "_rank": rank,
            }
        )
    ranked.sort(key=lambda item: (item["_rank"], item["size_gib"]))
    for item in ranked:
        item.pop("_rank")
    if args.json:
        print(json.dumps({"hardware": {"gpu": gpu, "available_ram_mib": ram // 1024**2}, "models": ranked}, indent=2))
    else:
        fit_color = {
            "excellent": ui.green,
            "viable": ui.cyan,
            "paging": ui.yellow,
            "unsupported": ui.red,
        }
        rows = []
        for item in ranked:
            colorize = fit_color.get(item["fit"], ui.dim)
            rows.append(
                f"  {colorize('[{}:9]'.format(item['fit'].upper()))} {ui.bold(item['source'])}"
            )
            rows.append(
                f"  {ui.dim('{:.2f} GiB'.format(item['size_gib']))}  {ui.dim(item['architecture'])}  "
                f"{ui.dim(item['reason'])}"
            )
        print(ui.box(
            f"Recommendations for {(gpu or {}).get('name', 'CPU-only host')}",
            "\n".join(rows),
        ))
        print(ui.dim("Fit is a memory classification, not a speed or quality guarantee; run `kestrel benchmark`."))


def cmd_models(args):
    from .model_store import (
        ModelStoreError,
        default_models_dir,
    )

    root = Path(USER_CONFIG.models_dir).expanduser() if USER_CONFIG.models_dir else default_models_dir()
    try:
        if args.models_command == "search":
            _models_search(args)
        elif args.models_command == "files":
            _models_files(args)
        elif args.models_command == "list":
            _models_list(args, root)
        elif args.models_command == "import":
            _models_import(args)
        elif args.models_command == "pull":
            _models_pull(args)
        elif args.models_command == "info":
            _models_info(args)
        elif args.models_command == "recommend":
            _models_recommend(args, root)
        else:
            raise SystemExit(
                "Choose a models command: search, files, list, recommend, info, pull, or import"
            )
    except ModelStoreError as exc:
        raise SystemExit(f"Model error: {exc}") from exc


def _menu_status_compact() -> str:
    gpu = detect_gpu()
    parts = []
    if gpu:
        parts.append(
            f"gpu: {gpu['name']} ({gpu['vram_free_mb']}/{gpu['vram_total_mb']} MiB free)"
        )
    else:
        parts.append("gpu: not detected")
    parts.append(f"ram: {_available_ram_mib()} MiB")
    text = "   " + "   ·   ".join(parts)
    return ui.dim(ui._truncate(text, ui.width()))


def cmd_menu(_args=None):
    """Dependency-free interactive front door over the scriptable CLI."""

    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise SystemExit("The interactive menu requires a terminal; use `kestrel --help`.")

    version = _kestrel_version()

    def launch(*arguments: str) -> None:
        try:
            _parser = build_parser()
            _run_dispatched(_parser, _parser.parse_args(arguments))
        except SystemExit:
            pass
        except KeyboardInterrupt:
            print()
        ui.pause()

    def prompt_required(label: str) -> str | None:
        try:
            return ui.ask(label).strip()
        except (EOFError, KeyboardInterrupt):
            return None

    def model_options() -> list[tuple[str, str]]:
        from .model_store import ModelStoreError, default_models_dir, discover_local_models, list_ollama_models

        config = load_config()
        options: list[tuple[str, str]] = []
        if config.default_model:
            options.append((config.default_model, "configured default"))
        root = Path(config.models_dir).expanduser() if config.models_dir else default_models_dir()
        for path in discover_local_models(root):
            options.append((str(path), f"{path.stat().st_size / 1024**3:.2f} GiB local GGUF"))
        try:
            for item in list_ollama_models():
                options.append((f"ollama://{item.name}", f"{item.size} via Ollama"))
        except ModelStoreError:
            pass
        return options

    def pick_model(title: str, *, keep_current: bool = False) -> str | None:
        options = model_options()
        if keep_current:
            options.insert(0, ("<keep current>", "leave the default model unchanged"))
        if not options:
            print(f"  {ui.warn_mark()} No models found; download one from the market first.")
            return None
        options.append(("<type a path>", "enter a model path, name, or alias manually"))
        chosen = ui.select(options, title=title, hint=ui.key_hint())
        if chosen < 0:
            return None
        label = options[chosen][0]
        if label == "<keep current>":
            return "<keep current>"
        if label == "<type a path>":
            return prompt_required("Model path, name, or alias")
        return label

    def pick_ollama_model() -> str | None:
        from .model_store import ModelStoreError, list_ollama_models

        try:
            models = list_ollama_models()
        except ModelStoreError as exc:
            print(f"  {ui.fail_mark()} {exc}")
            return None
        if not models:
            print(f"  {ui.warn_mark()} No Ollama models found; pull one first with `ollama pull <name>`.")
            return None
        options = [(item.name, f"{item.size}  {item.model_id}") for item in models]
        options.append(("<type a name>", "enter an Ollama model name manually"))
        chosen = ui.select(options, title="Import an Ollama model", hint=ui.key_hint())
        if chosen < 0:
            return None
        label = options[chosen][0]
        if label == "<type a name>":
            return prompt_required("Ollama model name")
        return label

    back = [("Go back", "")]

    def pick(items: list[tuple[str, str]], title: str) -> int:
        index = ui.select(items, title=title, hint=ui.key_hint())
        return index if index >= 0 else len(items) - 1

    while True:
        default_model = load_config().default_model
        header = "\n".join(
            [
                ui.bold("Kestrel") + (ui.dim(f"   v{version}") if ui.USE_ANSI else f"   v{version}"),
                _menu_status_compact(),
            ]
        )
        chosen = ui.select(
            [
                ("Chat with Model", default_model or "no model set"),
                ("Select Model(s)", ""),
                ("Import Models", ""),
                ("Manage Models", ""),
                ("Configure Kestrel", ""),
                ("Exit", ""),
            ],
            header=header,
            hint=ui.key_hint(),
        )
        if chosen < 0 or chosen >= 5:
            return

        if chosen == 0:
            model = default_model or pick_model("Chat with which model?")
            if model is None:
                continue
            context = ui.ask(
                "Context tokens",
                default="auto",
                validate=lambda value: (
                    None
                    if value == "auto" or (value.isdigit() and int(value) >= 512)
                    else "Context must be 'auto' or an integer of at least 512"
                ),
            )
            target = "auto"
            if ui.confirm("Choose the placement target explicitly", default=False):
                index = ui.select(
                    [
                        ("auto", "adaptive: pick from available free memory"),
                        ("balanced", "memory-aware default"),
                        ("quality", "most stable, slower"),
                        ("speed", "experimental throughput bias"),
                    ],
                    title="Placement target",
                    hint=ui.key_hint(),
                )
                if index < 0:
                    continue
                target = ["auto", "balanced", "quality", "speed"][index]
            warm = ui.confirm("Prime the page cache for a faster load", default=False)
            launch_args = ["chat", model, "--ctx-size", context, "--target", target]
            if warm:
                launch_args.append("--warm-cache")
            launch(*launch_args)
        elif chosen == 1:
            model = pick_model("Select a model")
            if model is None:
                continue
            launch("setup", "--model", model)
        elif chosen == 2:
            index = pick(
                [("Import an Ollama model", ""), ("Pull from Hugging Face", ""), *back],
                "Import Models",
            )
            if index >= 2:
                continue
            if index == 0:
                name = pick_ollama_model()
                if name is None:
                    continue
                launch("models", "import", f"ollama://{name}", "--set-default")
            else:
                repo = prompt_required("Hugging Face repository (OWNER/REPO)")
                if repo is None:
                    continue
                filename = prompt_required("Specific filename (Enter for entire repository)")
                if filename is None:
                    continue
                command = ["models", "pull", f"hf://{repo}"]
                if filename:
                    command.extend(["--file", filename])
                print(ui.dim("Checking download size first..."))
                launch(*command, "--dry-run")
                if ui.confirm("Proceed with the download", default=False):
                    launch(*command)
        elif chosen == 3:
            index = pick(
                [
                    ("Installed models", ""),
                    ("Search model market", ""),
                    ("Hardware diagnostics", ""),
                    ("Benchmark", ""),
                    ("Convert / prune a model", ""),
                    *back,
                ],
                "Manage Models",
            )
            if index == 0:
                launch("models", "list", "--resolve")
            elif index == 1:
                query = prompt_required("Search GGUF models")
                if query is None:
                    continue
                launch("models", "search", query)
            elif index == 2:
                launch("doctor")
            elif index == 3:
                launch("benchmark")
            elif index == 4:
                model_dir = prompt_required("Model directory (downloaded safetensors)")
                if model_dir is None:
                    continue
                output = prompt_required("Output GGUF path (Enter for automatic name)")
                if output is None:
                    continue
                keep = ui.ask(
                    "Experts per layer to keep (Enter to keep all)",
                    default="",
                    validate=lambda value: (
                        None
                        if value == "" or (value.isdigit() and int(value) > 0)
                        else "Enter a positive integer or leave blank to keep all"
                    ),
                )
                importance = None
                if keep and int(keep) > 0:
                    if ui.confirm(
                        "Choose which experts to keep by importance "
                        "(router-frequency JSON)?",
                        default=False,
                    ):
                        importance = prompt_required("Path to expert-importance JSON")
                        if importance is None:
                            continue
                command = ["convert", model_dir]
                if output:
                    command.extend(["-o", output])
                if keep and int(keep) > 0:
                    command.extend(["--experts-keep", keep])
                    if importance:
                        command.extend(["--expert-importance", importance])
                print(ui.dim("This is an experimental, opt-in quality trade-off."))
                launch(*command)
        else:
            index = pick([("Default model", ""), ("Models directory", ""), *back], "Configure Kestrel")
            if index == 0:
                model = pick_model("Set the default model", keep_current=True)
                if model is None:
                    continue
                command = ["setup"]
                if model != "<keep current>":
                    command.extend(["--model", model])
                launch(*command)
            elif index == 1:
                models_dir = prompt_required("Managed models directory (Enter to leave unchanged)")
                if models_dir is None:
                    continue
                launch("setup", "--models-dir", models_dir)


def _build_optimize_profile(args, model_arg, gpu, storage_path) -> dict:
    """Resolve the local model (if any) and build the hardware/model/plan
    document for the ``optimize`` command."""
    profile = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": {
            "cpu": platform.processor(),
            "logical_cpu_count": os.cpu_count() or 0,
            "available_ram_mib": _available_ram_mib(),
            "memory": _memory_snapshot(),
            "cpu_power_policy": _cpu_power_policy(),
            "gpu": gpu,
            "storage": {
                "path": str(storage_path),
                "free_gib": round(
                    shutil.disk_usage(storage_path).free / 1024**3,
                    2,
                ),
            },
        },
        "model": None,
        "plan": None,
    }
    cpu = profile["hardware"]["cpu"]
    if not cpu or cpu.lower() in {"x86_64", "amd64", "aarch64"}:
        try:
            cpu = next(
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            )
        except (OSError, StopIteration, IndexError):
            cpu = "unknown"
        profile["hardware"]["cpu"] = cpu
    if not model_arg:
        return profile
    engine = "llama.cpp"
    resolved_model_arg = model_arg
    if model_arg.startswith("ollama://"):
        from .model_store import ModelStoreError, resolve_ollama_blob

        engine = "ollama"
        try:
            blob = resolve_ollama_blob(model_arg.removeprefix("ollama://"))
        except ModelStoreError as exc:
            raise SystemExit(f"Error: {exc}") from exc
        if blob is None:
            raise SystemExit(
                "Error: cloud-only Ollama models do not expose local metadata for optimization"
            )
        resolved_model_arg = str(blob)
    model_info = detect_model(resolved_model_arg)
    if not model_info or model_info["type"] != "gguf" or not model_info["path"]:
        raise SystemExit("Error: optimize currently requires a local GGUF model")
    size = Path(model_info["path"]).stat().st_size
    if not args.storage_path:
        storage_path = Path(model_info["path"]).parent.resolve()
        profile["hardware"]["storage"] = {
            "path": str(storage_path),
            "free_gib": round(shutil.disk_usage(storage_path).free / 1024**3, 2),
        }
    if args.context:
        context = args.context
        context_reason = "explicit user setting"
    else:
        context, context_reason = _select_context_size(model_info, gpu)
    plan_args = argparse.Namespace(
        gpu_layers="auto",
        ctx_size=context,
        cpu_moe="auto",
        moe_cache="off",
        moe_cold_model=None,
        target=args.quality,
    )
    plan = estimate_config(model_info, gpu, plan_args)
    plan["context_reason"] = context_reason
    plan["quality_profile"] = args.quality
    plan["kv_cache_type"] = {
        "speed": "q4_0",
        "balanced": "q8_0",
        "quality": "f16",
    }[args.quality]
    profile["model"] = {
        "source": model_arg,
        "engine": engine,
        "path": model_info["path"],
        "size_gib": round(size / 1024**3, 2),
        **read_gguf_config(model_info["path"]),
    }
    profile["plan"] = plan
    return profile


def _run_optimize_benchmark(profile, model_arg, target, args) -> None:
    """Run the measurement sweep for ``optimize`` and fold the results into
    ``profile``, raising ``SystemExit`` when the run fails (the caller then
    re-raises after persisting the failure state)."""
    if not model_arg:
        raise SystemExit("Error: --benchmark requires a model")
    benchmark_path = target.with_name("hardware-benchmark.json")
    try:
        report = cmd_benchmark(
            argparse.Namespace(
                model=model_arg,
                prompt_tokens=128,
                generate_tokens=64,
                repetitions=3,
                ctx_size=profile["plan"]["context_size"],
                gpu_layers="auto",
                cpu_moe="auto",
                threads=(
                    _cpu_moe_thread_sweep(os.cpu_count() or 1)
                    if profile["plan"]["cpu_moe"]
                    else None
                ),
                batch_size=profile["plan"]["batch_size"],
                ubatch_size=profile["plan"]["ubatch_size"],
                kv_cache_type=profile["plan"]["kv_cache_type"],
                output=str(benchmark_path),
                quiet=True,
            )
        )
        profile["benchmark"] = {
            "status": "measured",
            "report": str(benchmark_path),
            "prompt_tokens_per_second": report["prompt_tokens_per_second"],
            "decode_tokens_per_second": report["decode_tokens_per_second"],
            "release_speed_floor_passed": report["release_speed_floor_passed"],
            "quality_gate": report.get("quality_gate", "not_run"),
            "selected_placement": report.get("placement"),
        }
        if report.get("placement", {}).get("threads") is not None:
            profile["plan"]["threads"] = report["placement"]["threads"]
    except SystemExit as exc:
        profile["benchmark"] = {"status": "failed", "error": str(exc)}
        if not args.no_save:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(profile, indent=2) + "\n")
            print(f"Wrote failed benchmark state to {target}", file=sys.stderr)
        print(json.dumps(profile, indent=2))
        raise


def cmd_optimize(args):
    """Create an explainable hardware/model plan and optionally measure it."""
    model_arg = args.model or os.environ.get("KESTREL_MODEL") or USER_CONFIG.default_model
    gpu = detect_gpu()
    storage_path = Path(args.storage_path or ".").expanduser().resolve()
    profile = _build_optimize_profile(args, model_arg, gpu, storage_path)
    target = (
        Path(args.output).expanduser()
        if args.output
        else config_path().with_name("hardware-profile.json")
    )
    profile["benchmark"] = {"status": "not_run"}
    if args.benchmark:
        _run_optimize_benchmark(profile, model_arg, target, args)
    if not args.no_save:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(profile, indent=2) + "\n")
        print(f"Wrote {target}", file=sys.stderr)
    print(json.dumps(profile, indent=2))


def _summarize_benchmark_rows(rows: list[dict]) -> tuple[dict | None, dict | None, list[dict]]:
    decode_rows = [row for row in rows if row.get("n_gen")]
    decode = max(decode_rows, key=lambda row: row.get("avg_ts", 0), default=None)
    best_threads = decode.get("n_threads") if decode else None
    prompt = next(
        (
            row
            for row in rows
            if row.get("n_prompt") and row.get("n_threads") == best_threads
        ),
        next((row for row in rows if row.get("n_prompt")), None),
    )
    sweep = [
        {
            "threads": row.get("n_threads"),
            "decode_tokens_per_second": row.get("avg_ts"),
        }
        for row in sorted(decode_rows, key=lambda item: item.get("n_threads", 0))
    ]
    return prompt, decode, sweep


def _print_benchmark_summary(report: dict) -> None:
    prompt = report.get("prompt_tokens_per_second")
    decode = report.get("decode_tokens_per_second")
    floor = report.get("release_speed_floor_passed")
    lines = [
        ui.kv(
            "Model",
            str(report.get("model") or "unknown"),
            value_color=ui.bold,
        ),
        ui.kv(
            "Prompt",
            f"{prompt} tok/s" if prompt is not None else "not recorded",
            value_color=ui.cyan,
        ),
        ui.kv(
            "Decode",
            f"{decode} tok/s" if decode is not None else "not recorded",
            value_color=ui.green if (decode and decode >= 10) else ui.yellow,
        ),
        ui.kv(
            "Speed floor (10 tok/s)",
            "pass" if floor else "fail",
            value_color=ui.green if floor else ui.red,
        ),
    ]
    placement = report.get("placement") or {}
    if placement:
        lines.append(ui.kv(
            "Placement",
            f"GPU layers={placement.get('gpu_layers', 'unknown')}, "
            f"threads={placement.get('threads', 'unknown')}, "
            f"CPU MoE={'on' if placement.get('cpu_moe') else 'off'}, "
            f"KV={placement.get('kv_cache_type', 'unknown')}",
        ))
    print(ui.box("Benchmark", "\n".join(lines)))


def cmd_benchmark(args):
    model_arg = _default_model(
        args,
        error="Error: no benchmark model selected; run `kestrel setup --model ...`",
    )
    if model_arg.startswith("ollama://"):
        from .providers.ollama import OllamaClient, OllamaError

        name = model_arg.removeprefix("ollama://")
        client = OllamaClient()
        prompt = ("Kestrel measures prompt and decode performance. " * max(2, args.prompt_tokens // 8)).strip()
        try:
            # Separate warmup prevents model load time from contaminating the
            # recorded token rates returned by Ollama.
            client.generate(name, "warmup", num_predict=1, num_ctx=args.ctx_size)
            samples = [
                client.generate(
                    name,
                    prompt,
                    num_predict=args.generate_tokens,
                    num_ctx=args.ctx_size,
                    seed=42,
                )
                for _ in range(args.repetitions)
            ]
        except OllamaError as exc:
            raise SystemExit(f"Ollama benchmark failed: {exc}") from exc
        prompt_rates = [item.prompt_tps for item in samples if item.prompt_tps is not None]
        decode_rates = [item.decode_tps for item in samples if item.decode_tps is not None]
        prompt_rate = sum(prompt_rates) / len(prompt_rates) if prompt_rates else None
        decode_rate = sum(decode_rates) / len(decode_rates) if decode_rates else None
        report = {
            "model": model_arg,
            "engine": "ollama",
            "context_size": args.ctx_size,
            "prompt_tokens_per_second": prompt_rate,
            "decode_tokens_per_second": decode_rate,
            "release_speed_floor_passed": bool(decode_rate and decode_rate >= 10),
            "quality_gate": "not_run",
            "sample_output": (
                samples[-1].response or samples[-1].thinking if samples else ""
            ),
            "raw": [
                {
                    "prompt_tokens": item.prompt_tokens,
                    "prompt_duration_ns": item.prompt_duration_ns,
                    "generated_tokens": item.generated_tokens,
                    "generation_duration_ns": item.generation_duration_ns,
                    "total_duration_ns": item.total_duration_ns,
                }
                for item in samples
            ],
        }
        encoded = json.dumps(report, indent=2)
        if args.output:
            output = Path(args.output).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(encoded + "\n")
            print(f"Wrote {output}", file=sys.stderr)
        if not getattr(args, "quiet", False):
            _print_benchmark_summary(report)
            print(encoded)
        return report
    model_info = detect_model(model_arg)
    if not model_info or model_info["type"] != "gguf" or not model_info["path"]:
        raise SystemExit("Error: benchmark requires a local GGUF model")
    binary = resolve_llama_binary("llama-bench")
    if not binary:
        raise SystemExit("Error: llama-bench was not found; build llama.cpp first")

    gpu = detect_gpu()
    plan_args = argparse.Namespace(
        gpu_layers=args.gpu_layers,
        ctx_size=args.ctx_size,
        cpu_moe=args.cpu_moe,
        moe_cache="off",
        moe_cold_model=None,
    )
    config = estimate_config(model_info, gpu, plan_args)
    threads = args.threads or config["threads"] or max(1, (os.cpu_count() or 2) - 2)
    if isinstance(threads, str) and not re.fullmatch(r"\d+(?:,\d+)*", threads):
        raise SystemExit("Error: --threads must be an integer or comma-separated integers")
    planned_gpu_layers = str(config["gpu_layers"])
    bench_gpu_layers = "99" if planned_gpu_layers in {"auto", "all"} else planned_gpu_layers
    command = [
        binary,
        "-m", model_info["path"],
        "-p", str(args.prompt_tokens),
        "-n", str(args.generate_tokens),
        "-r", str(args.repetitions),
        "-ngl", bench_gpu_layers,
        "--moe-cache", "off",
        "-t", str(threads),
        "-b", str(args.batch_size),
        "-ub", str(args.ubatch_size),
        "-fa", "on",
        "-ctk", args.kv_cache_type,
        "-ctv", args.kv_cache_type,
        "-o", "json",
    ]
    if planned_gpu_layers == "auto":
        command.extend(["-fitt", str(config["fit_target_mib"])])
    if config["cpu_moe"]:
        command.extend(["-ncmoe", str(config["n_layers"])])
    print("Benchmarking the exact configured placement...", file=sys.stderr)
    bench_timeout = 30 * 60 * max(1, args.repetitions or 1)
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=bench_timeout)
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(
            "Error: llama-bench exceeded its time budget "
            f"({bench_timeout}s). Reduce --repetitions or the token counts."
        ) from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout)[-3000:]
        raise SystemExit(f"llama-bench failed ({result.returncode}):\n{detail}")
    try:
        rows = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"llama-bench returned invalid JSON: {exc}") from exc
    prompt, decode, thread_sweep = _summarize_benchmark_rows(rows)
    report = {
        "model": model_info["path"],
        "model_size_gib": config["model_size_gib"],
        "gpu": gpu,
        "placement": {
            "gpu_layers": decode.get("n_gpu_layers") if decode else config["gpu_layers"],
            "requested_gpu_layers": config["gpu_layers"],
            "cpu_moe": config["cpu_moe"],
            "threads": decode.get("n_threads") if decode else threads,
            "requested_threads": threads,
            "batch_size": args.batch_size,
            "ubatch_size": args.ubatch_size,
            "kv_cache_type": args.kv_cache_type,
        },
        "prompt_tokens_per_second": prompt.get("avg_ts") if prompt else None,
        "decode_tokens_per_second": decode.get("avg_ts") if decode else None,
        "release_speed_floor_passed": bool(decode and decode.get("avg_ts", 0) >= 10),
        "quality_gate": "not_run",
        "thread_sweep": thread_sweep,
        "raw": rows,
    }
    encoded = json.dumps(report, indent=2)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n")
        print(f"Wrote {output}", file=sys.stderr)
    if not getattr(args, "quiet", False):
        _print_benchmark_summary(report)
        print(encoded)
    return report


def _add_json_flag(parser):
    parser.add_argument(
        "--json",
        dest="json",
        action="store_true",
        help="emit a single machine-readable JSON object on stdout; "
        "all human-readable output moves to stderr",
    )


def _default_model(args, *, error: str) -> str:
    model = (
        args.model
        or os.environ.get("KESTREL_MODEL")
        or USER_CONFIG.default_model
    )
    if not model and _resolve_model_alias("qwen3.5:122b-10a"):
        model = "qwen3.5:122b-10a"
    if not model:
        raise SystemExit(error)
    return model


def _add_local_run_options(parser, *, model_optional: bool = False):
    parser.add_argument(
        "model",
        nargs="?" if model_optional else None,
        help="Hugging Face model ID, model directory, or GGUF",
    )
    parser.add_argument("--no-convert", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the validated command")
    _add_json_flag(parser)
    parser.add_argument(
        "--prompt",
        help="run a one-shot, non-interactive generation with this prompt "
        "(otherwise launches the interactive CLI)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="token budget for --prompt one-shot generation (default: 256)",
    )
    parser.add_argument(
        "--ctx-size",
        type=_context_size_arg,
        default="auto",
        help="context tokens (default: hardware-aware auto)",
    )
    parser.add_argument("--gpu-layers", default="auto", help="auto, all, or an exact count")
    parser.add_argument("--cpu-moe", choices=("auto", "on", "off"), default="auto")
    parser.add_argument(
        "--target",
        choices=("auto", "balanced", "quality", "speed"),
        default="auto",
        help=(
            "placement target: auto (adaptive, from free RAM), balanced, "
            "quality (stable, slower), or speed (experimental-throughput; "
            "never auto-selected)"
        ),
    )
    parser.add_argument("--fit-target", type=int, help="VRAM margin in MiB")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--ubatch-size", type=int)
    parser.add_argument(
        "--threads",
        type=int,
        help="CPU generation and prompt threads (default: hardware-aware)",
    )
    parser.add_argument(
        "--kv-cache-type",
        choices=("f16", "bf16", "q8_0", "q4_0", "q4_1"),
        default="q8_0",
    )
    parser.add_argument(
        "--moe-cache",
        default="auto",
        help="llama.cpp MoE cache: auto, on, off, or a MiB budget",
    )
    parser.add_argument(
        "--moe-hot-model",
        help="immutable Q4 GGUF sidecar for a compact Q1 expert model",
    )
    parser.add_argument(
        "--moe-cold-model",
        help="immutable Q1 experts-only sidecar for a canonical Q4 model",
    )
    parser.add_argument("--no-mmap", action="store_true")
    parser.add_argument(
        "--mlock",
        action="store_true",
        help="lock model weights in RAM so CPU-offloaded/experts stay resident "
        "(removes page-in latency jitter; implies a memory footprint)",
    )
    parser.add_argument(
        "--tensor-split",
        help=(
            "comma-separated VRAM ratios for multi-GPU tensor splitting, "
            "e.g. '60,40' (default: auto-derived from detected device VRAM)"
        ),
    )
    parser.add_argument(
        "--extra",
        action="append",
        metavar="ARGS",
        help="space-separated passthrough args for the llama.cpp binary (repeatable); "
        "e.g. --extra '--temp 0.4 --top-p 0.9'",
    )
    parser.add_argument(
        "--direct-io",
        action="store_true",
        help=(
            "load weights with uncached direct I/O (bypasses the page cache) "
            "for faster cold loads from NVMe; ignore the page cache warning "
            "and prefer a warm mmap reload otherwise; disables --warm-cache"
        ),
    )
    parser.add_argument(
        "--warm-cache",
        action="store_true",
        help="prime the OS page cache for the model before launch (bounded pre-read)",
    )
    parser.add_argument("--no-mtp", action="store_true")
    parser.add_argument("--mtp-tokens", type=int, default=3)
    parser.add_argument(
        "--no-oom-retry",
        action="store_true",
        help="Do not retry startup with lower-memory settings",
    )


def cmd_audit(args):
    from .gguf.audit import audit_gguf

    report = audit_gguf(args.model, args.source, cold_sidecar=args.cold_sidecar)
    if args.json:
        import json

        print(json.dumps(report, indent=2))
    else:
        verdict = "PASS" if report["valid"] else "FAIL"
        severity_color = {
            "error": ui.red,
            "warning": ui.yellow,
            "info": ui.cyan,
        }
        findings = []
        for item in report["findings"]:
            label = item["severity"].upper()
            colorize = severity_color.get(item["severity"], ui.dim)
            findings.append(f"  {colorize(f'[{label}]')} {item['code']}: {item['message']}")
        print(ui.box(
            "Kestrel GGUF audit",
            "\n".join([
                ui.kv("Verdict", verdict, value_color=ui.green if report["valid"] else ui.red),
                ui.kv("Model", report["model"], value_color=ui.bold),
                ui.kv("Tensors", str(report["tensor_count"])),
                ui.kv(
                    "Errors / warnings",
                    f"{report['errors']} / {report['warnings']}",
                    value_color=ui.yellow if report["warnings"] else None,
                ),
                "",
                *findings,
            ]),
            title_color=ui.green if report["valid"] else ui.red,
        ))
    if not report["valid"]:
        raise SystemExit(1)


def _kestrel_version() -> str:
    """Resolve the package version without depending on the parent package.

    An editable install can be shadowed by a ``kestrel`` directory on
    ``sys.path`` that lacks ``__init__.py``, in which case ``kestrel`` becomes
    a namespace package without a ``__version__`` attribute. Prefer installed
    metadata, then the source ``pyproject.toml``.
    """
    try:
        from . import __version__ as _version

        return _version
    except (ImportError, AttributeError, NameError):
        pass
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("kestrel")
    except (PackageNotFoundError, ImportError):
        pass
    try:
        root = Path(__file__).resolve().parents[1] / "pyproject.toml"
        for line in root.read_text().splitlines():
            match = re.match(r'^version\s*=\s*["\']([^"\']+)["\']', line.strip())
            if match:
                return match.group(1)
    except (OSError, UnicodeDecodeError):
        pass
    return "unknown"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Kestrel - hardware-aware local model orchestration and management"
    )
    parser.add_argument(
        "--version", action="version", version=f"kestrel {_kestrel_version()}"
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("menu", help="Open the interactive Kestrel menu")
    status = sub.add_parser("status", help="Show active model, hardware plan, and benchmark state")
    _add_json_flag(status)

    run = sub.add_parser("run", help="Plan and run a local model")
    _add_local_run_options(run)
    chat = sub.add_parser("chat", help="Chat with the configured local model")
    _add_local_run_options(chat, model_optional=True)
    serve = sub.add_parser(
        "serve",
        help="Serve a GGUF model through an OpenAI-compatible llama-server endpoint",
    )
    _add_local_run_options(serve, model_optional=True)
    serve.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    serve.add_argument("--port", type=int, default=8080, help="listen port (default: 8080)")
    serve.add_argument("--alias", help="model alias reported by /v1/models")
    serve.add_argument(
        "--embeddings",
        action="store_true",
        help="enable the /v1/embeddings endpoint for RAG workloads "
        "(llama-server --embeddings)",
    )
    serve.add_argument(
        "--wait",
        type=float,
        default=0.0,
        help="after launch, poll /health until the server reports ready "
        "(seconds to wait before giving up; defaults to 30s when unset)",
    )


    setup = sub.add_parser("setup", help="Save safe local defaults (never API keys)")
    setup.add_argument("--model", help="default local GGUF, directory, or tested alias")
    setup.add_argument("--models-dir", help="managed model download directory")
    setup.add_argument("--llama-cpp-dir")
    setup.add_argument(
        "--reset",
        action="store_true",
        help="replace all saved defaults; also repairs a malformed config",
    )

    benchmark = sub.add_parser("benchmark", help="Measure prompt and decode rates reproducibly")
    benchmark.add_argument("model", nargs="?")
    benchmark.add_argument("--prompt-tokens", type=int, default=128)
    benchmark.add_argument("--generate-tokens", type=int, default=64)
    benchmark.add_argument("--repetitions", type=int, default=3)
    benchmark.add_argument("--ctx-size", type=int, default=2048)
    benchmark.add_argument("--gpu-layers", default="auto")
    benchmark.add_argument("--cpu-moe", choices=("auto", "on", "off"), default="auto")
    benchmark.add_argument(
        "--threads",
        help="one thread count or a comma-separated sweep, e.g. 8,10,12,14,16",
    )
    benchmark.add_argument("--batch-size", type=int, default=128)
    benchmark.add_argument("--ubatch-size", type=int, default=64)
    benchmark.add_argument("--kv-cache-type", choices=("f16", "bf16", "q8_0", "q4_0", "q4_1"), default="q8_0")
    benchmark.add_argument("--output", help="write the complete JSON report")

    optimize = sub.add_parser("optimize", help="Create and optionally benchmark a hardware-specific plan")
    optimize.add_argument("model", nargs="?")
    optimize.add_argument("--context", type=int, help="override automatic context selection")
    optimize.add_argument("--quality", choices=("speed", "balanced", "quality"), default="balanced")
    optimize.add_argument(
        "--storage-path",
        help="storage path to assess (default: selected model's filesystem)",
    )
    optimize.add_argument("--benchmark", action="store_true", help="measure the selected plan with llama-bench")
    optimize.add_argument("--output")
    optimize.add_argument("--no-save", action="store_true")

    models = sub.add_parser("models", help="Discover, inspect, pull, and import models")
    models_sub = models.add_subparsers(dest="models_command")
    models_search = models_sub.add_parser(
        "search", help="search current GGUF repositories on Hugging Face"
    )
    models_search.add_argument("query")
    models_search.add_argument("--limit", type=int, default=10)
    _add_json_flag(models_search)
    models_files = models_sub.add_parser(
        "files", help="list GGUF variants and sizes in a Hugging Face repository"
    )
    models_files.add_argument("source", help="hf://OWNER/REPO or OWNER/REPO")
    _add_json_flag(models_files)
    models_list = models_sub.add_parser("list", help="list Kestrel and Ollama models")
    models_list.add_argument("--resolve", action="store_true", help="resolve Ollama model blobs")
    _add_json_flag(models_list)
    models_recommend = models_sub.add_parser(
        "recommend", help="rank installed models by measured host memory fit"
    )
    _add_json_flag(models_recommend)
    models_info = models_sub.add_parser("info", help="inspect a local or Ollama GGUF")
    models_info.add_argument("source", help="local path or ollama://NAME")
    models_pull = models_sub.add_parser("pull", help="download from Hugging Face or Ollama")
    models_pull.add_argument("source", help="hf://OWNER/REPO, OWNER/REPO, or ollama://NAME")
    models_pull.add_argument("--file", help="specific Hugging Face file")
    models_pull.add_argument(
        "--include", help="Hugging Face glob, including all shards of a split GGUF"
    )
    models_pull.add_argument("--revision", help="Hugging Face commit, tag, or branch")
    models_pull.add_argument("--destination")
    models_pull.add_argument("--dry-run", action="store_true")
    models_pull.add_argument("--set-default", action="store_true")
    models_import = models_sub.add_parser("import", help="reuse a local GGUF or Ollama blob")
    models_import.add_argument("source", help="local path or ollama://NAME")
    models_import.add_argument("--set-default", action="store_true")

    sub.add_parser("build", help="Build llama.cpp with CUDA")

    engine_group = sub.add_parser(
        "engine",
        help="Manage the llama.cpp engine: provenance, upstream updates, self-rebuilds",
    )
    engine_sub = engine_group.add_subparsers(dest="engine_command")
    es_status = engine_sub.add_parser("status", help="Show provenance and staleness")
    es_status.add_argument("--dir", help="engine checkout (default: configured llama_cpp_dir)")
    _add_json_flag(es_status)
    es_update = engine_sub.add_parser(
        "update", help="Fetch upstream and fast-forward + rebuild if a newer revision exists"
    )
    es_update.add_argument("--dir", help="engine checkout (default: configured)"
    )
    es_update.add_argument("--remote", help="override the tracked remote URL")
    es_update.add_argument("--dry-run", action="store_true")
    es_update.add_argument(
        "--force",
        action="store_true",
        help="hard-reset the checkout, discarding local commits/edits",
    )
    _add_json_flag(es_update)
    es_rebuild = engine_sub.add_parser(
        "rebuild", help="Rebuild the engine from its checked-out source"
    )
    es_rebuild.add_argument("--dir", help="engine directory")
    es_rebuild.add_argument("--dry-run", action="store_true")
    _add_json_flag(es_rebuild)
    es_set = engine_sub.add_parser("set", help="Adopt a checkout as a managed engine")
    es_set.add_argument("--dir", required=True)
    es_set.add_argument("--remote", help="remote URL (default: the checkout's origin)")
    _add_json_flag(es_set)

    self_update = sub.add_parser(
        "self-update", help="Update the Kestrel package from a repo or a wheel"
    )
    self_update.add_argument("--repo", help="repo path or git URL (default: the package's source tree)")
    self_update.add_argument("--wheel", help="path or URL to a kestrel wheel to install")
    self_update.add_argument(
        "--sha256",
        help="expected SHA256 of --wheel (verified before install; trusted distribution)",
    )
    self_update.add_argument("--dry-run", action="store_true")
    self_update.add_argument("--yes", action="store_true")
    _add_json_flag(self_update)
    convert = sub.add_parser("convert", help="Convert supported NVFP4 safetensors")
    convert.add_argument("model")
    convert.add_argument("--output", "-o")
    convert.add_argument(
        "--generic",
        action="store_true",
        help="use llama.cpp's convert_hf_to_gguf.py for arbitrary HF safetensors models",
    )
    convert.add_argument(
        "--outtype",
        default="bf16",
        help="GGUF output type for --generic (default: bf16)",
    )
    convert.add_argument(
        "--include-mtp",
        action="store_true",
        help="include the optional speculative MTP draft block",
    )
    convert.add_argument(
        "--dense-q4",
        action="store_true",
        help="quantize dense matrices to Q4_0 instead of BF16",
    )
    convert.add_argument(
        "--cold-tier",
        choices=("off", "q1_0", "q1_only"),
        default="off",
        help="emit Q1_0 expert twins, or a compact Q1-only expert model",
    )
    convert.add_argument(
        "--q4-sidecar-source",
        help="derive q1_only experts from an existing canonical Q4 GGUF",
    )
    convert.add_argument(
        "--experts-only",
        action="store_true",
        help="emit only compact routed experts for use as a cold sidecar",
    )
    convert.add_argument(
        "--q2-edge-layers",
        type=int,
        default=0,
        metavar="N",
        help="use Q2_K experts for the first and last N layers of a direct q1_only conversion",
    )
    convert.add_argument(
        "--compact-expert-type",
        choices=("q1_0", "iq1_s"),
        default="q1_0",
        help="expert format used by a direct q1_only compact-primary conversion",
    )
    convert.add_argument(
        "--imatrix",
        help="llama-imatrix GGUF used to calibrate IQ1_S experts",
    )
    convert.add_argument(
        "--conversion-workers",
        type=int,
        help="parallel expert conversion workers (default: up to 4)",
    )
    convert.add_argument(
        "--experts-keep",
        type=int,
        metavar="K",
        help=(
            "EXPERIMENTAL: emit only the K most-used experts per layer and "
            "rewrite expert_count (must stay >= experts per token and < the "
            "full count). Smaller model, fits more in VRAM, quality may drop."
        ),
    )
    convert.add_argument(
        "--expert-importance",
        help=(
            "JSON list of per-expert importance values (length = num_experts) "
            "used to select which experts --experts-keep keeps; defaults to "
            "keeping the first K"
        ),
    )
    audit = sub.add_parser(
        "audit",
        help="Validate GGUF tokenizer and target/MTP structure",
    )
    audit.add_argument("model", help="GGUF artifact to audit")
    audit.add_argument(
        "--source",
        help="source Hugging Face model directory for cross-checks",
    )
    _add_json_flag(audit)
    audit.add_argument(
        "--cold-sidecar",
        action="store_true",
        help="validate an intentionally experts-only Q1 cold-sidecar GGUF",
    )
    doctor = sub.add_parser("doctor", help="Check hardware and llama.cpp capabilities")
    _add_json_flag(doctor)

    return parser


def main():
    parser = build_parser()
    _run_dispatched(parser, parser.parse_args())


_COMMAND_HANDLERS = {
    "menu": cmd_menu,
    "status": cmd_status,
    "run": cmd_run,
    "chat": cmd_run,
    "serve": cmd_serve,
    "setup": cmd_setup,
    "benchmark": cmd_benchmark,
    "optimize": cmd_optimize,
    "models": cmd_models,
    "build": cmd_build,
    "engine": cmd_engine,
    "self-update": cmd_self_update,
    "convert": cmd_convert,
    "audit": cmd_audit,
    "doctor": cmd_doctor,
}


def _run_dispatched(parser, args) -> None:
    if CONFIG_ERROR and not (args.command == "setup" and args.reset):
        parser.error(f"{CONFIG_ERROR}; run `kestrel setup --reset` to repair it")
    handler = _COMMAND_HANDLERS.get(args.command)
    if handler is not None:
        try:
            handler(args)
        except KestrelError as exc:
            raise SystemExit(_print_failure(exc, json_output=bool(getattr(args, "json", False)))) from None
    elif sys.stdin.isatty() and sys.stdout.isatty():
        cmd_menu(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
