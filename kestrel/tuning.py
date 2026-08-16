"""Measured runtime profiles for exact model and hardware identities.

Planner heuristics are intentionally conservative. A tuning profile may
override them only when it was measured for the same GGUF artifact and GPU
layout, and when the current free VRAM still satisfies the recorded floor.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path

from .config import config_path

PROFILE_SCHEMA = 2
TUNED_PROFILE_REASON = "exact measured model/hardware profile"
# Placement is weight/thread-bound and survives a larger request context; only
# the KV cache grows beyond the measured size, which the caller bounds by
# shrinking the prefill micro-batch and warning about memory pressure.
LARGER_CONTEXT_REASON = (
    "placement applied at a larger context than the tuned size (KV cache grows beyond the measured footprint)"
)
TUNABLE_PLAN_FIELDS = (
    "gpu_layers",
    "cpu_moe",
    "n_cpu_moe_layers",
    "batch_size",
    "ubatch_size",
    "threads",
    "cache_type_k",
    "cache_type_v",
)


def default_profile_path() -> Path:
    return config_path().with_name("hardware-profile.json")


def profile_path_for(model_info: dict, gpu: dict | None) -> Path:
    identity = {
        "model": model_identity(model_info),
        "hardware": hardware_identity(gpu),
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]
    return config_path().parent / "hardware-profiles" / f"{digest}.json"


def model_identity(model_info: dict) -> dict:
    path = Path(model_info.get("path") or "").expanduser()
    identity = {"path": str(path.resolve()) if path else ""}
    try:
        stat = path.stat()
    except OSError:
        identity.update({"size_bytes": 0, "mtime_ns": 0})
    else:
        identity.update({"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return identity


def hardware_identity(gpu: dict | None) -> dict:
    devices = (gpu or {}).get("devices") or ([] if not gpu else [gpu])
    return {
        "logical_cpu_count": os.cpu_count() or 0,
        "cpu": _cpu_identity(),
        "gpus": [
            {
                "name": device.get("name"),
                "vram_total_mb": device.get("vram_total_mb"),
            }
            for device in devices
        ],
    }


def engine_identity(dirs: tuple[str, ...] | None = None) -> dict:
    # Imported lazily to keep configuration/profile inspection cheap and avoid
    # coupling module import order to native engine discovery.
    from .backends.llama_cpp import resolve_llama_binary

    binary = resolve_llama_binary("llama-server", dirs=dirs)
    if not binary:
        return {"path": "", "size_bytes": 0, "mtime_ns": 0}
    path = Path(binary).resolve()
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path), "size_bytes": 0, "mtime_ns": 0}
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _cpu_identity() -> str:
    value = platform.processor().strip()
    if value and value.lower() not in {"x86_64", "amd64", "aarch64"}:
        return value
    try:
        return next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        )
    except (OSError, StopIteration, IndexError):
        return platform.machine() or "unknown"


def load_profile(path: str | Path | None = None) -> tuple[dict | None, str | None]:
    target = Path(path) if path else default_profile_path()
    if not target.is_file():
        return None, None
    try:
        profile = json.loads(target.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return None, str(exc)
    if not isinstance(profile, dict):
        return None, "profile root must be an object"
    return profile, None


def matching_tuned_plan(
    model_info: dict,
    gpu: dict | None,
    *,
    context_size: int,
    engine_dirs: tuple[str, ...] | None = None,
    path: str | Path | None = None,
) -> tuple[dict | None, str]:
    """Return a measured plan only when every safety identity still matches."""
    profile, error = load_profile(path or profile_path_for(model_info, gpu))
    if error:
        return None, f"profile unreadable: {error}"
    if not profile:
        return None, "no measured profile"
    tuning = profile.get("tuning")
    if profile.get("schema_version") != PROFILE_SCHEMA or not isinstance(tuning, dict):
        return None, "profile is not an adaptive tuning profile"
    if tuning.get("status") != "measured":
        return None, "profile tuning is not measured"
    if tuning.get("model_identity") != model_identity(model_info):
        return None, "model artifact changed"
    if tuning.get("hardware_identity") != hardware_identity(gpu):
        return None, "hardware layout changed"
    if tuning.get("engine_identity") != engine_identity(engine_dirs):
        return None, "llama.cpp engine changed"
    free_vram = (gpu or {}).get("vram_free_mb") or 0
    minimum_free = tuning.get("minimum_free_vram_mib")
    if not isinstance(minimum_free, int) or minimum_free < 0:
        return None, "profile has no valid VRAM floor"
    if free_vram < minimum_free:
        return None, f"free VRAM {free_vram} MiB is below tuned floor {minimum_free} MiB"
    tuned_context = tuning.get("context_size")
    if not isinstance(tuned_context, int) or tuned_context < 1:
        return None, "profile has no valid tuned context"
    plan = tuning.get("selected_plan")
    if not isinstance(plan, dict):
        return None, "profile has no selected plan"
    selected = {field: plan[field] for field in TUNABLE_PLAN_FIELDS if field in plan}
    if not selected:
        return None, "profile selected plan is empty"
    if context_size > tuned_context:
        return selected, LARGER_CONTEXT_REASON
    return selected, TUNED_PROFILE_REASON


def profile_measured_rates(
    model_info: dict,
    gpu: dict | None,
    *,
    context_size: int,
    engine_dirs: tuple[str, ...] | None = None,
    path: str | Path | None = None,
) -> dict | None:
    """Measured prompt/decode tok/s from a profile that still applies.

    Returns ``None`` when the profile is absent, stale, or its benchmark
    section was not measured, so callers fall back to the analytic estimate.
    Uses the same identity/headroom guards as :func:`matching_tuned_plan`.
    """
    if not matching_tuned_plan(
        model_info,
        gpu,
        context_size=context_size,
        engine_dirs=engine_dirs,
        path=path,
    )[0]:
        return None
    profile, _ = load_profile(path or profile_path_for(model_info, gpu))
    benchmark = (profile or {}).get("benchmark") or {}
    if benchmark.get("status") != "measured":
        return None
    rates = {
        "prompt_tokens_per_second": benchmark.get("prompt_tokens_per_second"),
        "decode_tokens_per_second": benchmark.get("decode_tokens_per_second"),
    }
    if not isinstance(rates["decode_tokens_per_second"], (int, float)) or isinstance(
        rates["decode_tokens_per_second"], bool
    ):
        return None
    return rates
