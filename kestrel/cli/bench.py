"""Benchmarking and optimization: ``benchmark`` and ``optimize``.

``optimize`` reuses ``cmd_benchmark`` (via the planned config) to measure the
selected placement, then folds the result into a persisted hardware profile.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .. import ui
from ..backends.llama_cpp import _load_capabilities, resolve_llama_binary
from ..config import config_path
from ..errors import BackendError, InputError, KestrelError, ModelError, ServiceError
from ..tuning import PROFILE_SCHEMA, engine_identity, hardware_identity, model_identity, profile_path_for
from ..util import write_atomic
from . import model_source, parser, planning, probes, state

# Tuning reasons that mean "the current plan has no measured basis" and therefore
# justify an automatic in-line placement scan at launch time. The VRAM-floor
# reason is deliberately excluded: scanning while VRAM is scarce cannot help and
# risks a wasteful OOM.
_AUTO_TUNE_REASONS = {
    "no measured profile",
    "profile unreadable",
    "model artifact changed",
    "hardware layout changed",
    "llama.cpp engine changed",
}


def _build_optimize_profile(args, model_arg, gpu, storage_path) -> dict:
    """Resolve the local model (if any) and build the hardware/model/plan
    document for the ``optimize`` command."""
    profile = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": {
            "cpu": platform.processor(),
            "logical_cpu_count": os.cpu_count() or 0,
            "available_ram_mib": probes._available_ram_mib(),
            "memory": probes._memory_snapshot(),
            "cpu_power_policy": probes._cpu_power_policy(),
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
        from ..model_store import ModelStoreError, resolve_ollama_blob

        engine = "ollama"
        try:
            blob = resolve_ollama_blob(model_arg.removeprefix("ollama://"))
        except ModelStoreError as exc:
            raise ModelError(str(exc)) from exc
        if blob is None:
            raise ModelError("cloud-only Ollama models do not expose local metadata for optimization")
        resolved_model_arg = str(blob)
    model_info = model_source.detect_model(resolved_model_arg)
    if not model_info or model_info["type"] != "gguf" or not model_info["path"]:
        raise ModelError("optimize currently requires a local GGUF model")
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
        overcommitted = False
    else:
        context, context_reason, overcommitted = planning._select_context_size(model_info, gpu)
    plan_args = argparse.Namespace(
        gpu_layers="auto",
        ctx_size=context,
        cpu_moe="auto",
        moe_cache="off",
        moe_cold_model=None,
        target=args.quality,
        use_tuning_profile=False,
    )
    plan = planning.estimate_config(model_info, gpu, plan_args)
    plan["context_reason"] = context_reason
    plan["memory_overcommit"] = overcommitted
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
        **model_source.read_gguf_config(model_info["path"]),
    }
    profile["plan"] = plan
    return profile


def _add_tuning_candidate(candidates: list[dict], seen: set[tuple], **candidate) -> None:
    key = tuple(candidate.get(name) for name in ("gpu_layers", "cpu_moe_layers", "batch_size", "ubatch_size"))
    if key not in seen:
        seen.add(key)
        candidates.append(candidate)


def _tuning_candidates(plan: dict) -> list[dict]:
    """Build a bounded, architecture-generic placement/prefill search.

    The search deliberately avoids model-name checks. MoE candidates keep all
    experts on CPU first, then try a bounded one-eighth expert-layer slice on
    the GPU. Dense and MoE models both test larger physical micro-batches.
    Failed/OOM candidates are evidence and are retained in the profile.
    """
    candidates: list[dict] = []
    seen: set[tuple] = set()
    layers = max(0, int(plan.get("n_layers") or 0))
    experts = max(0, int(plan.get("n_experts") or 0))
    baseline_cpu_layers = plan.get("n_cpu_moe_layers")
    if baseline_cpu_layers is None and experts and plan.get("cpu_moe"):
        baseline_cpu_layers = layers
    base = {
        "gpu_layers": str(plan.get("gpu_layers", "auto")),
        "cpu_moe_layers": baseline_cpu_layers,
        "batch_size": int(plan.get("batch_size") or 256),
        "ubatch_size": int(plan.get("ubatch_size") or 64),
        "label": "planner_baseline",
    }
    _add_tuning_candidate(candidates, seen, **base)

    full_gpu_layers = str(layers + 1) if layers else str(plan.get("gpu_layers", "auto"))
    all_cpu_experts = layers if experts else None
    for batch, ubatch, label in (
        (512, 128, "larger_prefill"),
        (512, 256, "wide_prefill"),
    ):
        _add_tuning_candidate(
            candidates,
            seen,
            gpu_layers=full_gpu_layers,
            cpu_moe_layers=all_cpu_experts,
            batch_size=batch,
            ubatch_size=ubatch,
            label=label,
        )
    if experts and layers >= 8:
        # Bounded expert slices climbing toward the VRAM ceiling: the
        # one-eighth slice stays safely inside the coarse floor on small
        # cards, while the one-quarter and one-third slices probe for more
        # GPU-expert headroom. Aggressive slices are only gated by the relaxed
        # preflight; llama-bench remains the authority on what actually loads
        # and records OOM candidates as evidence instead of guessing.
        for slice_divisor in (8, 4, 3):
            gpu_expert_layers = max(1, layers // slice_divisor)
            if gpu_expert_layers >= layers:
                continue
            _add_tuning_candidate(
                candidates,
                seen,
                gpu_layers=full_gpu_layers,
                cpu_moe_layers=layers - gpu_expert_layers,
                batch_size=512,
                ubatch_size=256,
                label=f"bounded_gpu_experts_{slice_divisor}",
            )
    return candidates


# The coarse preflight below overestimates VRAM for aggressive GPU-expert
# slices (measured placements can load at ~1.3x the estimate), and llama-bench
# records OOM failures gracefully as evidence. So candidates are only skipped
# when their estimate exceeds free VRAM by a clear margin, leaving the actual
# ceiling to the measurement.
PREFLIGHT_SKIP_MARGIN = 1.5


def _poll_peak_vram(process, timeout: float) -> int:
    """Poll nvidia-smi while the benchmark process runs and return peak VRAM.

    llama-bench frees its buffers on exit, so the footprint has to be observed
    while the process is alive. Returns 0 when no NVIDIA GPU / nvidia-smi is
    available; the coarse estimate then stands in for the stored floor.
    """
    deadline = time.monotonic() + max(0.0, timeout)
    peak = 0
    while process.poll() is None:
        if time.monotonic() >= deadline:
            return peak
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                peak = max(peak, int(result.stdout.splitlines()[0].strip()))
        except (OSError, ValueError, subprocess.SubprocessError):
            return peak
        time.sleep(0.3)
    return peak


def _estimated_required_free_vram_mib(profile: dict, candidate: dict) -> int:
    """Conservative preflight floor for a benchmark candidate.

    It is intentionally coarse: llama-bench remains the authority on whether
    the placement loads. This estimate only prevents obviously oversized
    expert slices from provoking an avoidable OOM during the search.
    """
    plan = profile.get("plan") or {}
    layers = max(1, int(plan.get("n_layers") or 1))
    experts = int(plan.get("n_experts") or 0)
    size_mib = float(plan.get("model_size_gib") or 0) * 1024
    cpu_layers = candidate.get("cpu_moe_layers")
    gpu_expert_layers = max(0, layers - int(cpu_layers)) if experts and cpu_layers is not None else 0
    expert_slice = size_mib * gpu_expert_layers / layers
    dense_and_runtime = min(2048.0, size_mib * 0.15) + 1536.0
    graph = 512.0 if candidate.get("ubatch_size", 0) >= 256 else 256.0
    return max(2048, math.ceil(expert_slice + dense_and_runtime + graph))


def _candidate_score(report: dict, quality: str) -> float:
    prompt = float(report["prompt_tokens_per_second"])
    decode = float(report["decode_tokens_per_second"])
    prompt_weight = {"speed": 0.7, "balanced": 0.6, "quality": 0.5}[quality]
    return (prompt**prompt_weight) * (decode ** (1.0 - prompt_weight))


def _run_optimize_benchmark(profile, model_arg, target, args) -> None:
    """Measure bounded candidates and persist the fastest safe exact profile."""
    if not model_arg:
        raise InputError("--benchmark requires a model")
    model_info = model_source.detect_model(model_arg)
    if not model_info or model_info.get("type") != "gguf" or not model_info.get("path"):
        raise ModelError("adaptive optimization requires a local GGUF model")
    gpu = probes.detect_gpu()
    free_vram = (gpu or {}).get("vram_free_mb") or 0
    candidates = _tuning_candidates(profile["plan"])
    results = []
    try:
        for candidate in candidates:
            required_free = _estimated_required_free_vram_mib(profile, candidate)
            result = {**candidate, "estimated_required_free_vram_mib": required_free}
            if free_vram and required_free > free_vram * PREFLIGHT_SKIP_MARGIN:
                results.append({**result, "status": "skipped", "error": "estimated VRAM floor exceeds free VRAM"})
                continue
            try:
                report = cmd_benchmark(
                    argparse.Namespace(
                        model=model_arg,
                        prompt_tokens=256,
                        generate_tokens=64,
                        repetitions=2,
                        ctx_size=profile["plan"]["context_size"],
                        gpu_layers=candidate["gpu_layers"],
                        cpu_moe="on" if candidate["cpu_moe_layers"] is not None else "auto",
                        cpu_moe_layers=candidate["cpu_moe_layers"],
                        threads=profile["plan"].get("threads") or max(1, (os.cpu_count() or 2) - 2),
                        batch_size=candidate["batch_size"],
                        ubatch_size=candidate["ubatch_size"],
                        kv_cache_type=profile["plan"]["kv_cache_type"],
                        output=None,
                        quiet=True,
                    )
                )
            except KestrelError as exc:
                results.append({**result, "status": "failed", "error": str(exc)})
                continue
            results.append(
                {
                    **result,
                    "status": "measured",
                    "prompt_tokens_per_second": report["prompt_tokens_per_second"],
                    "decode_tokens_per_second": report["decode_tokens_per_second"],
                    "score": _candidate_score(report, args.quality),
                    # The observed peak (when available) is the true floor for
                    # the profile gate; the coarse estimate overstates it for
                    # aggressive GPU-expert slices.
                    "estimated_required_free_vram_mib": report.get("peak_vram_mib")
                    or result["estimated_required_free_vram_mib"],
                }
            )
        measured = [item for item in results if item["status"] == "measured"]
        if not measured:
            raise ServiceError("all adaptive tuning candidates failed or exceeded the VRAM safety floor")
        selected = max(measured, key=lambda item: item["score"])
        selected_threads = profile["plan"].get("threads") or max(1, (os.cpu_count() or 2) - 2)
        thread_refinement: dict = {"status": "not_needed"}
        if selected["cpu_moe_layers"] is not None:
            try:
                thread_report = cmd_benchmark(
                    argparse.Namespace(
                        model=model_arg,
                        prompt_tokens=128,
                        generate_tokens=32,
                        repetitions=1,
                        ctx_size=profile["plan"]["context_size"],
                        gpu_layers=selected["gpu_layers"],
                        cpu_moe="on",
                        cpu_moe_layers=selected["cpu_moe_layers"],
                        threads=planning._cpu_moe_thread_sweep(os.cpu_count() or 1),
                        batch_size=selected["batch_size"],
                        ubatch_size=selected["ubatch_size"],
                        kv_cache_type=profile["plan"]["kv_cache_type"],
                        output=None,
                        quiet=True,
                    )
                )
            except KestrelError as exc:
                thread_refinement = {"status": "failed", "error": str(exc)}
            else:
                selected_threads = thread_report.get("placement", {}).get("threads") or selected_threads
                thread_refinement = {
                    "status": "measured",
                    "selected_threads": selected_threads,
                    "thread_sweep": thread_report.get("thread_sweep", []),
                }
        selected_plan = {
            "gpu_layers": selected["gpu_layers"],
            "cpu_moe": selected["cpu_moe_layers"] is not None,
            "n_cpu_moe_layers": selected["cpu_moe_layers"],
            "batch_size": selected["batch_size"],
            "ubatch_size": selected["ubatch_size"],
            "threads": selected_threads,
            "cache_type_k": profile["plan"].get("cache_type_k", "q8_0"),
            "cache_type_v": profile["plan"].get("cache_type_v", "q8_0"),
        }
        profile["schema_version"] = PROFILE_SCHEMA
        profile["tuning"] = {
            "status": "measured",
            "model_identity": model_identity(model_info),
            "hardware_identity": hardware_identity(gpu),
            "engine_identity": engine_identity((state.LLAMA_CPP_DIR,)),
            "context_size": profile["plan"]["context_size"],
            "minimum_free_vram_mib": selected["estimated_required_free_vram_mib"],
            "selected_plan": selected_plan,
            "objective": args.quality,
            "thread_refinement": thread_refinement,
            "candidates": results,
        }
        profile["plan"].update(selected_plan)
        profile["benchmark"] = {
            "status": "measured",
            "prompt_tokens_per_second": selected["prompt_tokens_per_second"],
            "decode_tokens_per_second": selected["decode_tokens_per_second"],
            "release_speed_floor_passed": selected["decode_tokens_per_second"] >= 10.0,
            "quality_gate": "same_artifact_placement_only",
            "selected_placement": selected_plan,
        }
    except KestrelError as exc:
        profile["schema_version"] = PROFILE_SCHEMA
        profile["tuning"] = {
            "status": "failed",
            "model_identity": model_identity(model_info),
            "hardware_identity": hardware_identity(gpu),
            "engine_identity": engine_identity((state.LLAMA_CPP_DIR,)),
            "context_size": profile["plan"]["context_size"],
            "candidates": results,
            "error": str(exc),
        }
        profile["benchmark"] = {"status": "failed", "error": str(exc)}
        if not args.no_save:
            target.parent.mkdir(parents=True, exist_ok=True)
            write_atomic(target, json.dumps(profile, indent=2) + "\n")
            print(f"Wrote failed benchmark state to {target}", file=sys.stderr)
        print(json.dumps(profile, indent=2))
        raise


def _build_auto_profile(model_info: dict, gpu: dict | None, config: dict) -> dict:
    """Build the persisted profile document for an in-line auto-tune scan."""
    return {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": {
            "cpu": platform.processor(),
            "logical_cpu_count": os.cpu_count() or 0,
            "available_ram_mib": probes._available_ram_mib(),
            "memory": probes._memory_snapshot(),
            "gpu": gpu,
        },
        "model": {
            "source": model_info.get("path"),
            "engine": "llama.cpp",
            "path": model_info.get("path"),
            "size_gib": config.get("model_size_gib"),
            **model_source.read_gguf_config(model_info["path"]),
        },
        "plan": config,
        "benchmark": {"status": "not_run"},
    }


def _persist_auto_profile(profile: dict, model_info: dict, gpu: dict | None) -> Path | None:
    target = profile_path_for(model_info, gpu)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(target, json.dumps(profile, indent=2) + "\n")
    except OSError as exc:
        print(ui.dim(f"could not persist auto-tune profile: {exc}"), file=sys.stderr)
        return None
    return target


def auto_tune_plan(model_info: dict, gpu: dict | None, config: dict, args) -> dict:
    """Measure and persist an adaptive profile for an unprofiled model at launch.

    Triggered when the plan has no measured basis (no profile, or a stale one)
    and auto-tune is enabled. Runs a bounded candidate scan with reduced token
    budgets, persists the fastest safe placement, and folds the measured rates
    back into ``config`` so this very launch uses it. Never raises: failures
    log a warning and return the conservative config unchanged.
    """
    if getattr(args, "no_auto_tune", False) or getattr(args, "dry_run", False):
        return config
    if config.get("tuning_profile_applied"):
        return config
    if config.get("tuning_profile_reason") not in _AUTO_TUNE_REASONS:
        return config
    model_path = model_info.get("path")
    if not model_path:
        return config
    if not resolve_llama_binary("llama-bench", dirs=(state.LLAMA_CPP_DIR,)):
        return config
    out = sys.stderr
    print(
        ui.dim("No measured tuning profile for this model; scanning placements..."),
        file=out,
    )
    print(ui.dim("(pass --no-auto-tune to keep the conservative planner defaults)"), file=out)

    profile = _build_auto_profile(model_info, gpu, config)
    candidates = _tuning_candidates(profile["plan"])
    results: list[dict] = []
    free_vram = (gpu or {}).get("vram_free_mb") or 0
    for candidate in candidates:
        required_free = _estimated_required_free_vram_mib(profile, candidate)
        result = {**candidate, "estimated_required_free_vram_mib": required_free}
        if free_vram and required_free > free_vram * PREFLIGHT_SKIP_MARGIN:
            results.append({**result, "status": "skipped", "error": "estimated VRAM floor exceeds free VRAM"})
            continue
        try:
            report = cmd_benchmark(
                argparse.Namespace(
                    model=model_path,
                    prompt_tokens=128,
                    generate_tokens=64,
                    repetitions=2,
                    ctx_size=config["context_size"],
                    gpu_layers=candidate["gpu_layers"],
                    cpu_moe="on" if candidate["cpu_moe_layers"] is not None else "auto",
                    cpu_moe_layers=candidate["cpu_moe_layers"],
                    threads=config.get("threads") or max(1, (os.cpu_count() or 2) - 2),
                    batch_size=candidate["batch_size"],
                    ubatch_size=candidate["ubatch_size"],
                    kv_cache_type="auto",
                    output=None,
                    quiet=True,
                )
            )
        except KestrelError as exc:
            results.append({**result, "status": "failed", "error": str(exc)})
            continue
        results.append(
            {
                **result,
                "status": "measured",
                "prompt_tokens_per_second": report["prompt_tokens_per_second"],
                "decode_tokens_per_second": report["decode_tokens_per_second"],
                "score": _candidate_score(report, "balanced"),
                "estimated_required_free_vram_mib": report.get("peak_vram_mib")
                or result["estimated_required_free_vram_mib"],
            }
        )
    measured = [item for item in results if item["status"] == "measured"]
    if not measured:
        profile["schema_version"] = PROFILE_SCHEMA
        profile["tuning"] = {
            "status": "failed",
            "model_identity": model_identity(model_info),
            "hardware_identity": hardware_identity(gpu),
            "engine_identity": engine_identity((state.LLAMA_CPP_DIR,)),
            "context_size": config["context_size"],
            "candidates": results,
            "error": "all auto-tune candidates failed or exceeded the VRAM safety floor",
        }
        profile["benchmark"] = {"status": "failed", "error": profile["tuning"]["error"]}
        _persist_auto_profile(profile, model_info, gpu)
        print(
            f"  {ui.warn_mark()} {ui.yellow('auto-tune found no usable placement; keeping the planner defaults')}",
            file=out,
        )
        return config
    selected = max(measured, key=lambda item: item["score"])
    selected_threads = config.get("threads") or max(1, (os.cpu_count() or 2) - 2)
    thread_refinement: dict = {"status": "not_needed"}
    if selected["cpu_moe_layers"] is not None:
        try:
            thread_report = cmd_benchmark(
                argparse.Namespace(
                    model=model_path,
                    prompt_tokens=64,
                    generate_tokens=16,
                    repetitions=1,
                    ctx_size=config["context_size"],
                    gpu_layers=selected["gpu_layers"],
                    cpu_moe="on",
                    cpu_moe_layers=selected["cpu_moe_layers"],
                    threads=planning._cpu_moe_thread_sweep(os.cpu_count() or 1),
                    batch_size=selected["batch_size"],
                    ubatch_size=selected["ubatch_size"],
                    kv_cache_type="auto",
                    output=None,
                    quiet=True,
                )
            )
        except KestrelError as exc:
            thread_refinement = {"status": "failed", "error": str(exc)}
        else:
            selected_threads = thread_report.get("placement", {}).get("threads") or selected_threads
            thread_refinement = {
                "status": "measured",
                "selected_threads": selected_threads,
                "thread_sweep": thread_report.get("thread_sweep", []),
            }
    selected_plan = {
        "gpu_layers": selected["gpu_layers"],
        "cpu_moe": selected["cpu_moe_layers"] is not None,
        "n_cpu_moe_layers": selected["cpu_moe_layers"],
        "batch_size": selected["batch_size"],
        "ubatch_size": selected["ubatch_size"],
        "threads": selected_threads,
        "cache_type_k": config.get("cache_type_k", "q8_0"),
        "cache_type_v": config.get("cache_type_v", "q8_0"),
    }
    profile["schema_version"] = PROFILE_SCHEMA
    profile["tuning"] = {
        "status": "measured",
        "model_identity": model_identity(model_info),
        "hardware_identity": hardware_identity(gpu),
        "engine_identity": engine_identity((state.LLAMA_CPP_DIR,)),
        "context_size": config["context_size"],
        "minimum_free_vram_mib": selected["estimated_required_free_vram_mib"],
        "selected_plan": selected_plan,
        "objective": "balanced",
        "thread_refinement": thread_refinement,
        "candidates": results,
    }
    profile["benchmark"] = {
        "status": "measured",
        "prompt_tokens_per_second": selected["prompt_tokens_per_second"],
        "decode_tokens_per_second": selected["decode_tokens_per_second"],
        "release_speed_floor_passed": selected["decode_tokens_per_second"] >= 10.0,
        "quality_gate": "same_artifact_placement_only",
        "selected_placement": selected_plan,
    }
    target = _persist_auto_profile(profile, model_info, gpu)
    config.update(selected_plan)
    config["predicted_decode_tps"] = selected["decode_tokens_per_second"]
    config["prediction_confidence"] = "measured"
    config["tuning_profile_applied"] = True
    config["tuning_profile_reason"] = "auto-tuned at launch"
    config["measured_prompt_tps"] = selected["prompt_tokens_per_second"]
    config["measured_decode_tps"] = selected["decode_tokens_per_second"]
    persisted = f"; profile: {target}" if target else ""
    print(
        ui.kv(
            "Auto-tuned",
            f"{selected['decode_tokens_per_second']} decode / {selected['prompt_tokens_per_second']} prompt tok/s"
            f"{persisted}",
        ),
        file=out,
    )
    return config


def cmd_optimize(args):
    """Create an explainable hardware/model plan and optionally measure it."""
    model_arg = args.model or os.environ.get("KESTREL_MODEL") or state.USER_CONFIG.default_model
    gpu = probes.detect_gpu()
    storage_path = Path(args.storage_path or ".").expanduser().resolve()
    profile = _build_optimize_profile(args, model_arg, gpu, storage_path)
    if args.output:
        target = Path(args.output).expanduser()
    elif model_arg:
        resolved_for_profile = model_source.detect_model(model_arg)
        target = (
            profile_path_for(resolved_for_profile, gpu)
            if resolved_for_profile and resolved_for_profile.get("type") == "gguf" and resolved_for_profile.get("path")
            else config_path().with_name("hardware-profile.json")
        )
    else:
        target = config_path().with_name("hardware-profile.json")
    profile["benchmark"] = {"status": "not_run"}
    if args.benchmark:
        _run_optimize_benchmark(profile, model_arg, target, args)
    if not args.no_save:
        target.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(target, json.dumps(profile, indent=2) + "\n")
        print(f"Wrote {target}", file=sys.stderr)
    print(json.dumps(profile, indent=2))


def _summarize_benchmark_rows(rows: list[dict]) -> tuple[dict | None, dict | None, list[dict]]:
    decode_rows = [row for row in rows if row.get("n_gen")]
    decode = max(decode_rows, key=lambda row: row.get("avg_ts", 0), default=None)
    best_threads = decode.get("n_threads") if decode else None
    prompt = next(
        (row for row in rows if row.get("n_prompt") and row.get("n_threads") == best_threads),
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


def _validate_benchmark_rows(rows) -> list[dict]:
    """Reject syntactically valid JSON that is not a usable llama-bench report."""
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise BackendError("llama-bench returned JSON with an invalid report schema")
    for row in rows:
        if row.get("n_prompt") or row.get("n_gen"):
            rate = row.get("avg_ts")
            threads = row.get("n_threads")
            if not isinstance(rate, (int, float)) or isinstance(rate, bool) or not math.isfinite(rate) or rate <= 0:
                raise BackendError("llama-bench returned a measurement without a positive avg_ts")
            if not isinstance(threads, int) or isinstance(threads, bool) or threads <= 0:
                raise BackendError("llama-bench returned a measurement without a positive n_threads")
    if not any(row.get("n_prompt") for row in rows) or not any(row.get("n_gen") for row in rows):
        raise BackendError("llama-bench report is missing prompt or decode measurements")
    return rows


def _validate_benchmark_args(args) -> None:
    for name in ("prompt_tokens", "generate_tokens", "repetitions", "ctx_size", "batch_size", "ubatch_size"):
        value = getattr(args, name)
        if name in {"batch_size", "ubatch_size"} and value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise InputError(f"--{name.replace('_', '-')} must be a positive integer")


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
        lines.append(
            ui.kv(
                "Placement",
                f"GPU layers={placement.get('gpu_layers', 'unknown')}, "
                f"threads={placement.get('threads', 'unknown')}, "
                f"CPU MoE={'on' if placement.get('cpu_moe') else 'off'}, "
                f"KV={placement.get('kv_cache_type', 'unknown')}",
            )
        )
    print(ui.box("Benchmark", "\n".join(lines)))


def cmd_benchmark(args):
    _validate_benchmark_args(args)
    model_arg = parser._default_model(
        args,
        error="Error: no benchmark model selected; run `kestrel setup --model ...`",
    )
    if model_arg.startswith("ollama://"):
        from ..providers.ollama import OllamaClient, OllamaError

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
            raise ServiceError(f"Ollama benchmark failed: {exc}") from exc
        prompt_rates = [item.prompt_tps for item in samples if item.prompt_tps is not None]
        decode_rates = [item.decode_tps for item in samples if item.decode_tps is not None]
        if (
            not prompt_rates
            or not decode_rates
            or any(not math.isfinite(rate) or rate <= 0 for rate in prompt_rates + decode_rates)
        ):
            raise ServiceError("Ollama benchmark returned no usable prompt or decode measurements")
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
            "sample_output": (samples[-1].response or samples[-1].thinking if samples else ""),
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
            write_atomic(output, encoded + "\n")
            print(f"Wrote {output}", file=sys.stderr)
        if getattr(args, "json", False):
            print(json.dumps(report))
        elif not getattr(args, "quiet", False):
            _print_benchmark_summary(report)
            print(encoded)
        return report
    model_info = model_source.detect_model(model_arg)
    if not model_info or model_info["type"] != "gguf" or not model_info["path"]:
        raise ModelError("benchmark requires a local GGUF model")
    # Benchmark the same configured engine used by run/serve. Searching every
    # known checkout can silently select an older, ABI-incompatible bench
    # binary even though the configured llama-cli loads the model correctly.
    binary = resolve_llama_binary("llama-bench", dirs=(state.LLAMA_CPP_DIR,))
    if not binary:
        raise BackendError("llama-bench was not found", hint="build llama.cpp first")

    gpu = probes.detect_gpu()
    plan_args = argparse.Namespace(
        gpu_layers=args.gpu_layers,
        ctx_size=args.ctx_size,
        cpu_moe=args.cpu_moe,
        moe_cache="off",
        moe_cold_model=None,
    )
    config = planning.estimate_config(model_info, gpu, plan_args)
    requested_cpu_moe_layers = getattr(args, "cpu_moe_layers", None)
    if requested_cpu_moe_layers is not None:
        if requested_cpu_moe_layers > config["n_layers"]:
            raise InputError("--cpu-moe-layers cannot exceed the model layer count")
        config["cpu_moe"] = requested_cpu_moe_layers > 0
        config["n_cpu_moe_layers"] = requested_cpu_moe_layers
    threads = args.threads or config["threads"] or max(1, (os.cpu_count() or 2) - 2)
    batch_size = args.batch_size or config["batch_size"]
    ubatch_size = args.ubatch_size or config["ubatch_size"]
    kv_cache_type = args.kv_cache_type or config["cache_type_k"]
    if (
        isinstance(threads, bool)
        or (isinstance(threads, int) and threads <= 0)
        or (isinstance(threads, str) and not re.fullmatch(r"[1-9]\d*(?:,[1-9]\d*)*", threads))
    ):
        raise InputError("--threads must be a positive integer or comma-separated positive integers")
    planned_gpu_layers = str(config["gpu_layers"])
    bench_gpu_layers = "99" if planned_gpu_layers in {"auto", "all"} else planned_gpu_layers
    # llama-bench lags llama-cli/server on some flags: emit --threads-batch only
    # when this build's llama-bench actually exposes it, so an older engine
    # still benchmarks instead of failing on an unknown option. A failed probe
    # (missing binary, older build) degrades to the legacy flag set.
    try:
        bench_caps = _load_capabilities(binary, refresh=False)
    except (BackendError, OSError):
        bench_caps = None
    command = [
        binary,
        "-m",
        model_info["path"],
        "-p",
        str(args.prompt_tokens),
        "-n",
        str(args.generate_tokens),
        "-r",
        str(args.repetitions),
        "-ngl",
        bench_gpu_layers,
        "-t",
        str(threads),
    ]
    if bench_caps is not None and bench_caps.supports("--threads-batch"):
        command += ["-tb", str(config.get("threads_batch") or threads)]
    command += [
        "-b",
        str(batch_size),
        "-ub",
        str(ubatch_size),
        "-fa",
        "on",
        "-ctk",
        kv_cache_type,
        "-ctv",
        kv_cache_type,
        "-o",
        "json",
    ]
    if planned_gpu_layers == "auto":
        command.extend(["-fitt", str(config["fit_target_mib"])])
    if config.get("n_cpu_moe_layers") is not None:
        command.extend(["-ncmoe", str(config["n_cpu_moe_layers"])])
    elif config["cpu_moe"]:
        command.extend(["-ncmoe", str(config["n_layers"])])
    if not getattr(args, "quiet", False):
        print("Benchmarking the exact configured placement...", file=sys.stderr)
    bench_timeout = 30 * 60 * max(1, args.repetitions or 1)
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        peak_vram_mib = _poll_peak_vram(process, bench_timeout)
        stdout, stderr = process.communicate(timeout=max(1, bench_timeout))
    except subprocess.TimeoutExpired as exc:
        process.kill()
        process.communicate()
        raise ServiceError(
            f"llama-bench exceeded its time budget ({bench_timeout}s)",
            hint="reduce --repetitions or the token counts",
        ) from exc
    except OSError as exc:
        raise BackendError(f"could not launch llama-bench: {exc}") from exc
    result = argparse.Namespace(returncode=process.returncode, stdout=stdout, stderr=stderr)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout)[-3000:]
        raise BackendError(f"llama-bench failed ({result.returncode}):\n{detail}")
    try:
        rows = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise BackendError(f"llama-bench returned invalid JSON: {exc}") from exc
    rows = _validate_benchmark_rows(rows)
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
            "batch_size": batch_size,
            "ubatch_size": ubatch_size,
            "kv_cache_type": kv_cache_type,
        },
        "prompt_tokens_per_second": prompt.get("avg_ts") if prompt else None,
        "decode_tokens_per_second": decode.get("avg_ts") if decode else None,
        "release_speed_floor_passed": bool(decode and decode.get("avg_ts", 0) >= 10),
        "quality_gate": "not_run",
        "thread_sweep": thread_sweep,
        "peak_vram_mib": peak_vram_mib,
        "raw": rows,
    }
    encoded = json.dumps(report, indent=2)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(output, encoded + "\n")
        print(f"Wrote {output}", file=sys.stderr)
    if getattr(args, "json", False):
        print(json.dumps(report))
    elif not getattr(args, "quiet", False):
        _print_benchmark_summary(report)
        print(encoded)
    return report
