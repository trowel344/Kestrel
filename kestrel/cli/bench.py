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
from ..backends.llama_cpp import resolve_llama_binary
from ..config import config_path
from ..errors import BackendError, InputError, KestrelError, ModelError, ServiceError
from ..util import write_atomic
from . import model_source, parser, planning, probes, state


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
    else:
        context, context_reason = planning._select_context_size(model_info, gpu)
    plan_args = argparse.Namespace(
        gpu_layers="auto",
        ctx_size=context,
        cpu_moe="auto",
        moe_cache="off",
        moe_cold_model=None,
        target=args.quality,
    )
    plan = planning.estimate_config(model_info, gpu, plan_args)
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
        **model_source.read_gguf_config(model_info["path"]),
    }
    profile["plan"] = plan
    return profile


def _run_optimize_benchmark(profile, model_arg, target, args) -> None:
    """Run the measurement sweep for ``optimize`` and fold the results into
    ``profile``, raising a :class:`KestrelError` when the run fails (the caller then
    re-raises after persisting the failure state)."""
    if not model_arg:
        raise InputError("--benchmark requires a model")
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
                threads=(planning._cpu_moe_thread_sweep(os.cpu_count() or 1) if profile["plan"]["cpu_moe"] else None),
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
    except KestrelError as exc:
        profile["benchmark"] = {"status": "failed", "error": str(exc)}
        if not args.no_save:
            target.parent.mkdir(parents=True, exist_ok=True)
            write_atomic(target, json.dumps(profile, indent=2) + "\n")
            print(f"Wrote failed benchmark state to {target}", file=sys.stderr)
        print(json.dumps(profile, indent=2))
        raise


def cmd_optimize(args):
    """Create an explainable hardware/model plan and optionally measure it."""
    model_arg = args.model or os.environ.get("KESTREL_MODEL") or state.USER_CONFIG.default_model
    gpu = probes.detect_gpu()
    storage_path = Path(args.storage_path or ".").expanduser().resolve()
    profile = _build_optimize_profile(args, model_arg, gpu, storage_path)
    target = Path(args.output).expanduser() if args.output else config_path().with_name("hardware-profile.json")
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
    threads = args.threads or config["threads"] or max(1, (os.cpu_count() or 2) - 2)
    if (
        isinstance(threads, bool)
        or (isinstance(threads, int) and threads <= 0)
        or (isinstance(threads, str) and not re.fullmatch(r"[1-9]\d*(?:,[1-9]\d*)*", threads))
    ):
        raise InputError("--threads must be a positive integer or comma-separated positive integers")
    planned_gpu_layers = str(config["gpu_layers"])
    bench_gpu_layers = "99" if planned_gpu_layers in {"auto", "all"} else planned_gpu_layers
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
        "-b",
        str(args.batch_size),
        "-ub",
        str(args.ubatch_size),
        "-fa",
        "on",
        "-ctk",
        args.kv_cache_type,
        "-ctv",
        args.kv_cache_type,
        "-o",
        "json",
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
        raise ServiceError(
            f"llama-bench exceeded its time budget ({bench_timeout}s)",
            hint="reduce --repetitions or the token counts",
        ) from exc
    except OSError as exc:
        raise BackendError(f"could not launch llama-bench: {exc}") from exc
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
        write_atomic(output, encoded + "\n")
        print(f"Wrote {output}", file=sys.stderr)
    if getattr(args, "json", False):
        print(json.dumps(report))
    elif not getattr(args, "quiet", False):
        _print_benchmark_summary(report)
        print(encoded)
    return report
