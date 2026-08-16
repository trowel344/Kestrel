"""The ``run``/``chat``/``serve`` entry commands.

These are the launch paths tests drive through the real parser, so the helper
modules below are accessed by module attribute at call time to keep the
monkeypatch contract (patching ``cli.runtime._configure_backend`` etc.) intact.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

from .. import ui
from ..errors import BackendError, InputError, ModelError, ServiceError
from . import bench, model_source, parser, planning, probes, runtime, telemetry


def _ollama_run_command(name: str, reasoning: str) -> list[str]:
    """Build the provider fallback command without pretending it owns context."""
    command = ["ollama", "run", name]
    if reasoning == "off":
        command.extend(["--think", "false"])
    elif reasoning in {"low", "medium", "high"}:
        command.extend(["--think", reasoning])
    elif reasoning == "maximum":
        command.extend(["--think", "true"])
    return command


def _prepare_run_model(args):
    """Resolve the requested model, converting/placing it for the Kestrel
    runtime and producing the fitting config.

    Falls back to passthrough for cloud-only Ollama models. Returns ``None``
    when the Ollama adapter has taken over in dry-run mode, otherwise a
    ``(model_info, config)`` pair."""
    if not args.model:
        args.model = parser._default_model(
            args,
            error="Error: no model selected. Pass a GGUF path or set KESTREL_MODEL.",
        )
    if args.model.startswith("ollama://"):
        name = args.model.removeprefix("ollama://")
        blob = runtime._resolve_ollama_native(name)
        if blob is not None:
            out = runtime._human_stream(args)
            print(
                ui.dim(f"Ollama model {name} resolved to local GGUF: {blob}"),
                file=out,
            )
            print(
                ui.dim(
                    "Running through the Kestrel runtime with fit and context guardrails (Ollama runtime bypassed)."
                ),
                file=out,
            )
            args.model = blob
        else:
            ollama_command = _ollama_run_command(name, getattr(args, "reasoning", "auto"))
            if args.dry_run:
                out = runtime._human_stream(args)
                print("Runtime: Ollama adapter", file=out)
                print("Command: " + shlex.join(ollama_command), file=out)
                print(
                    "Note: context and placement are owned by the Ollama runtime for this adapter.",
                    file=out,
                )
                runtime._finish_json(
                    args,
                    {
                        "model": args.model,
                        "dry_run": True,
                        "engine": "ollama",
                        "command": ollama_command,
                    },
                )
                return None
            try:
                if getattr(args, "json", False):
                    result = subprocess.run(ollama_command, stdout=sys.stderr, stderr=sys.stderr)
                    return runtime._finish_json(
                        args,
                        {
                            "model": args.model,
                            "engine": "ollama",
                            "exit_code": result.returncode,
                        },
                    )
                result = subprocess.run(ollama_command)
            except FileNotFoundError as exc:
                raise BackendError("Ollama is not installed", hint="install Ollama or pass a local GGUF path") from exc
            except OSError as exc:
                raise BackendError(f"could not start Ollama: {exc}") from exc
            raise SystemExit(result.returncode)
    model_info = model_source._resolve_model_source(args)

    gpu_info = probes.detect_gpu()
    args._gpu = gpu_info
    # Resolve node placement before command construction. A requested node
    # must become an explicit RPC endpoint or produce a typed error.
    args._node_plan = runtime._resolve_node_plan(args)
    out = runtime._human_stream(args)
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

    model_info = model_source._ensure_local_gguf(model_info, args)
    args._node_plan = runtime._annotate_node_model_fit(args._node_plan, model_info)

    config = planning.estimate_config(model_info, gpu_info, args)
    config = bench.auto_tune_plan(model_info, gpu_info, config, args)
    return model_info, config


def _print_run_plan(args, config, cmd, llama_cli_version, *, hot_model_path=None, cold_model_path=None):
    """Print the explainable runtime plan, notes and command block before
    launching the interactive/runtime execution phase of ``run``."""
    out = runtime._human_stream(args)
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
        ui.kv(
            "CPU MoE",
            "enabled" if config["cpu_moe"] else "disabled",
            value_color=ui.green if config["cpu_moe"] else ui.dim,
        ),
        ui.kv(
            "MoE cache",
            f"{config['moe_cache']} ({config['moe_cache_budget_mib']} MiB budget)",
        ),
    ]
    node_plan = runtime._node_plan_payload(getattr(args, "_node_plan", None))
    if node_plan["requested"]:
        plan_lines.append(ui.kv("Nodes", ", ".join(node.get("name", "?") for node in node_plan["nodes"]) or "selected"))
        plan_lines.append(ui.kv("RPC", ",".join(node_plan["rpc_endpoints"])))
        if node_plan.get("tensor_split"):
            plan_lines.append(ui.kv("Tensor split", str(node_plan["tensor_split"])))
    if hot_model_path:
        plan_lines.append(ui.kv("MoE Q4 hot sidecar", hot_model_path))
    if cold_model_path:
        plan_lines.append(ui.kv("MoE Q1 cold sidecar", cold_model_path))
    planned_threads = args.threads if args.threads is not None else config["threads"] or "llama.cpp default"
    planned_batch_threads = (
        args.threads_batch
        if getattr(args, "threads_batch", None) is not None
        else (config.get("threads_batch") or 0)
    )
    plan_lines.extend(
        [
            ui.kv("Threads", str(planned_threads)),
            ui.kv("Prompt threads", str(planned_batch_threads or "llama.cpp default")),
            ui.kv("Context", f"{config['context_size']} ({config['context_reason']})"),
            ui.kv("Reasoning", config.get("reasoning_level", "auto")),
            ui.kv("KV cache", str(args.kv_cache_type)),
            ui.kv(
                "Batch / micro-batch",
                f"{args.batch_size or config['batch_size']} / {args.ubatch_size or config['ubatch_size']}",
            ),
            ui.kv("MTP", "enabled" if mtp_enabled else "disabled", value_color=ui.green if mtp_enabled else ui.dim),
        ]
    )
    predicted = config["predicted_decode_tps"]
    confidence = config.get("prediction_confidence", "uncalibrated-model-estimate")
    confidence_note = "measured profile" if confidence == "measured" else f"{confidence}; benchmark required"
    plan_lines.append(
        ui.kv(
            "Decode estimate",
            f"{predicted} tok/s ({confidence_note})",
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
    if config.get("context_scaled"):
        notes.append(
            "Applied the measured placement at a context larger than the tuned "
            "size, so the KV cache grows beyond the profile's measured "
            "footprint. Free RAM/VRAM may become the limit; prefer the tuned "
            "context or free memory if launches become unstable."
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


def _cmd_run_live(args):
    prepared = _prepare_run_model(args)
    if prepared is None:
        return 0
    if isinstance(prepared, int):
        return prepared
    model_info, config = prepared
    run_env = None
    hot_model_path = None
    cold_model_path = None
    if args.moe_hot_model and args.moe_cold_model:
        raise InputError("--moe-hot-model and --moe-cold-model are mutually exclusive")
    if args.moe_hot_model:
        hot_model_path = str(Path(args.moe_hot_model).expanduser().resolve())
        if not Path(hot_model_path).is_file():
            raise ModelError(f"MoE hot sidecar does not exist: {hot_model_path}")
        if config["moe_cache"] != "on":
            raise InputError("--moe-hot-model requires --moe-cache on or an explicit MiB budget")
        run_env = os.environ.copy()
        run_env.pop("LLAMA_MOE_COLD_GGUF", None)
        run_env["LLAMA_MOE_HOT_GGUF"] = hot_model_path
    if args.moe_cold_model:
        cold_model_path = str(Path(args.moe_cold_model).expanduser().resolve())
        if not Path(cold_model_path).is_file():
            raise ModelError(f"MoE cold sidecar does not exist: {cold_model_path}")
        if config["moe_cache"] != "on":
            raise InputError("--moe-cold-model requires --moe-cache on or an explicit MiB budget")
        run_env = os.environ.copy()
        run_env.pop("LLAMA_MOE_HOT_GGUF", None)
        run_env["LLAMA_MOE_COLD_GGUF"] = cold_model_path
    backend = runtime._configure_backend(model_info, config, args)
    cmd = backend._build_interactive_cmd()
    if getattr(args, "session", None) and backend.capabilities().supports("--prompt-cache"):
        cmd = runtime._with_session_flags(cmd, args.session, args.model, interactive=True)
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
        return runtime._finish_json(
            args,
            {
                "model": args.model,
                "dry_run": True,
                "model_path": model_info["path"],
                "command": cmd,
                "nodes": runtime._node_plan_payload(getattr(args, "_node_plan", None)),
            },
        )
    if args.prompt:
        return runtime._oneshot_run(backend, cmd, args)
    if args.warm_cache:
        if direct_io:
            print(
                ui.dim("--warm-cache has no effect with --direct-io (direct I/O bypasses the page cache)."),
                file=runtime._human_stream(args),
            )
        else:
            warm_paths = [model_info["path"]]
            if cold_model_path:
                warm_paths.append(cold_model_path)
            if hot_model_path:
                warm_paths.append(hot_model_path)
            print(ui.dim("Warming the model page cache before launch..."), file=runtime._human_stream(args))
            probes._warm_page_cache(warm_paths)
    retries = 0 if args.no_oom_retry else 2
    rc = runtime._run_with_oom_retries(
        cmd,
        max_retries=retries,
        env=run_env,
        stdout=runtime._human_stream(args) if args.json else None,
    )
    return runtime._finish_json(
        args,
        {
            "model": args.model,
            "exit_code": rc,
            "nodes": runtime._node_plan_payload(getattr(args, "_node_plan", None)),
        },
    )


def cmd_run(args):
    """Run a model and always reap any managed SSH worker forwards."""

    try:
        return _cmd_run_live(args)
    finally:
        runtime._close_node_tunnels(args)


cmd_chat = cmd_run


def _stop_server_process(proc) -> int:
    """Terminate and reap a server process, escalating to kill after 5s."""
    if proc.poll() is not None:
        return proc.returncode
    try:
        proc.terminate()
        return proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        return proc.wait()
    except OSError:
        return proc.poll() if proc.poll() is not None else 1


def _wait_server_process(proc) -> int:
    """Wait for a server and normalize Ctrl-C to the conventional status 130."""
    try:
        return proc.wait()
    except KeyboardInterrupt:
        _stop_server_process(proc)
        return 130


def _wait_server_ready(proc, host: str, port: int, *, timeout: float) -> bool:
    """Poll readiness while also failing immediately when llama-server exits."""

    deadline = time.perf_counter() + timeout
    while proc.poll() is None:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return False
        if runtime._wait_ready(host, port, timeout=min(0.5, remaining), interval=0.1):
            return True
    return False


def _cmd_serve_live(args):
    """Serve the planned model over an OpenAI-compatible llama-server endpoint."""
    api_key_file = getattr(args, "api_key_file", None)
    if args.host not in {"127.0.0.1", "::1", "localhost"} and not api_key_file:
        raise InputError(
            "non-loopback serving requires --api-key-file",
            hint="keep the default 127.0.0.1 bind or provide a private API-key file and TLS-capable reverse proxy",
        )
    if api_key_file:
        key_path = Path(api_key_file).expanduser()
        if key_path.is_symlink() or not key_path.is_file():
            raise InputError("--api-key-file must be a regular, non-symlink file")
        try:
            key_mode = key_path.stat().st_mode
        except OSError as exc:
            raise InputError(f"could not inspect --api-key-file: {exc}") from exc
        if key_mode & 0o077:
            raise InputError("--api-key-file must not be readable or writable by group/other", hint="run chmod 600")
        try:
            if key_path.stat().st_size > 64 * 1024:
                raise InputError("--api-key-file is larger than the 64 KiB safety limit")
            key_lines = key_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            raise InputError(f"could not read --api-key-file: {exc}") from exc
        usable_keys = [line for line in key_lines if line and not line.startswith("#")]
        if not usable_keys:
            raise InputError("--api-key-file contains no usable API key")
        if any(
            line != line.strip()
            or len(line) < 16
            or len(line) > 512
            or any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in line)
            for line in usable_keys
        ):
            raise InputError("--api-key-file keys must be 16-512 non-whitespace characters with no surrounding space")
        args.api_key_file = str(key_path.resolve())
    if not args.model:
        args.model = parser._default_model(
            args,
            error="Error: no model selected. Pass a GGUF path or set KESTREL_MODEL.",
        )
    if args.model.startswith("ollama://"):
        name = args.model.removeprefix("ollama://")
        blob = runtime._resolve_ollama_native(name)
        if blob is not None:
            print(
                ui.dim(
                    f"Ollama model {name} resolved to local GGUF: {blob} (serving via Kestrel's llama-server runtime)"
                ),
                file=runtime._human_stream(args),
            )
            args.model = blob
        else:
            raise ModelError(
                "kestrel serve hosts GGUF models through llama-server",
                hint="for Ollama models use `ollama serve` or `ollama run`",
            )
    model_info = model_source._resolve_model_source(args)
    model_info = model_source._ensure_local_gguf(model_info, args)

    gpu_info = probes.detect_gpu()
    args._gpu = gpu_info
    args._node_plan = runtime._resolve_node_plan(args)
    config = planning.estimate_config(model_info, gpu_info, args)
    config = bench.auto_tune_plan(model_info, gpu_info, config, args)
    cmd = runtime._build_server_cmd(model_info, config, args)
    host = args.host
    port = args.port
    out = runtime._human_stream(args)
    print(
        ui.box(
            "Serving",
            "\n".join(
                [
                    ui.kv("Model", model_info["path"], value_color=ui.bold),
                    ui.kv("Engine", "llama-server"),
                    ui.kv("URL", f"http://{host}:{port}", value_color=ui.cyan),
                    ui.kv("Context", f"{config['context_size']} ({config['context_reason']})"),
                    ui.kv("Reasoning", config.get("reasoning_level", "auto")),
                    ui.kv("GPU layers", str(config["gpu_layers"])),
                    ui.kv("CPU MoE", "enabled" if config["cpu_moe"] else "disabled"),
                    *(
                        [
                            ui.kv(
                                "Nodes",
                                ", ".join(
                                    node.get("name", "?")
                                    for node in runtime._node_plan_payload(args._node_plan)["nodes"]
                                )
                                or "selected",
                            ),
                            ui.kv(
                                "RPC",
                                ",".join(runtime._node_plan_payload(args._node_plan)["rpc_endpoints"]),
                            ),
                        ]
                        if runtime._node_plan_payload(args._node_plan)["requested"]
                        else []
                    ),
                    "",
                    "Agent endpoints: /v1/chat/completions, /v1/responses, /v1/messages, /v1/models",
                    "Press Ctrl+C to stop the server.",
                ]
            ),
            title_color=ui.green,
        ),
        file=out,
    )
    if args.dry_run:
        print("  " + shlex.join(cmd), file=out)
        return runtime._finish_json(
            args,
            {
                "model": args.model,
                "dry_run": True,
                "url": f"http://{host}:{port}",
                "command": cmd,
                "nodes": runtime._node_plan_payload(getattr(args, "_node_plan", None)),
            },
        )
    if not args.wait and not getattr(args, "json", False):
        raise SystemExit(runtime._run_with_oom_retries(list(cmd)))
    try:
        proc = subprocess.Popen(list(cmd), **({"stdout": sys.stderr} if getattr(args, "json", False) else {}))
    except OSError as exc:
        raise ServiceError(f"could not start llama-server: {exc}") from exc
    result = {
        "model": args.model,
        "url": f"http://{host}:{port}",
        "host": host,
        "port": port,
        "command": cmd,
        "nodes": runtime._node_plan_payload(getattr(args, "_node_plan", None)),
    }
    timeout = float(args.wait) if args.wait else 30.0
    print(
        ui.dim(f"Waiting for the server on http://{host}:{port}/health ..."),
        file=out,
    )
    started = time.perf_counter()
    try:
        ready = _wait_server_ready(proc, host, port, timeout=timeout)
    except KeyboardInterrupt:
        _stop_server_process(proc)
        if getattr(args, "json", False):
            result.update(status="interrupted", exit_code=130)
            return runtime._finish_json(args, result)
        raise SystemExit(130) from None
    except BaseException:
        _stop_server_process(proc)
        raise
    elapsed = round(time.perf_counter() - started, 3)
    if not ready:
        _stop_server_process(proc)
        exited = proc.poll()
        summary = (
            f"llama-server exited with status {exited} before becoming ready"
            if exited is not None
            else f"llama-server did not become ready on http://{host}:{port}/health within {timeout:.0f}s"
        )
        raise ServiceError(
            summary + ".",
            hint=(
                "check the llama-server output above for load errors "
                "(CUDA OOM, missing GGUF, or an incompatible quant) and "
                "free GPU memory before retrying"
            ),
        )
    result["status"] = "ready"
    result["ready_after_s"] = elapsed
    print(ui.green("Server is ready."), file=out)
    stop_telemetry = threading.Event()
    dashboard = None
    if not getattr(args, "json", False):
        dashboard = threading.Thread(
            target=telemetry.live_dashboard,
            args=(stop_telemetry,),
            kwargs={"host": host, "port": port},
            daemon=True,
            name="kestrel-telemetry",
        )
        dashboard.start()
    try:
        returncode = _wait_server_process(proc)
    finally:
        if dashboard is not None:
            stop_telemetry.set()
            dashboard.join(timeout=1)
    if getattr(args, "json", False):
        result["exit_code"] = returncode
        return runtime._finish_json(args, result)
    raise SystemExit(returncode)


def cmd_serve(args):
    """Serve a model and always reap any managed SSH worker forwards."""

    try:
        return _cmd_serve_live(args)
    finally:
        runtime._close_node_tunnels(args)
