"""CLI and launch orchestration for external coding-agent integrations."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit

from .. import agent_service, integrations, ui
from ..config import load_config
from ..errors import InputError, IntegrationError
from . import parser, probes, runtime, telemetry

_CLIENTS = ("pi", "omp", "codex", "claude", "opencode")
_BINARIES = {"pi": "pi", "omp": "omp", "codex": "codex", "claude": "claude", "opencode": "opencode"}
_PROTOCOL = {
    "pi": "chat_completions",
    "omp": "chat_completions",
    "opencode": "chat_completions",
    "codex": "responses",
    "claude": "anthropic_messages",
}
_DEFAULT_WORK_CONTEXT = 32768
_MINIMUM_WORK_CONTEXT = 8192


def _selected(value: str) -> tuple[str, ...]:
    return _CLIENTS if value == "all" else (value,)


def _settings(args) -> tuple[str | int, str]:
    config = load_config()
    context = args.context if getattr(args, "context", None) is not None else getattr(config, "context_size", "auto")
    if context == "auto":
        context = _DEFAULT_WORK_CONTEXT
    if isinstance(context, int) and context < _MINIMUM_WORK_CONTEXT:
        raise IntegrationError(
            f"coding-agent context must be at least {_MINIMUM_WORK_CONTEXT} tokens",
            hint="use --context 32768 for the validated Pi/OMP work profile",
        )
    reasoning = (
        args.reasoning if getattr(args, "reasoning", None) is not None else getattr(config, "reasoning_level", "auto")
    )
    return context, reasoning


def _context_metadata(value: str | int) -> int | None:
    return value if isinstance(value, int) else None


def _effective_max_tokens(requested: int, context: str | int) -> int:
    """Keep an agent's advertised output reserve within its served context."""

    if isinstance(context, int):
        return min(requested, max(1, context // 4))
    return requested


def _default_model(args) -> str:
    if getattr(args, "model", None):
        return args.model
    return parser._default_model(args, error="Error: no model selected. Pass a model or configure a default.")


def _split_overrides(args) -> dict[str, str]:
    """Explicit layer-placement knobs passed through to the managed server."""

    config = load_config()
    gpu_layers = getattr(args, "gpu_layers", "auto")
    cpu_moe = getattr(args, "cpu_moe", "auto")
    return {
        "gpu_layers": gpu_layers if gpu_layers != "auto" else getattr(config, "gpu_layers", "auto"),
        "cpu_moe": cpu_moe if cpu_moe != "auto" else getattr(config, "cpu_moe", "auto"),
    }


def _sunmap_settings(args) -> tuple[str | None, int]:
    tokens = getattr(args, "sunmap_tokens", 4096)
    if tokens < 512:
        raise InputError("Sun Map token budget must be at least 512")
    custom = getattr(args, "sunmap_db", None)
    enabled = bool(getattr(args, "sunmap", False) or custom)
    return (
        (str(Path(custom).expanduser()) if custom else str(agent_service.sunmap_path(Path.cwd()))) if enabled else None,
        tokens,
    )


def _sunmap_trace(args) -> str | None:
    value = getattr(args, "sunmap_trace", None)
    if value and not (getattr(args, "sunmap", False) or getattr(args, "sunmap_db", None)):
        raise InputError("--sunmap-trace requires --sunmap or --sunmap-db")
    return str(Path(value).expanduser()) if value else None


def _metadata_payload(metadata) -> dict[str, object]:
    payload = asdict(metadata)
    payload["command"] = list(metadata.command)
    payload["config_path"] = str(metadata.config_path)
    return payload


def _integration_rows() -> list[dict[str, object]]:
    rows = []
    for client in _CLIENTS:
        status = integrations.status_agent_integration(client)
        rows.append(
            {
                "client": client,
                "installed": shutil.which(_BINARIES[client]) is not None,
                "configured": status.configured,
                "owned": status.owned,
                "path": str(status.path),
                "details": status.details,
            }
        )
    return rows


def _print_rows(rows: list[dict[str, object]]) -> None:
    for row in rows:
        state = "configured" if row["configured"] else "not configured"
        installed = "installed" if row["installed"] else "not installed"
        print(f"  {ui.ok_mark() if row['configured'] else ui.info_mark()} {row['client']}: {installed}, {state}")
        print(ui.dim(f"     {row['path']}"))


def _cmd_list(args):
    rows = _integration_rows()
    if not args.json:
        print(ui.box("Coding agents", "Kestrel-owned, reversible client profiles"))
        _print_rows(rows)
    return runtime._finish_json(args, {"integrations": rows})


def _cmd_setup(args):
    context, reasoning = _settings(args)
    max_tokens = _effective_max_tokens(args.max_tokens, context)
    endpoint = f"http://127.0.0.1:{args.port}"
    results = []
    for client in _selected(args.client):
        if args.dry_run:
            metadata = integrations.launch_metadata(
                client,
                model=args.alias,
                endpoint=endpoint,
                context_size=_context_metadata(context),
                reasoning=reasoning,
                max_tokens=max_tokens,
            )
        else:
            metadata = integrations.setup_agent_integration(
                client,
                model=args.alias,
                endpoint=endpoint,
                context_size=_context_metadata(context),
                reasoning=reasoning,
                max_tokens=max_tokens,
            )
        results.append(_metadata_payload(metadata))
        if not args.json:
            action = "Would configure" if args.dry_run else "Configured"
            print(f"  {ui.ok_mark()} {action} {client}: {metadata.config_path}")
            print(ui.dim("     Launch: " + " ".join(metadata.command)))
    if not args.dry_run:
        agent_service.ensure_credential()
    return runtime._finish_json(
        args,
        {
            "action": "setup",
            "dry_run": args.dry_run,
            "endpoint": endpoint,
            "model": args.alias,
            "context": context,
            "reasoning": reasoning,
            "max_tokens": max_tokens,
            "integrations": results,
        },
    )


def _cmd_remove(args):
    results = []
    for client in _selected(args.client):
        removed = integrations.remove_agent_integration(client)
        results.append({"client": client, "removed": removed})
        if not args.json:
            print(
                f"  {ui.ok_mark() if removed else ui.info_mark()} {client}: {'removed' if removed else 'not configured'}"
            )
    return runtime._finish_json(args, {"action": "remove", "integrations": results})


def _cmd_start(args):
    context, reasoning = _settings(args)
    sunmap_db, sunmap_tokens = _sunmap_settings(args)
    sunmap_trace = _sunmap_trace(args)
    result = agent_service.start_server(
        _default_model(args),
        alias=args.alias,
        port=args.port,
        context=str(context),
        reasoning=reasoning,
        timeout=args.timeout,
        sunmap_db=sunmap_db,
        sunmap_tokens=sunmap_tokens,
        sunmap_trace=sunmap_trace,
        **_split_overrides(args),
    )
    protocols = agent_service.probe_protocols()
    result["protocols"] = protocols
    if not args.json:
        print(
            ui.box(
                "Coding-agent server",
                "\n".join(
                    [ui.kv("Status", result["status"]), ui.kv("URL", result["url"]), ui.kv("Model", result["alias"])]
                ),
            )
        )
        print("  Protocols: " + ", ".join(name for name, ready in protocols.items() if ready))
    return runtime._finish_json(args, result)


def _cmd_stop(args):
    result = agent_service.stop_server()
    if not args.json:
        print(f"  {ui.ok_mark()} Coding-agent server stopped.")
    return runtime._finish_json(args, result)


def _status_payload() -> dict[str, object]:
    server = agent_service.server_status()
    protocols = (
        agent_service.probe_protocols()
        if server.get("healthy")
        else {
            "chat_completions": False,
            "responses": False,
            "anthropic_messages": False,
        }
    )
    return {"server": server, "protocols": protocols, "integrations": _integration_rows()}


def _cmd_status(args):
    payload = _status_payload()
    if not args.json:
        server = payload["server"]
        print(
            ui.box(
                "Coding-agent server",
                "\n".join([ui.kv("Status", server["status"]), ui.kv("URL", server.get("url", "not running"))]),
            )
        )
        _print_rows(payload["integrations"])
    return runtime._finish_json(args, payload)


def _cmd_doctor(args):
    payload = _status_payload()
    protocols = payload["protocols"]
    server = payload["server"]
    checks = []
    for row in payload["integrations"]:
        protocol = _PROTOCOL[row["client"]]
        details = row.get("details", {})
        endpoint_match = bool(server.get("url") and details.get("endpoint") == str(server["url"]).rstrip("/"))
        model_match = bool(server.get("alias") and details.get("model") == server.get("alias"))
        context_match = not details.get("context") or details.get("context") == str(server.get("context"))
        configuration_match = endpoint_match and model_match and context_match
        route_ready = bool(row["installed"] and row["configured"] and configuration_match and protocols.get(protocol))
        checks.append(
            {
                **row,
                "protocol": protocol,
                "protocol_ready": bool(protocols.get(protocol)),
                "configuration_match": configuration_match,
                "route_ready": route_ready,
                "client_smoke_test": "not_run",
                "runtime_validation": "route_only" if route_ready else "not_ready",
            }
        )
    considered = [check for check in checks if check["installed"] and check["configured"]]
    route_ready = bool(considered) and all(check["route_ready"] for check in considered)
    payload["status"] = "route_ready" if route_ready else "not_ready"
    payload["exit_code"] = 0 if route_ready else 1
    payload["checks"] = checks
    if not args.json:
        for check in checks:
            mark = ui.ok_mark() if check["route_ready"] else ui.warn_mark()
            label = "API route ready; client smoke not run" if check["route_ready"] else "not ready"
            print(f"  {mark} {check['client']}: {label} ({check['protocol']})")
    return runtime._finish_json(args, payload)


def _cmd_usage(args):
    server = agent_service.server_status()
    pid = server.get("pid")
    process = _process_usage(pid)
    tokens_per_second = None
    if server.get("healthy") and server.get("url"):
        host, port = _url_host_port(server["url"])
        tokens_per_second = telemetry._server_tps(host, port)
    payload = {
        "action": "usage",
        "server": server,
        "process": process,
        "tokens_per_second": tokens_per_second,
        "ram": probes._memory_snapshot(),
        "gpu": probes.detect_gpu(),
    }
    if not args.json:
        print(ui.box("Coding-agent usage", "\n".join(_usage_rows(payload))))
    return runtime._finish_json(args, payload)


def _url_host_port(url: str) -> tuple[str, int]:
    parsed = urlsplit(url)
    host = parsed.hostname or "127.0.0.1"
    return host, parsed.port or 8080


def _proc_field(pid: int, key: str) -> str | None:
    """Read a single field (e.g. ``VmRSS:``) from /proc/<pid>/status."""
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith(key):
                return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def _process_usage(pid: int | None) -> dict:
    if not pid:
        return {"available": False}
    rss_kib = _proc_field(pid, "VmRSS")
    cpu_percent = None
    elapsed_seconds = None
    try:
        result = subprocess.run(
            ["ps", "-o", "%cpu=,etimes=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.split()
            cpu_percent = float(parts[0]) if parts else None
            elapsed_seconds = int(parts[1]) if len(parts) > 1 else None
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    vram_mib = _process_vram_mib(pid)
    return {
        "available": True,
        "pid": pid,
        "rss_mib": int(rss_kib.split()[0]) // 1024 if rss_kib else None,
        "cpu_percent": cpu_percent,
        "vram_mib": vram_mib,
        "elapsed_seconds": elapsed_seconds,
    }


def _process_vram_mib(pid: int) -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        fields = line.split(",")
        if fields and fields[0].strip() == str(pid) and len(fields) > 1:
            try:
                return int(fields[1].strip())
            except ValueError:
                return None
    return None


def _usage_rows(payload: dict[str, object]) -> list[str]:
    server = payload["server"]
    rows = [ui.kv("Status", server["status"])]
    if payload["server"].get("running"):
        rows.append(ui.kv("Model", server.get("alias") or "unknown"))
        process = payload["process"]
        if process.get("available"):
            rows.append(ui.kv("PID", str(process.get("pid"))))
            if process.get("cpu_percent") is not None:
                rows.append(ui.kv("CPU", f"{process['cpu_percent']:.1f}%"))
            if process.get("rss_mib") is not None:
                rows.append(ui.kv("RAM", f"{process['rss_mib']} MiB"))
            if process.get("vram_mib") is not None:
                rows.append(ui.kv("VRAM", f"{process['vram_mib']} MiB"))
            if process.get("elapsed_seconds") is not None:
                rows.append(ui.kv("Uptime", f"{process['elapsed_seconds']}s"))
        tps = payload.get("tokens_per_second")
        if tps:
            rows.append(ui.kv("Tokens/s", f"{tps:.1f}"))
    return rows


def _cmd_logs(args):
    content = agent_service.tail_logs(args.lines)
    if args.json:
        return runtime._finish_json(args, {"log": str(agent_service.log_path()), "lines": content.splitlines()})
    print(content or ui.dim("No managed-server logs yet."))
    return 0


def _cmd_launch(args):
    if args.json and not args.dry_run:
        raise InputError("agents launch --json requires --dry-run because an interactive child owns the terminal")
    context, reasoning = _settings(args)
    sunmap_db, sunmap_tokens = _sunmap_settings(args)
    sunmap_trace = _sunmap_trace(args)
    model = _default_model(args)
    endpoint = f"http://127.0.0.1:{args.port}"
    if not args.dry_run and shutil.which(_BINARIES[args.client]) is None:
        raise IntegrationError(
            f"{args.client} is not installed",
            hint="install the client, or use --dry-run to inspect its Kestrel configuration",
        )
    integration_kwargs = {
        "model": args.alias,
        "endpoint": endpoint,
        "context_size": _context_metadata(context),
        "reasoning": reasoning,
        "max_tokens": _effective_max_tokens(8192, context),
    }
    metadata = (
        integrations.launch_metadata(args.client, **integration_kwargs)
        if args.dry_run
        else integrations.setup_agent_integration(args.client, **integration_kwargs)
    )
    command = list(metadata.command)
    if args.dry_run:
        return runtime._finish_json(
            args,
            {
                "dry_run": True,
                "server_model": model,
                "client": _metadata_payload(metadata),
                "command": command,
                "sunmap": {
                    "database": sunmap_db,
                    "token_budget": sunmap_tokens,
                    "trace": sunmap_trace,
                }
                if sunmap_db
                else None,
            },
        )
    status = agent_service.start_server(
        model,
        alias=args.alias,
        port=args.port,
        context=str(context),
        reasoning=reasoning,
        sunmap_db=sunmap_db,
        sunmap_tokens=sunmap_tokens,
        sunmap_trace=sunmap_trace,
        **_split_overrides(args),
    )
    protocols = agent_service.probe_protocols()
    required = _PROTOCOL[args.client]
    if not protocols.get(required):
        if not status.get("reused"):
            agent_service.stop_server()
        raise IntegrationError(
            f"the selected llama-server does not provide the {required} API required by {args.client}",
            hint="update the Kestrel engine and retry",
        )
    env = os.environ.copy()
    env.update(metadata.environment)
    if args.client == "omp":
        env.pop("OMP_PROFILE", None)
        env.pop("PI_PROFILE", None)
    if args.client == "opencode":
        env["KESTREL_API_KEY"] = agent_service.read_credential()
    print(f"  {ui.ok_mark()} Kestrel is ready at {status['url']}; launching {args.client}.")
    try:
        return subprocess.run(command, env=env).returncode
    except OSError as exc:
        raise IntegrationError(f"could not launch {args.client}: {exc}") from exc


def cmd_agents(args):
    action = args.agents_action
    if action == "token":
        print(agent_service.read_credential())
        return 0
    handlers = {
        "list": _cmd_list,
        "setup": _cmd_setup,
        "remove": _cmd_remove,
        "start": _cmd_start,
        "stop": _cmd_stop,
        "status": _cmd_status,
        "usage": _cmd_usage,
        "doctor": _cmd_doctor,
        "logs": _cmd_logs,
        "launch": _cmd_launch,
    }
    return handlers[action](args)


__all__ = ["cmd_agents"]
