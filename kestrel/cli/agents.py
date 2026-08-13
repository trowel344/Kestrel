"""CLI and launch orchestration for external coding-agent integrations."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import asdict

from .. import agent_service, integrations, ui
from ..config import load_config
from ..errors import InputError, IntegrationError
from . import parser, runtime

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
    result = agent_service.start_server(
        _default_model(args),
        alias=args.alias,
        port=args.port,
        context=str(context),
        reasoning=reasoning,
        timeout=args.timeout,
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
            },
        )
    status = agent_service.start_server(
        model,
        alias=args.alias,
        port=args.port,
        context=str(context),
        reasoning=reasoning,
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
        "doctor": _cmd_doctor,
        "logs": _cmd_logs,
        "launch": _cmd_launch,
    }
    return handlers[action](args)


__all__ = ["cmd_agents"]
