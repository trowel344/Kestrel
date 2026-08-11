"""CLI management and preflight commands for experimental RPC nodes."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import replace

from .. import engine, nodes, ui
from ..errors import InputError
from . import model_source, probes, runtime, state


def _store(args) -> nodes.NodeStore:
    return nodes.NodeStore(allow_insecure_direct_rpc=bool(getattr(args, "allow_insecure_rpc", False)))


def _local_provenance() -> tuple[list[int], str]:
    gpu = probes.detect_gpu() or {}
    devices = gpu.get("devices") or []
    capacities = [max(0, int(item.get("vram_free_mb") or 0)) for item in devices]
    if not capacities and gpu.get("vram_free_mb") is not None:
        capacities = [max(0, int(gpu.get("vram_free_mb") or 0))]
    manifest = engine.load_manifest(state.LLAMA_CPP_DIR)
    commit = (manifest.commit if manifest else None) or engine.git_head(state.LLAMA_CPP_DIR)
    if not commit:
        raise InputError(
            "selected llama.cpp engine has no verifiable git provenance",
            hint="adopt or build it with `kestrel engine set --dir PATH` before planning nodes",
        )
    return capacities, commit


def _emit(args, payload: dict, lines: list[str] = ()) -> int:
    if getattr(args, "json", False):
        return runtime._finish_json(args, payload)
    for line in lines:
        print(line)
    return int(payload.get("exit_code", 0))


@contextmanager
def _with_tunnels(entries, timeout: float):
    """Yield inventory entries with managed endpoints replaced by forwards."""

    stack = ExitStack()
    replacements = []
    try:
        for item in entries:
            if item.managed and item.enabled:
                tunnel = stack.enter_context(nodes.SshTunnel(item, timeout=timeout))
                replacements.append(replace(item, endpoint=tunnel.endpoint))
            else:
                replacements.append(item)
        yield tuple(replacements)
    finally:
        stack.close()


def _cmd_list(args) -> int:
    entries = _store(args).load()
    payload = {
        "status": "ok",
        "experimental": True,
        "nodes": [item.as_dict() for item in entries],
    }
    lines = [ui.bold("Experimental RPC nodes")]
    if not entries:
        lines.append("No nodes configured.")
    for item in entries:
        state_text = "enabled" if item.enabled else "disabled"
        suffix = f", managed SSH {item.ssh_user}@{item.ssh_host}" if item.managed else ", direct loopback endpoint"
        lines.append(
            f"{item.name}: {item.rpc_endpoint} ({state_text}, advertised {item.accelerator_memory_mib} MiB{suffix})"
        )
    return _emit(args, payload, lines)


def _cmd_add(args) -> int:
    managed = any(
        value is not None for value in (args.ssh_host, args.ssh_user, args.ssh_identity_file, args.ssh_host_key)
    )
    if managed and not all(
        value is not None for value in (args.ssh_host, args.ssh_user, args.ssh_identity_file, args.ssh_host_key)
    ):
        raise InputError("managed SSH requires --ssh-host, --ssh-user, --identity-file, and --host-key")
    if not managed and args.remote_rpc_port is not None:
        raise InputError("--remote-rpc-port requires managed SSH options")
    if not managed and args.ssh_port != 22:
        raise InputError("--ssh-port requires managed SSH options")
    if managed:
        endpoint = nodes.parse_endpoint(args.endpoint)
        if not nodes.is_loopback_host(endpoint.host):
            raise nodes.NodeSecurityError(
                "managed SSH nodes require a loopback coordinator endpoint",
                hint="use 127.0.0.1 or localhost as --endpoint; Kestrel allocates the forwarded port at launch",
            )
    entry = nodes.Node(
        name=args.name,
        endpoint=args.endpoint,
        accelerator_memory_mib=args.memory_mib,
        ram_mib=args.ram_mib,
        enabled=not args.disabled,
        engine_version=args.engine_version,
        engine_commit=args.engine_commit,
        model_cache_hashes=tuple(args.model_hash or ()),
        ssh_host=args.ssh_host,
        ssh_user=args.ssh_user,
        ssh_port=args.ssh_port,
        ssh_identity_file=args.ssh_identity_file,
        ssh_host_key=args.ssh_host_key,
        remote_rpc_port=(args.remote_rpc_port or 50052) if managed else None,
    )
    _store(args).upsert(entry)
    return _emit(
        args,
        {"status": "saved", "experimental": True, "node": entry.as_dict()},
        [f"Saved node {entry.name}: {entry.rpc_endpoint}"],
    )


def _cmd_remove(args) -> int:
    _store(args).remove(args.name)
    return _emit(args, {"status": "removed", "node": args.name}, [f"Removed node {args.name}."])


def _cmd_doctor(args) -> int:
    inventory = _store(args).load()
    wanted = set(args.name or ())
    selected = [item for item in inventory if not wanted or item.name in wanted]
    missing = sorted(wanted - {item.name for item in selected})
    if missing:
        raise nodes.NodeValidationError(f"unknown node(s): {', '.join(missing)}")
    if not selected:
        return _emit(
            args,
            {"status": "no_nodes", "experimental": True, "checks": [], "exit_code": 0},
            ["No nodes configured; no RPC checks were performed."],
        )
    checks = []
    failed = False
    with _with_tunnels(selected, args.timeout) as probed:
        for item in probed:
            if not item.enabled:
                checks.append({"name": item.name, "endpoint": item.rpc_endpoint, "status": "disabled"})
                continue
            result = nodes.probe_rpc(
                item, timeout=args.timeout, allow_insecure_direct_rpc=bool(args.allow_insecure_rpc)
            )
            status = "usable" if result.usable else "failed"
            failed = failed or not result.usable
            checks.append(
                {
                    "name": item.name,
                    "endpoint": item.rpc_endpoint,
                    "status": status,
                    "transport": "managed_ssh" if item.managed else "direct_loopback",
                    "tcp_reachable": result.tcp_reachable,
                    "rpc_protocol": result.rpc_protocol,
                    "protocol_version": result.version,
                    "protocol_compatible": result.protocol_compatible,
                    "device_count": result.device_count,
                    "device_memory": result.device_memory,
                    "configured_engine_commit": item.engine_commit,
                    "error": result.error,
                }
            )
    exit_code = 1 if failed else 0
    payload = {"status": "failed" if failed else "ok", "experimental": True, "checks": checks, "exit_code": exit_code}
    lines = [
        f"{check['name']}: {check['status']} ({check.get('protocol_version') or 'no RPC protocol'}, {check.get('device_count') or 0} device(s))"
        for check in checks
    ]
    return _emit(args, payload, lines)


def _cmd_plan(args) -> int:
    local_capacities, commit = _local_provenance()
    inventory = _store(args).load()
    selector = args.nodes or ("all" if not args.node else None)
    selected_names = set(args.node)
    if selector == "all":
        selected_names = {item.name for item in inventory}
    elif selector:
        selected_names.update(selector.split(","))
    selected = [item for item in inventory if item.name in selected_names]
    with _with_tunnels(selected, args.timeout) as replacement:

        class _MemoryNodeStore:
            def load(self):
                return tuple(replacement)

        plan = nodes.resolve_node_plan(
            names=args.node,
            selector=selector,
            allow_insecure_rpc=bool(args.allow_insecure_rpc),
            local_free_vram_mib=local_capacities,
            local_engine_commit=commit,
            expected_engine={"commit": commit},
            timeout=args.timeout,
            store=_MemoryNodeStore(),
        )
    plan["requested"] = True
    if args.model:
        info = model_source.detect_model(args.model)
        if info is None:
            raise InputError(f"could not resolve model or GGUF path: {args.model}")
        plan = runtime._annotate_node_model_fit(plan, info)
        plan["model"] = args.model
    payload = dict(runtime._node_plan_payload(plan), experimental=True)
    lines = [
        f"RPC endpoints: {', '.join(plan['rpc_endpoints'])}",
        f"Device order: {', '.join(plan['device_order'])}",
        f"Tensor split: {plan['tensor_split']}",
        f"Live accelerator capacity: {plan['total_capacity_mib']:.0f} MiB",
        "Fit result is weights-only; llama.cpp owns final tensor, KV-cache, and RAM placement.",
    ]
    return _emit(args, payload, lines)


def cmd_nodes(args) -> int:
    """Dispatch ``kestrel nodes`` without ever starting a worker implicitly."""

    command = getattr(args, "nodes_command", None)
    if command == "list":
        return _cmd_list(args)
    if command == "add":
        return _cmd_add(args)
    if command == "remove":
        return _cmd_remove(args)
    if command == "doctor":
        return _cmd_doctor(args)
    if command == "plan":
        return _cmd_plan(args)
    raise InputError("choose a nodes command: add, list, remove, doctor, or plan")


__all__ = ["cmd_nodes"]
