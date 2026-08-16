"""Runtime glue: backend construction and the process/stream plumbing.

This is the module under test for ``_configure_backend``/``_resolve_ollama_native``
and every helper whose contract is patching ``cli.runtime``.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
import time
from contextlib import ExitStack
from pathlib import Path
from typing import TextIO
from urllib.parse import urlsplit

from .. import nodes, ui
from ..backends.llama_cpp import LlamaCppBackend
from ..errors import BackendError, InputError
from . import probes, state


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


def _endpoint_is_loopback(endpoint: str) -> bool:
    """Return whether an RPC endpoint is bound to the local host.

    llama.cpp RPC has no authentication or encryption layer. Kestrel therefore
    permits registry entries on loopback by default and requires the explicit
    ``--allow-insecure-rpc`` acknowledgement for LAN/WAN endpoints.
    """
    value = str(endpoint).strip()
    if "://" not in value:
        value = "//" + value
    try:
        host = (urlsplit(value).hostname or "").lower().strip("[]")
    except ValueError:
        return False
    return host in {"localhost", "127.0.0.1", "::1"}


def _node_plan_payload(plan: dict | None) -> dict:
    """Return a stable, JSON-safe summary of a resolved node plan."""
    plan = dict(plan or {})
    nodes = []
    for node in plan.get("nodes") or []:
        if isinstance(node, str):
            nodes.append({"name": node})
        elif isinstance(node, dict):
            nodes.append(
                {
                    key: node[key]
                    for key in ("name", "rpc_endpoint", "endpoint", "status", "role", "transport", "vram_free_mb")
                    if key in node and isinstance(node[key], (str, int, float, bool, type(None)))
                }
            )
    endpoints = [str(item) for item in (plan.get("rpc_endpoints") or []) if str(item).strip()]
    return {
        "status": plan.get("status", "local"),
        "requested": bool(plan.get("requested", endpoints or nodes)),
        "nodes": nodes,
        "rpc_endpoints": endpoints,
        "tensor_split": plan.get("tensor_split"),
        "total_devices": plan.get("total_devices"),
        "device_order": list(plan.get("device_order") or []),
        "capacities_mib": list(plan.get("capacities_mib") or []),
        "total_capacity_mib": plan.get("total_capacity_mib"),
        "model_size_mib": plan.get("model_size_mib"),
        "coarse_accelerator_fit": plan.get("coarse_accelerator_fit"),
        "fit_scope": plan.get("fit_scope"),
        "probe_evidence": dict(plan.get("probe_evidence") or {}),
    }


def _local_node_inputs(args) -> tuple[list[int], str]:
    """Return ordered local device capacity and pinned engine provenance.

    llama.cpp applies ``--tensor-split`` to its enumerated local devices first
    and RPC devices afterward. Preserve the probe order exactly and require a
    git commit for the selected coordinator so remote workers can be matched
    before any model data is sent.
    """

    gpu = getattr(args, "_gpu", None) or {}
    devices = gpu.get("devices") or []
    capacities = [max(0, int(device.get("vram_free_mb") or 0)) for device in devices]
    if not capacities and gpu.get("vram_free_mb") is not None:
        capacities = [max(0, int(gpu.get("vram_free_mb") or 0))]
    from .. import engine

    manifest = engine.load_manifest(state.LLAMA_CPP_DIR)
    commit = (manifest.commit if manifest else None) or engine.git_head(state.LLAMA_CPP_DIR)
    if not commit:
        raise InputError(
            "selected llama.cpp engine has no verifiable git provenance",
            hint="adopt or build the engine with `kestrel engine set --dir PATH` before using nodes",
        )
    return capacities, commit


def _resolve_node_plan(args) -> dict:
    """Resolve selected node names through the optional node-inventory core.

    The node core is deliberately imported only when a node was requested, so
    ordinary single-host launches remain usable in installations that do not
    ship the experimental inventory module yet. The core contract is
    ``resolve_node_plan(names, selector, allow_insecure_rpc)`` returning a dict
    with ``nodes``, ``rpc_endpoints`` and (optionally) ``tensor_split``.
    """
    cached = getattr(args, "_node_plan", None)
    if cached is not None:
        return cached
    names = [str(item).strip() for item in (getattr(args, "node", None) or []) if str(item).strip()]
    selector = getattr(args, "nodes", None)
    selected_names = names + ([item for item in selector.split(",") if item] if selector and selector != "all" else [])
    if len(set(selected_names)) != len(selected_names):
        raise InputError("node names must not be selected more than once")
    if not names and not selector:
        return {
            "status": "local",
            "requested": False,
            "nodes": [],
            "rpc_endpoints": [],
            "tensor_split": None,
        }
    if selector == "all" and names:
        raise InputError("--nodes all cannot be combined with --node NAME")
    try:
        from .. import nodes
    except ImportError as exc:
        raise InputError(
            "node selection is unavailable because the node inventory is not installed",
            hint="remove --node/--nodes or install the Kestrel node inventory component",
        ) from exc
    allow_insecure = bool(getattr(args, "allow_insecure_rpc", False))
    resolver = getattr(nodes, "resolve_node_plan", None)
    if resolver is None:
        raise InputError(
            "installed node inventory cannot perform an RPC protocol preflight",
            hint="update Kestrel before using distributed nodes",
        )
    # Open pinned SSH forwards before the protocol resolver probes them.  The
    # resolver receives an in-memory inventory whose managed endpoints are the
    # newly allocated loopback ports; llama.cpp sees only those ports and no
    # SSH credentials. The ExitStack is closed by run/serve in every path.
    inventory_store = nodes.NodeStore(allow_insecure_direct_rpc=allow_insecure)
    inventory = inventory_store.load()
    selected_names = set(names)
    if selector == "all":
        selected_names = {item.name for item in inventory}
    elif selector:
        selected_names.update(item for item in selector.split(",") if item)
    stack = ExitStack()
    endpoint_overrides: dict[str, str] = {}
    active_tunnels: dict[str, nodes.SshTunnel] = {}
    managed_names: set[str] = set()
    try:
        # Managed forwards start concurrently so N nodes cost max(T_i) instead
        # of sum(T_i); each SshTunnel is self-contained in its own thread.
        to_start = [item for item in inventory if item.name in selected_names and item.managed and item.enabled]
        if to_start:
            started = nodes.start_managed_tunnels(to_start)
            for item, tunnel in zip(to_start, started, strict=True):
                stack.callback(tunnel.close)
                endpoint_overrides[item.name] = tunnel.endpoint
                active_tunnels[item.name] = tunnel
                managed_names.add(item.name)
        args._ssh_tunnel_stack = stack
    except BaseException:
        stack.close()
        raise

    try:
        local_capacities, local_commit = _local_node_inputs(args)
        raw = resolver(
            names=names,
            selector=selector,
            allow_insecure_rpc=allow_insecure,
            local_free_vram_mib=local_capacities,
            local_engine_commit=local_commit,
            expected_engine={"commit": local_commit},
            store=inventory_store,
            endpoint_overrides=endpoint_overrides,
        )
    except BaseException:
        _close_node_tunnels(args)
        raise
    dead = sorted(
        name for name, tunnel in active_tunnels.items() if tunnel.process is None or tunnel.process.poll() is not None
    )
    if dead:
        _close_node_tunnels(args)
        raise InputError(
            f"managed SSH tunnel exited during node preflight: {', '.join(dead)}",
            hint="inspect SSH authentication and the pinned worker host key",
        )
    if not isinstance(raw, dict):
        _close_node_tunnels(args)
        raise InputError("node inventory returned an invalid placement plan")
    plan = dict(raw)
    plan["requested"] = True
    entries = plan.get("nodes") or []
    endpoints = [str(item).strip() for item in (plan.get("rpc_endpoints") or []) if str(item).strip()]
    if not endpoints:
        for entry in entries:
            if isinstance(entry, dict):
                endpoint = entry.get("rpc_endpoint") or entry.get("endpoint") or entry.get("rpc")
                if endpoint:
                    endpoints.append(str(endpoint).strip())
    if not endpoints:
        _close_node_tunnels(args)
        raise InputError("selected nodes have no usable llama.cpp RPC endpoints")
    if not getattr(args, "allow_insecure_rpc", False):
        unsafe = [endpoint for endpoint in endpoints if not _endpoint_is_loopback(endpoint)]
        if unsafe:
            _close_node_tunnels(args)
            raise InputError(
                "selected RPC nodes are not loopback endpoints",
                hint="verify the nodes are protected by an authenticated tunnel, then pass --allow-insecure-rpc explicitly",
            )
    plan["rpc_endpoints"] = endpoints
    for entry in plan.get("nodes") or []:
        if isinstance(entry, dict) and entry.get("name") in managed_names:
            entry["transport"] = "managed_ssh"
    plan.setdefault("status", "planned")
    return plan


def _close_node_tunnels(args) -> None:
    """Close all managed SSH forwards associated with a launch namespace."""

    stack = getattr(args, "_ssh_tunnel_stack", None)
    args._ssh_tunnel_stack = None
    if stack is not None:
        stack.close()


def _annotate_node_model_fit(plan: dict | None, model_info: dict) -> dict:
    """Add a deliberately coarse accelerator-capacity comparison to a plan.

    This is not a promise that the model fits: llama.cpp owns tensor, KV-cache,
    and host-RAM placement. It is still valuable to expose whether the model's
    bytes alone fit inside the live accelerator memory used for tensor split.
    """

    plan = dict(plan or {})
    if not plan.get("requested"):
        return plan
    size = model_info.get("file_size_bytes")
    if not size:
        try:
            size = os.path.getsize(model_info["path"])
        except (KeyError, OSError, TypeError):
            size = None
    capacity = plan.get("total_capacity_mib")
    model_mib = round(size / 1024**2, 3) if isinstance(size, (int, float)) and size >= 0 else None
    plan.update(
        {
            "model_size_mib": model_mib,
            "coarse_accelerator_fit": bool(model_mib is not None and capacity is not None and model_mib <= capacity),
            "fit_scope": "weights-only live accelerator comparison; llama.cpp owns final tensor, KV-cache, and RAM placement",
        }
    )
    return plan


def _configure_backend(model_info: dict, config: dict, args=None) -> LlamaCppBackend:
    """Build the LlamaCppBackend from a planned runtime config.

    Shared by the interactive and server launch builders so backend
    construction and engine-tune arguments have a single source of truth.
    """
    use_mtp = config["use_mtp"] and not (args and args.no_mtp)
    node_plan = _resolve_node_plan(args) if args is not None else {"rpc_endpoints": [], "tensor_split": None}
    explicit_split = getattr(args, "tensor_split", None) if args else None
    planned_split = node_plan.get("tensor_split")
    if explicit_split and node_plan.get("requested"):
        actual = len(explicit_split.split(","))
        expected = node_plan.get("total_devices")
        if expected is not None and actual != expected:
            raise InputError(
                f"--tensor-split has {actual} entries but local plus RPC enumeration has {expected} devices",
                hint="omit --tensor-split to use live per-device free-memory ratios",
            )
    kv_cache_type = getattr(args, "kv_cache_type", "auto") if args else "auto"
    cache_type_k = config["cache_type_k"] if kv_cache_type in (None, "auto") else kv_cache_type
    cache_type_v = config["cache_type_v"] if kv_cache_type in (None, "auto") else kv_cache_type
    return LlamaCppBackend(
        model_path=model_info["path"],
        n_gpu_layers=config["gpu_layers"],
        n_ctx=config["context_size"],
        n_batch=(args.batch_size if args and args.batch_size else config["batch_size"]),
        n_ubatch=(args.ubatch_size if args and args.ubatch_size else config["ubatch_size"]),
        spec_type="mtp" if use_mtp else "none",
        spec_draft_n=args.mtp_tokens if args else 3,
        cpu_moe=config["cpu_moe"],
        n_cpu_moe_layers=config.get("n_cpu_moe_layers"),
        fit=config["fit"],
        fit_target_mib=(args.fit_target if args and args.fit_target else config["fit_target_mib"]),
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
        turbo_kv=(getattr(args, "turbo_kv", None) if args else None) or config.get("kv_cache_turbo", False),
        use_mmap=not (args and args.no_mmap),
        use_mlock=bool(args and args.mlock),
        direct_io=bool(args and args.direct_io),
        tensor_split=(
            explicit_split
            or planned_split
            or _tensor_split_arg(
                (args and getattr(args, "_gpu", None)) or probes.detect_gpu(),
                None,
            )
        ),
        rpc_endpoints=node_plan.get("rpc_endpoints") or [],
        extra_args=_flatten_extra(args.extra if args else None),
        reasoning_level=(getattr(args, "reasoning", "auto") if args else "auto"),
        n_threads=(args.threads if args and args.threads is not None else config["threads"]),
        threads_batch=(
            args.threads_batch
            if args and getattr(args, "threads_batch", None) is not None
            else (config.get("threads_batch") or 0)
        ),
        cache_reuse=(getattr(args, "cache_reuse", None) if args else None) or 0,
        ctx_checkpoints=(getattr(args, "ctx_checkpoints", None) if args else None) or 0,
        llama_cpp_dir=state.LLAMA_CPP_DIR,
        moe_cache=(
            str(config["moe_cache_budget_mib"])
            if config["moe_cache"] == "on" and config["moe_cache_budget_mib"]
            else config["moe_cache"]
        ),
    )


_OOM_MARKERS = (
    "out of memory",
    "cuda_error_out_of_memory",
    "failed to allocate",
    "allocation failed",
)


def _is_startup_oom(stderr: str, elapsed_seconds: float) -> bool:
    """Return whether a quick process failure is specifically memory-related.

    A generic ``CUDA error`` is not sufficient: invalid arguments, unsupported
    kernels, and driver failures should be surfaced immediately instead of
    being retried with unrelated memory flags.
    """
    return elapsed_seconds < 180 and any(marker in stderr.lower() for marker in _OOM_MARKERS)


def _lower_memory_command(command: list[str]) -> tuple[list[str], list[str]] | None:
    """Build the next lower-memory command, or ``None`` if nothing can change.

    Keeping this mutation pure makes the retry ladder independently testable
    and prevents repeating an identical failing launch when neither supported
    memory-control flag is present.
    """
    lowered = list(command)
    changes: list[str] = []

    def increase_or_reduce(flag: str, transform) -> int | None:
        try:
            # Extra llama.cpp arguments are appended after Kestrel's managed
            # flags. Its parser applies the final occurrence, so changing the
            # first duplicate would leave the effective launch untouched.
            index = len(lowered) - 1 - lowered[::-1].index(flag)
        except ValueError:
            return None
        if index + 1 >= len(lowered):
            return None
        try:
            value = transform(int(lowered[index + 1]))
        except ValueError:
            return None
        if str(value) == lowered[index + 1]:
            return None
        lowered[index + 1] = str(value)
        return value

    ubatch = increase_or_reduce("-ub", lambda value: max(16, value // 2))
    if ubatch is not None:
        changes.append(f"micro-batch {ubatch}")
    try:
        ngl_index = len(lowered) - 1 - lowered[::-1].index("-ngl")
        cpu_moe_index = len(lowered) - 1 - lowered[::-1].index("--n-cpu-moe")
        max_cpu_layers = max(0, int(lowered[ngl_index + 1]) - 1)
        current_cpu_layers = int(lowered[cpu_moe_index + 1])
    except (ValueError, IndexError):
        pass
    else:
        if current_cpu_layers < max_cpu_layers:
            step = max(1, (max_cpu_layers - current_cpu_layers + 1) // 2)
            safer_cpu_layers = min(max_cpu_layers, current_cpu_layers + step)
            lowered[cpu_moe_index + 1] = str(safer_cpu_layers)
            changes.append(f"CPU experts through layer {safer_cpu_layers}")
    fit_target = increase_or_reduce("--fit-target", lambda value: value + 512)
    if fit_target is not None:
        changes.append("a larger VRAM margin")
    return (lowered, changes) if changes else None


def _run_with_oom_retries(
    cmd: list[str],
    max_retries: int = 2,
    env: dict[str, str] | None = None,
    stdout: TextIO | None = None,
) -> int:
    """Run an interactive llama.cpp process and retry startup CUDA OOMs.

    Output and input stay attached to the terminal. Only stderr is mirrored so
    Kestrel can identify a CUDA allocation failure.
    """
    current = list(cmd)
    for attempt in range(max_retries + 1):
        tail: list[str] = []
        started = time.monotonic()
        try:
            process = subprocess.Popen(
                current,
                stdin=None,
                stdout=stdout,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
            )
        except OSError as exc:
            raise BackendError(f"could not start llama.cpp: {exc}") from exc

        def mirror_stderr(process=process, tail=tail):
            stderr = process.stderr
            if stderr is None:
                return
            for line in stderr:
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
            # The parent received Ctrl-C. Normalize the shell-visible result
            # instead of leaking the child's negative SIGTERM/SIGINT code
            # through sys.exit (which wraps it to an unrelated 8-bit status).
            return 130
        if returncode == 0:
            return 0

        error_text = "".join(tail)
        if not _is_startup_oom(error_text, time.monotonic() - started) or attempt >= max_retries:
            if returncode != 0:
                print(
                    f"\nKestrel: llama.cpp exited with status {returncode} (stderr shown above).",
                    file=sys.stderr,
                )
                from ..engine import matches_too_old_signature

                if matches_too_old_signature(error_text.lower()):
                    print(
                        "  Hint: this model may need a newer llama.cpp engine. "
                        "Run `kestrel engine status`, then `kestrel engine update`.",
                        file=sys.stderr,
                    )
            return returncode

        retry = _lower_memory_command(current)
        if retry is None:
            print(
                "\nKestrel: startup ran out of memory, but this command has no adjustable "
                "micro-batch or fit margin; not repeating the same launch.",
                file=sys.stderr,
            )
            return returncode
        current, changes = retry
        print(
            "\nKestrel: CUDA OOM during startup; retrying with " + " and ".join(changes) + ".",
            file=sys.stderr,
        )
        print("  " + shlex.join(current), file=sys.stderr)
    return 1


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
        print(json.dumps({"error": exc.as_dict()}, default=str))
    else:
        print(f"Error: {exc.message}", file=sys.stderr)
        if getattr(exc, "hint", None):
            print(f"  Hint: {exc.hint}", file=sys.stderr)
    return getattr(exc, "exit_code", 1)


def _session_root() -> Path:
    """Directory for persisted per-model run sessions."""
    base = os.environ.get("XDG_STATE_HOME")
    return Path(base if base else os.path.join(Path.home(), ".local", "state")) / "kestrel" / "sessions"


def _session_slug(model_key: str) -> str:
    """Stable per-model session subdirectory from the user's model string."""
    import re as _re

    return _re.sub(r"[^A-Za-z0-9._-]+", "_", Path(model_key).stem or "model")


def _session_cache_path(model_key: str, name: str) -> Path:
    return _session_root() / _session_slug(model_key) / f"{name}.gguf.cache"


def _session_transcript_path(model_key: str, name: str) -> Path:
    return _session_root() / _session_slug(model_key) / f"{name}.txt"


def _with_session_flags(cmd: list[str], name: str, model_key: str, *, interactive: bool) -> list[str]:
    """Add llama-cli warm-start flags for a named session.

    ``--prompt-cache`` writes the KV cache back on exit; ``--prompt-cache-ro``
    (only when a cache already exists) warm-starts from the previous run. An
    interactive resume also feeds the saved transcript with ``-f`` so the
    conversation continues where it left off. All flags are appended at the
    end, where llama-cli accepts them anywhere in the command line.
    """
    cache = _session_cache_path(model_key, name)
    transcript = _session_transcript_path(model_key, name)
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        cmd += ["--prompt-cache-ro", str(cache)]
    cmd += ["--prompt-cache", str(cache)]
    if interactive and transcript.exists():
        cmd += ["-f", str(transcript)]
    return cmd


def _append_session_transcript(model_key: str, name: str, prompt: str, output: str) -> None:
    """Persist one Q/A exchange so a later ``--session`` run can resume it."""
    path = _session_transcript_path(model_key, name)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"User: {prompt}\nKestrel: {output}\n")
    except OSError:
        pass


def _oneshot_run(backend, cmd: list[str], args) -> int:
    """Run a one-shot generation and return its exit code.

    The completion text is always shown on the human stream; under ``--json``
    a structured result (output + token counts + timings + command) is the
    only thing written to stdout.
    """
    oneshot = backend._build_cmd(args.prompt, int(args.max_tokens))
    session_name = getattr(args, "session", None)
    if session_name and backend.capabilities().supports("--prompt-cache"):
        oneshot = _with_session_flags(oneshot, session_name, args.model, interactive=False)
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
                "nodes": _node_plan_payload(getattr(args, "_node_plan", None)),
            },
        )
    if session_name:
        _append_session_transcript(args.model, session_name, args.prompt, output)
    metrics = backend.last_metrics
    print(output, file=_human_stream(args))
    if not getattr(args, "json", False) and metrics.output_tokens_per_second:
        print(
            ui.dim(
                f"… {metrics.output_tokens_per_second:.1f} tok/s "
                f"({metrics.output_tokens} tokens, {metrics.elapsed_seconds:.1f}s)"
            ),
            file=sys.stderr,
        )
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
            "nodes": _node_plan_payload(getattr(args, "_node_plan", None)),
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

    probe_host = {"0.0.0.0": "127.0.0.1", "::": "::1", "": "127.0.0.1"}.get(host, host)
    if ":" in probe_host and not probe_host.startswith("["):
        probe_host = f"[{probe_host}]"
    deadline = time.perf_counter() + timeout
    while True:
        for path in ("/health", "/v1/health"):
            url = f"http://{probe_host}:{port}{path}"
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
        from ..model_store import resolve_ollama_blob

        blob = resolve_ollama_blob(name)
    except Exception:
        return None
    return str(blob) if blob is not None else None


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
        api_key_file=getattr(args, "api_key_file", None),
        parallel=getattr(args, "parallel", None),
    )
