"""Runtime glue: backend construction and the process/stream plumbing.

This is the module under test for ``_configure_backend``/``_resolve_ollama_native``
and every helper whose contract is patching ``cli.runtime``.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import threading
import time
from typing import TextIO

from ..backends.llama_cpp import LlamaCppBackend
from ..errors import BackendError
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
        fit_target_mib=(args.fit_target if args and args.fit_target else config["fit_target_mib"]),
        cache_type_k=args.kv_cache_type if args else config["cache_type_k"],
        cache_type_v=args.kv_cache_type if args else config["cache_type_v"],
        use_mmap=not (args and args.no_mmap),
        use_mlock=bool(args and args.mlock),
        direct_io=bool(args and args.direct_io),
        tensor_split=_tensor_split_arg(
            (args and getattr(args, "_gpu", None)) or probes.detect_gpu(),
            getattr(args, "tensor_split", None) or None,
        ),
        extra_args=_flatten_extra(args.extra if args else None),
        n_threads=(args.threads if args and args.threads is not None else config["threads"]),
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
    )
