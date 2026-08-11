"""Health/setup commands: doctor, status, setup, and their shared probes.

Owns the ``setup`` persistence path, which refreshes the process-global config
snapshot through :func:`kestrel.cli.state.reload_state` so a same-process run
after ``kestrel setup`` sees the new settings.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .. import ui
from ..backends.llama_cpp import LlamaCppBackend, default_llama_cpp_dir
from ..config import REASONING_BUDGETS, KestrelConfig, config_path, load_config, save_config
from ..errors import ModelError
from ..model_store import default_models_dir
from ..util import available_disk_bytes
from . import model_source, probes, state

# byte thresholds for the disk-space sanity checks (fail below CRITICAL,
# warn below the softer bound; GGs are routinely many GiB).
_CHECK_DISK_CRITICAL = 1 << 30  # 1 GiB
_CHECK_DISK_WARN = 5 << 30  # 5 GiB


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
    fd = -1
    probe = None
    try:
        target = Path(path).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        fd, raw_probe = tempfile.mkstemp(dir=target, prefix=".kestrel-write-probe-")
        probe = Path(raw_probe)
        os.write(fd, b"ok")
        os.close(fd)
        fd = -1
        probe.unlink()
        return True, ""
    except OSError as exc:
        return False, str(exc)
    finally:
        if fd >= 0:
            os.close(fd)
        if probe is not None:
            try:
                probe.unlink(missing_ok=True)
            except OSError:
                pass


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
    models_dir = (
        Path(state.USER_CONFIG.models_dir).expanduser() if state.USER_CONFIG.models_dir else default_models_dir()
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
                "message": (f"llama-cli/llama-server unavailable: {cli_error or server_error}"),
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

    default = state.USER_CONFIG.default_model
    if not default:
        checks.append(
            {
                "name": "model cache integrity",
                "status": "warn",
                "message": "no default model configured",
            }
        )
    elif default.startswith("ollama://"):
        checks.append(
            {
                "name": "model cache integrity",
                "status": "ok",
                "message": f"configured Ollama model {default.removeprefix('ollama://')} is provider-managed",
            }
        )
    else:
        info = model_source.detect_model(default)
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
                                "message": (f"{gguf.name} {_fmt_bytes(size)}, GGUF header verified"),
                            }
                        )
                    else:
                        checks.append(
                            {
                                "name": "model cache integrity",
                                "status": "fail",
                                "message": (f"{gguf.name} header magic {magic!r} != GGUF"),
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
    gpu = probes.detect_gpu()
    memory = probes._memory_snapshot()
    power = probes._cpu_power_policy()
    backend = LlamaCppBackend("", llama_cpp_dir=state.LLAMA_CPP_DIR)
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

    checks = _doctor_checks(backend, cli_caps, cli_error, server_caps, server_error)
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
                "available_ram_mib": probes._available_ram_mib(),
                "cpu_power_policy": power,
            },
            "llama_cli": {
                "available": cli_caps is not None,
                "dir": backend.llama_cpp_dir,
                "binary": backend.binary if cli_caps is not None else None,
                "native": (
                    bool(
                        cli_caps is not None
                        and str(Path(backend.binary)).startswith(str(Path(backend.llama_cpp_dir)) + os.sep)
                    )
                ),
                "version": cli_caps.version if cli_caps is not None else None,
                "fit": cli_caps.supports("--fit") if cli_caps is not None else None,
                "cpu_moe": cli_caps.supports("--cpu-moe") if cli_caps is not None else None,
                "mmap": cli_caps.supports("--mmap") if cli_caps is not None else None,
                "direct_io": cli_caps.supports("--direct-io") if cli_caps is not None else None,
                "quantized_kv": (cli_caps.supports("--cache-type-k") if cli_caps is not None else None),
                "mtp": "draft-mtp" in cli_caps.spec_types if cli_caps is not None else None,
                "error": cli_error,
            },
            "llama_server": {
                "available": server_caps is not None,
                "dir": backend.llama_cpp_dir,
                "binary": backend.server_binary if server_caps is not None else None,
                "native": (
                    bool(
                        server_caps is not None
                        and str(Path(backend.server_binary)).startswith(str(Path(backend.llama_cpp_dir)) + os.sep)
                    )
                ),
                "version": server_caps.version if server_caps is not None else None,
                "moe_cache": (server_caps.supports("--moe-cache") if server_caps is not None else None),
                "spec_types": sorted(server_caps.spec_types) if server_caps is not None else [],
                "error": server_error,
            },
        }
        print(json.dumps(payload))
        if not (cli_caps is not None or server_caps is not None):
            return 1
        if checks_failed:
            return 1
        return 0

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
        ui.kv("Available RAM", f"{probes._available_ram_mib()} MiB", value_color=ui.cyan),
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
        host_lines.append(f"  {ui.warn_mark()} swap is over 50% used; cold large-model benchmarks may be distorted")
    print(ui.box("Host", "\n".join(host_lines)))

    cli_lines = [ui.kv("llama.cpp dir", backend.llama_cpp_dir)]
    cli_available = cli_caps is not None
    if cli_caps is not None:
        cli_native = str(Path(backend.binary)).startswith(str(Path(backend.llama_cpp_dir)) + os.sep)
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
    print(
        ui.box(
            "llama-cli",
            "\n".join(cli_lines),
            title_color=ui.green if cli_available else ui.yellow,
        )
    )

    server_available = server_caps is not None
    if server_caps is not None:
        server_native = str(Path(backend.server_binary)).startswith(str(Path(backend.llama_cpp_dir)) + os.sep)
        server_label = "" if server_native else " (fallback from another build)"
        server_lines = [
            ui.kv("llama-server", f"{backend.server_binary}{server_label}", value_color=ui.bold),
            ui.kv("version", server_caps.version),
            f"  {mark(server_caps.supports('--moe-cache'))} MoE cache",
            f"  {mark(bool(server_caps.spec_types))} server spec types: {sorted(server_caps.spec_types) or 'none'}",
        ]
    else:
        server_lines = [f"  {ui.fail_mark()} llama-server: unavailable ({server_error})"]
    print(
        ui.box(
            "llama-server",
            "\n".join(server_lines),
            title_color=ui.green if server_available else ui.yellow,
        )
    )

    check_marks = {
        "ok": ui.pass_mark(),
        "warn": ui.warn_mark(),
        "fail": ui.fail_mark(),
    }
    check_lines = [
        f"  {check_marks.get(check['status'], ui.info_mark())} {check['name']}: {check['message']}" for check in checks
    ]
    print(
        ui.box(
            f"Checks ({overall_status})",
            "\n".join(check_lines),
            title_color=(ui.green if overall_status == "ok" else (ui.yellow if overall_status == "warn" else ui.red)),
        )
    )
    if checks_failed:
        return 1
    if not (cli_available or server_available):
        return 1
    return 0


def cmd_status(args):
    """Show the configured runtime and last measured optimization state."""

    configured = load_config()
    gpu = probes.detect_gpu()
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
            **probes._memory_snapshot(),
            "cpu_power_policy": probes._cpu_power_policy(),
        },
        "optimization_profile": str(profile_path) if profile else None,
        "profile_model": profile_model,
        "profile_matches_model": profile_matches_model,
        "plan": profile.get("plan") if profile else None,
        "benchmark": profile.get("benchmark") if profile else None,
        "profile_error": profile_error,
    }
    if args.json:
        print(json.dumps(payload))
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
            from ..model_store import ModelStoreError, resolve_ollama_blob

            try:
                # Local and cloud models are both valid through the Ollama
                # adapter; this call also proves the model exists.
                resolve_ollama_blob(model.removeprefix("ollama://"))
            except ModelStoreError as exc:
                raise ModelError(f"setup could not resolve model: {exc}") from exc
        else:
            detected = model_source.detect_model(model)
            if detected is None or (detected["type"] == "safetensors" and not detected["path"]):
                raise ModelError(f"setup could not resolve model: {model}")
            if detected["path"]:
                model = str(Path(detected["path"]).expanduser().resolve())
    elif model_source._resolve_model_alias("qwen3.5:122b-10a"):
        model = "qwen3.5:122b-10a"

    llama_dir = args.llama_cpp_dir or current.llama_cpp_dir or default_llama_cpp_dir()
    llama_dir = str(Path(llama_dir).expanduser().resolve())
    configured = KestrelConfig(
        default_model=model,
        models_dir=args.models_dir or current.models_dir,
        llama_cpp_dir=llama_dir,
        context_size=getattr(args, "context", None)
        if getattr(args, "context", None) is not None
        else current.context_size,
        reasoning_level=getattr(args, "reasoning", None) or current.reasoning_level,
    )
    target = save_config(configured)
    state.reload_state()
    print(
        ui.box(
            "Kestrel configuration saved",
            "\n".join(
                [
                    ui.kv("File", str(target), value_color=ui.cyan),
                    ui.kv(
                        "Local model",
                        configured.default_model or "not configured",
                        value_color=ui.green if configured.default_model else ui.dim,
                    ),
                    ui.kv("Models directory", configured.models_dir or "platform default"),
                    ui.kv("llama.cpp", configured.llama_cpp_dir),
                    ui.kv("Context", str(configured.context_size)),
                    ui.kv("Reasoning", configured.reasoning_level),
                ]
            ),
        )
    )


def cmd_settings(args):
    """Show or atomically update the defaults applied to model launches."""
    current = load_config()
    configured = KestrelConfig(
        default_model=current.default_model,
        models_dir=current.models_dir,
        llama_cpp_dir=current.llama_cpp_dir,
        context_size=args.context if args.context is not None else current.context_size,
        reasoning_level=args.reasoning or current.reasoning_level,
    )
    changed = configured != current
    target = config_path()
    if changed:
        target = save_config(configured)
        state.reload_state()
    payload = {
        "context_size": configured.context_size,
        "reasoning_level": configured.reasoning_level,
        "reasoning_budgets": REASONING_BUDGETS,
        "changed": changed,
        "file": str(target),
    }
    if args.json:
        print(json.dumps(payload))
        return
    print(
        ui.box(
            "Default model settings" + (" saved" if changed else ""),
            "\n".join(
                [
                    ui.kv("Context", str(configured.context_size), value_color=ui.cyan),
                    ui.kv("Reasoning", configured.reasoning_level, value_color=ui.cyan),
                    ui.kv("File", str(target)),
                    "",
                    ui.dim("CLI overrides: --ctx-size and --reasoning"),
                ]
            ),
        )
    )


def _save_default_model(model: str | Path) -> None:
    current = load_config()
    value = str(model) if str(model).startswith("ollama://") else str(Path(model).resolve())
    save_config(
        KestrelConfig(
            default_model=value,
            models_dir=current.models_dir,
            llama_cpp_dir=current.llama_cpp_dir,
            context_size=current.context_size,
            reasoning_level=current.reasoning_level,
        )
    )
    state.reload_state()
