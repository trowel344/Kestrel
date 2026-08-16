"""Private, reusable llama-server lifecycle for coding-agent integrations.

The managed service is deliberately loopback-only and authenticated. State
contains a random command-line ownership token so stop never signals an
unrelated process after a stale PID is reused.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .errors import IntegrationError
from .util import write_atomic

_DEFAULT_OLLAMA_CONTEXT = 4096


def _state_root() -> Path:
    override = os.environ.get("KESTREL_AGENT_STATE_DIR")
    if override:
        return Path(override).expanduser()
    return Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "kestrel" / "agents"


def _config_root() -> Path:
    override = os.environ.get("KESTREL_AGENT_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "kestrel" / "agents"


def state_path() -> Path:
    return _state_root() / "server.json"


def log_path() -> Path:
    return _state_root() / "server.log"


def sunmap_path(project_dir: str | Path | None = None) -> Path:
    scope = Path(project_dir or Path.cwd()).expanduser().resolve()
    digest = hashlib.sha256(str(scope).encode()).hexdigest()[:16]
    return _state_root() / "sunmap" / f"{scope.name or 'project'}-{digest}.sqlite3"


def _require_sunmap() -> None:
    try:
        from sunmap import SunMap, TaskCheckpoint, TokenBudget, __version__  # noqa: F401
    except (ImportError, AttributeError) as exc:
        raise IntegrationError(
            "Sun Map memory was requested but sunmap>=0.2 is not installed",
            hint="install the companion project into Kestrel's environment and retry",
        ) from exc


def credential_path() -> Path:
    return _config_root() / "api-key"


def _lock_path() -> Path:
    return _state_root() / "lifecycle.lock"


def _private_parent(path: Path) -> None:
    if path.is_symlink():
        raise IntegrationError(f"refusing symlink integration directory: {path}")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not path.is_dir() or path.stat().st_uid != os.getuid():
        raise IntegrationError(f"integration directory is not a user-owned directory: {path}")
    try:
        path.chmod(0o700)
    except OSError as exc:
        raise IntegrationError(f"could not secure integration directory: {exc}") from exc


def ensure_credential() -> Path:
    """Return a private API-key file, creating one without exposing its value."""

    path = credential_path()
    _private_parent(path.parent)
    if path.is_symlink():
        raise IntegrationError(f"refusing symlink credential file: {path}")
    if path.exists():
        try:
            mode = path.stat().st_mode
        except OSError as exc:
            raise IntegrationError(f"could not inspect credential file: {exc}") from exc
        if not path.is_file():
            raise IntegrationError(f"credential path is not a regular file: {path}")
        if mode & 0o077:
            raise IntegrationError(
                f"credential file permissions are too broad: {path}",
                hint="run chmod 600 on the file and retry",
            )
        return path
    write_atomic(path, secrets.token_urlsafe(32) + "\n", backup=False)
    path.chmod(0o600)
    return path


def read_credential() -> str:
    path = ensure_credential()
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise IntegrationError(f"could not read integration credential: {exc}") from exc
    lines = raw.splitlines()
    if len(lines) != 1:
        raise IntegrationError(f"integration credential must contain exactly one key: {path}")
    value = lines[0]
    if (
        value.lstrip().startswith("#")
        or not 16 <= len(value) <= 512
        or value != value.strip()
        or any(not 0x21 <= ord(char) <= 0x7E for char in value)
    ):
        raise IntegrationError(f"integration credential is empty or invalid: {path}")
    return value


@contextmanager
def _lifecycle_lock():
    path = _lock_path()
    _private_parent(path.parent)
    if path.is_symlink():
        raise IntegrationError(f"refusing symlink lifecycle lock: {path}")
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
    except OSError as exc:
        raise IntegrationError(f"could not lock managed server lifecycle: {exc}") from exc


def _load_state() -> dict[str, Any] | None:
    path = state_path()
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise IntegrationError(f"managed server state is not a regular file: {path}")
    state_stat = path.stat()
    if state_stat.st_uid != os.getuid() or state_stat.st_mode & 0o077:
        raise IntegrationError(f"managed server state permissions are unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntegrationError(f"managed server state is unreadable: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema") != 1:
        raise IntegrationError("managed server state has an unsupported schema")
    return value


def _save_state(state: dict[str, Any]) -> None:
    path = state_path()
    _private_parent(path.parent)
    write_atomic(path, json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n", backup=False)
    path.chmod(0o600)


def _cmdline(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]


def _owned_alive(state: dict[str, Any]) -> bool:
    pid = state.get("pid")
    token = state.get("token")
    if not isinstance(pid, int) or pid <= 1 or not isinstance(token, str):
        return False
    command = _cmdline(pid)
    if command and "--managed-token" in command and token in command:
        return True
    return _owned_group_member(pid, token)


def _owned_group_member(pgid: int, token: str) -> bool:
    """Recover ownership when the proxy leader died but its backend survived."""

    if not sys.platform.startswith("linux"):
        return False
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return False
    for entry in entries:
        if not entry.name.isdigit():
            continue
        candidate = int(entry.name)
        try:
            if os.getpgid(candidate) != pgid:
                continue
        except OSError:
            continue
        command = _cmdline(candidate)
        if command and "--managed-token" in command and token in command:
            return True
    return False


def _request(path: str, *, port: int, token: str, payload: bytes | None = None, timeout: float = 2.0) -> int:
    headers = {"Authorization": f"Bearer {token}"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=payload,
        headers=headers,
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read(4096)
            return response.status
    except urllib.error.HTTPError as exc:
        exc.read(4096)
        return exc.code
    except (OSError, urllib.error.URLError):
        return 0


def _ready(port: int, token: str) -> bool:
    # llama-server intentionally leaves /health and /v1/models public.  A
    # malformed request to a protected generation route proves auth without
    # spending tokens: a matching key reaches validation (normally 400/422),
    # while a stale key is rejected with 401/403.
    protected = _request("/v1/chat/completions", port=port, token=token, payload=b"{}")
    rejected = _request("/v1/chat/completions", port=port, token="kestrel-deliberately-wrong", payload=b"{}")
    return (
        _request("/health", port=port, token=token) == 200
        and protected not in {0, 401, 403, 404, 405}
        and rejected in {401, 403}
    )


def _port_busy(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _ollama_context(model: str) -> int:
    try:
        result = subprocess.run(
            ["ollama", "show", model, "--modelfile"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise IntegrationError("could not inspect the selected Ollama model", hint=str(exc)) from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or "ollama show failed"
        raise IntegrationError(f"could not inspect Ollama model {model}: {detail}")
    configured = 4096
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0].upper() == "PARAMETER" and parts[1] == "num_ctx":
            try:
                configured = int(parts[2])
            except ValueError:
                raise IntegrationError(f"Ollama model {model} has an invalid num_ctx parameter") from None
    return configured


def _terminate_started_process(proc: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass


def server_status() -> dict[str, Any]:
    state = _load_state()
    if state is None:
        return {"status": "stopped", "running": False}
    owned = _owned_alive(state)
    token = read_credential() if owned else ""
    healthy = bool(owned and _ready(int(state["port"]), token))
    return {
        "status": "ready" if healthy else ("starting" if owned else "stale"),
        "running": owned,
        "healthy": healthy,
        "pid": state.get("pid"),
        "model": state.get("model"),
        "alias": state.get("alias"),
        "url": state.get("url"),
        "context": state.get("context"),
        "reasoning": state.get("reasoning"),
        "gpu_layers": state.get("gpu_layers", "auto"),
        "cpu_moe": state.get("cpu_moe", "auto"),
        "sunmap": state.get("sunmap"),
        "log": state.get("log"),
    }


def start_server(
    model: str,
    *,
    alias: str = "kestrel-local",
    port: int = 8080,
    context: str = "auto",
    reasoning: str = "auto",
    timeout: float = 180.0,
    sunmap_db: str | None = None,
    sunmap_tokens: int = 4096,
    sunmap_trace: str | None = None,
    gpu_layers: str = "auto",
    cpu_moe: str = "auto",
) -> dict[str, Any]:
    """Start a detached authenticated loopback server and wait for readiness."""

    if not sys.platform.startswith("linux"):
        raise IntegrationError(
            "managed coding-agent servers currently require Linux process ownership checks",
            hint="run `kestrel serve` manually on this platform",
        )
    with _lifecycle_lock():
        return _start_server_unlocked(
            model,
            alias=alias,
            port=port,
            context=context,
            reasoning=reasoning,
            timeout=timeout,
            sunmap_db=sunmap_db,
            sunmap_tokens=sunmap_tokens,
            sunmap_trace=sunmap_trace,
            gpu_layers=gpu_layers,
            cpu_moe=cpu_moe,
        )


def _start_server_unlocked(
    model: str,
    *,
    alias: str,
    port: int,
    context: str,
    reasoning: str,
    timeout: float,
    sunmap_db: str | None,
    sunmap_tokens: int,
    sunmap_trace: str | None,
    gpu_layers: str,
    cpu_moe: str,
) -> dict[str, Any]:
    if model.startswith("ollama://"):
        ollama_model = model.removeprefix("ollama://")
        if not ollama_model:
            raise IntegrationError("Ollama model name must not be empty")
        try:
            requested_context = int(context)
        except (TypeError, ValueError):
            requested_context = _DEFAULT_OLLAMA_CONTEXT
        configured_context = _ollama_context(ollama_model)
        if configured_context < requested_context:
            raise IntegrationError(
                f"Ollama model {ollama_model} is configured for {configured_context} tokens, "
                f"below Kestrel's advertised {requested_context}",
                hint=f"create an Ollama alias with `PARAMETER num_ctx {requested_context}` and use that alias",
            )
    workspace = str(Path.cwd().resolve()) if sunmap_db else None
    existing = _load_state()
    if existing and _owned_alive(existing):
        status = server_status()
        if status.get("healthy"):
            same = (
                existing.get("model") == model
                and existing.get("alias") == alias
                and existing.get("port") == port
                and existing.get("context") == str(context)
                and existing.get("reasoning") == reasoning
                and existing.get("gpu_layers") == gpu_layers
                and existing.get("cpu_moe") == cpu_moe
                and existing.get("sunmap")
                == (
                    {
                        "database": str(Path(sunmap_db).expanduser()),
                        "token_budget": sunmap_tokens,
                        "workspace": workspace,
                        "trace": str(Path(sunmap_trace).expanduser()) if sunmap_trace else None,
                    }
                    if sunmap_db
                    else None
                )
            )
            if same:
                status["reused"] = True
                return status
            raise IntegrationError(
                "a different Kestrel agent server is already running",
                hint="run kestrel agents stop before changing its model, port, context, or reasoning level",
            )
        raise IntegrationError(
            "the managed Kestrel server is still starting or unhealthy",
            hint="inspect kestrel agents logs",
        )
    if _port_busy(port):
        raise IntegrationError(f"port {port} is already owned by another process")

    credential = ensure_credential()
    api_token = read_credential()
    owner_token = uuid.uuid4().hex
    command = [
        sys.executable,
        "-m",
        "kestrel.agent_proxy",
        "--model",
        model,
        "--port",
        str(port),
        "--alias",
        alias,
        "--context",
        str(context),
        "--reasoning",
        reasoning,
        "--api-key-file",
        str(credential),
        "--managed-token",
        owner_token,
        "--timeout",
        str(timeout),
    ]
    if gpu_layers != "auto":
        command.extend(("--gpu-layers", gpu_layers))
    if cpu_moe != "auto":
        command.extend(("--cpu-moe", cpu_moe))
    if sunmap_db:
        if sunmap_tokens < 512:
            raise IntegrationError("Sun Map token budget must be at least 512")
        _require_sunmap()
        sunmap_db = str(Path(sunmap_db).expanduser())
        sunmap_trace = str(Path(sunmap_trace).expanduser()) if sunmap_trace else None
        command.extend(
            (
                "--sunmap-db",
                sunmap_db,
                "--sunmap-tokens",
                str(sunmap_tokens),
                "--workspace",
                workspace or str(Path.cwd().resolve()),
            )
        )
        if sunmap_trace:
            command.extend(("--sunmap-trace", sunmap_trace))
    log = log_path()
    _private_parent(log.parent)
    try:
        handle = log.open("ab", buffering=0)
        log.chmod(0o600)
        proc = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    except OSError as exc:
        raise IntegrationError(f"could not start managed Kestrel server: {exc}") from exc
    finally:
        if "handle" in locals():
            handle.close()

    state = {
        "schema": 1,
        "pid": proc.pid,
        "token": owner_token,
        "model": model,
        "alias": alias,
        "port": port,
        "url": f"http://127.0.0.1:{port}",
        "context": str(context),
        "reasoning": reasoning,
        "gpu_layers": gpu_layers,
        "cpu_moe": cpu_moe,
        "sunmap": (
            {
                "database": sunmap_db,
                "token_budget": sunmap_tokens,
                "workspace": workspace,
                "trace": sunmap_trace,
            }
            if sunmap_db
            else None
        ),
        "log": str(log),
        "started_at": int(time.time()),
    }
    try:
        _save_state(state)
    except BaseException:
        _terminate_started_process(proc)
        raise
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break
            if _ready(port, api_token):
                result = server_status()
                result["reused"] = False
                return result
            time.sleep(0.2)
    except BaseException:
        _terminate_started_process(proc)
        try:
            state_path().unlink()
        except OSError:
            pass
        raise
    _terminate_started_process(proc)
    try:
        state_path().unlink()
    except OSError:
        pass
    raise IntegrationError("managed Kestrel server did not become ready", hint=f"inspect {log}")


def stop_server() -> dict[str, Any]:
    if not sys.platform.startswith("linux"):
        raise IntegrationError("managed coding-agent server stop currently requires Linux")
    with _lifecycle_lock():
        return _stop_server_unlocked()


def _stop_server_unlocked() -> dict[str, Any]:
    state = _load_state()
    if state is None:
        return {"status": "stopped", "running": False, "already_stopped": True}
    if not _owned_alive(state):
        raise IntegrationError(
            "managed server state is stale; refusing to signal an unrelated PID",
            hint=f"remove the stale state file after inspection: {state_path()}",
        )
    pid = int(state["pid"])
    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError as exc:
        raise IntegrationError(f"could not stop managed server: {exc}") from exc
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline and _owned_alive(state):
        time.sleep(0.1)
    if _owned_alive(state):
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError as exc:
            raise IntegrationError(f"could not kill managed server: {exc}") from exc
    try:
        state_path().unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise IntegrationError(f"server stopped but state cleanup failed: {exc}") from exc
    return {"status": "stopped", "running": False, "already_stopped": False}


def probe_protocols() -> dict[str, bool]:
    """Probe live server routes without running an inference request."""

    status = server_status()
    if not status.get("healthy"):
        return {"chat_completions": False, "responses": False, "anthropic_messages": False}
    port = int(str(status["url"]).rsplit(":", 1)[1])
    token = read_credential()

    def present(path: str) -> bool:
        return _request(path, port=port, token=token, payload=b"{}") not in {0, 401, 403, 404, 405}

    return {
        "chat_completions": present("/v1/chat/completions"),
        "responses": present("/v1/responses"),
        "anthropic_messages": present("/v1/messages") and present("/v1/messages/count_tokens"),
    }


def tail_logs(lines: int = 80) -> str:
    path = log_path()
    if not path.exists():
        return ""
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise IntegrationError(f"could not read managed server log: {exc}") from exc
    return "\n".join(content[-max(1, min(lines, 500)) :])


__all__ = [
    "credential_path",
    "ensure_credential",
    "log_path",
    "probe_protocols",
    "read_credential",
    "server_status",
    "start_server",
    "state_path",
    "stop_server",
    "sunmap_path",
    "tail_logs",
]
