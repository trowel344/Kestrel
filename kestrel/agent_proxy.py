"""Bounded loopback proxy for coding-agent protocol adapters.

The proxy owns a private ``kestrel serve`` child and exposes only a loopback
listener to the selected coding agent.  It intentionally uses only the Python
standard library so the agent integration remains available in the runtime
installation.
"""

from __future__ import annotations

import argparse
import hmac
import http.client
import http.server
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path
from typing import Any

DEFAULT_MAX_BODY_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_CONCURRENCY = 8
DEFAULT_REQUEST_TIMEOUT = 3600.0
DEFAULT_MEMORY_CAPTURE_BYTES = 8 * 1024 * 1024
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)


def _text_from_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") in {"input_text", "text"}:
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def merge_responses_instructions(payload: dict[str, Any]) -> dict[str, Any]:
    """Fold leading Responses system/developer messages into ``instructions``.

    llama.cpp chat templates generally want exactly one leading system message.
    Codex's Responses API represents those messages as ``input`` items.  Only
    contiguous leading ``message`` items with ``system`` or ``developer``
    roles are folded; user, tool, function-call, and function-call-output
    items remain byte-for-byte structurally intact.
    """

    if not isinstance(payload, dict) or not isinstance(payload.get("input"), list):
        return payload
    items = payload["input"]
    consumed = 0
    prefix: list[str] = []
    for item in items:
        if (
            not isinstance(item, dict)
            or item.get("type") != "message"
            or item.get("role") not in {"system", "developer"}
        ):
            break
        prefix.append(_text_from_message_content(item.get("content")))
        consumed += 1
    if consumed == 0:
        return payload

    result = dict(payload)
    result["input"] = items[consumed:]
    existing = result.get("instructions")
    instruction_parts: list[str] = []
    if isinstance(existing, str) and existing:
        instruction_parts.append(existing)
    elif existing is not None:
        # Preserve an unusual non-string value instead of silently changing it.
        instruction_parts.append(json.dumps(existing, ensure_ascii=False, separators=(",", ":")))
    instruction_parts.extend(text for text in prefix if text)
    result["instructions"] = "\n\n".join(instruction_parts)
    return result


def _read_body(handler: http.server.BaseHTTPRequestHandler, max_bytes: int) -> bytes:
    raw_length = handler.headers.get("Content-Length")
    if raw_length is None:
        raise ValueError("request body requires Content-Length")
    try:
        length = int(raw_length)
    except ValueError as exc:
        raise ValueError("invalid Content-Length") from exc
    if length < 0 or length > max_bytes:
        raise OverflowError(f"request body exceeds {max_bytes} bytes")
    data = handler.rfile.read(length)
    if len(data) != length:
        raise ConnectionError("request body ended before Content-Length")
    return data


def _forward_headers(headers: http.client.HTTPMessage, body_length: int | None = None) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, value in headers.items():
        if name.lower() in _HOP_BY_HOP or name.lower() == "host":
            continue
        result[name] = value
    if body_length is not None:
        result["Content-Length"] = str(body_length)
    return result


def _forward_response(
    handler: http.server.BaseHTTPRequestHandler,
    response: http.client.HTTPResponse,
    *,
    capture_bytes: int = 0,
) -> tuple[bytes, str]:
    has_length = response.headers.get("Content-Length") is not None
    handler.send_response(response.status, response.reason)
    for name, value in response.headers.items():
        if name.lower() in _HOP_BY_HOP:
            continue
        handler.send_header(name, value)
    if not has_length:
        # urllib has already decoded upstream chunk framing.  Close-delimited
        # streaming keeps SSE/tool output flowing without buffering or lying
        # about a content length.
        handler.send_header("Connection", "close")
        handler.close_connection = True
    handler.end_headers()
    captured = bytearray()
    while True:
        chunk = response.read1(8 * 1024)
        if not chunk:
            break
        if len(captured) < capture_bytes:
            captured.extend(chunk[: capture_bytes - len(captured)])
        handler.wfile.write(chunk)
        handler.wfile.flush()
    return bytes(captured), response.headers.get("Content-Type", "")


class _ProxyHandler(http.server.BaseHTTPRequestHandler):
    server_version = "KestrelAgentProxy/1"
    protocol_version = "HTTP/1.1"

    @property
    def proxy_server(self) -> "AgentProxyServer":
        return self.server  # type: ignore[return-value]

    def log_message(self, _format: str, *args: Any) -> None:
        # Never print request headers, URLs, or upstream errors: they may carry
        # credentials or user source snippets.  The parent CLI owns status UI.
        return None

    def _authorized(self) -> bool:
        expected = self.proxy_server.api_token
        if expected is None:
            return True
        authorization = self.headers.get("Authorization", "")
        supplied = authorization[7:] if authorization.startswith("Bearer ") else self.headers.get("X-Api-Key", "")
        if supplied and hmac.compare_digest(supplied, expected):
            return True
        self.send_error(401, "Unauthorized")
        return False

    def do_GET(self) -> None:  # noqa: N802
        if not self._authorized():
            return
        if self.proxy_server.backend_model and self.path.split("?", 1)[0] not in {"/health", "/v1/models"}:
            self.send_error(404, "Route not supported by the managed Ollama backend")
            return
        if not self.proxy_server.request_slots.acquire(blocking=False):
            self.send_error(503, "Kestrel agent proxy is busy")
            return
        try:
            self._proxy(None)
        finally:
            self.proxy_server.request_slots.release()

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorized():
            return
        if self.proxy_server.backend_model and self.path.split("?", 1)[0] != "/v1/chat/completions":
            self.send_error(404, "Route not supported by the managed Ollama backend")
            return
        if not self.proxy_server.request_slots.acquire(blocking=False):
            self.send_error(503, "Kestrel agent proxy is busy")
            return
        try:
            try:
                body = _read_body(self, self.proxy_server.max_body_bytes)
            except OverflowError as exc:
                self.send_error(413, str(exc))
                return
            except (ValueError, ConnectionError) as exc:
                self.send_error(400, str(exc))
                return
            request_path = self.path.split("?", 1)[0]
            payload = None
            if request_path in {"/v1/responses", "/v1/chat/completions", "/v1/messages"}:
                try:
                    payload = json.loads(body)
                    if isinstance(payload, dict):
                        if self.proxy_server.backend_model and request_path in {
                            "/v1/responses",
                            "/v1/chat/completions",
                            "/v1/messages",
                        }:
                            payload = {**payload, "model": self.proxy_server.backend_model}
                            effort = {
                                "off": "none",
                                "low": "low",
                                "medium": "medium",
                                "high": "high",
                                "maximum": "max",
                            }.get(self.proxy_server.backend_reasoning)
                            if effort is not None:
                                # Ollama's OpenAI route controls thinking with
                                # reasoning_effort; native `think` is ignored.
                                payload["reasoning_effort"] = effort
                        if request_path == "/v1/responses":
                            payload = merge_responses_instructions(payload)
                        if self.proxy_server.memory is not None:
                            try:
                                payload = self.proxy_server.memory.enrich_request(request_path, payload)
                            except Exception:
                                # Memory is an optimization after startup. A
                                # transient SQLite or extraction failure must
                                # not destroy a long-running agent turn.
                                self.proxy_server.memory_errors += 1
                        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
                except (UnicodeDecodeError, json.JSONDecodeError):
                    pass
            self._proxy(body)
        finally:
            self.proxy_server.request_slots.release()

    def _proxy(self, body: bytes | None) -> None:
        path = self.path
        if self.proxy_server.backend_model and path.split("?", 1)[0] == "/health":
            path = "/api/tags"
        target = f"{self.proxy_server.backend_base_url}{path}"
        headers = _forward_headers(self.headers, len(body) if body is not None else None)
        if self.proxy_server.backend_model is not None:
            # The proxy authenticates the caller. Never disclose Kestrel's
            # private credential to an otherwise unauthenticated backend such
            # as the user's loopback Ollama daemon.
            headers.pop("Authorization", None)
            headers.pop("X-Api-Key", None)
        request = urllib.request.Request(target, data=body, headers=headers, method=self.command)
        try:
            with urllib.request.urlopen(request, timeout=self.proxy_server.request_timeout) as response:
                capture_limit = DEFAULT_MEMORY_CAPTURE_BYTES if self.proxy_server.memory is not None else 0
                captured, content_type = _forward_response(self, response, capture_bytes=capture_limit)
                if self.proxy_server.memory is not None and captured:
                    try:
                        self.proxy_server.memory.observe_response(self.path.split("?", 1)[0], content_type, captured)
                    except Exception:
                        self.proxy_server.memory_errors += 1
        except (BrokenPipeError, ConnectionResetError):
            # The coding agent may cancel a streamed request after receiving
            # enough output.  Its closed socket is not an upstream failure,
            # and attempting to send a second HTTP error only creates a noisy
            # handler traceback.
            self.close_connection = True
        except urllib.error.HTTPError as exc:
            try:
                _forward_response(self, exc)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
        except (urllib.error.URLError, TimeoutError, OSError):
            try:
                self.send_error(502, "Kestrel backend unavailable")
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True


class AgentProxyServer(http.server.ThreadingHTTPServer):
    """HTTP server carrying immutable proxy settings for each request."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        backend_port: int,
        *,
        max_body_bytes: int,
        request_timeout: float,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        memory: Any | None = None,
        api_token: str | None = None,
        backend_base_url: str | None = None,
        backend_model: str | None = None,
        backend_reasoning: str = "auto",
    ):
        super().__init__(address, _ProxyHandler)
        if max_concurrency <= 0:
            self.server_close()
            raise ValueError("max_concurrency must be positive")
        self.backend_port = backend_port
        self.backend_base_url = backend_base_url or f"http://127.0.0.1:{backend_port}"
        self.backend_model = backend_model
        self.backend_reasoning = backend_reasoning
        self.api_token = api_token
        self.max_body_bytes = max_body_bytes
        self.request_timeout = request_timeout
        self.request_slots = threading.BoundedSemaphore(max_concurrency)
        self.memory = memory
        self.memory_errors = 0


def _allocate_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _warm_ollama_model(model: str, *, timeout: float) -> None:
    """Load the selected Ollama model before advertising proxy readiness."""

    payload = json.dumps(
        {"model": model, "prompt": "", "stream": False, "keep_alive": "5m"},
        separators=(",", ":"),
    ).encode()
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Ollama could not load model {model!r}") from exc


def _llama_token_counter(base_url: str, api_token: str | None = None):
    """Return a cached exact counter backed by llama-server's tokenizer route."""

    cache: OrderedDict[str, int] = OrderedDict()

    def count(value: str) -> int:
        if not value:
            return 0
        cached = cache.get(value)
        if cached is not None:
            cache.move_to_end(value)
            return cached
        payload = json.dumps({"content": value}, separators=(",", ":")).encode()
        headers = {"Content-Type": "application/json"}
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"
        request = urllib.request.Request(
            f"{base_url}/tokenize",
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=5.0) as response:
                result = json.loads(response.read())
            tokens = result.get("tokens")
            if not isinstance(tokens, list):
                raise ValueError("tokenizer response omitted tokens")
            measured = len(tokens)
        except (OSError, ValueError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
            measured = max(1, len(value) // 4)
        cache[value] = measured
        cache.move_to_end(value)
        while len(cache) > 256:
            cache.popitem(last=False)
        return measured

    return count


def build_backend_command(
    model: str,
    backend_port: int,
    *,
    alias: str,
    context: str,
    reasoning: str,
    api_key_file: str | None,
    managed_token: str | None,
    timeout: float,
    gpu_layers: str = "auto",
    cpu_moe: str = "auto",
) -> list[str]:
    """Build shell-free child argv for a loopback ``kestrel serve``."""

    for value, label in ((model, "model"), (alias, "alias"), (context, "context"), (reasoning, "reasoning")):
        if not isinstance(value, str) or not value or any(ord(char) < 32 for char in value):
            raise ValueError(f"{label} must be a non-empty single-line value")
    for value, label in ((gpu_layers, "gpu_layers"), (cpu_moe, "cpu_moe")):
        if not isinstance(value, str) or value not in {"auto", "all", "on", "off"} and not value.isdigit():
            raise ValueError(f"{label} must be auto, all, on, off, or a layer count")
    if backend_port < 0 or backend_port > 65535:
        raise ValueError("backend_port must be between 0 and 65535")
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    command = [
        sys.executable,
        "-m",
        "kestrel",
        "serve",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(backend_port),
        "--alias",
        alias,
        "--ctx-size",
        context,
        "--reasoning",
        reasoning,
        "--parallel",
        "1",
        "--wait",
        str(timeout),
    ]
    if gpu_layers != "auto":
        command.extend(("--gpu-layers", gpu_layers))
    if cpu_moe != "auto":
        command.extend(("--cpu-moe", cpu_moe))
    if api_key_file:
        command.extend(("--api-key-file", api_key_file))
    if managed_token:
        command.extend(("--managed-token", managed_token))
    return command


def _terminate_child(child: subprocess.Popen[Any], *, timeout: float = 5.0) -> None:
    if child.poll() is not None:
        return
    shared_managed_group = os.getpgrp() == os.getpid()
    previous = None
    try:
        if shared_managed_group:
            previous = signal.signal(signal.SIGTERM, signal.SIG_IGN)
            os.killpg(os.getpgrp(), signal.SIGTERM)
        else:
            child.terminate()
        child.wait(timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        try:
            child.kill()
            child.wait(timeout=timeout)
        except (OSError, subprocess.TimeoutExpired):
            pass
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)


def run_proxy(
    model: str,
    *,
    alias: str = "kestrel-agent",
    public_port: int = 8080,
    context: str = "auto",
    reasoning: str = "auto",
    api_key_file: str | None = None,
    managed_token: str | None = None,
    timeout: float = 180.0,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    sunmap_db: str | None = None,
    sunmap_tokens: int = 4096,
    workspace: str | None = None,
    sunmap_trace: str | None = None,
    gpu_layers: str = "auto",
    cpu_moe: str = "auto",
) -> int:
    """Launch a managed child and serve until it exits or receives a signal."""

    backend_model = model.removeprefix("ollama://") if model.startswith("ollama://") else None
    if backend_model == "":
        raise ValueError("Ollama model name must not be empty")
    backend_port = 11434 if backend_model else _allocate_loopback_port()
    command = None
    if backend_model is None:
        command = build_backend_command(
            model,
            backend_port,
            alias=alias,
            context=context,
            reasoning=reasoning,
            api_key_file=api_key_file,
            managed_token=managed_token,
            timeout=timeout,
            gpu_layers=gpu_layers,
            cpu_moe=cpu_moe,
        )
    if public_port < 0 or public_port > 65535:
        raise ValueError("public_port must be between 0 and 65535")
    if timeout <= 0 or request_timeout <= 0 or max_body_bytes <= 0 or max_concurrency <= 0:
        raise ValueError("timeouts, max_body_bytes, and max_concurrency must be positive")
    api_token = None
    if api_key_file:
        try:
            api_token = Path(api_key_file).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ValueError(f"could not read API key file: {exc}") from exc
        if not api_token:
            raise ValueError("API key file is empty")
    if sunmap_trace and not sunmap_db:
        raise ValueError("Sun Map trajectory recording requires --sunmap-db")
    memory = None
    if sunmap_db:
        from .sunmap_memory import SunMapMemory  # noqa: E402

        memory = SunMapMemory(
            sunmap_db,
            token_budget=sunmap_tokens,
            workspace=workspace,
            trajectory_path=sunmap_trace,
            token_counter=(
                _llama_token_counter(f"http://127.0.0.1:{backend_port}", api_token) if backend_model is None else None
            ),
        )
    if backend_model is not None:
        _warm_ollama_model(backend_model, timeout=timeout)
    server = AgentProxyServer(
        ("127.0.0.1", public_port),
        backend_port,
        max_body_bytes=max_body_bytes,
        request_timeout=request_timeout,
        max_concurrency=max_concurrency,
        memory=memory,
        api_token=api_token,
        backend_base_url="http://127.0.0.1:11434" if backend_model else None,
        backend_model=backend_model,
        backend_reasoning=reasoning,
    )
    child = None
    if command is not None:
        try:
            child = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                close_fds=True,
                start_new_session=False,
                shell=False,
            )
        except BaseException:
            server.server_close()
            raise
    started = threading.Event()
    stop_requested = threading.Event()

    def watch_child() -> None:
        assert child is not None
        return_code = child.wait()
        if not stop_requested.is_set():
            started.wait(timeout=5.0)
            server.shutdown()
        server.child_returncode = return_code  # type: ignore[attr-defined]

    watcher = None
    if child is not None:
        watcher = threading.Thread(target=watch_child, name="kestrel-agent-child", daemon=True)
        watcher.start()
    previous: dict[int, Any] = {}

    def request_stop(signum: int, _frame: Any) -> None:
        del signum
        stop_requested.set()
        threading.Thread(target=server.shutdown, name="kestrel-agent-stop", daemon=True).start()

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, request_stop)
    try:
        started.set()
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        stop_requested.set()
    finally:
        stop_requested.set()
        server.server_close()
        if child is not None:
            _terminate_child(child)
        if watcher is not None:
            watcher.join(timeout=timeout)
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    child_returncode = child.returncode if child is not None else 0
    return int(getattr(server, "child_returncode", child_returncode or 0) or 0)


def _context_arg(value: str) -> str:
    if value == "auto":
        return value
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("context must be auto or a positive integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("context must be auto or a positive integer")
    return str(number)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Expose a managed Kestrel model to a coding agent")
    parser.add_argument("--model", required=True)
    parser.add_argument("--alias", default="kestrel-agent")
    parser.add_argument("--port", "--public-port", dest="public_port", type=int, default=8080)
    parser.add_argument("--context", "--ctx-size", dest="context", type=_context_arg, default="auto")
    parser.add_argument("--reasoning", choices=("auto", "off", "low", "medium", "high", "maximum"), default="auto")
    parser.add_argument("--gpu-layers", default="auto")
    parser.add_argument("--cpu-moe", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--api-key-file")
    parser.add_argument("--managed-token")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT)
    parser.add_argument("--sunmap-db", help="enable persistent Sun Map memory using this SQLite database")
    parser.add_argument("--sunmap-tokens", type=int, default=4096, help="maximum memory-context budget")
    parser.add_argument("--workspace", help="project root used to verify task-checkpoint file digests")
    parser.add_argument("--sunmap-trace", help="append a privacy-bounded replay trajectory as JSONL")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0 <= args.public_port <= 65535:
        raise SystemExit("public port must be between 0 and 65535")
    if args.timeout <= 0 or args.request_timeout <= 0:
        raise SystemExit("timeouts must be positive")
    if args.sunmap_tokens < 512:
        raise SystemExit("Sun Map token budget must be at least 512")
    return run_proxy(
        args.model,
        alias=args.alias,
        public_port=args.public_port,
        context=args.context,
        reasoning=args.reasoning,
        api_key_file=args.api_key_file,
        managed_token=args.managed_token,
        timeout=args.timeout,
        request_timeout=args.request_timeout,
        sunmap_db=args.sunmap_db,
        sunmap_tokens=args.sunmap_tokens,
        workspace=args.workspace,
        sunmap_trace=args.sunmap_trace,
        gpu_layers=args.gpu_layers,
        cpu_moe=args.cpu_moe,
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AgentProxyServer",
    "DEFAULT_MAX_BODY_BYTES",
    "DEFAULT_MAX_CONCURRENCY",
    "DEFAULT_MEMORY_CAPTURE_BYTES",
    "DEFAULT_REQUEST_TIMEOUT",
    "_warm_ollama_model",
    "build_backend_command",
    "build_parser",
    "main",
    "merge_responses_instructions",
    "run_proxy",
]
