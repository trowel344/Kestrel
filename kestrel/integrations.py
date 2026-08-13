"""Safe, reversible configuration for external coding agents.

This module deliberately only owns files in Kestrel's integration directory
(and the dedicated Codex profile).  It does not edit a user's existing agent
configuration.  The CLI can use :class:`LaunchMetadata` to start an agent
with the generated configuration without changing the user's shell.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from .errors import IntegrationError

SUPPORTED_CLIENTS = ("pi", "omp", "codex", "claude", "opencode")
_OWNER_PREFIX = "kestrel-integration"
_REASONING_TO_CODEX = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "maximum": "xhigh",
}


@dataclass(frozen=True)
class LaunchMetadata:
    """A client command plus the isolated environment it needs."""

    client: str
    command: tuple[str, ...]
    environment: dict[str, str]
    config_path: Path
    endpoint: str
    model: str
    context_size: int | None
    reasoning: str
    max_tokens: int = 8192


@dataclass(frozen=True)
class IntegrationStatus:
    """Current state of one Kestrel-owned integration file."""

    client: str
    configured: bool
    path: Path
    owned: bool
    details: dict[str, str]


@dataclass(frozen=True)
class IntegrationSpec:
    """Metadata for registering a running Kestrel server with Pi/OMP."""

    model_id: str
    base_url: str = "http://127.0.0.1:8080/v1"
    context_window: int | None = None
    max_tokens: int = 8192
    reasoning: bool = True

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[^\x00-\x20\x7f]+", self.model_id):
            raise IntegrationError("model_id must be non-empty and contain no whitespace or control characters")
        _validate_endpoint(self.base_url)
        if self.context_window is not None and (type(self.context_window) is not int or self.context_window < 512):
            raise IntegrationError("context_window must be at least 512 tokens")
        if type(self.max_tokens) is not int or self.max_tokens < 1:
            raise IntegrationError("max_tokens must be a positive integer")
        if self.context_window is not None and self.max_tokens > self.context_window:
            raise IntegrationError("max_tokens must not exceed context_window")


@dataclass(frozen=True)
class ProviderIntegrationResult:
    """Result from a Pi/OMP provider-file operation."""

    client: str
    action: str
    path: Path
    changed: bool
    installed: bool
    dry_run: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "client": self.client,
            "action": self.action,
            "path": str(self.path),
            "changed": self.changed,
            "installed": self.installed,
            "dry_run": self.dry_run,
        }


def _home(home: str | os.PathLike[str] | None = None) -> Path:
    return Path(home).expanduser() if home is not None else Path(os.environ.get("HOME", "~")).expanduser()


def integration_dir(home: str | os.PathLike[str] | None = None) -> Path:
    """Return Kestrel's private integration directory."""

    return _home(home) / ".config" / "kestrel" / "integrations"


def _codex_home(home: Path, codex_home: str | os.PathLike[str] | None) -> Path:
    if codex_home is not None:
        return Path(codex_home).expanduser()
    return Path(os.environ.get("CODEX_HOME", home / ".codex")).expanduser()


def integration_path(
    client: str,
    *,
    home: str | os.PathLike[str] | None = None,
    codex_home: str | os.PathLike[str] | None = None,
) -> Path:
    """Return the dedicated path used for *client*.

    The Codex filename is intentionally ``kestrel.config.toml`` because that
    is the official profile convention under ``$CODEX_HOME``.
    """

    client = _validate_client(client)
    root = _home(home)
    if client == "pi":
        if home is None and os.environ.get("PI_CODING_AGENT_DIR"):
            return Path(os.environ["PI_CODING_AGENT_DIR"]).expanduser() / "models.json"
        return root / ".pi" / "agent" / "models.json"
    if client == "omp":
        if home is None and os.environ.get("PI_CODING_AGENT_DIR"):
            return Path(os.environ["PI_CODING_AGENT_DIR"]).expanduser() / "models.yml"
        return root / ".omp" / "agent" / "models.yml"
    if client == "codex":
        return _codex_home(root, codex_home) / "kestrel.config.toml"
    suffix = "claude-settings.json" if client == "claude" else "opencode.jsonc"
    return integration_dir(root) / suffix


def _validate_client(client: str) -> str:
    value = str(client).strip().lower()
    if value not in SUPPORTED_CLIENTS:
        raise IntegrationError(f"unsupported integration client: {client!r}")
    return value


def _validate_scalar(value: str, label: str) -> str:
    result = str(value).strip()
    if not result or any(ord(char) < 32 for char in result):
        raise IntegrationError(f"{label} must be a non-empty single-line value")
    return result


def _validate_endpoint(endpoint: str) -> str:
    result = _validate_scalar(endpoint, "endpoint").rstrip("/")
    parsed = urlsplit(result)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise IntegrationError("endpoint must be an http(s) URL with a host")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise IntegrationError("endpoint must not contain credentials, query parameters, or fragments")
    return result


def _v1_endpoint(endpoint: str) -> str:
    return endpoint if endpoint.endswith("/v1") else f"{endpoint}/v1"


def _anthropic_endpoint(endpoint: str) -> str:
    return endpoint[:-3] if endpoint.endswith("/v1") else endpoint


def _validate_reasoning(reasoning: str) -> str:
    value = _validate_scalar(reasoning, "reasoning").lower()
    if value not in {"auto", "off", "low", "medium", "high", "maximum"}:
        raise IntegrationError("reasoning must be auto, off, low, medium, high, or maximum")
    return value


def _validate_context(context_size: int | None) -> int | None:
    if context_size is None:
        return None
    try:
        value = int(context_size)
    except (TypeError, ValueError) as exc:
        raise IntegrationError("context_size must be a positive integer") from exc
    if value <= 0:
        raise IntegrationError("context_size must be a positive integer")
    return value


def _owner_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.kestrel-owner")


def _owner_text(client: str, path: Path, digest: str) -> str:
    return f"{_OWNER_PREFIX}\nclient={client}\npath={path}\nsha256={digest}\n"


def _read_owned(path: Path, client: str) -> bool:
    marker = _owner_path(path)
    if path.is_symlink() or marker.is_symlink() or not path.is_file() or not marker.is_file():
        return False
    try:
        lines = marker.read_text(encoding="utf-8").splitlines()
        values = dict(line.split("=", 1) for line in lines[1:] if "=" in line)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, UnicodeError, ValueError):
        return False
    return (
        lines[:1] == [_OWNER_PREFIX]
        and values.get("client") == client
        and values.get("path") == str(path)
        and values.get("sha256") == digest
    )


def _atomic_write(path: Path, content: str, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink():
        raise IntegrationError(f"refusing to replace symlink: {path}")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(tmp_name)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except BaseException:
        try:
            temp_path.unlink()
        except OSError:
            pass
        raise


def _write_owned(path: Path, client: str, content: str) -> Path:
    if path.exists() and not _read_owned(path, client):
        raise IntegrationError(f"refusing to overwrite an unmanaged file: {path}")
    _atomic_write(path, content)
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    _atomic_write(_owner_path(path), _owner_text(client, path, digest))
    return path


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def render_codex_config(
    *,
    model: str,
    endpoint: str,
    context_size: int | None = None,
    reasoning: str = "auto",
) -> str:
    """Render an isolated Codex Responses API profile."""

    model = _validate_scalar(model, "model")
    endpoint = _v1_endpoint(_validate_endpoint(endpoint))
    context_size = _validate_context(context_size)
    reasoning = _validate_reasoning(reasoning)
    lines = [
        "# Kestrel-managed integration profile; do not edit by hand.",
        f"model = {_toml_string(model)}",
        'model_provider = "kestrel"',
    ]
    if context_size is not None:
        lines.append(f"model_context_window = {context_size}")
        lines.append(f"model_auto_compact_token_limit = {max(1, int(context_size * 0.8))}")
    if reasoning in _REASONING_TO_CODEX:
        lines.append(f"model_reasoning_effort = {_toml_string(_REASONING_TO_CODEX[reasoning])}")
    lines.extend(
        [
            "",
            "[model_providers.kestrel]",
            'name = "Kestrel"',
            f"base_url = {_toml_string(endpoint)}",
            'wire_api = "responses"',
            "",
            "[features]",
            "remote_models = false",
            "",
            "[model_providers.kestrel.auth]",
            f"command = {_toml_string(sys.executable)}",
            'args = ["-m", "kestrel", "agents", "token"]',
            "timeout_ms = 5000",
            "refresh_interval_ms = 300000",
            "",
        ]
    )
    return "\n".join(lines)


def render_claude_settings(*, model: str, endpoint: str) -> str:
    """Render a private Claude Code settings file for ``claude --settings``."""

    model = _validate_scalar(model, "model")
    endpoint = _anthropic_endpoint(_validate_endpoint(endpoint))
    payload = {
        "env": {
            "ANTHROPIC_BASE_URL": endpoint,
            "ANTHROPIC_MODEL": model,
            "ANTHROPIC_CUSTOM_MODEL_OPTION": model,
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        },
        "apiKeyHelper": _token_command(),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def render_opencode_config(
    *, model: str, endpoint: str, context_size: int | None = None, max_tokens: int = 8192
) -> str:
    """Render an OpenCode JSONC overlay using its supported custom config path."""

    model = _validate_scalar(model, "model")
    endpoint = _v1_endpoint(_validate_endpoint(endpoint))
    context_size = _validate_context(context_size)
    if type(max_tokens) is not int or max_tokens < 1:
        raise IntegrationError("max_tokens must be a positive integer")
    metadata: dict[str, object] = {"name": f"{model} via Kestrel"}
    if context_size is not None:
        metadata["limit"] = {"context": context_size, "output": max_tokens}
    payload = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "kestrel": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Kestrel (local)",
                "options": {
                    "baseURL": endpoint,
                    "apiKey": "{env:KESTREL_API_KEY}",
                },
                "models": {model: metadata},
            }
        },
    }
    body = json.dumps(payload, indent=2, ensure_ascii=False)
    return "// Kestrel-managed integration overlay; do not edit by hand.\n" + body + "\n"


def launch_metadata(
    client: str,
    *,
    model: str,
    endpoint: str = "http://127.0.0.1:8080",
    context_size: int | None = None,
    reasoning: str = "auto",
    max_tokens: int | None = None,
    home: str | os.PathLike[str] | None = None,
    codex_home: str | os.PathLike[str] | None = None,
) -> LaunchMetadata:
    """Return a deterministic client command and isolated environment."""

    client = _validate_client(client)
    model = _validate_scalar(model, "model")
    endpoint = _validate_endpoint(endpoint)
    context_size = _validate_context(context_size)
    reasoning = _validate_reasoning(reasoning)
    if max_tokens is None:
        max_tokens = min(8192, max(1, context_size // 4)) if context_size is not None else 8192
    if type(max_tokens) is not int or max_tokens < 1:
        raise IntegrationError("max_tokens must be a positive integer")
    if context_size is not None and max_tokens > context_size:
        raise IntegrationError("max_tokens must not exceed context_size")
    path = integration_path(client, home=home, codex_home=codex_home)
    if client == "pi":
        environment = {"PI_CODING_AGENT_DIR": str(path.parent)}
        command = ("pi", "--provider", "kestrel", "--model", model)
    elif client == "omp":
        environment = {"PI_CODING_AGENT_DIR": str(path.parent)}
        command = ("omp", "--model", f"kestrel/{model}")
    elif client == "codex":
        environment = {"CODEX_HOME": str(path.parent)}
        command = ("codex", "--profile", "kestrel")
    elif client == "claude":
        environment = {}
        command = ("claude", "--settings", str(path))
    else:
        environment = {"OPENCODE_CONFIG": str(path)}
        command = ("opencode", "--model", f"kestrel/{model}")
    return LaunchMetadata(client, command, environment, path, endpoint, model, context_size, reasoning, max_tokens)


def setup_agent_integration(
    client: str,
    *,
    model: str,
    endpoint: str = "http://127.0.0.1:8080",
    context_size: int | None = None,
    reasoning: str = "auto",
    max_tokens: int | None = None,
    home: str | os.PathLike[str] | None = None,
    codex_home: str | os.PathLike[str] | None = None,
) -> LaunchMetadata:
    """Create one Kestrel-owned config and return its launch metadata."""

    client = _validate_client(client)
    metadata = launch_metadata(
        client,
        model=model,
        endpoint=endpoint,
        context_size=context_size,
        reasoning=reasoning,
        max_tokens=max_tokens,
        home=home,
        codex_home=codex_home,
    )
    if client in {"pi", "omp"}:
        spec = IntegrationSpec(
            model_id=model,
            base_url=_v1_endpoint(_validate_endpoint(endpoint)),
            context_window=context_size,
            max_tokens=metadata.max_tokens,
            reasoning=reasoning != "off",
        )
        if client == "pi":
            install_pi_provider(spec, metadata.config_path)
        else:
            install_omp_provider(spec, metadata.config_path)
        return metadata
    if client == "codex":
        content = render_codex_config(
            model=model,
            endpoint=endpoint,
            context_size=context_size,
            reasoning=reasoning,
        )
    elif client == "claude":
        content = render_claude_settings(model=model, endpoint=endpoint)
    else:
        content = render_opencode_config(
            model=model,
            endpoint=endpoint,
            context_size=context_size,
            max_tokens=metadata.max_tokens,
        )
    _write_owned(metadata.config_path, client, content)
    return metadata


def status_agent_integration(
    client: str,
    *,
    home: str | os.PathLike[str] | None = None,
    codex_home: str | os.PathLike[str] | None = None,
) -> IntegrationStatus:
    """Inspect only the dedicated Kestrel-owned path for *client*."""

    client = _validate_client(client)
    path = integration_path(client, home=home, codex_home=codex_home)
    if client == "pi":
        state = status_pi_provider(path)
        return IntegrationStatus(
            client,
            configured=state.installed,
            path=path,
            owned=state.installed,
            details=_configured_details(client, path) if state.installed else {},
        )
    if client == "omp":
        state = status_omp_provider(path)
        return IntegrationStatus(
            client,
            configured=state.installed,
            path=path,
            owned=state.installed,
            details=_configured_details(client, path) if state.installed else {},
        )
    owned = _read_owned(path, client)
    details: dict[str, str] = {}
    if path.exists() and not owned:
        details["warning"] = "path exists but is not Kestrel-owned"
    elif owned:
        details.update(_configured_details(client, path))
    return IntegrationStatus(client, configured=owned, path=path, owned=owned, details=details)


def _configured_details(client: str, path: Path) -> dict[str, str]:
    """Read non-secret endpoint/model metadata from an owned config."""

    try:
        text = path.read_text(encoding="utf-8")
        if client == "pi":
            provider = json.loads(text)["providers"]["kestrel"]
            model = provider["models"][0]
            endpoint = provider["baseUrl"]
            context = model.get("contextWindow")
            model_id = model["id"]
        elif client == "omp":
            lines = text.splitlines(keepends=True)
            root = next(
                (i for i, line in enumerate(lines) if re.match(r"^providers:\s*(?:#.*)?$", line.rstrip("\n"))),
                -1,
            )
            bounds = _omp_provider_bounds(lines, root)
            if bounds is None:
                raise ValueError("owned provider block is missing")
            block = "".join(lines[bounds[0] : bounds[1]])
            endpoint_match = re.search(r'^    baseUrl:\s*"([^"]+)"', block, re.MULTILINE)
            model_match = re.search(r'^      - id:\s*"([^"]+)"', block, re.MULTILINE)
            context_match = re.search(r"^        contextWindow:\s*(\d+)", block, re.MULTILINE)
            if not endpoint_match or not model_match:
                raise ValueError("owned provider metadata is missing")
            endpoint = endpoint_match.group(1)
            model_id = model_match.group(1)
            context = context_match.group(1) if context_match else None
        elif client == "codex":
            payload = tomllib.loads(text)
            endpoint = payload["model_providers"]["kestrel"]["base_url"]
            model_id = payload["model"]
            context = payload.get("model_context_window")
        elif client == "claude":
            payload = json.loads(text)
            endpoint = payload["env"]["ANTHROPIC_BASE_URL"]
            model_id = payload["env"]["ANTHROPIC_MODEL"]
            context = None
        else:
            payload = json.loads(text.split("\n", 1)[1])
            provider = payload["provider"]["kestrel"]
            endpoint = provider["options"]["baseURL"]
            model_id, model = next(iter(provider["models"].items()))
            context = model.get("limit", {}).get("context")
    except (
        OSError,
        UnicodeError,
        ValueError,
        KeyError,
        IndexError,
        TypeError,
        json.JSONDecodeError,
        tomllib.TOMLDecodeError,
    ):
        return {"warning": "owned integration metadata could not be read"}
    details = {"endpoint": str(endpoint).removesuffix("/v1").rstrip("/"), "model": str(model_id)}
    if context is not None:
        details["context"] = str(context)
    return details


def remove_agent_integration(
    client: str,
    *,
    home: str | os.PathLike[str] | None = None,
    codex_home: str | os.PathLike[str] | None = None,
) -> bool:
    """Remove a config only when its ownership marker is valid."""

    client = _validate_client(client)
    path = integration_path(client, home=home, codex_home=codex_home)
    if client == "pi":
        return remove_pi_provider(path).changed
    if client == "omp":
        return remove_omp_provider(path).changed
    if not _read_owned(path, client):
        return False
    path.unlink()
    marker = _owner_path(path)
    try:
        marker.unlink()
    except FileNotFoundError:
        pass
    return True


# Pi/Oh My Pi provider registration intentionally lives alongside the isolated
# client launchers above, but targets only the provider key owned by Kestrel.
PI_MODELS_PATH = Path("~/.pi/agent/models.json")
OMP_MODELS_PATH = Path("~/.omp/agent/models.yml")
_PI_PROVIDER = "kestrel"


def _token_command() -> str:
    return shlex.join((sys.executable, "-m", "kestrel", "agents", "token"))


def _provider_token_command() -> str:
    return "!" + _token_command()


def _provider_path(value: str | os.PathLike[str] | None, default: Path) -> Path:
    if value is not None:
        target = Path(value).expanduser()
    elif os.environ.get("PI_CODING_AGENT_DIR") and default in {PI_MODELS_PATH, OMP_MODELS_PATH}:
        filename = "models.json" if default == PI_MODELS_PATH else "models.yml"
        target = Path(os.environ["PI_CODING_AGENT_DIR"]).expanduser() / filename
    else:
        target = default.expanduser()
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_file():
            raise IntegrationError(f"integration target must be a regular file, not {target}")
    elif target.parent.exists() and not target.parent.is_dir():
        raise IntegrationError(f"integration target parent is not a directory: {target.parent}")
    return target


def _provider_read(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise IntegrationError(f"cannot read integration config {path}: {exc}") from exc


def _provider_write(path: Path, before: str, after: str, *, dry_run: bool) -> bool:
    if before == after:
        return False
    if not dry_run:
        # Keep the client's existing file mode when updating a user-owned
        # config, while still refusing symlinks through _provider_path.
        mode = path.stat().st_mode & 0o7777 if path.exists() else 0o600
        _atomic_write(path, after, mode=mode)
    return True


def _provider_owner_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.kestrel-provider")


def _provider_digest(value: object) -> str:
    if isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _provider_owned(path: Path, client: str, value: object) -> bool:
    marker = _provider_owner_path(path)
    if marker.is_symlink() or not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return (
        payload.get("client") == client
        and payload.get("path") == str(path)
        and payload.get("sha256") == _provider_digest(value)
    )


def _write_provider_owner(path: Path, client: str, value: object, *, dry_run: bool) -> None:
    if dry_run:
        return
    marker = _provider_owner_path(path)
    content = json.dumps(
        {"client": client, "path": str(path), "sha256": _provider_digest(value)},
        sort_keys=True,
        separators=(",", ":"),
    )
    _atomic_write(marker, content + "\n")


def _remove_provider_owner(path: Path, *, dry_run: bool) -> None:
    if dry_run:
        return
    try:
        _provider_owner_path(path).unlink()
    except FileNotFoundError:
        pass


def _pi_provider_entry(spec: IntegrationSpec) -> dict[str, object]:
    model: dict[str, object] = {
        "id": spec.model_id,
        "name": f"Kestrel · {spec.model_id}",
        "reasoning": spec.reasoning,
        "input": ["text"],
        "maxTokens": spec.max_tokens,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        # Kestrel owns the startup reasoning budget; do not send a field that
        # llama.cpp's OpenAI-compatible endpoint does not promise to accept.
        "compat": {
            "supportsDeveloperRole": False,
            "supportsReasoningEffort": False,
            "supportsUsageInStreaming": True,
            "maxTokensField": "max_tokens",
        },
    }
    if spec.context_window is not None:
        model["contextWindow"] = spec.context_window
    return {
        "baseUrl": spec.base_url.rstrip("/"),
        "api": "openai-completions",
        "apiKey": _provider_token_command(),
        "models": [model],
    }


def _omp_provider_entry(spec: IntegrationSpec) -> dict[str, object]:
    model: dict[str, object] = {
        "id": spec.model_id,
        "name": f"Kestrel · {spec.model_id}",
        "reasoning": spec.reasoning,
        "input": ["text"],
        "maxTokens": spec.max_tokens,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "compat": {
            "supportsDeveloperRole": False,
            "supportsReasoningEffort": False,
            "supportsUsageInStreaming": True,
            "maxTokensField": "max_tokens",
        },
    }
    if spec.context_window is not None:
        model["contextWindow"] = spec.context_window
    return {
        "baseUrl": spec.base_url.rstrip("/"),
        "api": "openai-completions",
        # OMP resolves !-prefixed apiKey values by executing the command and
        # caching its trimmed stdout; the token never enters this config.
        "apiKey": _provider_token_command(),
        "models": [model],
    }


def _pi_provider_update(path: Path, spec: IntegrationSpec, *, dry_run: bool, remove: bool) -> ProviderIntegrationResult:
    before = _provider_read(path)
    if remove and not before:
        return ProviderIntegrationResult("pi", "remove", path, False, False, dry_run)
    if before:
        try:
            payload = json.loads(before)
        except json.JSONDecodeError as exc:
            raise IntegrationError(f"invalid Pi models JSON {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise IntegrationError(f"invalid Pi models JSON {path}: top level must be an object")
    else:
        payload = {}
    providers = payload.get("providers")
    if providers is None:
        providers = {}
        payload["providers"] = providers
    if not isinstance(providers, dict):
        raise IntegrationError(f"invalid Pi models JSON {path}: providers must be an object")
    if remove:
        if _PI_PROVIDER not in providers:
            return ProviderIntegrationResult("pi", "remove", path, False, False, dry_run)
        if not _provider_owned(path, "pi", providers[_PI_PROVIDER]):
            raise IntegrationError(f"refusing to remove an unmanaged Pi Kestrel provider from {path}")
        providers.pop(_PI_PROVIDER, None)
    else:
        if _PI_PROVIDER in providers and not _provider_owned(path, "pi", providers[_PI_PROVIDER]):
            raise IntegrationError(f"refusing to overwrite an unmanaged Pi Kestrel provider in {path}")
        providers[_PI_PROVIDER] = _pi_provider_entry(spec)
    if not providers:
        payload.pop("providers", None)
    after = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    changed = _provider_write(path, before, after, dry_run=dry_run)
    if remove:
        _remove_provider_owner(path, dry_run=dry_run)
    else:
        _write_provider_owner(path, "pi", providers[_PI_PROVIDER], dry_run=dry_run)
    return ProviderIntegrationResult("pi", "remove" if remove else "install", path, changed, not remove, dry_run)


def _validate_omp_yaml(text: str, path: Path) -> None:
    if "\t" in text:
        raise IntegrationError(f"invalid OMP YAML {path}: tabs are not supported")
    # PyYAML is optional and intentionally not part of Kestrel's runtime
    # dependencies. When present, use it for full syntax/type validation;
    # otherwise the structural checks in _omp_provider_bounds still protect
    # the provider block we own.
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        yaml = None
    if yaml is not None:
        try:
            parsed = yaml.safe_load(text) if text.strip() else {}
        except Exception as exc:
            raise IntegrationError(f"invalid OMP YAML {path}: {exc}") from exc
        if parsed is not None and not isinstance(parsed, dict):
            raise IntegrationError(f"invalid OMP YAML {path}: top level must be a mapping")
        if (
            isinstance(parsed, dict)
            and "providers" in parsed
            and not isinstance(parsed["providers"], dict | type(None))
        ):
            raise IntegrationError(f"invalid OMP YAML {path}: providers must be a mapping")
    if re.search(r"^providers:\s*\[[^\]]*\]\s*$", text, re.MULTILINE):
        raise IntegrationError(f"invalid OMP YAML {path}: providers must be a mapping")


def _omp_provider_block(spec: IntegrationSpec) -> str:
    model = _omp_provider_entry(spec)["models"][0]
    lines = [
        "  kestrel:",
        f"    baseUrl: {json.dumps(spec.base_url.rstrip('/'))}",
        "    api: openai-completions",
        f"    apiKey: {json.dumps(_provider_token_command())}",
        "    models:",
        f"      - id: {json.dumps(model['id'])}",
        f"        name: {json.dumps(model['name'])}",
        f"        reasoning: {'true' if model['reasoning'] else 'false'}",
        "        input: [text]",
    ]
    if spec.context_window is not None:
        lines.append(f"        contextWindow: {spec.context_window}")
    lines.extend(
        [
            f"        maxTokens: {model['maxTokens']}",
            "        cost:",
            "          input: 0",
            "          output: 0",
            "          cacheRead: 0",
            "          cacheWrite: 0",
            "        compat:",
            "          supportsDeveloperRole: false",
            "          supportsReasoningEffort: false",
            "          supportsUsageInStreaming: true",
            '          maxTokensField: "max_tokens"',
        ]
    )
    return "\n".join(lines) + "\n"


def _omp_provider_bounds(lines: list[str], root_start: int) -> tuple[int, int] | None:
    if root_start < 0:
        return None
    start = None
    for index in range(root_start + 1, len(lines)):
        line = lines[index].rstrip("\n")
        if re.match(r"^  kestrel:\s*(?:#.*)?$", line):
            start = index
            break
        if line.strip() and re.match(r"^\S", line) and not line.lstrip().startswith("#"):
            break
    if start is None:
        return None
    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index].rstrip("\n")
        if re.match(r"^  \S", line) or re.match(r"^\S", line):
            end = index
            break
    return start, end


def _omp_provider_update(
    path: Path, spec: IntegrationSpec, *, dry_run: bool, remove: bool
) -> ProviderIntegrationResult:
    before = _provider_read(path)
    _validate_omp_yaml(before, path)
    lines = before.splitlines(keepends=True)
    provider_roots = [line for line in lines if re.match(r"^providers:\s*", line.rstrip("\n"))]
    if len(provider_roots) > 1:
        raise IntegrationError(f"invalid OMP YAML {path}: duplicate top-level providers keys")
    root = next(
        (i for i, line in enumerate(lines) if re.match(r"^providers:\s*(?:#.*)?$", line.rstrip("\n"))),
        None,
    )
    empty_root = next(
        (i for i, line in enumerate(lines) if re.match(r"^providers:\s*\{\s*\}\s*(?:#.*)?$", line.rstrip("\n"))),
        None,
    )
    if root is None and empty_root is not None:
        if remove:
            return ProviderIntegrationResult("omp", "remove", path, False, False, dry_run)
        lines[empty_root] = "providers:\n"
        root = empty_root
    if root is None:
        if remove:
            return ProviderIntegrationResult("omp", "remove", path, False, False, dry_run)
        body = before.rstrip("\n")
        after = (body + "\n\n" if body else "") + "providers:\n" + _omp_provider_block(spec)
    else:
        bounds = _omp_provider_bounds(lines, root)
        if bounds is not None:
            start, end = bounds
            current_block = "".join(lines[start:end])
            if not _provider_owned(path, "omp", current_block):
                action = "remove" if remove else "overwrite"
                raise IntegrationError(f"refusing to {action} an unmanaged OMP Kestrel provider in {path}")
            if remove:
                del lines[start:end]
                # Use an inline empty mapping only when no other provider
                # children remain.  Replacing a non-empty root would leave
                # its indented children beneath `providers: {}` and corrupt
                # the user's YAML.
                has_other_provider = False
                for line in lines[root + 1 :]:
                    if line.strip() and re.match(r"^\S", line) and not line.lstrip().startswith("#"):
                        break
                    if re.match(r"^  \S", line):
                        has_other_provider = True
                        break
                if not has_other_provider:
                    lines[root] = "providers: {}\n"
                after = "".join(lines)
            else:
                lines[start:end] = [_omp_provider_block(spec)]
                after = "".join(lines)
        elif remove:
            return ProviderIntegrationResult("omp", "remove", path, False, False, dry_run)
        else:
            insert = root + 1
            while insert < len(lines) and (not lines[insert].strip() or lines[insert].lstrip().startswith("#")):
                insert += 1
            while insert < len(lines) and not re.match(r"^\S", lines[insert]):
                insert += 1
            lines[insert:insert] = [_omp_provider_block(spec)]
            after = "".join(lines)
    if not after.endswith("\n"):
        after += "\n"
    changed = _provider_write(path, before, after, dry_run=dry_run)
    if remove:
        _remove_provider_owner(path, dry_run=dry_run)
    else:
        new_lines = after.splitlines(keepends=True)
        new_root = next(
            (i for i, line in enumerate(new_lines) if re.match(r"^providers:\s*(?:#.*)?$", line.rstrip("\n"))),
            -1,
        )
        bounds = _omp_provider_bounds(new_lines, new_root)
        if bounds is None:
            raise IntegrationError("internal error: generated OMP provider block is missing")
        _write_provider_owner(path, "omp", "".join(new_lines[bounds[0] : bounds[1]]), dry_run=dry_run)
    return ProviderIntegrationResult("omp", "remove" if remove else "install", path, changed, not remove, dry_run)


def install_pi_provider(
    spec: IntegrationSpec, path: str | os.PathLike[str] | None = None, *, dry_run: bool = False
) -> ProviderIntegrationResult:
    return _pi_provider_update(_provider_path(path, PI_MODELS_PATH), spec, dry_run=dry_run, remove=False)


def install_omp_provider(
    spec: IntegrationSpec, path: str | os.PathLike[str] | None = None, *, dry_run: bool = False
) -> ProviderIntegrationResult:
    return _omp_provider_update(_provider_path(path, OMP_MODELS_PATH), spec, dry_run=dry_run, remove=False)


def remove_pi_provider(
    path: str | os.PathLike[str] | None = None, *, dry_run: bool = False
) -> ProviderIntegrationResult:
    return _pi_provider_update(
        _provider_path(path, PI_MODELS_PATH), IntegrationSpec("kestrel"), dry_run=dry_run, remove=True
    )


def remove_omp_provider(
    path: str | os.PathLike[str] | None = None, *, dry_run: bool = False
) -> ProviderIntegrationResult:
    return _omp_provider_update(
        _provider_path(path, OMP_MODELS_PATH), IntegrationSpec("kestrel"), dry_run=dry_run, remove=True
    )


def status_pi_provider(path: str | os.PathLike[str] | None = None) -> ProviderIntegrationResult:
    target = _provider_path(path, PI_MODELS_PATH)
    if not target.exists():
        return ProviderIntegrationResult("pi", "status", target, False, False)
    try:
        payload = json.loads(_provider_read(target))
    except json.JSONDecodeError as exc:
        raise IntegrationError(f"invalid Pi models JSON {target}: {exc}") from exc
    installed = (
        isinstance(payload, dict)
        and isinstance(payload.get("providers"), dict)
        and _PI_PROVIDER in payload["providers"]
        and _provider_owned(target, "pi", payload["providers"][_PI_PROVIDER])
    )
    return ProviderIntegrationResult("pi", "status", target, False, installed)


def status_omp_provider(path: str | os.PathLike[str] | None = None) -> ProviderIntegrationResult:
    target = _provider_path(path, OMP_MODELS_PATH)
    text = _provider_read(target)
    _validate_omp_yaml(text, target)
    lines = text.splitlines(keepends=True)
    root = next((i for i, line in enumerate(lines) if re.match(r"^providers:\s*(?:#.*)?$", line)), -1)
    bounds = _omp_provider_bounds(lines, root)
    installed = bounds is not None and _provider_owned(target, "omp", "".join(lines[bounds[0] : bounds[1]]))
    return ProviderIntegrationResult("omp", "status", target, False, installed)


__all__ = [
    "IntegrationError",
    "IntegrationSpec",
    "IntegrationStatus",
    "LaunchMetadata",
    "OMP_MODELS_PATH",
    "PI_MODELS_PATH",
    "ProviderIntegrationResult",
    "SUPPORTED_CLIENTS",
    "integration_dir",
    "integration_path",
    "install_omp_provider",
    "install_pi_provider",
    "launch_metadata",
    "remove_agent_integration",
    "remove_omp_provider",
    "remove_pi_provider",
    "render_claude_settings",
    "render_codex_config",
    "render_opencode_config",
    "setup_agent_integration",
    "status_agent_integration",
    "status_omp_provider",
    "status_pi_provider",
]
