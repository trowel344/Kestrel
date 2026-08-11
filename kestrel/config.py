from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .errors import ConfigError
from .util import write_atomic

REASONING_BUDGETS = {
    "auto": None,
    "off": 0,
    "low": 512,
    "medium": 2048,
    "high": 8192,
    "maximum": -1,
}
REASONING_LEVELS = tuple(REASONING_BUDGETS)


@dataclass(frozen=True)
class KestrelConfig:
    default_model: str | None = None
    models_dir: str | None = None
    llama_cpp_dir: str | None = None
    context_size: int | str = "auto"
    reasoning_level: str = "auto"


def config_path() -> Path:
    override = os.environ.get("KESTREL_CONFIG")
    if override:
        return Path(override).expanduser()
    root = Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser()
    return root / "kestrel" / "config.toml"


def load_config(path: str | Path | None = None) -> KestrelConfig:
    target = Path(path) if path else config_path()
    if not target.is_file():
        return KestrelConfig()
    try:
        payload = tomllib.loads(target.read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"invalid Kestrel config {target}: {exc}") from exc
    local = payload.get("local", {})
    if not isinstance(local, dict):
        raise ConfigError(f"invalid Kestrel config {target}: [local] must be a table")
    for field in ("default_model", "models_dir", "llama_cpp_dir"):
        value = local.get(field)
        if value is not None and not isinstance(value, str):
            raise ConfigError(f"invalid Kestrel config {target}: local.{field} must be a string")
    context_size = local.get("context_size", "auto")
    if context_size != "auto" and (type(context_size) is not int or context_size < 512):
        raise ConfigError(
            f"invalid Kestrel config {target}: local.context_size must be 'auto' or an integer of at least 512"
        )
    reasoning_level = local.get("reasoning_level", "auto")
    if not isinstance(reasoning_level, str) or reasoning_level not in REASONING_LEVELS:
        raise ConfigError(
            f"invalid Kestrel config {target}: local.reasoning_level must be one of {', '.join(REASONING_LEVELS)}"
        )
    return KestrelConfig(
        default_model=local.get("default_model"),
        models_dir=local.get("models_dir"),
        llama_cpp_dir=local.get("llama_cpp_dir"),
        context_size=context_size,
        reasoning_level=reasoning_level,
    )


def save_config(config: KestrelConfig, path: str | Path | None = None) -> Path:
    target = Path(path) if path else config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Kestrel configuration. API keys belong in environment variables.", "[local]"]
    if config.default_model:
        lines.append(f"default_model = {json.dumps(config.default_model)}")
    if config.models_dir:
        lines.append(f"models_dir = {json.dumps(config.models_dir)}")
    if config.llama_cpp_dir:
        lines.append(f"llama_cpp_dir = {json.dumps(config.llama_cpp_dir)}")
    lines.append(
        "context_size = "
        + (json.dumps(config.context_size) if config.context_size == "auto" else str(config.context_size))
    )
    lines.append(f"reasoning_level = {json.dumps(config.reasoning_level)}")
    write_atomic(target, "\n".join(lines) + "\n")
    return target
