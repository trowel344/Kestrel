from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .errors import ConfigError
from .util import write_atomic


@dataclass(frozen=True)
class KestrelConfig:
    default_model: str | None = None
    models_dir: str | None = None
    llama_cpp_dir: str | None = None


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
    return KestrelConfig(
        default_model=local.get("default_model"),
        models_dir=local.get("models_dir"),
        llama_cpp_dir=local.get("llama_cpp_dir"),
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
    write_atomic(target, "\n".join(lines) + "\n")
    return target
