from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KestrelConfig:
    default_model: str | None = None
    models_dir: str | None = None
    llama_cpp_dir: str | None = None
    kimi_base_url: str | None = None
    kimi_model: str = "kimi-k3"


def config_path() -> Path:
    override = os.environ.get("KESTREL_CONFIG")
    if override:
        return Path(override).expanduser()
    root = Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser()
    return root / "kestrel" / "config.toml"


def load_config(path: Path | None = None) -> KestrelConfig:
    target = path or config_path()
    if not target.is_file():
        return KestrelConfig()
    try:
        payload = tomllib.loads(target.read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"invalid Kestrel config {target}: {exc}") from exc
    local = payload.get("local", {})
    kimi = payload.get("kimi", {})
    return KestrelConfig(
        default_model=local.get("default_model"),
        models_dir=local.get("models_dir"),
        llama_cpp_dir=local.get("llama_cpp_dir"),
        kimi_base_url=kimi.get("base_url"),
        kimi_model=kimi.get("model", "kimi-k3"),
    )


def save_config(config: KestrelConfig, path: Path | None = None) -> Path:
    target = path or config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Kestrel configuration. API keys belong in environment variables.", "[local]"]
    if config.default_model:
        lines.append(f"default_model = {json.dumps(config.default_model)}")
    if config.models_dir:
        lines.append(f"models_dir = {json.dumps(config.models_dir)}")
    if config.llama_cpp_dir:
        lines.append(f"llama_cpp_dir = {json.dumps(config.llama_cpp_dir)}")
    lines.extend(["", "[kimi]"])
    if config.kimi_base_url:
        lines.append(f"base_url = {json.dumps(config.kimi_base_url)}")
    lines.append(f"model = {json.dumps(config.kimi_model)}")
    target.write_text("\n".join(lines) + "\n")
    return target
