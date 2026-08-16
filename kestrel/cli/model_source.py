"""Resolving a model string (alias, GGUF path, HF cache/source) to a source.

The resolved ``model_info`` document is the single input handed to the planner
and the backend builders, so every command resolves through the same logic.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .. import ui
from ..errors import MissingModelError, ModelError
from ..model_store import hf_snapshot_dir, model_total_size
from . import state


def _resolve_hf_model_dir(path: Path) -> Path | None:
    return hf_snapshot_dir(path)


def _resolve_model_alias(model_str: str) -> Path | None:
    candidates = state.MODEL_ALIASES.get(model_str.strip().lower())
    if candidates is None:
        return None
    env_name, *paths = candidates
    configured = os.environ.get(env_name)
    if configured:
        paths.insert(0, configured)
    for raw_path in paths:
        candidate = Path(raw_path).expanduser()
        if candidate.is_file():
            return candidate.resolve()
    return None


def detect_model(model_str: str) -> dict | None:
    alias_path = _resolve_model_alias(model_str)
    if alias_path is not None:
        path = str(alias_path)
        return {"type": "gguf", "path": path, "gguf_name": path}

    candidate = Path(model_str).expanduser()
    if candidate.is_file():
        try:
            if candidate.suffix.lower() == ".gguf":
                is_gguf = True
            else:
                with candidate.open("rb") as handle:
                    is_gguf = handle.read(4) == b"GGUF"
        except OSError:
            is_gguf = False
        if is_gguf:
            path = str(candidate.resolve())
            return {"type": "gguf", "path": path, "gguf_name": path}
    if candidate.is_dir():
        resolved = _resolve_hf_model_dir(candidate.resolve())
        if resolved:
            return {"type": "safetensors", "path": str(resolved), "hub_id": None}

    if model_str.startswith(("hf://", "huggingface://")):
        hub_id = model_str.split("://", 1)[1]
    else:
        hub_id = model_str
    if "/" not in hub_id:
        return None

    cache_root = Path(os.path.expanduser(f"~/.cache/huggingface/hub/models--{hub_id.replace('/', '--')}"))
    resolved = _resolve_hf_model_dir(cache_root)
    if resolved:
        return {"type": "safetensors", "path": str(resolved), "hub_id": hub_id}
    return {"type": "safetensors", "path": None, "hub_id": hub_id}


def read_gguf_config(gguf_path: str) -> dict:
    from ..gguf.metadata import read_planner_metadata

    return read_planner_metadata(gguf_path)


def _safetensors_info(directory: str) -> dict:
    """Report config.json summary + param estimate for a safetensors model dir."""
    base = Path(directory)
    info: dict = {
        "type": "safetensors",
        "path": str(base),
        "size_bytes": sum(p.stat().st_size for p in base.glob("*.safetensors")),
    }
    config_file = base / "config.json"
    if not config_file.is_file():
        info["config"] = {"error": "no config.json present"}
        return info
    try:
        config = json.loads(config_file.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        info["config"] = {"error": f"unreadable config.json: {exc}"}
        return info
    info["config"] = {
        key: config.get(key)
        for key in (
            "architectures",
            "model_type",
            "n_layer",
            "num_hidden_layers",
            "n_experts",
            "num_local_experts",
            "num_experts_per_tok",
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
            "max_position_embeddings",
        )
    }
    hidden = config.get("hidden_size") or config.get("d_model")
    layers = config.get("num_hidden_layers") or config.get("n_layer")
    if hidden and layers:
        intermediate = config.get("intermediate_size") or 4 * hidden
        params = layers * (4 * hidden * hidden + 3 * hidden * intermediate)
        info["estimated_params_b"] = round(params / 1e9, 2)
    return info


def _model_profile(model_info: dict):
    from ..core.planner import ModelProfile

    if model_info["type"] == "gguf":
        cfg = read_gguf_config(model_info["path"])
        return ModelProfile(
            path=model_info["path"],
            n_layers=cfg["n_layer"],
            n_experts=cfg["n_exp"],
            n_experts_used=cfg["n_used"],
            hidden_size=cfg["hidden"],
            expert_ff_size=cfg["n_ff"],
            has_mtp=cfg["mtp_layers"] > 0,
            file_size_bytes=_gguf_total_size(model_info["path"]),
            file_type=cfg["file_type"],
        )

    from ..gguf.converter import NVFP4Converter

    converter = NVFP4Converter(model_info["path"])
    return ModelProfile(
        path=model_info["path"],
        n_layers=converter.n_layer,
        n_experts=converter.n_exp,
        n_experts_used=converter.n_used,
        hidden_size=converter.hidden,
        expert_ff_size=converter.n_ff,
        has_mtp=converter.mtp_layers > 0,
        file_size_bytes=_safetensors_size(converter),
    )


def _gguf_total_size(path: str) -> int:
    """Total on-disk bytes for a GGUF, summing sibling shards of a split set.

    A split GGUF lives in several ``{prefix}-NNNNN-of-MMMMM`` files; sizing
    from a single shard undercounts a multi-file model and would silently
    disable CPU-MoE or pick an oversized context for a model that cannot fit.
    Degrades to 0 (as the old single-file probe did) when the path is gone.
    """
    candidate = Path(path)
    if not candidate.is_file():
        return 0
    try:
        return model_total_size(candidate)
    except (OSError, ValueError):
        return 0


def _safetensors_size(converter) -> int:
    """Total size of the shard files the source model's weight map references.

    This lets the planner apply its model-larger-than-VRAM heuristic to a
    safetensors source, matching what a converted GGUF would provide.
    """
    total = 0
    for shard in set(converter.wm.values()):
        try:
            total += os.path.getsize(os.path.join(converter.model_dir, shard))
        except OSError:
            continue
    return total


def _cached_gguf_path(source_path: str) -> str:
    return source_path.rstrip(os.sep) + ".gguf"


def _resolve_model_source(args) -> dict:
    """Detect ``args.model`` and surface unresolvable sources as errors.

    Returns ``model_info`` which may still be a safetensors directory; call
    :func:`_ensure_local_gguf` to convert it to the runnable GGUF.
    """
    model_info = detect_model(args.model)
    if model_info is None:
        raise MissingModelError(
            f"could not resolve model or GGUF path: {args.model}",
            hint="check the model name and run `kestrel models list`",
        )
    if model_info["type"] == "safetensors" and not model_info["path"]:
        raise MissingModelError(
            "model source exists but its safetensors are not downloaded",
            hint=f"huggingface-cli download {model_info['hub_id']}",
        )
    return model_info


def _ensure_local_gguf(model_info: dict, args) -> dict:
    """Return the runnable GGUF for ``model_info``, converting safetensors.

    A safetensors source is converted to a deterministic side-by-side ``.gguf``
    cache unless ``--no-convert`` is set. GGUF sources pass through unchanged.
    """
    if model_info["type"] != "safetensors":
        return model_info
    output = _cached_gguf_path(model_info["path"])
    if os.path.isfile(output):
        print(ui.kv("Cached GGUF", output), file=sys.stderr if getattr(args, "json", False) else sys.stdout)
    elif args.no_convert:
        raise ModelError(
            "model is safetensors but no cached GGUF exists",
            hint="remove --no-convert or pass a GGUF file",
        )
    else:
        from ..gguf.converter import NVFP4Converter

        converter = NVFP4Converter(model_info["path"], include_mtp=False)
        converter.convert(output)
    return {"type": "gguf", "path": output, "gguf_name": output}
