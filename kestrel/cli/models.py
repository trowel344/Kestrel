"""The ``models`` command family: search, files, list, import, pull, info,
recommend."""

from __future__ import annotations

import json
import shlex
import struct
from pathlib import Path

from .. import ui
from ..errors import InputError, ModelError
from . import health, model_source, probes, state


def _models_search(args):
    from ..model_store import search_huggingface

    rows = search_huggingface(args.query, limit=args.limit)
    if args.json:
        print(json.dumps({"models": rows}))
        return
    print(
        ui.box(
            f"Model market: {args.query}",
            "\n".join(
                f"  {ui.bold(item['id'])}\n"
                f"  {ui.dim('{} downloads, {} likes, {}'.format(item['downloads'], item['likes'], item['license'] or 'unspecified license'))}"
                for item in rows
            )
            or ui.dim("  no results"),
        )
    )
    print(ui.dim("Inspect model cards and choose a file before downloading; popularity is not a quality score."))


def _models_files(args):
    from ..model_store import list_huggingface_ggufs

    rows = list_huggingface_ggufs(args.source)
    if args.json:
        print(json.dumps({"repository": args.source, "files": rows}))
        return
    print(
        ui.box(
            f"GGUF files: {args.source.removeprefix('hf://')}",
            "\n".join(
                f"  {ui.bold(item['path'])}\n"
                f"  {ui.dim('{:.2f} GiB'.format(item['size_bytes'] / 1024**3))}  "
                f"Hub scan: {ui.yellow(item['security_status'])}"
                for item in rows
            )
            if rows
            else ui.dim("  none"),
        )
    )
    print(ui.dim("Hub scan status is metadata from Hugging Face, not a Kestrel security guarantee."))


def _models_list(args, root):
    from ..model_store import (
        complete_gguf_models,
        discover_local_models,
        list_ollama_models,
        model_total_size,
    )

    local = complete_gguf_models(discover_local_models(root))
    ollama = list_ollama_models(resolve_paths=args.resolve)
    payload = {
        "kestrel": [{"path": str(path), "size_bytes": model_total_size(path)} for path in local],
        "ollama": [
            {
                "name": item.name,
                "id": item.model_id,
                "size": item.size,
                "modified": item.modified,
                "local_path": str(item.local_path) if item.local_path else None,
            }
            for item in ollama
        ],
    }
    if args.json:
        print(json.dumps(payload))
        return
    kestrel_body = "\n".join(
        f"  {ui.bold(item['path'])}  {ui.dim('{:.2f} GiB'.format(item['size_bytes'] / 1024**3))}"
        for item in payload["kestrel"]
    ) or ui.dim("  none")
    print(ui.box(f"Kestrel models ({root})", kestrel_body))
    ollama_rows = [
        [
            item.name,
            item.size,
            str(item.local_path) if item.local_path else "cloud or unresolved",
        ]
        for item in ollama
    ]
    if args.resolve:
        ollama_headers = ["name", "size", "location"]
    else:
        ollama_headers = ["name", "size"]
        ollama_rows = [row[:2] for row in ollama_rows]
    print(
        ui.box(
            "Ollama models",
            ui.table(ollama_headers, ollama_rows) if ollama_rows else ui.dim("  none"),
        )
    )


def _models_import(args):
    from ..model_store import ModelStoreError, resolve_ollama_blob

    source = args.source
    default_value: str | Path
    if source.startswith("ollama://"):
        name = source.removeprefix("ollama://")
        path = resolve_ollama_blob(name)
        if path is None:
            print(f"Imported {name} through the Ollama provider (no local GGUF blob).")
            if args.set_default:
                health._save_default_model(source)
                print("Set as the default Kestrel model.")
            return
        default_value = source
    else:
        path = Path(source).expanduser().resolve()
        default_value = path
    detected = model_source.detect_model(str(path))
    if not detected or detected["type"] != "gguf":
        raise ModelStoreError(f"source is not a readable GGUF model: {path}")
    print(f"Imported {path}")
    if args.set_default:
        health._save_default_model(default_value)
        print("Set as the default Kestrel model.")


def _models_pull(args):
    from ..model_store import (
        ModelStoreError,
        choose_default_gguf,
        discover_local_models,
        pull_huggingface,
        pull_ollama,
    )

    source = args.source
    if source.startswith("ollama://"):
        name = source.removeprefix("ollama://")
        if args.dry_run:
            print(f"Would run: ollama pull {shlex.quote(name)}")
            return
        item = pull_ollama(name)
        print(f"Pulled Ollama model {item.name} ({item.size})")
        if item.local_path:
            print(f"Ollama-managed local blob: {item.local_path}")
            print("Kestrel will use Ollama's compatible engine for this model.")
            if args.set_default:
                health._save_default_model(f"ollama://{item.name}")
                print("Set as the default Kestrel model.")
        else:
            print("This model is served remotely by Ollama and has no local GGUF blob.")
        return
    result = pull_huggingface(
        source,
        filename=args.file,
        include=args.include,
        revision=args.revision,
        destination=Path(args.destination).expanduser() if args.destination else None,
        dry_run=args.dry_run,
    )
    print(result.stdout.strip())
    if args.set_default:
        if args.dry_run:
            raise ModelStoreError("cannot set a default model during a dry-run")
        try:
            selected = choose_default_gguf(discover_local_models(result.directory))
        except ModelStoreError as exc:
            raise ModelStoreError(f"{exc}. Choose one with `kestrel models import PATH --set-default`.") from exc
        metadata = model_source.read_gguf_config(str(selected))
        if metadata["architecture"] == "unknown" or not metadata["n_layer"]:
            raise ModelStoreError(f"downloaded GGUF has unusable planner metadata: {selected}")
        health._save_default_model(selected)
        print(f"Set {selected} as the default Kestrel model.")


def _models_info(args):
    from ..model_store import ModelStoreError, resolve_ollama_blob

    source = args.source
    if source.startswith("ollama://"):
        path = resolve_ollama_blob(source.removeprefix("ollama://"))
        if path is None:
            raise ModelStoreError("Ollama model has no reusable local GGUF blob")
    else:
        path = Path(source).expanduser()
    detected = model_source.detect_model(str(path))
    if not detected:
        raise ModelStoreError(f"could not resolve model: {source}")
    if detected["type"] == "safetensors" and detected["path"]:
        info = model_source._safetensors_info(detected["path"])
        print(json.dumps(info, indent=2))
        return
    if detected["type"] != "gguf":
        raise ModelStoreError("model info currently requires a local GGUF or safetensors directory")
    cfg = model_source.read_gguf_config(detected["path"])
    cfg.update(
        path=detected["path"],
        size_bytes=Path(detected["path"]).stat().st_size,
    )
    source_manifest = Path(detected["path"]).parent / ".kestrel-source.json"
    if source_manifest.is_file():
        try:
            cfg["source"] = json.loads(source_manifest.read_text())
        except (OSError, json.JSONDecodeError):
            cfg["source"] = {"error": "unreadable source manifest"}
    print(json.dumps(cfg, indent=2))


def _models_recommend(args, root):
    from ..model_store import (
        complete_gguf_models,
        discover_local_models,
        list_ollama_models,
        model_total_size,
    )

    gpu = probes.detect_gpu()
    vram = (gpu or {}).get("vram_total_mb", 0) * 1024**2
    ram = probes._available_ram_mib() * 1024**2
    candidates: dict[str, Path] = {str(path): path for path in complete_gguf_models(discover_local_models(root))}
    for item in list_ollama_models(resolve_paths=True):
        if item.local_path:
            candidates[f"ollama://{item.name}"] = item.local_path
    ranked = []
    for label, path in candidates.items():
        size = model_total_size(path)
        if vram and size <= vram * 0.82:
            fit, detail, rank = "excellent", "fits in safe GPU weight budget", 0
        elif size <= ram + vram * 0.65:
            fit, detail, rank = "viable", "requires CPU/RAM offload", 1
        else:
            fit, detail, rank = "paging", "exceeds working memory; expect storage stalls", 2
        try:
            cfg = model_source.read_gguf_config(str(path))
            architecture = cfg["architecture"]
            layers = cfg["n_layer"]
        except (OSError, struct.error, ValueError, KeyError):
            architecture, layers = "unreadable", 0
            fit, detail, rank = "unsupported", "GGUF metadata could not be read", 3
        ranked.append(
            {
                "source": label,
                "engine": "ollama" if label.startswith("ollama://") else "llama.cpp",
                "path": str(path),
                "size_gib": round(size / 1024**3, 2),
                "architecture": architecture,
                "layers": layers,
                "fit": fit,
                "reason": detail,
                "_rank": rank,
            }
        )
    ranked.sort(key=lambda item: (item["_rank"], item["size_gib"]))
    for item in ranked:
        item.pop("_rank")
    if args.json:
        print(json.dumps({"hardware": {"gpu": gpu, "available_ram_mib": ram // 1024**2}, "models": ranked}))
    else:
        fit_color = {
            "excellent": ui.green,
            "viable": ui.cyan,
            "paging": ui.yellow,
            "unsupported": ui.red,
        }
        rows = []
        for item in ranked:
            colorize = fit_color.get(item["fit"], ui.dim)
            rows.append(f"  {colorize('[{}]'.format(item['fit'].upper().ljust(13)))} {ui.bold(item['source'])}")
            rows.append(
                f"  {ui.dim('{:.2f} GiB'.format(item['size_gib']))}  {ui.dim(item['architecture'])}  "
                f"{ui.dim(item['reason'])}"
            )
        print(
            ui.box(
                f"Recommendations for {(gpu or {}).get('name', 'CPU-only host')}",
                "\n".join(rows),
            )
        )
        print(ui.dim("Fit is a memory classification, not a speed or quality guarantee; run `kestrel benchmark`."))


def cmd_models(args):
    from ..model_store import (
        ModelStoreError,
        default_models_dir,
    )

    root = Path(state.USER_CONFIG.models_dir).expanduser() if state.USER_CONFIG.models_dir else default_models_dir()
    try:
        if args.models_command == "search":
            _models_search(args)
        elif args.models_command == "files":
            _models_files(args)
        elif args.models_command == "list":
            _models_list(args, root)
        elif args.models_command == "import":
            _models_import(args)
        elif args.models_command == "pull":
            _models_pull(args)
        elif args.models_command == "info":
            _models_info(args)
        elif args.models_command == "recommend":
            _models_recommend(args, root)
        else:
            raise InputError("choose a models command: search, files, list, recommend, info, pull, or import")
    except ModelStoreError as exc:
        raise ModelError(str(exc)) from exc
