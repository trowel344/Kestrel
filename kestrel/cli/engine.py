"""Engine: provenance, upstream updates, rebuilds, and the ``build`` command.

Owns the llama.cpp engine checkout. ``EngineError`` (a :class:`KestrelError`)
propagates straight to the dispatch layer, which renders it as a failure.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .. import ui
from ..errors import InputError
from . import runtime, state


def cmd_build(args):
    """Backward-compatible alias for the transactional engine rebuild."""
    from .. import engine

    directory = _engine_dir(args)
    result = engine.rebuild(directory, dry_run=getattr(args, "dry_run", False))
    if getattr(args, "json", False):
        print(json.dumps(result, default=str))
        return
    if result["status"] == "dry_run":
        print(ui.kv("Engine", f"would rebuild {directory}"))
    else:
        print(ui.kv("Engine", f"rebuilt {directory}", value_color=ui.green))


def _engine_dir(args) -> str:
    from ..backends.llama_cpp import default_llama_cpp_dir

    directory = getattr(args, "dir", None)
    if directory:
        return str(Path(directory).expanduser().resolve())
    override = os.environ.get("KESTREL_LLAMA_CPP_DIR")
    if override:
        return override
    if state.USER_CONFIG.llama_cpp_dir:
        return state.USER_CONFIG.llama_cpp_dir
    return default_llama_cpp_dir()


def cmd_engine(args):
    from .. import engine

    directory = _engine_dir(args)
    command = args.engine_command
    out = runtime._human_stream(args)
    js = bool(getattr(args, "json", False))

    if command == "status":
        status = engine.engine_status(directory, check_remote=not getattr(args, "no_remote", False))
        if js:
            print(json.dumps(status, default=str))
            return
        rows: list[tuple[str, str, object]] = [
            ("Directory", status["directory"], None),
            ("Git", "yes" if status["git"] else "no", None),
            ("Remote", status["remote"] or "—", None),
            ("Branch", status["branch"] or "detached", None),
            ("Commit", (status["commit"] or "—")[:12], None),
            ("Managed", "yes" if status["manifest"] else "no", None),
        ]
        if status["git"] and status["remote"]:
            if status["behind"] is None:
                rows.append(("Stale", "unknown (remote unreachable)", ui.dim))
            elif status["stale"]:
                rows.append(
                    (
                        "Stale",
                        f"{status['behind']} behind upstream — run `kestrel engine update`",
                        ui.red,
                    )
                )
            else:
                rows.append(("Up to date", "current", None))
        if status["dirty"]:
            rows.append(("Dirty", "uncommitted changes", ui.yellow))
        if status.get("artifacts"):
            present = [name for name, artifact in status["artifacts"].items() if artifact["exists"]]
            rows.append(("Built", ", ".join(present) or "none", None))
        body = "\n".join(ui.kv(key, value, value_color=color) for key, value, color in rows)
        print(ui.box("Engine", body, title_color=ui.cyan), file=out)
        return

    if command == "update":
        result = engine.update(
            directory,
            dry_run=getattr(args, "dry_run", False),
            force=getattr(args, "force", False),
            remote=getattr(args, "remote", None),
        )
    elif command == "rebuild":
        result = engine.rebuild(directory, dry_run=getattr(args, "dry_run", False))
    elif command == "set":
        result = {
            "status": "adopted",
            "directory": directory,
            **engine.adopt(directory, getattr(args, "remote", None)).as_dict(),
        }
    else:
        raise InputError("choose an engine command: status, update, rebuild, or set")

    if js:
        print(json.dumps(result, default=str))
        return
    if result.get("status") == "up_to_date":
        print(ui.kv("Engine", "already up to date", value_color=ui.green), file=out)
        return
    if result.get("status") == "dry_run":
        print(
            ui.kv("Engine", "rebuild planned (dry run)"),
            file=out,
        )
    for key, value in result.items():
        if key in ("directory", "cmake_flags", "targets", "artifacts", "status"):
            continue
        print(ui.kv(key.replace("_", " ").title(), value), file=out)
    print(ui.kv("Status", result.get("status", "")), file=out)
