"""The interactive terminal menu (``kestrel menu``).

A dependency-free front door over the scriptable CLI. Re-enters the dispatch
loop in the same process, so it reloads the config snapshot through
:func:`kestrel.cli.state.reload_state` before each run (setup paths save first).

The menu deliberately keeps its interaction layer separate from the command
entry point.  ``_MenuSession`` owns the small amount of menu state and turns
each selection into an ordinary CLI argument vector.  That makes the menu
navigation testable without a real terminal or model runtime while preserving
the same visible prompts and dispatch behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

from .. import ui
from ..config import load_config
from . import probes


def _menu_status_compact() -> str:
    gpu = probes.detect_gpu()
    parts = []
    if gpu:
        parts.append(f"gpu: {gpu['name']} ({gpu['vram_free_mb']}/{gpu['vram_total_mb']} MiB free)")
    else:
        parts.append("gpu: not detected")
    parts.append(f"ram: {probes._available_ram_mib()} MiB")
    text = "   " + "   ·   ".join(parts)
    return ui.dim(ui._truncate(text, ui.width()))


_BACK = ("Go back", "")
_TARGETS = ("auto", "balanced", "quality", "speed")


class _MenuSession:
    """Interactive menu state and actions.

    Methods issue the same command vectors as the original menu, but each
    branch is an isolated action.  In particular, a cancelled action returns
    to :meth:`run` rather than carrying a half-built command into another
    action.
    """

    def __init__(self, launch: Callable[..., None] | None = None) -> None:
        self._launch_override = launch

    def run(self) -> None:
        from .parser import _kestrel_version

        actions = (
            self._chat,
            self._select_model,
            self._import_models,
            self._manage_models,
            self._configure,
        )
        while True:
            chosen = self._main_selection(_kestrel_version())
            if chosen < 0 or chosen >= len(actions):
                return
            actions[chosen]()

    def _main_selection(self, version: str) -> int:
        default_model = load_config().default_model
        header = "\n".join(
            [
                ui.bold("Kestrel") + (ui.dim(f"   v{version}") if ui.USE_ANSI else f"   v{version}"),
                _menu_status_compact(),
            ]
        )
        return ui.select(
            [
                ("Chat with Model", default_model or "no model set"),
                ("Select Model(s)", ""),
                ("Import Models", ""),
                ("Manage Models", ""),
                ("Configure Kestrel", ""),
                ("Exit", ""),
            ],
            header=header,
            hint=ui.key_hint(),
        )

    def _launch(self, *arguments: str) -> None:
        if self._launch_override is not None:
            self._launch_override(*arguments)
            return
        from .main import _run_dispatched
        from .parser import build_parser

        try:
            parser = build_parser()
            _run_dispatched(parser, parser.parse_args(arguments))
        except SystemExit:
            pass
        except KeyboardInterrupt:
            print()
        ui.pause()

    @staticmethod
    def _prompt_required(label: str) -> str | None:
        try:
            return ui.ask(label).strip()
        except (EOFError, KeyboardInterrupt):
            return None

    @staticmethod
    def _model_options() -> list[tuple[str, str]]:
        from ..model_store import ModelStoreError, default_models_dir, discover_local_models, list_ollama_models

        config = load_config()
        options: list[tuple[str, str]] = []
        if config.default_model:
            options.append((config.default_model, "configured default"))
        root = Path(config.models_dir).expanduser() if config.models_dir else default_models_dir()
        for path in discover_local_models(root):
            options.append((str(path), f"{path.stat().st_size / 1024**3:.2f} GiB local GGUF"))
        try:
            for item in list_ollama_models():
                options.append((f"ollama://{item.name}", f"{item.size} via Ollama"))
        except ModelStoreError:
            pass
        return options

    def _pick_model(self, title: str, *, keep_current: bool = False) -> str | None:
        options = self._model_options()
        if keep_current:
            options.insert(0, ("<keep current>", "leave the default model unchanged"))
        if not options:
            print(f"  {ui.warn_mark()} No models found; download one from the market first.")
            return None
        options.append(("<type a path>", "enter a model path, name, or alias manually"))
        chosen = ui.select(options, title=title, hint=ui.key_hint())
        if chosen < 0:
            return None
        label = options[chosen][0]
        if label == "<keep current>":
            return "<keep current>"
        if label == "<type a path>":
            return self._prompt_required("Model path, name, or alias")
        return label

    def _pick_ollama_model(self) -> str | None:
        from ..model_store import ModelStoreError, list_ollama_models

        try:
            models = list_ollama_models()
        except ModelStoreError as exc:
            print(f"  {ui.fail_mark()} {exc}")
            return None
        if not models:
            print(f"  {ui.warn_mark()} No Ollama models found; pull one first with `ollama pull <name>`.")
            return None
        options = [(item.name, f"{item.size}  {item.model_id}") for item in models]
        options.append(("<type a name>", "enter an Ollama model name manually"))
        chosen = ui.select(options, title="Import an Ollama model", hint=ui.key_hint())
        if chosen < 0:
            return None
        label = options[chosen][0]
        if label == "<type a name>":
            return self._prompt_required("Ollama model name")
        return label

    @staticmethod
    def _pick(items: list[tuple[str, str]], title: str) -> int:
        index = ui.select(items, title=title, hint=ui.key_hint())
        return index if index >= 0 else len(items) - 1

    def _chat(self) -> None:
        default_model = load_config().default_model
        model = default_model or self._pick_model("Chat with which model?")
        if model is None:
            return
        context = ui.ask(
            "Context tokens",
            default="auto",
            validate=lambda value: (
                None
                if value == "auto" or (value.isdigit() and int(value) >= 512)
                else "Context must be 'auto' or an integer of at least 512"
            ),
        )
        target = self._placement_target()
        if target is None:
            return
        launch_args = ["chat", model, "--ctx-size", context, "--target", target]
        if ui.confirm("Prime the page cache for a faster load", default=False):
            launch_args.append("--warm-cache")
        self._launch(*launch_args)

    @staticmethod
    def _placement_target() -> str | None:
        target = "auto"
        if not ui.confirm("Choose the placement target explicitly", default=False):
            return target
        index = ui.select(
            [
                ("auto", "adaptive: pick from available free memory"),
                ("balanced", "memory-aware default"),
                ("quality", "most stable, slower"),
                ("speed", "experimental throughput bias"),
            ],
            title="Placement target",
            hint=ui.key_hint(),
        )
        if index < 0:
            return None
        return _TARGETS[index]

    def _select_model(self) -> None:
        model = self._pick_model("Select a model")
        if model is not None:
            self._launch("setup", "--model", model)

    def _import_models(self) -> None:
        index = self._pick(
            [("Import an Ollama model", ""), ("Pull from Hugging Face", ""), _BACK],
            "Import Models",
        )
        if index == 0:
            self._import_ollama()
        elif index == 1:
            self._pull_hugging_face()

    def _import_ollama(self) -> None:
        name = self._pick_ollama_model()
        if name is not None:
            self._launch("models", "import", f"ollama://{name}", "--set-default")

    def _pull_hugging_face(self) -> None:
        repo = self._prompt_required("Hugging Face repository (OWNER/REPO)")
        if repo is None:
            return
        filename = self._prompt_required("Specific filename (Enter for entire repository)")
        if filename is None:
            return
        command = ["models", "pull", f"hf://{repo}"]
        if filename:
            command.extend(["--file", filename])
        print(ui.dim("Checking download size first..."))
        self._launch(*command, "--dry-run")
        if ui.confirm("Proceed with the download", default=False):
            self._launch(*command)

    def _manage_models(self) -> None:
        index = self._pick(
            [
                ("Installed models", ""),
                ("Search model market", ""),
                ("Hardware diagnostics", ""),
                ("Benchmark", ""),
                ("Convert / prune a model", ""),
                _BACK,
            ],
            "Manage Models",
        )
        if index == 0:
            self._launch("models", "list", "--resolve")
        elif index == 1:
            self._search_models()
        elif index == 2:
            self._launch("doctor")
        elif index == 3:
            self._launch("benchmark")
        elif index == 4:
            self._convert_model()

    def _search_models(self) -> None:
        query = self._prompt_required("Search GGUF models")
        if query is not None:
            self._launch("models", "search", query)

    def _convert_model(self) -> None:
        model_dir = self._prompt_required("Model directory (downloaded safetensors)")
        if model_dir is None:
            return
        output = self._prompt_required("Output GGUF path (Enter for automatic name)")
        if output is None:
            return
        keep = ui.ask(
            "Experts per layer to keep (Enter to keep all)",
            default="",
            validate=lambda value: (
                None
                if value == "" or (value.isdigit() and int(value) > 0)
                else "Enter a positive integer or leave blank to keep all"
            ),
        )
        importance = self._expert_importance(keep)
        if importance is False:
            return
        command = ["convert", model_dir]
        if output:
            command.extend(["-o", output])
        if keep and int(keep) > 0:
            command.extend(["--experts-keep", keep])
            if importance:
                command.extend(["--expert-importance", importance])
        print(ui.dim("This is an experimental, opt-in quality trade-off."))
        self._launch(*command)

    def _expert_importance(self, keep: str) -> str | None | bool:
        if not keep or int(keep) <= 0:
            return None
        if not ui.confirm("Choose which experts to keep by importance (router-frequency JSON)?", default=False):
            return None
        importance = self._prompt_required("Path to expert-importance JSON")
        return importance if importance is not None else False

    def _configure(self) -> None:
        index = self._pick([("Default model", ""), ("Models directory", ""), _BACK], "Configure Kestrel")
        if index == 0:
            self._configure_default()
        elif index == 1:
            self._configure_models_dir()

    def _configure_default(self) -> None:
        model = self._pick_model("Set the default model", keep_current=True)
        if model is None:
            return
        command = ["setup"]
        if model != "<keep current>":
            command.extend(["--model", model])
        self._launch(*command)

    def _configure_models_dir(self) -> None:
        models_dir = self._prompt_required("Managed models directory (Enter to leave unchanged)")
        if models_dir is not None:
            self._launch("setup", "--models-dir", models_dir)


def cmd_menu(_args=None):
    """Dependency-free interactive front door over the scriptable CLI."""

    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise SystemExit("The interactive menu requires a terminal; use `kestrel --help`.")
    _MenuSession().run()
