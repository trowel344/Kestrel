"""The interactive terminal menu (``kestrel menu``).

A dependency-free front door over the scriptable CLI. Re-enters the dispatch
loop in the same process, so it reloads the config snapshot through
:func:`kestrel.cli.state.reload_state` before each run (setup paths save first).

The menu deliberately keeps its interaction layer separate from the command
entry point.  ``_MenuSession`` owns the small amount of menu state and turns
each selection into an ordinary CLI argument vector. The first screen exposes
chat, models, model settings, and tools; technical placement controls remain
available in the scriptable CLI instead of blocking a first conversation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

from .. import ui
from ..config import load_config
from . import probes


def _memory_label(mib: int | float) -> str:
    return f"{mib / 1024:.1f} GiB" if mib >= 1024 else f"{int(mib)} MiB"


def _model_label(model: str | None) -> str:
    if not model:
        return "not set"
    if model.startswith("ollama://"):
        return model.removeprefix("ollama://")
    return Path(model).name or model


def _menu_status_compact() -> str:
    gpu = probes.detect_gpu()
    parts: list[str] = []
    if gpu:
        parts.append(str(gpu["name"]))
        parts.append(f"{_memory_label(gpu['vram_free_mb'])} VRAM free")
    else:
        parts.append("CPU mode")
    parts.append(f"{_memory_label(probes._available_ram_mib())} RAM free")
    return ui.dim(ui._truncate("  ·  ".join(parts), ui.width()))


_BACK = ("Go back", "")


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
            self._work,
            self._models,
            self._model_settings,
            self._tools,
        )
        while True:
            chosen = self._main_selection(_kestrel_version())
            if chosen < 0 or chosen >= len(actions):
                return
            actions[chosen]()

    def _main_selection(self, version: str) -> int:
        config = load_config()
        default_model = config.default_model
        default_display = ui._truncate(_model_label(default_model), max(12, ui.width() - 9))
        context = getattr(config, "context_size", "auto")
        reasoning = getattr(config, "reasoning_level", "auto")
        header = "\n".join(
            [
                ui.bold("Kestrel") + (ui.dim(f"  v{version}") if ui.USE_ANSI else f"  v{version}"),
                ui.dim(f"Default  {default_display}"),
                ui.dim(f"Context  {context}  ·  Reasoning  {reasoning}"),
                _menu_status_compact(),
            ]
        )
        return ui.select(
            [
                ("Start chat", "automatic settings" if default_model else "choose a model first"),
                ("Work with coding agents", "Pi, Oh My Pi, Codex, Claude Code, and OpenCode"),
                ("Models", "choose, add, or organize models"),
                ("Settings", "context, reasoning, KV cache, and GPU placement"),
                ("Tools", "system check, benchmark, and conversion"),
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
        seen: set[str] = set()

        def add(value: str, description: str) -> None:
            if value not in seen:
                seen.add(value)
                options.append((value, description))

        if config.default_model:
            add(config.default_model, "configured default")
        root = Path(config.models_dir).expanduser() if config.models_dir else default_models_dir()
        for path in discover_local_models(root):
            add(str(path), f"{path.stat().st_size / 1024**3:.2f} GiB local GGUF")
        try:
            for item in list_ollama_models():
                add(f"ollama://{item.name}", f"{item.size} via Ollama")
        except ModelStoreError:
            pass
        return options

    def _pick_model(self, title: str) -> str | None:
        options = self._model_options()
        if not options:
            print(f"  {ui.warn_mark()} No models found. Add a model first.")
            return None
        options.append(("<type a path>", "enter a model path, name, or alias manually"))
        chosen = ui.select(options, title=title, hint=ui.key_hint())
        if chosen < 0:
            return None
        label = options[chosen][0]
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
        if not default_model:
            print(f"  {ui.info_mark()} Choose a model before starting a chat.")
            self._import_models()
            return
        self._launch("chat", default_model)

    def _work(self) -> None:
        config = load_config()
        if not config.default_model:
            print(f"  {ui.info_mark()} Choose a model before launching a coding agent.")
            self._import_models()
            return
        index = self._pick(
            [
                ("Launch Pi", "local coding agent with shell, read, edit, and write tools"),
                ("Launch Oh My Pi", "enhanced Pi workflow using the Kestrel model"),
                ("Launch Codex", "Responses API profile with Kestrel-owned settings"),
                ("Launch Claude Code", "experimental Anthropic-compatible local endpoint"),
                ("Launch OpenCode", "OpenAI-compatible local provider"),
                ("Set up all integrations", "create private, reversible client profiles"),
                ("Integration status", "server, API routes, and installed clients"),
                ("Stop work server", "release model memory when work is finished"),
                ("View server logs", "inspect model load or API failures"),
                _BACK,
            ],
            "Work with coding agents",
        )
        launch_clients = ("pi", "omp", "codex", "claude", "opencode")
        if index < len(launch_clients):
            self._launch("agents", "launch", launch_clients[index])
        elif index == 5:
            self._launch("agents", "setup", "all")
        elif index == 6:
            self._launch("agents", "status")
        elif index == 7:
            self._launch("agents", "stop")
        elif index == 8:
            self._launch("agents", "logs")

    def _select_model(self) -> None:
        model = self._pick_model("Select a model")
        if model is not None:
            self._launch("setup", "--model", model)

    def _import_models(self) -> None:
        index = self._pick(
            [
                ("Use a local GGUF", "choose a file already on this computer"),
                ("Import from Ollama", "reuse an installed Ollama model"),
                ("Download from Hugging Face", "pull a model repository or file"),
                _BACK,
            ],
            "Add a model",
        )
        if index == 0:
            self._use_local_model()
        elif index == 1:
            self._import_ollama()
        elif index == 2:
            self._pull_hugging_face()

    def _use_local_model(self) -> None:
        model = self._prompt_required("Local GGUF path")
        if model:
            self._launch("setup", "--model", model)

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

    def _models(self) -> None:
        index = self._pick(
            [
                ("Choose default model", "used when chat starts"),
                ("Installed models", "show local and provider models"),
                ("Add a model", "local GGUF, Ollama, or Hugging Face"),
                ("Find models online", "search the GGUF model market"),
                ("Model storage", "change the managed models folder"),
                _BACK,
            ],
            "Models",
        )
        if index == 0:
            self._select_model()
        elif index == 1:
            self._launch("models", "list", "--resolve")
        elif index == 2:
            self._import_models()
        elif index == 3:
            self._search_models()
        elif index == 4:
            self._configure_models_dir()

    def _model_settings(self) -> None:
        config = load_config()
        context = str(getattr(config, "context_size", "auto"))
        reasoning = str(getattr(config, "reasoning_level", "auto"))
        kv_cache_type = str(getattr(config, "kv_cache_type", "auto"))
        turbo = "on" if getattr(config, "kv_cache_turbo", False) else "off"
        gpu_layers = str(getattr(config, "gpu_layers", "auto"))
        cpu_moe = str(getattr(config, "cpu_moe", "auto"))
        index = self._pick(
            [
                ("Context window", f"currently {context}"),
                ("Reasoning level", f"currently {reasoning}"),
                ("KV cache type", f"currently {kv_cache_type}"),
                ("Turbo KV", f"currently {turbo}"),
                ("GPU layers", f"currently {gpu_layers}"),
                ("CPU MoE", f"currently {cpu_moe}"),
                _BACK,
            ],
            "Model settings",
        )
        if index == 0:
            self._configure_context()
        elif index == 1:
            self._configure_reasoning()
        elif index == 2:
            self._configure_kv_cache_type()
        elif index == 3:
            self._configure_turbo_kv()
        elif index == 4:
            self._configure_gpu_layers()
        elif index == 5:
            self._configure_cpu_moe()

    def _configure_context(self) -> None:
        options = [
            ("Automatic", "fit the context to current memory"),
            ("2K", "2,048 tokens"),
            ("4K", "4,096 tokens"),
            ("8K", "8,192 tokens"),
            ("16K", "16,384 tokens"),
            ("32K", "32,768 tokens"),
            ("Custom", "enter any value of at least 512"),
            _BACK,
        ]
        index = self._pick(options, "Context window")
        values = ("auto", "2048", "4096", "8192", "16384", "32768")
        if index < len(values):
            self._launch("settings", "--context", values[index])
        elif index == 6:
            value = self._prompt_required("Context tokens (minimum 512)")
            if value:
                self._launch("settings", "--context", value)

    def _configure_reasoning(self) -> None:
        levels = [
            ("Automatic", "use the model and engine default"),
            ("Off", "answer without a reasoning budget"),
            ("Low", "up to 512 reasoning tokens"),
            ("Medium", "up to 2,048 reasoning tokens"),
            ("High", "up to 8,192 reasoning tokens"),
            ("Maximum", "allow unrestricted reasoning"),
            _BACK,
        ]
        index = self._pick(levels, "Reasoning level")
        values = ("auto", "off", "low", "medium", "high", "maximum")
        if index < len(values):
            self._launch("settings", "--reasoning", values[index])

    def _configure_kv_cache_type(self) -> None:
        types = [
            ("Automatic", "use the engine's default KV cache type"),
            ("FP16", "full precision"),
            ("BF16", "bfloat16 precision"),
            ("Q8_0", "8-bit quantized"),
            ("Q4_0", "4-bit quantized, halves KV use (lossless on hybrid models)"),
            ("Q4_1", "4-bit with residual"),
            _BACK,
        ]
        index = self._pick(types, "KV cache type")
        values = ("auto", "f16", "bf16", "q8_0", "q4_0", "q4_1")
        if index < len(values):
            self._launch("settings", "--kv-cache-type", values[index])

    def _configure_turbo_kv(self) -> None:
        options = [
            ("Off", "standard KV cache (recommended until engines ship --turbo-kv)"),
            ("On", "opt-in TurboQuant KV compression, only if the engine build supports it"),
            _BACK,
        ]
        index = self._pick(options, "Turbo KV cache")
        if index == 0:
            self._launch("settings", "--no-turbo-kv")
        elif index == 1:
            self._launch("settings", "--turbo-kv")

    def _configure_gpu_layers(self) -> None:
        options = [
            ("Automatic", "use the tuned placement profile"),
            ("All", "place every layer on the GPU"),
            ("24", "a balanced middle ground"),
            ("Custom", "enter any exact layer count"),
            _BACK,
        ]
        index = self._pick(options, "GPU layers")
        if index == 0:
            self._launch("settings", "--gpu-layers", "auto")
        elif index == 1:
            self._launch("settings", "--gpu-layers", "all")
        elif index == 2:
            self._launch("settings", "--gpu-layers", "24")
        elif index == 3:
            value = self._prompt_required("GPU layers (auto, all, or a count)")
            if value:
                self._launch("settings", "--gpu-layers", value)

    def _configure_cpu_moe(self) -> None:
        options = [
            ("Automatic", "use the tuned placement profile"),
            ("On", "run all MoE experts on the CPU (stable on small GPUs)"),
            ("Off", "offload MoE experts to the GPU when VRAM allows"),
            _BACK,
        ]
        index = self._pick(options, "CPU MoE")
        values = ("auto", "on", "off")
        if index < len(values):
            self._launch("settings", "--cpu-moe", values[index])

    def _tools(self) -> None:
        index = self._pick(
            [
                ("Check system", "hardware, engine, storage, and model health"),
                ("Benchmark", "measure prompt and generation speed"),
                ("Convert or prune", "advanced GGUF conversion tools"),
                _BACK,
            ],
            "Tools",
        )
        if index == 0:
            self._launch("doctor")
        elif index == 1:
            self._launch("benchmark")
        elif index == 2:
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

    def _configure_models_dir(self) -> None:
        models_dir = self._prompt_required("Managed models directory (Enter to leave unchanged)")
        if models_dir is not None:
            self._launch("setup", "--models-dir", models_dir)


def cmd_menu(_args=None):
    """Dependency-free interactive front door over the scriptable CLI."""

    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise SystemExit("The interactive menu requires a terminal; use `kestrel --help`.")
    _MenuSession().run()
