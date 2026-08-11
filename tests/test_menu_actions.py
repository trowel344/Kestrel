"""Behavioral tests for the dependency-free interactive menu layer."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from kestrel.cli import menu


class FakeUI:
    USE_ANSI = False

    def __init__(self, *, selections=(), answers=(), confirmations=()):
        self.selections = list(selections)
        self.answers = list(answers)
        self.confirmations = list(confirmations)
        self.pauses = 0

    def select(self, *_args, **_kwargs):
        return self.selections.pop(0)

    def ask(self, *_args, **_kwargs):
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def confirm(self, *_args, **_kwargs):
        return self.confirmations.pop(0)

    @staticmethod
    def bold(text):
        return text

    @staticmethod
    def dim(text):
        return text

    @staticmethod
    def key_hint():
        return ""

    @staticmethod
    def warn_mark():
        return "WARN"

    @staticmethod
    def fail_mark():
        return "FAIL"

    def pause(self):
        self.pauses += 1


def _config(default_model="model.gguf"):
    return SimpleNamespace(default_model=default_model, models_dir="")


def test_cmd_menu_requires_a_terminal(monkeypatch):
    monkeypatch.setattr(menu.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    monkeypatch.setattr(menu.sys, "stdout", SimpleNamespace(isatty=lambda: False))

    with pytest.raises(SystemExit, match="requires a terminal"):
        menu.cmd_menu()


def test_chat_action_builds_the_same_dispatch_vector_and_exit_returns(monkeypatch):
    fake_ui = FakeUI(selections=[0, 5], answers=["1024"], confirmations=[False, False])
    launches = []
    monkeypatch.setattr(menu, "ui", fake_ui)
    monkeypatch.setattr(menu, "load_config", lambda: _config())
    monkeypatch.setattr(menu, "_menu_status_compact", lambda: "status")

    menu._MenuSession(launch=lambda *args: launches.append(args)).run()

    assert launches == [("chat", "model.gguf", "--ctx-size", "1024", "--target", "auto")]


def test_submenu_back_is_a_noop_and_returns_to_main(monkeypatch):
    fake_ui = FakeUI(selections=[2, 2, 5])
    launches = []
    monkeypatch.setattr(menu, "ui", fake_ui)
    monkeypatch.setattr(menu, "load_config", lambda: _config())
    monkeypatch.setattr(menu, "_menu_status_compact", lambda: "status")

    menu._MenuSession(launch=lambda *args: launches.append(args)).run()

    assert launches == []


def test_cancelled_prompt_does_not_dispatch(monkeypatch):
    fake_ui = FakeUI(selections=[3, 1, 5], answers=[KeyboardInterrupt()])
    launches = []
    monkeypatch.setattr(menu, "ui", fake_ui)
    monkeypatch.setattr(menu, "load_config", lambda: _config())
    monkeypatch.setattr(menu, "_menu_status_compact", lambda: "status")

    menu._MenuSession(launch=lambda *args: launches.append(args)).run()

    assert launches == []


def test_launch_converts_dispatch_system_exit_to_pause(monkeypatch):
    fake_ui = FakeUI()
    monkeypatch.setattr(menu, "ui", fake_ui)

    class Parser:
        @staticmethod
        def parse_args(arguments):
            return arguments

    monkeypatch.setattr("kestrel.cli.parser.build_parser", lambda: Parser())

    def fail_dispatch(_parser, _args):
        raise SystemExit(7)

    main_module = importlib.import_module("kestrel.cli.main")
    monkeypatch.setattr(main_module, "_run_dispatched", fail_dispatch)

    menu._MenuSession()._launch("doctor")

    assert fake_ui.pauses == 1


def test_prompt_required_handles_end_of_input(monkeypatch):
    fake_ui = FakeUI(answers=[EOFError()])
    monkeypatch.setattr(menu, "ui", fake_ui)

    assert menu._MenuSession._prompt_required("Model") is None


@pytest.mark.parametrize(
    ("gpu", "expected"),
    [
        (None, "gpu: not detected"),
        ({"name": "RTX", "vram_free_mb": 4, "vram_total_mb": 8}, "gpu: RTX (4/8 MiB free)"),
    ],
)
def test_menu_status_reports_gpu_and_ram(monkeypatch, gpu, expected):
    monkeypatch.setattr(menu.probes, "detect_gpu", lambda: gpu)
    monkeypatch.setattr(menu.probes, "_available_ram_mib", lambda: 123)

    status = menu._menu_status_compact()

    assert expected in status
    assert "ram: 123 MiB" in status


def test_model_options_include_configured_local_and_ollama_models(monkeypatch, tmp_path):
    from kestrel import model_store

    local = tmp_path / "local.gguf"
    local.write_bytes(b"gguf")
    monkeypatch.setattr(menu, "load_config", lambda: _config(default_model="configured.gguf"))
    monkeypatch.setattr(model_store, "default_models_dir", lambda: tmp_path)
    monkeypatch.setattr(model_store, "discover_local_models", lambda _root: [local])
    monkeypatch.setattr(model_store, "list_ollama_models", lambda: [SimpleNamespace(name="phi", size="2 GiB")])

    options = menu._MenuSession._model_options()

    assert options[0] == ("configured.gguf", "configured default")
    assert options[1][0] == str(local)
    assert options[2] == ("ollama://phi", "2 GiB via Ollama")


def test_model_pickers_cover_empty_cancel_keep_and_manual_paths(monkeypatch, capsys):
    session = menu._MenuSession()
    fake_ui = FakeUI(selections=[0, 1], answers=["manual.gguf"])
    monkeypatch.setattr(menu, "ui", fake_ui)
    monkeypatch.setattr(session, "_model_options", lambda: [])
    assert session._pick_model("Select") is None
    assert "No models found" in capsys.readouterr().out

    monkeypatch.setattr(session, "_model_options", lambda: [("configured.gguf", "default")])
    assert session._pick_model("Select", keep_current=True) == "<keep current>"
    assert session._pick_model("Select") == "manual.gguf"


def test_ollama_picker_handles_backend_error_empty_and_manual_name(monkeypatch, capsys):
    from kestrel import model_store

    session = menu._MenuSession()
    fake_ui = FakeUI(selections=[1], answers=["phi:latest"])
    monkeypatch.setattr(menu, "ui", fake_ui)
    monkeypatch.setattr(
        model_store, "list_ollama_models", lambda: (_ for _ in ()).throw(model_store.ModelStoreError("offline"))
    )
    assert session._pick_ollama_model() is None
    assert "offline" in capsys.readouterr().out

    monkeypatch.setattr(model_store, "list_ollama_models", lambda: [])
    assert session._pick_ollama_model() is None
    assert "No Ollama models" in capsys.readouterr().out

    monkeypatch.setattr(
        model_store,
        "list_ollama_models",
        lambda: [SimpleNamespace(name="phi", size="2 GiB", model_id="sha")],
    )
    assert session._pick_ollama_model() == "phi:latest"


def test_chat_explicit_target_and_warm_cache(monkeypatch):
    fake_ui = FakeUI(selections=[2], answers=["2048"], confirmations=[True, True])
    launches = []
    monkeypatch.setattr(menu, "ui", fake_ui)
    monkeypatch.setattr(menu, "load_config", lambda: _config())
    session = menu._MenuSession(launch=lambda *args: launches.append(args))

    session._chat()

    assert launches == [("chat", "model.gguf", "--ctx-size", "2048", "--target", "quality", "--warm-cache")]


def test_actions_dispatch_selection_import_pull_search_convert_and_configure(monkeypatch):
    fake_ui = FakeUI(
        answers=["OWNER/REPO", "file.safetensors", "query", "dir", "out.gguf", "2", "importance.json", "/models"],
        confirmations=[True, True],
    )
    launches = []
    monkeypatch.setattr(menu, "ui", fake_ui)
    session = menu._MenuSession(launch=lambda *args: launches.append(args))
    monkeypatch.setattr(session, "_pick_model", lambda *_args, **_kwargs: "selected.gguf")
    monkeypatch.setattr(session, "_pick_ollama_model", lambda: "phi")

    session._select_model()
    session._import_ollama()
    session._pull_hugging_face()
    session._search_models()
    session._convert_model()
    session._configure_default()
    session._configure_models_dir()

    assert launches == [
        ("setup", "--model", "selected.gguf"),
        ("models", "import", "ollama://phi", "--set-default"),
        ("models", "pull", "hf://OWNER/REPO", "--file", "file.safetensors", "--dry-run"),
        ("models", "pull", "hf://OWNER/REPO", "--file", "file.safetensors"),
        ("models", "search", "query"),
        ("convert", "dir", "-o", "out.gguf", "--experts-keep", "2", "--expert-importance", "importance.json"),
        ("setup", "--model", "selected.gguf"),
        ("setup", "--models-dir", "/models"),
    ]


def test_submenu_dispatchers_route_every_manage_and_configure_action(monkeypatch):
    launches = []
    session = menu._MenuSession(launch=lambda *args: launches.append(args))
    monkeypatch.setattr(menu, "ui", FakeUI())
    routed = []
    monkeypatch.setattr(session, "_search_models", lambda: routed.append("search"))
    monkeypatch.setattr(session, "_convert_model", lambda: routed.append("convert"))
    monkeypatch.setattr(session, "_configure_default", lambda: routed.append("default"))
    monkeypatch.setattr(session, "_configure_models_dir", lambda: routed.append("directory"))
    monkeypatch.setattr(session, "_import_ollama", lambda: routed.append("ollama"))
    monkeypatch.setattr(session, "_pull_hugging_face", lambda: routed.append("huggingface"))

    for index in range(5):
        monkeypatch.setattr(session, "_pick", lambda _items, _title, index=index: index)
        session._manage_models()
    monkeypatch.setattr(session, "_pick", lambda _items, _title: 0)
    session._configure()
    monkeypatch.setattr(session, "_pick", lambda _items, _title: 1)
    session._configure()
    monkeypatch.setattr(session, "_pick", lambda _items, _title: 0)
    session._import_models()
    monkeypatch.setattr(session, "_pick", lambda _items, _title: 1)
    session._import_models()

    assert launches == [("models", "list", "--resolve"), ("doctor",), ("benchmark",)]
    assert routed == ["search", "convert", "default", "directory", "ollama", "huggingface"]
