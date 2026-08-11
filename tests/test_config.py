import pytest

from kestrel.config import KestrelConfig, config_path, load_config, save_config


def test_config_roundtrip(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    cfg = KestrelConfig(
        default_model="ollama://qwen3.6:27b",
        models_dir="/tmp/models",
        llama_cpp_dir="/tmp/llama.cpp",
        context_size=8192,
        reasoning_level="high",
    )
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded == cfg


def test_empty_config_defaults(tmp_path):
    loaded = load_config(tmp_path / "nonexistent.toml")
    assert loaded == KestrelConfig()
    assert loaded.default_model is None


def test_load_accepts_str_path(tmp_path):
    path = tmp_path / "config.toml"
    save_config(KestrelConfig(default_model="x"), str(path))
    assert load_config(str(path)).default_model == "x"


def test_save_config_creates_parent_dir(tmp_path):
    nested = tmp_path / "a" / "b" / "config.toml"
    result = save_config(KestrelConfig(), str(nested))
    assert result.parent.is_dir()
    assert result.is_file()


def test_invalid_config_raises(tmp_path, monkeypatch):
    from kestrel.errors import ConfigError

    path = tmp_path / "bad.toml"
    path.write_text("not [valid toml")
    with pytest.raises(ConfigError):
        load_config(path)


@pytest.mark.parametrize(
    "text, message",
    [
        ("local = []", "local.*must be a table"),
        ("[local]\nmodels_dir = 42", "local.models_dir must be a string"),
        ("[local]\ncontext_size = 128", "local.context_size"),
        ("[local]\ncontext_size = true", "local.context_size"),
        ("[local]\nreasoning_level = 'extreme'", "local.reasoning_level"),
    ],
)
def test_invalid_config_schema_is_typed(tmp_path, text, message):
    from kestrel.errors import ConfigError

    path = tmp_path / "bad-schema.toml"
    path.write_text(text)
    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_config_path_env_override(tmp_path, monkeypatch):
    target = tmp_path / "custom" / "k.toml"
    monkeypatch.setenv("KESTREL_CONFIG", str(target))
    assert config_path() == target


def test_config_path_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("KESTREL_CONFIG", raising=False)
    assert config_path() == tmp_path / "kestrel" / "config.toml"
