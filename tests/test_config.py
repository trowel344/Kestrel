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
        kv_cache_turbo=True,
    )
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded == cfg


def test_config_kv_cache_turbo_roundtrip(tmp_path):
    path = tmp_path / "config.toml"
    save_config(KestrelConfig(kv_cache_turbo=True), path)
    assert load_config(path).kv_cache_turbo is True
    assert load_config(path) == KestrelConfig(kv_cache_turbo=True)


def test_config_kv_cache_turbo_defaults_false(tmp_path):
    assert KestrelConfig().kv_cache_turbo is False
    assert load_config(tmp_path / "nonexistent.toml").kv_cache_turbo is False


def test_config_placement_and_kv_defaults_roundtrip(tmp_path):
    path = tmp_path / "config.toml"
    save_config(
        KestrelConfig(kv_cache_type="q4_0", gpu_layers="24", cpu_moe="on", kv_cache_turbo=True),
        path,
    )
    loaded = load_config(path)
    assert loaded == KestrelConfig(kv_cache_type="q4_0", gpu_layers="24", cpu_moe="on", kv_cache_turbo=True)
    assert "kv_cache_type = \"q4_0\"" in path.read_text()
    assert "gpu_layers = \"24\"" in path.read_text()
    assert "cpu_moe = \"on\"" in path.read_text()


def test_config_rejects_invalid_new_fields(tmp_path):
    from kestrel.errors import ConfigError

    cases = {
        "kv_cache_type = \"q2_0\"": "kv_cache_type",
        "gpu_layers = \"banana\"": "gpu_layers",
        "gpu_layers = -4": "gpu_layers",
        "cpu_moe = \"sometimes\"": "cpu_moe",
    }
    for line, _match in cases.items():
        path = tmp_path / f"bad-{line.split()[0]}.toml"
        path.write_text("[local]\n" + line + "\n")
        with pytest.raises(ConfigError):
            load_config(path)


def test_config_auto_placement_defaults_omit_lines(tmp_path):
    path = tmp_path / "config.toml"
    save_config(KestrelConfig(), path)
    text = path.read_text()
    assert "kv_cache_type" not in text
    assert "gpu_layers" not in text
    assert "cpu_moe" not in text


def test_config_toml_integer_gpu_layers_roundtrip(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[local]\ngpu_layers = 24\n')
    assert load_config(path).gpu_layers == "24"


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
        ("[local]\nkv_cache_turbo = 'yes'", "local.kv_cache_turbo"),
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
