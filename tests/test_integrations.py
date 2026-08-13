import json
import shlex
import sys
import tomllib

import pytest

from kestrel.integrations import (
    IntegrationError,
    launch_metadata,
    remove_agent_integration,
    render_claude_settings,
    render_codex_config,
    render_opencode_config,
    setup_agent_integration,
    status_agent_integration,
)


def test_codex_profile_is_responses_only_and_command_authenticated():
    payload = tomllib.loads(
        render_codex_config(
            model="qwen3.5:35b",
            endpoint="http://127.0.0.1:8080",
            context_size=32768,
            reasoning="high",
        )
    )

    assert payload["model"] == "qwen3.5:35b"
    assert payload["model_provider"] == "kestrel"
    assert payload["model_context_window"] == 32768
    assert payload["model_auto_compact_token_limit"] == 26214
    assert payload["model_reasoning_effort"] == "high"
    provider = payload["model_providers"]["kestrel"]
    assert provider["base_url"] == "http://127.0.0.1:8080/v1"
    assert provider["wire_api"] == "responses"
    assert payload["features"]["remote_models"] is False
    assert provider["auth"]["command"] == sys.executable
    assert provider["auth"]["args"] == ["-m", "kestrel", "agents", "token"]


def test_claude_settings_uses_anthropic_root_and_helper():
    payload = json.loads(render_claude_settings(model="local-model", endpoint="http://127.0.0.1:8080/v1"))

    assert payload["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8080"
    assert payload["env"]["ANTHROPIC_MODEL"] == "local-model"
    assert payload["env"]["ANTHROPIC_CUSTOM_MODEL_OPTION"] == "local-model"
    assert payload["env"]["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert payload["apiKeyHelper"] == shlex.join((sys.executable, "-m", "kestrel", "agents", "token"))


def test_opencode_overlay_uses_chat_completions_and_env_key():
    rendered = render_opencode_config(
        model="local-model",
        endpoint="http://127.0.0.1:8080",
        context_size=16384,
    )
    payload = json.loads(rendered.split("\n", 1)[1])
    provider = payload["provider"]["kestrel"]
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"]["baseURL"] == "http://127.0.0.1:8080/v1"
    assert provider["options"]["apiKey"] == "{env:KESTREL_API_KEY}"
    assert provider["models"]["local-model"]["limit"]["context"] == 16384
    assert provider["models"]["local-model"]["limit"]["output"] == 8192


def test_setup_status_and_remove_are_scoped_to_owned_files(tmp_path):
    metadata = setup_agent_integration(
        "codex",
        model="local-model",
        endpoint="http://127.0.0.1:8080",
        context_size=4096,
        home=tmp_path,
        codex_home=tmp_path / "codex-home",
    )
    assert metadata.config_path == tmp_path / "codex-home" / "kestrel.config.toml"
    assert metadata.command == ("codex", "--profile", "kestrel")
    assert metadata.environment["CODEX_HOME"] == str(tmp_path / "codex-home")
    assert status_agent_integration("codex", home=tmp_path, codex_home=tmp_path / "codex-home").configured

    assert remove_agent_integration("codex", home=tmp_path, codex_home=tmp_path / "codex-home")
    assert not status_agent_integration("codex", home=tmp_path, codex_home=tmp_path / "codex-home").configured
    assert not remove_agent_integration("codex", home=tmp_path, codex_home=tmp_path / "codex-home")


def test_setup_refuses_to_overwrite_unmanaged_client_file(tmp_path):
    target = tmp_path / ".config" / "kestrel" / "integrations" / "opencode.jsonc"
    target.parent.mkdir(parents=True)
    target.write_text('{"provider": {"other": {}}}\n', encoding="utf-8")

    with pytest.raises(IntegrationError, match="unmanaged"):
        setup_agent_integration("opencode", model="model", home=tmp_path)

    assert "other" in target.read_text(encoding="utf-8")


def test_launch_metadata_is_reversible_and_client_specific(tmp_path):
    claude = launch_metadata("claude", model="model", home=tmp_path)
    assert claude.command == ("claude", "--settings", str(claude.config_path))
    assert claude.environment == {}
    assert claude.endpoint == "http://127.0.0.1:8080"

    opencode = launch_metadata("opencode", model="model", home=tmp_path)
    assert opencode.command == ("opencode", "--model", "kestrel/model")
    assert opencode.environment["OPENCODE_CONFIG"] == str(opencode.config_path)
    assert "KESTREL_API_KEY" not in opencode.environment


@pytest.mark.parametrize(("client", "filename"), [("pi", "models.json"), ("omp", "models.yml")])
def test_pi_agent_directory_override_is_honored(monkeypatch, tmp_path, client, filename):
    agent_dir = tmp_path / "custom-agent"
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent_dir))

    metadata = launch_metadata(client, model="model")

    assert metadata.config_path == agent_dir / filename


@pytest.mark.parametrize("client", ["pi", "omp"])
def test_setup_status_remove_supports_pi_and_omp(tmp_path, client):
    metadata = setup_agent_integration(
        client,
        model="local-model",
        endpoint="http://127.0.0.1:8080",
        context_size=4096,
        max_tokens=2048,
        home=tmp_path,
    )
    assert metadata.max_tokens == 2048
    assert metadata.config_path.exists()
    assert status_agent_integration(client, home=tmp_path).configured
    assert remove_agent_integration(client, home=tmp_path)
    assert not status_agent_integration(client, home=tmp_path).configured


def test_launch_metadata_rejects_invalid_max_tokens(tmp_path):
    with pytest.raises(IntegrationError, match="max_tokens"):
        launch_metadata("pi", model="model", max_tokens=0, home=tmp_path)


@pytest.mark.parametrize(
    ("client", "endpoint"),
    [
        ("codex", "http://user:secret@127.0.0.1:8080"),
        ("claude", "http://127.0.0.1:8080?secret=yes"),
        ("opencode", "ftp://127.0.0.1:8080"),
    ],
)
def test_rejects_unsafe_endpoints(client, endpoint):
    with pytest.raises(IntegrationError):
        launch_metadata(client, model="model", endpoint=endpoint)


def test_rejects_unknown_client_and_invalid_reasoning():
    with pytest.raises(IntegrationError, match="unsupported"):
        launch_metadata("unknown", model="model")
    with pytest.raises(IntegrationError, match="reasoning"):
        launch_metadata("codex", model="model", reasoning="turbo")
