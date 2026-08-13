import json
import shlex
import sys

import pytest

from kestrel.integrations import (
    IntegrationError,
    IntegrationSpec,
    install_omp_provider,
    install_pi_provider,
    remove_omp_provider,
    remove_pi_provider,
    status_omp_provider,
    status_pi_provider,
)


def _spec(**changes):
    values = {"model_id": "qwen3.6:35B", "context_window": 32768, "max_tokens": 8192}
    values.update(changes)
    return IntegrationSpec(**values)


def test_pi_provider_preserves_other_providers_and_uses_token_command(tmp_path):
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"providers": {"other": {"baseUrl": "http://other/v1"}}, "extra": True}) + "\n")
    result = install_pi_provider(_spec(), path)
    payload = json.loads(path.read_text())
    assert result.changed and result.installed
    assert payload["extra"] is True
    assert payload["providers"]["other"] == {"baseUrl": "http://other/v1"}
    assert payload["providers"]["kestrel"]["api"] == "openai-completions"
    expected_auth = "!" + shlex.join((sys.executable, "-m", "kestrel", "agents", "token"))
    assert payload["providers"]["kestrel"]["apiKey"] == expected_auth
    assert payload["providers"]["kestrel"]["models"][0]["contextWindow"] == 32768


def test_pi_provider_is_idempotent_and_dry_run_does_not_write(tmp_path):
    path = tmp_path / "models.json"
    assert install_pi_provider(_spec(), path).changed
    original = path.read_text()
    assert not install_pi_provider(_spec(), path).changed
    dry = install_pi_provider(_spec(model_id="other"), path, dry_run=True)
    assert dry.changed and dry.dry_run
    assert path.read_text() == original


def test_pi_provider_rejects_invalid_json_and_symlink(tmp_path):
    path = tmp_path / "models.json"
    path.write_text("not json")
    with pytest.raises(IntegrationError, match="invalid Pi models JSON"):
        install_pi_provider(_spec(), path)
    target = tmp_path / "target.json"
    target.write_text("{}")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(IntegrationError, match="regular file"):
        install_pi_provider(_spec(), link)


def test_omp_provider_preserves_other_provider_and_default_role(tmp_path):
    path = tmp_path / "models.yml"
    path.write_text("providers:\n  other:\n    baseUrl: http://other/v1\nmodelRoles:\n  default: other/model\n")
    result = install_omp_provider(_spec(), path)
    text = path.read_text()
    assert result.changed and result.installed
    assert "  other:\n    baseUrl: http://other/v1\n" in text
    assert "  kestrel:\n" in text
    assert "  default: other/model\n" in text
    assert "api: openai-completions" in text
    expected_auth = json.dumps("!" + shlex.join((sys.executable, "-m", "kestrel", "agents", "token")))
    assert f"apiKey: {expected_auth}" in text
    assert "auth: none" not in text
    assert "contextWindow: 32768" in text

    remove_omp_provider(path)
    removed_text = path.read_text()
    assert "providers:\n  other:\n" in removed_text
    assert "providers: {}" not in removed_text
    assert "kestrel:" not in removed_text


def test_omp_provider_replace_remove_and_idempotence(tmp_path):
    path = tmp_path / "models.yml"
    assert install_omp_provider(_spec(), path).changed
    original = path.read_text()
    assert not install_omp_provider(_spec(), path).changed
    assert path.read_text() == original
    install_omp_provider(_spec(model_id="new-model", context_window=None), path)
    assert "new-model" in path.read_text() and "qwen3.6:35B" not in path.read_text()
    removed = remove_omp_provider(path)
    assert removed.changed and not removed.installed
    assert "kestrel:" not in path.read_text()


def test_omp_provider_rejects_malformed_and_nonregular_targets(tmp_path):
    path = tmp_path / "models.yml"
    path.write_text("providers: [bad]\n")
    with pytest.raises(IntegrationError, match="providers must be a mapping"):
        install_omp_provider(_spec(), path)
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(IntegrationError, match="regular file"):
        install_omp_provider(_spec(), directory)


def test_provider_status_and_remove_missing_are_safe(tmp_path):
    pi = tmp_path / "pi.json"
    omp = tmp_path / "omp.yml"
    assert not status_pi_provider(pi).installed
    assert not status_omp_provider(omp).installed
    assert not remove_pi_provider(pi).changed
    assert not remove_omp_provider(omp).changed


def test_pi_refuses_preexisting_unmanaged_kestrel_provider(tmp_path):
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"providers": {"kestrel": {"baseUrl": "http://user-owned/v1"}}}))

    with pytest.raises(IntegrationError, match="unmanaged"):
        install_pi_provider(_spec(), path)
    with pytest.raises(IntegrationError, match="unmanaged"):
        remove_pi_provider(path)

    assert "user-owned" in path.read_text()


def test_omp_refuses_modified_owned_provider(tmp_path):
    path = tmp_path / "models.yml"
    install_omp_provider(_spec(), path)
    path.write_text(path.read_text().replace("qwen3.6:35B", "user-edit", 1))

    with pytest.raises(IntegrationError, match="unmanaged"):
        install_omp_provider(_spec(), path)
    with pytest.raises(IntegrationError, match="unmanaged"):
        remove_omp_provider(path)


def test_omp_remove_reinstall_has_one_provider_key(tmp_path):
    path = tmp_path / "models.yml"
    install_omp_provider(_spec(), path)
    remove_omp_provider(path)
    install_omp_provider(_spec(), path)

    assert sum(line.startswith("providers:") for line in path.read_text().splitlines()) == 1
    assert status_omp_provider(path).installed


def test_omp_rejects_duplicate_provider_roots(tmp_path):
    path = tmp_path / "models.yml"
    path.write_text("providers:\n  first: {}\nproviders:\n  second: {}\n", encoding="utf-8")

    with pytest.raises(IntegrationError, match="duplicate top-level providers"):
        install_omp_provider(_spec(), path)
