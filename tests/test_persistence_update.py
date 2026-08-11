"""Focused persistence and self-update safety checks."""

from __future__ import annotations

import importlib.metadata
import sys
import urllib.request
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from kestrel import config
from kestrel.cli import updater
from kestrel.errors import InputError, IntegrityError


def test_save_config_uses_atomic_writer(monkeypatch, tmp_path):
    calls = []

    def fake_write(path, payload, **kwargs):
        calls.append((Path(path), payload, kwargs))
        return Path(path)

    monkeypatch.setattr(config, "write_atomic", fake_write)
    target = tmp_path / "nested" / "config.toml"
    result = config.save_config(config.KestrelConfig(default_model="m"), target)
    assert result == target
    assert calls and calls[0][0] == target
    assert 'default_model = "m"\n' in calls[0][1]
    assert calls[0][1].endswith('reasoning_level = "auto"\n')


def test_planner_cache_write_is_atomic(monkeypatch, tmp_path):
    import kestrel.gguf.metadata as metadata

    calls = []
    monkeypatch.setenv("KESTREL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(
        metadata,
        "write_atomic",
        lambda path, payload, **kwargs: calls.append((Path(path), payload, kwargs)),
    )
    metadata._planner_cache_write(tmp_path / "model.gguf", 12, 34, {"n_layer": 1})
    assert calls and calls[0][0].name.startswith("planner-")
    assert calls[0][2] == {"backup": False}


def test_hf_manifest_write_is_atomic(monkeypatch, tmp_path):
    import kestrel.model_store as model_store

    destination = tmp_path / "model"
    fake_process = SimpleNamespace(stdout="{}", stderr="", returncode=0)
    monkeypatch.setattr(model_store, "_hf_cli", lambda: "/usr/bin/hf")
    monkeypatch.setattr(model_store, "_run", lambda command, **kwargs: fake_process)
    calls = []
    monkeypatch.setattr(
        model_store,
        "write_atomic",
        lambda path, payload, **kwargs: calls.append((Path(path), payload, kwargs)),
    )
    model_store.pull_huggingface("owner/repo", destination=destination)
    assert calls and calls[0][0] == destination / ".kestrel-source.json"


def _wheel(path: Path, name: str = "kestrel", version: str = "1.5.0") -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"{name}-1.5.0.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        )
        archive.writestr(f"{name}-1.5.0.dist-info/WHEEL", "Wheel-Version: 1.0\n")
    return path


def test_wheel_identity_requires_kestrel_distribution(tmp_path):
    good = _wheel(tmp_path / "good.whl")
    assert updater._validate_wheel_identity(good) == "1.5.0"
    bad = _wheel(tmp_path / "bad.whl", name="unrelated")
    with pytest.raises(IntegrityError, match="not a valid Kestrel"):
        updater._validate_wheel_identity(bad)


def test_remote_wheels_require_https_and_checksum(monkeypatch, tmp_path):
    args = SimpleNamespace(
        repo=None,
        wheel="http://example.invalid/kestrel.whl",
        sha256=None,
        dry_run=True,
        json=False,
        yes=False,
    )
    with pytest.raises(IntegrityError, match="SHA256 checksum"):
        updater.cmd_self_update(args)

    with pytest.raises(IntegrityError, match="HTTPS"):
        updater._materialize_wheel("http://example.invalid/kestrel.whl")

    missing = SimpleNamespace(
        repo=None,
        wheel=str(tmp_path / "missing.whl"),
        sha256="0" * 64,
        dry_run=True,
        json=True,
        yes=False,
    )
    with pytest.raises(IntegrityError, match="unable to read update artifact"):
        updater.cmd_self_update(missing)


def test_repo_identity_rejects_unrelated_project(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "other"\nversion = "1.0.0"\n')
    with pytest.raises(InputError, match="not a versioned Kestrel"):
        updater._validate_repo_identity(tmp_path)


def test_remote_wheel_size_is_bounded(monkeypatch):
    class Response:
        headers = {"Content-Length": str(updater.MAX_WHEEL_BYTES + 1)}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def geturl(self):
            return "https://example.invalid/kestrel.whl"

    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: Response())
    with pytest.raises(IntegrityError, match="exceeds"):
        updater._materialize_wheel("https://example.invalid/kestrel.whl")


def test_post_install_timeout_is_typed(monkeypatch):
    def timeout(*args, **kwargs):
        raise updater.subprocess.TimeoutExpired(args[0], 30)

    monkeypatch.setattr(updater.subprocess, "run", timeout)
    with pytest.raises(IntegrityError, match="timed out"):
        updater._post_install_check()


def test_post_install_spawn_error_is_typed(monkeypatch):
    monkeypatch.setattr(
        updater.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("denied"))
    )
    with pytest.raises(IntegrityError, match="unable to start"):
        updater._post_install_check()


def test_pip_spawn_error_attempts_rollback(monkeypatch, tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "kestrel"\nversion = "1.5.0"\n')
    root = tmp_path / "snapshot"
    root.mkdir()
    calls = {"restore": 0}
    monkeypatch.setattr(updater, "_snapshot_installed", lambda: {"root": root})
    monkeypatch.setattr(updater, "_restore_install", lambda snapshot: calls.update(restore=1) or True)
    monkeypatch.setattr(updater, "_post_install_check", lambda: (True, "1.5.0"))
    monkeypatch.setattr(
        updater.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("denied"))
    )
    args = SimpleNamespace(repo=str(tmp_path), wheel=None, sha256=None, dry_run=False, json=True, yes=False)
    with pytest.raises(IntegrityError, match="unable to start pip"):
        updater.cmd_self_update(args)
    assert calls["restore"] == 1


def test_snapshot_cleanup_removes_temporary_root(tmp_path):
    root = tmp_path / "snapshot"
    root.mkdir()
    (root / "data").write_text("x")
    updater._cleanup_snapshot({"root": root})
    assert not root.exists()


def test_snapshot_copies_installed_package_metadata_and_script(monkeypatch, tmp_path):
    site = tmp_path / "site-packages"
    package = site / "kestrel"
    dist_info = site / "kestrel-1.4.0.dist-info"
    package.mkdir(parents=True)
    dist_info.mkdir()
    (package / "__init__.py").write_text("old package")
    (dist_info / "METADATA").write_text("Name: kestrel\nVersion: 1.4.0\n")
    script = tmp_path / "bin" / "kestrel"
    script.parent.mkdir()
    script.write_text("old script")
    fake_module = SimpleNamespace(__file__=str(package / "__init__.py"))
    monkeypatch.setitem(sys.modules, "kestrel", fake_module)
    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        lambda name: SimpleNamespace(_path=dist_info),
    )
    monkeypatch.setattr(updater, "_console_script_candidates", lambda: [script])

    snapshot = updater._snapshot_installed()
    assert snapshot is not None
    (package / "__init__.py").write_text("new package")
    (dist_info / "METADATA").write_text("Name: kestrel\nVersion: 1.5.0\n")
    new_dist = site / "kestrel-1.5.0.dist-info"
    new_dist.mkdir()
    (new_dist / "METADATA").write_text("new metadata")
    script.write_text("new script")

    assert updater._restore_install(snapshot) is True
    assert (package / "__init__.py").read_text() == "old package"
    assert (dist_info / "METADATA").read_text().endswith("1.4.0\n")
    assert not new_dist.exists()
    assert script.read_text() == "old script"
    updater._cleanup_snapshot(snapshot)


def test_editable_snapshot_preserves_metadata_without_touching_source(monkeypatch, tmp_path):
    repo = tmp_path / "checkout"
    package = repo / "kestrel"
    egg_info = repo / "kestrel.egg-info"
    package.mkdir(parents=True)
    egg_info.mkdir()
    (repo / ".git").mkdir()
    (repo / "pyproject.toml").write_text('[project]\nname = "kestrel"\nversion = "1.5.0"\n')
    (package / "__init__.py").write_text("source old")
    (egg_info / "PKG-INFO").write_text("old metadata")
    script = tmp_path / "bin" / "kestrel"
    script.parent.mkdir()
    script.write_text("old script")
    editable_pth = repo / "site-packages" / "__editable__.kestrel-1.5.0.pth"
    editable_finder = repo / "site-packages" / "__editable___kestrel_1_5_0_finder.py"
    editable_pth.parent.mkdir()
    editable_pth.write_text("old pth")
    editable_finder.write_text("old finder")
    monkeypatch.setitem(sys.modules, "kestrel", SimpleNamespace(__file__=str(package / "__init__.py")))
    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        lambda name: SimpleNamespace(_path=egg_info),
    )
    monkeypatch.setattr(updater, "_console_script_candidates", lambda: [script])
    monkeypatch.setattr(
        updater.sysconfig, "get_path", lambda key: str(editable_pth.parent) if key == "purelib" else str(script.parent)
    )

    snapshot = updater._snapshot_installed()
    assert snapshot is not None
    (package / "__init__.py").write_text("source changed")
    (egg_info / "PKG-INFO").write_text("new metadata")
    script.write_text("new script")
    editable_pth.write_text("new pth")
    editable_finder.write_text("new finder")
    replacement_package = editable_pth.parent / "kestrel"
    replacement_dist = editable_pth.parent / "kestrel-1.5.0.dist-info"
    replacement_package.mkdir()
    replacement_dist.mkdir()
    assert updater._restore_install(snapshot) is True
    assert (package / "__init__.py").read_text() == "source changed"
    assert (egg_info / "PKG-INFO").read_text() == "old metadata"
    assert script.read_text() == "old script"
    assert editable_pth.read_text() == "old pth"
    assert editable_finder.read_text() == "old finder"
    assert not replacement_package.exists()
    assert not replacement_dist.exists()
    updater._cleanup_snapshot(snapshot)


def test_failed_pip_install_is_typed_and_attempts_rollback(monkeypatch, tmp_path):
    calls = {"restore": 0}
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "kestrel"\nversion = "1.5.0"\n')
    monkeypatch.setattr(updater, "_snapshot_installed", lambda: {"root": tmp_path / "snapshot"})
    monkeypatch.setattr(updater, "_restore_install", lambda snapshot: calls.update(restore=1) or True)
    monkeypatch.setattr(updater, "_post_install_check", lambda: (True, "1.4.0"))
    monkeypatch.setattr(
        updater.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=7, stdout="", stderr="pip broke"),
    )
    args = SimpleNamespace(
        repo=str(tmp_path),
        wheel=None,
        sha256=None,
        dry_run=False,
        json=True,
        yes=False,
    )
    with pytest.raises(IntegrityError, match="succeeded rollback"):
        updater.cmd_self_update(args)
    assert calls["restore"] == 1


def test_failed_rollback_retains_snapshot_for_manual_recovery(monkeypatch, tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "kestrel"\nversion = "1.5.0"\n')
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()
    monkeypatch.setattr(updater, "_snapshot_installed", lambda: {"root": snapshot_root})
    monkeypatch.setattr(updater, "_restore_install", lambda snapshot: False)
    monkeypatch.setattr(updater, "_post_install_check", lambda: (False, "1.6.0"))
    monkeypatch.setattr(
        updater.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    args = SimpleNamespace(
        repo=str(tmp_path),
        wheel=None,
        sha256=None,
        dry_run=False,
        json=True,
        yes=False,
    )
    with pytest.raises(IntegrityError) as exc_info:
        updater.cmd_self_update(args)
    assert "snapshot retained" in (exc_info.value.hint or "")
    assert snapshot_root.exists()


def test_post_install_check_isolated_and_requires_matching_cli(monkeypatch, tmp_path):
    executable = tmp_path / "kestrel"
    executable.write_text("script")
    monkeypatch.setattr(updater, "_console_script_candidates", lambda: [executable])
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[0] == sys.executable:
            return SimpleNamespace(returncode=0, stdout="1.5.0\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="kestrel 1.4.0\n", stderr="")

    monkeypatch.setattr(updater.subprocess, "run", fake_run)
    assert updater._post_install_check() == (False, "1.5.0")
    assert calls[0][1]["cwd"] != str(Path.cwd())
    assert "PYTHONPATH" not in calls[0][1]["env"]
