import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))


def _git(repo, *args, check=True):
    subprocess.run(
        ["git", "-C", str(repo), *args], check=check, capture_output=True
    )


@pytest.fixture
def upstream(tmp_path):
    """A bare remote and a local checkout 'engine', with 'one' committed."""
    bare = tmp_path / "upstream.git"
    subprocess.run(["git", "init", "--bare", str(bare)], check=True, capture_output=True)

    engine = tmp_path / "engine"
    engine.mkdir()
    _git(engine, "init", "-b", "main")
    _git(engine, "config", "user.email", "t@e")
    _git(engine, "config", "user.name", "t")
    (engine / "model.txt").write_text("one")
    _git(engine, "add", ".")
    _git(engine, "commit", "-m", "one")
    _git(engine, "remote", "add", "origin", str(bare))
    _git(engine, "push", "-u", "origin", "main")
    subprocess.run(
        ["git", "--git-dir", str(bare), "symbolic-ref", "HEAD", "refs/heads/main"],
        check=True,
        capture_output=True,
    )
    return {"bare": bare, "engine": engine}


def _advance(upstream):
    """Add 'two' to the upstream and push."""
    clone = upstream["bare"].parent / "tmp-clone"
    subprocess.run(["git", "clone", str(upstream["bare"]), str(clone)], check=True, capture_output=True)
    _git(clone, "config", "user.email", "t@e")
    _git(clone, "config", "user.name", "t")
    (clone / "model.txt").write_text("two")
    _git(clone, "add", ".")
    _git(clone, "commit", "-m", "two")
    _git(clone, "push", "origin", "HEAD:main")


def _mock_cmake(monkeypatch):
    import subprocess as sp

    real_run = sp.run

    def fake_run(cmd, **kwargs):
        if cmd and cmd[0] == "cmake":
            return sp.CompletedProcess(cmd, 0, stdout="", stderr="")
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(sp, "run", fake_run)


def test_adopt_creates_managed_manifest(upstream):
    import kestrel.engine as engine

    manifest = engine.adopt(str(upstream["engine"]))
    assert manifest.remote == str(upstream["bare"])
    assert manifest.branch == "main"

    status = engine.engine_status(str(upstream["engine"]), check_remote=False)
    assert status["git"] is True
    assert status["manifest"] is not None
    assert status["commit"]


def test_update_reports_and_applies_when_behind(upstream, monkeypatch):
    import kestrel.engine as engine

    engine.adopt(str(upstream["engine"]))
    _advance(upstream)
    _mock_cmake(monkeypatch)

    dry = engine.update(str(upstream["engine"]), dry_run=True)
    assert dry["status"] == "update_available"
    assert dry["behind"] == 1

    result = engine.update(str(upstream["engine"]))
    assert result["status"] == "rebuilt"
    assert (upstream["engine"] / "model.txt").read_text() == "two"


def test_update_refuses_dirty_without_force(upstream, monkeypatch):
    import kestrel.engine as engine

    engine.adopt(str(upstream["engine"]))
    _advance(upstream)
    _mock_cmake(monkeypatch)
    (upstream["engine"] / "local-edit.txt").write_text("x")

    with pytest.raises(engine.EngineError):
        engine.update(str(upstream["engine"]))
    assert (upstream["engine"] / "local-edit.txt").exists()

    engine.update(str(upstream["engine"]), force=True)
    assert not (upstream["engine"] / "local-edit.txt").exists()
    assert (upstream["engine"] / "model.txt").read_text() == "two"


def test_update_refuses_local_ahead_without_force(upstream):
    import kestrel.engine as engine

    engine.adopt(str(upstream["engine"]))
    (upstream["engine"] / "model.txt").write_text("local-only")
    _git(upstream["engine"], "add", ".")
    _git(upstream["engine"], "commit", "-m", "local")
    _advance(upstream)

    with pytest.raises(engine.EngineError):
        engine.update(str(upstream["engine"]))


def test_update_up_to_date(upstream, monkeypatch):
    import kestrel.engine as engine

    engine.adopt(str(upstream["engine"]))
    _mock_cmake(monkeypatch)
    result = engine.update(str(upstream["engine"]))
    assert result["status"] == "up_to_date"


def test_rebuild_writes_artifacts(upstream, monkeypatch):
    import kestrel.engine as engine

    engine.adopt(str(upstream["engine"]))
    _mock_cmake(monkeypatch)
    build_bin = upstream["engine"] / "build" / "bin"
    (build_bin / "llama-cli").parent.mkdir(parents=True, exist_ok=True)
    (build_bin / "llama-cli").write_bytes(b"x" * 10)

    result = engine.rebuild(str(upstream["engine"]))
    assert result["status"] == "rebuilt"
    manifest = engine.load_manifest(str(upstream["engine"]))
    assert manifest.artifacts["llama-cli"]["size"] == 10
    assert manifest.built_at


def test_rebuild_dry_run_returns_planned(upstream):
    import kestrel.engine as engine

    engine.adopt(str(upstream["engine"]))
    result = engine.rebuild(str(upstream["engine"]), dry_run=True)
    assert result["status"] == "dry_run"
    assert "llama-cli" in result["targets"]


def test_matches_too_old_signature():
    import kestrel.engine as engine

    assert engine.matches_too_old_signature(
        "key qwen35moe.rope.dimension_sections has wrong array length; expected 4, got 3"
    )
    assert engine.matches_too_old_signature("error loading model hyperparameters")
    assert not engine.matches_too_old_signature("model loaded successfully")


def test_cmd_engine_status_json(capsys, monkeypatch, upstream):
    from types import SimpleNamespace

    import kestrel.cli as cli
    import kestrel.engine as engine

    engine.adopt(str(upstream["engine"]))
    args = SimpleNamespace(engine_command="status", dir=str(upstream["engine"]), json=True, no_remote=True)
    cli.cmd_engine(args)
    payload = json.loads(capsys.readouterr().out)
    assert payload["git"] is True
    assert payload["manifest"] is not None


def test_self_update_dry_run_reports(capsys, monkeypatch):
    from types import SimpleNamespace

    import kestrel.cli as cli

    args = SimpleNamespace(repo=str(Path(__file__).resolve().parents[1]), dry_run=True, json=False, yes=False, wheel=None, sha256=None)
    cli.cmd_self_update(args)
    out = capsys.readouterr().out
    assert "Would install Kestrel" in out
