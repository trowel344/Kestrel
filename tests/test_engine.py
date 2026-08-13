import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))


def _git(repo, *args, check=True):
    subprocess.run(["git", "-C", str(repo), *args], check=check, capture_output=True)


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


def test_status_fetches_remote_object_and_reports_behind(upstream):
    import kestrel.engine as engine

    engine.adopt(str(upstream["engine"]))
    _advance(upstream)

    status = engine.engine_status(str(upstream["engine"]), check_remote=True)

    assert status["behind"] == 1
    assert status["ahead"] == 0
    assert status["stale"] is True
    assert status["remote_head"] != status["commit"]


def test_ls_remote_head_does_not_treat_url_as_working_directory(monkeypatch):
    import kestrel.engine as engine

    seen = []

    def fake_run(command, **kwargs):
        seen.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="abc123\trefs/heads/main\n", stderr="")

    monkeypatch.setattr(engine.subprocess, "run", fake_run)

    assert engine.ls_remote_head("https://example.invalid/repo.git", "main") == "abc123"
    assert seen[0][0] == [
        "git",
        "ls-remote",
        "https://example.invalid/repo.git",
        "refs/heads/main",
    ]


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


def test_update_does_not_claim_current_when_revision_compare_fails(upstream, monkeypatch):
    import kestrel.engine as engine

    engine.adopt(str(upstream["engine"]))
    monkeypatch.setattr(
        engine,
        "_resolve_target",
        lambda *args: ("main", "f" * 40),
    )
    monkeypatch.setattr(engine, "_rev_counts", lambda *args, **kwargs: None)

    with pytest.raises(engine.EngineError, match="cannot compare"):
        engine.update(str(upstream["engine"]))


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


def test_snapshot_and_rollback_round_trip(upstream):
    import kestrel.engine as engine

    engine.adopt(str(upstream["engine"]))
    build_bin = upstream["engine"] / "build" / "bin"
    build_bin.mkdir(parents=True, exist_ok=True)
    (build_bin / "llama-cli").write_bytes(b"old-good")
    (build_bin / "llama-cli").chmod(0o755)

    snap = engine._snapshot_previous(
        str(upstream["engine"]), ["llama-cli", "rpc-server"], commit="a" * 40, last_good="b" * 40
    )
    assert snap is not None
    assert "llama-cli" in snap["artifacts"]
    assert snap["artifacts"]["llama-cli"]["size"] == len(b"old-good")

    prev = upstream["engine"] / ".kestrel-engine.previous" / "llama-cli"
    assert prev.read_bytes() == b"old-good"
    assert prev.stat().st_mode & 0o777 == 0o755

    (build_bin / "llama-cli").write_bytes(b"broken-build")
    restored = engine._rollback(str(upstream["engine"]), ["llama-cli"])
    assert restored["restored"] is True
    assert (build_bin / "llama-cli").read_bytes() == b"old-good"


def test_rebuild_dry_run_returns_planned(upstream):
    import kestrel.engine as engine

    engine.adopt(str(upstream["engine"]))
    result = engine.rebuild(str(upstream["engine"]), dry_run=True)
    assert result["status"] == "dry_run"
    assert "llama-cli" in result["targets"]


def test_parse_bench_speed_extracts_generation_rate():
    import kestrel.engine as engine

    sample = (
        '[{"backend":"CPU","model":"m","n_prompt":16,"n_gen":32,"tg16":"12.34","tg32":"42.56 +- 0.90","pp16":"180.00"}]'
    )
    assert engine._parse_bench_speed(sample) == 42.56
    assert engine._parse_bench_speed("no numbers here") is None


def test_bench_engine_returns_none_when_binary_missing(tmp_path):
    import kestrel.engine as engine

    assert engine._bench_engine(str(tmp_path), "/models/model.gguf") is None


def test_bench_delta_reports_regression():
    import kestrel.engine as engine

    delta = engine._bench_delta(100.0, 88.0, "/models/model.gguf")
    assert delta["delta_pct"] == -12.0
    assert delta["regressed"] is True
    assert engine._bench_delta(88.0, 100.0, "/models/model.gguf")["regressed"] is False
    assert engine._bench_delta(None, 100.0, "/models/model.gguf") is None


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
    output = capsys.readouterr().out
    assert output.count("\n") == 1
    payload = json.loads(output)
    assert payload["git"] is True
    assert payload["manifest"] is not None


def test_cmd_build_uses_transactional_rebuild_json(capsys, monkeypatch, tmp_path):
    from types import SimpleNamespace

    import kestrel.cli as cli
    import kestrel.engine as engine

    monkeypatch.setattr(
        engine,
        "rebuild",
        lambda directory, dry_run: {"status": "dry_run", "directory": directory},
    )

    cli.cmd_build(SimpleNamespace(dir=str(tmp_path), dry_run=True, json=True))

    output = capsys.readouterr().out
    assert output.count("\n") == 1
    assert json.loads(output) == {"status": "dry_run", "directory": str(tmp_path)}


def test_self_update_dry_run_reports(capsys, monkeypatch):
    from types import SimpleNamespace

    import kestrel.cli as cli

    args = SimpleNamespace(
        repo=str(Path(__file__).resolve().parents[1]), dry_run=True, json=False, yes=False, wheel=None, sha256=None
    )
    cli.cmd_self_update(args)
    out = capsys.readouterr().out
    assert "Would install Kestrel" in out
