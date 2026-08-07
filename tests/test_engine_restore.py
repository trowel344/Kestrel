"""Crash-safety of ``engine rebuild``/``update``: last-good snapshot & rollback."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

GOOD = b"GOOD_BINARY"


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
    (engine / ".gitignore").write_text("build/\n")
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


def _mock_cmake(monkeypatch, *, configure_code=0, build_code=0, build_writes=None):
    import subprocess as sp

    real_run = sp.run

    def fake_run(cmd, **kwargs):
        if cmd and cmd[0] == "cmake":
            if "--build" in cmd:
                if build_writes is not None:
                    build_writes()
                return sp.CompletedProcess(cmd, build_code, stdout="", stderr="")
            return sp.CompletedProcess(cmd, configure_code, stdout="", stderr="")
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(sp, "run", fake_run)


def _install_binary(engine_dir, content=GOOD):
    build_bin = Path(engine_dir) / "build" / "bin"
    build_bin.mkdir(parents=True, exist_ok=True)
    (build_bin / "llama-cli").write_bytes(content)
    return build_bin


def _previous_json(engine_dir):
    return Path(engine_dir) / ".kestrel-engine.previous.json"


def test_rebuild_build_failure_rolls_back(upstream, monkeypatch):
    import kestrel.engine as engine

    engine_dir = str(upstream["engine"])
    engine.adopt(engine_dir)
    build_bin = _install_binary(engine_dir)

    # Establish a last-good snapshot via a successful build.
    _mock_cmake(monkeypatch, build_code=0)
    engine.rebuild(engine_dir)

    # The next build clobbers the binary and fails; the snapshot must win.
    def clobber():
        (build_bin / "llama-cli").write_bytes(b"garbage")

    _mock_cmake(monkeypatch, build_code=1, build_writes=clobber)
    with pytest.raises(engine.EngineError) as exc:
        engine.rebuild(engine_dir)

    assert exc.value.restored_from_previous is True
    assert "restored last-good binaries" in str(exc.value)
    assert (build_bin / "llama-cli").read_bytes() == GOOD


def test_rebuild_success_rotates_snapshot(upstream, monkeypatch):
    import kestrel.engine as engine

    engine_dir = str(upstream["engine"])
    engine.adopt(engine_dir)
    _install_binary(engine_dir, content=b"old")

    def write_new():
        build_bin = Path(engine_dir) / "build" / "bin"
        (build_bin / "llama-cli").write_bytes(b"new_good")

    _mock_cmake(monkeypatch, build_writes=write_new)
    result = engine.rebuild(engine_dir)

    assert result["status"] == "rebuilt"
    assert result["restored_from_previous"] is False
    sidecar = json.loads(_previous_json(engine_dir).read_text())
    assert sidecar["artifacts"]["llama-cli"]["size"] == len(b"new_good")
    manifest = engine.load_manifest(engine_dir)
    assert manifest.artifacts["llama-cli"]["size"] == len(b"new_good")


def test_rebuild_smoke_failure_rolls_back(upstream, monkeypatch):
    import kestrel.engine as engine

    engine_dir = str(upstream["engine"])
    engine.adopt(engine_dir)
    build_bin = _install_binary(engine_dir)

    # Establish a last-good snapshot; the binary is not runnable, so the smoke
    # check simply has nothing to run and the rebuild succeeds.
    _mock_cmake(monkeypatch, build_code=0)
    engine.rebuild(engine_dir)

    # A "successful" build that emits a broken executable must roll back.
    def write_broken():
        script = "#!/bin/sh\nexit 1\n"
        (build_bin / "llama-cli").write_text(script)
        (build_bin / "llama-cli").chmod(0o755)

    _mock_cmake(monkeypatch, build_writes=write_broken)
    with pytest.raises(engine.EngineError) as exc:
        engine.rebuild(engine_dir)

    assert exc.value.restored_from_previous is True
    assert "smoke test failed" in str(exc.value)
    assert (build_bin / "llama-cli").read_bytes() == GOOD


def test_update_failure_after_reset_rolls_back_and_notes_ahead(upstream, monkeypatch):
    import kestrel.engine as engine

    engine_dir = str(upstream["engine"])
    engine.adopt(engine_dir)
    build_bin = _install_binary(engine_dir)

    # Establish a last-good snapshot at the current revision.
    _mock_cmake(monkeypatch, build_code=0)
    engine.rebuild(engine_dir)

    _advance(upstream)

    # The fetch/reset moves HEAD forward, but the rebuild fails: roll back the
    # artifacts while noting the working tree is now ahead of them.
    _mock_cmake(monkeypatch, build_code=1)
    with pytest.raises(engine.EngineError) as exc:
        engine.update(engine_dir)

    assert exc.value.restored_from_previous is True
    assert "artifacts were rolled back" in str(exc.value)
    assert (upstream["engine"] / "model.txt").read_text() == "two"
    assert (build_bin / "llama-cli").read_bytes() == GOOD


def test_rebuild_failure_without_snapshot_just_raises(upstream, monkeypatch):
    import kestrel.engine as engine

    engine_dir = str(upstream["engine"])
    engine.adopt(engine_dir)

    _mock_cmake(monkeypatch, build_code=1)
    with pytest.raises(engine.EngineError) as exc:
        engine.rebuild(engine_dir)

    assert exc.value.restored_from_previous is False
    assert "no previous snapshot" in str(exc.value)
    assert not _previous_json(engine_dir).exists()
