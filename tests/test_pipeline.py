import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from kestrel.core.pipeline import InferencePipeline, _resolve_model_dir  # noqa: E402


def _make_pipeline(model_dir: str) -> InferencePipeline:
    pipeline = object.__new__(InferencePipeline)
    pipeline.model_dir = model_dir
    return pipeline


def test_discover_gguf_empty_dir_returns_default(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    pipeline = _make_pipeline(str(empty))
    assert pipeline._discover_gguf() == str(empty) + ".gguf"


def test_discover_gguf_missing_dir_returns_default(tmp_path):
    missing = tmp_path / "does-not-exist"
    pipeline = _make_pipeline(str(missing))
    assert pipeline._discover_gguf() == str(missing) + ".gguf"


def test_discover_gguf_prefers_model_gguf_inside(tmp_path):
    (tmp_path / "model.gguf").write_bytes(b"GGUF" + b"\x00" * 8)
    pipeline = _make_pipeline(str(tmp_path))
    assert pipeline._discover_gguf() == str(tmp_path / "model.gguf")


def test_discover_gguf_beside_directory_wins_over_other_ggufs(tmp_path):
    # An exact "<dir>.gguf" next to the directory outranks arbitrary gguFs inside.
    (tmp_path / "other.gguf").write_bytes(b"GGUF" + b"\x00" * 8)
    (str(tmp_path) + ".gguf") and Path(str(tmp_path) + ".gguf").write_bytes(b"GGUF" + b"\x00" * 8)
    pipeline = _make_pipeline(str(tmp_path))
    assert pipeline._discover_gguf() == str(tmp_path) + ".gguf"


def test_discover_gguf_falls_back_to_split_shard(tmp_path):
    # A dir holding only split shards: discovery must pick the first shard,
    # which llama.cpp resolves into the full model.
    (tmp_path / "m-00001-of-00002.gguf").write_bytes(b"GGUF" + b"\x00" * 8)
    (tmp_path / "m-00002-of-00002.gguf").write_bytes(b"GGUF" + b"\x00" * 8)
    pipeline = _make_pipeline(str(tmp_path))
    assert pipeline._discover_gguf() == str(tmp_path / "m-00001-of-00002.gguf")


def test_resolve_model_dir_direct_layout(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text("{}")
    assert _resolve_model_dir(str(tmp_path)) == str(tmp_path)


def test_resolve_model_dir_hf_refs_main(tmp_path):
    refs = tmp_path / "refs"
    refs.mkdir()
    (refs / "main").write_text("abc123")
    snapshot = tmp_path / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors.index.json").write_text("{}")
    assert _resolve_model_dir(str(tmp_path)) == str(snapshot)


def test_resolve_model_dir_hf_newest_snapshot(tmp_path):
    snapshots = tmp_path / "snapshots"
    (snapshots / "old").mkdir(parents=True)
    (snapshots / "old" / "model.safetensors.index.json").write_text("{}")
    (snapshots / "new").mkdir()
    (snapshots / "new" / "model.safetensors.index.json").write_text("{}")
    # Force "old" to have the older mtime so "new" is picked as newest.
    import os
    import time

    old = snapshots / "old"
    past = time.time() - 1000
    os.utime(old, (past, past))
    result = _resolve_model_dir(str(tmp_path))
    assert result == str(snapshots / "new")
