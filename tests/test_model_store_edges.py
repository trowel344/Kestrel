import os

import pytest

from kestrel.model_store import (
    CorruptModelError,
    MissingModelError,
    ModelStoreError,
    complete_gguf_models,
    discover_local_models,
    model_total_size,
)


def _touch(path, data=b"GGUF"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_zero_byte_partial_download_is_skipped(tmp_path):
    _touch(tmp_path / "partial.gguf", b"")
    _touch(tmp_path / "fine.gguf")
    found = discover_local_models(tmp_path)
    assert [p.name for p in found] == ["fine.gguf"]


def test_zero_byte_file_raises_corrupt_when_read_directly(tmp_path):
    empty = _touch(tmp_path / "empty.gguf", b"")
    with pytest.raises(CorruptModelError, match="empty or truncated"):
        model_total_size(empty)


def test_missing_shard_split_is_dropped_gracefully(tmp_path):
    single = _touch(tmp_path / "single.gguf")
    shard1 = _touch(tmp_path / "m-00001-of-00003.gguf")
    result = complete_gguf_models([single, shard1])
    assert result == [single]
    assert model_total_size(shard1) == len(b"GGUF")


def test_dangling_symlink_is_skipped(tmp_path):
    link = tmp_path / "link.gguf"
    link.symlink_to(tmp_path / "gone.gguf")
    _touch(tmp_path / "real.gguf")
    found = discover_local_models(tmp_path)
    assert [p.name for p in found] == ["real.gguf"]


def test_file_symlink_is_resolved_and_deduplicated(tmp_path):
    real = _touch(tmp_path / "real.gguf")
    (tmp_path / "alias.gguf").symlink_to(real)
    found = discover_local_models(tmp_path)
    assert found == [real.resolve()]


def test_directory_named_gguf_is_not_a_model(tmp_path):
    d = tmp_path / "tricks.gguf"
    d.mkdir()
    inner = _touch(d / "inner.gguf")
    _touch(tmp_path / "real.gguf")
    found = discover_local_models(tmp_path)
    assert all(p.is_file() for p in found)
    assert not any(p == d for p in found)
    assert inner.resolve() in found


def test_directory_path_raises_corrupt_when_read_directly(tmp_path):
    d = tmp_path / "tricks.gguf"
    d.mkdir()
    with pytest.raises(CorruptModelError, match="not a regular file"):
        model_total_size(d)


def test_discovery_survives_unreadable_subtree(tmp_path):
    good = _touch(tmp_path / "good.gguf")
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    _touch(blocked / "hidden.gguf")
    try:
        blocked.chmod(0o000)
        found = discover_local_models(tmp_path)
        assert good.resolve() in found
        assert not any("blocked" in str(p) for p in found)
    finally:
        blocked.chmod(0o755)


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permission bits")
def test_readonly_root_raises_model_store_error(tmp_path):
    root = tmp_path / "store"
    root.mkdir()
    try:
        root.chmod(0o000)
        with pytest.raises(ModelStoreError, match="cannot read model store"):
            discover_local_models(root)
    finally:
        root.chmod(0o755)


def test_missing_nondefault_root_raises_missing(tmp_path):
    missing = tmp_path / "does-not-exist"
    with pytest.raises(MissingModelError, match="does not exist"):
        discover_local_models(missing)


def test_missing_primary_raises_missing_model(tmp_path):
    with pytest.raises(MissingModelError, match="not found on disk"):
        model_total_size(tmp_path / "nope.gguf")
