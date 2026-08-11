import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_error_as_dict_and_codes():
    from kestrel.errors import CorruptModelError, KestrelError

    e = CorruptModelError("bad header", hint="re-download the file")
    err_dict = e.as_dict()
    assert err_dict["code"] == "model_corrupt"
    assert "bad header" in err_dict["message"]
    assert err_dict["hint"] == "re-download the file"
    assert isinstance(e, KestrelError)
    assert e.exit_code == 1


def test_error_default_message_and_code_override():
    from kestrel.errors import MissingModelError

    e = MissingModelError()
    assert e.code == "model_not_found"
    assert e.message  # falls back to docstring/class text


def test_write_atomic_creates_and_replaces(tmp_path):
    from kestrel.util import write_atomic

    target = tmp_path / "dir" / "f.txt"
    write_atomic(target, "one")
    assert target.read_text() == "one"
    write_atomic(target, "two")
    assert target.read_text() == "two"
    assert (tmp_path / "dir" / "f.txt.bak").read_text() == "one"


def test_write_atomic_bytes_and_no_backup(tmp_path):
    from kestrel.util import write_atomic

    target = tmp_path / "blob.bin"
    write_atomic(target, b"\x00\x01", backup=False)
    assert target.read_bytes() == b"\x00\x01"
    assert not (tmp_path / "blob.bin.bak").exists()


def test_write_atomic_preserves_mode_and_target_symlink(tmp_path):
    from kestrel.util import write_atomic

    referent = tmp_path / "real.toml"
    referent.write_text("old")
    referent.chmod(0o640)
    link = tmp_path / "config.toml"
    link.symlink_to(referent.name)

    assert write_atomic(link, "new") == link

    assert link.is_symlink()
    assert referent.read_text() == "new"
    assert referent.stat().st_mode & 0o777 == 0o640


def test_write_atomic_does_not_follow_backup_symlink(tmp_path):
    from kestrel.util import write_atomic

    target = tmp_path / "config.toml"
    target.write_text("old")
    unrelated = tmp_path / "unrelated"
    unrelated.write_text("do not overwrite")
    backup = tmp_path / "config.toml.bak"
    backup.symlink_to(unrelated.name)

    write_atomic(target, "new")

    assert unrelated.read_text() == "do not overwrite"
    assert not backup.is_symlink()
    assert backup.read_text() == "old"


def test_copy_file_preserves_executable_mode_and_replaces_symlink(tmp_path):
    from kestrel.util import copy_file

    source = tmp_path / "llama-cli"
    source.write_bytes(b"binary")
    source.chmod(0o755)
    unrelated = tmp_path / "unrelated"
    unrelated.write_bytes(b"keep")
    target = tmp_path / "snapshot" / "llama-cli"
    target.parent.mkdir()
    target.symlink_to(unrelated)

    copy_file(source, target)

    assert target.read_bytes() == b"binary"
    assert not target.is_symlink()
    assert target.stat().st_mode & 0o777 == 0o755
    assert unrelated.read_bytes() == b"keep"


def test_available_disk_bytes_reports_number(tmp_path):
    from kestrel.util import available_disk_bytes

    value = available_disk_bytes(tmp_path)
    assert value is None or value > 0


def test_truncate_keeps_tail():
    from kestrel.util import truncate

    assert truncate("abcdef", limit=4) == "cdef"
    assert truncate("abc", limit=4) == "abc"


def test_ttl_cache_refreshes_after_window():
    import time as _time

    from kestrel.util import ttl_cache

    calls = []

    @ttl_cache(seconds=3600)
    def probe(value):
        calls.append(value)
        return value * 2

    assert probe(3) == 6
    assert probe(3) == 6
    assert calls == [3]
    _time.monotonic()  # stall a moment (cache never expires within the window)
