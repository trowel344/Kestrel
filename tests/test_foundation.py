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


def test_available_disk_bytes_reports_number(tmp_path):
    from kestrel.util import available_disk_bytes

    value = available_disk_bytes(tmp_path)
    assert value is None or value > 0
