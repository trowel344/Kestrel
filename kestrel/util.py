"""Reusable, crash-safe filesystem and process helpers.

The safety property that makes Kestrel's state trustworthy: any persisted file
(config, engine manifest, caches) is written atomically in place — a crash
leaves either the old or the new contents, never a torn document.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def write_atomic(
    path: str | Path,
    data: str | bytes,
    *,
    backup: bool = True,
) -> Path:
    """Write ``data`` to ``path`` atomically via a same-directory temp + rename.

    If ``backup`` is true and a previous file exists, its contents are first
    preserved next to ``path`` as ``<name>.bak`` so a torn read by another
    tool never loses the last-good version. The parent directory is created as
    needed. Raises :class:`OSError` on failure (caller may wrap it).
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if backup and target.exists():
        copy_file(target, target.with_name(target.name + ".bak"))
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            payload = data.encode() if isinstance(data, str) else data
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return target


def copy_file(src: str | Path, dst: str | Path) -> Path:
    """Copy ``src`` to ``dst``, creating the destination directory."""
    from shutil import copyfile

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    copyfile(src, dst)
    return dst


def available_disk_bytes(path: str | Path) -> int | None:
    """Free bytes on the filesystem holding ``path``, or ``None`` if unknown."""
    try:
        stat = os.statvfs(str(path))
    except OSError:
        return None
    return stat.f_bavail * stat.f_frsize
