"""Reusable, crash-safe filesystem and process helpers.

The safety property that makes Kestrel's state trustworthy: any persisted file
(config, engine manifest, caches) is written atomically in place — a crash
leaves either the old or the new contents, never a torn document.
"""

from __future__ import annotations

import functools
import os
import tempfile
import time
from pathlib import Path


def sync_directory(directory: str | Path) -> None:
    """Best-effort sync of a directory entry after an atomic rename.

    ``os.replace`` makes readers see either complete version, but a power loss
    can still lose the new directory entry unless the parent is synced.  Some
    platforms/filesystems do not allow opening directories, so durability is
    deliberately best effort while replacement itself remains strict.
    """
    try:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        fd = os.open(Path(directory), flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


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
    requested = Path(path)
    # Configuration files are commonly symlinked into dotfile repositories.
    # Update the referent atomically instead of replacing the user's symlink.
    target = requested.resolve(strict=False) if requested.is_symlink() else requested
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = data.encode() if isinstance(data, str) else data
    mode = target.stat().st_mode & 0o7777 if target.exists() else None
    if backup and target.exists():
        # Replace the backup path itself. Do not use copyfile here: it follows
        # a pre-existing .bak symlink and could overwrite an unrelated file.
        _replace_file(target.with_name(target.name + ".bak"), target.read_bytes(), mode=mode)
    _replace_file(target, payload, mode=mode)
    return requested


def _replace_file(target: Path, payload: bytes, *, mode: int | None) -> None:
    """Durably replace one concrete path without following its final symlink."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
    try:
        if mode is not None:
            os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
        sync_directory(target.parent)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def copy_file(src: str | Path, dst: str | Path) -> Path:
    """Atomically copy ``src`` to ``dst`` while preserving executable mode."""
    from shutil import copyfileobj

    source = Path(src)
    target = Path(dst)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = source.stat().st_mode & 0o7777
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as output_handle:
            fd = -1
            with source.open("rb") as input_handle:
                copyfileobj(input_handle, output_handle, length=1024 * 1024)
                output_handle.flush()
                os.fsync(output_handle.fileno())
        os.replace(tmp, target)
        sync_directory(target.parent)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return target


def clone_or_copy(src: str | Path, dst: str | Path) -> Path:
    """Copy ``src`` to ``dst`` via a copy-on-write reflink when possible.

    ``copy_file`` re-writes every byte, which makes multi-GB engine binary
    snapshots expensive.  Filesystems that support reflinks (btrfs/XFS via
    FICLONE on Linux, APFS via clonefile on macOS) can instead share the
    source's extents and only copy-on-write later, so a snapshot costs a
    handful of syscalls regardless of file size.

    Unlike a hard link, a reflink is safe for build artifacts: a later
    in-place write to ``src`` (linkers truncate the output file in place)
    copies on write and can never corrupt the snapshot.  The destination is
    written to a same-directory temp file and atomically renamed into place,
    mirroring ``copy_file``'s durability story.  Falls back to a byte copy
    when the platform or filesystem cannot reflink.
    """
    source = Path(src)
    target = Path(dst)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        _reflink(source, target)
    except OSError:
        copy_file(source, target)
    return target


def _reflink(source: Path, target: Path) -> None:
    """Clone ``source`` onto ``target`` as a copy-on-write reflink.

    Raises :class:`OSError` when the platform or filesystem cannot reflink so
    :func:`clone_or_copy` can fall back to a byte copy.
    """
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
    try:
        os.close(fd)
        os.unlink(tmp)
        if hasattr(os, "clonefile"):  # macOS (APFS)
            os.clonefile(source, tmp)
        else:  # Linux: ioctl(FS_IOC_FICLONE)
            _ficlone(source, tmp)
        os.chmod(tmp, source.stat().st_mode & 0o7777)
        os.replace(tmp, target)
        sync_directory(target.parent)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _ficlone(source: Path, target: Path) -> None:
    """Clone ``source``'s extents onto ``target`` using ``FS_IOC_FICLONE``."""
    import fcntl

    FICLONE = 0x40049409  # _IOW(0x94, 9, int), linux/fs.h
    with source.open("rb") as src_handle, open(target, "wb") as dst_handle:
        fcntl.ioctl(dst_handle.fileno(), FICLONE, src_handle.fileno())


def available_disk_bytes(path: str | Path) -> int | None:
    """Free bytes on the filesystem holding ``path``, or ``None`` if unknown."""
    try:
        stat = os.statvfs(str(path))
    except OSError:
        return None
    return stat.f_bavail * stat.f_frsize


def truncate(text: str, limit: int = 2000) -> str:
    """Keep only the last ``limit`` characters of ``text``.

    Error detail from subprocess output is trimmed this way everywhere so a
    single long line (e.g. a CUDA trace) cannot flood the terminal or a JSON
    report. Pure, so tests can assert on it directly.
    """
    return text[-limit:] if len(text) > limit else text


def ttl_cache(seconds: float):
    """Memoize a probe result for ``seconds`` in one process lifetime.

    Repeated reads in a single invocation (or rapid menu redraws) re-read
    ``nvidia-smi`` or ``/proc/meminfo``; the small staleness window keeps live
    hardware reasonably fresh in a long-running interactive session.
    """

    def deco(fn):
        cached_at = None
        cached_key = None
        value = None

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            nonlocal cached_at, cached_key, value
            now = time.monotonic()
            key = (args, tuple(sorted(kwargs.items())))
            try:
                same_key = cached_key == key
            except (TypeError, ValueError):
                same_key = False
            if cached_at is not None and same_key and now - cached_at < seconds:
                return value
            value = fn(*args, **kwargs)
            cached_at = now
            cached_key = key
            return value

        return wrapped

    return deco
