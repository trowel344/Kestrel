"""Self-update: wheel/repo install with integrity checks and limited rollback.

The updater snapshots Kestrel's package and distribution metadata before an
install and can restore those files when verification fails.  It cannot roll
back dependencies changed by pip, so callers must treat rollback as a best
effort recovery aid rather than a transaction over the whole environment.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import tomllib
import zipfile
from email.parser import Parser
from pathlib import Path

from .. import ui
from ..errors import InputError, IntegrityError
from . import parser, runtime

MAX_WHEEL_BYTES = 512 * 1024 * 1024
MAX_METADATA_BYTES = 4 * 1024 * 1024
PIP_TIMEOUT_SECONDS = 30 * 60
CHECK_TIMEOUT_SECONDS = 30


def cmd_self_update(args):
    """Update the running Kestrel package from its source tree or a wheel.

    ``--wheel`` may be a local path or an ``https://`` URL. When ``--sha256``
    is given the artifact is verified before install (trusted distribution).
    """
    repo = getattr(args, "repo", None) or os.environ.get("KESTREL_REPO")
    wheel = getattr(args, "wheel", None)
    downloaded_artifact = False
    expected_version = None
    if wheel:
        source_sha = _normalize_sha256(getattr(args, "sha256", None))
        if _is_remote_url(wheel) and not source_sha:
            raise IntegrityError(
                "a SHA256 checksum is required for remote wheels",
                hint="pass --sha256=<64 hex characters> and use an HTTPS URL",
            )
        artifact = _materialize_wheel(wheel)
        downloaded_artifact = _is_remote_url(wheel)
        source_label = str(wheel)
    else:
        if repo is None:
            root = Path(__file__).resolve().parents[2]
            repo = str(root) if (root / ".git").is_dir() else None
        if repo is None:
            raise InputError(
                "cannot locate the Kestrel repository",
                hint="set --repo or --wheel",
            )
        artifact, source_sha = repo, None
        expected_version = _validate_repo_identity(Path(repo))
        source_label = repo

    from_version = parser._kestrel_version()
    out = runtime._human_stream(args)
    js = bool(getattr(args, "json", False))
    plan = {
        "status": "planned",
        "from_version": from_version,
        "to_version": None,
        "rolled_back": False,
        "source": source_label,
        "dry_run": bool(args.dry_run),
    }

    backup = None
    cleanup_snapshot = False
    try:
        if source_sha:
            try:
                actual = _sha256_file(Path(artifact))
            except OSError as exc:
                raise IntegrityError(
                    f"unable to read update artifact {source_label}: {exc}",
                    hint="check that the wheel still exists and is readable",
                ) from exc
            if actual != source_sha:
                raise IntegrityError(
                    f"SHA256 mismatch for {source_label}",
                    hint="refusing to install an unverified artifact",
                )

        # Validate wheel identity even for dry runs so the preview cannot
        # claim an unrelated artifact would be installed.
        if wheel:
            expected_version = _validate_wheel_identity(Path(artifact))

        if args.dry_run:
            plan["status"] = "dry_run"
            plan["to_version"] = "pending"
            if js:
                print(json.dumps(plan, default=str))
                return
            print(ui.kv("Source", source_label, value_color=ui.bold), file=out)
            print(f"  Would install Kestrel from {source_label} via pip.", file=out)
            return

        backup = _snapshot_installed()
        print(ui.kv("Source", source_label, value_color=ui.bold), file=out)
        try:
            install = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", artifact],
                capture_output=True,
                text=True,
                timeout=PIP_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            rollback = _rollback_and_verify(backup)
            cleanup_snapshot = rollback
            snapshot_hint = _snapshot_hint(backup)
            raise IntegrityError(
                f"pip install timed out after {PIP_TIMEOUT_SECONDS}s ({'succeeded' if rollback else 'failed'} rollback)",
                hint=(
                    "the previous Kestrel files were restored"
                    if rollback
                    else f"manual recovery required{snapshot_hint}"
                ),
            ) from exc
        except OSError as exc:
            rollback = _rollback_and_verify(backup)
            cleanup_snapshot = rollback
            snapshot_hint = _snapshot_hint(backup)
            raise IntegrityError(
                f"unable to start pip ({'succeeded' if rollback else 'failed'} rollback): {exc}",
                hint=(
                    "the previous Kestrel files were restored"
                    if rollback
                    else f"manual recovery required{snapshot_hint}"
                ),
            ) from exc
        if install.returncode != 0:
            rollback = _rollback_and_verify(backup)
            status = "succeeded" if rollback else "unavailable or failed"
            if rollback:
                cleanup_snapshot = True
            snapshot_hint = _snapshot_hint(backup)
            raise IntegrityError(
                f"install failed ({status} rollback): {install.stderr[-2000:].strip()}",
                hint=(
                    "the previous Kestrel files were restored"
                    if rollback
                    else f"reinstall the previous environment manually; rollback was unavailable{snapshot_hint}"
                ),
            )

        try:
            ok, version = _post_install_check()
        except IntegrityError as exc:
            rollback = _rollback_and_verify(backup)
            cleanup_snapshot = rollback
            snapshot_hint = _snapshot_hint(backup)
            raise IntegrityError(
                f"post-install verification error ({'succeeded' if rollback else 'failed'} rollback): {exc}",
                hint=(
                    "the previous Kestrel files were restored"
                    if rollback
                    else f"manual recovery required{snapshot_hint}"
                ),
            ) from exc
        if expected_version and version != expected_version:
            ok = False
        verification_failed = not ok
        rolled_back = False
        rollback_verified = False
        if not ok and backup is not None:
            restored = _restore_install(backup)
            try:
                ok, version = _post_install_check()
            except IntegrityError:
                ok, version = False, ""
            if expected_version and version != from_version:
                ok = False
            rolled_back = True
            if restored is False:
                ok = False
            rollback_verified = restored is not False and ok
        if verification_failed:
            status = "succeeded" if rolled_back and ok else "unavailable or failed"
            if rollback_verified:
                cleanup_snapshot = True
            snapshot_hint = _snapshot_hint(backup)
            raise IntegrityError(
                f"post-install verification failed for {source_label} ({status} rollback)",
                hint=(
                    "the previous Kestrel files were restored; the update was not applied"
                    if rolled_back and ok
                    else f"Kestrel package files could not be verified; reinstall the previous environment manually{snapshot_hint}"
                ),
            )

        cleanup_snapshot = True
        if js:
            plan["status"] = "rolled_back" if rolled_back else "updated"
            plan["to_version"] = version
            plan["rolled_back"] = rolled_back
            print(json.dumps(plan, default=str))
            return
        if rolled_back:
            print(f"Installation rolled back; Kestrel still at {version}.", file=out)
        else:
            print(
                f"Updated Kestrel to {version}. Restart the shell for changes to take effect.",
                file=out,
            )
    finally:
        if backup is not None and cleanup_snapshot:
            _cleanup_snapshot(backup)
        if downloaded_artifact:
            try:
                Path(artifact).unlink(missing_ok=True)
            except OSError:
                pass


def _is_remote_url(target: str) -> bool:
    return target.lower().startswith(("http://", "https://"))


def _validate_repo_identity(repo: Path) -> str:
    """Validate a local update source is Kestrel and return its version."""
    if not repo.is_dir():
        raise InputError(
            f"update repository does not exist or is not a directory: {repo}",
            hint="pass --repo pointing to a local Kestrel checkout",
        )
    manifest = repo / "pyproject.toml"
    try:
        project = tomllib.loads(manifest.read_text()).get("project", {})
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise InputError(
            f"unable to read update repository metadata: {exc}",
            hint="the repository must contain a valid pyproject.toml",
        ) from exc
    name = str(project.get("name") or "").strip().casefold()
    version = str(project.get("version") or "").strip()
    if name != "kestrel" or not re.fullmatch(r"\d+(?:\.\d+)+", version):
        raise InputError(
            f"update repository is not a versioned Kestrel project (name={name or 'missing'}, version={version or 'missing'})",
            hint="pass --repo pointing to a Kestrel checkout with project metadata",
        )
    return version


def _normalize_sha256(value: str | None) -> str | None:
    if not value:
        return None
    digest = value.lower().removeprefix("sha256:").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise IntegrityError(
            "invalid SHA256 checksum",
            hint="provide exactly 64 hexadecimal characters, optionally prefixed with sha256:",
        )
    return digest


def _sha256_file(path: Path) -> str:
    from hashlib import sha256

    digest = sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise IntegrityError(f"unable to read update artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _materialize_wheel(target: str) -> str:
    """Return a local path for ``target`` (downloading https wheels to temp)."""
    if not target.startswith(("http://", "https://")):
        return target
    import tempfile
    import urllib.parse
    import urllib.request

    if not target.startswith("https://"):
        raise IntegrityError(
            "remote wheel downloads must use HTTPS",
            hint="download the wheel over HTTPS or provide a local path",
        )

    temporary = None
    try:
        with urllib.request.urlopen(target, timeout=60) as response:
            if not response.geturl().lower().startswith("https://"):
                raise IntegrityError("wheel download redirected away from HTTPS")
            suffix = Path(urllib.parse.urlparse(target).path).suffix or ".whl"
            if suffix.lower() != ".whl":
                suffix = ".whl"
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_WHEEL_BYTES:
                raise IntegrityError(f"remote wheel exceeds {MAX_WHEEL_BYTES} byte limit")
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                temporary = Path(handle.name)
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_WHEEL_BYTES:
                        raise IntegrityError(f"remote wheel exceeds {MAX_WHEEL_BYTES} byte limit")
                    handle.write(chunk)
            return str(temporary)
    except IntegrityError:
        _unlink_quiet(temporary)
        raise
    except (OSError, ValueError) as exc:
        _unlink_quiet(temporary)
        raise IntegrityError(
            f"unable to download update wheel: {exc}",
            hint="check the HTTPS URL and retry with a matching SHA256 digest",
        ) from exc


def _validate_wheel_identity(path: Path) -> str:
    """Verify a wheel contains exactly one distribution named ``kestrel``."""
    if path.suffix.lower() != ".whl":
        raise IntegrityError(f"wheel path does not end in .whl: {path}")
    try:
        with zipfile.ZipFile(path) as archive:
            metadata_names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
            if len(metadata_names) != 1:
                raise IntegrityError("wheel must contain exactly one dist-info/METADATA file")
            if archive.getinfo(metadata_names[0]).file_size > MAX_METADATA_BYTES:
                raise IntegrityError(f"wheel metadata exceeds {MAX_METADATA_BYTES} byte limit")
            metadata = Parser().parsestr(archive.read(metadata_names[0]).decode("utf-8", "replace"))
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise IntegrityError(f"unable to inspect wheel {path}: {exc}") from exc
    name = (metadata.get("Name") or "").strip()
    version = (metadata.get("Version") or "").strip()
    if name.casefold() != "kestrel" or not version:
        raise IntegrityError(
            f"wheel is not a valid Kestrel distribution (Name={name or 'missing'} Version={version or 'missing'})",
            hint="pass a wheel built from the Kestrel project",
        )
    return version


def _snapshot_installed():
    """Copy Kestrel package and dist metadata for limited rollback, or ``None``.

    For editable installs the checkout source itself is left untouched, while
    generated metadata, launchers, and PEP 660 import hooks are snapshotted.
    """
    try:
        import kestrel
    except ImportError:
        return None
    src = Path(kestrel.__file__).resolve().parent
    if not src.is_dir():
        return None
    source_tree = _is_source_tree(src)
    backup_root = Path(tempfile.mkdtemp(prefix="kestrel-snapshot-"))
    # Editable installs point at the checkout. Never copy or remove that
    # source package, but still preserve its generated metadata and launcher.
    entries = [] if source_tree else [(src, backup_root / src.name)]
    cleanup_paths: list[Path] = []
    metadata_parents: list[Path] = []
    try:
        from importlib import metadata

        distribution = metadata.distribution("kestrel")
        dist_info = Path(distribution._path).resolve()  # setuptools importlib metadata path
        if dist_info.exists() and dist_info != src:
            entries.append((dist_info, backup_root / dist_info.name))
            metadata_parents.append(dist_info.parent)
    except (ImportError, OSError, metadata.PackageNotFoundError, AttributeError):
        pass
    if source_tree:
        purelib = sysconfig.get_path("purelib")
        if purelib:
            # pip may replace an editable checkout with a regular package and
            # dist-info in site-packages before a failed update is rolled back.
            # Remove those replacement artifacts, while leaving the checkout
            # package itself untouched.
            purelib_root = Path(purelib)
            metadata_parents.append(purelib_root)
            cleanup_paths.append(purelib_root / "kestrel")
    for candidate in _console_script_candidates():
        cleanup_paths.append(candidate)
        if candidate.is_file():
            entries.append((candidate, backup_root / "scripts" / candidate.name))
    for candidate in _editable_artifact_candidates():
        cleanup_paths.append(candidate)
        entries.append((candidate, backup_root / "editable" / candidate.name))
    try:
        for original, backup in entries:
            if original.is_dir():
                shutil.copytree(original, backup)
            else:
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, backup)
    except BaseException:
        shutil.rmtree(backup_root, ignore_errors=True)
        raise
    return {"entries": entries, "root": backup_root, "cleanup": cleanup_paths, "metadata_parents": metadata_parents}


def _restore_install(snapshot) -> None:
    entries = snapshot.get("entries")
    if entries is None:
        # Compatibility with callers/tests that supplied the old two-path
        # shape; real snapshots always include package metadata as well.
        entries = [(snapshot["from"], snapshot["to"])]
    try:
        for path in snapshot.get("cleanup", []):
            _remove_path(path)
        for parent in snapshot.get("metadata_parents", []):
            for path in parent.glob("kestrel-*.dist-info"):
                _remove_path(path)
            for path in parent.glob("kestrel.egg-info"):
                _remove_path(path)
        for src, backup in entries:
            _remove_path(src)
            if backup.is_dir():
                shutil.copytree(backup, src)
            else:
                src.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup, src)
    except OSError:
        return False
    return True


def _rollback_and_verify(snapshot) -> bool:
    if snapshot is None:
        return False
    restored = _restore_install(snapshot)
    if restored is False:
        return False
    try:
        ok, _ = _post_install_check()
    except IntegrityError:
        return False
    return ok


def _snapshot_hint(snapshot) -> str:
    root = snapshot.get("root") if isinstance(snapshot, dict) else None
    return f"; snapshot retained at {root}" if root else ""


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _unlink_quiet(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _is_source_tree(package_dir: Path) -> bool:
    """Return true only for a checkout package, not any site-packages child."""
    root = package_dir.parent
    return (root / "pyproject.toml").is_file() and (root / ".git").exists()


def _console_script_candidates() -> list[Path]:
    candidates = [Path(sys.executable).with_name("kestrel")]
    scripts = sysconfig.get_path("scripts")
    if scripts:
        candidates.append(Path(scripts) / "kestrel")
    # Never adopt an arbitrary PATH entry: rollback cleanup is destructive and
    # must stay inside the interpreter environment being updated.
    return list(dict.fromkeys(candidates))


def _editable_artifact_candidates() -> list[Path]:
    """Return PEP 660 artifacts belonging to Kestrel in this interpreter."""
    purelib = sysconfig.get_path("purelib")
    if not purelib:
        return []
    root = Path(purelib)
    patterns = ("__editable__.kestrel-*.pth", "__editable___kestrel_*_finder.py")
    return list(dict.fromkeys(item for pattern in patterns for item in root.glob(pattern) if item.is_file()))


def _cleanup_snapshot(snapshot) -> None:
    """Remove the temporary rollback copy after the update attempt."""
    root = snapshot.get("root") if isinstance(snapshot, dict) else None
    if root:
        shutil.rmtree(root, ignore_errors=True)


def _post_install_check() -> tuple[bool, str]:
    """Import Kestrel and exercise the installed console script in isolation."""
    check_env = os.environ.copy()
    check_env.pop("PYTHONPATH", None)
    check_env["PYTHONNOUSERSITE"] = "1"
    with tempfile.TemporaryDirectory(prefix="kestrel-update-check-") as check_dir:
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    "import importlib.metadata as m; import kestrel; v=kestrel.__version__; assert m.version('kestrel') == v; print(v)",
                ],
                capture_output=True,
                text=True,
                cwd=check_dir,
                env=check_env,
                timeout=CHECK_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise IntegrityError(f"post-install import check timed out after {CHECK_TIMEOUT_SECONDS}s") from exc
        except OSError as exc:
            raise IntegrityError(f"unable to start post-install import check: {exc}") from exc
        version = result.stdout.strip()
        if result.returncode != 0 or not version or any(char.isspace() for char in version):
            return False, version
        executable = next((item for item in _console_script_candidates() if item.is_file()), None)
        if executable is None:
            return False, version
        try:
            cli_result = subprocess.run(
                [str(executable), "--version"],
                capture_output=True,
                text=True,
                cwd=check_dir,
                env=check_env,
                timeout=CHECK_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise IntegrityError(f"post-install CLI check timed out after {CHECK_TIMEOUT_SECONDS}s") from exc
        except OSError as exc:
            raise IntegrityError(f"unable to start post-install CLI check: {exc}") from exc
        cli_version = cli_result.stdout.strip().removeprefix("kestrel ").strip()
        return cli_result.returncode == 0 and cli_version == version, version
