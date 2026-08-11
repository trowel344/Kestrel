"""Engine lifecycle: provenance, upstream updates, and self-rebuilds.

Kestrel wraps an external llama.cpp checkout (the "engine"). This module
records where an engine came from (remote/branch/commit/build flags) in a
small manifest, lets ``engine status`` report provenance and staleness, and
lets ``engine update`` fetch a newer upstream revision and rebuild safely.

Safety: ``update`` refuses to destroy uncommitted work or a divergence unless
the caller explicitly passes ``force=True``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import util
from .errors import EngineError

DEFAULT_CMAKE_FLAGS = [
    "-DLLAMA_CUDA=ON",
    "-DLLAMA_CUDA_NVFP4=ON",
    "-DGGML_RPC=ON",
    "-DCMAKE_BUILD_TYPE=Release",
]
BUILD_TARGETS = ["llama-cli", "llama-server", "llama-bench", "rpc-server"]
RPC_TARGETS = ("rpc-server", "ggml-rpc-server")
CONFIGURE_TIMEOUT_SECONDS = 10 * 60
BUILD_TIMEOUT_SECONDS = 2 * 60 * 60
MANIFEST_NAME = ".kestrel-engine.json"

# "Last-good" snapshot/rollback storage, kept next to the manifest.
PREVIOUS_DIR_NAME = ".kestrel-engine.previous"
PREVIOUS_JSON_NAME = ".kestrel-engine.previous.json"

# Stderr signatures that mean "this engine is too old for the model".
ENGINE_TOO_OLD_SIGNATURES = (
    "wrong array length",
    "dimension_sections",
    "error loading model hyperparameters",
    "unknown architecture",
)


@dataclass
class EngineManifest:
    remote: str
    branch: str | None
    commit: str
    cmake_flags: list[str] = field(default_factory=lambda: list(DEFAULT_CMAKE_FLAGS))
    built_at: str | None = None
    targets: list[str] = field(default_factory=lambda: list(BUILD_TARGETS))
    artifacts: dict[str, dict] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "remote": self.remote,
            "branch": self.branch,
            "commit": self.commit,
            "cmake_flags": list(self.cmake_flags),
            "built_at": self.built_at,
            "targets": list(self.targets),
            "artifacts": dict(self.artifacts),
        }


def manifest_path(directory: str) -> Path:
    return Path(directory) / MANIFEST_NAME


def _rpc_target_for_source(directory: str) -> str:
    """Select the RPC target name used by this llama.cpp checkout.

    Upstream renamed ``ggml-rpc-server`` to ``rpc-server`` across revisions;
    passing both to CMake fails because one is necessarily unknown. Keep the
    target in the manifest so rebuilds remain reproducible for that checkout.
    """
    rpc_cmake = Path(directory) / "tools" / "rpc" / "CMakeLists.txt"
    try:
        text = rpc_cmake.read_text(errors="strict")
    except OSError:
        text = ""
    match = re.search(r"(?m)^\s*set\s*\(\s*TARGET\s+([A-Za-z0-9_-]+)\s*\)", text)
    if match and match.group(1) in RPC_TARGETS:
        return match.group(1)
    for target in RPC_TARGETS:
        if re.search(rf"add_executable\s*\(\s*{re.escape(target)}(?:\s|\))", text):
            return target
    return RPC_TARGETS[0]


def load_manifest(directory: str) -> EngineManifest | None:
    path = manifest_path(directory)
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    try:
        cmake_flags = list(payload.get("cmake_flags") or DEFAULT_CMAKE_FLAGS)
        if "-DGGML_RPC=ON" not in cmake_flags:
            cmake_flags.append("-DGGML_RPC=ON")
        targets = list(payload.get("targets") or BUILD_TARGETS)
        rpc_target = _rpc_target_for_source(directory)
        old_rpc = next((target for target in RPC_TARGETS if target in targets), None)
        if old_rpc is None:
            targets.append(rpc_target)
        elif old_rpc != rpc_target:
            targets[targets.index(old_rpc)] = rpc_target
        return EngineManifest(
            remote=payload["remote"],
            branch=payload.get("branch"),
            commit=payload["commit"],
            cmake_flags=cmake_flags,
            built_at=payload.get("built_at"),
            targets=targets,
            artifacts=payload.get("artifacts") or {},
        )
    except KeyError as exc:
        raise EngineError(f"corrupt engine manifest at {path}: missing {exc}") from exc


def save_manifest(directory: str, manifest: EngineManifest) -> Path:
    path = manifest_path(directory)
    try:
        util.write_atomic(path, json.dumps(manifest.as_dict(), indent=2), backup=False)
    except OSError as exc:
        raise EngineError(f"cannot write engine manifest: {exc}") from exc
    return path


def _git(
    directory: str, args: list[str], *, capture: bool = True, check: bool = False, timeout: int = 60
) -> subprocess.CompletedProcess:
    cmd = ["git", "-C", directory, *args]
    try:
        result = subprocess.run(cmd, capture_output=capture, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise EngineError("git is not installed") from exc
    except subprocess.TimeoutExpired as exc:
        raise EngineError(f"git {args[0]} timed out in {directory}") from exc
    except OSError as exc:
        raise EngineError(f"cannot run git {args[0]} in {directory}: {exc}") from exc
    if check and result.returncode != 0:
        raise EngineError(f"git {args[0]} failed in {directory}: {result.stderr.strip()}")
    return result


def _rev_counts(directory: str, *, head: str, remote_head: str) -> tuple[int, int] | None:
    """Return ``(behind, ahead)`` commit counts; ``None`` when git can't answer.

    Both ``engine_status`` and ``update`` compute the same ``git rev-list
    --count`` pair to decide staleness; one helper keeps the two in lock-step.
    """
    behind = _git(directory, ["rev-list", "--count", f"{head}..{remote_head}"], check=False)
    ahead = _git(directory, ["rev-list", "--count", f"{remote_head}..{head}"], check=False)
    if behind.returncode != 0 or ahead.returncode != 0:
        return None
    return int(behind.stdout.strip() or 0), int(ahead.stdout.strip() or 0)


def _binary_stat(binary: Path) -> dict[str, bool | int | float | None]:
    """``(exists, size, mtime)`` report for a build artifact."""
    if not binary.is_file():
        return {"exists": False, "size": 0, "mtime": None}
    st = binary.stat()
    return {"exists": True, "size": st.st_size, "mtime": st.st_mtime}


def is_git(directory: str) -> bool:
    if not Path(directory).is_dir():
        return False
    return _git(directory, ["rev-parse", "--is-inside-work-tree"], check=False).returncode == 0


def git_head(directory: str) -> str | None:
    result = _git(directory, ["rev-parse", "HEAD"], check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def git_branch(directory: str) -> str | None:
    result = _git(directory, ["symbolic-ref", "--short", "HEAD"], check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def git_remote(directory: str, remote: str = "origin") -> str | None:
    result = _git(directory, ["remote", "get-url", remote], check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def git_dirty(directory: str) -> bool:
    result = _git(directory, ["status", "--porcelain"], check=False)
    if result.returncode != 0:
        return True
    # Kestrel's own bookkeeping (manifest + last-good snapshot) must never
    # block an engine update.
    ignored = {MANIFEST_NAME, PREVIOUS_DIR_NAME, PREVIOUS_JSON_NAME}
    for line in result.stdout.splitlines():
        path = line[3:].strip()
        if path.split("/")[0] in ignored:
            continue
        return True
    return False


def ls_remote_head(remote: str, branch: str | None, timeout: int = 15) -> str | None:
    ref = f"refs/heads/{branch}" if branch else "HEAD"
    command = ["git", "ls-remote", remote, ref]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise EngineError("git is not installed") from exc
    except subprocess.TimeoutExpired as exc:
        raise EngineError(f"git ls-remote timed out for {remote}") from exc
    except OSError as exc:
        raise EngineError(f"cannot query remote {remote}: {exc}") from exc
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 2:
            return fields[0]
    return None


def _resolve_target(directory: str, remote: str, branch: str | None) -> tuple[str | None, str | None]:
    """Return ``(branch, remote_head_commit)`` for the tracked revision.

    Fetches the upstream (which may be a bare URL) into a stable
    remote-tracking ref so the target revision is always addressable.
    """
    target_branch = branch or git_branch(directory) or "master"
    tracking = f"refs/remotes/kestrel-upstream/{target_branch}"
    refspec = f"+refs/heads/{target_branch}:{tracking}"
    result = _git(directory, ["fetch", remote, refspec], check=False, timeout=120)
    if result.returncode != 0:
        raise EngineError(f"git fetch {remote} failed in {directory}: {result.stderr.strip()}")
    probe = _git(directory, ["rev-parse", "--verify", tracking], check=False)
    if probe.returncode != 0:
        raise EngineError(f"remote {remote} has no branch {target_branch}")
    return target_branch, probe.stdout.strip()


def engine_status(directory: str, *, check_remote: bool = True) -> dict:
    """Report provenance, staleness, and build artifacts for an engine dir."""
    manifest = load_manifest(directory)
    git_ok = is_git(directory)
    head = git_head(directory) if git_ok else None
    branch = git_branch(directory) if git_ok else None
    remote = manifest.remote if manifest else (git_remote(directory) if git_ok else None)

    behind = ahead = stale = None
    remote_head = None
    if check_remote and git_ok and remote:
        try:
            remote_head = ls_remote_head(remote, manifest.branch if manifest else branch)
            if remote_head and head:
                if remote_head != head:
                    _target_branch, remote_head = _resolve_target(
                        directory,
                        remote,
                        manifest.branch if manifest else branch,
                    )
                counts = _rev_counts(directory, head=head, remote_head=remote_head)
                if counts is not None:
                    behind, ahead = counts
                    stale = bool(behind) and not ahead
        except EngineError:
            remote_head = None

    artifacts = None
    if manifest and manifest.targets:
        build_bin = Path(directory) / "build" / "bin"
        artifacts = {name: _binary_stat(build_bin / name) for name in manifest.targets}

    return {
        "directory": str(Path(directory).resolve()),
        "git": git_ok,
        "remote": remote,
        "branch": branch,
        "commit": head,
        "remote_head": remote_head,
        "behind": behind,
        "ahead": ahead,
        "stale": stale,
        "dirty": git_dirty(directory) if git_ok else True,
        "manifest": manifest.as_dict() if manifest else None,
        "artifacts": artifacts,
    }


def adopt(directory: str, remote: str | None = None) -> EngineManifest:
    """Record an existing (or fresh) checkout as a managed engine."""
    if not is_git(directory):
        raise EngineError(f"{directory} is not a git checkout; clone it first")
    commit = git_head(directory)
    if not commit:
        raise EngineError(f"{directory} has no commits")
    resolved = remote or git_remote(directory)
    if not resolved:
        raise EngineError(f"{directory} has no remote; pass --remote <url> to adopt it")
    manifest = EngineManifest(
        remote=resolved,
        branch=git_branch(directory),
        commit=commit,
        targets=["llama-cli", "llama-server", "llama-bench", _rpc_target_for_source(directory)],
        built_at=None,
    )
    save_manifest(directory, manifest)
    return manifest


def _artifact_report(directory: str, targets: list[str]) -> dict[str, dict]:
    build_bin = Path(directory) / "build" / "bin"
    report = {}
    for name in targets:
        info = _binary_stat(build_bin / name)
        if info["exists"]:
            report[name] = {"size": info["size"], "mtime": info["mtime"]}
    return report


def _previous_dir(directory: str) -> Path:
    return Path(directory) / PREVIOUS_DIR_NAME


def _previous_json_path(directory: str) -> Path:
    return Path(directory) / PREVIOUS_JSON_NAME


def _load_previous(directory: str) -> dict | None:
    """Return the recorded last-good snapshot sidebar, or ``None`` if absent."""
    try:
        return json.loads(_previous_json_path(directory).read_text())
    except (OSError, ValueError):
        return None


def _snapshot_previous(directory: str, targets: list[str], *, commit: str | None, last_good: str | None) -> dict | None:
    """Copy currently-installed binaries into the side dir + write a sidebar.

    Only binaries that actually exist are snapshotted; if none exist this is a
    no-op and returns ``None``. The sidebar records which ``git_head`` it came
    from (``commit``) and which build it corresponds to (``last_good``).
    """
    prev = _previous_dir(directory)
    build_bin = Path(directory) / "build" / "bin"
    snap = {"git": commit, "last_good": last_good, "artifacts": {}}
    wrote = False
    for name in targets:
        info = _binary_stat(build_bin / name)
        if not info["exists"]:
            continue
        util.copy_file(build_bin / name, prev / name)
        snap["artifacts"][name] = {"size": info["size"], "mtime": info["mtime"]}
        wrote = True
    if not wrote:
        return None
    snap["at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    util.write_atomic(_previous_json_path(directory), json.dumps(snap, indent=2), backup=False)
    return snap


def _rollback(directory: str, targets: list[str]) -> dict:
    """Restore last-good binaries from the sidecar snapshot; report integrity."""
    snapshot = _load_previous(directory)
    result = {"restored": False, "restored_commit": None, "ahead": None}
    if snapshot is None:
        return result
    prev = _previous_dir(directory)
    build_bin = Path(directory) / "build" / "bin"
    os.makedirs(build_bin, exist_ok=True)
    for name in targets:
        src = prev / name
        if src.is_file():
            try:
                util.copy_file(src, build_bin / name)
                result["restored"] = True
            except OSError:
                pass
    result["restored_commit"] = snapshot.get("last_good") or snapshot.get("git")
    head = git_head(directory)
    if head and result["restored_commit"] and head != result["restored_commit"]:
        # The checkout moved (e.g. `update` hard-reset) but artifacts are ours.
        result["ahead"] = head
    return result


def _smoke_test(directory: str) -> tuple[bool | None, str | None]:
    """Run a freshly built binary; return ``(ok, detail)``.

    ``ok`` is ``True`` when the binary reported a version, ``False`` when a
    runnable binary failed (non-zero exit / no output), and ``None`` when there
    is nothing runnable to smoke (used in mock environments that produce no
    real binaries).
    """
    build_bin = Path(directory) / "build" / "bin"
    for name in ("llama-cli", "llama-server"):
        binary = build_bin / name
        if not binary.is_file() or not os.access(binary, os.X_OK):
            continue
        try:
            result = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            continue
        output = ((result.stdout or "") + (result.stderr or "")).strip()
        if result.returncode != 0:
            return False, f"{name} --version exited with {result.returncode}"
        if not output:
            return False, f"{name} --version produced no output"
        return True, output
    return None, None


def _raise_with_rollback(directory: str, targets: list[str], cause: str) -> None:
    """Roll back to the last-good snapshot (if any) and raise EngineError."""
    rb = _rollback(directory, targets)
    message = cause
    if rb["restored"]:
        message += "; restored last-good binaries"
        if rb["restored_commit"]:
            message += f" (commit {rb['restored_commit']})"
        if rb["ahead"]:
            message += (
                f"; working tree is at {rb['ahead']} but artifacts were rolled back - "
                "run `kestrel engine rebuild` again deliberately"
            )
    else:
        message += " (no previous snapshot to restore)"
    err = EngineError(message)
    err.restored_from_previous = bool(rb["restored"])
    err.restored_commit = rb["restored_commit"]
    err.rolled_back_ahead = rb["ahead"]
    raise err


def _run_build_step(
    command: list[str],
    *,
    directory: str,
    targets: list[str],
    phase: str,
    timeout: int,
) -> subprocess.CompletedProcess:
    """Run one bounded CMake phase and roll artifacts back on any failure."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        _raise_with_rollback(directory, targets, f"cmake {phase} timed out after {timeout}s in {directory}")
    except OSError as exc:
        _raise_with_rollback(directory, targets, f"cannot start cmake {phase} in {directory}: {exc}")
    if result.returncode != 0:
        detail = util.truncate(result.stderr or result.stdout)
        _raise_with_rollback(directory, targets, f"cmake {phase} failed in {directory}:\n{detail}")
    return result


def rebuild(directory: str, *, dry_run: bool = False, flags: list[str] | None = None) -> dict:
    """Rebuild the engine from its checked-out source using recorded flags.

    Crash-safe: the currently-installed binaries are snapshotted as the
    last-good copy before the build starts, so a failed or interrupted build
    (or a smoke-test failure) rolls the engine back instead of leaving it
    without working binaries.
    """
    manifest = load_manifest(directory)
    if manifest is None:
        manifest = adopt(directory)
    chosen = flags or manifest.cmake_flags or list(DEFAULT_CMAKE_FLAGS)
    commit = git_head(directory) or manifest.commit
    build_dir = Path(directory) / "build"

    if dry_run:
        return {
            "status": "dry_run",
            "directory": directory,
            "commit": commit,
            "cmake_flags": chosen,
            "targets": list(manifest.targets),
            "restored_from_previous": False,
        }

    os.makedirs(build_dir, exist_ok=True)
    _snapshot_previous(directory, manifest.targets, commit=commit, last_good=manifest.commit)
    _run_build_step(
        ["cmake", "-S", str(Path(directory)), "-B", str(build_dir), *chosen],
        directory=directory,
        targets=manifest.targets,
        phase="configure",
        timeout=CONFIGURE_TIMEOUT_SECONDS,
    )
    _run_build_step(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            *manifest.targets,
            "-j",
            str(os.cpu_count() or 4),
        ],
        directory=directory,
        targets=manifest.targets,
        phase="build",
        timeout=BUILD_TIMEOUT_SECONDS,
    )

    smoke_ok, smoke_detail = _smoke_test(directory)
    if smoke_ok is False:
        _raise_with_rollback(directory, manifest.targets, f"smoke test failed: {smoke_detail}")

    # Rotate the last-good snapshot to the freshly verified artifacts.
    _snapshot_previous(directory, manifest.targets, commit=commit, last_good=commit)
    manifest.commit = commit
    manifest.built_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest.artifacts = _artifact_report(directory, manifest.targets)
    save_manifest(directory, manifest)
    return {
        "status": "rebuilt",
        "directory": directory,
        "commit": commit,
        "built_at": manifest.built_at,
        "targets": list(manifest.targets),
        "artifacts": manifest.artifacts,
        "restored_from_previous": False,
    }


def update(directory: str, *, dry_run: bool = False, force: bool = False, remote: str | None = None) -> dict:
    """Fetch upstream, fast-forward to a newer revision, and rebuild.

    Refuses to destroy uncommitted work or a diverged history unless
    ``force=True`` (which then hard-resets the checkout to the remote).
    """
    manifest = load_manifest(directory)
    if manifest is None:
        manifest = adopt(directory, remote)
    fetch_remote = remote or manifest.remote

    branch, remote_head = _resolve_target(directory, fetch_remote, manifest.branch)
    head = git_head(directory) or manifest.commit

    behind = ahead = 0
    if remote_head and head:
        counts = _rev_counts(directory, head=head, remote_head=remote_head)
        if counts is None:
            raise EngineError(f"cannot compare local commit {head} with fetched upstream {remote_head}")
        behind, ahead = counts

    if behind == 0:
        return {
            "status": "up_to_date",
            "directory": directory,
            "commit": head,
            "behind": behind,
            "ahead": ahead,
            "restored_from_previous": False,
        }

    if ahead and not force:
        raise EngineError(
            f"{directory} is {ahead} commit(s) ahead of {fetch_remote}/{branch} "
            "(local work); pass --force to hard-reset and rebuild"
        )
    if git_dirty(directory) and not force:
        raise EngineError(f"{directory} has uncommitted changes; commit or stash them, or pass --force")

    if dry_run:
        return {
            "status": "update_available",
            "directory": directory,
            "from_commit": head,
            "to_commit": remote_head,
            "behind": behind,
            "ahead": ahead,
            "branch": branch,
            "restored_from_previous": False,
        }

    reset = _git(
        directory,
        ["reset", "--hard", f"refs/remotes/kestrel-upstream/{branch}"],
        check=False,
    )
    if reset.returncode != 0:
        raise EngineError(f"git reset --hard failed in {directory}: {reset.stderr.strip()}")

    if force:
        clean = _git(
            directory,
            [
                "clean",
                "-fd",
                "-e",
                MANIFEST_NAME,
                "-e",
                "build",
                "-e",
                PREVIOUS_DIR_NAME,
                "-e",
                PREVIOUS_JSON_NAME,
            ],
            check=False,
        )
        if clean.returncode != 0:
            raise EngineError(f"git clean failed in {directory}: {clean.stderr.strip()}")

    updated_manifest = load_manifest(directory) or manifest
    updated_manifest.remote = fetch_remote
    updated_manifest.branch = branch
    # Keep the last-good commit recorded until rebuild() succeeds, so a failed
    # build after the reset still knows which commit the rolled-back artifacts
    # came from (and can report that the tree is now ahead).
    updated_manifest.commit = manifest.commit or head
    save_manifest(directory, updated_manifest)
    return rebuild(directory)


def matches_too_old_signature(stderr: str) -> bool:
    lowered = stderr.lower()
    return any(signature in lowered for signature in ENGINE_TOO_OLD_SIGNATURES)
