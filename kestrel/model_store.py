from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .errors import CorruptModelError, KestrelError, MissingModelError
from .util import write_atomic


class ModelStoreError(KestrelError):
    """A model source could not be queried, acquired, or resolved safely."""

    code = "model_store_error"


@dataclass(frozen=True)
class OllamaModel:
    name: str
    model_id: str
    size: str
    modified: str
    local_path: Path | None = None

    @property
    def is_cloud(self) -> bool:
        return self.local_path is None


@dataclass(frozen=True)
class HuggingFaceDownload:
    repository: str
    directory: Path
    process: subprocess.CompletedProcess

    @property
    def stdout(self) -> str:
        return self.process.stdout


def default_models_dir() -> Path:
    override = os.environ.get("KESTREL_MODELS_DIR")
    if override:
        return Path(override).expanduser()
    return Path("~/.local/share/kestrel/models").expanduser()


def _hf_cli(*, modern_only: bool = False) -> str | None:
    """Locate the Hugging Face CLI binary, or ``None`` when unavailable.

    ``modern_only`` requires the current ``hf`` binary (``huggingface-cli``
    lacks ``hf models ls``); ``pull`` works with either.
    """
    binary = shutil.which("hf") or shutil.which("huggingface-cli")
    if not binary:
        return None
    if modern_only and Path(binary).name != "hf":
        return None
    return binary


def hf_snapshot_dir(path: str | Path, *, marker: str = "config.json") -> Path | None:
    """Resolve the materialized HF snapshot for a cached repo root, if any.

    Prefers the newly downloaded ``refs/main`` pin, then falls back to the
    most recently modified snapshot, mirroring ``huggingface_hub``'s own
    resolution. Returns ``None`` when the directory is not an HF cache tree.
    """
    root = Path(path).expanduser()
    if (root / marker).is_file():
        return root
    refs_main = root / "refs" / "main"
    snapshots = root / "snapshots"

    def valid_snapshot(candidate: Path) -> Path | None:
        try:
            resolved_root = snapshots.resolve()
            resolved = candidate.resolve()
        except OSError:
            return None
        if resolved == resolved_root or resolved_root not in resolved.parents:
            return None
        return resolved if (resolved / marker).is_file() else None

    if refs_main.is_file():
        try:
            revision = refs_main.read_text().strip()
        except OSError:
            revision = ""
        if revision:
            snapshot = valid_snapshot(snapshots / revision)
            if snapshot is not None:
                return snapshot
    if snapshots.is_dir():
        try:
            items = list(snapshots.iterdir())
        except OSError:
            return None

        def modified(item: Path) -> float:
            try:
                return item.stat().st_mtime
            except OSError:
                return 0.0

        candidates = sorted(
            (resolved for item in items if (resolved := valid_snapshot(item)) is not None),
            key=modified,
            reverse=True,
        )
        if candidates:
            return candidates[0]
    return None


def _run(command: list[str], *, timeout: float = 60.0) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise ModelStoreError(f"required command is not installed: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ModelStoreError(f"command timed out: {command[0]}") from exc
    except OSError as exc:
        raise ModelStoreError(f"could not run {command[0]}: {exc}") from exc
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-2000:]
        raise ModelStoreError(f"{' '.join(command[:2])} failed: {detail}")
    return result


def _ollama_model_roots() -> list[str]:
    """Directories that may legitimately hold local Ollama model blobs."""
    roots = []
    if os.environ.get("OLLAMA_MODELS"):
        roots.append(os.environ["OLLAMA_MODELS"])
    roots.extend(
        [
            os.path.join(os.path.expanduser("~"), ".ollama", "models"),
            "/usr/share/ollama/.ollama/models",
            "/opt/ollama/.ollama/models",
        ]
    )
    return roots


def _is_ollama_blob(path: Path) -> bool:
    try:
        candidate = path.resolve()
    except OSError:
        return False
    for root in _ollama_model_roots():
        root_path = Path(root).expanduser().resolve()
        if root_path != candidate and root_path in candidate.parents:
            return candidate.is_file()
    return False


def resolve_ollama_blob(name: str) -> Path | None:
    """Resolve an Ollama model to its immutable local GGUF/blob, if it has one.

    Reads the model's manifest JSON straight off disk (no Ollama daemon round
    trip). Falls back to ``ollama show --modelfile`` if the manifest layout is
    unusual or a model referenced a non-manifest source. Results are TTL-cached
    so listing many models never re-parses the same file.
    """
    cached = _ollama_blob_cached(name)
    if cached is not _UNSET:
        return cached
    blob = _resolve_blob_from_manifest(name) or _resolve_ollama_blob_shell(name)
    return _ollama_blob_cache_set(name, blob)


def _ollama_manifest_paths(name: str) -> list[tuple[Path, Path]]:
    """Candidate ``(models_root, manifest_file)`` pairs for ``name``."""
    reference = name.strip()
    prefix = "registry.ollama.ai/"
    if reference.startswith(prefix):
        reference = reference.removeprefix(prefix)
    tail = reference.rsplit("/", 1)[-1]
    if ":" in tail:
        base, tag = reference.rsplit(":", 1)
    else:
        base, tag = reference, "latest"
    components = base.split("/")
    valid_component = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    if not components or any(not valid_component.fullmatch(item) for item in components):
        raise ModelStoreError(f"invalid Ollama model reference: {name}")
    if not valid_component.fullmatch(tag):
        raise ModelStoreError(f"invalid Ollama model tag: {tag}")
    if len(components) == 1:
        components.insert(0, "library")
    relative = Path("manifests") / "registry.ollama.ai" / Path(*components)
    result = []
    for raw_root in _ollama_model_roots():
        root = Path(raw_root).expanduser()
        if root.is_dir():
            # Ollama's native layout uses the bare tag filename. Keep the
            # older .json candidate for stores produced by earlier tooling.
            result.extend(((root, root / relative / tag), (root, root / relative / f"{tag}.json")))
    return result


def _resolve_blob_from_manifest(name: str) -> Path | None:
    """Map an Ollama model to its GGUF blob from its disk manifest, if possible."""
    for root, manifest_path in _ollama_manifest_paths(name):
        try:
            payload = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("layers", []), list):
            continue
        for layer in payload.get("layers", []):
            if not isinstance(layer, dict):
                continue
            media_type = layer.get("mediaType", "")
            if "gguf" not in media_type and "model" not in media_type:
                continue
            digest = layer.get("digest", "") or ""
            if not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", digest):
                continue
            blob = root / "blobs" / digest.replace(":", "-")
            if blob.is_file() and _is_ollama_blob(blob):
                return blob.resolve()
    return None


def _resolve_ollama_blob_shell(name: str) -> Path | None:
    result = _run(["ollama", "show", name, "--modelfile"])
    for line in result.stdout.splitlines():
        if not line.startswith("FROM "):
            continue
        source = line[5:].strip().strip('"')
        candidate = Path(source).expanduser()
        if candidate.is_file() and _is_ollama_blob(candidate):
            return candidate.resolve()
        # Cloud models and model-name parents have no directly reusable blob.
        return None
    return None


_UNSET = object()
_ollama_blob_cache: dict[str, tuple[float, Path | None]] = {}


def _ollama_blob_cached(name: str) -> Path | None:
    entry = _ollama_blob_cache.get(name)
    if entry is None or time.monotonic() - entry[0] > _ollama_list_ttl:
        return _UNSET
    return entry[1]


def _ollama_blob_cache_set(name: str, value: Path | None) -> Path | None:
    _ollama_blob_cache[name] = (time.monotonic(), value)
    return value


def list_ollama_models(*, resolve_paths: bool = False) -> list[OllamaModel]:
    rows = _ollama_list_cached()
    if rows is None:
        rows = _ollama_list()
        _ollama_list_cache_update(rows)
    if resolve_paths and rows:
        with ThreadPoolExecutor(max_workers=min(8, len(rows))) as executor:
            paths = list(executor.map(resolve_ollama_blob, (item.name for item in rows)))
        rows = [
            OllamaModel(item.name, item.model_id, item.size, item.modified, path)
            for item, path in zip(rows, paths, strict=True)
        ]
    return rows


_ollama_list_ttl = 4.0
_ollama_list_cache: tuple[float, list[OllamaModel]] | None = None


def _ollama_list() -> list[OllamaModel]:
    result = _run(["ollama", "list"])
    rows = []
    for line in result.stdout.splitlines()[1:]:
        columns = re.split(r"\s{2,}", line.strip(), maxsplit=3)
        if len(columns) < 2:
            continue
        name, model_id = columns[:2]
        size = columns[2] if len(columns) > 2 else "unknown"
        modified = columns[3] if len(columns) > 3 else "unknown"
        rows.append(OllamaModel(name, model_id, size, modified, None))
    return rows


def _ollama_list_cache_update(rows: list[OllamaModel]) -> None:
    global _ollama_list_cache
    _ollama_list_cache = (time.monotonic(), rows)


def _ollama_list_cached() -> list[OllamaModel] | None:
    """Return the last listing when it is fresh enough for the menu.

    Re-spawning ``ollama list`` on every menu/model open is wasteful on hosts
    with many models; a short TTL keeps the menu snappy while staying fresh.
    """
    if _ollama_list_cache is None:
        return None
    at, rows = _ollama_list_cache
    if time.monotonic() - at > _ollama_list_ttl:
        return None
    return rows


def pull_ollama(name: str) -> OllamaModel:
    global _ollama_list_cache
    _run(["ollama", "pull", name], timeout=24 * 60 * 60)
    # The just-pulled model may post date a cached listing; force a fresh read.
    _ollama_list_cache = None
    path = resolve_ollama_blob(name)
    match = next((item for item in list_ollama_models() if item.name == name), None)
    return OllamaModel(
        name=name,
        model_id=match.model_id if match else "unknown",
        size=match.size if match else "unknown",
        modified=match.modified if match else "just now",
        local_path=path,
    )


def pull_huggingface(
    repo_id: str,
    *,
    filename: str | None = None,
    include: str | None = None,
    revision: str | None = None,
    destination: Path | None = None,
    dry_run: bool = False,
) -> HuggingFaceDownload:
    """Acquire an HF model with the official CLI and a deterministic destination."""

    repo = repo_id.removeprefix("hf://")
    if "/" not in repo:
        raise ModelStoreError("Hugging Face source must be OWNER/REPOSITORY")
    if filename and include:
        raise ModelStoreError("choose either an exact --file or an --include glob")
    if filename:
        if filename.startswith("-"):
            raise ModelStoreError("filenames may not start with '-'; use --include to filter instead")
        file_path = Path(filename)
        if file_path.is_absolute() or ".." in file_path.parts:
            raise ModelStoreError("filenames must stay within the Hugging Face repository")
    binary = _hf_cli()
    if not binary:
        raise ModelStoreError("Hugging Face CLI is missing; install `huggingface_hub[cli]`")
    target = destination or default_models_dir() / repo.replace("/", "--")
    target.mkdir(parents=True, exist_ok=True)
    command = [binary, "download", repo]
    if filename:
        command.append(filename)
    if include:
        command.extend(["--include", include])
    if revision:
        command.extend(["--revision", revision])
    command.extend(["--local-dir", str(target), "--format", "json"])
    if dry_run:
        command.append("--dry-run")
    process = _run(command, timeout=24 * 60 * 60)
    if not dry_run:
        manifest = {
            "schema_version": 1,
            "provider": "huggingface",
            "repository": repo,
            "revision": revision or "default",
            "file": filename,
            "include": include,
            "acquired_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        write_atomic(target / ".kestrel-source.json", json.dumps(manifest, indent=2) + "\n")
    return HuggingFaceDownload(repo, target.resolve(), process)


def search_huggingface(query: str, *, limit: int = 10) -> list[dict]:
    """Search current GGUF repositories through Hugging Face's official CLI."""

    binary = _hf_cli(modern_only=True)
    if binary is None:
        raise ModelStoreError("model search requires the current `hf` CLI")
    if not query.strip():
        raise ModelStoreError("search query cannot be empty")
    result = _run(
        [
            binary,
            "models",
            "ls",
            "--search",
            query.strip(),
            "--filter",
            "gguf",
            "--sort",
            "downloads",
            "--limit",
            str(max(1, min(limit, 50))),
            "--expand",
            "downloads,likes,lastModified,tags",
            "--format",
            "json",
        ]
    )
    try:
        rows = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ModelStoreError("Hugging Face search returned invalid JSON") from exc
    if not isinstance(rows, list):
        raise ModelStoreError("Hugging Face search returned an unexpected response")
    result = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("id"):
            continue
        tags = row.get("tags") if isinstance(row.get("tags"), list) else []
        try:
            downloads = int(row.get("downloads") or 0)
            likes = int(row.get("likes") or 0)
        except (TypeError, ValueError) as exc:
            raise ModelStoreError("Hugging Face search returned invalid popularity fields") from exc
        result.append(
            {
                "id": str(row["id"]),
                "downloads": downloads,
                "likes": likes,
                "last_modified": row.get("last_modified") or row.get("lastModified"),
                "license": next(
                    (
                        tag.removeprefix("license:")
                        for tag in tags
                        if isinstance(tag, str) and tag.startswith("license:")
                    ),
                    None,
                ),
            }
        )
    return result


def list_huggingface_ggufs(repo_id: str) -> list[dict]:
    """List downloadable GGUF files and integrity metadata for a Hub repo."""

    binary = _hf_cli(modern_only=True)
    if binary is None:
        raise ModelStoreError("model file listing requires the current `hf` CLI")
    repo = repo_id.removeprefix("hf://")
    if "/" not in repo:
        raise ModelStoreError("Hugging Face source must be OWNER/REPOSITORY")
    result = _run([binary, "models", "ls", repo, "-R", "--format", "json"])
    try:
        rows = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ModelStoreError("Hugging Face file listing returned invalid JSON") from exc
    files = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        path = str(row.get("path") or "")
        if not path.lower().endswith(".gguf"):
            continue
        security = row.get("security") if isinstance(row.get("security"), dict) else {}
        lfs = row.get("lfs") if isinstance(row.get("lfs"), dict) else {}
        try:
            size_bytes = int(row.get("size") or lfs.get("size") or 0)
        except (TypeError, ValueError) as exc:
            raise ModelStoreError(f"Hugging Face file listing returned an invalid size for {path}") from exc
        files.append(
            {
                "path": path,
                "size_bytes": size_bytes,
                "sha256": lfs.get("sha256"),
                "security_status": security.get("status") or "unknown",
            }
        )
    return sorted(files, key=lambda item: item["path"].lower())


_SPLIT_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


def split_shard(path: Path) -> tuple[str, int, int] | None:
    """Return ``(prefix, part, total)`` when ``path`` is a llama.cpp split shard.

    Split GGUFs follow the ``{prefix}-00001-of-00005.gguf`` convention produced
    by ``gguf-split`` and ``convert_hf_to_gguf.py --split-*``.
    """
    match = _SPLIT_RE.match(path.name)
    if not match:
        return None
    prefix, part, total = match.groups()
    return prefix, int(part), int(total)


def complete_gguf_models(paths: list[Path]) -> list[Path]:
    """Return one representative path per complete GGUF model.

    Standalone files are returned as-is. Shards of a split model are grouped by
    their shared prefix and total count; a complete set contributes its first
    shard (llama.cpp resolves sibling shards from it automatically), while
    incomplete sets are dropped so a half-downloaded split is not offered as a
    broken model. ``mmproj`` vision projections are treated as standalone files.
    """
    groups: dict[tuple[str, int], dict[int, Path]] = {}
    standalone: list[Path] = []
    for path in paths:
        shard = split_shard(path)
        if shard is None:
            standalone.append(path)
            continue
        prefix, part, total = shard
        groups.setdefault((prefix, total), {})[part] = path
    complete = []
    for (_prefix, total), shards in groups.items():
        if set(shards) == set(range(1, total + 1)):
            complete.append(min(shards.items())[1])
    return sorted(standalone + complete)


def _stat_model(path: Path) -> os.stat_result:
    """Stat a model path, surfacing low-level failures as typed errors.

    Missing files raise :class:`MissingModelError`; permission and other OS
    failures raise :class:`ModelStoreError` with an actionable hint. A real but
    unusable target (a directory or a zero-byte/truncated file) raises
    :class:`CorruptModelError` instead of leaking a raw ``OSError``.
    """
    try:
        st = path.stat()
    except FileNotFoundError as exc:
        raise MissingModelError(
            f"model not found on disk: {path}",
            hint="Re-download the model or check that the file still exists.",
        ) from exc
    except (OSError, ValueError) as exc:
        raise ModelStoreError(
            f"cannot stat model: {path}",
            hint=f"grant the process read permission on this file: {exc}",
        ) from exc
    if not stat.S_ISREG(st.st_mode):
        raise CorruptModelError(
            f"model path is not a regular file: {path}",
            hint="The path resolves to a directory or special file, not a GGUF model.",
        )
    if st.st_size <= 0:
        raise CorruptModelError(
            f"model file is empty or truncated: {path}",
            hint="Re-download the model; the file on disk has no usable bytes.",
        )
    return st


def model_total_size(path: Path) -> int:
    """On-disk bytes for a GGUF model, summing sibling shards of a split set.

    A standalone file is stat'd defensively; a missing or unreadable sibling of
    a split set is skipped so a graceful partial total is returned instead of a
    raw ``OSError``.
    """
    shard = split_shard(path)
    if shard is None:
        return _stat_model(path).st_size
    prefix, _part, total = shard
    total_bytes = 0
    for part in range(1, total + 1):
        sibling = path.with_name(f"{prefix}-{part:05d}-of-{total:05d}.gguf")
        try:
            total_bytes += sibling.stat().st_size
        except (OSError, ValueError):
            continue
    return total_bytes


def _walk_ggufs(root: Path) -> list[Path]:
    """Yield real, non-empty ``.gguf`` files under ``root``.

    Gracefully downgrades unusable entries instead of failing the whole walk:
    zero-byte downloads, files behind dangling symlinks, directory-like ``.gguf``
    names, and paths that recurse into permission-denied subtrees are skipped.
    Real (resolved) paths are deduplicated so file symlinks pointing at the same
    blob are reported once. Never follows directory symlinks, so cycles are
    impossible even on deeply nested stores.
    """
    found: set[Path] = set()
    try:
        for _dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                if not name.lower().endswith(".gguf"):
                    continue
                try:
                    resolved = (Path(_dirpath) / name).resolve()
                except (OSError, ValueError):
                    continue
                try:
                    if not resolved.is_file() or resolved.stat().st_size <= 0:
                        continue
                except (OSError, ValueError):
                    continue
                found.add(resolved)
    except (OSError, ValueError):
        # A store that becomes unreachable mid-walk must not abort the whole
        # discovery; report whatever usable entries were already collected.
        return sorted(found)
    return sorted(found)


def discover_local_models(root: Path | None = None) -> list[Path]:
    target = root or default_models_dir()
    if not target.exists():
        # A missing store is only fatal when the caller asked for a specific
        # non-default root; the default store simply has nothing to list yet.
        if root is not None and target != default_models_dir():
            raise MissingModelError(
                f"model store directory does not exist: {target}",
                hint=(
                    "Create the directory, run a pull/download first, or point KESTREL_MODELS_DIR at an existing store."
                ),
            )
        return []
    try:
        with os.scandir(target) as it:
            next(it, None)
    except OSError as exc:
        raise ModelStoreError(
            f"cannot read model store: {target}",
            hint=f"grant the process read permission on the directory: {exc}",
        ) from exc
    return _walk_ggufs(target)


def choose_default_gguf(paths: list[Path]) -> Path:
    """Select one model or the first shard of one complete split model."""

    candidates = [path for path in paths if not path.name.lower().startswith("mmproj")]
    choices = complete_gguf_models(candidates)
    if len(choices) != 1:
        raise ModelStoreError("download does not contain exactly one unambiguous complete GGUF model")
    return choices[0]
