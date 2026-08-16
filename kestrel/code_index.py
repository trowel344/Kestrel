"""Small Git-aware lexical code index for current workspace evidence.

This is intentionally separate from Sun Map's durable project memory.  Indexed
source is local, hash-versioned, bounded, and rendered as untrusted evidence.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path

MAX_FILE_BYTES = 512 * 1024
MAX_INDEX_BYTES = 32 * 1024 * 1024
MAX_FILES = 2000
CHUNK_LINES = 60
CHUNK_OVERLAP = 10

_TEXT_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".cs",
        ".css",
        ".gd",
        ".go",
        ".h",
        ".hpp",
        ".html",
        ".java",
        ".js",
        ".json",
        ".jsx",
        ".kt",
        ".lua",
        ".md",
        ".php",
        ".py",
        ".rb",
        ".rs",
        ".sh",
        ".sql",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".vue",
        ".xml",
        ".yaml",
        ".yml",
    }
)
_SECRET_NAMES = frozenset({".env", "id_rsa", "id_ed25519", "credentials", "secrets"})
_SECRET_SUFFIXES = frozenset({".key", ".pem", ".p12", ".pfx"})
_SYMBOL_PATTERNS = (
    re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE),
    re.compile(
        r"^\s*(?:export\s+)?(?:async\s+)?(?:function|class|interface|type)\s+"
        r"([A-Za-z_$][A-Za-z0-9_$]*)",
        re.MULTILINE,
    ),
    re.compile(r"^\s*(?:pub\s+)?(?:fn|struct|enum|trait)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE),
    re.compile(r"^\s*(?:func|type)\s+(?:\([^)]*\)\s*)?([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE),
)
_QUERY_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,63}")
_QUERY_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "answer",
        "current",
        "from",
        "have",
        "inspect",
        "into",
        "only",
        "please",
        "should",
        "that",
        "their",
        "this",
        "using",
        "what",
        "when",
        "where",
        "which",
        "with",
        "without",
    }
)


@dataclass(frozen=True)
class CodeEvidence:
    path: str
    start_line: int
    digest: str
    symbols: str
    content: str
    score: float


class CodeIndex:
    """Persistent FTS5 index over bounded, non-secret workspace text files."""

    def __init__(self, workspace: str | Path, database_path: str | Path) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        if not self.workspace.is_dir():
            raise ValueError(f"code-index workspace is not a directory: {self.workspace}")
        self.database_path = Path(database_path).expanduser()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()
        try:
            self.database_path.chmod(0o600)
        except OSError:
            pass

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS indexed_files (
                    path TEXT PRIMARY KEY,
                    digest TEXT NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    size_bytes INTEGER NOT NULL
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS code_chunks USING fts5(
                    path UNINDEXED,
                    start_line UNINDEXED,
                    digest UNINDEXED,
                    symbols,
                    content,
                    tokenize = 'unicode61 tokenchars _'
                );
                """
            )

    @staticmethod
    def _allowed(path: Path) -> bool:
        lower_name = path.name.lower()
        return (
            path.suffix.lower() in _TEXT_SUFFIXES
            and lower_name not in _SECRET_NAMES
            and path.stem.lower() not in _SECRET_NAMES
            and path.suffix.lower() not in _SECRET_SUFFIXES
            and not lower_name.startswith(".env.")
        )

    def _git_paths(self) -> list[Path] | None:
        try:
            result = subprocess.run(
                ["git", "ls-files", "-co", "--exclude-standard", "-z"],
                cwd=self.workspace,
                capture_output=True,
                check=False,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None
        paths = []
        for raw in result.stdout.split(b"\0"):
            if not raw:
                continue
            value = raw.decode("utf-8", "surrogateescape")
            candidate = (self.workspace / value).resolve()
            try:
                candidate.relative_to(self.workspace)
            except ValueError:
                continue
            paths.append(candidate)
        return paths

    def _workspace_paths(self) -> list[Path]:
        candidates = self._git_paths()
        if candidates is None:
            excluded = {".git", ".venv", "node_modules", "build", "dist", "__pycache__"}
            candidates = [
                path
                for path in self.workspace.rglob("*")
                if path.is_file() and not any(part in excluded for part in path.parts)
            ]
        result = []
        total = 0
        for path in sorted(set(candidates)):
            if len(result) >= MAX_FILES or not self._allowed(path):
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > MAX_FILE_BYTES or total + size > MAX_INDEX_BYTES:
                continue
            result.append(path)
            total += size
        return result

    @staticmethod
    def _read(path: Path) -> tuple[str, str] | None:
        try:
            payload = path.read_bytes()
        except OSError:
            return None
        if b"\0" in payload:
            return None
        text = payload.decode("utf-8", "replace")
        digest = hashlib.sha256(payload).hexdigest()[:16]
        return text, digest

    @staticmethod
    def _symbols(text: str) -> str:
        symbols = []
        for pattern in _SYMBOL_PATTERNS:
            symbols.extend(pattern.findall(text))
        return " ".join(list(dict.fromkeys(symbols))[:100]) if symbols else ""

    @staticmethod
    def _chunks(text: str) -> list[tuple[int, str]]:
        lines = text.splitlines()
        if not lines:
            return [(1, "")]
        stride = CHUNK_LINES - CHUNK_OVERLAP
        return [(start + 1, "\n".join(lines[start : start + CHUNK_LINES])) for start in range(0, len(lines), stride)]

    def refresh(self) -> dict[str, int]:
        """Synchronize changed files and remove deleted entries."""

        paths = self._workspace_paths()
        relative_paths = {str(path.relative_to(self.workspace)): path for path in paths}
        indexed = updated = removed = 0
        with self._connect() as db:
            existing = {
                row[0]: (row[1], int(row[2]), int(row[3]))
                for row in db.execute("SELECT path, digest, mtime_ns, size_bytes FROM indexed_files")
            }
            for relative in set(existing) - set(relative_paths):
                db.execute("DELETE FROM indexed_files WHERE path = ?", (relative,))
                db.execute("DELETE FROM code_chunks WHERE path = ?", (relative,))
                removed += 1
            for relative, path in relative_paths.items():
                try:
                    stat = path.stat()
                except OSError:
                    continue
                previous = existing.get(relative)
                if previous and previous[1:] == (stat.st_mtime_ns, stat.st_size):
                    indexed += 1
                    continue
                loaded = self._read(path)
                if loaded is None:
                    continue
                text, digest = loaded
                if previous and previous[0] == digest:
                    db.execute(
                        "UPDATE indexed_files SET mtime_ns = ?, size_bytes = ? WHERE path = ?",
                        (stat.st_mtime_ns, stat.st_size, relative),
                    )
                    indexed += 1
                    continue
                symbols = self._symbols(text)
                db.execute("DELETE FROM code_chunks WHERE path = ?", (relative,))
                db.executemany(
                    "INSERT INTO code_chunks(path, start_line, digest, symbols, content) VALUES (?, ?, ?, ?, ?)",
                    [(relative, start, digest, symbols, content) for start, content in self._chunks(text)],
                )
                db.execute(
                    "INSERT INTO indexed_files(path, digest, mtime_ns, size_bytes) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(path) DO UPDATE SET digest = excluded.digest, "
                    "mtime_ns = excluded.mtime_ns, size_bytes = excluded.size_bytes",
                    (relative, digest, stat.st_mtime_ns, stat.st_size),
                )
                updated += 1
            db.commit()
        return {"indexed": indexed + updated, "updated": updated, "removed": removed}

    def search(self, query: str, *, k: int = 5) -> list[CodeEvidence]:
        tokens = list(
            dict.fromkeys(
                token.lower() for token in _QUERY_TOKEN.findall(query) if token.lower() not in _QUERY_STOPWORDS
            )
        )[:12]
        if not tokens or k <= 0:
            return []
        expression = " OR ".join(f'"{token}"' for token in tokens)
        with self._connect() as db:
            rows = db.execute(
                "SELECT path, CAST(start_line AS INTEGER), digest, symbols, content, "
                "bm25(code_chunks, 0.0, 0.0, 0.0, 2.0, 1.0) AS score "
                "FROM code_chunks WHERE code_chunks MATCH ? ORDER BY score LIMIT ?",
                (expression, k),
            ).fetchall()
        return [
            CodeEvidence(
                path=str(row[0]),
                start_line=int(row[1]),
                digest=str(row[2]),
                symbols=str(row[3]),
                content=str(row[4]),
                score=float(row[5]),
            )
            for row in rows
        ]

    def stats(self) -> dict[str, int]:
        with self._connect() as db:
            files = int(db.execute("SELECT COUNT(*) FROM indexed_files").fetchone()[0])
            chunks = int(db.execute("SELECT COUNT(*) FROM code_chunks").fetchone()[0])
        return {"files": files, "chunks": chunks}


def _focused_body(content: str, query: str, max_lines: int = 32) -> tuple[int, str]:
    lines = content.splitlines()
    if len(lines) <= max_lines:
        return 0, content
    tokens = {token.lower() for token in _QUERY_TOKEN.findall(query) if token.lower() not in _QUERY_STOPWORDS}
    match = next(
        (index for index, line in enumerate(lines) if any(token in line.lower() for token in tokens)),
        0,
    )
    start = max(0, min(match - 8, len(lines) - max_lines))
    return start, "\n".join(lines[start : start + max_lines])


def _focused_symbols(symbols: str, query: str) -> str:
    values = symbols.split()
    query_tokens = {token.lower() for token in _QUERY_TOKEN.findall(query)}
    relevant = [value for value in values if value.lower() in query_tokens]
    return " ".join((relevant or values)[:8])


def _ordered_evidence(evidence: list[CodeEvidence], query: str) -> list[CodeEvidence]:
    """Prefer chunks that contain query terms and skip redundant overlap."""

    tokens = {token.lower() for token in _QUERY_TOKEN.findall(query) if token.lower() not in _QUERY_STOPWORDS}
    ranked = sorted(
        evidence,
        key=lambda item: (
            -sum(token in item.content.lower() for token in tokens),
            item.score,
            item.path,
            item.start_line,
        ),
    )
    selected: list[CodeEvidence] = []
    covered_by_path: dict[str, set[str]] = {}
    for item in ranked:
        content_tokens = {token for token in tokens if token in item.content.lower()}
        covered = covered_by_path.setdefault(item.path, set())
        if item.path in covered_by_path and selected and not (content_tokens - covered):
            if any(existing.path == item.path for existing in selected):
                continue
        selected.append(item)
        covered.update(content_tokens)
    return selected


def render_evidence(evidence: list[CodeEvidence], *, max_chars: int, query: str = "") -> str:
    parts = []
    used = 0
    for item in _ordered_evidence(evidence, query):
        line_offset, focused = _focused_body(item.content, query)
        symbols = _focused_symbols(item.symbols, query)
        header = (
            f"--- {item.path}:{item.start_line + line_offset} digest={item.digest}"
            + (f" symbols={symbols}" if symbols else "")
            + " ---\n"
        )
        remaining = max_chars - used - len(header)
        if remaining <= 0:
            break
        body = focused[:remaining]
        parts.append(header + body)
        used += len(header) + len(body)
    return "\n".join(parts)


__all__ = ["CodeEvidence", "CodeIndex", "render_evidence"]
