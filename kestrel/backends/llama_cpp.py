from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

from .. import util
from ..errors import BackendError

NATIVE_LLAMA_CPP_DIRS = (
    os.path.expanduser("~/llama.cpp-moe-cache"),
    os.path.expanduser("~/llama.cpp"),
)

_CAPABILITY_CACHE_VERSION = 2
_CAPABILITY_CACHE_MAX_ENTRIES = 8
_CAPABILITY_CACHE_MAX_BYTES = 2 * 1024 * 1024


def _find_binary(directory: str, name: str) -> str | None:
    """Return the executable path for ``name`` under a llama.cpp build tree."""
    for subdir in (
        os.path.join(directory, "build", "bin"),
        os.path.join(directory, "build"),
        directory,
    ):
        path = os.path.join(subdir, name)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def _candidate_dirs(dirs: tuple[str, ...] | None = None) -> list[str]:
    """Search order: KESTREL_LLAMA_CPP_DIR override, then the known builds."""
    override = os.environ.get("KESTREL_LLAMA_CPP_DIR")
    ordered = ([override] if override else []) + list(dirs or NATIVE_LLAMA_CPP_DIRS)
    seen: set[str] = set()
    result: list[str] = []
    for directory in ordered:
        if directory and directory not in seen:
            seen.add(directory)
            result.append(directory)
    return result


def default_llama_cpp_dir(dirs: tuple[str, ...] | None = None) -> str:
    """Kestrel's native llama.cpp directory.

    Prefers the MoE-native build (the fork that can load the corrected NVFP4
    GGUFs) over the stock upstream build, so running and benchmarking work
    without setting KESTREL_LLAMA_CPP_DIR. An explicit override still wins.
    """
    override = os.environ.get("KESTREL_LLAMA_CPP_DIR")
    ordered = _candidate_dirs(dirs)
    if override and ordered and ordered[0] == override:
        return override
    for directory in ordered:
        if _find_binary(directory, "llama-server") or _find_binary(directory, "llama-cli"):
            return directory
    return os.path.expanduser("~/llama.cpp")


def resolve_llama_binary(name: str, dirs: tuple[str, ...] | None = None) -> str | None:
    """Find a llama.cpp executable across the native build locations."""
    for directory in _candidate_dirs(dirs):
        found = _find_binary(directory, name)
        if found:
            return found
    return None


@dataclass(frozen=True)
class LlamaCppCapabilities:
    help_text: str
    version: str = "unknown"

    def supports(self, flag: str) -> bool:
        # Avoid treating ``--rpc-server`` as support for ``--rpc`` (or
        # ``--fit-target`` as support for ``--fit``). Capability gating must
        # reflect the actual option exposed by this engine binary.
        return re.search(rf"(?<![A-Za-z0-9_-]){re.escape(flag)}(?![A-Za-z0-9_-])", self.help_text) is not None

    @property
    def spec_types(self) -> set[str]:
        match = re.search(r"--spec-type\s+([^\n]+)", self.help_text)
        if not match:
            return set()
        return set(match.group(1).split()[0].split(","))


@dataclass
class RunMetrics:
    elapsed_seconds: float = 0.0
    returncode: int = 0
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    prompt_tokens_per_second: float | None = None
    output_tokens_per_second: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _capability_cache_path() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "kestrel", "llama-cli-capabilities.json")


def _capability_cache_key(binary: str) -> str:
    """Identify one concrete binary build, including atomic replacements."""
    stat = os.stat(binary)
    return "\0".join(
        str(value)
        for value in (
            os.path.realpath(binary),
            stat.st_dev,
            stat.st_ino,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
            stat.st_size,
        )
    )


def _capability_cache_read(binary: str) -> tuple[str, str] | None:
    """Return cached ``(help_text, version)`` when the binary is unchanged.

    Each llama-cli capability probe spawns ``--help`` and ``--version`` (and on
    some builds each can initialize a CUDA context, adding seconds to session
    startup). Persisting the result keyed by the binary's identity lets repeat
    launches skip both subprocesses entirely.
    """
    try:
        path = _capability_cache_path()
        if os.stat(path).st_size > _CAPABILITY_CACHE_MAX_BYTES:
            return None
        key = _capability_cache_key(binary)
        with open(path) as f:
            data = json.load(f)
        entries = data.get("entries") if data.get("version") == _CAPABILITY_CACHE_VERSION else None
        entry = entries.get(key) if isinstance(entries, dict) else None
        if isinstance(entry, dict) and isinstance(entry.get("help"), str) and isinstance(entry.get("version"), str):
            return entry["help"], entry["version"]
    except (OSError, ValueError, KeyError, AttributeError, TypeError):
        pass
    return None


def _capability_cache_write(binary: str, help_text: str, version: str) -> None:
    try:
        key = _capability_cache_key(binary)
        path = _capability_cache_path()
        entries: dict[str, dict[str, str]] = {}
        try:
            if os.stat(path).st_size <= _CAPABILITY_CACHE_MAX_BYTES:
                with open(path) as handle:
                    cached = json.load(handle)
                loaded = cached.get("entries") if cached.get("version") == _CAPABILITY_CACHE_VERSION else None
                if isinstance(loaded, dict):
                    entries = {
                        cached_key: entry
                        for cached_key, entry in loaded.items()
                        if isinstance(cached_key, str)
                        and isinstance(entry, dict)
                        and isinstance(entry.get("help"), str)
                        and isinstance(entry.get("version"), str)
                    }
        except (OSError, ValueError, KeyError, AttributeError, TypeError):
            pass

        # Reinsert the current binary last, then retain only the newest bounded
        # set. A bounded inventory lets cli/server and multiple builds coexist
        # without turning the cache into an unbounded history file.
        entries.pop(key, None)
        entries[key] = {"help": help_text, "version": version}
        entries = dict(list(entries.items())[-_CAPABILITY_CACHE_MAX_ENTRIES:])
        util.write_atomic(
            path,
            json.dumps({"version": _CAPABILITY_CACHE_VERSION, "entries": entries}),
            backup=False,
        )
    except OSError:
        pass


def _load_capabilities(binary: str, refresh: bool) -> LlamaCppCapabilities:
    """Probe ``binary``'s capabilities, using/populating the on-disk cache.

    Avoids the per-launch ``--help``/``--version`` subprocesses (and the CUDA
    contexts some builds initialize per invocation) by keying the persisted
    result on the binary's identity unless ``refresh`` forces a re-probe.
    """
    cached = None if refresh else _capability_cache_read(binary)
    if cached is not None:
        help_text, version = cached
    else:
        try:
            help_result = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=15)
            version_result = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=15)
        except subprocess.TimeoutExpired as exc:
            raise BackendError(f"llama.cpp capability probe timed out: {binary}") from exc
        except OSError as exc:
            raise BackendError(f"could not probe llama.cpp binary {binary}: {exc}") from exc
        help_text = help_result.stdout + help_result.stderr
        version = (version_result.stdout or version_result.stderr).strip()
        if not help_text.strip():
            raise BackendError(f"llama.cpp capability probe produced no help output: {binary}")
        _capability_cache_write(binary, help_text, version)
    return LlamaCppCapabilities(help_text=help_text, version=version)


class LlamaCppBackend:
    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int | str = "auto",
        n_ctx: int = 2048,
        n_batch: int = 512,
        n_ubatch: int = 128,
        temp: float = 0.7,
        seed: int = 42,
        spec_type: str = "none",
        spec_draft_n: int = 3,
        cpu_moe: bool = False,
        fit: bool = True,
        fit_target_mib: int = 1024,
        cache_type_k: str = "q8_0",
        cache_type_v: str = "q8_0",
        use_mmap: bool = True,
        use_mlock: bool = False,
        n_threads: int = 0,
        llama_cpp_dir: str | None = None,
        moe_cache: str = "auto",
        direct_io: bool = False,
        tensor_split: str | None = None,
        rpc_endpoints: list[str] | tuple[str, ...] | None = None,
        extra_args: list[str] | None = None,
    ):
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.n_ubatch = n_ubatch
        self.temp = temp
        self.seed = seed
        self.spec_type = spec_type
        self.spec_draft_n = spec_draft_n
        self.cpu_moe = cpu_moe
        self.fit = fit
        self.fit_target_mib = fit_target_mib
        self.cache_type_k = cache_type_k
        self.cache_type_v = cache_type_v
        self.use_mmap = use_mmap
        self.use_mlock = use_mlock
        self.direct_io = direct_io
        self.tensor_split = tensor_split
        self.rpc_endpoints = tuple(str(endpoint).strip() for endpoint in (rpc_endpoints or ()) if str(endpoint).strip())
        self.extra_args = list(extra_args or ())
        self.n_threads = max(0, n_threads)
        self.moe_cache = moe_cache
        self.llama_cpp_dir = llama_cpp_dir or default_llama_cpp_dir()
        self._proc: subprocess.Popen | None = None
        self._stderr_file = None
        self._capabilities: LlamaCppCapabilities | None = None
        self._server_capabilities: LlamaCppCapabilities | None = None
        self.last_metrics = RunMetrics()

    def _search_dirs(self) -> list[str]:
        """The selected build first, then the other native builds as fallback."""
        return [self.llama_cpp_dir] + [directory for directory in _candidate_dirs() if directory != self.llama_cpp_dir]

    def _find_bin(self, name: str) -> str | None:
        for directory in self._search_dirs():
            found = _find_binary(directory, name)
            if found:
                return found
        return None

    @property
    def binary(self) -> str:
        binary = self._find_bin("llama-cli") or self._find_bin("main")
        if not binary:
            raise BackendError(
                f"llama-cli was not found under {self.llama_cpp_dir} "
                "or any native llama.cpp build. Run 'kestrel build' or set "
                "KESTREL_LLAMA_CPP_DIR."
            )
        return binary

    @property
    def server_binary(self) -> str:
        binary = self._find_bin("llama-server")
        if not binary:
            raise BackendError(
                f"llama-server was not found under {self.llama_cpp_dir} "
                "or any native llama.cpp build. Run 'kestrel build' or set "
                "KESTREL_LLAMA_CPP_DIR."
            )
        return binary

    def server_capabilities(self, refresh: bool = False) -> LlamaCppCapabilities:
        if self._server_capabilities is not None and not refresh:
            return self._server_capabilities
        self._server_capabilities = _load_capabilities(self.server_binary, refresh)
        return self._server_capabilities

    def capabilities(self, refresh: bool = False) -> LlamaCppCapabilities:
        if self._capabilities is not None and not refresh:
            return self._capabilities
        self._capabilities = _load_capabilities(self.binary, refresh)
        return self._capabilities

    def _resolved_spec_type(self, caps: LlamaCppCapabilities) -> str | None:
        if self.spec_type in ("", "none", None):
            return None
        requested = self.spec_type
        if requested == "mtp":
            requested = "draft-mtp"
        return requested if requested in caps.spec_types else None

    def _common_engine_args(self, caps: LlamaCppCapabilities) -> list[str]:
        """Flags shared by llama-cli (interactive/one-shot) and llama-server.

        Single source of truth for model sizing and engine tuning so the CLI
        does not re-assemble them and drift (e.g. the server path once dropped
        ``--fit``, letting serve OOM where run would not).
        """
        args = [
            "-ngl",
            str(self.n_gpu_layers),
            "-c",
            str(self.n_ctx),
            "-b",
            str(self.n_batch),
            "-ub",
            str(self.n_ubatch),
        ]
        if self.n_threads and caps.supports("--threads"):
            args += ["--threads", str(self.n_threads)]
        if self.n_threads and caps.supports("--threads-batch"):
            args += ["--threads-batch", str(self.n_threads)]

        # Direct I/O bypasses the page cache and loads uncached model weights
        # at sequential disk speed; it is fastest on a cold launch from NVMe.
        # It is opt-in (the default mmap path wins on warm, cache-resident
        # reloads), and direct I/O implicitly disables mmap.
        if self.direct_io and caps.supports("--direct-io"):
            args += ["--no-mmap", "--direct-io"]
        elif self.use_mmap and caps.supports("--mmap"):
            args.append("--mmap")
        elif caps.supports("--no-mmap"):
            args.append("--no-mmap")
        if self.tensor_split and caps.supports("--tensor-split"):
            args += ["--tensor-split", self.tensor_split]
        if self.rpc_endpoints:
            # RPC is an explicit distributed execution request. Never drop it
            # when an older llama.cpp is selected: silently falling back to a
            # local-only launch can load the wrong placement or OOM.
            if not caps.supports("--rpc"):
                raise BackendError(
                    "selected llama.cpp engine does not support RPC nodes",
                    hint="rebuild the engine with -DGGML_RPC=ON or remove --node/--nodes",
                )
            args += ["--rpc", ",".join(self.rpc_endpoints)]
        if self.use_mlock and caps.supports("--mlock"):
            args.append("--mlock")
        if self.fit and caps.supports("--fit"):
            args += ["--fit", "on"]
            if caps.supports("--fit-target"):
                args += ["--fit-target", str(self.fit_target_mib)]
        if self.cpu_moe and caps.supports("--cpu-moe"):
            args.append("--cpu-moe")
        if self.moe_cache != "auto" and caps.supports("--moe-cache"):
            args += ["--moe-cache", self.moe_cache]
        if self.cache_type_k and caps.supports("--cache-type-k"):
            args += ["--cache-type-k", self.cache_type_k]
        if self.cache_type_v and caps.supports("--cache-type-v"):
            args += ["--cache-type-v", self.cache_type_v]
        if caps.supports("--flash-attn"):
            args += ["--flash-attn", "auto"]

        spec_type = self._resolved_spec_type(caps)
        if spec_type:
            args += ["--spec-type", spec_type]
            if caps.supports("--spec-draft-n-max"):
                args += ["--spec-draft-n-max", str(self.spec_draft_n)]
        if self.extra_args:
            args += self.extra_args
        return args

    def _base_cmd(self) -> list[str]:
        caps = self.capabilities()
        if not Path(self.model_path).is_file():
            raise FileNotFoundError(f"GGUF model not found: {self.model_path}")

        cmd = [
            self.binary,
            "-m",
            self.model_path,
            "--temp",
            str(self.temp),
            "--seed",
            str(self.seed),
        ]
        cmd += self._common_engine_args(caps)
        return cmd

    def build_server_cmd(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8080,
        alias: str | None = None,
        embeddings: bool = False,
    ) -> list[str]:
        caps = self.server_capabilities()
        cmd = [
            self.server_binary,
            "-m",
            self.model_path,
            "--host",
            host,
            "--port",
            str(port),
        ]
        if alias:
            cmd += ["--alias", alias]
        cmd += self._common_engine_args(caps)
        if embeddings and caps.supports("--embeddings"):
            cmd.append("--embeddings")
        return cmd

    def _build_interactive_cmd(self) -> list[str]:
        return self._base_cmd()

    def _build_cmd(self, prompt: str, max_tokens: int) -> list[str]:
        caps = self.capabilities()
        cmd = self._base_cmd()
        # Models with a chat template auto-enable conversation mode in recent
        # llama.cpp builds. A captured one-shot generation must explicitly
        # exit after its first response or EOF can produce an unbounded stream
        # of empty interactive prompts.
        if caps.supports("--single-turn"):
            cmd.append("--single-turn")
        if caps.supports("--simple-io"):
            cmd.append("--simple-io")
        if caps.supports("--perf"):
            cmd.append("--perf")
        cmd += [
            "-n",
            str(max_tokens),
            "--no-display-prompt",
            "-p",
            prompt,
        ]
        return cmd

    @staticmethod
    def _parse_metrics(stderr: str, elapsed: float, returncode: int) -> RunMetrics:
        metrics = RunMetrics(elapsed_seconds=elapsed, returncode=returncode)
        patterns = {
            "prompt_tokens": r"^\S+:\s+prompt eval time\s*=.*?/\s*(\d+) tokens",
            "output_tokens": r"^\S+:\s+eval time\s*=.*?/\s*(\d+) runs",
            "prompt_tokens_per_second": (r"^\S+:\s+prompt eval time\s*=.*?([\d.]+) tokens per second"),
            "output_tokens_per_second": (r"^\S+:\s+eval time\s*=.*?([\d.]+) tokens per second"),
        }
        for field, pattern in patterns.items():
            match = re.search(pattern, stderr, flags=re.MULTILINE)
            if match:
                value = float(match.group(1)) if "second" in field else int(match.group(1))
                setattr(metrics, field, value)
        return metrics

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        cmd = self._build_cmd(prompt, max_tokens)
        started = time.perf_counter()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        except subprocess.TimeoutExpired as exc:
            raise BackendError("llama.cpp generation exceeded 30 minutes and was stopped") from exc
        except OSError as exc:
            raise BackendError(f"could not start llama.cpp generation: {exc}") from exc
        elapsed = time.perf_counter() - started
        self.last_metrics = self._parse_metrics(result.stderr, elapsed, result.returncode)
        if result.returncode != 0:
            detail = util.truncate(result.stderr.strip())
            raise BackendError(f"llama.cpp failed with exit {result.returncode}:\n{detail}")
        return result.stdout.strip()

    def _finalize_stream(self, started: float, *, suppress_error: bool = False, terminate: bool = False) -> None:
        """Collect metrics and tear down the streaming subprocess exactly once.

        Runs even when a consumer abandons the generator early (GeneratorExit),
        so the child is always terminated and the temporary stderr file is
        always closed instead of leaking.
        """
        if self._proc is None:
            return
        err = None
        try:
            try:
                if terminate and self._proc.poll() is None:
                    self._proc.terminate()
                    try:
                        returncode = self._proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self._proc.kill()
                        returncode = self._proc.wait()
                else:
                    returncode = self._proc.wait()
            except OSError as exc:
                returncode = 1
                err = BackendError(f"could not finalize llama.cpp stream: {exc}")
            elapsed = time.perf_counter() - started
            self._stderr_file.seek(0)
            stderr = self._stderr_file.read()
            self.last_metrics = self._parse_metrics(stderr, elapsed, returncode)
            if returncode != 0 and err is None:
                err = BackendError(f"llama.cpp failed with exit {returncode}:\n{stderr[-2000:]}")
        finally:
            if self._proc.stdout is not None:
                self._proc.stdout.close()
            self._stderr_file.close()
            self._stderr_file = None
            self._proc = None
        if err is not None and not suppress_error:
            raise err

    def generate_stream(self, prompt: str, max_tokens: int = 256) -> Iterator[str]:
        cmd = self._build_cmd(prompt, max_tokens)
        self._stderr_file = tempfile.TemporaryFile(mode="w+")
        started = time.perf_counter()
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=self._stderr_file,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            self._stderr_file.close()
            self._stderr_file = None
            raise BackendError(f"could not start llama.cpp stream: {exc}") from exc
        terminating = False
        try:
            if self._proc.stdout is None:
                raise BackendError("llama.cpp stream has no stdout pipe")
            while True:
                chunk = self._proc.stdout.readline()
                if chunk == "":
                    break
                yield chunk
        except GeneratorExit:
            # Consumer stopped early: still clean up, but don't replace the
            # GeneratorExit with a failure raised from the cleanup path.
            terminating = True
            self._finalize_stream(started, suppress_error=True, terminate=True)
            raise
        finally:
            if not terminating:
                # Drain/record metrics and always clean up the child + temp
                # file, whether the stream completed or an error was raised.
                self._finalize_stream(started)

    def close(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
        if self._stderr_file:
            self._stderr_file.close()
            self._stderr_file = None
