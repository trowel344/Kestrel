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

NATIVE_LLAMA_CPP_DIRS = (
    "/tmp/llama.cpp-moe-cache",
    os.path.expanduser("~/llama.cpp-moe-cache"),
    os.path.expanduser("~/llama.cpp"),
)


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
    if override:
        return override
    for directory in _candidate_dirs(dirs):
        if _find_binary(directory, "llama-server") or _find_binary(
            directory, "llama-cli"
        ):
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
        return flag in self.help_text

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
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    return os.path.join(base, "kestrel", "llama-cli-capabilities.json")


def _capability_cache_read(binary: str) -> tuple[str, str] | None:
    """Return cached ``(help_text, version)`` when the binary is unchanged.

    Each llama-cli capability probe spawns ``--help`` and ``--version`` (and on
    some builds each can initialize a CUDA context, adding seconds to session
    startup). Persisting the result keyed by the binary's identity lets repeat
    launches skip both subprocesses entirely.
    """
    try:
        stat = os.stat(binary)
        key = f"{binary}\0{stat.st_mtime_ns}\0{stat.st_size}"
        with open(_capability_cache_path()) as f:
            data = json.load(f)
        if data.get("key") == key:
            return data["help"], data["version"]
    except (OSError, ValueError, KeyError):
        pass
    return None


def _capability_cache_write(binary: str, help_text: str, version: str) -> None:
    try:
        stat = os.stat(binary)
        key = f"{binary}\0{stat.st_mtime_ns}\0{stat.st_size}"
        path = _capability_cache_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump({"key": key, "help": help_text, "version": version}, f)
        os.replace(tmp, path)
    except OSError:
        pass


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
        n_threads: int = 0,
        llama_cpp_dir: str | None = None,
        moe_cache: str = "auto",
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
        return [self.llama_cpp_dir] + [
            directory
            for directory in _candidate_dirs()
            if directory != self.llama_cpp_dir
        ]

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
            raise RuntimeError(
                f"llama-cli was not found under {self.llama_cpp_dir} "
                "or any native llama.cpp build. Run 'kestrel build' or set "
                "KESTREL_LLAMA_CPP_DIR."
            )
        return binary

    @property
    def server_binary(self) -> str:
        binary = self._find_bin("llama-server")
        if not binary:
            raise RuntimeError(
                f"llama-server was not found under {self.llama_cpp_dir} "
                "or any native llama.cpp build. Run 'kestrel build' or set "
                "KESTREL_LLAMA_CPP_DIR."
            )
        return binary

    def server_capabilities(self, refresh: bool = False) -> LlamaCppCapabilities:
        if self._server_capabilities is not None and not refresh:
            return self._server_capabilities
        binary = self.server_binary
        cached = None if refresh else _capability_cache_read(binary)
        if cached is not None:
            help_text, version = cached
        else:
            help_result = subprocess.run(
                [binary, "--help"], capture_output=True, text=True, timeout=15
            )
            version_result = subprocess.run(
                [binary, "--version"], capture_output=True, text=True, timeout=15
            )
            help_text = help_result.stdout + help_result.stderr
            version = (version_result.stdout or version_result.stderr).strip()
            _capability_cache_write(binary, help_text, version)
        self._server_capabilities = LlamaCppCapabilities(help_text=help_text, version=version)
        return self._server_capabilities

    def capabilities(self, refresh: bool = False) -> LlamaCppCapabilities:
        if self._capabilities is not None and not refresh:
            return self._capabilities
        binary = self.binary
        cached = None if refresh else _capability_cache_read(binary)
        if cached is not None:
            help_text, version = cached
        else:
            help_result = subprocess.run(
                [binary, "--help"], capture_output=True, text=True, timeout=15
            )
            version_result = subprocess.run(
                [binary, "--version"], capture_output=True, text=True, timeout=15
            )
            help_text = help_result.stdout + help_result.stderr
            version = (version_result.stdout or version_result.stderr).strip()
            _capability_cache_write(binary, help_text, version)
        self._capabilities = LlamaCppCapabilities(help_text=help_text, version=version)
        return self._capabilities

    def _resolved_spec_type(self, caps: LlamaCppCapabilities) -> str | None:
        if self.spec_type in ("", "none", None):
            return None
        requested = self.spec_type
        if requested == "mtp":
            requested = "draft-mtp"
        return requested if requested in caps.spec_types else None

    def _base_cmd(self) -> list[str]:
        caps = self.capabilities()
        if not Path(self.model_path).is_file():
            raise FileNotFoundError(f"GGUF model not found: {self.model_path}")

        cmd = [
            self.binary,
            "-m", self.model_path,
            "-ngl", str(self.n_gpu_layers),
            "-c", str(self.n_ctx),
            "-b", str(self.n_batch),
            "-ub", str(self.n_ubatch),
            "--temp", str(self.temp),
            "--seed", str(self.seed),
        ]
        if self.n_threads and caps.supports("--threads"):
            cmd += ["--threads", str(self.n_threads)]
        if self.n_threads and caps.supports("--threads-batch"):
            cmd += ["--threads-batch", str(self.n_threads)]

        if self.use_mmap and caps.supports("--mmap"):
            cmd.append("--mmap")
        elif caps.supports("--no-mmap"):
            cmd.append("--no-mmap")
        if self.fit and caps.supports("--fit"):
            cmd += ["--fit", "on"]
            if caps.supports("--fit-target"):
                cmd += ["--fit-target", str(self.fit_target_mib)]
        if self.cpu_moe and caps.supports("--cpu-moe"):
            cmd.append("--cpu-moe")
        if self.moe_cache != "auto" and caps.supports("--moe-cache"):
            cmd += ["--moe-cache", self.moe_cache]
        if self.cache_type_k and caps.supports("--cache-type-k"):
            cmd += ["--cache-type-k", self.cache_type_k]
        if self.cache_type_v and caps.supports("--cache-type-v"):
            cmd += ["--cache-type-v", self.cache_type_v]
        if caps.supports("--flash-attn"):
            cmd += ["--flash-attn", "auto"]

        spec_type = self._resolved_spec_type(caps)
        if spec_type:
            cmd += ["--spec-type", spec_type]
            if caps.supports("--spec-draft-n-max"):
                cmd += ["--spec-draft-n-max", str(self.spec_draft_n)]
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
            "-n", str(max_tokens),
            "--no-display-prompt",
            "-p", prompt,
        ]
        return cmd

    @staticmethod
    def _parse_metrics(stderr: str, elapsed: float, returncode: int) -> RunMetrics:
        metrics = RunMetrics(elapsed_seconds=elapsed, returncode=returncode)
        patterns = {
            "prompt_tokens": r"^\S+:\s+prompt eval time\s*=.*?/\s*(\d+) tokens",
            "output_tokens": r"^\S+:\s+eval time\s*=.*?/\s*(\d+) runs",
            "prompt_tokens_per_second": (
                r"^\S+:\s+prompt eval time\s*=.*?([\d.]+) tokens per second"
            ),
            "output_tokens_per_second": (
                r"^\S+:\s+eval time\s*=.*?([\d.]+) tokens per second"
            ),
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
            raise RuntimeError(
                "llama.cpp generation exceeded 30 minutes and was stopped"
            ) from exc
        elapsed = time.perf_counter() - started
        self.last_metrics = self._parse_metrics(result.stderr, elapsed, result.returncode)
        if result.returncode != 0:
            detail = result.stderr.strip()[-2000:]
            raise RuntimeError(f"llama.cpp failed with exit {result.returncode}:\n{detail}")
        return result.stdout.strip()

    def generate_stream(self, prompt: str, max_tokens: int = 256) -> Iterator[str]:
        cmd = self._build_cmd(prompt, max_tokens)
        self._stderr_file = tempfile.TemporaryFile(mode="w+")
        started = time.perf_counter()
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=self._stderr_file,
            text=True,
            bufsize=1,
        )
        assert self._proc.stdout is not None
        while True:
            chunk = self._proc.stdout.readline()
            if chunk == "":
                break
            yield chunk
        returncode = self._proc.wait()
        elapsed = time.perf_counter() - started
        self._stderr_file.seek(0)
        stderr = self._stderr_file.read()
        self.last_metrics = self._parse_metrics(stderr, elapsed, returncode)
        if returncode != 0:
            raise RuntimeError(f"llama.cpp failed with exit {returncode}:\n{stderr[-2000:]}")

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
