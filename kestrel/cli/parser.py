"""Argument surface: the one ``argparse`` tree for every Kestrel command.

``build_parser`` is the single authority for flags, defaults, and help; the
handlers in :mod:`kestrel.cli.run` and friends receive exactly the namespace
produced here. Version resolution lives here too (``--version`` needs it before
any subcommand import).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

from ..config import REASONING_LEVELS
from ..errors import InputError
from . import model_source, planning, state


class _ArgumentParser(argparse.ArgumentParser):
    """Preserve argparse's exit status while honoring the JSON CLI contract."""

    def parse_known_args(self, args=None, namespace=None):
        requested_args = sys.argv[1:] if args is None else args
        self._json_requested = "--json" in requested_args
        return super().parse_known_args(args, namespace)

    def error(self, message):
        if getattr(self, "_json_requested", False) or "--json" in sys.argv[1:]:
            print(json.dumps({"error": InputError(message).as_dict()}))
            raise SystemExit(2)
        super().error(message)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_int_sweep(value: str) -> str:
    if not re.fullmatch(r"[1-9]\d*(?:,[1-9]\d*)*", value):
        raise argparse.ArgumentTypeError("must be a positive integer or comma-separated positive integers")
    return value


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a nonnegative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def _port(value: str) -> int:
    parsed = _positive_int(value)
    if parsed > 65535:
        raise argparse.ArgumentTypeError("must be between 1 and 65535")
    return parsed


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite nonnegative number") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite nonnegative number")
    return parsed


def _positive_float(value: str) -> float:
    parsed = _nonnegative_float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def _node_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise argparse.ArgumentTypeError("must be 1-128 letters, digits, '.', '-' or '_'")
    return value


def _gpu_layers(value: str) -> str:
    lowered = value.lower()
    if lowered in {"auto", "all"}:
        return lowered
    return str(_nonnegative_int(value))


def _moe_cache(value: str) -> str:
    lowered = value.lower()
    if lowered in {"auto", "on", "off"}:
        return lowered
    return str(_positive_int(value))


def _tensor_split(value: str) -> str:
    try:
        ratios = [float(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be comma-separated nonnegative numbers with a positive total") from exc
    if not ratios or any(not math.isfinite(item) or item < 0 for item in ratios) or sum(ratios) <= 0:
        raise argparse.ArgumentTypeError("must be comma-separated nonnegative numbers with a positive total")
    return value


def _nodes_selector(value: str) -> str:
    """Validate the compact node selector used by ``--nodes``.

    ``all`` selects every node returned by the configured inventory. Otherwise
    the value is a comma-separated list; ``--node`` remains the convenient
    repeatable spelling for names containing no commas.
    """
    value = value.strip()
    if not value:
        raise argparse.ArgumentTypeError("must be 'all' or a comma-separated node name list")
    names = [item.strip() for item in value.split(",")]
    if any(not item for item in names):
        raise argparse.ArgumentTypeError("must be 'all' or a comma-separated node name list")
    if value.lower() == "all":
        return "all"
    if any(item.lower() == "all" for item in names):
        raise argparse.ArgumentTypeError("'all' cannot be combined with named nodes")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", item) for item in names):
        raise argparse.ArgumentTypeError("node names must use only letters, digits, '.', '-' or '_'")
    if len(set(names)) != len(names):
        raise argparse.ArgumentTypeError("node names must not be repeated")
    return ",".join(names)


def _add_json_flag(parser):
    parser.add_argument(
        "--json",
        dest="json",
        action="store_true",
        help="emit a single machine-readable JSON object on stdout; all human-readable output moves to stderr",
    )


def _default_model(args, *, error: str) -> str:
    model = args.model or os.environ.get("KESTREL_MODEL") or state.USER_CONFIG.default_model
    if not model and model_source._resolve_model_alias("qwen3.5:122b-10a"):
        model = "qwen3.5:122b-10a"
    if not model:
        raise InputError(error.removeprefix("Error: "))
    return model


def _add_local_run_options(parser, *, model_optional: bool = False):
    parser.add_argument(
        "model",
        nargs="?" if model_optional else None,
        help="Hugging Face model ID, model directory, or GGUF",
    )
    parser.add_argument("--no-convert", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the validated command")
    _add_json_flag(parser)
    parser.add_argument(
        "--node",
        action="append",
        type=_node_name,
        default=[],
        metavar="NAME",
        help="use a configured llama.cpp RPC node (repeatable; requires node inventory)",
    )
    parser.add_argument(
        "--nodes",
        type=_nodes_selector,
        metavar="all|NAME[,NAME...]",
        help="select all configured nodes or a comma-separated node set",
    )
    parser.add_argument(
        "--allow-insecure-rpc",
        action="store_true",
        help="DEV-ONLY acknowledgement for unauthenticated RPC to non-loopback endpoints (never network-safe)",
    )
    parser.add_argument(
        "--prompt",
        help="run a one-shot, non-interactive generation with this prompt (otherwise launches the interactive CLI)",
    )
    parser.add_argument(
        "--max-tokens",
        type=_positive_int,
        default=256,
        help="token budget for --prompt one-shot generation (default: 256)",
    )
    parser.add_argument(
        "--ctx-size",
        type=planning._context_size_arg,
        default=getattr(state.USER_CONFIG, "context_size", "auto"),
        help="context tokens (default: saved model setting, initially hardware-aware auto)",
    )
    parser.add_argument(
        "--reasoning",
        choices=REASONING_LEVELS,
        default=getattr(state.USER_CONFIG, "reasoning_level", "auto"),
        help="reasoning budget: auto, off, low, medium, high, or maximum (default: saved model setting)",
    )
    parser.add_argument("--gpu-layers", type=_gpu_layers, default="auto", help="auto, all, or an exact count")
    parser.add_argument("--cpu-moe", choices=("auto", "on", "off"), default="auto")
    parser.add_argument(
        "--target",
        choices=("auto", "balanced", "quality", "speed"),
        default="auto",
        help=(
            "placement target: auto (adaptive, from free RAM), balanced, "
            "quality (stable, slower), or speed (experimental-throughput; "
            "never auto-selected)"
        ),
    )
    parser.add_argument("--fit-target", type=_nonnegative_int, help="VRAM margin in MiB")
    parser.add_argument("--batch-size", type=_positive_int)
    parser.add_argument("--ubatch-size", type=_positive_int)
    parser.add_argument(
        "--threads",
        type=_positive_int,
        help="CPU generation and prompt threads (default: hardware-aware)",
    )
    parser.add_argument(
        "--kv-cache-type",
        choices=("f16", "bf16", "q8_0", "q4_0", "q4_1"),
        default="q8_0",
    )
    parser.add_argument(
        "--moe-cache",
        type=_moe_cache,
        default="auto",
        help="llama.cpp MoE cache: auto, on, off, or a MiB budget",
    )
    parser.add_argument(
        "--moe-hot-model",
        help="immutable Q4 GGUF sidecar for a compact Q1 expert model",
    )
    parser.add_argument(
        "--moe-cold-model",
        help="immutable Q1 experts-only sidecar for a canonical Q4 model",
    )
    parser.add_argument("--no-mmap", action="store_true")
    parser.add_argument(
        "--mlock",
        action="store_true",
        help="lock model weights in RAM so CPU-offloaded/experts stay resident "
        "(removes page-in latency jitter; implies a memory footprint)",
    )
    parser.add_argument(
        "--tensor-split",
        type=_tensor_split,
        help=(
            "comma-separated VRAM ratios for multi-GPU tensor splitting, "
            "e.g. '60,40' (default: auto-derived from detected device VRAM)"
        ),
    )
    parser.add_argument(
        "--extra",
        action="append",
        metavar="ARGS",
        help="space-separated passthrough args for the llama.cpp binary (repeatable); "
        "e.g. --extra '--temp 0.4 --top-p 0.9'",
    )
    parser.add_argument(
        "--direct-io",
        action="store_true",
        help=(
            "load weights with uncached direct I/O (bypasses the page cache) "
            "for faster cold loads from NVMe; ignore the page cache warning "
            "and prefer a warm mmap reload otherwise; disables --warm-cache"
        ),
    )
    parser.add_argument(
        "--warm-cache",
        action="store_true",
        help="prime the OS page cache for the model before launch (bounded pre-read)",
    )
    parser.add_argument("--no-mtp", action="store_true")
    parser.add_argument("--mtp-tokens", type=_positive_int, default=3)
    parser.add_argument(
        "--no-oom-retry",
        action="store_true",
        help="Do not retry startup with lower-memory settings",
    )


def _kestrel_version() -> str:
    """Resolve the package version without depending on the parent package.

    An editable install can be shadowed by a ``kestrel`` directory on
    ``sys.path`` that lacks ``__init__.py``, in which case ``kestrel`` becomes
    a namespace package without a ``__version__`` attribute. Prefer installed
    metadata, then the source ``pyproject.toml``.
    """
    try:
        from .. import __version__ as _version

        return _version
    except (ImportError, AttributeError, NameError):
        pass
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("kestrel")
    except (PackageNotFoundError, ImportError):
        pass
    try:
        root = Path(__file__).resolve().parents[2] / "pyproject.toml"
        for line in root.read_text().splitlines():
            match = re.match(r'^version\s*=\s*["\']([^"\']+)["\']', line.strip())
            if match:
                return match.group(1)
    except (OSError, UnicodeDecodeError):
        pass
    return "unknown"


def build_parser():
    parser = _ArgumentParser(description="Kestrel - hardware-aware local model orchestration and management")
    parser.add_argument("--version", action="version", version=f"kestrel {_kestrel_version()}")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("menu", help="Open the interactive Kestrel menu")
    status = sub.add_parser("status", help="Show active model, hardware plan, and benchmark state")
    _add_json_flag(status)

    run = sub.add_parser("run", help="Plan and run a local model")
    _add_local_run_options(run)
    chat = sub.add_parser("chat", help="Chat with the configured local model")
    _add_local_run_options(chat, model_optional=True)
    serve = sub.add_parser(
        "serve",
        help="Serve a GGUF model through an OpenAI-compatible llama-server endpoint",
    )
    _add_local_run_options(serve, model_optional=True)
    serve.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    serve.add_argument("--port", type=_port, default=8080, help="listen port (default: 8080)")
    serve.add_argument("--alias", help="model alias reported by /v1/models")
    serve.add_argument(
        "--embeddings",
        action="store_true",
        help="enable the /v1/embeddings endpoint for RAG workloads (llama-server --embeddings)",
    )
    serve.add_argument(
        "--wait",
        type=_nonnegative_float,
        default=0.0,
        help="after launch, poll /health until the server reports ready "
        "(seconds to wait before giving up; defaults to 30s when unset)",
    )

    setup = sub.add_parser("setup", help="Save safe local defaults (never API keys)")
    setup.add_argument("--model", help="default local GGUF, directory, or tested alias")
    setup.add_argument("--models-dir", help="managed model download directory")
    setup.add_argument("--llama-cpp-dir")
    setup.add_argument("--context", type=planning._context_size_arg, help="saved context: auto or at least 512 tokens")
    setup.add_argument("--reasoning", choices=REASONING_LEVELS, help="saved reasoning level")
    setup.add_argument(
        "--reset",
        action="store_true",
        help="replace all saved defaults; also repairs a malformed config",
    )

    settings = sub.add_parser("settings", help="View or change default model context and reasoning")
    settings.add_argument("--context", type=planning._context_size_arg, help="auto or at least 512 tokens")
    settings.add_argument("--reasoning", choices=REASONING_LEVELS, help="auto, off, low, medium, high, or maximum")
    _add_json_flag(settings)

    benchmark = sub.add_parser("benchmark", help="Measure prompt and decode rates reproducibly")
    benchmark.add_argument("model", nargs="?")
    benchmark.add_argument("--prompt-tokens", type=_positive_int, default=128)
    benchmark.add_argument("--generate-tokens", type=_positive_int, default=64)
    benchmark.add_argument("--repetitions", type=_positive_int, default=3)
    benchmark.add_argument("--ctx-size", type=_positive_int, default=2048)
    benchmark.add_argument("--gpu-layers", default="auto")
    benchmark.add_argument("--cpu-moe", choices=("auto", "on", "off"), default="auto")
    benchmark.add_argument(
        "--threads",
        type=_positive_int_sweep,
        help="one thread count or a comma-separated sweep, e.g. 8,10,12,14,16",
    )
    benchmark.add_argument("--batch-size", type=_positive_int, default=128)
    benchmark.add_argument("--ubatch-size", type=_positive_int, default=64)
    benchmark.add_argument("--kv-cache-type", choices=("f16", "bf16", "q8_0", "q4_0", "q4_1"), default="q8_0")
    benchmark.add_argument("--output", help="write the complete JSON report")
    _add_json_flag(benchmark)

    evaluate = sub.add_parser(
        "evaluate",
        aliases=("model-test",),
        help="Run deterministic Qwen quality/capacity tests against a running llama-server",
    )
    evaluate.add_argument("model", nargs="?", help="model ID reported to the OpenAI-compatible server")
    evaluate.add_argument("--endpoint", default="http://127.0.0.1:8080", help="llama-server base URL")
    evaluate.add_argument("--seed", type=int, default=42, help="deterministic generation seed (default: 42)")
    evaluate.add_argument("--timeout", type=_nonnegative_float, default=120.0, help="per-request timeout in seconds")
    evaluate.add_argument("--max-cases", type=_positive_int, help="run only the first N manifest cases")
    evaluate.add_argument("--artifact", help="local GGUF path to record in the report; never modified")
    evaluate.add_argument("--sha256", action="store_true", help="hash the artifact (slow for very large GGUFs)")
    evaluate.add_argument("--output", help="write the complete JSON report")
    _add_json_flag(evaluate)

    optimize = sub.add_parser("optimize", help="Create and optionally benchmark a hardware-specific plan")
    optimize.add_argument("model", nargs="?")
    optimize.add_argument("--context", type=int, help="override automatic context selection")
    optimize.add_argument("--quality", choices=("speed", "balanced", "quality"), default="balanced")
    optimize.add_argument(
        "--storage-path",
        help="storage path to assess (default: selected model's filesystem)",
    )
    optimize.add_argument("--benchmark", action="store_true", help="measure the selected plan with llama-bench")
    optimize.add_argument("--output")
    optimize.add_argument("--no-save", action="store_true")

    models = sub.add_parser("models", help="Discover, inspect, pull, and import models")
    models_sub = models.add_subparsers(dest="models_command")
    models_search = models_sub.add_parser("search", help="search current GGUF repositories on Hugging Face")
    models_search.add_argument("query")
    models_search.add_argument("--limit", type=int, default=10)
    _add_json_flag(models_search)
    models_files = models_sub.add_parser("files", help="list GGUF variants and sizes in a Hugging Face repository")
    models_files.add_argument("source", help="hf://OWNER/REPO or OWNER/REPO")
    _add_json_flag(models_files)
    models_list = models_sub.add_parser("list", help="list Kestrel and Ollama models")
    models_list.add_argument("--resolve", action="store_true", help="resolve Ollama model blobs")
    _add_json_flag(models_list)
    models_recommend = models_sub.add_parser("recommend", help="rank installed models by measured host memory fit")
    _add_json_flag(models_recommend)
    models_info = models_sub.add_parser("info", help="inspect a local or Ollama GGUF")
    models_info.add_argument("source", help="local path or ollama://NAME")
    models_pull = models_sub.add_parser("pull", help="download from Hugging Face or Ollama")
    models_pull.add_argument("source", help="hf://OWNER/REPO, OWNER/REPO, or ollama://NAME")
    models_pull.add_argument("--file", help="specific Hugging Face file")
    models_pull.add_argument("--include", help="Hugging Face glob, including all shards of a split GGUF")
    models_pull.add_argument("--revision", help="Hugging Face commit, tag, or branch")
    models_pull.add_argument("--destination")
    models_pull.add_argument("--dry-run", action="store_true")
    models_pull.add_argument("--set-default", action="store_true")
    models_import = models_sub.add_parser("import", help="reuse a local GGUF or Ollama blob")
    models_import.add_argument("source", help="local path or ollama://NAME")
    models_import.add_argument("--set-default", action="store_true")

    nodes_group = sub.add_parser(
        "nodes",
        help="Manage experimental trusted llama.cpp RPC workers",
    )
    nodes_sub = nodes_group.add_subparsers(dest="nodes_command")

    def add_node_common(command_parser):
        command_parser.add_argument(
            "--allow-insecure-rpc",
            action="store_true",
            help="DEV-ONLY acknowledgement for unauthenticated direct RPC outside loopback (never a network security control)",
        )
        _add_json_flag(command_parser)

    nodes_list = nodes_sub.add_parser("list", help="List configured RPC workers")
    add_node_common(nodes_list)
    nodes_add = nodes_sub.add_parser("add", help="Add or replace a named RPC worker")
    nodes_add.add_argument("name", type=_node_name)
    nodes_add.add_argument("--endpoint", required=True, help="loopback RPC endpoint, normally an SSH tunnel")
    nodes_add.add_argument("--memory-mib", type=_nonnegative_int, required=True, help="advertised accelerator memory")
    nodes_add.add_argument("--ram-mib", type=_nonnegative_int)
    nodes_add.add_argument("--engine-version")
    nodes_add.add_argument("--engine-commit", required=True, help="worker llama.cpp git commit")
    nodes_add.add_argument("--model-hash", action="append", default=[], help="cached model SHA-256 (repeatable)")
    nodes_add.add_argument("--disabled", action="store_true")
    nodes_add.add_argument("--ssh-host", help="worker SSH host; creates a managed loopback forward")
    nodes_add.add_argument("--ssh-user", help="worker SSH user (required with --ssh-host)")
    nodes_add.add_argument("--ssh-port", type=_port, default=22)
    nodes_add.add_argument("--remote-rpc-port", type=_port, default=None)
    nodes_add.add_argument(
        "--identity-file", dest="ssh_identity_file", help="absolute SSH private-key path (passed only to ssh)"
    )
    nodes_add.add_argument(
        "--host-key",
        dest="ssh_host_key",
        required=False,
        help="pinned OpenSSH public host key (for example ssh-ed25519 AAAA...)",
    )
    add_node_common(nodes_add)
    nodes_remove = nodes_sub.add_parser("remove", help="Remove a named RPC worker")
    nodes_remove.add_argument("name", type=_node_name)
    add_node_common(nodes_remove)
    nodes_doctor = nodes_sub.add_parser("doctor", help="Perform live RPC protocol and device preflight")
    nodes_doctor.add_argument("name", nargs="*", type=_node_name, help="specific nodes (default: all)")
    nodes_doctor.add_argument("--timeout", type=_positive_float, default=1.0)
    add_node_common(nodes_doctor)
    nodes_plan = nodes_sub.add_parser("plan", help="Probe nodes and show local plus remote tensor placement")
    nodes_plan.add_argument("model", nargs="?", help="optional local model for a coarse weights-only fit comparison")
    nodes_plan.add_argument("--node", action="append", type=_node_name, default=[])
    nodes_plan.add_argument("--nodes", type=_nodes_selector, metavar="all|NAME[,NAME...]")
    nodes_plan.add_argument("--timeout", type=_positive_float, default=1.0)
    add_node_common(nodes_plan)

    build = sub.add_parser("build", help="Transactionally build llama.cpp with CUDA")
    build.add_argument("--dir", help="engine checkout (default: configured llama_cpp_dir)")
    build.add_argument("--dry-run", action="store_true")
    _add_json_flag(build)

    engine_group = sub.add_parser(
        "engine",
        help="Manage the llama.cpp engine: provenance, upstream updates, self-rebuilds",
    )
    engine_sub = engine_group.add_subparsers(dest="engine_command")
    es_status = engine_sub.add_parser("status", help="Show provenance and staleness")
    es_status.add_argument("--dir", help="engine checkout (default: configured llama_cpp_dir)")
    _add_json_flag(es_status)
    es_update = engine_sub.add_parser(
        "update", help="Fetch upstream and fast-forward + rebuild if a newer revision exists"
    )
    es_update.add_argument("--dir", help="engine checkout (default: configured)")
    es_update.add_argument("--remote", help="override the tracked remote URL")
    es_update.add_argument("--dry-run", action="store_true")
    es_update.add_argument(
        "--force",
        action="store_true",
        help="hard-reset the checkout, discarding local commits/edits",
    )
    _add_json_flag(es_update)
    es_rebuild = engine_sub.add_parser("rebuild", help="Rebuild the engine from its checked-out source")
    es_rebuild.add_argument("--dir", help="engine directory")
    es_rebuild.add_argument("--dry-run", action="store_true")
    _add_json_flag(es_rebuild)
    es_set = engine_sub.add_parser("set", help="Adopt a checkout as a managed engine")
    es_set.add_argument("--dir", required=True)
    es_set.add_argument("--remote", help="remote URL (default: the checkout's origin)")
    _add_json_flag(es_set)

    self_update = sub.add_parser("self-update", help="Update the Kestrel package from a repo or a wheel")
    self_update.add_argument(
        "--repo",
        help="local Kestrel checkout (default: the package's source tree; use a checksummed HTTPS wheel for remote updates)",
    )
    self_update.add_argument(
        "--wheel",
        help="path or HTTPS URL to a Kestrel wheel (remote URLs require --sha256)",
    )
    self_update.add_argument(
        "--sha256",
        help="expected SHA256 of --wheel (required for remote wheels; verified before install)",
    )
    self_update.add_argument("--dry-run", action="store_true")
    _add_json_flag(self_update)
    convert = sub.add_parser("convert", help="Convert supported NVFP4 safetensors")
    convert.add_argument("model")
    convert.add_argument("--output", "-o")
    convert.add_argument(
        "--generic",
        action="store_true",
        help="use llama.cpp's convert_hf_to_gguf.py for arbitrary HF safetensors models",
    )
    convert.add_argument(
        "--outtype",
        default="bf16",
        help="GGUF output type for --generic (default: bf16)",
    )
    convert.add_argument(
        "--include-mtp",
        action="store_true",
        help="include the optional speculative MTP draft block",
    )
    convert.add_argument(
        "--dense-q4",
        action="store_true",
        help="quantize dense matrices to Q4_0 instead of BF16",
    )
    convert.add_argument(
        "--cold-tier",
        choices=("off", "q1_0", "q1_only"),
        default="off",
        help="emit Q1_0 expert twins, or a compact Q1-only expert model",
    )
    convert.add_argument(
        "--q4-sidecar-source",
        help="derive q1_only experts from an existing canonical Q4 GGUF",
    )
    convert.add_argument(
        "--experts-only",
        action="store_true",
        help="emit only compact routed experts for use as a cold sidecar",
    )
    convert.add_argument(
        "--q2-edge-layers",
        type=int,
        default=0,
        metavar="N",
        help="use Q2_K experts for the first and last N layers of a direct q1_only conversion",
    )
    convert.add_argument(
        "--all-q2",
        action="store_true",
        help="use Q2_K for every routed expert in a direct compact conversion",
    )
    convert.add_argument(
        "--compact-expert-type",
        choices=("q1_0", "q2_k", "q3_k", "iq1_s"),
        default="q1_0",
        help="expert format for direct q1_only conversion; q2_k is the all-Q2 mode",
    )
    convert.add_argument(
        "--imatrix",
        help="llama-imatrix GGUF used to calibrate IQ1_S, Q2_K, or Q3_K experts",
    )
    convert.add_argument(
        "--conversion-workers",
        type=int,
        help="parallel expert conversion workers (default: up to 4)",
    )
    convert.add_argument(
        "--experts-keep",
        type=int,
        metavar="K",
        help=(
            "EXPERIMENTAL: emit only the K most-used experts per layer and "
            "rewrite expert_count (must stay >= experts per token and < the "
            "full count). Smaller model, fits more in VRAM, quality may drop."
        ),
    )
    convert.add_argument(
        "--expert-importance",
        help=(
            "JSON list of per-expert importance values (length = num_experts) "
            "used to select which experts --experts-keep keeps; defaults to "
            "keeping the first K"
        ),
    )
    audit = sub.add_parser(
        "audit",
        help="Validate GGUF tokenizer and target/MTP structure",
    )
    audit.add_argument("model", help="GGUF artifact to audit")
    audit.add_argument(
        "--source",
        help="source Hugging Face model directory for cross-checks",
    )
    _add_json_flag(audit)
    audit.add_argument(
        "--cold-sidecar",
        action="store_true",
        help="validate an intentionally experts-only Q1 cold-sidecar GGUF",
    )
    doctor = sub.add_parser("doctor", help="Check hardware and llama.cpp capabilities")
    _add_json_flag(doctor)

    return parser
