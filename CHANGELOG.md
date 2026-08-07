# Changelog

All notable changes to Kestrel are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.5.0] - 2026-08-05

The spotlight release: faster cold loads, multi-GPU awareness, split-model
support, and a fully hermetic, CI-covered test suite.

### Added

- **`--direct-io`** for `run`/`chat`/`serve`: uncached direct I/O
  (`--no-mmap --direct-io`) loads cold NVMe models at sequential disk speed.
  Capability-gated against the llama.cpp binary, opt-in, and correctly
  disables `--warm-cache`.
- **Multi-GPU support**: `detect_gpu` aggregates VRAM across every nvidia-smi
  device, planning fits against combined capacity, and `--tensor-split` emits
  per-device ratios (auto-derived from free VRAM when more than one GPU is
  present).
- **`--mlock`**: pin CPU-offloaded and MoE weights in RAM to remove page-in
  latency jitter.
- **`--extra "..."`**: repeatable escape hatch that passes arbitrary llama.cpp
  flags through `run` and `serve`.
- **`serve --embeddings`**: enables the OpenAI-compatible `/v1/embeddings`
  endpoint for RAG workloads.
- **`convert --generic`**: converts any Hugging Face safetensors model to GGUF
  by wrapping llama.cpp's `convert_hf_to_gguf.py` (beyond the NVFP4 Qwen3.5
  path).
- **Split-GGUF support**: shard families (`-00001-of-00005.gguf`) are grouped
  into complete models in `models list`/`recommend`, sized across all shards,
  and model discovery falls back to any `.gguf` in a model directory.
- **Persistent metadata cache**: GGUF planner metadata is cached on disk keyed
  by `(path, size, mtime)` so repeated `run`/`recommend`/`models info` calls
  skip re-reading large file headers.

### Changed

- Conversion hot path: safetensors shard headers are parsed once per shard
  (cached handles) and Q1_0/Q4_0 quantization consumes buffers in place,
  cutting ~460 GB of copy traffic over a full 122B-MoE conversion. Output is
  byte-identical.
- `tests/` now lints clean under `ruff` and runs hermetically (per-test cache
  isolation, `KESTREL_CACHE_DIR`), with `pytest` configured in `pyproject.toml`.
- CI now runs the full test suite on Python 3.11/3.12 and lints both `kestrel/`
  and `tests/`.

### Fixed

- `choose_default_gguf` and model discovery reject incomplete split sets
  instead of offering broken models.

### Foundation

The reliability spine that lets Kestrel stand on its own: every failure is a
typed, machine-readable error; every state file is written crash-safe; and
updates can never brick the tool.

- **Structured error taxonomy** (`kestrel/errors.py`): all failures surface a
  stable `code`/`message`/`hint` through one handler — human prose on stderr,
  or a single `{"error": {...}}` JSON document on stdout under `--json`, with
  per-class exit codes. `EngineError` and `ModelStoreError` migrated onto it.
- **Crash-safe writes** (`kestrel/util.py`): `write_atomic` persists config,
  engine manifests, and caches via same-directory temp + `fsync` + rename,
  keeping a `.bak` of the last-good file. No torn documents on crash.
- **`engine update`/`rebuild` last-good restore**: binaries are snapshotted
  before a build; a failed or interrupted build rolls back the previous
  working engine and reports `restored_from_previous`. A smoke test
  (`llama-cli --version`) must pass before a build is accepted.
- **Trusted `self-update`**: `--wheel` + `--sha256` verify the artifact before
  installing; the running package is snapshotted first and restored on a
  failed post-install check, so Kestrel is never left broken.
- **Deepened `doctor`**: disk-free, writability, llama.cpp compatibility, and
  GGUF-magic model-integrity checks, each `ok`/`warn`/`fail`, in both human
  and `--json` forms with a non-zero aggregate exit code.
- **`serve` healthcheck**: readiness polled via `/health` then `/v1/health`;
  a server that never comes up is terminated with a `service_error` and a
  targeted hint instead of an endless hang.
- **Monolith splits**: `cli.py`'s `cmd_run`/`cmd_models`/`cmd_optimize`, the
  planner's `plan_runtime`, and `audit.py`'s `audit_snapshot` are now thin
  orchestrators over single-purpose helpers. `--json`, model defaults, and
  menu dispatch are shared, not copy-pasted.
- **Edge-case hardening**: model discovery survives partial/symlinked/read-only
  and corrupt split sets with typed errors; the GGUF parser never leaks raw
  `struct.error`/`EOF` (fuzzed over 4000 mutations); quantizers are proven
  byte-exact by deterministic property tests.
- **CI matrix & install verification**: GitHub Actions runs the suite on
  Linux/macOS across Python 3.11/3.13, and `install.sh` fails loudly (exit 1)
  if the fresh install cannot print a version.

## [1.4.0] - Previous release

Hardware-aware placement planning, NVFP4 conversion, GGUF auditing, model
store, Ollama provider, and the interactive menu.
