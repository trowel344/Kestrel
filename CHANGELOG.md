# Changelog

All notable changes to Kestrel are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- The Settings tab of the `kestrel menu` now also manages KV cache type,
  Turbo KV, GPU layers, and CPU MoE. These live in `config.toml`
  (`kv_cache_type` auto/f16/bf16/q8_0/q4_0/q4_1, `kv_cache_turbo` bool,
  `gpu_layers` auto/all/count, `cpu_moe` auto/on/off) and are applied by
  `chat`, `run`, `serve`, and the managed agent server: `auto` keeps the
  tuned placement profile (measured ~23 tok/s on the 8 GiB target), while an
  explicit value overrides it. `kestrel settings` gained `--kv-cache-type`,
  `--turbo-kv`/`--no-turbo-kv`, `--gpu-layers`, and `--cpu-moe`.
- `kestrel agents start` and `kestrel agents launch` now accept `--gpu-layers`
  and `--cpu-moe`, forwarding explicit layer placement to the managed
  llama-server (mirroring `kestrel serve`). This gives an escape hatch on GPUs
  where the tuned aggressive split can exhaust VRAM during real decode: e.g.
  an 8 GiB card with a 30B-A3B model OOMs during agentic generation at the
  tuned placement but runs stably with `--cpu-moe on` (experts on CPU).
  The override is threaded through the managed server, its reuse check, and the
  loopback proxy child command, and is surfaced in `agents status`/`start`
  output.
- Fixed `kestrel benchmark` failing on engine builds whose `llama-bench` does
  not expose `--threads-batch`: the flag is now capability-gated, so older
  engines benchmark with the legacy flag set instead of erroring on an unknown
  option.
- A measured tuning profile now keeps its placement when the requested context
  is larger than the tuned context instead of being discarded. The prefill
  micro-batch is bounded to the conservative small-VRAM tier and the runtime
  plan warns that the KV cache grows beyond the measured footprint, while the
  measured decode/prompt rates still replace the analytic estimate. This fixes
  an unprofiled-style fallback (e.g. `n_cpu_moe_layers=None`, batch 256/64, and
  a wildly pessimistic Tok/s prediction) appearing for users whose default
  context exceeds the tuned size.
- Parameter estimates for GGUF files now anchor the total against the file's
  own size via the declared `general.file_type` quant. Hybrid-MoE conversions
  (e.g. a 30B-A3B class model) whose advertised `feed_forward_length` does not
  match the tensor payloads previously inflated the estimate to a multi-TB
  phantom and produced absurdly slow analytic predictions; the active/total
  set is rescaled to the file instead. The planner metadata reader now recovers
  `general.file_type` even when a producer emits it after the tokenizer
  section (bounded resume scan), and the on-disk metadata cache was bumped so
  stale entries are rebuilt.
- The auto-tune scan measures each candidate with a longer decode window and
  two repetitions (128 prompt / 64 generated) for a more stable selection, and
  probes one-eighth, one-quarter, and one-third GPU-expert slices toward the
  VRAM ceiling.
- The preflight VRAM estimate no longer skips aggressive GPU-expert candidates
  that a coarse formula would reject: candidates within 1.5x of free VRAM are
  measured and llama-bench records any OOM as evidence. Each measured candidate
  now records its *sampled peak* VRAM (nvidia-smi polled while llama-bench runs)
  so the persisted `minimum_free_vram_mib` gate reflects the real footprint
  instead of the conservative estimate. This lets the search find the fastest
  placement that actually fits (e.g. the 30B-A3B class model on an 8 GiB card
  jumped from 15.1 to ~23.2 tok/s decode by moving 18 of 53 layers' experts to
  the GPU, right at the 7.8 GiB VRAM ceiling).
- Fixed `kestrel optimize --benchmark` unpacking `_select_context_size`, which
  now returns an extra `overcommitted` flag; the flag is folded into the plan.
- KV-cache sizing for GGUF models now counts only the layers that actually
  allocate K/V. Hybrid-attention architectures (interleaved full-attention,
  SSM/linear-attention, and FFN-only layers, e.g. `nemotron_h_moe`) were sized
  as if every layer stored KV, which over-estimated the cache by the full layer
  count (for a 30B-A3B class model the 32K-context KV estimate dropped from
  ~9.8 GiB to ~0.12 GiB). The metadata reader scans the tensor-info section for
  `blk.*.attn_k.weight` tensors to count KV-bearing layers and their value dim,
  cached on disk (schema bumped), so context selection and the 
  `context_scaled` footprint are no longer wildly conservative for hybrid
  models.
- `kestrel serve` gained `--ctx-checkpoints N` (default 0) which passes
  llama-server `--ctx-checkpoints`, keeping per-slot KV state restorable across
  context truncation. Combined with `--cache-reuse` this lets long coding
  sessions reuse a cached prefix after the ring buffer wraps instead of forcing
  a full re-prefill; the flag is capability-gated per engine build.
- Added an opt-in TurboQuant KV cache switch: `local.kv_cache_turbo = true` in
  the config and `--turbo-kv` on `run`/`chat`/`serve`. It passes llama.cpp's
  `--turbo-kv` (random orthogonal rotation before low-bit K/V quantization)
  for higher KV compression, and is silently skipped until the selected engine
  exposes the flag. TurboQuant is still an open upstream PR rather than merged
  llama.cpp, so the switch is inert on stock builds and activates automatically
  once a turbo-capable engine (or a forked build) is installed.
- `kestrel run`/`serve` now auto-tune on first launch of a model that has no
  measured tuning profile: a bounded llama-bench scan measures candidate
  placements, persists the fastest safe profile, and this very launch uses it
  with the measured decode/prompt rates replacing the analytic Tok/s estimate.
  Pass `--no-auto-tune` to keep the conservative planner defaults. Existing
  measured profiles also now feed the plan: `predicted_decode_tps` becomes the
  profile's measured rate (`prediction_confidence: "measured"`).
- `kestrel serve` now enables llama-server KV chunk reuse (`--cache-reuse 256`)
  by default so multi-turn chat and coding-agent requests reuse a matching
  cached prefix instead of re-processing the full system prompt and tool schema
  every turn. The chunk size is configurable with `--cache-reuse N` and
  `--cache-reuse 0` disables it; the flag is skipped when the engine build does
  not expose it.
- CPU-MoE plans now use a dedicated prompt-processing thread budget
  (`--threads-batch`, up to all logical cores) separate from the tuned decode
  thread count, speeding up prefill over the CPU-resident layers. An explicit
  `--threads-batch N` override is available on `run`/`chat`/`serve`, and the
  runtime plan reports the prompt-thread count.
- Added opt-in Sun Map persistence to `kestrel agents start/launch --sunmap`.
  The loopback proxy compiles bounded memory into Responses, Chat Completions,
  and Anthropic Messages requests, observes completed assistant output without
  buffering live streams, deduplicates replayed messages, and excludes raw
  tool output from privileged persistent memory. Compiled context remains
  stable during a tool loop and refreshes after its terminal response.
- Sun Map mode now adds a durable task checkpoint, deterministic tool-event
  reducer, repetition and protected-file supervision, provenance/staleness
  tracking, and a separate hash-versioned current-code index. Client-reported
  passes stop at `ready_for_verification`; only an independent gate can mark a
  task complete.
- Long-horizon injection is now query- and token-aware: compact checkpoint
  projections omit repeated objectives and irrelevant fields, code evidence
  uses focused non-overlapping windows, stale replayed user turns no longer
  broaden retrieval, and status exposes estimated tokens by evidence channel.
- Added mode-0600 redacted trajectory recording and direct Sun Map replay.
  Runtime allocation returns unused code capacity to semantic memory,
  suppresses noisy history for explicit checkpoint questions, and can use the
  llama.cpp tokenizer route with a bounded fallback. Ollama reasoning settings
  now forward through `reasoning_effort`, including `none` for off.
- Exact user-mandated verification commands are now durable task-contract
  state. Divergent gates create warnings, and repeated no-progress failures
  produce a strong bounded recovery directive for small local models.
- Managed agent servers now accept `ollama://MODEL` for authenticated,
  memory-aware Chat Completions proxying. Kestrel never forwards its private
  credential to Ollama and does not claim unsupported Responses or Anthropic
  routes. Startup preloads the selected model before reporting readiness and
  rejects Ollama aliases whose configured `num_ctx` is smaller than Kestrel's
  advertised agent context.
- Managed llama.cpp coding-agent servers use one inference slot so the full
  configured context is not divided across idle slots. Cancelled downstream
  streams are also handled as normal disconnects instead of emitting noisy
  secondary proxy errors.

- Added persistent model settings for context size and reasoning level. They
  are visible from the top-level **Settings** screen, `kestrel settings`, and
  per-launch `--ctx-size` / `--reasoning` overrides; explicit reasoning levels
  now become real llama.cpp reasoning budgets rather than UI-only labels.

### Added

- **Coding-agent work mode**: a top-level interactive UI and `kestrel agents`
  commands configure, launch, inspect, and stop Pi, Oh My Pi, Codex, Claude
  Code, and OpenCode against one reusable local model server. Client profiles
  are reversible and Kestrel-owned; live route probes distinguish Chat
  Completions, Responses, and Anthropic Messages compatibility.
- The managed work server is loopback-only and API-key authenticated, with
  private credential/state files, stale-PID ownership checks, bounded startup,
  logs, and deterministic stop/reap behavior.

- **Experimental trusted RPC nodes**: strict atomic named-node inventory,
  loopback/SSH-tunnel secure default, live ggml-rpc HELLO/device/memory
  preflight, pinned engine-commit matching, and deterministic local-plus-remote
  tensor split ordering for `run`, `chat`, and `serve`.
- **`kestrel nodes`** commands for add/list/remove, protocol-aware doctor, and
  a coarse weights-only capacity plan. JSON reports retain probe evidence,
  per-device order, and live capacities instead of equating TCP reachability
  with compatibility.
- Managed node registration now supports pinned public host keys and
  Kestrel-supervised SSH local forwards. SSH children use strict batch/key
  checking, loopback-only ephemeral endpoints, process-group cleanup, and
  never pass credentials to llama.cpp.

### Changed

- The GitHub landing page now leads with Kestrel's measured out-of-core result,
  a two-command start, and a compact explanation of the memory planner. A new
  layered off-white block-pixel K with a complete tonal shadow on a clean,
  softly lit deep-navy tile gives the project a
  terminal-native visual identity, and
  the detailed trusted-node guide has moved into focused documentation.
- The interactive front door is reduced to Start chat, Models, Tools, and Exit.
  Chat launches with automatic settings; model onboarding groups local GGUF,
  Ollama, and Hugging Face sources under one plain-language path. Selection
  styling and key hints are quieter and hardware memory is shown in readable
  GiB units.
- llama.cpp capability discovery now retains a bounded, binary-identity keyed
  cache for CLI, server, and alternate builds. Repeated doctor/startup checks
  avoid redundant engine processes while atomic binary replacements invalidate
  stale capabilities.
- Model discovery resolves invariant Hugging Face snapshot roots once and
  avoids a redundant filesystem stat per GGUF candidate.
- Generic upstream Hugging Face conversion now stages, fsyncs, and atomically
  replaces its target, preserving an existing GGUF across conversion failure,
  timeout, or interruption.
- Config and Ollama provider responses now have strict typed schemas; provider
  bodies and error details are bounded before decoding.

### Security

- `kestrel serve` now requires a private API-key file for non-loopback binds.
  Coding-agent profiles use command/environment-backed credentials rather than
  embedding the managed server token.

- Direct non-loopback llama.cpp RPC is rejected unless the user supplies the
  explicit `--allow-insecure-rpc` acknowledgement. Documentation states that
  upstream RPC has no authentication/encryption and recommends loopback-bound
  workers carried through authenticated SSH tunnels.
- Managed inventory rejects symlink/non-regular redirection and insecure parent
  directories. RPC probing uses one absolute deadline so a byte-dripping worker
  cannot multiply the configured timeout across protocol fields and devices.

## [1.6.0] - 2026-08-11

The reliability and maintainability release: a modular CLI, deterministic
model evaluation, bounded recovery paths, stricter conversion validation, and
release artifacts that are tested before publication.

### Added

- **`evaluate` / `model-test`**: deterministic capability tests for a running
  OpenAI-compatible llama-server, with exact artifact provenance, optional
  SHA-256 recording, machine-readable reports, and failure exit codes.
- **Quantization candidate analysis** for comparing calibrated Q2_K/Q3_K
  error on representative expert rows before committing to a full conversion.
- **Behavioral CLI tests** through the real parser/dispatch: `doctor --json`,
  `serve --json --dry-run`, and `run --json --dry-run` assert the stdout JSON
  contract end-to-end.
- CI test matrix now covers `ubuntu-latest`/`macos-latest` across Python
  3.11/3.12/3.13, plus a no-torch `import kestrel` smoke job and a dedicated
  coverage-gate job (`pytest --cov=kestrel --cov-report=term`; rising floor,
  currently 55%).
- `release` workflow: pushing a version-matched `v*` tag runs the complete
  test/coverage/lint/format gate, builds and smoke-tests an sdist + wheel,
  then creates an idempotent GitHub Release containing the exact artifacts.
- PyPI publication is intentionally disabled because the `kestrel`
  distribution name belongs to an unrelated project. Source and GitHub Release
  installation remain available while a distinct distribution name is chosen.
- Dependabot configuration for pip and GitHub Actions (weekly).
- `[tool.coverage]` configuration (branch coverage, rising fail floor).

### Changed

- **`cli.py` split into a `kestrel/cli/` package** (main, parser, probes,
  model_source, planning, runtime, run, health, models, bench, menu,
  engine, updater, convert) — ~3,700-line single file becomes organized
  command families; the public entry point and `cli.*` symbol surface are
  preserved.
- **`gguf/converter.py` split**: low-level quant/dequant primitives moved to
  a new `gguf/quants.py`; `torch`/`safetensors` imports are now lazy/guarded so
  `import kestrel` never requires the `[convert]` extra.
- **Dedup**: shared `util.truncate`/`util.ttl_cache` replace five inline TTL
  caches and every `[-2000:]` stderr tail; `util.write_atomic` replaces three
  hand-rolled temp+rename writers; `model_store.hf_snapshot_dir` is the single
  HF-snapshot resolver; the "hf CLI present" probe is one helper.
- **Dead code removed**: five orphaned error classes, `FP8_LUT`, unused
  `_pack_gguf_kv_*` wrappers, `_write_data_bf16`, unused `ui.blue`/`ui.magenta`,
  the dead `--yes` flag, and no-op `except EngineError` boundaries.
- **Error taxonomy wired**: `ConfigError` raised by `config.load_config`,
  `BackendError` (a `RuntimeError` subclass) raised by the llama.cpp backend.
- **Bug fixes**: `[EXCELLENT:9]` label typo, `--json` exit codes now propagate
  to the process exit status, and interactive sessions reload the on-disk
  config instead of using the stale import-time snapshot.
- **Formatter enforced**: `ruff format` adopted (all files reformatted once)
  and `ruff format --check` added to lint CI; ruff pinned to `ruff==0.9.*`.
- CI actions are SHA-pinned (`actions/checkout@<sha>`,
  `actions/setup-python@<sha>`) with the resolved version in comments; the
  duplicate `ruff check` step was removed from the test workflow (lint owns it).
- Packaging: Python <=3.12 uses `numpy>=1.26,<2` with the torch 2.1 converter
  floor, while Python 3.13+ uses NumPy 2.1+ with torch 2.6+; this keeps the
  tensor/ndarray bridge valid without dropping Python 3.13. `pytest-cov` is
  added to the `dev` extra.
- `install.sh` reuses an existing `.venv` when present and fully anchors the
  `kestrel --version` check regex.
- Engine builds now use the same bounded, transactional rebuild path as
  `engine rebuild`; remote status fetches the upstream object before computing
  ahead/behind counts, and spawn/timeout failures restore executable last-good
  artifacts.
- Runtime numeric flags are validated before launch, server children are
  reaped across readiness failures and interrupts, abandoned streaming
  generations terminate promptly, and process spawn failures remain typed.
- Model-store resolution now contains Hugging Face snapshots beneath their
  cache root, validates Ollama references/digests against the native manifest
  layout, and rejects exact-file traversal. Executables, shared libraries, and
  model aliases are no longer auto-discovered from world-writable `/tmp`.
- State/report copies preserve permissions and replace atomically; the Q1
  cascade report now uses the same crash-safe writer.

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

The reliability spine that lets Kestrel stand on its own: supported command
failures use typed, machine-readable errors; durable state uses crash-safe
writes where applicable; and updates are verified with an explicitly bounded
recovery path.

- **Structured error taxonomy** (`kestrel/errors.py`): supported typed failures
  surface a stable `code`/`message`/`hint` through one handler — human prose on
  stderr, or a single `{"error": {...}}` JSON document on stdout under
  `--json`, with per-class exit codes. `EngineError` and `ModelStoreError`
  migrated onto it.
- **Crash-safe writes** (`kestrel/util.py`): persisted config, engine
  manifests, benchmark profiles, and caches use same-directory temp + `fsync`
  + rename via `write_atomic`, with last-good backups where useful. This
  prevents torn target documents when the atomic writer is used.
- **`engine update`/`rebuild` last-good restore**: binaries are snapshotted
  before a build; a failed or interrupted build rolls back the previous
  working engine and reports `restored_from_previous`. A smoke test
  (`llama-cli --version`) must pass before a build is accepted.
- **Guarded `self-update`**: remote wheels require HTTPS and an explicit
  SHA-256 digest, and local wheels are checked for Kestrel dist metadata before
  installation. The package and its dist metadata are snapshotted for a
  best-effort rollback after an import/CLI smoke check; pip dependency changes
  are outside that rollback boundary and may still require a manual reinstall.
  `--repo` accepts only a local, versioned Kestrel checkout; remote updates use
  a checksummed HTTPS wheel.
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
