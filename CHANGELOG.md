# Changelog

All notable changes are documented here. Kestrel follows semantic versioning.

## Unreleased

- Add a stdlib-only terminal UI layer (`kestrel.ui`) with ANSI colors, framed
  boxes, width-aware wrapping, aligned tables, and validated prompts; the
  interactive menu, chat, and report commands all render through it.
- Redesign the interactive menu with a status header (default model, GPU, RAM),
  colored options, input validation, error feedback, and a pause before
  returning from a launched command.
- Upgrade the Kimi K3 chat loop with a session header, colored prompt, and
  `/help` and `/model` commands; keep preserved-thinking history.
- Render the runtime plan, launch command, benchmark summary, doctor, status,
  model-market, recommendation, audit, and setup output as framed, colored
  sections that degrade cleanly on non-TTY terminals and under `NO_COLOR`.
- Replace the numbered menu with an arrow-key interactive menu (Up/Down/Home/End,
  Enter, `q` to quit) built on termios raw input; it degrades to a plain list
  on non-TTY terminals.
- Make the interactive menu compact and elegant: a slim `kestrel · version` header
  with a one-line hardware status, and options on a vertical `│` rail with a
  `▸` cursor. The active model is shown inline on "Chat with Model". Long lists
  scroll with `↑ N more` / `↓ N more` indicators and rows truncate to the column
  width.
- Redraw the menu on a freshly cleared screen each time it is entered, so
  output from a launched command is wiped before the menu returns and there are
  no leftover frames or duplicated blocks.
- Group top-level actions into submenus so the main screen stays small: Import
  Models, Manage Models (installed/search/diagnostics/benchmark), and
  Configure Kestrel each open a short picker.
- Pick models from a list instead of typing: Chat, Select, and Configure show a
  picker of the configured default, local GGUFs, and Ollama models; importing an
  Ollama model lists everything `ollama list` reports, with a manual entry as a
  fallback.
- Fix `python -m kestrel.cli` crashing with `ImportError: cannot import name
  '__version__'` when run from a parent directory where the package directory
  is picked up as a namespace package; version resolution now falls back to
  importlib metadata and `pyproject.toml`.
- Add a simpler `kestrel chat [model]` local entry point.
- Add an interactive menu, `status`, hardware-aware auto-context, and measured
  optimization profiles with explicit benchmark state.
- Add Ollama provider/import support and a live Hugging Face GGUF market with
  search, variant inspection, dry-run downloads, provenance, and split-shard
  validation.
- Replace full tensor mapping during model inspection with a bounded GGUF
  metadata reader; a 72.2 GiB model now inspects in 0.05 seconds locally.
- Add parallel IQ1_S conversion workers and clearer per-stage progress.
- Force the experimental synchronous MoE cache off for CPU-MoE production
  plans after full-model testing showed that automatic cache selection can
  disable repacking and reduce throughput.
- Add sparse-MoE mmap handling to the native runtime fork so 8-of-256 routing
  does not force-populate a 24.33 GiB GGUF on a 16 GiB machine.
- Strengthen the full-model suite with exact artifact hashing, a 64-token
  decode floor, 128-token stability checks, thermal metrics, and an automated
  release-gate summary.
- Add dependency-free Moonshot Kimi K3 API chat with preserved-thinking history.
- Add honest local Kimi K3 hardware and engine capability reporting.
- Add direct-source IQ1_S compact expert conversion and native correctness tests.
- Define measurable local-quality, performance, packaging and Kimi release gates.
- Complete the calibrated 122B qualification: 8/8 deterministic cases and the
  stability gate pass, while the measured 6.33 tok/s decode rate correctly
  keeps the 10-15 tok/s release gate closed.

## 1.1.0

- Add Qwen3.5-122B-A10B NVFP4 conversion, GGUF auditing and memory-aware
  llama.cpp launch planning.
- Add conservative CPU-MoE placement, quantized KV defaults and CUDA OOM retry.
