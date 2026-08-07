# Kestrel

Hardware-aware local model orchestration for large AI work — run, convert, and
manage multi-hundred-GB models on the hardware you actually have.

Kestrel profiles the machine, inspects the model, and enables only the
capabilities the installed llama.cpp binary supports. It plans a memory layout
that fits your VRAM and RAM, loads weights as fast as your disk allows, and
manages GGUFs (including split shards) without a cloud account.

[![CI](https://img.shields.io/github/actions/workflow/status/trowel344/Kestrel/test.yml?label=tests)](https://github.com/trowel344/Kestrel/actions)
[![Lint](https://img.shields.io/github/actions/workflow/status/trowel344/Kestrel/lint.yml?label=lint)](https://github.com/trowel344/Kestrel/actions)
[![PyPI version](https://img.shields.io/pypi/v/kestrel)](https://pypi.org/project/kestrel/)
[![Python](https://img.shields.io/pypi/pyversions/kestrel)](https://pypi.org/project/kestrel/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Why Kestrel

- **Plans before it runs.** VRAM, free RAM, and the llama.cpp binary's real
  capability set are measured first, so Kestrel picks a memory layout that
  fits — CPU-MoE placement, quantized KV caches, VRAM margins, and an automatic
  CUDA-OOM retry ladder.
- **Loads at disk speed.** `--direct-io` bypasses the page cache for cold NVMe
  loads (opt-in; warm mmap reloads stay the default).
- **Uses every GPU.** Multi-GPU rigs are detected and `--tensor-split` balances
  tensors across all of them; planning fits against combined VRAM.
- **Handles 100 GB+ models.** Split-GGUF shards are discovered as one model and
  sized across all shards; NVFP4 conversion compacts MoE experts to Q1/IQ1_S
  with optional MTP drafts.
- **Dependency-light.** `import kestrel` pulls only `numpy` and `gguf`;
  conversion dependencies stay behind `kestrel[convert]`.

## Quick start

```bash
# one-command install (isolated virtual environment)
./install.sh
.venv/bin/kestrel doctor              # GPU, RAM, and llama.cpp capabilities
.venv/bin/kestrel setup --model /path/to/model.gguf
.venv/bin/kestrel chat                # chat with the configured default model
```

Or install with pip: `pip install '.[convert]'`. Skip setup by passing a model
directly: `kestrel chat /path/to/model.gguf`. `kestrel` (or `kestrel menu`)
opens the interactive menu.

> No llama.cpp binary yet? `kestrel build` builds the CUDA-enabled MoE-native
> build, or set `KESTREL_LLAMA_CPP_DIR` to an existing llama.cpp checkout.

## Command reference

```text
kestrel menu        interactive menu
kestrel doctor      check hardware and llama.cpp capabilities
kestrel status      active model, hardware plan, benchmark state
kestrel setup       save safe local defaults (never API keys)
kestrel run         plan and run a local model
kestrel chat        chat with the configured local model
kestrel serve       serve a GGUF over an OpenAI-compatible HTTP endpoint
kestrel benchmark   reproduce prompt/decode measurements
kestrel optimize    create and optionally benchmark a hardware profile
kestrel models      search, list, recommend, info, pull, import
kestrel convert     convert safetensors to GGUF (NVFP4 or --generic)
kestrel audit       validate a GGUF against its source
kestrel build       build llama.cpp with CUDA
```

## Run-time controls

| Flag | What it does |
| --- | --- |
| `--direct-io` | Uncached direct I/O for fast cold loads from NVMe; disables `--warm-cache`. |
| `--mlock` | Pin CPU-offloaded/MoE weights in RAM; removes page-in latency jitter. |
| `--tensor-split 60,40` | Multi-GPU split ratios; auto-derived from detected VRAM. |
| `--extra "--temp 0.4 --top-p 0.9"` | Pass arbitrary llama.cpp flags through (repeatable). |
| `--kv-cache-type q8_0` | Quantized KV cache to stretch context under VRAM pressure. |
| `--cpu-moe on` | Route MoE experts through CPU to keep the GPU dense-only. |
| `--warm-cache` | Pre-read the model into the OS page cache before launch. |

`serve --embeddings` enables the `/v1/embeddings` endpoint for RAG workloads.
`convert --generic` wraps llama.cpp's `convert_hf_to_gguf.py` so any Hugging
Face safetensors model can become GGUF, not just the NVFP4 Qwen3.5 layout.

## Agent-friendly output

Kestrel is a runner *for* agents, so every run/serve command speaks machine
readable too:

| Invocation | What you get on stdout |
| --- | --- |
| `kestrel run --json --dry-run MODEL` | The validated launch plan as one JSON object (`command`, `model`, `dry_run`). |
| `kestrel run --json --prompt "..." MODEL` | One-shot generation: `output`, token counts, tok/s, `duration_s`, `command`. |
| `kestrel serve --json --wait 120 MODEL` | `status: ready|timeout`, `url`, `port`, and `ready_after_s` once `/health` answers. |
| `kestrel serve --json --dry-run MODEL` | Connection info + the full `llama-server` command. |

With `--json`, human-readable output is redirected to stderr so stdout stays a
single parseable JSON document (the contract agents rely on). `--prompt`
accepts an inline one-shot prompt; `--max-tokens` caps its budget.

## Features

- **Hardware-aware planning** — measures VRAM, RAM, and llama.cpp support, then
  picks a conservative layout (CPU-MoE placement, mmap/direct-IO, Q8 KV
  caches, VRAM margins, startup CUDA OOM retry).
- **Multi-GPU and fast loading** — aggregate-VRAM planning, `--tensor-split`,
  `--direct-io`, `--mlock`, and persistent metadata caching so repeated runs
  skip re-reading 100 GB headers.
- **Local + remote models** — discover, inspect, rank, pull, and import models
  from disk, Hugging Face (GGUF-filtered), and Ollama; split-GGUF shard sets
  are treated as one complete model.
- **NVFP4 conversion** — Qwen3.5-MoE safetensors to GGUF with dense Q4, compact
  Q1/IQ1_S expert tiers, expert pruning, and optional MTP drafts.
- **GGUF auditing** — validates an artifact against its source model.
- **Reproducible benchmark** — separates measured results from projections.

## Install

```bash
./install.sh                  # isolated .venv install (recommended)
pip install .                 # runtime only (numpy + gguf)
pip install '.[convert]'      # + NVFP4 conversion (torch, safetensors)
pip install '.[dev]'          # + pytest, ruff (for contributors)
```

See `./install.sh --help` for options (`--convert`, `--dir`).

## Development

```bash
pip install -e '.[dev,convert]'
python -m pytest              # hermetic suite; caches are per-test isolated
ruff check kestrel/ tests/
```

Tests never touch a real user config, model directory, or cache — `tests/`
runs fully offline. CI runs the same suite on Python 3.11 and 3.12.

## Repository layout

```text
kestrel/                 installable package (the product)
  cli.py                 command-line entry point, menus, planning
  config.py              config/env loading
  ui.py                  std library terminal UI
  model_store.py         model discovery and blob resolution
  core/planner.py        memory placement planning
  core/pipeline.py       llama.cpp wrapper
  backends/llama_cpp.py  process launch, capability probe, metrics
  gguf/converter.py      NVFP4 -> GGUF conversion
  gguf/metadata.py       GGUF metadata parsing
  gguf/audit.py          audit
  providers/             Ollama adapter
```

The public API is `kestrel.__version__`, `kestrel.InferencePipeline`,
`kestrel.LlamaCppBackend`, and `kestrel.NVFP4Converter`.

## Getting help

- `kestrel --help`, `kestrel <command> --help`
- Report issues: see [CONTRIBUTING.md](CONTRIBUTING.md#reporting-an-issue)
- Contact: trowel344@gmail.com

## License

MIT. See `LICENSE`.
