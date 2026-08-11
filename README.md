<div align="center">

# Kestrel

**Run frontier-scale GGUF models on the edge hardware you already own.**

Hardware-aware orchestration, conversion, evaluation, and model management for
llama.cpp—without requiring a cloud account or hiding the real memory plan.

[![Tests](https://img.shields.io/github/actions/workflow/status/trowel344/Kestrel/test.yml?branch=main&label=tests&logo=github)](https://github.com/trowel344/Kestrel/actions/workflows/test.yml)
[![Lint](https://img.shields.io/github/actions/workflow/status/trowel344/Kestrel/lint.yml?branch=main&label=lint&logo=ruff)](https://github.com/trowel344/Kestrel/actions/workflows/lint.yml)
[![Distribution](https://img.shields.io/github/actions/workflow/status/trowel344/Kestrel/distribution.yml?branch=main&label=wheel&logo=github)](https://github.com/trowel344/Kestrel/actions/workflows/distribution.yml)
[![Python](https://img.shields.io/badge/python-3.11--3.13-3776ab?logo=python&logoColor=white)](pyproject.toml)
[![License](https://img.shields.io/github/license/trowel344/Kestrel)](LICENSE)
[![Status](https://img.shields.io/badge/status-beta-2f81f7)](CHANGELOG.md)

[Quick start](#quick-start) · [Commands](#command-reference) ·
[How it works](#why-kestrel) · [Contributing](CONTRIBUTING.md) ·
[Security](SECURITY.md)

</div>

Kestrel profiles the host, inspects the model, and enables only capabilities
the installed llama.cpp binary actually supports. It plans a layout across
VRAM, RAM, and disk; handles split GGUFs and sparse MoE models; and keeps its
decisions visible through human and machine-readable CLI output.

> [!NOTE]
> Kestrel is beta software. It is designed to make oversized local models
> runnable and measurable; it does not promise that every model will be fast
> or capable on every machine. Use `kestrel benchmark` and `kestrel evaluate`
> to record what a specific artifact can actually do on your hardware.

> [!IMPORTANT]
> This project is not affiliated with the unrelated `kestrel` distribution on
> PyPI. Until Kestrel has a distinct distribution name, install it from this
> repository or from an attached GitHub Release artifact—not with
> `pip install kestrel`.

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
kestrel evaluate    deterministic capability tests against llama-server
kestrel optimize    create and optionally benchmark a hardware profile
kestrel models      search, list, recommend, info, pull, import
kestrel convert     convert safetensors to GGUF (NVFP4 or --generic)
kestrel audit       validate a GGUF against its source
kestrel build       transactionally build llama.cpp with CUDA and rollback
kestrel self-update install a verified local checkout or checksummed wheel
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

## Measure capability, not just speed

`benchmark` measures prompt processing and decode throughput. `evaluate`
checks whether a running OpenAI-compatible llama-server can complete a small,
deterministic capability manifest and records the model and artifact evidence
in JSON:

```bash
kestrel benchmark /models/model.gguf --repetitions 3 --json
kestrel serve /models/model.gguf --port 8080
kestrel evaluate --endpoint http://127.0.0.1:8080 \
  --artifact /models/model.gguf --sha256 --output evaluation.json
```

These reports are deliberately separate. A model can fit and run reliably yet
remain slow, or generate quickly while failing capability checks. Kestrel does
not collapse those outcomes into a projected score.

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
- **Deterministic evaluation** — records capability results, generation
  metadata, and artifact provenance without silently treating a partial run as
  a full pass.

## Install

```bash
./install.sh                  # isolated .venv install (recommended)
pip install .                 # runtime only (numpy + gguf)
pip install '.[convert]'      # + NVFP4 conversion (torch, safetensors)
pip install '.[dev]'          # + pytest, ruff (for contributors)
```

See `./install.sh --help` for options (`--convert`, `--dir`).

## Supported platform

Kestrel targets Linux for native CUDA/llama.cpp operation. The dependency-light
Python package and offline test suite are continuously checked on Linux and
macOS with Python 3.11, 3.12, and 3.13. GPU behavior still depends on the
selected llama.cpp build and driver stack; `kestrel doctor` is the authoritative
compatibility report for a host.

## Development

```bash
pip install -e '.[dev,convert]'
python -m pytest              # hermetic suite; caches are per-test isolated
ruff check kestrel/ tests/
```

Tests never touch a real user config, model directory, or cache — `tests/`
runs fully offline. CI runs the same suite on Python 3.11, 3.12, and 3.13.

## Repository layout

```text
kestrel/                 installable package (the product)
  cli/                   command-line entry point (main), parser, and command families
    main.py              dispatch + entry point
    parser.py            argument parser
    probes.py            hardware probes (GPU, RAM, power)
    model_source.py      model/GGUF/safetensors resolution
    planning.py          memory-placement planning glue
    runtime.py           llama.cpp launch glue, JSON contracts
    run.py               run / serve / chat
    health.py            doctor / status / setup
    models.py            models subcommands
    bench.py             benchmark / optimize
    menu.py              interactive menu
    engine.py            engine build / status
    updater.py           self-update
    convert.py           convert / audit
  config.py              config/env loading
  ui.py                  std library terminal UI
  model_store.py         model discovery, HF/Ollama stores, snapshot resolution
  core/planner.py        memory placement planning
  core/pipeline.py       llama.cpp wrapper
  backends/llama_cpp.py  process launch, capability probe, metrics
  gguf/converter.py      NVFP4 -> GGUF conversion
  gguf/quants.py         low-level quant/dequant primitives
  gguf/metadata.py       GGUF metadata parsing
  gguf/audit.py          audit
  evals/model_eval.py    deterministic model capability evaluation
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
