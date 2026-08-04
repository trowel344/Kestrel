# Kestrel

Hardware-aware local model launcher, conversion toolkit, and model manager.

Kestrel profiles the machine, inspects the model, and picks only the capabilities
the installed llama.cpp binary actually supports. Its native local runtime is
llama.cpp; provider adapters let it reuse models Ollama already manages without
copying weights.

## Quick start

```bash
# one-command install (creates a virtual environment)
./install.sh
.venv/bin/kestrel doctor          # check GPU, RAM, and llama.cpp capabilities
.venv/bin/kestrel setup --model /path/to/model.gguf
.venv/bin/kestrel chat            # chat with the configured default model
```

Or install with pip directly: `pip install .`. For the interactive menu, run
`kestrel` or `kestrel menu`. Skip setup and pass a model:
`kestrel chat /path/to/model.gguf`.

## Command reference

```text
kestrel menu        interactive menu
kestrel doctor      check hardware and llama.cpp capabilities
kestrel status      active model, hardware plan, benchmark state
kestrel setup       save safe local defaults (never API keys)
kestrel run         plan and run a local model
kestrel chat        chat with the configured local model
kestrel benchmark   reproduce prompt/decode measurements
kestrel optimize    create and optionally benchmark a hardware profile
kestrel models      search, list, recommend, info, pull, import
kestrel convert     convert supported NVFP4 safetensors to GGUF
kestrel audit       validate a GGUF against its source
kestrel build       build llama.cpp with CUDA
```

`kestrel doctor` reports GPU memory, available RAM, the selected llama.cpp
binary, and support for fitting, CPU MoE, mmap, quantized KV cache, and MTP.

## Features

- **Hardware-aware planning** — measures VRAM, RAM, and llama.cpp support and
  picks a conservative memory layout that avoids OOM (CPU-MoE placement, mmap,
  Q8 KV caches, VRAM margins, startup CUDA OOM retry).
- **Local + remote models** — discover, inspect, rank, pull, and import models
  from disk, Hugging Face (GGUF-filtered), and Ollama.
- **NVFP4 conversion** — converts Qwen3.5-MoE safetensors to GGUF with dense Q4,
  compact Q1/IQ1_S expert tiers, expert pruning, optional MTP drafts.
- **GGUF auditing** — validates an artifact against its source model.
- **Reproducible benchmark** — separates measured results from projections.
- **Dependency-light** — importing `kestrel` pulls only numpy and gguf;
  conversion deps stay behind `kestrel[convert]`.

## Install

```bash
./install.sh                  # isolated .venv install (recommended)
pip install .                 # install into current environment
pip install '.[convert]'      # also install NVFP4 conversion extras (torch)
```

See `./install.sh --help` for options (`--convert`, `--dir`).

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