<div align="center">

<img src="assets/kestrel-mark.svg" width="132" alt="Kestrel block-pixel K logo">

# Kestrel

**Run models larger than your memory.**

Hardware-aware local inference for GGUF models across VRAM, RAM, NVMe, and
trusted edge machines—without hiding the plan or requiring a cloud account.

[![Tests](https://img.shields.io/github/actions/workflow/status/trowel344/Kestrel/test.yml?branch=main&label=tests&logo=github)](https://github.com/trowel344/Kestrel/actions/workflows/test.yml)
[![Lint](https://img.shields.io/github/actions/workflow/status/trowel344/Kestrel/lint.yml?branch=main&label=lint&logo=ruff)](https://github.com/trowel344/Kestrel/actions/workflows/lint.yml)
[![Distribution](https://img.shields.io/github/actions/workflow/status/trowel344/Kestrel/distribution.yml?branch=main&label=wheel&logo=github)](https://github.com/trowel344/Kestrel/actions/workflows/distribution.yml)
[![Python](https://img.shields.io/badge/python-3.11--3.13-3776ab?logo=python&logoColor=white)](pyproject.toml)
[![License](https://img.shields.io/github/license/trowel344/Kestrel)](LICENSE)
[![Status](https://img.shields.io/badge/status-beta-f2a33a)](CHANGELOG.md)

[Start](#start-in-two-commands) · [How it works](#how-kestrel-works) ·
[Commands](#command-line) · [Trusted nodes](docs/TRUSTED_NODES.md) ·
[Security](SECURITY.md)

</div>

Kestrel turns an oversized model from **cannot load** into **runs locally**. It
inspects the real GGUF and the installed llama.cpp binary, measures available
hardware, then builds a visible placement plan before launching anything.

> [!IMPORTANT]
> This project is not affiliated with the unrelated `kestrel` distribution on
> PyPI. Install from this repository or a Kestrel GitHub Release artifact—not
> with `pip install kestrel` from PyPI.

## Proven beyond memory

A reference development run on one consumer laptop:

| | Measured configuration |
| --- | --- |
| **Host** | Intel i7-13620H · RTX 4060 Laptop GPU, 8 GiB · 15 GiB system RAM |
| **Model** | Qwen3.5-122B-A10B · sparse MoE |
| **Artifact** | 24.33 GiB IQ1_S GGUF · out of core |
| **Result** | 6.2–6.33 tokens/s sustained decode · stable · 8/8 deterministic capability gate |

The artifact could not fit conventionally in the machine's RAM or VRAM. Sparse
CPU-MoE placement and disk-backed mmap made it runnable without silently
shrinking the model. This is a measured reference, not a universal speed claim:
hardware, quantization, context, storage, and model architecture all matter.

## Start in two commands

```bash
./install.sh
.venv/bin/kestrel
```

Choose **Models → Add a model** once, then **Start chat**. Kestrel automatically
chooses conservative memory and hardware settings for the first run.

Already have a GGUF and prefer the scriptable path?

```bash
.venv/bin/kestrel setup --model /models/model.gguf
.venv/bin/kestrel chat
```

No llama.cpp binary yet? `kestrel build` creates the CUDA-enabled MoE build, or
set `KESTREL_LLAMA_CPP_DIR` to an existing checkout.

## How Kestrel works

```text
GGUF + host + llama.cpp
          │
          ▼
  inspect capabilities
          │
          ▼
  plan the working set
          │
          ├── VRAM
          ├── system RAM
          ├── NVMe / mmap
          └── trusted RPC devices
          │
          ▼
 launch · measure · report
```

1. **Inspect.** Read the complete split-GGUF set, model metadata, free memory,
   accelerator inventory, and the selected engine's actual flags.
2. **Plan.** Choose GPU layers, CPU-MoE placement, KV quantization, mmap/direct
   I/O, tensor splits, and safety margins from live capacity.
3. **Run.** Launch llama.cpp with a bounded CUDA-OOM retry ladder. Kestrel never
   silently changes a distributed run into a local one.
4. **Prove.** Keep projected plans separate from measured benchmark and
   deterministic capability reports.

## Built for oversized local models

- **100 GB+ and split GGUFs** — discover a shard set as one model and cache
  metadata instead of repeatedly reading giant headers.
- **Sparse MoE placement** — keep dense work on the GPU while routing experts
  through CPU and disk-backed memory.
- **Fast cold and warm loading** — opt-in direct I/O for cold NVMe reads; mmap
  remains the warm-reload path.
- **Multiple GPUs and trusted machines** — balance across local accelerators or
  authenticated, protocol-checked llama.cpp RPC workers.
- **Conversion and pruning** — convert NVFP4 Qwen3.5-MoE safetensors to GGUF,
  compact experts to Q1/IQ1_S tiers, prune experts, and carry optional MTP
  drafts.
- **Agent-safe output** — every JSON-mode command emits one machine-readable
  document on stdout while human diagnostics stay on stderr.
- **Dependency-light runtime** — Torch and safetensors are conversion-only
  extras, not an inference requirement.

## Command line

```text
kestrel                 open the minimal interactive menu
kestrel doctor          inspect hardware, storage, models, and llama.cpp
kestrel run             plan and run a local model
kestrel chat            chat with the configured model
kestrel serve           expose an OpenAI-compatible local endpoint
kestrel benchmark       measure prompt and decode throughput
kestrel evaluate        run deterministic capability checks
kestrel models          find, inspect, pull, and import models
kestrel nodes           manage experimental trusted RPC workers
kestrel convert         convert or prune a model into GGUF
kestrel audit           compare a GGUF with its source
kestrel build           build the pinned llama.cpp engine safely
```

<details>
<summary><strong>Runtime controls</strong></summary>

| Flag | Purpose |
| --- | --- |
| `--direct-io` | Bypass the page cache for a cold NVMe load. |
| `--mlock` | Pin CPU-offloaded weights when enough RAM exists. |
| `--tensor-split 60,40` | Override multi-GPU split ratios. |
| `--kv-cache-type q8_0` | Stretch context with a quantized KV cache. |
| `--cpu-moe on` | Keep sparse experts on CPU. |
| `--warm-cache` | Pre-read the model before launch. |
| `--node NAME` | Include a configured, verified RPC worker. |
| `--extra "--temp 0.4"` | Pass an advanced llama.cpp flag through. |

`kestrel run --json --dry-run MODEL` prints the validated plan and complete
command without launching the model. `serve --embeddings` enables the local
OpenAI-compatible embeddings endpoint.

</details>

## Measure capability, not just speed

```bash
kestrel benchmark /models/model.gguf --repetitions 3 --json
kestrel serve /models/model.gguf --port 8080
kestrel evaluate --endpoint http://127.0.0.1:8080 \
  --artifact /models/model.gguf --sha256 --output evaluation.json
```

Benchmark and evaluation reports are deliberately separate. A model can fit
and run reliably while remaining slow, or generate quickly while failing a
capability check. Kestrel does not collapse those outcomes into a projected
score.

## Install and develop

Kestrel targets Linux for native CUDA and llama.cpp operation. The Python
package and offline suite are continuously checked on Linux and macOS with
Python 3.11–3.13.

```bash
./install.sh                  # isolated .venv, recommended
pip install .                 # runtime only: NumPy + gguf
pip install '.[convert]'      # add Torch + safetensors conversion
pip install -e '.[dev,convert]'
python -m pytest
ruff check kestrel/ tests/
```

Tests run offline and do not touch the real user configuration, model store,
or cache. GPU behavior still depends on the selected llama.cpp build and
driver stack; `kestrel doctor` is authoritative for a host.

## Documentation

- [Trusted-node setup and threat boundary](docs/TRUSTED_NODES.md)
- [Security policy](SECURITY.md)
- [Contributing and issue reports](CONTRIBUTING.md)
- [Release history](CHANGELOG.md)

The public Python API is `kestrel.__version__`, `InferencePipeline`,
`LlamaCppBackend`, and `NVFP4Converter`.

## License

MIT. See [LICENSE](LICENSE).
