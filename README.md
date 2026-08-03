# Kestrel

Kestrel is a hardware-aware model launcher, conversion toolkit, and model
manager for running local models on consumer hardware. Its native large-MoE
path uses llama.cpp; its Ollama adapter can reuse models already managed by
Ollama without copying their weights.

Kestrel does not implement a second inference engine beside llama.cpp. It
profiles the machine, inspects the model and the installed llama.cpp binary,
then selects only capabilities that binary actually supports.

## Quick start

```bash
python -m pip install .
kestrel doctor
kestrel setup --model /path/to/model.gguf
kestrel chat
```

For the interactive front door, run `kestrel` in a terminal or use the
explicit command:

```bash
kestrel menu
```

The menu exposes chat, adjustable context, local/Ollama model discovery,
Hugging Face downloads, benchmarks, diagnostics, and configuration. Every
menu action invokes the same scriptable commands available to automation.
Local llama.cpp launches use a conservative hardware-aware context tier by
default; override it explicitly with `--ctx-size 8192` or request the policy
directly with `--ctx-size auto`.

## Models: Kestrel, Ollama, and Hugging Face

List Kestrel-managed models and models already installed through Ollama:

```bash
kestrel models list
kestrel models list --resolve       # show reusable local Ollama blobs
kestrel models list --json
kestrel models recommend           # rank installed models for this hardware
```

Search the live Hugging Face GGUF catalog and inspect the actual variants
before downloading one:

```bash
kestrel models search qwen3.5 --limit 10
kestrel models files hf://unsloth/Qwen3.5-4B-GGUF
```

Search results come from Hugging Face's official CLI and are explicitly GGUF
filtered. Kestrel shows popularity, license tags, file sizes, hashes in JSON,
and Hub scan state, but does not mistake popularity or a remote scan for a
quality/security guarantee.

Kestrel reads Ollama's own model metadata rather than guessing storage paths.
It records Ollama imports as `ollama://` provider references, so Ollama keeps
responsibility for format compatibility and no multi-gigabyte copy is needed:

```bash
kestrel models info ollama://qwen3.5:4b
kestrel models import ollama://qwen3.5:4b --set-default
```

Cloud-only Ollama entries are reported explicitly. They can be selected
through the provider adapter, but are never described as local weights.
Pulling works through the installed upstream client:

```bash
kestrel models pull ollama://qwen3.5:4b
```

`models recommend` uses detected VRAM and currently available RAM to classify
installed artifacts as GPU-resident, viable with offload, paging-bound, or
unsupported. It deliberately does not turn that memory classification into a
made-up performance claim; qualify the selected placement with
`kestrel benchmark`.

Tune CPU-MoE thread count in one model load instead of repeatedly reloading a
large GGUF:

```bash
kestrel benchmark /path/to/model.gguf --threads 8,10,12,14,16
```

The report retains every result and promotes the fastest decode row together
with its matching prompt rate.

Create an explainable hardware plan, optionally followed by a real benchmark:

```bash
kestrel optimize ollama://qwen3.5:4b
kestrel optimize ollama://qwen3.5:4b --context 8192 --benchmark
```

The saved hardware profile records the benchmark as `measured`, `failed`, or
`not_run`. A failed engine compatibility check is never silently presented as
a completed optimization. Analytical throughput projections are labelled
uncalibrated and never count toward the release speed gate; only the recorded
benchmark does.

Hugging Face acquisition uses the official `hf` CLI, supports revision-pinned
sources and performs a dry-run before large downloads:

```bash
kestrel models pull hf://OWNER/REPOSITORY --dry-run
kestrel models pull hf://OWNER/REPOSITORY --revision COMMIT
kestrel models pull hf://OWNER/REPOSITORY --file model.gguf
kestrel models pull hf://OWNER/REPOSITORY --include 'model-Q4_K_M-*.gguf' --set-default
```

`--include` is the safe path for split GGUFs. Kestrel sets a downloaded model
as default only when it finds one standalone GGUF or every shard of exactly
one split model; incomplete or ambiguous downloads fail explicitly.

Downloads land under `~/.local/share/kestrel/models` by default. Override that
with `KESTREL_MODELS_DIR` or persist it with:

```bash
kestrel setup --models-dir /fast/storage/kestrel-models
```

You can also skip persistent setup and pass the model directly:

```bash
kestrel chat /path/to/model.gguf
```

`kestrel setup` stores only safe defaults in
`~/.config/kestrel/config.toml`; it never stores provider credentials. After
setup—or after placing the Qwen3.5 artifact at Kestrel's managed path or
configuring `KESTREL_QWEN35_122B_GGUF`—the model argument is optional:

```bash
kestrel chat
```

Kimi K3 is available through Moonshot's official API path:

```bash
export KIMI_API_KEY="..."
kestrel kimi --check
kestrel kimi "Explain this repository's architecture"
kestrel kimi                         # interactive
```

Kestrel does not claim that this laptop can run Kimi K3 locally. The complete
checkpoint is 2.8T parameters in native MXFP4—roughly 1.4 TB of weight payload
before runtime overhead—and its KDA, gated-MLA, Attention Residuals and vision
graph are not implemented by the current llama.cpp path. `kestrel kimi --local`
fails with that boundary explicitly instead of attempting a doomed download.

## What the default runtime does

- Uses a measured fixed offload only for a recognized model/hardware profile;
  other layouts keep a conservative fallback and llama.cpp allocation fitting.
- Reserves a VRAM safety margin to absorb CUDA context and fragmentation.
- Keeps MoE weights on CPU when the model is much larger than available VRAM.
- Keeps GGUF mmap enabled, avoiding a full model copy in process memory.
- Uses Q8 KV caches and conservative micro-batches by default.
- Uses the benchmarked 12-layer, 14-thread, 256/64 batch profile for the exact
  Qwen3.5-122B-A10B shape on an 8 GiB GPU.
- Enables MTP only when model metadata, llama.cpp support, and the memory
  profile make it viable; oversized CPU-MoE models on 8 GiB GPUs disable it.
- Retries startup CUDA OOMs with a smaller micro-batch and larger VRAM margin.
- Never allocates Kestrel's experimental Python expert cache during inference.

These defaults prioritize completing a launch without an OOM. They can be
overridden for benchmarking.

## Install

The default installation is intentionally light:

```bash
python -m pip install -e .
```

Install conversion dependencies only when needed:

```bash
python -m pip install -e '.[convert]'
```

## Check the machine

```bash
kestrel doctor
```

This reports GPU memory, available RAM, the selected llama.cpp binary, and
support for fitting, CPU MoE, mmap, quantized KV cache, and MTP.

## Run a GGUF

Inspect the complete launch without starting a large allocation:

```bash
kestrel run /models/model.gguf --dry-run
```

Run with conservative defaults:

```bash
kestrel run /models/model.gguf
```

Useful memory controls:

```bash
kestrel run model.gguf \
  --cpu-moe on \
  --fit-target 1800 \
  --ctx-size 2048 \
  --threads 14 \
  --ubatch-size 64 \
  --kv-cache-type q8_0
```

If a launch still OOMs, reduce physical micro-batch first, increase the fit
margin second, and reduce context length third:

```bash
kestrel run model.gguf --ubatch-size 64 --fit-target 2048 --ctx-size 1024
```

Q4 KV cache is available as a more aggressive context-memory tradeoff:

```bash
kestrel run model.gguf --kv-cache-type q4_0
```

The native MoE expert cache is experimental and is forced off for CPU-MoE
plans. Its current CPU-graph hook synchronizes every cached expert matrix, can
disable CPU repacking, and is slower than the compact and full-model fallbacks
on the reference RTX 4060 Laptop. Use `--moe-cache 2048` only for controlled
cache experiments until the native CUDA graph path lands.

A compact Q1 expert model can reuse an existing canonical Q4 GGUF as immutable
hot backing without duplicating its Q4 payload:

```bash
kestrel run compact-q1.gguf \
  --moe-cache 2048 \
  --moe-hot-model canonical-q4.gguf
```

The patched llama.cpp runtime mmaps the sidecar without prefetching it. Q1
misses remain the immediate CPU fallback; admitted Q4 experts are copied from
the sidecar into GPU cache slots for later routes. It never creates, deletes,
or requantizes an expert during inference. Supplying a sidecar is explicit and
fail-closed: missing files, missing layers, wrong shapes, and non-Q4 expert
tensors abort context creation. This remains an experimental cache path, not a
claim that Q1 fallback preserves Q4 quality on cache misses.

The smaller recommended artifact layout keeps that canonical Q4 GGUF as the
primary model and adds only an experts-only Q1 cold sidecar:

```bash
kestrel run canonical-q4.gguf \
  --moe-cache 2048 \
  --moe-cold-model experts-q1.gguf
```

This retains the exact canonical dense tensors and avoids duplicating them in a
compact primary model. The cold sidecar is also mmap-backed and fail-closed.
The two sidecar directions are mutually exclusive.

This layout is a research mechanism, not a quality-preserving serving profile.
On the complete 48-layer Qwen3.5-122B-A10B model, cache-off Q1 fallback measured
3.21 generated tok/s and only 3.94 prompt tok/s at a 512-token prompt. A
deterministic end-to-end smoke test produced a repeated mixed-language loop,
while the same canonical Q4 control remained coherent. Fixed 1 GiB and 2 GiB
Q4 caches were slower still (2.21 and 2.23 tok/s). Keep the cache disabled and
do not treat this Q4-derived Q1 sidecar as a usable quality profile.

Use `--no-oom-retry` when benchmarking a fixed configuration.

## Convert supported NVIDIA NVFP4 models

```bash
kestrel convert /path/to/safetensors-model -o /path/to/model.gguf
```

For the compact half of the sidecar layout, use:

```bash
kestrel convert /path/to/safetensors-model \
  --dense-q4 --cold-tier q1_only \
  --q4-sidecar-source /path/to/canonical-q4.gguf \
  -o /path/to/compact-q1.gguf
```

`--cold-tier q1_0` emits a self-contained GGUF containing both complete tiers;
`q1_only` avoids duplicating the canonical Q4 payload when a separate hot
sidecar already exists. `--q4-sidecar-source` derives the cold experts directly
from that existing Q4 GGUF, so the converter never builds or retains another
complete Q4 expert tier. This is the fast path, but it adds Q4-to-Q1 cascading
error; omit the option to derive Q1 directly from the source NVFP4 weights when
maximizing cold-tier quality matters more than conversion time.

For sensitivity experiments, a direct-source compact primary can keep the
first and last `N` routed-expert layers at Q2_K while using Q1_0 elsewhere:

```bash
kestrel convert /path/to/safetensors-model \
  --dense-q4 --cold-tier q1_only --q2-edge-layers 1 \
  -o /path/to/mixed-q2-q1.gguf
```

This requires a built llama.cpp `libggml-base.so` (or an explicit
`KESTREL_GGML_BASE_LIB`) and cannot be combined with `--q4-sidecar-source`.
It is an R&D format, not a quality recommendation: on Qwen3.5-122B-A10B,
Q2_K at only the first and last layers did not prevent Q1_0 generation
collapse in the full model.

IQ1_S is the higher-quality compact candidate. It requires an activation
importance matrix; Kestrel refuses an uncalibrated full conversion:

```bash
llama-imatrix -m canonical-q4.gguf -f calibration.txt \
  -o calibration-imatrix.gguf --no-ppl

kestrel convert /path/to/safetensors-model \
  --dense-q4 --cold-tier q1_only \
  --compact-expert-type iq1_s --imatrix calibration-imatrix.gguf \
  -o compact-iq1s.gguf
```

This remains experimental until a complete artifact passes the performance
and quality requirements in `RELEASE_GATES.md`.

To omit dense tensors as well and produce the recommended cold sidecar:

```bash
kestrel convert /path/to/safetensors-model \
  --cold-tier q1_only --experts-only \
  --q4-sidecar-source /path/to/canonical-q4.gguf \
  -o /path/to/experts-q1.gguf

kestrel audit /path/to/experts-q1.gguf --cold-sidecar
```

For this 48-layer model the experts-only payload is approximately 15.19 GiB,
about 3 GiB smaller than a complete compact-primary GGUF.

The converter currently targets the Qwen3.5-MoE NVFP4 layout. It preflights
output disk space, writes to an atomic `.partial` file, streams experts instead
of retaining a converted layer in RAM, preserves or explicitly quantizes dense
matrices, writes
elementwise vectors as F32 for llama.cpp CPU-kernel compatibility, and fails on
missing tensors. It does not yet justify an "any MoE model" claim.
The optional MTP draft block is omitted by default because it does not affect
target-model quality and consumes several GiB. Pass `--include-mtp` only when
benchmarking speculative decoding on hardware with enough memory.

Audit an artifact before spending hours on a real-model benchmark:

```bash
kestrel audit /path/to/model.gguf --source /path/to/safetensors-model
```

The source cross-check is strongly recommended for Kestrel conversions. It
detects missing or replaced target blocks, MTP placement errors, fabricated
token IDs, empty normal tokens, and missing tokenizer metadata. The command
returns a non-zero status when a correctness error is found; `--json` emits a
machine-readable report.

Before choosing the faster Q4-to-Q1 cascade for a full build, measure its added
weight error across every layer:

```bash
python -m kestrel.analysis.q1_cascade_quality \
  --q4-gguf /path/to/canonical-q4.gguf \
  --experts-per-layer 4 \
  --output /path/to/q1-cascade-audit.json
```

This bounded-memory audit compares the actual decoded Q1 encodings from direct
NVFP4-to-Q1 and cascaded NVFP4-to-Q4-to-Q1 conversion. It is a preflight, not a
substitute for end-to-end full-model quality evaluation.

## Verification

```bash
python scripts/run_test.py
python scripts/benchmark.py /path/to/model.gguf
```

`run_test.py` is the automated unit suite. `benchmark.py` is the real-model
benchmark. MTP is opt-in there because the corrected converter omits the draft
block by default; use `--mtp` only with a GGUF that intentionally includes it.

## Production boundary

The installable package contains only:

- `kestrel.cli`: hardware/model detection and the command-line entry point.
- `kestrel.core.planner`: conservative runtime policy.
- `kestrel.core.pipeline`: the supported programmatic llama.cpp wrapper.
- `kestrel.backends.llama_cpp`: process launch and metrics.
- `kestrel.gguf.converter`: the narrowly scoped Qwen3.5-MoE conversion path.

Old Python inference, expert-cache, router, DFlash, controller, Triton, and
unwired vLLM prototypes are preserved under `research/archive`, outside the
installed package. See `research/README.md` for the evidence and promotion
gate. An external Python process cannot manage llama.cpp tensor residency
without a supported API or patch.

## Repository layout

```
kestrel/                 installable package (the supported surface)
  cli.py                 command-line entry point, menus, --target planning
  config.py              config/env loading
  ui.py                  stdlib terminal UI (menus, tables, prompts)
  model_store.py         local + Ollama model discovery, blob resolution
  core/planner.py        memory-adaptive placement planning
  core/pipeline.py       programmatic llama.cpp wrapper
  backends/llama_cpp.py  process launch, capability probe, metrics
  gguf/converter.py      Qwen3.5-MoE NVFP4 -> GGUF conversion (+ expert pruning)
  gguf/metadata.py       GGUF metadata parsing
  gguf/audit.py          sidecar / cold-snapshot auditing
  analysis/              heuristic analysis (MoE routes, Q1 cascade quality)
  providers/             remote runtimes (Ollama, Kimi)

scripts/                 developer / release tooling
  run_test.py            dependency-light unit suite
  full_model_suite.py    end-to-end real-model performance suite
  benchmark.py           real-GGUF benchmark
  benchmark_speedups.py  stdlib-only loading-speedup benchmark (no real binary)
  check_wheel.py         CI wheel-content boundary inspection

tools/                   small native binaries + sources for GGUF repairs
research/archive/        pre-llama.cpp prototypes (deliberately unsupported)
tests/                   unit tests mirroring the kestrel package
docs (root)              README.md, RELEASE_GATES.md, PRODUCTION_AUDIT.md,
                         SECURITY.md, CONTRIBUTING.md, CHANGELOG.md
```

Everything outside `kestrel/` is tooling, tests, or history. The public API is
only `kestrel.__version__`, `kestrel.InferencePipeline`,
`kestrel.LlamaCppBackend` and `kestrel.NVFP4Converter` (exported lazily so
importing `kestrel` never pulls torch/transformers/vllm/safetensors).
