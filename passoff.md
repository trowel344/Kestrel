# Kestrel Engineering Passoff

## Current supported architecture

Kestrel is a memory-aware orchestration layer over llama.cpp. llama.cpp owns
model execution and tensor residency. Kestrel owns:

1. Local model and GPU detection.
2. GGUF metadata inspection.
3. Installed llama.cpp capability detection.
4. Conservative runtime planning.
5. Supported Qwen3.5-MoE NVFP4-to-GGUF conversion.
6. Launch and metrics collection.

The default runtime uses a conservative explicit offload cap on 8 GiB GPUs
(llama.cpp automatic fitting on larger devices), a VRAM target margin,
mmap, CPU-resident MoE weights for oversized models, Q8 KV caches, conservative
micro-batches, and MTP only when the memory profile makes it viable.

## Evidence as of July 30, 2026

- Local GPU: RTX 4060 Laptop, 8188 MiB.
- Local llama.cpp: build 9285 (`9c92e96a6`).
- Supported by that binary: `--fit`, `--fit-target`, `--cpu-moe`, mmap,
  quantized K/V caches, and `--spec-type draft-mtp`.
- The old flags `--spec-type mtp` and `--moe-hot-expert-k` are invalid in this
  build and are no longer generated.
- The Qwen3.5-122B-A10B source model is approximately 78 GiB.
- The existing `/tmp/qwen3.5-122b-a10b-nvfp4.gguf` is an old 86.96 GiB
  conversion. It loads with four GPU layers, but every tested completion
  detokenizes as `?`; it is not a valid model artifact.
- Measured baseline decode is 0.138-0.145 tok/s. Constrained MTP is slower at
  0.0719 tok/s and accepted 0/87 draft tokens. See
  `benchmark_results/FULL_MODEL_REPORT.md`.
- The old converter emitted reserved vocabulary IDs as empty normal tokens,
  omitted token types/pre-tokenizer/chat-template metadata, and fabricated a
  BOS ID. The converter now emits named UNUSED padding tokens and preserves the
  source tokenizer metadata.
- The old converter also replaced target block 47 with the MTP block. Block
  accounting now preserves all 48 target layers and treats MTP as an optional
  49th block. Target-only conversion is the default because omitting a draft
  head does not change target quality.
- The revised target-only GGUF converter estimates a 72.22 GiB output. The current disk has
  only about 19 GiB free, so conversion now fails during preflight without
  producing a partial file.
- Dependency-light unit tests run both in the system Python and the repository
  virtual environment.

## Retired or experimental paths

These modules have been moved out of the installable package to
`research/archive`:

| Component | Status | Reason |
|---|---|---|
| Custom PyTorch Colibri engine | Retired | Measured around 0.57 tok/s; duplicates llama.cpp work |
| DFlash self-speculation | Invalidated | Changing MoE router top-k caused zero acceptance |
| Python `MultiTierCache` | Experimental | Cannot control llama.cpp tensor residency |
| All-layer router-first scan | Approximate research | Deeper routers require deeper hidden states |
| Triton NVFP4 dequantizer | Reference | Native llama.cpp NVFP4 is the supported path |

The research cache now loads L2 weights on CPU, evicts before promotion, and
fails on load errors instead of fabricating random weights. It still must not be
described as active llama.cpp orchestration.

## Converter state

The converter is intentionally narrow: it targets the NVIDIA
Qwen3.5-MoE NVFP4 tensor layout. Improvements now in place:

- Dense tensors are BF16 rather than incorrectly expanded to F32.
- E2M1 FP4 values use the correct nonlinear codebook.
- Expert tensors stream to the output instead of accumulating a layer in RAM.
- Output space is checked before conversion.
- Output is written atomically through a `.partial` path.
- Missing tensors and byte-count mismatches fail loudly.

Before claiming converter completion, produce a full GGUF on a volume with at
least 80 GiB free, run llama.cpp tensor validation, compare deterministic
prompts with the Hugging Face source, and measure output divergence.

## Immediate next engineering work

1. Validate a full converted model on adequate storage.
2. Parse llama.cpp structured metrics only if the installed build exposes a
   stable interface and the existing parser becomes a demonstrated problem.
3. Do not resume expert-cache or remote-runtime work unless a supported API,
   target deployment, and end-to-end benchmark make it a real product need.

## Commands

```bash
kestrel doctor
kestrel run /path/to/model.gguf --dry-run
kestrel run /path/to/model.gguf
python scripts/run_test.py
python scripts/benchmark.py /path/to/model.gguf
```
