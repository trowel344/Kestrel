# Kestrel production upgrade audit

Date: 2026-08-02

## Supported product

Kestrel is a hardware-aware launcher, model manager, metrics wrapper, and
converter. The native local runtime remains llama.cpp; provider adapters do
not create a second in-process inference engine. The production import graph is:

```text
kestrel CLI
├── hardware/model inspection
├── runtime planner
├── llama.cpp backend
├── Ollama and Moonshot provider adapters
├── Hugging Face/Ollama model store
└── Qwen3.5-MoE NVFP4 converter (loaded only for conversion)
```

The programmatic `InferencePipeline` also wraps the same llama.cpp backend. No
Python-managed expert cache is on the runtime path. Ollama execution is an
explicit `ollama://` adapter and preserves Ollama's format/runtime ownership.

## Upgrade decisions

| Change | Decision | Current evidence |
|---|---|---|
| Tokenizer metadata repair | Required | Old GGUF emits only `?`; reserved token IDs and tokenizer metadata were incorrect |
| Preserve all 48 target blocks | Required | Old conversion replaced target block 47 with MTP |
| E2M1 FP4 decoding | Required | Source NVFP4 uses a nonlinear E2M1 codebook; signed-int4 decoding is numerically wrong |
| BF16 dense matrices, F32 elementwise vectors | Required | Avoids broad F32 expansion while satisfying llama.cpp CPU binary-op type support |
| Atomic output and space preflight | Required | Conversion writes multi-gigabyte artifacts and must not leave a plausible-looking partial GGUF |
| Twelve-layer dense offload for calibrated IQ1_S on 8 GiB GPU | Required locally | Full-model thread sweep and quality runs validated the 12-layer CPU-MoE placement without OOM |
| Disable MTP on the local oversized profile | Required locally | Equal settings OOM; constrained MTP was slower and accepted 0/87 drafts |
| Q8 KV cache, mmap, and OOM retry | Required operational safeguards | They reduce resident/temporary memory and were exercised by the live launch and test harness |
| Disable full mmap prefetch for strongly sparse MoE | Required locally | The 24.33 GiB artifact runs on 16 GiB RAM with only 8/256 experts active; eager population causes avoidable cache churn and swap pressure |
| Force experimental MoE cache off for CPU-MoE | Required locally | The synchronous CUDA cache path can disable repacking and was slower than direct IQ1_S CPU fallback |
| Python expert cache/router/controller | Archived | Not reachable from CLI and cannot control llama.cpp residency |
| Colibri/Hugging Face engine | Archived | Duplicate inference path, slower prototype, no production entry point |
| Standalone vLLM backend | Archived | Never wired into CLI, planner, packaging, or end-to-end tests |
| DFlash self-speculation | Archived | Router manipulation produced zero accepted draft tokens |
| Triton FP4 decoder | Archived and unsafe | Historical kernel interprets FP4 as signed int4 |

## Enforced boundary

- `kestrel.__init__` exports only `InferencePipeline`, `LlamaCppBackend`, and
  `NVFP4Converter`.
- Retired cache keywords now raise `TypeError` instead of being silently
  ignored.
- `InferencePipeline` and the normal benchmark default to no speculation.
- MTP requires an explicit `--mtp` in `scripts/benchmark.py` or `--mode mtp`
  in the full suite.
- The full benchmark requires an explicit model path; it no longer defaults to
  the known-invalid `/tmp` artifact.
- Packaging has no `research`, `vllm`, or `all` extras. Conversion dependencies
  remain isolated behind `kestrel[convert]`.
- Archived source is under `research/archive` and is absent from the wheel.

## Verification gates

The production boundary is considered intact only if:

1. `python scripts/run_test.py` passes.
2. A wheel build contains only the supported package modules.
3. Importing `kestrel` does not import PyTorch, Transformers, Triton, vLLM, or
   safetensors.
4. Alpha public exports remain absent.
5. The calibrated full artifact must continue to pass at least 7/8 accuracy
   cases plus the stability gate. The current artifact passes 8/8, but its
   measured 6.33 tok/s decode rate does not pass the 10-15 tok/s release gate.
