# Kestrel release gates

Kestrel is releasable only when the supported paths meet these gates. Passing
unit tests alone is not a release claim.

## Local Qwen3.5-122B-A10B profile

- Generated-token rate is 10-15 tok/s or better on the documented reference
  machine, measured over at least 64 generated tokens after warmup.
- Prompt ingestion, generated-token rate, model hash, placement, KV type,
  context, batch sizes, thread count, RAM, swap and VRAM are recorded.
- At least 7/8 deterministic accuracy cases pass and the 128-token stability
  response remains coherent, factual, valid UTF-8 and free of repetition
  collapse.
- The suite's automated gate summary passes; the stored stability text still
  receives an explicit human coherence review before a release claim.
- The exact shipped profile survives cold start, a second warm request and an
  interactive chat smoke without CUDA OOM or parser failure.
- Faster profiles that fail quality are labelled experiments and are never
  selected automatically.

## Kimi K3 boundary

- Remote Kimi K3 uses Moonshot's documented OpenAI-compatible API and preserves
  `reasoning_content` in multi-turn history.
- Secrets come only from environment/config and are never printed or stored by
  default.
- Local Kimi K3 is reported unsupported until an engine can load its KDA,
  gated-MLA, Attention Residuals, LatentMoE and vision checkpoint correctly.
- Hardware checks account for the complete 2.8T MXFP4 checkpoint, not only its
  104B active parameters. This laptop must never be presented as locally
  capable of hosting Kimi K3.

## Product and repository

- `kestrel doctor`, local chat, Kimi API chat, conversion and auditing have
  actionable errors and dependency-light tests.
- Wheel and source distribution build from a clean checkout and contain no
  benchmark models, secrets, temporary files or archived experimental engines.
- CI runs tests, compile checks, package builds and wheel-content inspection.
- README quickstarts describe one recommended path before advanced research
  controls.
- License, changelog, contribution guide, security policy and release notes are
  present before a public GitHub release.
