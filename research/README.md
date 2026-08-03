# Archived alpha research

This directory is a source snapshot of Kestrel's pre-llama.cpp experiments. It
is deliberately outside the installable `kestrel` package and is not supported
as a runnable backend.

The archive exists only to preserve investigation history:

| Area | Why it is not production code |
|---|---|
| Python expert caches and router prediction | Cannot control llama.cpp tensor residency |
| DFlash and controller policy | MoE draft acceptance was invalidated; the controller was not wired into the CLI |
| Colibri/Hugging Face engines | Duplicate the supported llama.cpp execution path and were substantially slower |
| Standalone vLLM wrapper | Was never connected to the CLI, planner, tests, or packaging entry point |
| Triton FP4 decoder | Decodes nibbles as signed integers instead of NVIDIA E2M1 FP4 and is numerically unsafe |

Files under `archive/` retain their historical imports and may not run from
their new location. Do not import them from production code. A future feature
should be implemented against a supported llama.cpp or remote-runtime API and
must have an end-to-end benchmark before it is moved into `kestrel/`.

## Promotion gate

Code may move back into the installable package only when all of these are
true:

1. It is reachable from a documented production entry point.
2. Its output correctness is compared with the source model.
3. Its throughput and memory use are measured on the target hardware.
4. Failure is explicit; no fabricated weights, ignored options, or silent
   fallback is allowed.
5. The default dependency set remains free of unused CUDA frameworks.
