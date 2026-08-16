# Adaptive runtime tuning

Kestrel separates safe first-launch planning from measured optimization.
Planner heuristics never guess aggressive expert placement or large physical
micro-batches. Instead, a one-time tuning run searches a bounded set of
placements for the exact model artifact and host:

```bash
kestrel optimize /path/to/model.gguf --context 16384 --benchmark
```

The search measures:

- the planner baseline;
- full non-expert GPU offload with all MoE experts on CPU;
- larger logical and physical prompt-processing batches;
- a bounded one-eighth slice of expert layers on GPU when the estimated VRAM
  floor fits; and
- a CPU thread sweep for the selected hybrid placement.

Candidates that exceed the conservative VRAM floor are skipped. Backend
failures and OOMs are recorded as failed candidates rather than selected.
The balanced objective weights prompt processing more heavily than decode
because coding-agent turns repeatedly ingest tool history.

## Profile identity and activation

Profiles live under `~/.config/kestrel/hardware-profiles/`. They are keyed and
validated against:

- the resolved GGUF path, byte size, and modification time;
- CPU identity and logical CPU count;
- GPU names and total VRAM;
- the exact `llama-server` path, size, and modification time;
- the context size used during tuning; and
- a minimum free-VRAM floor.

`run`, `serve`, coding-agent servers, and the local benchmark path consume a
profile automatically only when every identity matches, the requested context
does not exceed the tuned context, and current free VRAM meets the floor.
Otherwise Kestrel fails closed to its conservative planner. Explicit
`--gpu-layers` or `--cpu-moe` choices always override automatic tuning.

The profile changes placement and execution batching only. It does not replace,
requantize, or modify model weights.

## Startup recovery

Interactive llama.cpp launches retain an OOM retry ladder. A retry:

1. halves the physical micro-batch;
2. moves half of the remaining GPU expert layers back to CPU; and
3. increases the VRAM fit margin.

Kestrel never retries an identical command and does not treat unrelated CUDA
errors as memory failures.
