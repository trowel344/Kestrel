# Long-horizon agent memory

Kestrel's Sun Map mode is a hybrid, not a larger transcript. It separates four
jobs that compaction and ordinary RAG often conflate:

1. **Task checkpoint:** a deterministic reducer materializes the objective,
   changed and protected files, failures, attempted approaches, verification,
   warnings, and next action. Replayed tool events are idempotent and SQLite
   revisions reject stale writers.
2. **Semantic memory:** Sun Map retains bounded profile, event, and
   source-grounded record classes with provenance, validity intervals,
   contradiction handling, and HOT/WARM/COLD lifecycle state.
3. **Current code:** a separate hash-versioned FTS index retrieves bounded
   excerpts from the current workspace. Source files do not become privileged
   durable memory and secret-like or binary files are excluded.
4. **Execution supervision:** structured observations detect repeated actions
   without progress, failed edits and tests, stale files, and protected-file
   changes. Raw tool payloads are never persisted in the checkpoint.

The request prefix contains stable semantic memory. Current code and checkpoint
state are appended as labelled, untrusted live evidence. This keeps prompt
caching useful while ensuring operational state refreshes during a tool loop.

## Completion trust boundary

A client tool result can report that tests passed, but the model and proxy do
not thereby prove it. A terminal answer after a client-reported pass becomes
`ready_for_verification`. Only a checkpoint observation from an
`external_gate` or `verified_test_runner` can mark it `complete`; stale files
or protected-file warnings still block completion.

## Research basis

- [LongMemEval](https://arxiv.org/abs/2410.10813) motivates testing information
  extraction, multi-session reasoning, temporal reasoning, knowledge updates,
  and abstention separately. Its session decomposition and fact-augmented keys
  support keeping structured task state alongside retrieval.
- [LeanMem](https://arxiv.org/abs/2608.03463) supports separating profile,
  event, and source-grounded record memories rather than treating every fact as
  one undifferentiated embedding record.
- [SWE-Cycle](https://arxiv.org/abs/2605.13139) treats execution-capable,
  dynamically verified trajectories as the meaningful evaluation boundary.
  That is why Kestrel does not equate route readiness, a terminal answer, or a
  client-reported test string with completion.

These papers inform the architecture; they do not validate this implementation.
Kestrel's bundled interrupted-loop fixture is a deterministic regression. The
remaining research gate is a larger corpus of external multi-session coding
trajectories with real crashes and clean, independent test reruns.

## Current benchmark boundary

On Sun Map's synthetic 84-event interrupted-loop fixture, hybrid checkpoint
plus semantic memory reaches 1.0 task recall and 1.0 resume recall. Extractive
compaction reaches 0.6 and 0.667; keyword RAG reaches 0.6 and 0.333. Hybrid uses
103.4 estimated tokens on average, versus 758.0 for compaction and 184.8 for
RAG. This is down from the initial hybrid's 287.6-token result through
query-aware task projection and compact provenance labels, with recall
unchanged. The fixture is intentionally tailored to test loss of operational state,
so these numbers are evidence for the reducer invariant, not universal
superiority over learned retrieval or model-generated summaries.

## Controlled managed smoke (2026-08-14)

An isolated Git workspace was served through Kestrel's authenticated managed
proxy using an 8K `qwen3.5:4b` Ollama alias and a private Sun Map database. The
model received a hash-versioned `ledger.py` excerpt and correctly identified
that `total(values)` returns `sum(values)`. After stopping and restarting the
managed server, its reasoning recovered the prior objective and correctly said
that no passing verification was recorded. Route authentication, code evidence,
checkpoint persistence, model restart, and process cleanup therefore passed.

The model repeatedly spent its response allowance in a verbose reasoning field;
one response produced a truncated final sentence and the restart query produced
no final-content field before the token cap. This is a Qwen/Ollama response-
quality failure, so the smoke is not counted as a clean end-user answer pass.

## Token allocation policy

The model-facing checkpoint is a projection of the durable full state. It
always retains protected paths, omits an objective already present in the user
request, and selects objective, change, verification, failure, or next-action
fields for direct state questions. Diagnostic revisions, duplicate failures,
raw hashes, scores, and source UUIDs remain stored but are not injected unless
needed.

Kestrel caps task state at 192 tokens and current code at one fifth of the total
budget, but allocates those channels only when the query needs them. Unused code
capacity returns to semantic evidence; direct operational questions use the
canonical checkpoint without unrelated semantic history. Code results use
query-centred 32-line windows, at most eight relevant
symbols, and suppress overlapping chunks that add no new query terms. Runtime
status reports the last estimated semantic, code, task, and total injected
tokens so future quality gains can be evaluated against their prompt cost.
Low-information questions, acknowledgements, and task commands already covered
by the checkpoint remain in the immutable event archive but are not admitted as
retrievable memories. This prevents long sessions from turning routine chatter
into future prompt competition.

For small-model containment, commands introduced with wording such as
`run exactly 'python -m pytest -q'` are stored as task-contract state and placed
ahead of semantic context on every turn. A different verification command is
recorded as a contract warning. Three failed actions without workspace progress,
or a repeated identical action, emits a strong `STOP` recovery guard that tells
the model to use one direct read/edit and then the exact required gate. This
cannot make a 4B model reason like a larger model, but it reduces command drift,
looping, and false completion claims.

### Managed prompt-token comparison

The isolated `qwen3.5:4b` 8K managed smoke was repeated with the same workspace,
question, 2,048-token Sun Map ceiling, and Ollama API usage accounting. The
first code-evidence request fell from 269 to 159 actual prompt tokens (40.9%).
After a complete Kestrel stop/start, the persisted-state question fell from 418
to 236 prompt tokens (43.5%). Qwen's reasoning still recovered `total(values)`
and correctly recognized that no passing verification existed.

These are paired measurements on one local model/template, not universal
tokenizer results. Qwen continued to spend its completion allowance in the
reasoning field without producing final content, so this establishes retained
context utility and lower prompt cost, not improved answer generation.

## Recorded Pi replay and model compaction baseline (2026-08-14)

Kestrel now has an opt-in privacy-bounded trajectory recorder and Sun Map can
replay that schema directly. A real two-session Pi run through a 16K Qwen 3.5
4B Ollama alias produced 106 records and 1,733 replay tokens. The first session
issued many malformed diagnostic commands, repeated actions, made two edits,
and stopped with verification failed. A fresh session recovered the checkpoint,
fixed the remaining zero-handling bug, and recorded 3 passed, 0 failed. An
independent `python -m pytest -q` rerun passed all three tests.
The run also found that unquoted comma-separated protected filenames were not
materialized into the checkpoint. That parser now recognizes filename-shaped
tokens plus explicit database and trace categories.

Under the same 256-token ceiling and with Qwen as both reader and real
model-generated compactor, hybrid Sun Map and compaction each recalled all
three current facts. Hybrid averaged 39 injected tokens; compaction used 256.
Keyword RAG recalled none. This supports checkpoint-first conditional context
on this noisy trajectory, but is too small to establish general superiority.
Exact-string oracle scoring also gives paraphrased compaction zero, which is
why both deterministic retention and semantic-reader results are retained.
