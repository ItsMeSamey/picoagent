# Data protocol and honest evidence

## What exists

`data/curriculum-v2` contains 512 train, 64 development, and 64 test **original procedural task specifications**, plus equally many clearly labeled authored references. Eight domains are represented: instruction following, arithmetic, Bash/file processing, Python, local fixture search, fictional documentation, persistent key/value operations, and SVG visualization. These are a narrow bootstrap curriculum, not evidence of broad agent competence.

The whole versioned template/family is assigned to one split. Changing a seed never changes its split. For example `python.square_sort` is train, `python.group_counts` is development, and `python.nested_transform` is test. This is more restrictive than a random row split, but held-out families still share broad skills. Test definitions are published, not a secret benchmark. Keep test files out of training/selection pipelines, freeze a checkpoint before inspecting results, and use independent external evaluation later. Nothing here establishes what was in a base model's pretraining data.

No benchmark examples, external corpora, or web search text are used in the generators. Fixture search explicitly returns `source: original_fixture_corpus`; it is not live SearXNG/web-search evidence. Fictional documentation randomizes flag meanings, defaults, and step ordering rather than merely renaming an API.

The early `data/curriculum/v1` and `data/curriculum-v1` are preserved but excluded in `data/EXCLUSIONS.json`: their authored visualization commands embedded the expected SVG instead of computing it from input. The collector rejects their generator version. Never merge these with v2.

## Reproduce task specs

```bash
python -m picoagent.data generate \
  --output-dir data/new-curriculum \
  --seeds-per-family 64 --holdout-seeds-per-family 8
python -m picoagent.data validate --manifest data/new-curriculum/manifest.json
```

Creation is exclusive: an existing directory is never replaced. JSON serialization, task IDs, seeded generation, and split assignment are deterministic. The manifest hashes each file, the split policy, and the implementation files present at generation. It also records counts and provenance. The source hashes record the generation-time implementation; later collector fixes do not silently rewrite old manifests. Reproduction from the recorded implementation yields identical bytes; current implementation changes can intentionally change source hashes.

## Task contract

Each task includes `schema_version`, `task_id`, `family`, `template_id`, `domain`, `split`, `seed`, `prompt`, `environment`, `oracle`, `reference`, `provenance`, and `input_sha256`. Environment fixtures are bounded UTF-8 files under relative workspace paths, a fresh KV mapping, and an original per-episode document corpus. `input_sha256` covers the prompt and all fixture content.

**Do not expose `oracle` or `reference` to a student.** The shared harness supplies only the system/user messages and tool schemas. It seeds files/KV separately and exposes document content only through tools. Evaluators check final output, actual artifact bytes, and KV postconditions. Oracles are trusted pure Python/JSON/XML checks: no generated code is executed in the evaluator.

The SVG checker verifies actual artifact existence, well-formed static SVG, accessible title, labeled bar values, vertical ordering, and absence of scripts/external references. This is structural visualization validation, not perceptual/aesthetic evaluation.

## Evidence levels

- `authored_example`: generated reference answer/planned actions, **not** observed environment output. `status=unexecuted`, `verification.passed=false`. Pure author-answer checks are recorded separately. These records are rejected by default training admission
- `unexecuted`: a rollout attempt never obtained a real container lifecycle receipt, including unavailable runtime/daemon/image. It cannot claim success
- `verified_environment`: the collector observed a Docker/Podman probe and actual container ID, then recorded actual model/tool interactions and ran task oracles. A successful answer needs `status=success` and `verification.passed=true`

These are evidence contracts, not cryptographic attestations against a malicious archive author. Schema validation checks structure; archive verification ties traces to raw attempts/tasks and hashes. It cannot independently prove that a fabricated remote receipt is true.

## Collect real demonstrations

Preinstall an audited Python/Bash image with GNU `timeout`; prefer a pinned digest. Docker/Podman must already work. The runtime uses `--pull=never`, no network, dropped capabilities, read-only root, non-root UID, resource/output/time limits, and a fresh mounted task workspace. It never falls back to host command execution.

```bash
python -m picoagent.data collect \
  --tasks data/curriculum-v2/train.tasks.jsonl \
  --archive-dir data/attempts/train-v2 \
  --runtime docker --image python:3.11-slim
python -m picoagent.data collect \
  --tasks data/curriculum-v2/dev.tasks.jsonl \
  --archive-dir data/attempts/dev-v2 \
  --runtime docker --image python:3.11-slim
python -m picoagent.data export \
  --archive-dir data/attempts/train-v2 --output-dir data/exports/train-v2
python -m picoagent.data export \
  --archive-dir data/attempts/dev-v2 --output-dir data/exports/dev-v2
python -m picoagent.training prepare \
  --train data/exports/train-v2/train.jsonl \
  --dev data/exports/dev-v2/dev.jsonl \
  --output-dir data/snapshots/v2
```

The CLI uses an explicitly labeled scripted procedural teacher. It requests real tools; it never invents their replies. File-task final answers come from actual stdout; docs/search finals come from retrieved text; KV updates come from actual reads; visualization code loads and transforms the input fixture. Instruction/math reference answers are computed from fully visible problem statements. Scripted-teacher success is **data generation**, never a learned policy evaluation result.

An unavailable runtime is preserved as an error attempt and the CLI exits 2 rather than claiming completion. Individual oracle failures are preserved and make the CLI exit 1. A retry creates a new attempt. Test collection additionally requires `--include-test`; training admission never accepts the test split.

## Full preservation and training admission

Each attempt owns a unique directory containing the original task, request, append-only hash-chained events, raw model requests/responses before protocol validation, shared-harness events, raw final result, artifact bytes/KV snapshot, trace, and an integrity manifest. Provider failures, rejected outputs, tool failures, oracle failures, and retries remain. Hard interruption leaves a partial directory and journal; it is reported as incomplete and never admitted. Invalid trace structure is archived in `invalid_trace.json` after `raw.json` is written.

`export` writes all valid attempts to `all_attempts.jsonl`, including failures. Its success-only per-split view takes one successful attempt per task using a declared deterministic attempt-ID ordering. It does not erase rejected/duplicate attempts. `verify-attempt --attempt-dir PATH` checks hashes, event chaining, task identity, and raw/trace correspondence.

The training preparation step accepts only successful, verified original train/development traces, checks whole-family/template disjointness and conversation/task duplicates, and creates immutable byte snapshots. Authored, benchmark, test, failed, and unexecuted records are excluded.

## Shared protocol, context, and sequence length

All messages use `role`, `content`, canonical `tool_calls` (`id`, `type=function`, `function.name`, JSON-string `function.arguments`), and `tool_call_id` on tool replies. Calls and responses are linked and cannot be orphaned, duplicated, or reordered. Executed Bash/Python/write-file events need actual container receipts. The collector imports the exact inference `DEFAULT_SYSTEM_PROMPT`.

A trace preserves three views:

1. `messages`: full uncompacted chronological interaction for audit and complete tool-event linkage
2. `effective_messages`: final context after any compaction
3. `model_events`: exact input context and output for every assistant decision, plus compaction summary requests/responses. Training can reproduce the context the model actually saw and supervise only that decision

The initial scripted CLI bootstrap deliberately has `context_compaction_enabled=false` and provides **no trained compaction behavior**. The `collect_task` API accepts a real policy callback. With a supplied policy it uses the same `ContextManager`, defaults to 4096 context/512 response reserve, and requires the policy's actual `count_tokens` or an explicit tokenizer counter. Compaction calls are journaled; accepted summaries are separate supervision examples. Set `context_max_tokens=None` only to explicitly disable it. This API support alone is not evidence that a model was trained to summarize.

Prefer a 4096-token initial training configuration, then measure with the exact pinned tokenizer. The encoder rejects oversized traces/examples and never silently truncates final-answer supervision. Tool schemas, escaping, receipt fields, and individual decision contexts all contribute tokens. Do not assert a measured token bound until encoding the actually collected traces.

## Remaining experiments

The bootstrap lacks live web search, rich failure recovery, broad tool APIs, and long-horizon/compaction examples. Add original tasks that require actual installed CLI `--help`, `python -m pydoc`, available man/info pages, and reading novel local module source, then executing the discovered behavior. Keep independently authored API/grammar/composition families held out. Counterfactual paired manuals and state-changing tasks distinguish real observation use from ceremonial tool calls.
