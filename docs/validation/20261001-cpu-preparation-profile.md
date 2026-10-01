# CPU preparation cost and checkpoint timing

Read-only local profile of source commit
`c2f82ade626f1fc6501885315c507572fbd25bb7` and artificial-plan dataset manifest
`e5774693bc42815ab44858754303b3970c7e34f26a1140425dccdd5853ccfbd9`.
No model download/load, training, accelerator, provider action, or recorded tool
command was executed. The active Colab run was not modified.

## Measured locally

- Strict `verify_dataset`: **487.5 seconds** (8.1 minutes)
- Production `encode_trace` calls: **174.6 seconds**
- Exact model-event expansion: **7.0 seconds**
- Comparable verification + encoding work: **669.1 seconds** (11.2 minutes)
- Complete instrumented run: **697.6 seconds**, including extra fingerprint and
  statistics work
- Streaming profiler peak RSS: **1,294.9 MiB** (about 1.26 GiB)

The 49,905 examples match the previously audited final dataset: 48,702 train and
1,203 dev; 53,929,387 total tokens; 6,146,621 supervised tokens; maximum length
3,445 with a frozen 4,096 limit. Encoding was streamed and each completed item
discarded. The actual trainer currently retains the encoded lists. Their Python
container/integer storage is estimated at **2,848,666,268 bytes** (2.65 GiB),
excluding source rows, tokenizer, model, framework and allocator overhead. This
is an estimate, not a measured trainer RSS increase.

Largest verification costs, with overlapping inclusive timers:

- Frozen-tokenizer compaction budget checks: **247.5 seconds**, including all
  648 original mode variants and 16 retention variants, not only selected views
- Gzip/JSON canonical metadata checks: **64.7 seconds**
- Receipt/projection bindings: **56.9 seconds**
- Physical file hashing: **7.9 seconds**
- Independent fixture oracles: **3.3 seconds**

Do not sum these breakdown entries: some parent timings contain child work.
Page-cache, CPU and environment differences limit comparisons with Colab. This
profile does not establish the cause of a particular remote delay. In the pinned
trainer, strict verification precedes Hub tokenizer lookup, encoding and model
load. `run_status.json` is created after those phases, so its absence alone does
not identify the current initialization phase.

## Safe improvement for future runs

Prepare a separately admitted token artifact on CPU before allocating a GPU:

1. Run the existing strict verifier with both native and artificial-plan opt-ins.
   Encode its returned augmented rows, including the 5,004 selected CLI notes.
   Validate the actual pinned tokenizer, exact assistant masks and no truncation.
2. Bind the exact raw-data manifest and evidence inventory, tokenizer revision and
   asset bytes, effective special-token/padding settings, encoder/protocol source
   identities, package versions, max length, and label policy.
3. Preserve split order, source-row and original event ordinal, task/trace IDs,
   input/output/tool-schema identities, and every `input_ids`, `attention_mask`
   and `labels` value. Use bounded non-executable serialization, never pickle.
   Round-trip and independently rebuild for exact logical equality.
4. Independently approve the prepared-manifest digest through a reviewed release
   record/allowlist or trusted build attestation. A self-hashed cache claiming
   `verified=true` is not proof of correct derivation.
5. On the accelerator, check the approved digest and all source/cache bytes,
   paths, integer arrays, lengths, labels, counts and order. Bind the artifact to
   immutable run/resume identity. Fail closed on mismatches; preserve ordinary
   strict raw admission. The existing private copy helper is same-process-only
   and must not be reused as cross-process admission.

Primitive memory-mapped arrays could reduce retained Python-object memory, but
need exact production-equivalence tests. A cache-enabled trainer has a different
source/resume identity; it must not be silently substituted into the active run.
Future startup phase receipts would also make validation, tokenization, model
loading and snapshot-copy delays observable without exposing auth values.

## Checkpoint evaluation delay

Source inspection confirms the wall-clock callback sets both `should_save` and
`should_evaluate`. The installed Transformers `Trainer._maybe_log_save_evaluate`
runs evaluation **before** `_save_checkpoint` and the sealing `on_save` callback.

The parent observed approximately **2 dev examples/second** on T4. With 1,203
examples, full dev evaluation is therefore estimated at **601.5 seconds**. A
nominal 600-second save timer can imply roughly **1,201.5 seconds before a durable
save**, plus serialization/export. This is a calculation from a reported remote
throughput, not a locally measured evaluation or guaranteed upper bound.

For a future reviewed run, separate durability from evaluation: seal and export
at a completed optimizer step first, then evaluate that exact immutable checkpoint
on a separately frozen cadence. Preserve training RNG/state and test resumed
versus uninterrupted equivalence. Dev loss remains distinct from agent success
and benchmark evaluation. No current checkpoint policy was changed.

## Reproduce and inspect

- [Raw measurements](20261001-cpu-preparation-profile.json)
- [Machine-readable recommendations](20261001-cpu-preparation-recommendation.json)
- [Readable reproducible profiler](profile_cpu_preparation.py)
- [Exact executed profiler bytes](profile_cpu_preparation.executed.py.txt)

Run from a checkout with the recorded source hashes and pinned training packages:

```sh
PYTHONPATH=src .venv/bin/python docs/validation/profile_cpu_preparation.py
```

The script hashes the exact dataset, runs CPU-only validation and streaming
production encoding, and writes reports under `/tmp/picoagent-preparation-profile`.
It does not train or alter the dataset. Recorded source hashes identify the core
implementation, and each fresh result records its actual Git revision. Repeated
profiling produces new timings rather than reproducing wall time exactly.

The readable script differs from the executed source only by formatting, import
splitting and a lint annotation. The test normalizes split imports and checks AST
equality. The original report's profiler hash names the preserved executed bytes:
`addf6de43d92e9fd414c97905bd12c0a1b3e75ac5e75743e62a8da418b03a8f0`.
