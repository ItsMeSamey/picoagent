# Reproducibility and contamination boundary

## Claims

Pretrained checkpoints may already contain programming languages, public
benchmarks, documentation, or near duplicates. Fine-tuning cannot erase that
history. We can audit our added data; we cannot certify unknown base pretraining.
An experiment starting from random initialization must be labeled separately.

Do not train on public benchmark prompts, solutions, traces, scoring feedback,
or paraphrases. Do not use official benchmark errors to generate new training
tasks. Use original procedural training families and a separate development
set. Keep a final sealed evaluation set out of training and model selection.
Repeatedly choosing checkpoints on a benchmark overfits the benchmark even if
its literal questions never enter an SFT dataset.

## Required run artifacts

- Repository commit, dirty-diff status, dependency lock/freeze and Python version
- Exact base model and tokenizer revisions and license
- Dataset manifest and SHA-256 hashes of every file and raw trajectory
- Generator revision, task-family split, seeds, all failed/successful attempts
- Actual sandbox image digest, tool schemas, limits and software versions
- Hardware, dtype, optimizer, hyperparameters, batch size, random seeds
- Step/loss logs, checkpoint hashes, durable storage locations, evaluation outputs
- Explicit distinction between authored examples, executed oracle demonstrations,
  model-generated attempts, and held-out evaluation

Do not overwrite a failed attempt with a repaired success. Record the repair as
a separate linked trace. Commit generated JSONL or preserve sharded data with
hashes and access-tested durable URIs; a temporary local path is not preservation.

## What repeatability means

Pin and publish the recipe. Bitwise equality can fail across GPU architectures,
kernels and libraries even with fixed seeds. Report that distinction and run
multiple seeds before claiming robust improvement. CPU smoke tests only verify
plumbing; they are not model-quality results.

## Late documentation-only language test

After instruction-following and tool-use training, freeze the checkpoint and
test on newly constructed small language/API specifications with fresh syntax
and randomized semantics. Provide the specification and interpreter errors,
not training examples. C, JavaScript and HTML are useful practical tests but
cannot honestly be called unseen for a web/code-pretrained model.
