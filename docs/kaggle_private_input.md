# Kaggle private-input package mode

`scripts/kaggle_job.py` only builds local folders. It does not call Kaggle, upload
files, create/version datasets, submit kernels, or inspect account credentials.
Package construction verifies data structure and hashes locally; private
visibility cannot be verified until an authorized provider-side check is done.

## Package layout and guarantees

For production, select one self-contained native-teacher dataset manifest under
`data/` and use `--input-dataset owner/slug` plus `--data-output`:

- The builder calls `colab_run.archive_source(root, archive, dataset_manifest)`.
  The project archive contains the allowed code files and the selected manifest
  with only its hash-listed dependencies. Secret-filtered paths, symlinks,
  missing dependencies, and hash mismatches fail closed
- `source_staging.pack_archive` streams the archive into bounded chunks. The
  default chunk is 16 MiB; the builder refuses chunk sizes over 32 MiB. The
  separate input folder contains `transfer_manifest.json`, individual chunk
  files, `dataset-metadata.json`, and a local package receipt
- The manifest preserves the `picoagent.source-transfer.v1` bundle identity,
  ordered chunk hashes/byte counts, source-manifest hash, and extracted-byte
  count. The exact transfer-manifest hash and reviewed staging-helper hash are
  pinned in the generated `main.py` and kernel package receipt
- The generated kernel refers to the exact `owner/slug` in `dataset_sources`.
  It compares the mounted manifest with its pinned identity, checks each chunk,
  reconstructs the archive by streaming, and uses the shared safe extractor to
  verify the source manifest and every extracted file before training
- Production package mode requires a native-SFT config that opts into
  `native_teacher_observed`, matches the selected manifest, and excludes test
  and smoke data. Native-teacher manifests must have exactly `splits.train` and
  `splits.dev`; artificial-action-plan manifests must have exactly positive
  `counts.train` and `counts.dev`. Missing or additional split keys are rejected.
  The kernel runs only `picoagent.training train --config …`; it does not execute
  learner tool calls
- `--dataset-license` is required and offers `unknown`, `copyright-authors`, or
  `other` (the last also requires a description). No CC0 license or public
  visibility is chosen implicitly
- Kernel privacy is emitted as `is_private: true`. The dataset metadata format
  has no locally verifiable access status, so package receipts explicitly say
  visibility is unverified

Legacy inline mode is only for tiny fixture smoke tests. The archive is capped
at 8 MiB and rejected above that limit; larger source/data must use the separate
private-input dataset package. The optional GPU smoke is fixture-only and its
entire install, hardware-check, and smoke subprocess has a hard maximum of 60
seconds. Production SFT skips that smoke mode.

## Local package workflow (no provider actions)

After preparing an approved native training config whose `dataset_manifest`
matches the selected snapshot, build both local packages:

```sh
.venv/bin/python scripts/kaggle_job.py \
  --root "$PWD" \
  --owner KAGGLE_OWNER \
  --slug picoagent-native-sft \
  --output /tmp/picoagent-kernel-package \
  --config configs/smol360m_native_t4.json \
  --input-dataset KAGGLE_OWNER/picoagent-native-input \
  --data-output /tmp/picoagent-input-dataset \
  --dataset-manifest data/native-training-plans-v1/manifest.json \
  --dataset-license copyright-authors \
  --chunk-bytes 16777216
```

Inspect `kernel-package-receipt.json`, `package-receipt.json`, the manifest
hashes, chosen license, selected data manifest, and chunk count before any
provider action. Rebuilds intentionally refuse existing output directories.

If separately authorized to publish the input package, the current Kaggle CLI
documents `kaggle datasets create` as private by default; it makes a dataset
public only with `--public`. Create the dataset without that flag and preserve
tabular/binary files:

```sh
kaggle datasets create -p /tmp/picoagent-input-dataset --keep-tabular --dir-mode skip
```

Then independently verify the created dataset's visibility and completed
status in Kaggle before pushing the private kernel package:

```sh
kaggle datasets status KAGGLE_OWNER/picoagent-native-input
kaggle kernels push -p /tmp/picoagent-kernel-package
```

These provider commands are documentation only and were not run while preparing
this change. See the [official Kaggle CLI dataset command reference](https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/datasets.md)
for private-by-default creation and the [official dataset metadata reference](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets_metadata.md)
for supported licenses, including `unknown`, `copyright-authors`, and `other`.

## Long-run checkpoint readiness blocker

The local package is **not launch-ready for a long training run**. Its generated
kernel waits synchronously for `picoagent.training train` to finish, and only
then writes `job_status.json`. It does not stream or copy verified checkpoints
off the Kaggle runtime, restart training, or collect checkpoints during a run.
The training configuration saves at 100 steps and at approximately 600-second
intervals; Hugging Face `TrainingArguments` currently sets
`save_total_limit=None`, so every full-state checkpoint is retained. Do not
work around the space problem by deleting unsynchronized checkpoints.

This is especially risky for the current full-precision 360M-parameter run.
Model weights alone are approximately 1.44 GB (360 million FP32 parameters),
and Adam's two FP32 moment tensors add approximately 2.88 GB before scheduler,
RNG, metadata, and serialization overhead. Thus a full resumable checkpoint is
roughly 4.3 GB as an estimate, not a measured Kaggle footprint. Four such
checkpoints would already be about 17 GB; a fifth is about 21.6 GB, before the
final model export or other saved outputs. The 100-step and 10-minute triggers
can produce many saves if the run proceeds quickly. Measure actual checkpoint
size and save duration on an explicitly authorized pilot before estimating the
long run's capacity.

Kaggle's current [Notebook documentation](https://www.kaggle.com/docs/notebooks)
states that up to 20 GB in `/kaggle/working` is auto-saved output, that scratch
space outside it is not saved beyond the current session, and lists a 12-hour
CPU/GPU notebook-session execution limit (9 hours for TPU) at time of writing.
The [official Kaggle CLI kernel reference](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)
documents `kernels output` as retrieving output from the latest run and
`kernels status` as reporting whether it is running, completed, or failed. It
does not document a guarantee that checkpoint files are downloadable while a
run is active. Therefore assume that a running kernel may be interrupted before
its checkpoint output can be retrieved; successful final output retrieval is
not a recovery plan.

Before a long run, one of these recovery paths needs separate design and
verification:

1. A short, approved provider pilot must measure actual save size, free-space
   behavior, session limit, output visibility after success and failure, and
   whether any supported path can collect a completed checkpoint before the
   kernel exits. Do not infer these facts from local packaging tests.
2. The production runner must gain a bounded checkpoint handoff: either a
   supported off-runtime transfer that verifies the complete checkpoint and
   run-manifest hashes before the next save, or deliberately segmented kernel
   runs that end with a verified resumable checkpoint and resume only from a
   separately retrieved and verified artifact. This needs an end-to-end resume
   test and explicit storage/retention policy. The current builder implements
   neither path.

Until that work is complete, treat this package as a verified input/code
staging prototype only. Do not launch a long training job, claim a checkpoint
is durable, or claim it can recover from a Kaggle timeout, runtime loss, output
quota, or failed kernel. No Kaggle provider action or checkpoint pilot has been
performed as part of this local review.
