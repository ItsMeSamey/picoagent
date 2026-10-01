# Exact CPU-prepared training tokens

`picoagent.training.prepared` separates expensive strict source admission and
assistant-only encoding from a newly identified training run. It does not weaken
the ordinary raw-data path and it cannot approve its own output for production.

## Build and review

Build on CPU from the raw, immutable config and a fresh output directory:

```sh
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  python -m picoagent.training prepare-tokens \
  --config configs/smol360m_native_t4.json \
  --output-dir /tmp/picoagent-prepared-rebuild \
  --tokenizer-path data/native-training-plans-v1/base/components/000/evidence/native_compaction/source/tokenizer
```

The builder invokes the original strict dataset verifier, the original
`event_examples` expansion, and the original `encode_trace`. No model or
accelerator is allocated. It rejects overlong examples, performs no truncation or
packing, and preserves source-row and model-event order. Every token array is
checked for an exact binary round trip. Reviewed action-plan notes are recounted
with the same pinned tokenizer assets.

Before production use, independently rebuild/review the output and compare the
complete ordered `input_ids`, `attention_mask`, and assistant-only `labels` to
the ordinary raw encoding path. Per-split ordered-stream SHA256 digests cover
every example identity and all three arrays, not sampled examples or aggregate
counts alone. Review source-manifest identity, source/encoder hashes, tokenizer
assets and effective settings, example counts, maximum lengths, and both total
and supervised token counts.

Only after that review may an exact prepared-manifest digest be added to
`prepared_approvals.py`. A new training config must also supply both
`prepared_manifest` and the same `prepared_manifest_sha256`. A self-consistent
artifact or user-supplied digest alone cannot enter production training. Smoke
fixtures remain explicitly smoke-only.

Use a new output directory and frozen run config. Do not retrofit a prepared
artifact or changed training source into an existing checkpoint's resume
identity. The original raw manifest and evidence remain required and unchanged.

## Artifact and trust boundary

- The root manifest is canonical JSON with an explicitly pinned SHA256
- Source evidence is bound by its complete file inventory, sizes and hashes
- Every core Python source path and hash is bound, including newly added or
  deleted modules. Only the reviewed approval registry is excluded to avoid a
  digest cycle. The full source map is captured before strict admission and
  checked after admission and encoding; freeze core code for the whole build
- Tokenizer assets are saved locally and inventoried. Effective vocabulary,
  special tokens, backend, padding settings and exact Transformers/Tokenizers
  versions are bound to the artifact
- Each deterministic gzip token shard contains a `PICOTOK1` header followed by
  length-prefixed, little-endian signed int32 arrays. There is no pickle or
  executable serialization
- Each gzip JSONL index binds split, source-row ordinal, trace/task identity,
  exact model-event index, example identity, last-assistant-only supervision,
  lengths, and encoded-array hashes
- Compressed files, decompressed shards and the root manifest have explicit
  byte bounds. Symlinks, paths outside the artifact, duplicate JSON keys,
  unknown inventory entries, bad lengths, out-of-vocabulary tokens, altered
  attention masks, missing supervised tokens and reordered examples fail closed
- Files are created exclusively and made read-only. Hash validation remains
  mandatory because filesystem permissions are not tamper-proof

Production fast-loading verifies the independent approval, config pin, source
evidence bytes, full token artifact, effective tokenizer and transform identity.
It deliberately does not repeat source semantic validation or raw encoding after
that approved admission. Exact packed buffers avoid retaining millions of Python
integer objects; each dataset read returns fresh arrays.

The new training run snapshots both raw evidence and prepared bytes, checking
every copied file against the approved inventory. Its ordinary whole-source and
environment identity still applies to checkpoint resumes, including the approval
registry. CPU preparation excludes Torch/Python from the *encoding transform*
identity so different CPU and CUDA stacks can use the same reviewed token bytes;
the builder Python version is recorded and the full training identity still
records its own Python, Torch, hardware and package environment.

## Regression checks

```sh
python -m pytest tests/test_training_prepared.py \
  tests/test_training_config.py tests/test_training_data.py \
  tests/test_training_encoding.py tests/test_training_resume.py
```

The focused tests include deterministic independent rebuilds, every-array
equivalence, accepted/rejected compaction event gaps, legacy and current-output
masks, immutable dataset reads, source/cache corruption, malformed/rehashed
artifacts, bounded gzip expansion, snapshot relocation, fail-closed production
approval and unchanged ordinary raw admission.

For the full native dataset, the standalone standard-library auditor decodes
both complete artifacts without importing the builder or its decoder. It checks
every payload byte, reconstructs the ordered arrays independently, and compares
their digests to the separately recorded complete raw-encoder profile:

```sh
python scripts/audit_prepared_equivalence.py \
  --candidate data/prepared-native-training-plans-v1/manifest.json \
  --rebuilt data/prepared-native-training-plans-v2/manifest.json \
  --raw-profile docs/validation/20261001-cpu-preparation-profile.json \
  --output /tmp/picoagent-prepared-equivalence-check.json
```

The auditor never changes the approval registry. Its report is evidence for
independent review, not an automatic production admission decision. Candidate v1
is retained as historical equivalence evidence; only a separately approved final
digest may appear in the new production config.

## Qualified Colab revision

Revision `84f32312c35b07f923d558e30122b2ee2554a8a5` includes the approved
v2 artifact and config. On an already authorized runtime, execute
`scripts/colab_public_bootstrap.py` with an explicit
`PICOAGENT_SOURCE_COMMIT=84f32312c35b07f923d558e30122b2ee2554a8a5`.
There is no implicit bootstrap revision. The public checkout and model/data
downloads occur on Colab; no GitHub credential is copied there.

The initial bounded training invocation is:

```sh
PYTHONPATH=/content/picoagent/src TOKENIZERS_PARALLELISM=false \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=20261001 \
  python -u -m picoagent.training train \
  --config configs/smol360m_native_t4_prepared_v2.json \
  --segment-steps 64 --output-budget-bytes 20000000000 \
  --output-budget-root /content
```

This pauses at the first sealed checkpoint or the 64-update cap; it does not
complete or shorten the frozen 3,044-update schedule. Run from
`/content/picoagent` through the detached `colab_run.py start` controller.
Keep the off-runtime SDK collector running with `--download-workers 4`, and
verify its durable receipt before releasing the runtime. Subsequent segments
must resume the exact same identity; do not treat a paused segment as a finished
model. A failed or partial download is not a resumable backup.

The prepared loader was measured locally at 34.33 seconds including imports and
config loading, versus 669.1 seconds of recorded raw preparation. These are CPU
measurements with host/cache differences, not a guaranteed Colab speedup.
