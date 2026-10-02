# Colab: explicit sessions, recoverable checkpoints, no surprise spend

This controller uses Google's [official Colab CLI](https://github.com/googlecolab/google-colab-cli), pinned to `google-colab-cli==0.7.4`. It operates an **existing explicitly named session**. It never allocates/replaces/stops sessions, opens payment pages, changes authentication, or creates keepalive traffic. Do not use multiple accounts, quota cycling, or unattended provisioning to evade Colab limits.

## Boundaries and current evidence

- Current approved fresh run prefers a **free T4** with the pinned prepared-data full-finetune config. Private free Kaggle is the approved fallback if Colab is unavailable. Do not allocate paid accelerators or touch another task’s existing session. Earlier TPU-first notes describe prior attempts, not this run’s preference.
- No purchase, subscription, compute-unit consumption that would charge the user, or paid upgrade is authorized by these scripts. A hardware request alone does not prove an account allocation is free; check its billing/usage information before allocation
- `configs/smol360m_tpu.json` requires XLA and refuses silent CPU fallback. `configs/smol360m_full.json` supports auto backend selection; confirm CUDA/T4 when using it as the fallback
- CPU smoke and fake-transport tests demonstrate plumbing. They do not demonstrate TPU/GPU success, model quality, or benchmark results. See the run's actual manifests/status for hardware validation
- A directory on the Colab VM, a different directory under `/content`, and the exported chunks beside it **are not durable backups**. The controller must run on a separate machine, writing to independently persistent storage. A transient controller VM can also disappear; local fsync cannot prove physical durability
- A genuine disconnect/termination can lose work since the last completed, verified off-runtime checkpoint. There is no zero-loss guarantee

Installation/authentication and session allocation are separate operator steps. Do not copy OAuth client files, session state, API keys, or tokens into the repository/archive. Do not commit runtime `.config` directories. All commands below use an explicit session to avoid accidentally operating on unrelated work.

```sh
./scripts/install_colab.sh
colab status --session picoagent-tpu
```

If no authorized session exists, stop here. The operator can inspect `colab new --help` and request the free v5e1 option after confirming account cost/availability; a quota denial is not a reason to retry with another identity. The `colab run` convenience command automatically provisions/tears down resources, so it is intentionally not used.

## Prepare an auditable code/data archive

Use a clean reviewed checkout plus the admitted, immutable train/development snapshot under `data/`. Keep every raw attempt and trace, including failures, on persistent storage. Keep sealed evaluation/lockbox data separate and never expose it to training. Prefer `--dataset-manifest` to archive code plus exactly one selected snapshot, without historical snapshots or raw working copies. Selection does not delete or change the originals. File-name filtering cannot detect a secret pasted into an ordinary source/config file: review before upload.

```sh
python scripts/colab_run.py --session picoagent-tpu stage \
  --archive /persistent/picoagent/source-v1.tar.gz \
  --dataset-manifest data/selected-train-dev/manifest.json
```

Replace the illustrative manifest path with the actual reviewed dataset. It must reside under the source checkout's `data/` directory and contain a nonempty `files` map with SHA256 and byte lengths for **every self-contained dependency**, including nested snapshots. Required files are included even if git-ignored. Missing or changed files, path traversal, symlinks (including ancestors), and required files filtered as credential-like cause failure before upload. The existing public tokenizer/metadata filename exceptions remain allowed. A transport hash check does not replace the training CLI's dataset/provenance verification.

For compatibility, omitting `--dataset-manifest` retains the previous archive scope: code, configuration, docs, tests, dependency recipes/locks and all eligible non-ignored `data/` files. This can be much larger and must not contain lockbox data. Both modes exclude private account state and run checkpoints. Keep the archive outside approved source/data roots; the persistent path above is recommended.

Uploads now use content-addressed **32 MiB chunks** by default (`--chunk-bytes` can change this, up to 256 MiB). The CLI never receives the whole tarball. Local chunks remain beside the archive in `<archive>.chunks/`; remote chunks use `--transfer-root /content/picoagent-source-uploads`, under the archive hash. Each uploaded chunk is independently checked on the runtime. Repeat the identical command with the same frozen source after a disconnect: valid chunks, including complete uploads whose acknowledgment was lost, are reused. Corrupt/incomplete chunks are retried. Changed source produces a new archive identity.

The runtime reconstructs and hashes the archive with bounded reads, extracts into a fresh temporary directory, verifies the exact file tree and hashes, then atomically publishes the project. Unsafe/duplicate tar entries and insufficient space fail closed. An exact completed retry is accepted after re-verification; an existing project with changed files or run outputs is never overwritten. Select a fresh `--project /content/picoagent-v2` for another experiment. Reserve space for local archive plus chunks, and remote chunks plus reconstructed archive plus extracted files; no automatic cleanup deletes originals or checkpoints. Preserve the archive on the controller and push the reviewed source to the authorized private GitHub repository separately. Diagnostics report exception types/locations without raw CLI output or credential-bearing exception text. Do not put multi-GB model checkpoints into ordinary Git.

For TPU, preserve the runtime's matched `torch`/`torch_xla` pair and use `requirements-tpu.txt`. Never independently replace torch in a working TPU image. Use `requirements-cpu.lock.txt` only for the CPU reference environment. Record the exact actual environment in the run manifest. The same package identity is required for strict resume.

## Smoke, train and collect

The user limits any optional accelerator smoke to **60 seconds total**, including compilation. Prefer skipping it after CPU plumbing qualification. The unbounded smoke CLI examples in generic development documentation must not be run on an accelerator without an external whole-process deadline. Production training is distinct from a smoke test; never disguise a long smoke as production.

`start` launches one argv-only subprocess under a detached supervisor on the **selected** runtime, with a log at `/content/picoagent/controller-job.log` and status at `.picoagent-job.json`. A disconnected CLI does not itself cancel this subprocess, but Colab remains free to terminate the runtime. No shell expansion is used. An existing recorded running job prevents a duplicate launch; inspect its PID/log rather than launching a second copy. Kernel execution remains free for collection while training runs in the child process.

```sh
# Skip optional accelerator smoke by default. Complete CPU preflight first.
# Start production only after verified train/dev data and durable transfer are ready:
python scripts/colab_run.py --session picoagent-tpu start \
  --command-json '["python","-m","picoagent.training","train","--config","configs/smol360m_tpu.json"]'

# Run this on a separate persistent host, not inside Colab.
python scripts/colab_run.py --session picoagent-tpu watch \
  --run-dir /content/picoagent/runs/smol360m-tpu-v1 \
  --destination /persistent/picoagent/runs/smol360m-tpu-v1 \
  --off-runtime --interval 60 --prune
```

The `--off-runtime` flag is the operator's attestation of storage topology, not a discovery mechanism. Set `save_steps` before training and measure how long that interval and transfer take. The TPU preset saves every 10 optimizer steps; a 60-second watcher only finds **completed** checkpoints and cannot force earlier saves. Full 360M training stores FP32 parameters and Adam states: checkpoints are several GB, so bandwidth and staging space may dominate. Use a shorter save interval only after measuring sustainable transfer throughput.

For T4 fallback, deliberately select the authorized T4 session name and use a reviewed configuration with `device: "cuda"`, appropriate precision and save interval. This is a distinct run. Never silently re-label TPU training as T4 success or claim bitwise continuity across hardware.

`collect` performs one collection; `watch` repeats until the run reports completed/failed. Both exit visibly on a transfer/connection failure, leave resumable chunk state, and never allocate a replacement session. Restart the same command after checking the existing session. If a long command's response is lost, inspect the remote job status/log before retrying the launch: the job may have started even when the CLI reported an error. The controller's subprocess and Colab execution timeouts are explicit rather than the CLI's default 30 seconds. On interruption/timeout it terminates and reaps only its own local CLI process group, preventing abandoned reconnect threads from contending with a later command. CLI 0.7.4 can skip client cleanup when the initial connection fails; do not leave a traceback-producing CLI process running in another terminal.

```sh
python scripts/colab_run.py --session picoagent-tpu status \
  --run-dir /content/picoagent/runs/smol360m-tpu-v1
```

## What a verified checkpoint means

1. Hugging Face saves full model/adapter weights, optimizer, scheduler, Trainer state and RNG state. The training callback publishes `checkpoint_manifest.json` atomically after all saved files are hashed
2. `checkpoint_sync.py pack` verifies the seal and produces bounded 32 MiB content-addressed chunks. The transfer manifest is published last. CLI 0.7.4 base64-loads whole files, so sending a multi-GB optimizer file directly can exhaust memory; chunking avoids that transport peak
3. The host downloads into `.incoming/`, retaining only verified chunks for retry. It verifies chunk hashes, reconstructed file hashes, the exact checkpoint tree, and the original run-manifest binding
4. It fsyncs files/directories, atomically renames the completed checkpoint, verifies again, then writes `receipts/checkpoint-N.json`. A partial transfer or wrong hash produces no new durable receipt and cannot replace a committed checkpoint
5. Optional `--prune` sends acknowledgements only for checkpoints just verified on the host. The runtime rechecks each exact manifest before removing older checkpoint directories. It keeps the latest two plus the lowest-development-loss checkpoint. Unsynced, changed, incomplete, newly created, and trace-containing directories are not deleted

**Durable checkpoint history is preserved in this first pass.** Pruning frees Colab runtime copies and their export chunks; it does not prune the persistent archive. Traces, datasets, source archives and receipts are never pruning targets. Keep traces outside checkpoint directories. Final model export, current log/status, and arbitrary files outside the checkpoint/run snapshots are not automatically collected by this checkpoint transfer; archive those explicitly before releasing the runtime. The final model-only export is not a resumable training checkpoint.

Synchronizers are single-writer per run. The controller and in-process mounted-storage helper share `.checkpoint-retention.lock`; a crashed process can leave a stale lock. Inspect whether its recorded PID is still running before manually removing the stale lock. Do not run both synchronization modes concurrently. Hidden incomplete staging directories are never considered backups. Verified partial chunks can be reused on restart.

## Restore after runtime loss

Keep the code/data archive, original configuration, package environment and full checkpoint together. Allocate/reconnect a permitted session deliberately, stage the original source revision, install the same environment, and restore the latest verified checkpoint. Check whether the original run had already completed: a restore operation does not authorize treating a completed run as an unfinished one. Do not change max steps or hyperparameters under the same run identity.

```sh
python scripts/colab_run.py --session picoagent-tpu restore \
  --source /persistent/picoagent/runs/smol360m-tpu-v1 \
  --checkpoint checkpoint-25 \
  --run-dir /content/picoagent/runs/smol360m-tpu-v1 \
  --local-export /persistent/picoagent/restore-exports \
  --export-root /content/picoagent-restore-attempt-1

python scripts/colab_run.py --session picoagent-tpu start \
  --command-json '["python","-m","picoagent.training","train","--config","configs/smol360m_tpu.json","--resume","runs/smol360m-tpu-v1/checkpoint-25"]'
```

Restore uploads bounded chunks, reads them back, then publishes/verifies the transfer manifest. It materializes the checkpoint atomically in the new runtime and **does not issue a durable-backup receipt for that runtime copy**. A completed remote export prefix is not overwritten; choose a fresh prefix if intentionally restarting the transfer. The training CLI revalidates original source/data/config/package/backend identity and optimizer/RNG integrity. Inspect/quarantine incomplete later checkpoint directories rather than overwriting them. Hardware changes or incompatible optimizer/XLA state may require a separately documented continuation experiment, not an exact resume claim.

## Other storage CLIs

`checkpoint_sync.py` accepts trusted JSON argv templates with whole-argument `{local}` and `{remote}` placeholders. Shell interpolation is never used. Configure authenticated storage beforehand; the script never reads or injects credentials. The storage backend must support reliable read-after-write reads, bounded chunk objects and manifest-last publication. Use a unique destination prefix for each immutable bundle.

```sh
python scripts/checkpoint_sync.py pack --run-dir runs/example \
  --checkpoint checkpoint-25 --export-root /tmp/picoagent-exports

# Filesystem transfer into an independently persistent, authorized destination:
python scripts/checkpoint_sync.py pull \
  --remote /mounted/export/checkpoint-25 \
  --destination /persistent/example --off-runtime

# Example of exact argv configuration for downloading an existing Colab export:
python scripts/checkpoint_sync.py pull \
  --remote content/picoagent-checkpoint-exports/checkpoint-25 \
  --destination /persistent/example --off-runtime \
  --download-command-json '["colab","download","--session","picoagent-tpu","{remote}","{local}"]'
```

`push` takes both `--upload-command-json` and `--download-command-json` and requires read-back checksum verification before publishing its completion manifest. A failed upload can leave orphan chunks; it must not be reported as a completed durable backup. Do not pass commands from downloaded manifests or external training traces.

## Tests

```sh
python -m pytest tests/test_checkpoint_sync.py tests/test_training_retention.py -q
```

Fake files/processes cover interrupted upload/download, restart reuse of verified chunks, checksum corruption, manifest path traversal, conflicting run identity, sealed resumable states, retained best/latest checkpoints, source-secret exclusions, and explicit session targeting. These tests do not contact Colab or validate actual storage durability.

For bounded storage on the off-runtime controller, see [reviewed archive retention](checkpoint_workflow.md). Runtime-only pruning does not bound the controller archive.

## Public-release durability barrier

For a trainer configured with the public-backup durability barrier, keep an
independent controller running with both publication flags:

```bash
PYTHONPATH=src:scripts python scripts/colab_sdk_watch.py \
  --session EXISTING_SESSION --run-dir /content/picoagent/runs/RUN \
  --destination /durable/controller/RUN --off-runtime \
  --publish-repository OWNER/REPO --approve-public-checkpoints
```

`colab_run.py collect` and `colab_run.py watch` support the same flags. The
approval flag authorizes public disclosure of the complete verified checkpoint
(model, optimizer, scheduler, RNG and trainer state), pinned run metadata and
source snapshot to the named repository. Review this scope before enabling it;
GitHub authentication belongs on the controller, never on the training runtime.
Without publication flags the existing off-runtime collection behavior remains
unchanged and no trainer durability acknowledgement is created.

The controller first completes the normal SHA256-verified download. It plans
publication against the exact retained transfer manifest under
`.incoming/checkpoint-N/BUNDLE_SHA256/transfer_manifest.json`, rather than
repacking a potentially different archive. It saves `plan.json` and then the
verified published `receipt.json` under
`release-backups/checkpoint-N/BUNDLE_SHA256/` on the controller before atomically
writing `RUN/durability/checkpoint-N.json` on the runtime. Acknowledgement is
independent of `--prune`; this example does not request pruning.

Upload, public readback, identity or persistence failures stop collection
without acknowledging that checkpoint. Restart the same command to reuse exact
verified chunks and matching release assets. Published releases are reverified
before acknowledgement; a saved receipt alone never skips verification. Within
one running controller, a confirmed acknowledgement can skip repeated public
readback only while the exact plan, preserved receipt and current remote
acknowledgement hash still match. A controller restart or missing acknowledgement
forces release re-verification. No
release, tag or asset is clobbered or deleted. Keep the controller destination
on persistent storage and retain its plan and receipt for restoration. The SDK
watcher also performs a final collection after seeing terminal trainer status.

Enable `checkpoint_before_eval: true` in the training configuration and pass a
positive `--durability-timeout-seconds` when starting a new run. Pass that flag
again on every resume; the requirement is recorded in the immutable run
manifest. Choose enough time for download, upload and complete readback. A
missing or invalid acknowledgement halts progress; timeout leaves the sealed
checkpoint available for recovery. The acknowledgement is integrity evidence
from a trusted controller, not cryptographic authentication: protect both run
and controller directories against untrusted writers.

## Fresh full durable v3 run

`configs/smol360m_native_t4_prepared_v3_durable.json` preserves the v2 model,
admitted data, optimizer settings and full one-epoch schedule (3,044 optimizer
updates for 48,702 examples at accumulation 16), with a fresh output directory.
Use `--durability-timeout-seconds` and the approved public-release watcher. A
durability-enabled new run saves at optimizer step 1 to prove its real full-size
backup before continuing; later saves retain the 100-step/600-second cadence.
The GPU smoke-only limit remains 60 seconds; setup, full training, checkpoint
transfer and evaluation may take longer. The first-step pause does not shorten
the training schedule. Loss is a diagnostic; agent capability requires the
separate full/half/manual tool evaluation.

### Bounded controller checkpoint cache (explicit opt-in)

For the full-run watcher, add `--prune-local-published-cache` alongside
`--publish-repository OWNER/REPO --approve-public-checkpoints`. This separately
approves removal of older local, fully verified public-release checkpoint
payloads. It does not delete or change any GitHub release. Keep the collector's
exclusive destination lock; do not run simultaneous collectors against it.

Each newer checkpoint must finish local integrity verification, public release
publication/readback, saved plan/receipt persistence, and runtime acknowledgement
before older payloads can be evicted. The latest local checkpoint is retained.
Only conventional model/optimizer/scheduler/RNG/scaler payload filenames are
eligible. JSON metadata, all manifests, source/data snapshots, release plans and
receipts, evaluations, and incoming partial transfers are preserved. Unexpected
entries, traces, symlinks or integrity mismatches fail closed before eviction.
A small per-release eviction record preserves the exact plan/receipt hashes and
payload list, including if local unlink is interrupted.

Previously published older exports are not downloaded again when their exact
runtime acknowledgement digest matches the saved release evidence. Within one
controller process, only already verified receipt hashes are reused. After a
restart, every release asset is read back and hashed, its public identity and
asset IDs are rechecked, and the manifest is fetched anonymously before skipping
an evicted export. Evaluation sidecars are still collected on subsequent polls.
Do not remove the saved evidence: it is required for safe deduplication/recovery.

Peak disk usage still includes the previous full checkpoint plus the new
checkpoint's download chunks and atomic reconstruction (roughly three checkpoint
sizes plus source/metadata and headroom). This flag does not remove unique or
incomplete partials to force a download to fit; insufficient space stops safely.
If eviction is interrupted after its durable intent is saved, a later collection
reports the incomplete eviction and retains remaining bytes for recovery rather
than silently declaring cleanup successful or deleting unverified leftovers.
