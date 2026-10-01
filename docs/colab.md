# Colab: explicit sessions, recoverable checkpoints, no surprise spend

This controller uses Google's [official Colab CLI](https://github.com/googlecolab/google-colab-cli), pinned to `google-colab-cli==0.7.4`. It operates an **existing explicitly named session**. It never allocates/replaces/stops sessions, opens payment pages, changes authentication, or creates keepalive traffic. Do not use multiple accounts, quota cycling, or unattended provisioning to evade Colab limits.

## Boundaries and current evidence

- Prefer a **free TPU v5e1**. Run the XLA plumbing smoke before any production training. Use a **free T4** only if TPU availability or a recorded compatibility test blocks progress. A fallback is an explicit operator action, not an automatic allocation loop
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

Use a clean reviewed checkout plus the admitted, immutable train/development snapshot under `data/`. Keep every raw attempt and trace, including failures. Keep sealed evaluation/lockbox data separate and never expose it to training. `stage` archives approved source/data roots with a SHA256 manifest, excludes `.git`, virtual environments, hidden credential directories and credential-like names, uploads the archive, then verifies it remotely. Symlinks are rejected. File-name filtering cannot detect a secret pasted into an ordinary source/config file: review before upload.

```sh
python scripts/colab_run.py --session picoagent-tpu stage \
  --archive /persistent/picoagent/source-v1.tar.gz
```

`stage` refuses to overwrite a nonempty project directory. This prevents accidentally mixing revisions. Select a fresh `--project /content/picoagent-v2` for another experiment. The archive includes code, configuration, docs, tests, dependency recipe/locks, and eligible `data/` files; it excludes run checkpoints and private account state. Preserve the archive on the controller and push the reviewed source to the authorized private GitHub repository separately. Do not put multi-GB model checkpoints into ordinary Git.

For TPU, preserve the runtime's matched `torch`/`torch_xla` pair and use `requirements-tpu.txt`. Never independently replace torch in a working TPU image. Use `requirements-cpu.lock.txt` only for the CPU reference environment. Record the exact actual environment in the run manifest. The same package identity is required for strict resume.

## Smoke, train and collect

`start` launches one argv-only subprocess under a detached supervisor on the **selected** runtime, with a log at `/content/picoagent/controller-job.log` and status at `.picoagent-job.json`. A disconnected CLI does not itself cancel this subprocess, but Colab remains free to terminate the runtime. No shell expansion is used. An existing recorded running job prevents a duplicate launch; inspect its PID/log rather than launching a second copy. Kernel execution remains free for collection while training runs in the child process.

```sh
# After dependency setup, first exercise the actual TPU backend.
python scripts/colab_run.py --session picoagent-tpu start \
  --command-json '["python","-m","picoagent.training","smoke","--device","xla","--output-dir","/content/picoagent-xla-smoke"]'

# Only after the smoke succeeds and verified train/dev data is prepared:
python scripts/colab_run.py --session picoagent-tpu start \
  --command-json '["python","-m","picoagent.training","train","--config","configs/smol360m_tpu.json"]'

# Run this on a separate persistent host, not inside Colab.
python scripts/colab_run.py --session picoagent-tpu watch \
  --run-dir /content/picoagent/runs/smol360m-tpu-v1 \
  --destination /persistent/picoagent/runs/smol360m-tpu-v1 \
  --off-runtime --interval 60 --prune
```

The `--off-runtime` flag is the operator's attestation of storage topology, not a discovery mechanism. Set `save_steps` before training and measure how long that interval and transfer take. The TPU preset saves every 25 optimizer steps; a 60-second watcher only finds **completed** checkpoints and cannot force earlier saves. Full 360M training stores FP32 parameters and Adam states: checkpoints are several GB, so bandwidth and staging space may dominate. Use a shorter save interval only after measuring sustainable transfer throughput.

For T4 fallback, deliberately select the authorized T4 session name, run `smoke --device cuda`, then use a reviewed configuration with `device: "cuda"`, appropriate precision and save interval. This is a distinct run. Never silently re-label TPU training as T4 success or claim bitwise continuity across hardware.

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
