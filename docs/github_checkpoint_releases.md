# Optional GitHub release checkpoint backup

This helper is prepared for **explicitly approved public checkpoint publication**.
Preparing its code or a local plan does not authorize creating a draft, uploading
assets, or publishing. A draft also transmits data to GitHub and requires approval.
Keep all GitHub credentials on the controller; never copy them into Colab.

## Local review

Use an already verified archive and the exact original `transfer_manifest.json`.
The helper verifies full checkpoint contents, each original transfer chunk,
Trainer step, run identity, and source snapshots before creating a plan. It does
not download, repack, allocate an accelerator, or contact GitHub during planning.

```sh
python scripts/github_checkpoint_release.py plan \
  --run-dir /persistent/run-archive \
  --transfer-manifest /persistent/transfer_manifest.json \
  --repository OWNER/REPO \
  --output checkpoint-release-plan.json
```

The output reports the canonical `plan_sha256`. Preserve that printed hash
independently of the plan file. It hashes canonical JSON, not the file's trailing
newline. The plan binds the public repository, full source/run/checkpoint hashes,
original Git commit, deterministic source/run/step tag, exact manifest bytes,
and every immutable asset's name, length and SHA256. The original source snapshot
is authoritative even if the original Git checkout was dirty.

Review the actual transfer manifest before approving the scope. It includes
weights, optimizer/RNG/Trainer state, run metadata and transferred source,
tokenizer and dataset snapshots. Public release assets are publicly downloadable.
Late evaluation sidecars, logs, traces and files outside that manifest are not
included. This script never changes that scope or automatically adds newer files.

## Only after approval

The following command creates or resumes a **draft**. Its exact plan hash must be
approved; supplying a hash is an operator assertion of approval, not a substitute
for obtaining the user's permission.

```sh
python scripts/github_checkpoint_release.py upload \
  --run-dir /persistent/run-archive \
  --transfer-manifest /persistent/transfer_manifest.json \
  --plan checkpoint-release-plan.json \
  --approved-plan-sha256 PRINTED_APPROVED_HASH \
  --receipt checkpoint-draft-verification.json
```

Only when publication is explicitly approved, repeat that command with
`--publish` and a new `--receipt` path. Existing matching assets are independently
read back, not overwritten. Every asset is streamed and hashed, including when
GitHub supplies a matching API SHA256. The transfer manifest is uploaded last,
after all chunks have passed read-back. Publication is confirmed, the source tag
is rechecked, and anonymous access to the exact public manifest is verified
before a published-release receipt is returned. A draft staging report is not a
durable public-backup receipt.

Any release-body, target-commit, asset-name, size, digest, state, or identity
collision fails closed. A failed upload can leave partial draft assets; an
incomplete or conflicting asset requires inspection and separately authorized
remediation. There is no automatic deletion, clobber, force push or cleanup of
checkpoint data. A retry reuses correct assets. Keep one writer per run/release.
The CLI holds the existing run lock; do not remove a stale lock without checking
its process. The helper's immutability rule is not a claim that GitHub prevents
repository owners from later editing or deleting releases.

Uploads and read-back are serial by default. For an explicitly selected bounded
parallel transfer, add `--upload-workers 2` (integer 1–4). Parallel mode requires
all assets to be at most 32 MiB and materializes at most one temporary chunk per
worker (at most 128 MiB total with four workers). It uploads one bounded batch,
waits for every upload to finish before checking the asset inventory, then fully
reads back that batch in parallel. The completion manifest stays serial and last,
after all chunk read-backs succeed. Existing matching assets are read back again.
On failure or interruption, queued work is canceled and active operations are
joined before temporary cleanup and CLI lock release; draining a running CLI
operation may take its remaining timeout. No automatic retry or remote cleanup
is attempted. Do not change worker settings by restarting an active publisher;
use the option only on a separately coordinated later invocation.

The official `gh`
CLI handles authenticated uploads and streamed read-back. It must already be
installed and signed into an authorized account on the controller. CLI calls
and asset read-back are bounded to 600 seconds; failure does not automatically
retry or change authentication.

## Collector concurrency and coordinated stopping

Both `colab_run.py collect/watch` and `colab_sdk_watch.py` accept
`--upload-workers 1..4` (default 1). A value above 1 requires
`--publish-repository OWNER/REPO` and `--approve-public-checkpoints`. The setting
changes only controller-side upload/read-back concurrency; approved plans,
release receipts, and runtime acknowledgement identities stay unchanged.
It does not change training source, download concurrency, or runtime execution.

The two watch entry points also accept `--stop-file /explicit/local/path`.
Create that marker locally when a coordinated controller stop is needed. The
watcher checks it only after its current collection has fully returned, including
all uploads, verified publication, persisted receipts, acknowledgements, and any
requested pruning. It exits normally, releases the controller lock, and emits a
`controller_stopped` record with `collection_completed: true`. It never interrupts
an in-progress upload or sends a stop signal to training. Collection failures
still propagate as failures, rather than reporting a safe stop. An existing
marker still allows one collection to finish before stopping; a marker arriving
during the sleep interval is observed after the next collection. The marker is
not removed automatically, so remove or choose a new marker path before restarting.

This option must have been configured when the watcher started; adding a marker
does not retrofit an already-running watcher that lacks `--stop-file`. Coordinate
existing controllers separately, and never start a second writer while their
lock is held.

## Anonymous pinned restore

No GitHub credential is used or required for a published public release:

```sh
python scripts/github_checkpoint_release.py restore \
  --plan checkpoint-release-plan.json \
  --expected-plan-sha256 INDEPENDENTLY_SAVED_PLAN_HASH \
  --destination /content/restored-run
```

The normal chunk/full-file/checkpoint integrity checks, destination lock, atomic
checkpoint publication, and identity collision rules apply. Restore does not
issue an off-runtime durability receipt. The plan and exact manifest bind all
downloaded bytes; no moving `latest` release or branch is used.

## Limits and local validation

GitHub documents [at most 1,000 assets per release, each under 2 GiB, with no
total-release-size or bandwidth limit](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases#storage-and-bandwidth-quotas).
Both asset caps are enforced before upload. The existing 32 MiB chunks fit.
The helper checks the optional [release-asset API digest](https://docs.github.com/en/rest/releases/assets#get-a-release-asset)
and never uses [`gh release upload --clobber`](https://cli.github.com/manual/gh_release_upload).

```sh
python -m pytest tests/test_github_checkpoint_release.py tests/test_github_checkpoint_parallel.py -q
python -m ruff check scripts/github_checkpoint_release.py tests/test_github_checkpoint*.py
```

Tests use tiny local checkpoints, fake GitHub state/HTTP responses and a fake
`gh` executable. They do not publish or demonstrate a real GitHub backup.

Hard execution-tool or host termination can bypass Python `finally` blocks.
Worker-draining guarantees describe cooperative Python exceptions, not a killed
process tree. Prefer the watcher's `--stop-file` for later planned controller
changes. After hard termination, verify the owned execution session is terminal,
all detached children have stopped, and the remote inventory is complete and
unchanged before treating its local lock as stale. Never clear a live lock or
start a second writer. Interrupted assets remain preserved and fail closed.
