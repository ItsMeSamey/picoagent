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

Only one bounded temporary chunk is materialized at a time. The official `gh`
CLI handles authenticated uploads and streamed read-back. It must already be
installed and signed into an authorized account on the controller. CLI calls
and asset read-back are bounded to 600 seconds; failure does not automatically
retry or change authentication.

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
python -m pytest tests/test_github_checkpoint_release.py -q
python -m ruff check scripts/github_checkpoint_release.py tests/test_github_checkpoint_release.py
```

Tests use tiny local checkpoints, fake GitHub state/HTTP responses and a fake
`gh` executable. They do not publish or demonstrate a real GitHub backup.
