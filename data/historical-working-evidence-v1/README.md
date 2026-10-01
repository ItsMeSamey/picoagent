# Historical raw-byte preservation

These files preserve historical working copies and rejected/superseded artifacts. They are **not training admission evidence by themselves**. No original was deleted or changed, and no commands stored in a trajectory were executed by this preservation process.

`manifest.json` binds `files.jsonl`, the canonical manifests, and the new bounded archive. The inventory describes 14,767 original files (1,287,267,048 bytes), preserving original path, byte length, SHA256 and permission bits. A representation is a verified canonical file, decompressed gzip stream, tar member, or ordered concatenation of those byte streams. All canonical containers and archive members were read back; source inventory hashes matched before and after packaging.

14,758 files reuse existing canonical evidence. Nine otherwise unrepresented contents are in `unrepresented-00000.tar.gz` (894,719 bytes). They include the rejected unexecuted manual-retention contract and its source, compaction helper/README files, task manifests, and the superseded failed Python seal's original selection index. Archive metadata is deterministic: sorted source traversal, fixed zero uid/gid/mtime, regular members, and gzip mtime zero. Logical archive parts are limited to 16 MiB and physical files to 25 MiB.

`readback-verification.json` records an additional reconstruction check for all nine new contents and both oversized Python raw JSONL files (132,430,771 and 204,936,939 bytes). `preserve_data.py` is the exact packaging source snapshot bound by the preservation manifest. The reusable utility is also in `scripts/preserve_data.py`.

`ignore-verification.json` documents the eight exact versioned directories excluded only after a fresh full-file hash comparison against this inventory. Compaction's top-level pilot observations, tokenizer snapshot, helper scripts, and retention metadata remain directly stageable so artifact regression tests still have their ordinary inputs.

## Existing ignore-rule audit

`../historical-existing-ignore-audit-v1.json` independently rechecks the earlier CLI working-copy rule and four historical pilot archives. The CLI aggregate's exact 103,559,325 bytes are reconstructed by its canonical raw gzip shards. Three late/unrepresented files, totaling 5,769 bytes (including one empty test-packet placeholder), are preserved in `cli-late-audit-supplement.tar.gz`. No nonempty test task contents were inspected for this check. The audit report binds the supplement's hash and each member hash.

## Restore one original file

Run from the repository root, using a new destination. This copies verified bytes without executing them:

```sh
python scripts/preserve_data.py restore-file \
  --manifest data/historical-working-evidence-v1/manifest.json \
  --original data/native-compaction-v1/manual-retention-contracts-v1/attempt-001.json \
  --destination /tmp/restored-retention-attempt.json
```

The destination must not already exist. Canonical component directories listed in the preservation manifest must accompany this archive; they are not external storage dependencies. To inspect the CLI supplement, verify its hash against the existing-ignore audit and extract only the three listed regular members into a fresh directory.

High-confidence private-key, AWS access-key and Hugging Face-token patterns were screened without printing matched content. Name/content checks cannot prove the absence of an arbitrary secret pasted into an ordinary file; these are reviewed synthetic curriculum artifacts, not a place to store credentials.
