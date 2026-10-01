# Preserved data

- `curriculum-v2/`: admitted original task specs: 512 train, 64 dev, 64 test. `*.authored.jsonl` are explicitly unexecuted references, not verified training traces
- `curriculum/v1/` and `curriculum-v1/`: preserved early fixtures, excluded by `EXCLUSIONS.json`
- `tool-docs-v1/`: separate unexecuted installed-help/pydoc/source-reading expansion, 16 tasks per split. Not merged into the initial bootstrap
- Real execution attempts/exports are created by the container collector, not invented by generation

See [the data protocol](../docs/data_protocol.md) for schemas, split policy, collection, hashes, limitations, and strict training admission. Never combine repeated task IDs from separate curriculum generations or use test specifications for training/checkpoint selection.

Reviewed native CPU demonstrations use a separate opt-in evidence category:

- `native-training-collection-v1/`: final unaugmented collection, 20,326 train and 221 development tasks. It preserves the complete baseline and knowledge/search component trees byte-for-byte, with a flat outer hash inventory
- `native-training-v1/`: five-source baseline, 20,198 train and 189 development tasks; retention variants replace matching underlying compaction problems
- `native-python-10k-v2/`: corrected 9,997 train and 128 development unique conversations; all 10,128 actual raw observations remain preserved
- `native-knowledge-search-160-v1/`: 128 train and 32 development tasks, with 312 actual in-process knowledge/fixture-search receipts and complete state/file evidence
- `native-pilot-v2/` and `native-pilot-v3/`: sealed CLI admission pilots (one train and one dev task each), with raw observations, frozen reviewed source, original candidates, and independent oracle audits
- `native-pilot-v1/`: preserved incomplete sealing attempt, no final manifest, never admissible
- `native-source-reviews/`: exact source-review records required by the native admission verifier
- `native-cli-10k-v1/`: sealed 10,008 train and 36 development tasks; all observed raw byte shards and bounded evidence archives are retained
- `native-python-pilot-v3/`: sealed 16 train and 8 development tasks from the corrected Python/documentation/SVG source
- `native-compaction-pilot-v1/`: sealed 3 train and 3 development base tasks, with all 18 actual mode variants preserved
- `native-compaction-216-v1/`: sealed 192 train and 24 development base tasks, with all 648 mode variants preserved; its manual selections are all empty
- `native-retention-16-v2/`: 16 reviewed context variants of existing compaction bases, with 80 actual nonempty keep/drop decisions. In combined data these replace the matching original preferred views, adding zero independent problems
- `native-recovery-pilot-v5/`: sealed one train and one development recovery task with genuine initial errors, repairs, and artifact byte checks
- `native-combined-small-v1/`: plumbing-only combined snapshot; it reuses pilot tasks and must not be mixed with their standalone snapshots

These are reviewed procedural replays (`teacher_model=null` in new seals). They do not claim adaptive model sampling, container isolation, or broad learned capabilities. Original authorship evidence and unexecuted candidates remain separately preserved. Generic training JSONL admission still rejects native records; explicit native manifest verification is required.

`native-python-10k-v1/` is a preserved failed admission attempt: three identical
training conversations triggered the unchanged duplicate guard. All 10,128 raw
observations are retained in the corrected seal; only 9,997 train and 128 dev
unique conversations are eligible. Cross-split overlap remains a fatal error.

The final unaugmented collection supplies 49,905 exact per-decision SFT examples:
53,765,506 total tokens and 6,067,391 supervised tokens, maximum 3,445 at the frozen
4,096-token configuration. These are production-encoder counts from unchanged
components, not learned-policy evaluation scores. Reports are in
`native-source-reviews/`; the artificial-action-plan view is managed separately.
