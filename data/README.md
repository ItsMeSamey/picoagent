# Preserved data

- `curriculum-v2/`: admitted original task specs: 512 train, 64 dev, 64 test. `*.authored.jsonl` are explicitly unexecuted references, not verified training traces
- `curriculum/v1/` and `curriculum-v1/`: preserved early fixtures, excluded by `EXCLUSIONS.json`
- `tool-docs-v1/`: separate unexecuted installed-help/pydoc/source-reading expansion, 16 tasks per split. Not merged into the initial bootstrap
- Real execution attempts/exports are created by the container collector, not invented by generation

See [the data protocol](../docs/data_protocol.md) for schemas, split policy, collection, hashes, limitations, and strict training admission. Never combine repeated task IDs from separate curriculum generations or use test specifications for training/checkpoint selection.

Reviewed native CPU demonstrations use a separate opt-in evidence category:

- `native-pilot-v2/` and `native-pilot-v3/`: sealed CLI admission pilots (one train and one dev task each), with raw observations, frozen reviewed source, original candidates, and independent oracle audits
- `native-pilot-v1/`: preserved incomplete sealing attempt, no final manifest, never admissible
- `native-source-reviews/`: exact source-review records required by the native admission verifier

These are deterministic replays of Luna-authored programs. They do not claim adaptive Luna sampling, container isolation, or broad learned capabilities. Generic training JSONL admission still rejects them; explicit native manifest verification is required.
