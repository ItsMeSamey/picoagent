# Preserved data

- `curriculum-v2/`: admitted original task specs: 512 train, 64 dev, 64 test. `*.authored.jsonl` are explicitly unexecuted references, not verified training traces
- `curriculum/v1/` and `curriculum-v1/`: preserved early fixtures, excluded by `EXCLUSIONS.json`
- Real execution attempts/exports are created by the container collector, not invented by generation

See [the data protocol](../docs/data_protocol.md) for schemas, split policy, collection, hashes, limitations, and strict training admission. Never combine repeated task IDs from separate curriculum generations or use test specifications for training/checkpoint selection.
