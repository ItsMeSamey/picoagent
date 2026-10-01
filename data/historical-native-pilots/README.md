# Historical native pilot evidence

These deterministic tar.gz archives preserve every original file byte, including
manifest-listed interpreter bytecode from early source imports. They are retained
for audit, not selected training data. Failed/partial pilots are not relabelled as
successes. Early model-identity metadata is superseded by the reviewed procedural
replay attribution in later seals; immutable old observations are unchanged.

The manifest gives the archive hash and each decompressed member hash/size. If
restoring, use an empty directory, reject absolute/traversing paths, links and
special files, then check every member against this manifest. Do not execute
archived bytecode. The production CLI snapshot is `data/native-cli-10k-v1`; the
small distributed integration fixture is `data/native-sharded-pilot-v1`.
