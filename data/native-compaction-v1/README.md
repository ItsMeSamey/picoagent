# Native compaction observations

These are real, reviewed read-only `cat` subprocess observations in the platform's
native CPU environment. They are not Docker/Podman parity evidence or learned
model performance results. Raw observations remain `sft_admissible=false`; only a
separately reviewed native admission snapshot can be used for training.

## Completed observations

- The root pilot has 6 train/dev base tasks and 18 full/half/manual mode variants
- `scale-216-v1` has 216 base problems (192 train, 24 dev), 648 successful observed
  mode variants, 10,368 actual subprocess receipts, and 10,152 accepted compactions
- The scale source's 3,456 manual decisions all retain an empty set. They supervise
  summarization and empty-selection syntax, not meaningful group-retention variety
- `manual-retention-v2` adds 16 context-retention variants of existing scale bases,
  with mixed-length records and a tokenizer-aware reference selector. It does not
  add 16 independent factual problems. `derivation.json` binds each context variant
  to its original base task and matches the goal/fact hashes
- These 16 variants all succeeded: 256 real subprocesses and 176 actual manual
  decisions, including 80 nonempty keep-and-drop decisions and 96 empty selections
- Nonempty selections keep the latest single atomic tool group (`keep_groups=[2]`)
  and add 590–614 rendered tokens. This narrow contiguous policy is not evidence
  of arbitrary selection competence. Maximum request/target/complete-event lengths
  are 2,826 / 619 / 3,445 tokens under the pinned 4K/768-token configuration

There are no executed test/lockbox families here. Counting the scale and retention
extension together still yields 216 distinct underlying base problems, with
additional context/mode supervision. Future independently new task instances
should use disjoint seeds instead of reusing the scale's seeds.

## Evidence and rejected attempts

Each actual batch freezes source files, tokenizer files, tool schemas and runtime
identities before execution. Every subprocess retains stdin/stdout/stderr bytes
and hashes, argv, environment, exit status, duration, and before/after fixture
hashes. Callback requests/responses, exact context transitions and complete
harness events are retained. Source helpers admit only exact reviewed linked-file
reads; arbitrary learner commands cannot execute through this producer.

Observation/task/candidate gzip shard paths and hashes are in each batch's
`manifest.json`. Auxiliary archives retain every per-attempt journal, raw snapshot
and workspace file, with read-back-verified member hashes. Loose originals are
also preserved locally. Integrity reports are audit records, not execution
attestations or admission capabilities.

`manual-retention-contracts-v1` preserves an explicitly **unexecuted unit fixture**
that exceeded the request budget, plus its original source. That failed contract
led to the shorter versioned mixed-record specification. It is not a native
execution artifact and must never enter training.

The prior pilots, rejected contracts, complete scale, and executed retention
variants have not been rewritten or relabelled. Separate sidecars clarify their
relationships and limitations without changing the original evidence.
