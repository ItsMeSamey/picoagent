# Bounded controller archive retention

Full-state checkpoints consume several GB each; keeping every mirrored save is
not a bounded storage policy. `scripts/durable_retention.py plan` produces an
exact, read-only deletion plan for a single run archive, retaining the latest two
plus the best finite development loss. It requires complete checkpoint integrity,
matching steps, and hash-bound off-runtime transfer receipts for every checkpoint.

```sh
python scripts/durable_retention.py plan --root /durable/picoagent/run-v1 \
  --output /durable/picoagent/run-v1-retention-plan.json
```

Review and authorize the exact listed old generated checkpoints before applying:

```sh
python scripts/durable_retention.py apply \
  --plan /durable/picoagent/run-v1-retention-plan.json \
  --confirm-delete-old-checkpoints
```

This deletion is permanent. It does not remove datasets, source, trajectories,
receipts or audit logs. It records the complete approved plan and each removal.
A changed archive invalidates the plan before deletion. Pause the transfer watcher
before applying a plan; both operations use the same exclusive archive lock.
The planning tool is not automatic authorization or an automatic pruning daemon.
No real training checkpoints have been deleted with this tool yet.

Export and download also preflight free space, including simultaneous downloaded
chunks and atomic reconstruction, plus a 256MiB margin. That check can stop a
transfer; it cannot replace a confirmed retention policy or protect against other
processes concurrently filling the disk. Establish storage/retention before a
long accelerator run, rather than waiting for storage exhaustion.

## Independent evaluation and multi-checkpoint segments (new runs)

New experiments can explicitly freeze `checkpoint_before_eval: true` and an
independent positive `eval_steps` in their training config. The defaults remain
`false` and `null`: legacy runs still evaluate on the save cadence and timed
saves request evaluation. Do not alter an existing run's config or migrate its
identity to opt in; preserve it and create a separately named new experiment.

With the new policy, a timed durability save does not itself request development
evaluation. When save and evaluation coincide, weights, optimizer, scheduler,
RNG and Trainer state are saved and `checkpoint_manifest.json` is sealed first.
Only then does evaluation run. Its finite metrics are published separately as
`evaluations/checkpoint-N.json`, bound to the exact checkpoint manifest SHA256 and
global step. A failed or interrupted evaluation leaves the sealed checkpoint
usable for resume while its step is below the configured training target. At the
final target, the checkpoint remains usable directly for inference, but the
existing resume guard rejects further training; retrying final metrics/export
needs a separate finalization workflow, which this change does not add. Every
evaluation in this policy restores training RNG state, including eval-only steps. The sealed checkpoint is never edited to append loss.
An unevaluated checkpoint has no inferred loss from a different training step.

`--segment-steps N` continues to pause at the first complete checkpoint by default.
Add `--continue-through-checkpoints` to preserve normal/timed checkpoints while
continuing to exactly N additional optimizer steps, or the configured complete
training target, whichever comes first. This is an invocation boundary, not a
new optimizer schedule. The operational choice is recorded in run status. A
continuing segment may produce multiple sealed checkpoints.

The bounded-output preflight runs initially and before every checkpoint write.
Use `--output-budget-root` to include the entire saved tree, particularly export
chunks outside the run directory. It reserves the next full checkpoint, final
model and margin without deleting artifacts. A failing recheck stops before that
checkpoint is written; the latest earlier sealed checkpoint remains available.
Space checks cannot guarantee quota safety against unrelated concurrent writers.
Kaggle's local package builder accepts the same optional continuation flag and
keeps the default off; building a package does not submit a provider job.

CPU-only verification for this policy uses fresh tiny random GPT-2 fixtures with
nonzero dropout, not production model or benchmark results:

```sh
PICOAGENT_RUN_ML_TESTS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -m pytest tests/test_training_efficiency.py tests/test_training_smoke.py -q
```

The eight tests cover bitwise final-weight, optimizer, RNG and scheduler equality
after a capped multi-checkpoint resume and after interruption between sealing
and evaluation; an odd-sized, accumulated cross-epoch dropout resume;
independent eval-only steps; checkpoint immutability; real bounded-output failure
caused by export files; legacy pause/cadence behavior; and the final-step recovery
limit. The opt-in policy also reuses final-step metrics instead of repeating the same development evaluation.
These CPU proofs do not establish accelerator speed, cross-device determinism,
or agent task quality.

Late evaluation is transferred as a separate content-addressed artifact,
`evaluations/<SHA256>.json`, without modifying the checkpoint's immutable
`transfer_manifest.json` or expanding its allowed paths. Collection advertises
and verifies the evaluation digest; collecting again also fetches late evidence
for a checkpoint already covered by a durable receipt. Generic CLI transfers
require `--expected-evaluation-sha256` for this optional artifact; filesystem
transfers can discover it locally. Invalid name, step, manifest hash or nonfinite
metrics fail closed. Best-checkpoint selection uses valid exact-checkpoint
sidecars, then exact-step legacy logs. Durable-retention plans bind sidecar
presence and hashes, so late evidence requires a fresh reviewed plan.

Before any payload download, the verified transfer manifest's exact original
bytes are atomically persisted beside the digest-scoped incoming chunk cache.
An interrupted transfer therefore retains the ordered file/chunk mapping as well
as verified chunks if the remote disappears. Partial cache data is not a complete
checkpoint or durability receipt; only fully verified reconstruction publishes
those completion markers. A conflicting cached manifest fails closed.

Runtime pruning removes an old acknowledged checkpoint together with only its
corresponding derived export bundle. Before deletion, both identities and exact
regular-file inventories must match the durable acknowledgements; symlinks,
path overlaps, unexpected files and changed evidence stop pruning. All proposed
pairs are validated before any removal. Latest-two, best and newly appearing
checkpoints and their bundles remain untouched. Late evaluation evidence defers
pruning until collection verifies it. This hardens the existing export cleanup;
it does not change transfer manifests or delete controller-side incoming caches.
Configure the bounded output root as a common parent of the run and export roots,
so retained checkpoints and exported chunks both count toward the same cap.
Paired deletion is not an atomic multi-directory transaction: if an export changes
or an error occurs after its acknowledged checkpoint is removed, the remaining
export is preserved for inspection rather than swept up by a retry.
