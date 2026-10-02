# Half-hour snapshots with independent publication

This policy is for a **fresh run**, not a migration of the original step-1
checkpoint. Preserve the original run and its manifests. Resume identity checks
must remain unchanged.

The production preset is `configs/smol360m_native_t4_async_30min_v4.json`.
After its exact prepared-data admission and hardware/disk preflight pass on a
separately authorized free T4 runtime, the full-training command is:

```sh
PYTHONPATH=src TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  PYTHONHASHSEED=20261001 python -u -m picoagent.training train \
  --config configs/smol360m_native_t4_async_30min_v4.json \
  --output-budget-bytes 50000000000 --output-budget-root /content
```

Do not add `--durability-timeout-seconds`: that selects the old blocking
acknowledgement policy and is rejected by this async configuration. The raw
companion config is for CPU token preparation, not the designated production
launch. Run the command through the existing supervised Colab controller so
job status remains available after a disconnected response; inspect an
uncertain launch before retrying.

The asynchronous preset uses a 1,800-second monotonic timer checked at completed
optimizer-step boundaries. Saving model, optimizer, scheduler, scaler and RNG
state still pauses the training thread briefly so the snapshot is coherent.
Transfer and verification happen in a separate controller while training resumes.
There is no mandatory step-1 snapshot or blocking public-backup acknowledgement.
Development evaluation remains independently scheduled every 500 steps.

## Bounded backlog

The trainer retains at most two intermediate sealed snapshots and reserves an
additional final snapshot slot. If an upload or retention cannot finish in time,
new intermediate snapshot requests coalesce rather than overwriting an in-flight
snapshot, deleting unverified data, growing an unbounded queue, or waiting for
the network. Status explicitly reports coalescing. This is a degraded mode: a
30-minute requested cadence cannot guarantee a newly durable snapshot every
30 minutes when the destination is unavailable or too slow.

The output budget must include runtime export chunks, checkpoint payloads, a
reserved final checkpoint, final model, and safety margin. Other processes can
consume free space, so preflight checks are not a guarantee against exhaustion.
A recoverable transfer failure is distinct from a fatal integrity/authentication
failure or exhaustion of the filesystem holding training outputs.

## Publication and retention

Start the collector independently of training. With an authorized public
destination, enable explicit latest-published retention rather than legacy
latest-two/best retention. The older local snapshot is eligible only after a
newer complete publication is independently verified; an incomplete or failed
upload never authorizes deleting it. Existing public releases are not deleted.

Without authorized publication, omit publication and pruning flags. Verified
controller-local copies preserve data outside the runtime, but do not count as
public GitHub publication and do not authorize public-upload-gated deletion.
Do not claim rolling half-hour durability if the bounded queue is coalescing.

After training stops, keep collecting until its final sealed snapshot has been
verified at the approved destination. Training completion and publication
completion are separate outcomes. Do not terminate a runtime merely because
its optimizer reached the final step while final transfer is still pending.

## Current recovery constraints

The original step-1 checkpoint remains preserved. GitHub checkpoint publication
has not been resumed, and no new runtime has been allocated as part of these
code changes. The controller previously had about 13 GiB of free space; at
approximately 4.35 GB per full snapshot this is insufficient for an entire long
run without a working destination and retention. Check actual free capacity
and free-compute availability before launch. Never use another project's Colab
session or assume a new GPU allocation is free.
