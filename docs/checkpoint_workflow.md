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
