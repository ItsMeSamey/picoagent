# Interrupted first T4 training segment

The pinned full-parameter FP16 run reached step 46 of 3,044 and sealed a local checkpoint. Colab ended before its full-state off-runtime transfer completed. The termination cause is unknown. No complete resumable checkpoint or learned-agent score is claimed. Partial hash-verified chunks remain preserved outside Git.

The validation pass took 587.7 seconds before checkpoint serialization. The next explicitly identified run separates CPU preparation, durability saves and validation, and uses bounded parallel backup. It starts from the original pinned base model, rather than pretending to resume the incomplete checkpoint.

See outcome.json for timestamps, limitations and hashes of the original run manifest, paused status and captured training log.
