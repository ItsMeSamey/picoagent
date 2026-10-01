# Context modes and qualification

One shared context controller is used by collection and inference. Select
`--compaction-mode full|half|manual` at inference.

- **Full:** summarize the entire non-system history. No historical suffix is retained verbatim.
- **Half:** summarize the oldest token-half, moving the boundary to preserve complete tool-call/result groups. Keep the newer half verbatim.
- **Manual:** the model receives numbered atomic message groups and returns `keep_groups` plus a factual `summary`. Selected groups remain byte-for-byte unchanged in chronological order. Invalid IDs, partial tool groups, malformed decisions and non-shrinking results fail explicitly.

System instructions remain pinned. Summary text has user/reference authority,
never system authority. The full original transcript, exact compactor requests,
responses, model-visible contexts, selected groups, failures and token counts
remain in the audit archive. Compaction is not deletion of the source trajectory.
`ContextManager.compact(..., force=True)` also permits explicit compaction.

Full/manual modes proactively leave half the input budget as tool-response
headroom by default. Half mode retains its original trigger. Headroom is
configurable through `compaction_headroom_tokens`; it is recorded in collection
provenance and each event records the actual trigger budget. This is not a
promise that arbitrarily large tool output fits. Every compactor request is
checked against the tokenizer budget before a learned model call; a single
oversized response fails explicitly rather than being silently truncated.

Use an appropriate generation reserve: manual decisions include JSON group
selection and escaped summary text. The 4K tokenizer contracts use a 768-token
reserve. A configured 512-token generation cap is not sufficient for every
scripted reference summary. Longer-context SFT must also use a sequence length
that covers the complete request and target; examples must not be silently cut.

## What is established

Controller/replay tests cover all three modes, system preservation, atomic tool
pairs, non-contiguous manual retention, malformed selection rejection, repeated
summaries, exact source-only summaries and private-oracle isolation. The original
train/dev linked-record curriculum is checked separately under each mode with
the pinned SmolLM2 tokenizer at a 4096-token window. Fixtures are explicitly
unexecuted and are **not training or learned-model performance evidence**.

The procedural teacher provides mode-correct example decisions. It is not a
learned agent and cannot establish generalization. Current reference manual
selection prefers the latest observed tool group when it fits, otherwise carries
its relevant facts in the summary. Broader learned selection needs training and
evaluation beyond this initial reference strategy.

## Learned-model evaluation protocol

`picoagent.evaluation.evaluate` runs identical dev tasks with the same stateless,
deterministic learned policy in full, half and manual modes, in fresh isolated
workspaces and fresh KV stores. It preserves each attempt and never substitutes
a teacher action. It reports task success, protocol validity, compaction usage
and input-token cost separately. A task must have all three results; failures
stay in denominators. Family-macro scores prevent large easy families from
hiding poor coverage. `selection_key` maximizes the weakest mode's family-macro
success, then the mean, using verified dev runs only, with compaction exercised
on every qualification task. Short tasks belong in a separate basic suite.

Final test tasks require explicit unlocking after checkpoint selection. They are
not training data and cannot be used for checkpoint selection. The current
collector needs an approved Docker/Podman environment for learned-model code.
No trained checkpoint or per-mode learned-model score is available yet.
