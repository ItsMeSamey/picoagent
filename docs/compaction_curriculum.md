# Half-context compaction curriculum

This is a separate, original expansion track. It does not replace the basic
512-train / 64-dev / 64-test corpus. `data/compaction-v1` contains **24 unexecuted
task specifications** (8 train, 8 dev, 8 test) and initial authored tool requests.
There are **zero verified rollouts** in that directory. No learner competence or
benchmark result is established by the generator or its unit tests.

## What the tasks exercise

Every task has 16 linked files containing `[amount, status]` observations plus
long, irrelevant, procedurally generated audit text. Only the first path appears
in the prompt; each read reveals the next path. The task requires one file per
call and a final calculation using the whole observed sequence. Remembering only
the latest file is insufficient. The files are ordinary workspace inputs, not
protected secrets; this is memory supervision, not an anti-cheating benchmark.

Whole families, including all seeds and horizon variants, have fixed splits:

| Split | Families |
| --- | --- |
| Train | Running balance; last qualifying status |
| Dev | Threshold count; first qualifying status |
| Test | Amount range; adjacent status changes |

The acquisition protocol is deliberately shared; family-held-out results do not
demonstrate arbitrary-task generalization. No benchmark examples or downloaded
task corpus are used. An oracle lives in collector-side task metadata but is never
passed to the policy callback.

## Exactly the shared compaction scheme

`VisibleContextTeacher` has no constructor arguments, task handle, hidden history,
fixture access, or retained state. It implements the same `(messages, tools)`
callback for ordinary actions and summary generation. The action branch derives
the next read and final answer solely from visible messages. The summary branch
receives only the exact historical source serialized by `ContextManager`, with
tools disabled. It cannot consult the unchanged suffix, future observations, or
the private answer.

The existing harness triggers compaction when measured input exceeds the context
budget. It pins the system prefix, summarizes the oldest half by measured tokens
at the nearest complete assistant/tool-group boundary, and leaves the recent
suffix byte-for-byte unchanged. The summary is untrusted reference data in a
user-role message. Repeated compaction also summarizes previous summaries.

The structured summary retains the goal parameters, every observed amount/status
pair, the next path, failures, and execution outcomes. Repeated receipt outcomes
are grouped while retaining every exact tool call ID as a receipt reference. It
never supplies missing container IDs or infers that an unobserved tool succeeded.
Irrelevant audit prose is discarded. Exact read arguments, each unique container
identity, full output and per-command timing remain in the raw immutable attempt,
keyed by those call IDs, rather than being repeated in memory.

This bootstrapping teacher uses a narrow JSON memory format. Strict semantic
admission currently requires that format; it is not an evaluator for arbitrary
free-form model-written summaries. The harness's default 384-token summary target
is advisory. Its actual budget check is shrinking measured context; long exact
observation ledgers can exceed the target. Check rendered event lengths against the
frozen SFT configuration instead of truncating summaries or changing the budget
after looking at development loss.

## Generate and verify without executing anything

```bash
python -m picoagent.data.compaction_curriculum \
  --output-dir data/compaction-v1 \
  --seeds-per-family 4 --horizon 16 --detail-words 160
python -m picoagent.data validate --manifest data/compaction-v1/manifest.json
python -m pytest tests/test_compaction_curriculum.py -q
```

Generation uses exclusive creation; reruns require a new directory. The authored
examples end with an unanswered first tool request and carry `status=unexecuted`.
They contain no synthesized tool results, receipts, or final answers and fail
production training admission. Unit tests use explicitly unexecuted in-memory
fixture replies. They test contracts; they do not test container isolation or
establish successful external execution.

## Collect genuine training traces later

`collect_compaction_task` wraps the existing `collect_task` and `ContainerSandbox`.
There is no host subprocess, mock-runtime, or weaker-isolation fallback. A runtime
startup failure is preserved as an error attempt, not relabelled as a success.
Use the already-authorized isolated runtime and an immutable image identity.

Supply the **actual intended policy tokenizer**, including ordinary tool schema
cost. The byte-counting estimate used in unit tests is not a production tokenizer.
For example, with an already-loaded pinned tokenizer:

```python
from picoagent.data.audit import read_jsonl
from picoagent.data.compaction_curriculum import (
    collect_compaction_task, export_compaction_attempts,
)
from picoagent.harness.protocol import render_messages
from picoagent.harness.tools import TOOL_SCHEMAS

def count_tokens(messages):
    rendered = render_messages(messages, TOOL_SCHEMAS, add_generation_prompt=True)
    return len(tokenizer.encode(rendered, add_special_tokens=False))

for split in ("train", "dev"):
    for task in read_jsonl(f"data/compaction-v1/{split}.tasks.jsonl"):
        result = collect_compaction_task(
            task,
            "data/attempts/compaction-v1",
            runtime="docker",
            image=verified_image_digest,
            token_counter=count_tokens,
            context_max_tokens=4096,
            context_reserve_tokens=512,
        )
        print(result["trace"]["status"], result["compaction_audit"])

export_compaction_attempts(
    "data/attempts/compaction-v1",
    "data/rollouts/compaction-v1",
    token_counter=count_tokens,
)
```

Here `tokenizer` and `verified_image_digest` must be explicitly resolved by the
runner; neither is a guessed package/model/runtime identity. A supplied learned
model can use the same collector callback, but its summaries must satisfy the
current memory contract to enter this teacher-supervised track. Test-family
collection is disabled unless `include_test=True`; test is never an SFT input.

## Admission, retention, and SFT

Use `export_compaction_attempts`, not the generic exporter, to assert compaction
supervision. A correct answer obtained without an accepted, budget-triggered
compaction does **not** qualify. Admission verifies:

- Standard successful original-procedural execution provenance and container
  receipts, plus immutable archive hashes
- Exact model-event replay, source/request alignment, unchanged suffix and the
  oldest-half group boundary
- At least one accepted summary that actually reduced an over-budget context
- Exact preservation of visible goal, facts, receipts and unresolved failures,
  with no unsupported extra facts
- Required recounting of token budgets and the half boundary using the supplied
  policy tokenizer

Every valid attempt and its admission report remains in `all_attempts.jsonl` and
`compaction_audits.jsonl`. Incomplete or invalid attempts stay in their original
archive directories and are listed in the export report. No attempt is deleted
or overwritten. One deterministically selected audited success per task enters
the train/dev/test export views. Hashes and container metadata are audit evidence,
not independent cryptographic proof against a malicious collector.

The existing `training.encoding.event_examples` already expands accepted
compactions into their exact two-message request plus assistant summary response,
with `tools=[]` and last-response-only supervision. Ordinary actions train on the
actual effective context after each compaction. No flattening, inferred history,
teacher-oracle answer copying, or silent truncation is needed. Snapshot verified
train/dev exports with the existing immutable dataset workflow before SFT. Keep
this track's inclusion and loss results distinct from the basic corpus.
