# Shared training/inference harness

The harness imports only Python's standard library. No runtime service, model,
network endpoint, or credential is created implicitly. The default schema lives
in `picoagent.harness.TOOL_SCHEMAS` and is identical in training and inference.

## Policy and data contract

```python
from picoagent.harness import AgentHarness, ContextManager, ToolRegistry

def model(messages: list[dict], tools: list[dict]) -> dict:
    # Return ONE assistant message, optionally containing structured tool_calls.
    ...

registry = ToolRegistry(backend, knowledge, search_client=search)
context = ContextManager(model, max_tokens=4096, reserve_tokens=512,
                         token_counter=count_with_actual_tokenizer)
agent = AgentHarness(model, registry, context=context, max_steps=16,
                     trace_path="run-events.jsonl")
result = agent.run("Task", initial_messages=None)
```

`result` is a `RunResult` dataclass (`to_dict()` is available) with `messages`,
`events`, `final`, `stop_reason`, `steps`, and `error`. Stop reasons are `final`,
`max_steps`, `invalid_model_output`, and `context_budget`. Unexpected model/network
exceptions propagate to the caller; semantic tool errors become recoverable tool
messages. Models receive deep copies, preventing callback mutation of history.

Message examples:

```json
{"role":"user","content":"Inspect the input file"}
{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"bash","arguments":"{\"command\":\"cat input.txt\"}"}}]}
{"role":"tool","name":"bash","tool_call_id":"c1","content":"{\"stdout\":\"data\",\"exit_code\":0}"}
{"role":"assistant","content":"The input contains data"}
```

Tool call IDs are unique across the active conversation. A complete assistant
batch must be followed by exactly one result per call before the next message.
`validate_conversation` checks this; `allow_pending=True` is reserved for the
newly generated assistant response before dispatch. Arguments may initially be
objects but are normalized to canonical JSON strings before rendering/execution.

## Canonical text and loss masking

Use `render_messages(messages, tools, add_generation_prompt=True)` for inference;
use `render_segments(messages, tools)` for SFT. Segments have `text`, `trainable`,
and `role`. Only assistant JSON plus its end marker is trainable. Headers, tools,
user text, and system text are masked. Concatenating segment text exactly equals
`render_messages(messages, tools)`. Train and infer with the SAME tool header.
Do not independently choose a Hugging Face chat template or tokenizer defaults.

The format is `TOOLS:\n<canonical JSON>\nEND_TOOLS\n`, then repeated
`ROLE_UPPER:\n<canonical full message JSON>\nEND_MESSAGE\n`; the generation prefix
is `ASSISTANT:\n`. Embedded newlines in JSON strings are escaped. `parse_assistant`
accepts one assistant JSON object and an optional trailing `END_MESSAGE`, rejecting
extra messages. The protocol has no dependency on special tokenizer tokens.

## Tool surface

- `bash(command, timeout?)`: Bash with no shell profile; absolute maximum timeout
  is imposed by sandbox config
- `python(code, timeout?)`: code sent as stdin, written to a temporary `.py` file
  inside the container, executed with `runpy`, then removed
- `write_file(path, content, executable?)`: atomic UTF-8 file/script creation in
  `/workspace`; rejects traversal and symlink destinations
- `search(query, limit?)`: only the preconfigured SearXNG endpoint; model cannot
  choose a request URL; HTTP redirects disabled, response size capped
- `knowledge(operation, key?, value?, prefix?)`: bounded JSON KV notes with
  get/set/delete/list and atomic-file persistence

Search and knowledge values are untrusted reference data. `SearXNGSearch` uses
`format=json`; the server must permit its JSON API. For offline tests, inject a
`search(query, limit=5) -> dict` object, and label its provenance accurately.

## Sandbox and receipts

```python
from picoagent.harness import ContainerSandbox, KnowledgeStore
with ContainerSandbox(task_root="runs", image="python:3.11-slim") as backend:
    backend.seed_files({"input.txt": "trusted task fixture"})
    receipt = backend.probe()       # Fails if daemon/image/limits unavailable
    kv = KnowledgeStore("notes/episode-unique.json") # OUTSIDE mounted workspace
    # Run AgentHarness using this backend and KV store.
    text = backend.read_file("answer.txt", max_bytes=4096)
```

Fixtures must be seeded BEFORE any sandbox execution, including `probe()`. Never
mount hidden graders, oracle answers, host home, sockets, credentials, or the KV
store in the task workspace. Seed only public task inputs/documentation. Each
command runs a fresh container over the same task-unique workspace; shell exports,
processes, and `/tmp` contents do not persist across calls, but workspace files do.
`close()` removes the task workspace, so collect needed artifacts before closing.

Container launches enforce no network, read-only root filesystem, dropped Linux
capabilities, no privilege escalation, non-host-home user, process/CPU/memory/file
size limits, capped tmpfs, capped stdout/stderr, and wall-clock deadlines. An
in-container GNU `timeout` additionally bounds execution if the host runtime
client is interrupted. Docker/Podman removal is attempted after every call.
Images must contain `python`, `bash`, and GNU `timeout`; the default official
Debian-based Python image satisfies this. Images are NEVER pulled implicitly
(`--pull=never`); explicitly provision and pin a digest for reproducibility.
Only the per-task directory is bind-mounted. Runtime client env is allowlisted;
API tokens, cloud credentials, and proxy secrets are not forwarded to containers.

`ExecutionResult.to_dict()` includes stdout/stderr, exit_code, timed_out, truncated,
backend (`container`), duration_seconds, runtime (`docker`/`podman`), image, and an
actual 64-character container_id read from the runtime's `--cidfile` output.
`runtime_metadata()` returns the latest receipt identity. The container has already
been removed when a successful call returns. `probe()` requires both exit code 0
and a real container ID. A runtime executable alone does not prove a daemon or
image is available. A command's nonzero exit code is an observed failure, not a
successful task outcome.

This is a baseline, not a strong multi-tenant security boundary. Use a disposable,
rootless runtime worker with a patched kernel and an external filesystem quota.
The per-file limit does not cap the total bind-mounted workspace size. Container
images and runtime config are trusted operator inputs. Use a stronger VM-based
sandbox for actively hostile or production multi-tenant workloads. Real container
integration cannot be validated when no Docker/Podman daemon is available.

`TrustedLocalSandbox` exists only for explicitly hand-authored developer tests.
It raises for `model_generated=True` (the default), and `ToolRegistry` refuses to
use it. It is NEVER an automatic fallback. No container runtime means model shell
execution fails closed. Do not label local fixture tests as agent rollouts.

## Compaction and training traces

`ContextManager.compact(messages)` preserves initial system messages, summarizes
roughly the first half of the remaining messages, and retains the second half
byte-for-byte at the message level. The split moves to a complete assistant/tool
batch boundary. Summaries use the same model callback with no tools; inject a
separate callback only when explicitly labeling teacher-generated data. The
summary is inserted as a user-role historical-data message, not system authority.
The original input is never mutated. `fit()` repeats shrinking compactions until
the input budget fits, or fails explicitly if no reduction is possible. It never
silently drops a recent suffix or splits tool call/result pairs.

Supply the REAL tokenizer counter, including tools/header/generation prefix:

```python
def count_with_actual_tokenizer(messages):
    text = render_messages(messages, TOOL_SCHEMAS, add_generation_prompt=True)
    return len(tokenizer.encode(text, add_special_tokens=False))
```

The fallback counter is a conservative UTF-8-byte estimate, not a measurement of
model tokens. `summary_tokens` is a target instruction, not an enforced generation
limit; configure the model callback's generation limit too. Nonshrinking summaries
are rejected and recorded.

Events are JSON-serializable and optionally appended as JSONL. `assistant` events
contain the exact `input_messages` and generated `message`; this is the authoritative
per-decision SFT source after compaction. `tool_execution` events include call ID,
arguments, result, and `verified` (dispatch returned, NOT oracle success or genuine
container provenance). `compaction` events contain `source_messages`,
`summary_request`, `summary_response`, `summary_message`, unchanged
`retained_messages`, pinned messages, result messages, token counts, and `accepted`.
Train summary behavior on `summary_request + [summary_response]`. A shared trace
path between ContextManager and AgentHarness writes each compaction once.
`RunResult.messages` is the final effective context, not a complete pre-compaction
transcript; use events to reconstruct every model decision faithfully.

For verified training records require actual container receipts AND executable
oracle checks. Reset workspace and KV per episode. Freeze any documentation/search
snapshot used for held-out evaluation and keep oracle metadata outside the agent's
visible inputs. Never infer runtime verification from a hand-authored answer or
successful JSON formatting alone.
