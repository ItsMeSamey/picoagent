# Memory-only Colab controller

Commands below assume the reviewed adapter has been copied into `scripts/ephemeral_colab.py`.

Development adapter only; no runtime was allocated or contacted during tests.
It uses the installed vendor's `state.client` credential loader unchanged and
keeps runtime proxy credentials solely in a vendor `SessionState` in memory.
It does not call session stores, history, logging setup, authentication rewrites,
TLS overrides, network monkeypatches, kernel restart, or runtime termination.

## Reconnect

Create a descriptor from a previously verified, nonsecret endpoint identity:

```json
{"schema":"picoagent.ephemeral-colab.v1","name":"YOUR-NAME","phase":"ready","endpoint":"EXACT-SAVED-ENDPOINT","kernel_id":"SAVED-KERNEL-ID","session_id":"SAVED-SESSION-ID"}
```

Omit unknown kernel/session IDs. A read-only reconnect does not start a kernel;
the first explicitly requested execution may establish a kernel on the selected
existing runtime. Endpoint absence or ambiguity fails; it never allocates a
replacement. A lost kernel connection is closed locally, never shut down remotely.
IDs newly returned by the vendor are saved atomically. URL/token fields are
prohibited in descriptors.

```bash
PYTHONPATH=src:scripts .venv/bin/python \
  scripts/ephemeral_colab.py --descriptor /path/reconnect.json
```

The command reads assignments and reports only nonsecret descriptor fields.
The descriptor stores no runtime URL, proxy credential or expiry. Refresh uses
`Client.list_assignments()` with exact endpoint matching; vendor proxy credentials
are renewed in memory when within five minutes of expiry, or by explicit
`controller.refresh()`.

## Authorized operations from another controller script

```python
from ephemeral_colab import EphemeralController
from colab_run import start, collect, restore
from colab_sdk_watch import watch

with EphemeralController('/path/reconnect.json', download_workers=4) as controller:
    result = controller.execute(code)  # Must emit exactly one PICOAGENT_RESULT JSON object.
    # controller.command('upload', local_path, 'content/destination')
    # controller.transfer().download('content/file', local_path)
    # start(controller, '/content/picoagent', approved_argv)
    # watch(controller, project=..., run_dir=..., export_root=..., destination=...)
```

The normal checkpoint publication flags/approval remain required. Use a
persistent external archive and hold its normal retention lock around `watch`,
as the SDK CLI does. Do not use an existing live publisher's locked archive
concurrently. Do not print controller/session objects or vendor exceptions.
No execution/transfer method changes its user's authorization scope.

## Explicit allocation only

An entirely separate allocation entry point is present for a specifically
approved allocation. This example is documentation, not authorization to run it:

```bash
PYTHONPATH=src:scripts .venv/bin/python \
  scripts/ephemeral_colab.py \
  --descriptor /path/NEW-reconnect.json --allocate --name APPROVED-NAME \
  --variant GPU --accelerator T4 --shape 0
```

The new descriptor is reserved before the one vendor `assign` call. Existing
paths fail rather than allocate twice. An uncertain assignment retains only its
nonsecret notebook UUID and phase; it is never retried or unassigned. Inspect
assignments read-only and explicitly reconcile the exact endpoint before use.
The allocation function's `approved=True` is only an application gate; the
caller must separately have the user's applicable authorization.

Tests are entirely fake-provider and cover exact selection, no automatic
allocation, credential serialization rejection, untouched vendor state/history,
refresh, ID persistence, cleanup, explicit allocation, and uncertain outcomes.

## Transfer and private-log limits

Remote transfer paths must be canonical `content/...` or `/content/...`; parent
traversal, empty/dot segments, URL/query syntax and paths outside content are
rejected. Local paths and all their ancestors must not be symlinks; source and
destination files must be regular files. Only ordinary ASCII filename segments
are accepted remotely. Uploaded files are snapshotted through a no-follow open,
limited to 32 MiB, checked for concurrent change, and the private temporary copy
is removed afterward. Downloads use the public `ContentsClient.list_dir(parent)` method and require a
unique child with the exact requested filename/path and bounded size before fetching,
use private temporary destinations, verify the exact declared length within
32 MiB, then atomically replace the validated local destination. Failed downloads
preserve previous destination bytes and clean their own temporary file.

The unmodified vendor ContentsClient buffers responses. Metadata preflight and
post-download size checking are **not a streaming memory cap against a hostile
server lying about its file size**. Remote canonical paths also do not prove the
remote runtime has no symlinks; its filesystem is trusted, while caller-controlled
traversal and local symlinks are rejected. Checkpoint integrity still comes from
the existing transfer-manifest and per-chunk SHA256 checks.

A reference-counted logging filter drops logs during vendor calls and throughout
an open runtime/websocket connection, including vendor `stop()` exception logs.
Overlapping worker calls do not prematurely restore handlers or serialize
network IO. Original handlers/levels are restored after the last protected call
or runtime connection closes. Use the adapter in a dedicated controller process
and always close it with a context manager; unrelated logs in that process are
intentionally suppressed while a runtime connection is open.

Do not add logging handlers, enable debug configuration, or run logging setup
while a protected call or runtime connection is active. The filter protects the
handler inventory present at its first entry, not handlers added afterward.
