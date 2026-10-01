# Colab CLI transport diagnostics

The supported launcher is `scripts/colab_safe_cli.py`, run with the Python
environment containing Google's official `google-colab-cli==0.7.4`. That release
requires `jupyter-kernel-client==0.9.0`; the diagnostic environment also used
`websocket-client==1.9.2`. It delegates ordinary session/authentication behavior
to the installed CLI. It does not provision sessions, refresh credentials itself,
change TLS verification, retry user code, restart kernels, or alter proxy settings.

## What the launcher fixes

CLI 0.7.4's `exec` performs its initial `/content` setup execution before entering
the `finally` block that calls `runtime.stop()`. A failure at that stage can leave
a local WebSocket listener thread alive. The launcher tracks runtimes created
during the entire invocation and closes their local clients in an outer `finally`,
including on `SystemExit`. It never requests remote kernel shutdown. This is
complementary to the controller's process-group timeout cleanup.

The launcher also installs an `on_error` handler because the underlying client
otherwise hides the useful WebSocket error behind the later closed-connection
exception. Diagnostics print exception types and traceback locations only:

- No exception message, source line, frame locals, HTTP headers or URL is printed
- Unhandled main-thread exceptions receive the same treatment instead of a rich
  traceback that might expose runtime credentials
- Runtime/socket patches are restored after the invocation

The official CLI has its own private log/state directory. Keep it out of source
archives and published diagnostics. Do not enable verbose logging or export its
raw logs: those can include authenticated runtime URLs and executed code.

## Observed status, 2026-10-01

Earlier TPU and T4 sessions encountered `AttributeError` involving a missing
`sock`, followed by `WebSocketConnectionClosedException`, before user code ran.
Those sessions were explicitly terminated without training or data loss.

A single new standard/free CPU session named `picoagent-cpu` successfully executed
Python and reconnected in a second invocation. Docker was installed from the
official Ubuntu package repository and its daemon reported version 29.1.3. The
daemon was launched with networking configuration disabled:

```text
--iptables=false --bridge=none --ip-forward=false --ip-masq=false --storage-driver=vfs
```

Another invocation reconnected after Docker started, pulled the official Python
image, and attempted a hardened container. It **failed before any container code
executed**:

```text
image: python:3.11-slim-bookworm
digest: python@sha256:a36c24f9cbdf4fd0f52d67f0823eeac19c2028c637cecc392d97f980d4fec56b
diagnostic container ID: 44d98c3a19197a220a48789781f458f297b70cae7fe8e412f3d214ccd1f65ec1
exit code: 125
runc: unable to apply cgroup configuration:
mkdir /sys/fs/cgroup/docker: read-only file system
```

Docker also warned that swap limiting was unsupported. The attempted container
retained `--network=none`, `--read-only`, `--cap-drop=ALL`,
`--security-opt=no-new-privileges`, PID, memory, CPU and file-size limits. No
security settings were weakened and no cgroup remount was attempted. This runtime
therefore did **not** qualify for the isolated rollout harness. No rollout or
training data was collected. The recorded container ID proves only a creation
attempt, not successful execution.

The failed diagnostic container was confirmed absent afterward. A separate
read-only inspection reported the cgroup mount flags as
`ro,nosuid,nodev,noexec,relatime`, corroborating the daemon's failure.
The unused `picoagent-cpu` session was explicitly stopped, with the CLI confirming
`Session terminated`. The diagnostic receipt's relevant non-secret fields are
preserved above; its original runtime-local file was ephemeral.

This is evidence that authenticated CPU execution works. It does not establish
the root cause of the earlier accelerator-session failures or demonstrate that
TPU/T4 reconnection has been fixed. A missing-socket error can be secondary to
connection teardown; collect the sanitized traceback locations before diagnosing
its cause. Do not blindly repeat a possibly started training job.

CLI `exec` can itself exit zero even when the remote Python cell raises an error.
Require a success sentinel and a valid operation receipt; never infer remote
success from the local process's exit code alone.

## Use and verify

Use the already authenticated CLI environment and an explicit existing session:

```sh
path/to/colab-environment/bin/python scripts/colab_safe_cli.py \
  --auth=oauth2 exec --session picoagent-cpu --timeout 60 \
  --file trusted_diagnostic.py
```

Leave the platform's trusted CA/proxy configuration intact. If the connection
fails, do not disable certificate verification, bypass an access denial, allocate
additional accelerators or modify account/security settings as a workaround.
Check the session/job state before retrying any operation with side effects.

Run the local, account-free tests:

```sh
python -m pytest tests/test_colab_transport.py -q
python -m ruff check scripts/colab_safe_cli.py tests/test_colab_transport.py
```

The eight tests cover early startup failure, normal/error/interrupted CLI exits,
unopened runtimes, secret-safe diagnostics, callback preservation and continuing
cleanup after an individual client fails. They do not prove remote reliability.
